import hashlib
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

# Vergara et al., UCI Machine Learning Repository, CC BY 4.0
URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
SHA256 = "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"
BATCHES = 10
FEATURES = 128
GASES = ("ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene")
# 16 rows a step make the first 200 steps, the first model's window, about the first three batches
ROWS = 16
KNOWN = 3
SHUFFLE = 0
# outside the repository, the data is never committed
CACHE = Path(os.environ.get("DRIFTGATE_DATA", Path.home() / ".cache/driftgate"))


def fetch(cache=CACHE):
    """the path to the archive in cache, downloaded the first time.

    ValueError when its sha256 isn't the one this code was written against.
    """
    path = Path(cache) / "gas.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL) as response:
            path.write_bytes(response.read())
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256:
        raise ValueError(f"{path} hashes to {got}, not {SHA256}: remove it and fetch again")
    return path


def parse(text):
    """one batch file as (x, y): libsvm style lines, the gas from 1 then index:value pairs.

    y counts the gases from 0, in GASES order. some lines give the gas as
    gas;concentration, the concentration is dropped.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    x, y = np.zeros((len(lines), FEATURES)), np.empty(len(lines), dtype=np.int64)
    for i, line in enumerate(lines):
        label, *pairs = line.split()
        y[i] = int(label.split(";")[0]) - 1
        for pair in pairs:
            k, v = pair.split(":")
            x[i, int(k) - 1] = float(v)
    return x, y


@cache
def batches(path):
    """every batch of the archive at path in time order, as a tuple of (x, y)."""
    with zipfile.ZipFile(path) as z:
        return tuple(parse(z.read(f"Dataset/batch{b}.dat").decode()) for b in range(1, BATCHES + 1))


def ordered(parts):
    """the batches as one stream: (x, y, the row each batch starts at).

    the batches stay in time order, the rows inside each are shuffled, the
    same way every time: the archive lists the early batches gas by gas, and
    served like that every new gas would look like a change. x is
    standardized with the means and spreads of the first KNOWN batches, the
    year the first model is trained on, all that was known before serving.
    """
    xs, ys = [], []
    for b, (x, y) in enumerate(parts):
        order = np.random.default_rng([SHUFFLE, b]).permutation(len(y))
        xs.append(x[order])
        ys.append(y[order])
    known = np.concatenate(xs[:KNOWN])
    mean, sd = known.mean(axis=0), known.std(axis=0)
    x = (np.concatenate(xs) - mean) / np.where(sd > 0, sd, 1.0)
    return x, np.concatenate(ys), np.cumsum([0] + [len(y) for y in ys[:-1]])


@cache
def _stream():
    return ordered(batches(fetch()))


@dataclass(frozen=True)
class Gas:
    """the gas sensor batches served ROWS rows a step, in time order.

    seed only drives which labels come back and when, as with a Scenario.
    data is the archive's sha256, so a recipe names the exact file, and only
    that one is accepted. the rows left over after the last full step are
    never served. at is None: when the sensors drifted isn't known.
    """

    dataset: str = "gas"
    seed: int = 0
    data: str = SHA256

    # the shape of what it serves, as on a Scenario
    features, labels, rows = FEATURES, len(GASES), ROWS
    kind, at = "gas", None

    def __post_init__(self):
        if self.data != SHA256:
            raise ValueError(f"this code reads the archive with sha256 {SHA256}, not {self.data}")

    @property
    def steps(self):
        return len(_stream()[1]) // ROWS

    def starts(self):
        """the step each batch starts at, rounded down."""
        return [int(r) // ROWS for r in _stream()[2]]

    def batch(self, step):
        """the x and y served at step, copies."""
        if not 0 <= step < self.steps:
            raise IndexError(f"the gas stream has steps 0 to {self.steps - 1}, not {step}")
        x, y, _ = _stream()
        at = slice(step * ROWS, (step + 1) * ROWS)
        return x[at].copy(), y[at].copy()


def main():
    path = fetch()
    print(f"{path}, sha256 {SHA256[:16]}")
    for b, (x, y) in enumerate(batches(path), 1):
        counts = " ".join(f"{n:4d}" for n in np.bincount(y, minlength=len(GASES)))
        print(f"batch {b:2d}: {len(y):4d} rows, per gas {counts}")


if __name__ == "__main__":
    main()
