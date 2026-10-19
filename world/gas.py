import hashlib
import os
import urllib.request
import zipfile
from functools import cache
from pathlib import Path

import numpy as np

# Vergara et al., UCI Machine Learning Repository, CC BY 4.0
URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
SHA256 = "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"
BATCHES = 10
FEATURES = 128
GASES = ("ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene")
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


def main():
    path = fetch()
    print(f"{path}, sha256 {SHA256[:16]}")
    for b, (x, y) in enumerate(batches(path), 1):
        counts = " ".join(f"{n:4d}" for n in np.bincount(y, minlength=len(GASES)))
        print(f"batch {b:2d}: {len(y):4d} rows, per gas {counts}")


if __name__ == "__main__":
    main()
