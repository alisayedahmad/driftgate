import hashlib
from pathlib import Path

import numpy as np

from world.scenarios import KINDS, Scenario

SEEDS = range(3)
LAST = 999
FILE = Path(__file__).with_name("fingerprints.txt")


def fingerprint(s):
    """first 16 hex digits of a sha256 over the x and y bytes of a few steps.

    the steps are 0, LAST, and the two around the change when there is one.
    """
    steps = [0, LAST] if s.at is None else [0, s.at - 1, s.at, LAST]
    h = hashlib.sha256()
    for t in steps:
        x, y = s.batch(t)
        h.update(x.astype("<f8").tobytes())
        h.update(y.astype("<i8").tobytes())
    return h.hexdigest()[:16]


def table():
    return "".join(f"{kind} {seed} {fingerprint(Scenario.draw(seed, kind))}\n"
                   for kind in KINDS for seed in SEEDS)


if __name__ == "__main__":
    FILE.write_text(table())
    print(f"wrote {FILE}")
