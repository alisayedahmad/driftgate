import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from world.labels import fate
from world.scenarios import Scenario

ROOT = Path(__file__).parents[1]


def canonical(doc):
    """doc as JSON bytes with sorted keys and no spaces, the same doc always gives the same bytes."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def data_hash(x, y):
    """sha256 of the rows, shapes included, as 64 hex characters."""
    h = hashlib.sha256(repr((x.shape, y.shape)).encode())
    h.update(x.astype("<f8").tobytes())
    h.update(y.astype("<i8").tobytes())
    return h.hexdigest()


def rows(recipe):
    """the (x, y) a recipe stands for.

    the rows of its scenario over steps [first, stop) whose label, under its
    labeling, had come back by step as_of. y is the label the pipeline saw.
    """
    s = Scenario(**recipe["scenario"])
    xs, ys = [], []
    for t in range(*recipe["steps"]):
        x, y = s.batch(t)
        labeled, arrives = fate(s.seed, t, recipe["labeling"], len(y))
        known = labeled & (arrives <= recipe["as_of"])
        xs.append(x[known])
        ys.append(y[known])
    return np.concatenate(xs), np.concatenate(ys)


def dataset(s, first, stop, labeling, as_of):
    """what was known at step as_of about steps [first, stop) of scenario s.

    returns x, y and the document naming them, which holds the recipe, the
    number of rows and their hash, not the rows.
    """
    recipe = {"scenario": asdict(s), "steps": [first, stop], "labeling": labeling,
              "as_of": as_of}
    x, y = rows(recipe)
    return x, y, {"recipe": recipe, "rows": len(y), "data": data_hash(x, y)}


def verify(doc):
    """regenerate a dataset document's rows and check they still hash the same."""
    return data_hash(*rows(doc["recipe"])) == doc["data"]


def code(root=ROOT):
    """the commit checked out in root, whether the tree had changes, and their hash.

    changes covers modified tracked files and untracked ones git doesn't ignore.
    it is None when the tree is clean.
    """
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout

    commit = git("rev-parse", "HEAD").decode().strip()
    if not git("status", "--porcelain"):
        return {"commit": commit, "dirty": False, "changes": None}
    h = hashlib.sha256(git("diff", "HEAD", "--binary"))
    for name in sorted(git("ls-files", "--others", "--exclude-standard").decode().splitlines()):
        h.update(name.encode() + b"\0" + (Path(root) / name).read_bytes())
    return {"commit": commit, "dirty": True, "changes": h.hexdigest()}


def versions():
    return {"python": platform.python_version(),
            "numpy": version("numpy"), "scikit-learn": version("scikit-learn")}
