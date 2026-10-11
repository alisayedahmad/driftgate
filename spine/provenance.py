import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

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
    """the (x, y) a recipe stands for: its scenario's batches over steps [first, stop)."""
    s = Scenario(**recipe["scenario"])
    x, y = zip(*(s.batch(t) for t in range(*recipe["steps"])))
    return np.concatenate(x), np.concatenate(y)


def dataset(s, first, stop):
    """rows of scenario s over steps [first, stop), as x, y and the document naming them.

    the document holds the recipe and the hash of the rows, not the rows.
    """
    recipe = {"scenario": asdict(s), "steps": [first, stop]}
    x, y = rows(recipe)
    return x, y, {"recipe": recipe, "data": data_hash(x, y)}


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
