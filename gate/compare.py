import json

import numpy as np

from retrain.train import from_bytes
from spine.provenance import canonical, dataset
from world.scenarios import Scenario
from world.stream import CLASSES


def paired(new, old, y):
    """how much better new does than old on the same rows, with the error of that estimate.

    new and old are bools, whether each model got each row right, y the labels.
    gain is the difference in accuracy and se its standard error, from the
    per row differences, which is much smaller than for two separate test sets
    when the models agree on most rows. per class, over the rows of that label,
    the same for recall, in arrays of length CLASSES + 1. a gain is nan without
    rows, a se without two.
    """
    d = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    y = np.asarray(y)
    out = {"rows": len(d), **_mean(d), "class rows": np.bincount(y, minlength=CLASSES + 1),
           "class gain": np.full(CLASSES + 1, np.nan), "class se": np.full(CLASSES + 1, np.nan)}
    for c in np.unique(y):
        stats = _mean(d[y == c])
        out["class gain"][c], out["class se"][c] = stats["gain"], stats["se"]
    return out


def _mean(d):
    n = len(d)
    return {"gain": float(d.mean()) if n else np.nan,
            "se": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan}


def compare(registry, candidate, incumbent, look):
    """the candidate against the incumbent on the candidate's held-out rows, as known at step look.

    those are the rows of the steps the candidate was tested on, with the
    labels that had come back by look, which is the step it was trained at or
    later. returns the document naming those rows, stored, and paired()'s result.
    """
    record = registry.get(candidate)
    recipe = json.loads(registry.store.get(record["evaluated_on"]))["recipe"]
    if look < recipe["as_of"]:
        raise ValueError(f"the candidate was trained at {recipe['as_of']}, it can't be judged at {look}")
    x, y, doc = dataset(Scenario(**recipe["scenario"]), *recipe["steps"], recipe["labeling"], look)
    registry.store.put(canonical(doc))
    new, old = (from_bytes(registry.store.get(m)).predict(x) == y for m in (candidate, incumbent))
    return doc, paired(new, old, y)
