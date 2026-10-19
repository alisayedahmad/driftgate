import json

import numpy as np

from retrain.train import from_bytes
from spine.provenance import canonical, dataset, stream


def paired(new, old, y, labels):
    """how much better new does than old on the same rows, with the error of that estimate.

    new and old are bools, whether each model got each row right, y the labels.
    gain is the difference in accuracy and se its standard error, from the
    per row differences, which is much smaller than for two separate test sets
    when the models agree on most rows. per class, over the rows of that label,
    the same for recall, in arrays of length labels, the labels the stream can
    give. a gain is nan without rows, a se without two.
    """
    d = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    y = np.asarray(y)
    out = {"rows": len(d), **_mean(d), "class rows": np.bincount(y, minlength=labels),
           "class gain": np.full(labels, np.nan), "class se": np.full(labels, np.nan)}
    for c in np.unique(y):
        stats = _mean(d[y == c])
        out["class gain"][c], out["class se"][c] = stats["gain"], stats["se"]
    return out


def _mean(d):
    n = len(d)
    return {"gain": float(d.mean()) if n else np.nan,
            "se": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan}


def compare(registry, candidate, incumbent, look, first=None):
    """the candidate against the incumbent on the rows it hasn't seen, as known at step look.

    those are the rows of the steps from first, by default the first the
    candidate was tested on, up to look excluded, with the labels that had
    come back by look. look is the step it was trained at or later; whichever
    model served those rows, the other answers them as if it had been
    shadowing it. returns the document naming the rows, stored, and
    paired()'s result.
    """
    record = registry.get(candidate)
    recipe = json.loads(registry.store.get(record["evaluated_on"]))["recipe"]
    if look < recipe["as_of"]:
        raise ValueError(f"the candidate was trained at {recipe['as_of']}, it can't be judged at {look}")
    first = recipe["steps"][0] if first is None else first
    s = stream(recipe["scenario"])
    x, y, doc = dataset(s, first, look, recipe["labeling"], look)
    registry.store.put(canonical(doc))
    new, old = (from_bytes(registry.store.get(m)).predict(x) == y for m in (candidate, incumbent))
    return doc, paired(new, old, y, s.labels)
