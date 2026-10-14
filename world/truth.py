import numpy as np

from world.stream import CLASSES

JUDGED = 20_000


def judge(s, predict, step, n=JUDGED):
    """how a predictor really does at `step` of scenario s, on n fresh rows.

    predict maps x of shape (n, FEATURES) to class indices. accuracy is the
    expected share of right answers under the true P(y|x) at step, recall the
    expected share of each class's rows predicted as that class, of length
    CLASSES + 1, nan for a class with no rows then. labels are integrated
    out rather than drawn, and under label noise the truth is the clean one.
    every predictor judged at the same step sees the same rows.
    """
    x = s.fresh(step, n)
    p = np.zeros((n, CLASSES + 1))
    truth = s.proba(x, step)
    p[:, :truth.shape[1]] = truth
    pred = np.asarray(predict(x))
    hit = p[np.arange(n), pred]
    mass = p.sum(axis=0)
    caught = np.bincount(pred, weights=hit, minlength=CLASSES + 1)
    with np.errstate(invalid="ignore"):
        recall = np.where(mass > 0, caught / mass, np.nan)
    return {"accuracy": float(hit.mean()), "recall": recall}
