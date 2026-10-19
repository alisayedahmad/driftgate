import numpy as np


JUDGED = 20_000


def judge(s, predict, step, n=JUDGED):
    """how a predictor really does at `step` of scenario s, on n fresh rows.

    predict maps x of shape (n, s.features) to class indices. accuracy is the
    expected share of right answers under the true P(y|x) at step, recall the
    expected share of each class's rows predicted as that class, of length
    s.labels, nan for a class with no rows then. labels are integrated
    out rather than drawn, and under label noise the truth is the clean one.
    every predictor judged at the same step sees the same rows.
    """
    x = s.fresh(step, n)
    p = np.zeros((n, s.labels))
    truth = s.proba(x, step)
    p[:, :truth.shape[1]] = truth
    pred = np.asarray(predict(x))
    hit = p[np.arange(n), pred]
    mass = p.sum(axis=0)
    caught = np.bincount(pred, weights=hit, minlength=s.labels)
    with np.errstate(invalid="ignore"):
        recall = np.where(mass > 0, caught / mass, np.nan)
    return {"accuracy": float(hit.mean()), "recall": recall}


def realized(s, predict, step, every):
    """the share of rows served at steps [step, step + every) that predict gets right.

    against their real labels, every one of them, where the pipeline only
    sees those that come back, and late. the truth there is for real data,
    with no P(y|x) to judge on; a stream with an end can cut the last block
    short.
    """
    stop = step + every if s.steps is None else min(step + every, s.steps)
    x, y = zip(*(s.batch(t) for t in range(step, stop)))
    return float((np.asarray(predict(np.concatenate(x))) == np.concatenate(y)).mean())
