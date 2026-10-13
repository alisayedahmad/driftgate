import numpy as np

from world.stream import LABELS, ROWS

LABELING = {"rate": 0.3, "median": 10.0, "spread": 1.0}


def fate(seed, step, labeling, n=ROWS):
    """which rows of `step` ever get a label, and the step it arrives at.

    labeling is like LABELING: each row is labeled with probability rate, after
    a lognormal delay with that median in steps and spread as the sigma of its
    log, rounded up. returns a bool mask of shape (n,) and arrival steps of
    shape (n,), always after `step`. depends on the seed and the step only, so
    a scenario and its twin label the same rows at the same steps.
    """
    g = np.random.default_rng([seed, LABELS, step])
    labeled = g.random(n) < labeling["rate"]
    delay = np.ceil(labeling["median"] * np.exp(labeling["spread"] * g.standard_normal(n)))
    return labeled, step + np.maximum(delay, 1).astype(np.int64)


class Arrivals:
    """labels on their way back, as fate decides for a seed and a labeling.

    send each served step's labels; due(t) gives back those arriving at t as
    (served_at, rows, labels), rows being indices within step served_at.
    """

    def __init__(self, seed, labeling):
        self.seed, self.labeling, self.pending = seed, labeling, {}

    def send(self, step, y):
        labeled, arrives = fate(self.seed, step, self.labeling, len(y))
        for at in np.unique(arrives[labeled]):
            rows = np.flatnonzero(labeled & (arrives == at))
            self.pending.setdefault(int(at), []).append((step, rows, y[rows]))

    def due(self, step):
        return self.pending.pop(step, [])
