import numpy as np

from monitor.cusum import Cusum
from monitor.reference import window
from retrain.train import from_bytes
from world.stream import CLASSES


class PredictionDrift:
    """a CUSUM per class on the share of each step's rows predicted as that class.

    the reference shares are the model's own predictions on every row of its
    training window, see monitor.reference. classes it never predicted there
    are left out. update is a replay watcher; after step t, path holds the
    CUSUM statistic and steps the t.
    """

    def __init__(self, registry, model, k=0.5):
        pred = from_bytes(registry.store.get(model)).predict(window(registry, model))
        self.share = np.bincount(pred, minlength=CLASSES + 1) / len(pred)
        self.cusum = Cusum(CLASSES + 1, k)
        self.steps, self.path = [], []

    def update(self, step, log):
        pred = log.batch(step)["proba"].argmax(axis=1)
        share = np.bincount(pred, minlength=CLASSES + 1) / len(pred)
        z = np.full(CLASSES + 1, np.nan)
        known = self.share > 0
        z[known] = (share[known] - self.share[known]) / np.sqrt(
            self.share[known] * (1 - self.share[known]) / len(pred))
        self.steps.append(step)
        self.path.append(self.cusum.update(z))
