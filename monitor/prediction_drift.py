import numpy as np

from monitor.cusum import Monitor
from monitor.reference import window
from retrain.train import from_bytes


class PredictionDrift(Monitor):
    """a CUSUM per class on the share of each step's rows predicted as that class.

    the reference shares are the model's own predictions on every row of its
    training window, see monitor.reference. classes it never predicted there
    are left out. update is a replay watcher, Monitor says what it keeps.
    """

    def __init__(self, registry, model, k=0.5):
        pred = from_bytes(registry.store.get(model)).predict(window(registry, model))
        labels = registry.stream(model).labels
        self.share = np.bincount(pred, minlength=labels) / len(pred)
        super().__init__(labels, k)

    def update(self, step, log):
        pred = log.batch(step)["proba"].argmax(axis=1)
        share = np.bincount(pred, minlength=len(self.share)) / len(pred)
        z = np.full(len(self.share), np.nan)
        known = self.share > 0
        z[known] = (share[known] - self.share[known]) / np.sqrt(
            self.share[known] * (1 - self.share[known]) / len(pred))
        self.record(step, z)
