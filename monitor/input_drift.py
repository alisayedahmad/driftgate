import numpy as np

from monitor.cusum import Monitor
from monitor.reference import window


class InputDrift(Monitor):
    """a CUSUM per feature on each step's mean, against the model's training window.

    the reference mean and spread come from every row of the window, see
    monitor.reference. update is a replay watcher, Monitor says what it keeps.
    """

    def __init__(self, registry, model, k=0.5):
        x = window(registry, model)
        self.mean, self.sd = x.mean(axis=0), x.std(axis=0)
        super().__init__(x.shape[1], k)

    def update(self, step, log):
        x = log.batch(step)["x"]
        z = (x.mean(axis=0) - self.mean) / (self.sd / np.sqrt(len(x)))
        self.record(step, z)
