import json

import numpy as np

from monitor.cusum import Cusum
from world.scenarios import Scenario
from world.stream import FEATURES


class InputDrift:
    """a CUSUM per feature on each step's mean, against the model's training window.

    the reference mean and spread come from every row of the window, labeled
    or not, regenerated from the model's registry record. update is a replay
    watcher; after step t, path holds the CUSUM statistic and steps the t.
    """

    def __init__(self, registry, model, k=0.5):
        record = registry.get(model)
        fit, test = (json.loads(registry.store.get(record[field]))["recipe"]
                     for field in ("trained_on", "evaluated_on"))
        # all rows, not only the labeled ones the model saw: the reference's own
        # error is a constant offset the CUSUM would add up every step
        s = Scenario(**fit["scenario"])
        x = np.concatenate([s.batch(t)[0] for t in range(fit["steps"][0], test["steps"][1])])
        self.mean, self.sd = x.mean(axis=0), x.std(axis=0)
        self.cusum = Cusum(FEATURES, k)
        self.steps, self.path = [], []

    def update(self, step, log):
        x = log.batch(step)["x"]
        z = (x.mean(axis=0) - self.mean) / (self.sd / np.sqrt(len(x)))
        self.steps.append(step)
        self.path.append(self.cusum.update(z))
