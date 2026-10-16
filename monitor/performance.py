import json

import numpy as np

from monitor.cusum import Monitor
from spine.provenance import rows
from world.stream import CLASSES


class Performance(Monitor):
    """a CUSUM on the error rate of the labels that land at each step, for the rows the model served.

    the reference is one minus the accuracy the model was tested at, from its
    registry record. labels of rows another model served are left out, and a
    step where none of its own land leaves the CUSUM as it was. update is a
    replay watcher, Monitor says what it keeps. its steps are the ones labels
    land at, which come after the rows they belong to were served.
    """

    def __init__(self, registry, model, k=0.5):
        self.model = model
        self.error = 1 - registry.get(model)["metrics"]["accuracy"]
        super().__init__(1, k)

    def update(self, step, log):
        landed = log.landed(step)
        mine = landed["model"] == (log.models.index(self.model) if self.model in log.models else -1)
        n = int(mine.sum())
        z = np.nan
        if n:
            wrong = (landed["proba"][mine].argmax(axis=1) != landed["label"][mine]).mean()
            z = (wrong - self.error) / np.sqrt(self.error * (1 - self.error) / n)
        self.record(step, [z])


class LabelShares(Monitor):
    """a CUSUM per class on its share of the labels that land at each step.

    the reference shares are those of the labels known when the model was
    trained, the ones it was fit and tested on. a label of a class that never
    showed up there sends the statistic to infinity, an unseen class being an
    alarm on its own. update is a replay watcher, like Performance's.
    """

    def __init__(self, registry, model, k=0.5):
        record = registry.get(model)
        labels = np.concatenate([rows(json.loads(registry.store.get(record[field]))["recipe"])[1]
                                 for field in ("trained_on", "evaluated_on")])
        self.share = np.bincount(labels, minlength=CLASSES + 1) / len(labels)
        super().__init__(CLASSES + 1, k)

    def update(self, step, log):
        y = log.landed(step)["label"]
        z = np.full(CLASSES + 1, np.nan)
        if len(y):
            share = np.bincount(y, minlength=CLASSES + 1) / len(y)
            known = self.share > 0
            z[known] = (share[known] - self.share[known]) / np.sqrt(
                self.share[known] * (1 - self.share[known]) / len(y))
            z[~known & (share > 0)] = np.inf
        self.record(step, z)
