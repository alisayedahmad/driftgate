import numpy as np

from retrain.train import from_bytes
from serving.log import Log
from world.labels import Arrivals
from world.stream import CLASSES


class Service:
    """predictions from one registered model, every one of them logged."""

    def __init__(self, registry, model):
        if registry.get(model) is None:
            raise ValueError(f"model {model} isn't registered, it can't be served")
        self.model = model
        self.estimator = from_bytes(registry.store.get(model))
        self.log = Log()

    def predict(self, step, x):
        """probabilities for the rows x of `step`, as (n, CLASSES + 1), logged first.

        classes the model never saw get 0.
        """
        p = np.zeros((len(x), CLASSES + 1))
        p[:, self.estimator.classes_] = self.estimator.predict_proba(x)
        self.log.served(step, x, self.model, p)
        return p


def replay(service, s, labeling, steps):
    """serve the given steps of scenario s in order, labels coming back as labeling says."""
    arrivals = Arrivals(s.seed, labeling)
    for t in steps:
        x, y = s.batch(t)
        service.predict(t, x)
        arrivals.send(t, y)
        for served_at, rows, labels in arrivals.due(t):
            service.log.arrived(t, served_at, rows, labels)
