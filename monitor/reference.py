import json

import numpy as np

from world.scenarios import Scenario


def window(registry, model):
    """x of every row of the model's training window, labeled or not, from its record.

    the window runs from the first step it was fit on to the last it was tested
    on. monitors use all of it: watching inputs or predictions needs no labels,
    and with only the labeled rows the reference's own error becomes a constant
    offset that a CUSUM adds up step after step.
    """
    record = registry.get(model)
    fit, test = (json.loads(registry.store.get(record[field]))["recipe"]
                 for field in ("trained_on", "evaluated_on"))
    s = Scenario(**fit["scenario"])
    return np.concatenate([s.batch(t)[0] for t in range(fit["steps"][0], test["steps"][1])])
