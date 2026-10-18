import numpy as np

from retrain.train import from_bytes
from world.truth import judge


def best_available(s, registries, steps, n):
    """the true accuracy at each of steps of the best model trained by then, and that model.

    the models are those of every registry given, the runs of several policies
    on one scenario, so that every policy is held to the same oracle. a model
    is available from the step it was trained at on, whether it ever served or
    not. every model is judged on the same n fresh rows at a step, so a model
    that served there scores exactly what it scored serving. returns an array
    of accuracies and a list of model hashes.
    """
    records = {r["model"]: (r, registry) for registry in registries for r in registry.records()}
    predict = {m: from_bytes(registry.store.get(m)).predict for m, (_, registry) in records.items()}
    records = [r for r, _ in records.values()]
    best, which = [], []
    for t in steps:
        scores = {r["model"]: judge(s, predict[r["model"]], t, n)["accuracy"]
                  for r in records if r["step"] <= t}
        model = max(scores, key=scores.get)
        best.append(scores[model])
        which.append(model)
    return np.array(best), which
