import numpy as np

from retrain.train import from_bytes


def best_available(registries, steps, score):
    """the score at each of steps of the best model trained by then, and that model.

    the models are those of every registry given, the runs of several policies
    on one stream, so that every policy is held to the same oracle. a model is
    available from the step it was trained at on, whether it ever served or
    not. score(predict, step) is how a model does at a step: the judge's
    accuracy on fresh rows for a scenario, or the accuracy on every row's real
    label for real data. it has to score a model that served there exactly as
    it scored serving. returns an array of scores and a list of model hashes.
    """
    records = {r["model"]: (r, registry) for registry in registries for r in registry.records()}
    predict = {m: from_bytes(registry.store.get(m)).predict for m, (_, registry) in records.items()}
    records = [r for r, _ in records.values()]
    best, which = [], []
    for t in steps:
        scores = {r["model"]: score(predict[r["model"]], t) for r in records if r["step"] <= t}
        model = max(scores, key=scores.get)
        best.append(scores[model])
        which.append(model)
    return np.array(best), which
