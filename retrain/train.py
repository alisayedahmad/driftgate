import argparse
import hashlib
import json
import pickle

from sklearn.linear_model import LogisticRegression

from spine.provenance import canonical, code, dataset, row_weights, rows, verify, versions
from spine.registry import fresh
from world.labels import LABELING
from world.scenarios import KINDS, Scenario

CONFIG = {"C": 1.0, "max_iter": 1000}
WINDOW = 200
TEST = 50


def fit(x, y, config, weights=None):
    """multinomial logistic regression of labels y on rows x, each row counting its weight.

    config goes to LogisticRegression as keyword arguments, CONFIG is the default.
    no weights is every row weighing 1, and gives the same bytes as all ones.
    """
    return LogisticRegression(**config).fit(x, y, sample_weight=weights)


def to_bytes(model):
    """pickle protocol 5. the same fit gives the same bytes in any process."""
    return pickle.dumps(model, protocol=5)


def from_bytes(data):
    # pickle runs code when it loads, only feed it bytes from our own store
    return pickle.loads(data)


def train(registry, s, step, parent, about, labeling, plan=None):
    """fit a model at `step` on what was known then, test it, register it.

    only rows of s whose label had come back by step under labeling are used.
    plan is a dict as retrain.window makes them: fit and test, [first, stop)
    step ranges to fit and test on, and weights, [first, stop, w] lists for
    the fit rows. without one the window is the WINDOW steps before step, the
    first WINDOW - TEST to fit on and the last TEST to test on, every row
    weighing 1. about is {"code": ..., "versions": ...}, read once per run.
    returns the model hash.
    """
    if plan is None:
        plan = {"fit": [step - WINDOW, step - TEST], "weights": [], "test": [step - TEST, step]}
    store = registry.store
    x, y, trained_on = dataset(s, *plan["fit"], labeling, step, plan["weights"])
    tx, ty, evaluated_on = dataset(s, *plan["test"], labeling, step)
    model = fit(x, y, CONFIG, row_weights(trained_on["recipe"]))
    return registry.register({
        "model": store.put(to_bytes(model)),
        "trained_on": store.put(canonical(trained_on)),
        "evaluated_on": store.put(canonical(evaluated_on)),
        "config": store.put(canonical(CONFIG)),
        **about,
        "metrics": score(model, tx, ty),
        "parent": parent,
        "step": step,
    })


def score(model, x, y):
    return {"accuracy": float((model.predict(x) == y).mean()), "rows": len(y)}


def check(registry, model):
    """rebuild a registered model from its record alone, return what doesn't match.

    regenerates both datasets from their recipes, refits with the stored config
    and the weights in the recipe, and recomputes the metrics. an empty list
    means the record holds up.
    """
    record = registry.get(model)
    doc = {field: json.loads(registry.store.get(record[field]))
           for field in ("trained_on", "evaluated_on", "config")}
    problems = [f"{field} rows don't hash the same" for field in ("trained_on", "evaluated_on")
                if not verify(doc[field])]
    recipe = doc["trained_on"]["recipe"]
    refit = fit(*rows(recipe), doc["config"], row_weights(recipe))
    if hashlib.sha256(to_bytes(refit)).hexdigest() != model:
        problems.append("the refit model has other bytes")
    if score(refit, *rows(doc["evaluated_on"]["recipe"])) != record["metrics"]:
        problems.append("the metrics come out different")
    return problems


def describe(registry, model):
    r = registry.get(model)
    fitted, tested = (json.loads(registry.store.get(r[field]))
                      for field in ("trained_on", "evaluated_on"))
    parent = r["parent"][:12] if r["parent"] else "none"
    dirty = ", uncommitted changes" if r["code"]["dirty"] else ""
    (a, b), (c, d) = fitted["recipe"]["steps"], tested["recipe"]["steps"]
    return (f"model   {model[:12]} at step {r['step']}, parent {parent}, "
            f"code {r['code']['commit'][:7]}{dirty}\n"
            f"        fit on {fitted['rows']} labels from steps {a}-{b - 1}, accuracy "
            f"{r['metrics']['accuracy']:.3f} on {tested['rows']} labels from steps {c}-{d - 1}")


def main():
    parser = argparse.ArgumentParser(description="train two models on a scenario and trace them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=KINDS, default="concept")
    args = parser.parse_args()

    registry = fresh(f"train-{args.kind}-{args.seed}")
    about = {"code": code(), "versions": versions()}

    s = Scenario.draw(args.seed, args.kind)
    first = train(registry, s, WINDOW, None, about, LABELING)
    after = s.at if s.at is not None else 500
    second = train(registry, s, after + WINDOW, first, about, LABELING)
    print(f"registry {registry.store.path}")
    for model in (first, second):
        print(describe(registry, model))
    print("lineage " + " <- ".join(r["model"][:12] for r in registry.lineage(second)))
    problems = {m[:12]: check(registry, m) for m in (first, second)}
    print("check   " + ("every record rebuilds to the same hashes and metrics"
                        if not any(problems.values()) else str(problems)))


if __name__ == "__main__":
    main()
