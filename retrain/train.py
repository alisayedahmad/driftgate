import argparse
import hashlib
import json
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from spine.provenance import canonical, code, dataset, rows, verify, versions
from spine.registry import Registry
from spine.store import Store
from world.scenarios import KINDS, Scenario

CONFIG = {"C": 1.0, "max_iter": 1000}
WINDOW = 200
TEST = 50


def fit(x, y, config):
    """multinomial logistic regression of labels y on rows x.

    config goes to LogisticRegression as keyword arguments, CONFIG is the default.
    """
    return LogisticRegression(**config).fit(x, y)


def to_bytes(model):
    """pickle protocol 5. the same fit gives the same bytes in any process."""
    return pickle.dumps(model, protocol=5)


def from_bytes(data):
    # pickle runs code when it loads, only feed it bytes from our own store
    return pickle.loads(data)


def train(registry, s, first, parent, about):
    """fit on WINDOW steps of s from first, test on the TEST after, register it.

    about is {"code": ..., "versions": ...} for the record, read once per run.
    the model counts as trained at the step after its window. returns its hash.
    """
    store = registry.store
    x, y, trained_on = dataset(s, first, first + WINDOW)
    tx, ty, evaluated_on = dataset(s, first + WINDOW, first + WINDOW + TEST)
    model = fit(x, y, CONFIG)
    return registry.register({
        "model": store.put(to_bytes(model)),
        "trained_on": store.put(canonical(trained_on)),
        "evaluated_on": store.put(canonical(evaluated_on)),
        "config": store.put(canonical(CONFIG)),
        **about,
        "metrics": {"accuracy": float((model.predict(tx) == ty).mean())},
        "parent": parent,
        "step": first + WINDOW,
    })


def check(registry, model):
    """rebuild a registered model from its record alone, return what doesn't match.

    regenerates both datasets from their recipes, refits with the stored config,
    and recomputes the metrics. an empty list means the record holds up.
    """
    record = registry.get(model)
    doc = {field: json.loads(registry.store.get(record[field]))
           for field in ("trained_on", "evaluated_on", "config")}
    problems = [f"{field} rows don't hash the same" for field in ("trained_on", "evaluated_on")
                if not verify(doc[field])]
    refit = fit(*rows(doc["trained_on"]["recipe"]), doc["config"])
    if hashlib.sha256(to_bytes(refit)).hexdigest() != model:
        problems.append("the refit model has other bytes")
    tx, ty = rows(doc["evaluated_on"]["recipe"])
    if {"accuracy": float((refit.predict(tx) == ty).mean())} != record["metrics"]:
        problems.append("the metrics come out different")
    return problems


def describe(registry, model):
    r = registry.get(model)
    trained = json.loads(registry.store.get(r["trained_on"]))["recipe"]["steps"]
    tested = json.loads(registry.store.get(r["evaluated_on"]))["recipe"]["steps"]
    parent = r["parent"][:12] if r["parent"] else "none"
    dirty = ", uncommitted changes" if r["code"]["dirty"] else ""
    return (f"model   {model[:12]} trained on steps {trained[0]}-{trained[1] - 1}, "
            f"parent {parent}, accuracy {r['metrics']['accuracy']:.3f} on steps "
            f"{tested[0]}-{tested[1] - 1}, code {r['code']['commit'][:7]}{dirty}")


def main():
    parser = argparse.ArgumentParser(description="train two models on a scenario and trace them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=KINDS, default="concept")
    args = parser.parse_args()

    # one registry per run, a rerun after an edit would clash with the old records
    path = Path.home() / ".local/share/driftgate" / f"train-{args.kind}-{args.seed}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    registry = Registry(Store(path))
    about = {"code": code(), "versions": versions()}

    s = Scenario.draw(args.seed, args.kind)
    first = train(registry, s, 0, None, about)
    second = train(registry, s, s.at if s.at is not None else 500, first, about)
    print(f"registry {path}")
    for model in (first, second):
        print(describe(registry, model))
    print("lineage " + " <- ".join(r["model"][:12] for r in registry.lineage(second)))
    problems = {m[:12]: check(registry, m) for m in (first, second)}
    print("check   " + ("every record rebuilds to the same hashes and metrics"
                        if not any(problems.values()) else str(problems)))


if __name__ == "__main__":
    main()
