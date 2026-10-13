import argparse

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrain.train import WINDOW, from_bytes, train
from serving.log import Log
from spine.provenance import code, versions
from spine.registry import fresh
from world.labels import LABELING, Arrivals
from world.scenarios import KINDS, Scenario
from world.stream import CLASSES, FEATURES


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


def replay(service, s, labeling, steps, watch=()):
    """serve the given steps of scenario s in order, labels coming back as labeling says.

    after each step, once its labels are in, every callable in watch gets (step, log).
    """
    arrivals = Arrivals(s.seed, labeling)
    for t in steps:
        x, y = s.batch(t)
        service.predict(t, x)
        arrivals.send(t, y)
        for served_at, rows, labels in arrivals.due(t):
            service.log.arrived(t, served_at, rows, labels)
        for see in watch:
            see(t, service.log)


class Rows(BaseModel):
    step: int
    rows: list[list[float]]


def http(service):
    """the service over HTTP, a thin layer the pipeline itself never goes through.

    POST /predict takes {"step": int, "rows": [[FEATURES floats], ...]} and
    answers {"model": hash, "proba": [[CLASSES + 1 floats], ...]}. 422 when there
    are no rows or a row isn't FEATURES finite numbers, 409 when the step isn't
    after the last one served. GET /model answers {"model": hash}.
    """
    app = FastAPI(title="driftgate")

    @app.post("/predict")
    def predict(request: Rows):
        if not request.rows or any(len(row) != FEATURES for row in request.rows):
            raise HTTPException(422, f"rows need {FEATURES} values each")
        x = np.array(request.rows)
        if not np.isfinite(x).all():
            raise HTTPException(422, "rows have to be finite")
        if service.log.last is not None and request.step <= service.log.last:
            raise HTTPException(409, f"step {request.step} isn't after {service.log.last}")
        return {"model": service.model, "proba": service.predict(request.step, x).tolist()}

    @app.get("/model")
    def model():
        return {"model": service.model}

    return app


def main():
    parser = argparse.ArgumentParser(description="serve a model trained on a scenario over HTTP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=KINDS, default="concept")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    registry = fresh(f"serve-{args.kind}-{args.seed}")
    s = Scenario.draw(args.seed, args.kind)
    model = train(registry, s, WINDOW, None, {"code": code(), "versions": versions()}, LABELING)
    print(f"serving {model[:12]}, trained at step {WINDOW} of seed {s.seed} ({s.kind}), "
          f"on http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(http(Service(registry, model)), host="127.0.0.1", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
