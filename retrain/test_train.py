import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from retrain.train import CONFIG, TEST, WINDOW, check, fit, from_bytes, to_bytes, train
from spine.provenance import canonical, dataset, versions
from spine.registry import Registry
from spine.store import Store
from world.scenarios import Scenario

ABOUT = {"code": {"commit": "0" * 40, "dirty": False, "changes": None}, "versions": versions()}


def rows(seed, steps=200):
    s = Scenario(seed)
    x, y = zip(*(s.batch(t) for t in range(steps)))
    return np.concatenate(x), np.concatenate(y)


@pytest.fixture(scope="module")
def data():
    return rows(3)


def test_thread_count_does_not_change_the_bytes(data):
    x, y = data
    first = to_bytes(fit(x, y, CONFIG))
    with threadpool_limits(1):
        one = to_bytes(fit(x, y, CONFIG))
    with threadpool_limits(4):
        four = to_bytes(fit(x, y, CONFIG))
    assert first == one == four
    assert first != to_bytes(fit(*rows(4), CONFIG))


def test_another_process_gets_the_same_bytes(data):
    code = ("import hashlib; from retrain.train import CONFIG, fit, to_bytes; "
            "from retrain.test_train import rows; "
            "print(hashlib.sha256(to_bytes(fit(*rows(3), CONFIG))).hexdigest())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=Path(__file__).parents[1])
    assert out.stdout.strip() == hashlib.sha256(to_bytes(fit(*data, CONFIG))).hexdigest()


def test_bytes_load_back_into_the_same_model(data):
    x, y = data
    model = fit(x, y, CONFIG)
    again = from_bytes(to_bytes(model))
    assert np.array_equal(again.predict_proba(x), model.predict_proba(x))


def test_trained_models_rebuild_from_their_records():
    registry = Registry(Store(":memory:"))
    s = Scenario.draw(3, "concept")
    first = train(registry, s, 0, None, ABOUT)
    second = train(registry, s, s.at, first, ABOUT)
    assert [r["model"] for r in registry.lineage(second)] == [second, first]
    assert registry.get(second)["step"] == s.at + WINDOW
    assert check(registry, first) == check(registry, second) == []


def test_check_catches_a_record_that_lies_about_its_config():
    registry = Registry(Store(":memory:"))
    store = registry.store
    x, y, trained_on = dataset(Scenario(3), 0, WINDOW)
    tx, ty, evaluated_on = dataset(Scenario(3), WINDOW, WINDOW + TEST)
    model = fit(x, y, {**CONFIG, "C": 0.01})
    h = registry.register({
        "model": store.put(to_bytes(model)),
        "trained_on": store.put(canonical(trained_on)),
        "evaluated_on": store.put(canonical(evaluated_on)),
        "config": store.put(canonical(CONFIG)),
        **ABOUT,
        "metrics": {"accuracy": float((model.predict(tx) == ty).mean())},
        "parent": None,
        "step": WINDOW,
    })
    assert "the refit model has other bytes" in check(registry, h)
