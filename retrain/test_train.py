import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from retrain.train import CONFIG, fit, from_bytes, to_bytes
from world.scenarios import Scenario


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
