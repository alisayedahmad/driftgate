import numpy as np
import pytest

from retrain.test_train import ABOUT
from retrain.train import TEST, WINDOW, train
from serving.app import Service, replay
from serving.log import UNKNOWN
from spine.provenance import dataset
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING, fate
from world.scenarios import Scenario
from world.stream import CLASSES

STEPS = range(120)


@pytest.fixture(scope="module")
def served():
    registry = Registry(Store(":memory:"))
    s = Scenario.draw(3, "noise")
    model = train(registry, s, WINDOW, None, ABOUT, LABELING)
    return registry, s, model


def test_only_registered_models_are_served(served):
    registry, _, _ = served
    with pytest.raises(ValueError):
        Service(registry, "0" * 64)


def test_probabilities_cover_every_class(served):
    registry, s, model = served
    service = Service(registry, model)
    x, _ = s.batch(0)
    p = service.predict(0, x)
    assert p.shape == (len(x), CLASSES + 1)
    assert np.allclose(p.sum(axis=1), 1)
    assert (p[:, CLASSES] == 0).all()
    assert np.array_equal(p[:, :CLASSES], service.estimator.predict_proba(x))
    assert np.array_equal(service.log.as_of(0)["proba"], p)


def test_labels_reach_the_log_when_fate_says(served):
    registry, s, model = served
    service = Service(registry, model)
    replay(service, s, LABELING, STEPS)
    known = service.log.as_of(STEPS[-1])
    expected = []
    for t in STEPS:
        labeled, arrives = fate(s.seed, t, LABELING)
        expected.append(np.where(labeled & (arrives <= STEPS[-1]), arrives, UNKNOWN))
    assert np.array_equal(known["arrived"], np.concatenate(expected))
    y = np.concatenate([s.batch(t)[1] for t in STEPS])
    has = known["label"] != UNKNOWN
    assert np.array_equal(known["label"][has], y[has])


def test_a_scenario_and_its_twin_get_labels_at_the_same_steps(served):
    registry, s, model = served
    logs = []
    for scenario in (s, s.twin()):
        service = Service(registry, model)
        replay(service, scenario, LABELING, range(s.at - 30, s.at + 30))
        logs.append(service.log.as_of(s.at + 29))
    assert np.array_equal(logs[0]["arrived"], logs[1]["arrived"])
    assert not np.array_equal(logs[0]["label"], logs[1]["label"])


def test_a_training_set_is_what_the_log_knew_at_that_step(served):
    registry, s, model = served
    service = Service(registry, model)
    replay(service, s, LABELING, range(WINDOW + 1))
    known = service.log.as_of(WINDOW)
    keep = (known["label"] != UNKNOWN) & (known["step"] < WINDOW - TEST)
    x, y, _ = dataset(s, 0, WINDOW - TEST, LABELING, WINDOW)
    assert np.array_equal(known["x"][keep], x)
    assert np.array_equal(known["label"][keep], y)
