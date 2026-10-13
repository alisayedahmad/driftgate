import numpy as np
import pytest

from retrain.test_train import ABOUT
from retrain.train import WINDOW, train
from serving.app import Service, replay
from serving.log import UNKNOWN
from serving.replay import RECENT, halfway, lag
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario

LAST = WINDOW + 150


@pytest.fixture(scope="module")
def run():
    s = Scenario(3, "concept", at=WINDOW + 60, size=0.25)
    registry = Registry(Store(":memory:"))
    service = Service(registry, train(registry, s, WINDOW, None, ABOUT, LABELING))
    replay(service, s, LABELING, range(WINDOW, LAST + 1))
    return s, service, lag(s, service.log, WINDOW, LAST)


def test_known_only_uses_labels_back_by_each_step(run):
    s, service, r = run
    for t in (WINDOW + 10, s.at + 20, LAST):
        log = service.log.as_of(t)
        recent = (log["step"] > t - RECENT) & (log["label"] != UNKNOWN)
        hits = log["proba"][recent].argmax(axis=1) == log["label"][recent]
        assert r["labels"][t - WINDOW] == recent.sum()
        assert r["known"][t - WINDOW] == hits.mean()


def test_true_is_the_expected_accuracy_of_what_was_served(run):
    s, service, r = run
    t = s.at + 10
    log = service.log.as_of(t)
    served = log["step"] == t
    p = s.proba(log["x"][served], t)
    assert r["true"][t - WINDOW] == p[np.arange(served.sum()), log["proba"][served].argmax(axis=1)].mean()


def test_what_is_known_trails_the_truth(run):
    s, _, r = run
    step = halfway(s, r, WINDOW)
    assert step is not None and step > s.at
    assert halfway(s.twin(), r, WINDOW) is None
