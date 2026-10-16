import pytest

from retrain.test_train import ABOUT
from retrain.test_window import AT, DECIDED
from retrain.train import WINDOW, train
from retrain.window import Window
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario


def trained(s, decision):
    """(registry, serving, candidate, step): a model at WINDOW, a candidate after decision."""
    registry = Registry(Store(":memory:"))
    serving = train(registry, s, WINDOW, None, ABOUT, LABELING)
    step, plan = Window().plan(decision)
    return registry, serving, train(registry, s, step, serving, ABOUT, LABELING, plan), step


@pytest.fixture(scope="session")
def concept():
    s = Scenario(3, "concept", at=AT, size=0.25)
    return s, *trained(s, DECIDED)
