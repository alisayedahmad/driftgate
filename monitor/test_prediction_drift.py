import numpy as np
import pytest

from monitor.cusum import first_alarm
from monitor.prediction_drift import PredictionDrift
from monitor.reference import window
from retrain.test_train import ABOUT
from retrain.train import WINDOW, from_bytes, train
from serving.app import Service, replay
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario
from world.stream import CLASSES

H = 10.0


def watched(s, steps):
    registry = Registry(Store(":memory:"))
    model = train(registry, s, WINDOW, None, ABOUT, LABELING)
    monitor = PredictionDrift(registry, model)
    replay(Service(registry, model), s, LABELING, steps, [monitor.update])
    return registry, model, monitor


def test_the_reference_is_the_models_predictions_on_its_window():
    registry, model, monitor = watched(Scenario(3), range(WINDOW, WINDOW + 1))
    pred = from_bytes(registry.store.get(model)).predict(window(registry, model))
    assert np.array_equal(monitor.share, np.bincount(pred, minlength=CLASSES + 1) / len(pred))
    assert monitor.share[CLASSES] == 0


@pytest.mark.parametrize("kind", ["concept", "noise"])
def test_changes_that_leave_x_alone_are_invisible(kind):
    s = Scenario(3, kind, at=WINDOW + 40, size=0.2)
    steps = range(WINDOW, WINDOW + 120)
    assert watched(s, steps)[2].path == watched(s.twin(), steps)[2].path


@pytest.mark.parametrize("kind, size", [("covariate", 0.6), ("new", 0.15)])
def test_changes_that_move_predictions_are_caught(kind, size):
    s = Scenario(3, kind, at=WINDOW + 40, size=size)
    monitor = watched(s, range(WINDOW, WINDOW + 80))[2]
    alarm = first_alarm(monitor.path, H)
    assert alarm is not None and s.at <= monitor.steps[alarm] <= s.at + 10
