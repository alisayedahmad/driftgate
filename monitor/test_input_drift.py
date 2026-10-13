import numpy as np
import pytest

from monitor.cusum import first_alarm
from monitor.input_drift import InputDrift
from retrain.test_train import ABOUT
from retrain.train import WINDOW, train
from serving.app import Service, replay
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario

H = 10.0


def watched(s, steps):
    # one registry per run: an earlier scenario can give the same model with another recipe
    registry = Registry(Store(":memory:"))
    model = train(registry, s, WINDOW, None, ABOUT, LABELING)
    monitor = InputDrift(registry, model)
    replay(Service(registry, model), s, LABELING, steps, [monitor.update])
    return monitor


def test_the_reference_is_every_row_of_the_training_window():
    monitor = watched(Scenario(3), range(WINDOW, WINDOW + 1))
    x = np.concatenate([Scenario(3).batch(t)[0] for t in range(0, WINDOW)])
    assert np.array_equal(monitor.mean, x.mean(axis=0))
    assert np.array_equal(monitor.sd, x.std(axis=0))


@pytest.mark.parametrize("kind", ["concept", "noise"])
def test_changes_that_leave_x_alone_are_invisible(kind):
    s = Scenario(3, kind, at=WINDOW + 40, size=0.2)
    steps = range(WINDOW, WINDOW + 120)
    assert watched(s, steps).path == watched(s.twin(), steps).path


def test_a_covariate_shift_is_caught_within_a_few_steps():
    s = Scenario(3, "covariate", at=WINDOW + 40, size=0.6)
    monitor = watched(s, range(WINDOW, WINDOW + 80))
    alarm = first_alarm(monitor.path, H)
    assert alarm is not None and s.at <= monitor.steps[alarm] <= s.at + 3
