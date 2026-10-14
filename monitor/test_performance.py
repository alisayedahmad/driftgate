import json

import numpy as np

from monitor.cusum import first_alarm
from monitor.performance import LabelShares, Performance
from retrain.test_train import ABOUT
from retrain.train import WINDOW, train
from serving.app import Service, replay
from spine.provenance import rows
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario
from world.stream import CLASSES

H = 10.0
AT = WINDOW + 40
STEPS = range(WINDOW, AT + 200)


def watched(s, steps=STEPS):
    registry = Registry(Store(":memory:"))
    model = train(registry, s, WINDOW, None, ABOUT, LABELING)
    monitors = Performance(registry, model), LabelShares(registry, model)
    replay(Service(registry, model), s, LABELING, steps, [m.update for m in monitors])
    return registry, model, monitors


def alarm(monitor):
    i = first_alarm(monitor.path, H)
    return None if i is None else monitor.steps[i]


def test_the_references_come_from_the_record():
    registry, model, (performance, shares) = watched(Scenario(3), range(WINDOW, WINDOW + 1))
    record = registry.get(model)
    assert performance.error == 1 - record["metrics"]["accuracy"]
    labels = np.concatenate([rows(json.loads(registry.store.get(record[field]))["recipe"])[1]
                             for field in ("trained_on", "evaluated_on")])
    assert np.array_equal(shares.share, np.bincount(labels, minlength=CLASSES + 1) / len(labels))


def test_a_concept_break_shows_in_the_errors_but_not_in_the_labels():
    _, _, (performance, shares) = watched(Scenario(3, "concept", at=AT, size=0.25))
    assert AT < alarm(performance) <= AT + 100
    assert alarm(shares) is None


def test_label_noise_shows_in_both():
    # seed 3 relabels part of class 0, the most common, as class 1
    _, _, (performance, shares) = watched(Scenario(3, "noise", at=AT, size=0.4))
    assert AT < alarm(performance) <= AT + 100
    assert AT < alarm(shares) <= AT + 100


def test_the_first_label_of_a_new_class_is_an_alarm():
    _, _, (_, shares) = watched(Scenario(3, "new", at=AT, size=0.05), range(WINDOW, AT + 60))
    step = alarm(shares)
    assert step is not None and step > AT
    assert shares.path[shares.steps.index(step)] == np.inf
    # its stream had no value until then
    assert shares.starts[shares.steps.index(step)] == step
