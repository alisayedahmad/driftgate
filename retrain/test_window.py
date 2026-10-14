import hashlib
import json

import pytest

from retrain.test_train import ABOUT
from retrain.train import CONFIG, WINDOW, check, fit, from_bytes, to_bytes, train
from retrain.trigger import Decision
from retrain.window import Window
from spine.provenance import rows
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario
from world.truth import judge

AT = WINDOW + 40
# what the informed trigger decides on this concept break
DECIDED = Decision(AT + 12, "retrain", "only the errors went up, the concept changed", AT + 1)


def decision(step, start):
    return Decision(step, "retrain", "", start)


def test_the_plan_follows_the_knobs():
    w = Window(margin=10, old_weight=0.2, old_share=0.5, patience=20, gate=20)
    # trained at 320 on [120, 300), cut at 260, the 70 old steps nearest the cut kept
    assert w.plan(decision(300, 270)) == (320, {"fit": [190, 300], "weights": [[190, 260, 0.2]],
                                                "test": [300, 320]})


def test_the_cut_stays_inside_the_window():
    assert Window().plan(decision(300, 50)) == (320, {"fit": [120, 300], "weights": [],
                                                      "test": [300, 320]})


def test_a_cut_among_the_held_out_steps_is_refused():
    with pytest.raises(ValueError):
        Window(margin=0, patience=0, gate=20).plan(decision(300, 290))


def candidates(s, windows):
    registry = Registry(Store(":memory:"))
    serving = train(registry, s, WINDOW, None, ABOUT, LABELING)
    out = []
    for w in windows:
        step, plan = w.plan(DECIDED)
        out.append((step, plan, train(registry, s, step, serving, ABOUT, LABELING, plan)))
    return registry, serving, out


def test_a_weighted_candidate_rebuilds_from_its_record():
    registry, _, [(_, plan, model)] = candidates(Scenario(3, "concept", at=AT, size=0.25),
                                                  [Window(old_share=1.0, old_weight=0.2)])
    assert check(registry, model) == []
    recipe = json.loads(registry.store.get(registry.get(model)["trained_on"]))["recipe"]
    assert recipe["weights"] == plan["weights"]
    assert hashlib.sha256(to_bytes(fit(*rows(recipe), CONFIG))).hexdigest() != model


def test_cutting_at_the_change_beats_keeping_the_whole_window():
    s = Scenario(3, "concept", at=AT, size=0.25)
    registry, serving, [(step, _, cut), (_, _, whole)] = candidates(
        s, [Window(), Window(old_share=1.0)])
    judged = {name: judge(s, from_bytes(registry.store.get(m)).predict, step)["accuracy"]
              for name, m in [("serving", serving), ("cut", cut), ("whole", whole)]}
    assert judged["cut"] > judged["whole"] > judged["serving"]
