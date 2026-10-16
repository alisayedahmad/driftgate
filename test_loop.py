import numpy as np

from gate.policy import load
from loop import Loop
from retrain.test_train import ABOUT
from retrain.train import WINDOW, from_bytes, train
from serving.app import replay
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario
from world.truth import judge

AT = WINDOW + 40


def looped(s, stop):
    loop = Loop(Registry(Store(":memory:")), s, ABOUT, load())
    replay(loop.service, s, LABELING, range(WINDOW, stop), [loop.update])
    return loop


def test_each_promotion_after_a_concept_break_serves_a_better_model():
    # a first candidate fit on the few steps since the break, then one on more
    s = Scenario(3, "concept", at=AT, size=0.25)
    loop = looped(s, AT + 150)
    lineage = [r["model"] for r in loop.registry.lineage(loop.service.model)]
    assert lineage[-1] == loop.service.log.models[0] and len(lineage) >= 2
    assert loop.service.log.models == lineage[::-1]
    assert loop.monitors["errors"].model == loop.service.model
    judged = [judge(s, from_bytes(loop.registry.store.get(m)).predict, AT + 150)["accuracy"]
              for m in lineage[::-1]]
    assert (np.diff(judged) > 0).all()
    assert not any(what.startswith("roll back") for _, what in loop.events)


def test_label_noise_is_held_and_watched_again_later_each_time():
    loop = looped(Scenario(4, "noise", at=AT, size=0.4), AT + 500)
    assert loop.service.model == loop.service.log.models[0]
    holds = [step for step, what in loop.events if what.startswith("hold")]
    assert len(holds) >= 3
    gaps = np.diff(holds)
    assert (gaps >= load()["retry"] * 2 ** np.arange(len(gaps))).all()


def test_a_promotion_that_turns_out_worse_is_rolled_back():
    # the gate's mistake made on purpose: a model fit on 5 steps replaces a good one
    s, at = Scenario(3), WINDOW + 100
    loop = looped(s, at + 1)
    good = loop.service.model
    bad = train(loop.registry, s, at, good, ABOUT, LABELING,
                {"fit": [at - 25, at - 20], "weights": [], "test": [at - 20, at]})
    loop.service.switch(loop.registry, bad)
    loop.promoted(at, good)
    replay(loop.service, s, LABELING, range(at + 1, at + 40), [loop.update])
    assert loop.service.model == good
    assert [step for step, what in loop.events if what.startswith("roll back")] == [
        at + load()["rollback"]]
