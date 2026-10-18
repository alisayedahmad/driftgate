import numpy as np

from eval.oracle import best_available
from gate.policy import load
from loop import Loop
from retrain.test_train import ABOUT
from retrain.train import WINDOW
from serving.app import replay
from spine.registry import Registry
from spine.store import Store
from world.labels import LABELING
from world.scenarios import Scenario
from world.truth import judge

AT = WINDOW + 40
N = 2000


def test_the_best_available_model_is_never_below_the_one_serving():
    s = Scenario(3, "concept", at=AT, size=0.25)
    loop = Loop(Registry(Store(":memory:")), s, ABOUT, load())
    steps = np.arange(WINDOW, AT + 150, 10)
    served = {}

    def truth(step, log):
        if step in steps:
            served[step] = judge(s, loop.service.estimator.predict, step, N)["accuracy"]

    replay(loop.service, s, LABELING, range(WINDOW, AT + 150), [truth, loop.update])
    best, which = best_available(s, [loop.registry], steps, N)
    got = np.array([served[t] for t in steps])
    assert (best >= got).all()
    # the first model alone is available until the first candidate is trained,
    # which is before the gate lets it serve
    first = steps < loop.registry.get(loop.reviews[0]["candidate"])["step"]
    assert (best[first] == got[first]).all()
    assert {w for w, f in zip(which, first) if f} == {loop.service.log.models[0]}
