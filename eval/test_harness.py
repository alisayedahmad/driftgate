from types import SimpleNamespace

import numpy as np

from eval.harness import HURT, alarms, recovery
from world.scenarios import Scenario

S = Scenario(3, "concept", at=300, size=0.1)


def test_an_alarm_counts_as_false_when_the_model_had_not_got_worse():
    timeline = [(280, "a", 0.80), (290, "a", 0.80), (300, "a", 0.80), (310, "a", 0.70),
                (320, "b", 0.78), (330, "b", 0.78)]
    loop = SimpleNamespace(decisions=[{"step": 295, "action": "retrain"},
                                      {"step": 312, "action": "retrain"},
                                      {"step": 335, "action": "hold"}])
    # 295 is before the change, 312 is after a drop, 335 is b, steady since it started
    assert alarms(S, loop, timeline) == (12, 3, 2)


def test_recovery_counts_until_the_gap_is_back_near_where_it_was():
    steps = np.arange(200, 400, 10)
    gap = np.where(steps < 300, 0.05, np.where(steps < 330, 0.15, 0.05 + HURT / 2))
    assert recovery(S, steps, gap) == 30
    assert recovery(S, steps, np.full(len(steps), 0.05)) == 0
    assert np.isnan(recovery(S, steps, np.where(steps < 300, 0.05, 0.15)))
    assert np.isnan(recovery(Scenario(3), steps, gap))
