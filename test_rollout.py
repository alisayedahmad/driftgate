from types import SimpleNamespace

import numpy as np
import pytest

from rollout import HURT, SCHEDULES, measured, table
from world.scenarios import Scenario


def test_recovery_counts_from_the_change_until_the_gap_comes_back():
    s = Scenario(3, "concept", at=300, size=0.1)
    steps = range(200, 400, 10)
    best = dict.fromkeys(steps, 0.85)
    # 0.05 below the best before the change, 0.15 for three steps, then back near 0.05
    gap = np.array([0.05 if t < 300 else 0.15 if t < 330 else 0.05 + HURT / 2 for t in steps])
    nothing = SimpleNamespace(reviews=[], events=[])
    got = measured(s, nothing, 0.85 - gap, best)
    assert got["recovery"] == 30
    assert got["after"] == pytest.approx(gap[10:].mean())
    assert got["promotions"] == got["rollbacks"] == got["worse"] == 0


def test_schedules_are_summed_per_kind():
    def run(lost, recovery, worse):
        return {"lost": lost, "after": 2 * lost, "recovery": recovery, "promotions": 1,
                "rollbacks": 0, "worse": worse}
    runs = [("concept", 1, dict.fromkeys(SCHEDULES, run(0.05, 40, 0))),
            ("concept", 2, dict.fromkeys(SCHEDULES, run(0.07, np.nan, 30)))]
    row = table(runs)["concept", "default"]
    assert row["runs"] == 2 and row["lost"] == pytest.approx(0.06)
    assert row["recovery"] == 40 and row["never"] == 0.5 and row["worse"] == 15
    assert ("noise", "default") not in table(runs)
