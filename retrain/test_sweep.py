import pytest

from retrain.sweep import KNOWN, decisions, plans, windows
from retrain.trigger import Decision
from world.scenarios import Scenario


def decision(step, start, action="retrain"):
    return Decision(step, action, "", start)


def test_the_true_start_is_planned_from_the_change():
    got = {name: plan for name, _, plan in plans(Scenario(3, "concept", at=400, size=0.1),
                                                 decision(430, 420))}
    assert got["default"]["fit"][0] == 410
    assert got[KNOWN]["fit"][0] == 400
    assert KNOWN not in {name for name, *_ in plans(Scenario(3), decision(430, 420))}


def test_windows_that_cant_be_planned_are_left_out():
    # gate 40 holds out [410, 450), past a cut at 420
    names = {name for name, *_ in plans(Scenario(3), decision(430, 430))}
    assert "gate 40" not in names and "default" in names


def candidate(policy, gain):
    return {"policy": policy, "window": "default", "step": 330, "rows": 100,
            "candidate": 0.8 + gain, "serving": 0.8, "best": 0.85, "worst": -0.1}


def test_decisions_and_gains_are_counted_per_kind_and_policy():
    runs = [("noise", 1, 300, {"informed": decision(320, 305, "hold"), "naive": decision(310, 305)},
             [candidate("naive", -0.02)]),
            ("noise", 2, 300, {"informed": None, "naive": decision(250, 240)},
             [candidate("naive", 0.04)])]
    d = decisions(runs)
    assert d["noise", "informed"] == {"runs": 2, "retrain": 0.0, "hold": 0.5, "early": 0.0,
                                      "delay": 20.0}
    assert d["noise", "naive"] == {"runs": 2, "retrain": 1.0, "hold": 0.0, "early": 0.5,
                                   "delay": 10.0}
    t = windows(runs)
    assert t["noise", "informed", "default"] == {"candidates": 0}
    naive = t["noise", "naive", "default"]
    assert naive["candidates"] == 2 and naive["better"] == 0.5
    assert naive["gain"] == pytest.approx(0.01)
    assert naive["short"] == pytest.approx(0.04)
