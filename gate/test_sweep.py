import numpy as np
import pytest

from gate.sweep import SHADOWS, gates, table
from gate.policy import load


def candidate(gain, worst, **went):
    return {"shadow": 10, "gain": gain, "worst": worst, **dict.fromkeys(gates(load()), False), **went}


def test_calls_are_scored_against_the_truth():
    rows = [candidate(0.05, 0.0, naive=True, both=True),   # better, promoted by both
            candidate(0.02, -0.2, both=True),               # better but gives up a class
            candidate(-0.03, 0.0, naive=True),              # worse, promoted by naive
            candidate(-0.01, 0.0)]
    out = table([("concept", 1, rows)], tolerance=0.05)
    naive, both = out["concept", "naive", 10], out["concept", "both", 10]
    assert naive["candidates"] == 4 and naive["promoted"] == 0.5
    assert naive["false"] == 0.5 and naive["missed"] == 0.5 and naive["gave"] == 0.0
    assert naive["left"] == pytest.approx((0 + 0.02 + 0.03 + 0) / 4)
    assert both["false"] == 0.0 and both["missed"] == 0.0 and both["gave"] == 1.0
    assert both["left"] == 0.0
    assert out["concept", "both", SHADOWS[0]]["candidates"] == 0
    assert np.isnan(out["noise", "both", 10]["promoted"])
