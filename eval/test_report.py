import numpy as np
import pytest

from eval.report import metrics, table


def run(policy="default", kind="concept", at=300, **got):
    r = {"kind": kind, "seed": 1, "policy": policy, "at": at, "regret": 2.0, "recovery": 40,
         "delay": 20, "decisions": 2, "false alarms": 1, "candidates": 4, "wasted": 1,
         "promotions": 2, "worse promoted": 0, "kept": 2, "better kept": 1, "rollbacks": 0}
    return {**r, **got}


def test_rates_are_pooled_over_runs_and_times_are_medians():
    got = metrics([run(), run(promotions=0, kept=4, **{"better kept": 0}, delay=np.nan,
                              recovery=np.nan, regret=4.0)])
    assert got["false promotion"] == 0.0
    assert got["missed promotion"] == pytest.approx(1 / 6)
    assert got["detection delay"] == 20 and got["undetected"] == 0.5
    assert got["recovery"] == 40 and got["unrecovered"] == 0.5
    assert got["false alarm"] == 0.5 and got["wasted retrains"] == 0.25
    assert got["regret"] == 3.0


def test_runs_without_a_change_have_no_delay():
    got = metrics([run(kind="none", at=None, delay=np.nan, recovery=np.nan)])
    assert np.isnan(got["detection delay"]) and np.isnan(got["undetected"])


def test_the_table_has_every_policy_over_all_kinds():
    out = table([run(), run(policy="naive", kind="noise")])
    assert out["default", "all"]["runs"] == 1 and out["naive", "noise"]["runs"] == 1
    assert ("naive", "concept") not in out
