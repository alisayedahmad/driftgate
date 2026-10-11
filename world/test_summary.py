import numpy as np

from world.scenarios import Scenario
from world.summary import WINDOW, rolling, trace


def test_before_a_break_the_old_model_is_the_best_one():
    r = trace(Scenario(3, "concept", at=5, size=0.2), steps=10)
    assert np.array_equal(r["true"][:5], r["best"][:5])
    assert np.all(r["true"][5:] < r["best"][5:])
    assert np.allclose(r["freq"].sum(axis=1), 1)


def test_noise_moves_what_is_measured_and_nothing_true():
    s = Scenario(3, "noise", at=5, size=0.4)
    r, twin = trace(s, steps=10), trace(s.twin(), steps=10)
    assert np.array_equal(r["true"], twin["true"])
    assert np.array_equal(r["best"], twin["best"])
    assert np.array_equal(r["measured"][:5], twin["measured"][:5])
    assert not np.array_equal(r["measured"][5:], twin["measured"][5:])


def test_rolling_is_a_trailing_mean_over_full_windows():
    a = np.arange(100.0)
    out = rolling(a)
    assert len(out) == len(a) - WINDOW + 1
    assert out[0] == a[:WINDOW].mean()
    assert out[-1] == a[-WINDOW:].mean()
