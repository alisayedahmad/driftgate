import math

import numpy as np

from monitor.cusum import Cusum, first_alarm

RUNS = 4000
H = 4.0


def siegmund(shift, k=0.5, h=H):
    # one-sided average run length, Siegmund's approximation
    d, b = shift - k, h + 1.166
    return (math.exp(-2 * d * b) + 2 * d * b - 1) / (2 * d * d)


def run_lengths(shift, steps=3000):
    # RUNS independent streams, each until its first crossing of H
    g = np.random.default_rng(11)
    c = Cusum(RUNS)
    first = np.full(RUNS, steps)
    for t in range(steps):
        c.update(g.standard_normal(RUNS) + shift)
        crossed = ((c.up > H) | (c.down > H)) & (first == steps)
        first[crossed] = t + 1
    return first


def test_false_alarms_come_as_rarely_as_the_theory_says():
    # two-sided: the false alarm rates of both sides add up
    lengths = run_lengths(0.0)
    assert abs(lengths.mean() / (siegmund(0.0) / 2) - 1) < 0.1


def test_a_shift_of_one_is_caught_as_fast_as_the_theory_says():
    lengths = run_lengths(1.0)
    assert abs(lengths.mean() / siegmund(1.0) - 1) < 0.1


def test_a_missing_value_leaves_a_stream_alone():
    c = Cusum(2)
    c.update([3.0, 3.0])
    up = c.up.copy()
    c.update([np.nan, 0.0])
    assert c.up[0] == up[0]
    assert c.up[1] == up[1] - 0.5


def test_first_alarm():
    assert first_alarm([0.0, 2.0, 5.0, 6.0], 4.0) == 2
    assert first_alarm([0.0, 1.0], 4.0) is None
