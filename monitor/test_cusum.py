import math

import numpy as np

from monitor.cusum import Cusum, Monitor, first_alarm

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


def test_the_start_follows_the_largest_sum():
    m = Monitor(1, 0.5)
    for step, z in zip(range(10, 15), [1.0, -2.0, 0.6, 3.0, 3.0]):
        m.record(step, [z])
    # up rises at 10 and falls back, down rises at 11, up rises again at 12 and overtakes
    assert m.starts == [10, 11, 11, 12, 12]


def test_a_stream_starts_when_it_first_gets_a_value():
    c = Cusum(2)
    c.update([np.nan, 0.0])
    c.update([np.nan, 0.0])
    assert c.start() is None
    c.update([2.0, 0.0])
    assert c.start() == 2


def starts(shift, at=50, h=8.0, runs=500):
    # one stream per run, shifted from update `at` on, its start read at the alarm
    g = np.random.default_rng(12)
    out = []
    for _ in range(runs):
        c = Cusum(1)
        t = 0
        while c.update(g.standard_normal(1) + shift * (t >= at)) <= h:
            t += 1
        if t >= at:
            out.append(c.start() - at)
    return np.array(out)


def test_the_start_of_a_shift_of_2k_is_found():
    # for that shift the last zero is the maximum likelihood change point
    error = starts(1.0)
    assert np.median(error) == 0
    assert np.mean(np.abs(error) <= 5) > 0.8


def test_first_alarm():
    assert first_alarm([0.0, 2.0, 5.0, 6.0], 4.0) == 2
    assert first_alarm([0.0, 1.0], 4.0) is None
