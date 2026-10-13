import math

import numpy as np

from world.labels import LABELING, fate


def test_fate_depends_only_on_seed_and_step():
    labeled, arrives = fate(3, 10, LABELING)
    again = fate(3, 10, LABELING)
    assert np.array_equal(labeled, again[0]) and np.array_equal(arrives, again[1])
    assert not np.array_equal(arrives - 10, fate(3, 11, LABELING)[1] - 11)
    assert not np.array_equal(arrives, fate(4, 10, LABELING)[1])


def test_rate_and_delays_match_the_labeling():
    labeled, delay = [], []
    for step in range(1000):
        mask, arrives = fate(3, step, LABELING)
        labeled.append(mask)
        delay.append(arrives - step)
    labeled, delay = np.concatenate(labeled), np.concatenate(delay)
    n, rate = len(labeled), LABELING["rate"]
    assert delay.min() >= 1
    assert abs(labeled.mean() - rate) < 5 * math.sqrt(rate * (1 - rate) / n)
    for steps in (10, 36, 103):
        z = math.log(steps / LABELING["median"]) / LABELING["spread"]
        share = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        assert abs((delay <= steps).mean() - share) < 5 * math.sqrt(share * (1 - share) / n)
