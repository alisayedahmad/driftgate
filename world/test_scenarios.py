import numpy as np
import pytest

from world.scenarios import AT, SHIFT, Scenario
from world.stream import FEATURES, INFORMATIVE


def covariate(seed):
    return Scenario.draw(seed, "covariate")


def test_draw_depends_only_on_the_seed():
    s = covariate(3)
    assert s == covariate(3)
    assert AT[0] <= s.at <= AT[1]
    assert SHIFT[0] <= s.size <= SHIFT[1]
    assert np.count_nonzero(s.shift) == 1


def test_rows_match_the_twin_before_the_change():
    s = covariate(3)
    for step in (0, s.at - 1):
        x, y = s.batch(step)
        tx, ty = s.twin().batch(step)
        assert np.array_equal(x, tx)
        assert np.array_equal(y, ty)


def test_covariate_shift_moves_the_inputs_only():
    s = covariate(3)
    x, _ = s.batch(s.at)
    tx, _ = s.twin().batch(s.at)
    assert np.array_equal(x, tx + s.shift)
    z = np.random.default_rng(0).standard_normal((500, FEATURES))
    assert np.array_equal(s.proba(z, s.at), s.twin().proba(z, s.at))


def test_shifting_a_column_without_signal_keeps_every_label():
    s = next(s for s in map(covariate, range(100)) if s.shift[INFORMATIVE:].any())
    _, y = s.batch(s.at + 10)
    _, ty = s.twin().batch(s.at + 10)
    assert np.array_equal(y, ty)


def test_bad_scripts_are_refused():
    with pytest.raises(ValueError):
        Scenario(0, "drift", at=10)
    with pytest.raises(ValueError):
        Scenario(0, "covariate")
    with pytest.raises(ValueError):
        Scenario(0, "none", at=10)
