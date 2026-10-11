import numpy as np
import pytest

from world.stream import (BAYES, CALIBRATION_ROWS, CLASSES, FEATURES, INFORMATIVE,
                          PRIORS, World)


@pytest.fixture(scope="module")
def worlds():
    return [World(seed) for seed in range(5)]


def test_a_step_depends_only_on_seed_and_step():
    first = World(7).batch(5)
    w = World(7)
    for t in range(5):
        w.batch(t)
    x, y = w.batch(5)
    assert np.array_equal(x, first[0])
    assert np.array_equal(y, first[1])
    assert not np.array_equal(x, w.batch(6)[0])
    assert not np.array_equal(x, World(8).batch(5)[0])


def test_calibration_holds_on_fresh_rows(worlds):
    x = np.random.default_rng(12345).standard_normal((200_000, FEATURES))
    k = np.sqrt(1 / CALIBRATION_ROWS + 1 / len(x))
    for w in worlds:
        p = w.proba(x)
        best = p.max(axis=1)
        assert np.all(np.abs(p.mean(axis=0) - PRIORS) < 5 * k * p.std(axis=0))
        assert abs(best.mean() - BAYES) < 5 * k * best.std()


def test_labels_follow_proba(worlds):
    w = worlds[3]
    x, y = (np.concatenate(a) for a in zip(*(w.batch(t) for t in range(300))))
    p = w.proba(x)
    freq = np.bincount(y, minlength=CLASSES) / len(y)
    se = np.sqrt((p * (1 - p)).mean(axis=0) / len(y))
    assert np.all(np.abs(freq - p.mean(axis=0)) < 5 * se)
    # per row too: if y ~ p(x), the mean of p(x)[y] is the mean of sum p(x)^2
    hit = p[np.arange(len(y)), y]
    square = (p * p).sum(axis=1)
    se = np.sqrt(((p**3).sum(axis=1) - square**2).mean() / len(y))
    assert abs(hit.mean() - square.mean()) < 5 * se


def test_uninformative_features_do_not_change_proba(worlds):
    x = np.random.default_rng(1).standard_normal((1000, FEATURES))
    moved = x.copy()
    moved[:, INFORMATIVE:] += 5
    assert np.array_equal(worlds[0].proba(x), worlds[0].proba(moved))


def test_seed_outside_32_bits_is_refused():
    with pytest.raises(ValueError):
        World(2**32)
