import numpy as np
import pytest

from world.scenarios import AT, SIZES, Scenario
from world.stream import BAYES, CALIBRATION_ROWS, FEATURES, INFORMATIVE, PRIORS

KINDS = ["covariate", "concept"]


def covariate(seed):
    return Scenario.draw(seed, "covariate")


@pytest.fixture(scope="module")
def concept():
    return Scenario.draw(3, "concept")


@pytest.fixture(scope="module")
def fresh():
    return np.random.default_rng(12345).standard_normal((200_000, FEATURES))


@pytest.mark.parametrize("kind", KINDS)
def test_draw_depends_only_on_the_seed(kind):
    s = Scenario.draw(3, kind)
    assert s == Scenario.draw(3, kind)
    assert AT[0] <= s.at <= AT[1]
    assert SIZES[kind][0] <= s.size <= SIZES[kind][1]


@pytest.mark.parametrize("kind", KINDS)
def test_rows_match_the_twin_before_the_change(kind):
    s = Scenario.draw(3, kind)
    for step in (0, s.at - 1):
        x, y = s.batch(step)
        tx, ty = s.twin().batch(step)
        assert np.array_equal(x, tx)
        assert np.array_equal(y, ty)


def test_covariate_shift_moves_the_inputs_only():
    s = covariate(3)
    assert np.count_nonzero(s.shift) == 1
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


def test_concept_break_keeps_the_inputs_and_moves_labels(concept):
    steps = range(concept.at, concept.at + 20)
    for step in steps:
        assert np.array_equal(concept.batch(step)[0], concept.twin().batch(step)[0])
    y = np.concatenate([concept.batch(t)[1] for t in steps])
    ty = np.concatenate([concept.twin().batch(t)[1] for t in steps])
    assert (y != ty).any()


def test_concept_break_costs_the_old_classifier_its_size(concept, fresh):
    before = concept.proba(fresh, concept.at - 1)
    after = concept.proba(fresh, concept.at)
    loss = before.max(axis=1) - after[np.arange(len(fresh)), before.argmax(axis=1)]
    k = np.sqrt(1 / CALIBRATION_ROWS + 1 / len(fresh))
    assert abs(loss.mean() - concept.size) < 5 * k * loss.std()


def test_concept_break_leaves_priors_and_bayes_accuracy_alone(concept, fresh):
    p = concept.proba(fresh, concept.at)
    best = p.max(axis=1)
    k = np.sqrt(1 / CALIBRATION_ROWS + 1 / len(fresh))
    assert np.all(np.abs(p.mean(axis=0) - PRIORS) < 5 * k * p.std(axis=0))
    assert abs(best.mean() - BAYES) < 5 * k * best.std()


def test_bad_scripts_are_refused():
    with pytest.raises(ValueError):
        Scenario(0, "drift", at=10)
    with pytest.raises(ValueError):
        Scenario(0, "covariate")
    with pytest.raises(ValueError):
        Scenario(0, "none", at=10)
