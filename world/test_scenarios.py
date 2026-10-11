import numpy as np
import pytest

from world.scenarios import AT, SIZES, Scenario
from world.stream import BAYES, CALIBRATION_ROWS, CHANGE, FEATURES, INFORMATIVE, PRIORS, label

KINDS = ["covariate", "concept", "noise"]


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
    assert s != Scenario.draw(4, kind)
    assert AT[0] <= s.at <= AT[1]
    assert SIZES[kind][0] <= s.size <= SIZES[kind][1]


def test_draw_reaches_both_ends_of_the_step_range():
    steps = [Scenario.draw(seed, "covariate").at for seed in range(5000)]
    assert min(steps) == AT[0]
    assert max(steps) == AT[1]


@pytest.mark.parametrize("kind", KINDS)
def test_rows_match_the_twin_before_the_change(kind):
    s = Scenario.draw(3, kind)
    for step in (0, s.at - 1):
        x, y = s.batch(step)
        tx, ty = s.twin().batch(step)
        assert np.array_equal(x, tx)
        assert np.array_equal(y, ty)


def test_covariate_shift_moves_the_inputs_only():
    s = next(s for s in map(covariate, range(100)) if s.shift[:INFORMATIVE].any())
    assert np.count_nonzero(s.shift) == 1
    x, y = s.batch(s.at)
    tx, _ = s.twin().batch(s.at)
    assert np.array_equal(x, tx + s.shift)
    _, u = s.world.draws(s.at)
    assert np.array_equal(y, label(s.world.proba(x), u))


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


def test_concept_angle_hits_its_size_on_its_own_sample(concept):
    g = np.random.default_rng([concept.seed, CHANGE])
    v = g.standard_normal((CALIBRATION_ROWS, INFORMATIVE))
    before = concept.proba(v, concept.at - 1)
    after = concept.proba(v, concept.at)
    harm = (before.max(axis=1) - after[np.arange(len(v)), before.argmax(axis=1)]).mean()
    assert concept.size <= harm < concept.size + 1e-3


def test_concept_sizes_share_one_plane():
    small = Scenario(3, "concept", at=300, size=0.05).rotation
    large = Scenario(3, "concept", at=300, size=0.25).rotation
    eye = np.eye(INFORMATIVE)
    assert np.linalg.matrix_rank(np.hstack([small - eye, large - eye]), tol=1e-9) == 2


def test_concept_break_leaves_priors_and_bayes_accuracy_alone(concept, fresh):
    p = concept.proba(fresh, concept.at)
    best = p.max(axis=1)
    k = np.sqrt(1 / CALIBRATION_ROWS + 1 / len(fresh))
    assert np.all(np.abs(p.mean(axis=0) - PRIORS) < 5 * k * p.std(axis=0))
    assert abs(best.mean() - BAYES) < 5 * k * best.std()


def test_noise_only_relabels_part_of_the_source_class():
    s = Scenario.draw(3, "noise")
    source, target = s.flip
    y, ty = [], []
    for step in range(s.at, s.at + 300):
        x, labels = s.batch(step)
        tx, true = s.twin().batch(step)
        assert np.array_equal(x, tx)
        assert np.array_equal(s.proba(x, step), s.twin().proba(x, step))
        y.append(labels)
        ty.append(true)
    y, ty = np.concatenate(y), np.concatenate(ty)
    moved = y != ty
    assert np.all(ty[moved] == source) and np.all(y[moved] == target)
    n = (ty == source).sum()
    assert abs(moved.sum() / n - s.size) < 5 * np.sqrt(s.size * (1 - s.size) / n)


def test_bad_scripts_are_refused():
    with pytest.raises(ValueError):
        Scenario(0, "drift", at=10)
    with pytest.raises(ValueError):
        Scenario(0, "covariate")
    with pytest.raises(ValueError):
        Scenario(0, "none", at=10)
    with pytest.raises(ValueError):
        Scenario(3, "concept", at=10, size=0.9).rotation
