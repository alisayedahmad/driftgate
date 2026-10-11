import numpy as np
import pytest

from world.scenarios import AT, NEW_SPREAD, SIZES, Scenario
from world.stream import (BAYES, CALIBRATION_ROWS, CHANGE, CLASSES, FEATURES, INFORMATIVE, PRIORS,
                          ROWS, label)

CHANGES = ["covariate", "concept", "noise", "new"]


def covariate(seed):
    return Scenario.draw(seed, "covariate")


@pytest.fixture(scope="module")
def concept():
    return Scenario.draw(3, "concept")


@pytest.fixture(scope="module")
def fresh():
    return np.random.default_rng(12345).standard_normal((200_000, FEATURES))


@pytest.mark.parametrize("kind", CHANGES)
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


@pytest.mark.parametrize("kind", CHANGES)
def test_rows_match_the_twin_before_the_change(kind):
    s = Scenario.draw(3, kind)
    for step in (0, s.at - 1):
        x, y = s.batch(step)
        twin_x, twin_y = s.twin().batch(step)
        assert np.array_equal(x, twin_x)
        assert np.array_equal(y, twin_y)


def test_covariate_shift_moves_the_inputs_only():
    s = next(s for s in map(covariate, range(100)) if s.shift[:INFORMATIVE].any())
    assert np.count_nonzero(s.shift) == 1
    x, y = s.batch(s.at)
    twin_x, _ = s.twin().batch(s.at)
    assert np.array_equal(x, twin_x + s.shift)
    _, u = s.world.draws(s.at)
    assert np.array_equal(y, label(s.world.proba(x), u))


def test_shifting_a_column_without_signal_keeps_every_label():
    s = next(s for s in map(covariate, range(100)) if s.shift[INFORMATIVE:].any())
    _, y = s.batch(s.at + 10)
    _, twin_y = s.twin().batch(s.at + 10)
    assert np.array_equal(y, twin_y)


def test_concept_break_keeps_the_inputs_and_moves_labels(concept):
    steps = range(concept.at, concept.at + 20)
    for step in steps:
        assert np.array_equal(concept.batch(step)[0], concept.twin().batch(step)[0])
    y = np.concatenate([concept.batch(t)[1] for t in steps])
    twin_y = np.concatenate([concept.twin().batch(t)[1] for t in steps])
    assert (y != twin_y).any()


def test_concept_break_costs_the_old_classifier_its_size(concept, fresh):
    before = concept.proba(fresh, concept.at - 1)
    after = concept.proba(fresh, concept.at)
    loss = before.max(axis=1) - after[np.arange(len(fresh)), before.argmax(axis=1)]
    tol = 5 * np.sqrt(1 / CALIBRATION_ROWS + 1 / len(fresh))
    assert abs(loss.mean() - concept.size) < tol * loss.std()


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
    tol = 5 * np.sqrt(1 / CALIBRATION_ROWS + 1 / len(fresh))
    assert np.all(np.abs(p.mean(axis=0) - PRIORS) < tol * p.std(axis=0))
    assert abs(best.mean() - BAYES) < tol * best.std()


def test_noise_only_relabels_part_of_the_source_class():
    s = Scenario.draw(3, "noise")
    source, target = s.flip
    y, twin_y = [], []
    for step in range(s.at, s.at + 300):
        x, labels = s.batch(step)
        twin_x, true = s.twin().batch(step)
        assert np.array_equal(x, twin_x)
        assert np.array_equal(s.proba(x, step), s.twin().proba(x, step))
        y.append(labels)
        twin_y.append(true)
    y, twin_y = np.concatenate(y), np.concatenate(twin_y)
    moved = y != twin_y
    assert np.all(twin_y[moved] == source) and np.all(y[moved] == target)
    n = (twin_y == source).sum()
    assert abs(moved.sum() / n - s.size) < 5 * np.sqrt(s.size * (1 - s.size) / n)


def test_new_class_only_replaces_some_rows():
    s = Scenario.draw(3, "new")
    steps = range(s.at, s.at + 200)
    count = 0
    for step in steps:
        x, y = s.batch(step)
        twin_x, twin_y = s.twin().batch(step)
        new = y == CLASSES
        assert np.array_equal(x[~new], twin_x[~new])
        assert np.array_equal(y[~new], twin_y[~new])
        moved = s.center + NEW_SPREAD * twin_x[new, :INFORMATIVE]
        assert np.array_equal(x[new, :INFORMATIVE], moved)
        assert np.array_equal(x[new, INFORMATIVE:], twin_x[new, INFORMATIVE:])
        count += new.sum()
    n = len(steps) * ROWS
    assert abs(count / n - s.size) < 5 * np.sqrt(s.size * (1 - s.size) / n)


def test_new_class_labels_follow_the_posterior():
    s = Scenario.draw(3, "new")
    x, y = (np.concatenate(a) for a in zip(*(s.batch(t) for t in range(s.at, s.at + 300))))
    assert s.proba(x, s.at - 1).shape[1] == CLASSES
    p = s.proba(x, s.at)
    freq = np.bincount(y, minlength=CLASSES + 1) / len(y)
    se = np.sqrt((p * (1 - p)).mean(axis=0) / len(y))
    assert np.all(np.abs(freq - p.mean(axis=0)) < 5 * se)
    hit = p[np.arange(len(y)), y]
    square = (p * p).sum(axis=1)
    se = np.sqrt(((p**3).sum(axis=1) - square**2).mean() / len(y))
    assert abs(hit.mean() - square.mean()) < 5 * se


def test_bad_scripts_are_refused():
    with pytest.raises(ValueError):
        Scenario(0, "drift", at=10)
    with pytest.raises(ValueError):
        Scenario.draw(0, "drift")
    with pytest.raises(ValueError):
        Scenario.draw(2**32, "covariate")
    with pytest.raises(ValueError):
        Scenario(0, "covariate")
    with pytest.raises(ValueError):
        Scenario(0, "none", at=10)
    with pytest.raises(ValueError):
        Scenario(3, "concept", at=10, size=0.9).rotation
