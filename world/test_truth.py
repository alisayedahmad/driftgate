import numpy as np

from world.scenarios import Scenario
from world.stream import BAYES, CLASSES, PRIORS
from world.truth import JUDGED, judge, realized


def best(s, step):
    # the best possible predictor at step: the argmax of the true P(y|x)
    return lambda x: s.proba(x, step).argmax(axis=1)


def test_the_best_predictor_gets_the_bayes_accuracy():
    s = Scenario(3)
    got = judge(s, best(s, 400), 400)
    assert got["accuracy"] == s.proba(s.fresh(400, JUDGED), 400).max(axis=1).mean()
    assert abs(got["accuracy"] - BAYES) < 0.01


def test_a_concept_break_costs_the_old_model_its_size():
    s = Scenario.draw(3, "concept")
    old = best(s.twin(), s.at)
    lost = judge(s, old, s.at - 1)["accuracy"] - judge(s, old, s.at)["accuracy"]
    assert abs(lost - s.size) < 0.015


def test_a_one_class_predictor_gets_that_class_and_its_prior():
    got = judge(Scenario(3), lambda x: np.zeros(len(x), dtype=int), 400)
    assert np.isclose(got["recall"][0], 1.0)
    assert (got["recall"][1:CLASSES] == 0).all()
    assert np.isnan(got["recall"][CLASSES])
    assert abs(got["accuracy"] - PRIORS[0]) < 0.01


def test_fresh_rows_follow_the_world_but_never_the_stream():
    s = Scenario.draw(3, "covariate")
    x = s.fresh(s.at, 500)
    assert np.array_equal(x, s.twin().fresh(s.at, 500) + s.shift)
    assert np.array_equal(s.fresh(s.at - 1, 500), s.twin().fresh(s.at - 1, 500))
    assert not np.isin(x, s.batch(s.at)[0]).any()


def test_label_noise_is_judged_on_the_clean_labels():
    s = Scenario.draw(3, "noise")
    model = best(s.twin(), s.at)
    noisy, clean = judge(s, model, s.at + 10), judge(s.twin(), model, s.at + 10)
    assert noisy["accuracy"] == clean["accuracy"]
    assert np.array_equal(noisy["recall"], clean["recall"], equal_nan=True)


def test_realized_counts_the_right_answers_on_every_row():
    s = Scenario(3)
    model = best(s, 400)
    x, y = zip(*(s.batch(t) for t in range(400, 410)))
    assert realized(s, model, 400, 10) == (model(np.concatenate(x)) == np.concatenate(y)).mean()
    # a thousand drawn labels land near what the judge expects
    assert abs(realized(s, model, 400, 10) - judge(s, model, 400)["accuracy"]) < 0.05
