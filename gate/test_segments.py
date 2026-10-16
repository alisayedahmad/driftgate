import numpy as np

from gate.policy import decide, load, review
from gate.segments import given_up


def result(class_gain, class_rows, gain=0.05, se=0.01):
    return {"rows": 400, "gain": gain, "se": se, "class gain": np.array(class_gain),
            "class rows": np.array(class_rows)}


def test_a_class_losing_more_than_the_tolerance_is_given_up():
    got = result([-0.10, 0.20, -0.04, -0.30, np.nan], [60, 80, 40, 10, 0])
    # class 2 is within the tolerance, class 3 has too few rows to judge
    assert given_up(got, load()) == [(0, -0.10, 60)]


def test_a_class_given_up_holds_a_candidate_that_wins_overall():
    got = result([-0.10, 0.20, 0.0, 0.0, np.nan], [60, 80, 40, 30, 0])
    assert decide(got, load()).promote is False
    assert "class 0" in decide(got, load()).reason
    got["class gain"][0] = -0.01
    assert decide(got, load()).promote is True


def test_the_class_guard_catches_what_label_noise_hides(noise):
    # seed 4 relabels part of class 0 as class 1: the rows still labeled 0
    # show the candidate giving that class away
    _, registry, serving, candidate, _ = noise
    _, _, got, verdict = review(registry, candidate, serving, load())
    assert got["gain"] > 0
    assert not verdict.promote and verdict.reason.startswith("class 0 ")
