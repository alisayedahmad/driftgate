import numpy as np

from gate.policy import decide, load, review
from gate.segments import given_up


def result(class_gain, class_rows, class_se, gain=0.05, se=0.01):
    return {"rows": 400, "gain": gain, "se": se, "class gain": np.array(class_gain),
            "class rows": np.array(class_rows), "class se": np.array(class_se)}


def test_a_class_given_up_is_one_whose_loss_stands_out_from_its_noise():
    got = result([-0.10, 0.20, -0.12, -0.30, np.nan], [60, 80, 40, 10, 0],
                 [0.02, 0.03, 0.05, 0.05, np.nan])
    # class 2 loses as much as class 0 but on noisier rows, class 3 has too few
    assert given_up(got, load()) == [(0, -0.10, 60)]


def test_a_class_given_up_holds_a_candidate_that_wins_overall():
    got = result([-0.10, 0.20, 0.0, 0.0, np.nan], [60, 80, 40, 30, 0],
                 [0.02, 0.03, 0.03, 0.04, np.nan])
    assert decide(got, load()).promote is False
    assert "class 0" in decide(got, load()).reason
    got["class gain"][0] = -0.04
    assert decide(got, load()).promote is True


def test_the_class_guard_catches_what_label_noise_hides(noise):
    # seed 4 relabels part of class 0 as class 1: the rows still labeled 0
    # show the candidate giving that class away
    _, registry, serving, candidate, _ = noise
    _, _, got, verdict = review(registry, candidate, serving, load())
    assert got["gain"] > 0
    assert not verdict.promote and verdict.reason.startswith("class 0 ")
