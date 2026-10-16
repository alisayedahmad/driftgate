import numpy as np
import pytest

from gate.policy import decide, expected, load, review
from retrain.train import from_bytes
from world.truth import judge


def result(gain, se, rows=400):
    # no class with rows enough to be judged
    return {"rows": rows, "gain": gain, "se": se, "class rows": np.zeros(5, dtype=int),
            "class gain": np.full(5, np.nan), "class se": np.full(5, np.nan)}


def test_expected_gain_and_loss_match_a_simulation():
    g = np.random.default_rng(9)
    for gain, se in [(0.02, 0.03), (-0.01, 0.02), (0.0, 0.05)]:
        draws = gain + se * g.standard_normal(1_000_000)
        up, down = expected(gain, se)
        assert up == pytest.approx(np.maximum(draws, 0).mean(), abs=2e-4)
        assert down == pytest.approx(np.maximum(-draws, 0).mean(), abs=2e-4)


def test_the_shipped_policy_reads():
    assert set(load()) == {"penalty", "shadow", "min_rows", "class_rows", "tolerance",
                           "class_confidence", "retry"}


def test_too_few_rows_hold_whatever_they_say():
    assert not decide(result(0.2, 0.01, rows=50), load()).promote


def test_without_a_penalty_any_expected_gain_promotes():
    policy = {**load(), "penalty": 1.0}
    assert decide(result(0.001, 0.05), policy).promote
    assert not decide(result(-0.001, 0.05), policy).promote


def test_a_penalty_wants_more_evidence_the_noisier_the_gain():
    policy = {**load(), "penalty": 4.0}
    assert decide(result(0.02, 0.01), policy).promote
    assert not decide(result(0.02, 0.05), policy).promote


def test_a_concept_break_candidate_is_promoted(concept):
    _, registry, serving, candidate, step = concept
    look, _, got, verdict = review(registry, candidate, serving, load())
    assert look == step + load()["shadow"]
    assert verdict.promote and got["gain"] > 0


def test_the_overall_gain_cant_see_label_noise(noise):
    # the held-out labels are noisy too, the candidate fits them better
    s, registry, serving, candidate, step = noise
    _, _, got, _ = review(registry, candidate, serving, load())
    true = (judge(s, from_bytes(registry.store.get(candidate)).predict, step)["accuracy"]
            - judge(s, from_bytes(registry.store.get(serving)).predict, step)["accuracy"])
    assert got["gain"] > 0 > true
    assert decide(got, {**load(), "class_rows": 10_000}).promote
