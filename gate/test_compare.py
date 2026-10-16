import hashlib

import numpy as np
import pytest

from gate.compare import compare, paired
from retrain.train import from_bytes
from spine.provenance import canonical
from world.truth import judge


def test_paired_on_rows_worked_by_hand():
    got = paired([1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 2, 2])
    d = np.array([0, 1, 0, 0, -1, 1])
    assert got["rows"] == 6
    assert got["gain"] == pytest.approx(1 / 6)
    assert got["se"] == pytest.approx(d.std(ddof=1) / np.sqrt(6))
    assert got["class rows"].tolist() == [2, 2, 2, 0, 0]
    assert got["class gain"][:3].tolist() == [0.5, 0.0, 0.0]
    assert np.isnan(got["class gain"][3:]).all()


def test_pairing_shrinks_the_error_when_models_mostly_agree():
    g = np.random.default_rng(5)
    old = g.random(2000) < 0.7
    new = np.where(g.random(2000) < 0.05, ~old, old)
    p, q = old.mean(), new.mean()
    apart = np.sqrt(p * (1 - p) / 2000 + q * (1 - q) / 2000)
    # 5% of rows differ, the theory gives about a third
    assert paired(new, old, np.zeros(2000, dtype=int))["se"] < apart / 2


def test_the_paired_gain_finds_the_true_one(concept):
    s, registry, serving, candidate, step = concept
    doc, got = compare(registry, candidate, serving, step + 10)
    true = (judge(s, from_bytes(registry.store.get(candidate)).predict, step)["accuracy"]
            - judge(s, from_bytes(registry.store.get(serving)).predict, step)["accuracy"])
    assert abs(got["gain"] - true) < 2 * got["se"]
    assert hashlib.sha256(canonical(doc)).hexdigest() in registry.store


def test_waiting_brings_more_rows_and_the_past_is_refused(concept):
    _, registry, serving, candidate, step = concept
    assert compare(registry, candidate, serving, step)[1]["rows"] < compare(
        registry, candidate, serving, step + 10)[1]["rows"]
    with pytest.raises(ValueError):
        compare(registry, candidate, serving, step - 1)
