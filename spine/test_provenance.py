import argparse
import json
import subprocess

import numpy as np

from spine.provenance import (canonical, code, data_hash, dataset, row_weights, stamp, verify,
                              versions)
from world.labels import LABELING
from world.scenarios import Scenario


def test_canonical_ignores_key_order():
    assert canonical({"b": 1, "a": [1.5, None]}) == canonical({"a": [1.5, None], "b": 1})
    assert canonical({"a": 1}) == b'{"a":1}'


def test_data_hash_sees_any_change():
    x, y = np.zeros((10, 6)), np.zeros(10, dtype=int)
    h = data_hash(x, y)
    moved = x.copy()
    moved[3, 2] = 1e-300
    relabeled = y.copy()
    relabeled[0] = 1
    assert data_hash(moved, y) != h
    assert data_hash(x, relabeled) != h
    assert data_hash(x.reshape(20, 3), y) != h


def test_a_dataset_document_regenerates_its_rows():
    s = Scenario.draw(3, "covariate")
    x, y, doc = dataset(s, s.at - 5, s.at + 5, LABELING, s.at + 5)
    assert len(x) == len(y) == doc["rows"]
    again = json.loads(canonical(doc))
    assert verify(again)
    again["recipe"]["as_of"] += 1000
    assert not verify(again)


def test_row_weights_follow_the_steps_of_the_rows():
    s = Scenario(3)
    _, y, doc = dataset(s, 0, 30, LABELING, 50, [[10, 20, 0.5]])
    before, inside = (dataset(s, a, b, LABELING, 50)[2]["rows"] for a, b in ((0, 10), (10, 20)))
    w = row_weights(doc["recipe"])
    assert len(w) == len(y)
    assert (w[before:before + inside] == 0.5).all()
    assert (w[:before] == 1).all() and (w[before + inside:] == 1).all()
    plain = dataset(s, 0, 30, LABELING, 50)[2]["recipe"]
    assert "weights" not in plain and (row_weights(plain) == 1).all()


def test_only_labels_back_by_as_of_are_kept():
    s = Scenario(3)
    early = dataset(s, 0, 50, LABELING, 50)[2]["rows"]
    late = dataset(s, 0, 50, LABELING, 10_000)[2]["rows"]
    every = dataset(s, 0, 50, {**LABELING, "rate": 1.0}, 10_000)[2]["rows"]
    assert early < late < every == 50 * 100


def test_a_stamp_says_what_made_a_result(tmp_path):
    about = {"code": {"commit": "0" * 40, "dirty": False, "changes": None}, "versions": versions()}
    stamp(tmp_path, argparse.Namespace(seeds=3, out=tmp_path), about)
    assert json.loads((tmp_path / "about.json").read_text()) == {"args": {"seeds": 3}, **about}


def git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                   check=True, capture_output=True)


def test_code_tells_a_clean_tree_from_a_changed_one(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "a.py")
    git(tmp_path, "commit", "-q", "-m", "first")
    clean = code(tmp_path)
    assert clean["dirty"] is False and clean["changes"] is None and len(clean["commit"]) == 40
    (tmp_path / "a.py").write_text("x = 2\n")
    edited = code(tmp_path)
    assert edited["dirty"] and edited["commit"] == clean["commit"]
    (tmp_path / "b.py").write_text("y = 1\n")
    assert code(tmp_path)["changes"] != edited["changes"]
