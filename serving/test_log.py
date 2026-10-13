import sqlite3

import numpy as np
import pytest

from serving.log import UNKNOWN, Log
from world.stream import CLASSES, FEATURES


def rows(step, n=100):
    g = np.random.default_rng(step)
    return g.standard_normal((n, FEATURES)), g.dirichlet(np.ones(CLASSES + 1), n)


def filled(steps, model="a" * 64):
    log = Log()
    for step in range(steps):
        x, proba = rows(step)
        log.served(step, x, model, proba)
    return log


def test_rows_come_back_in_serving_order():
    log = filled(30)
    known = log.as_of(29)
    assert np.array_equal(known["x"], np.concatenate([rows(t)[0] for t in range(30)]))
    assert np.array_equal(known["proba"], np.concatenate([rows(t)[1] for t in range(30)]))
    assert np.array_equal(known["step"], np.repeat(np.arange(30), 100))
    assert len(log.as_of(9)["step"]) == 1000


def test_a_label_is_unknown_until_it_arrives():
    log = filled(5)
    log.arrived(3, 0, [2, 7], [1, 3])
    assert (log.as_of(2)["label"] == UNKNOWN).all()
    known = log.as_of(4)
    assert known["label"][[2, 7]].tolist() == [1, 3]
    assert known["arrived"][[2, 7]].tolist() == [3, 3]
    assert (np.delete(known["label"], [2, 7]) == UNKNOWN).all()


def test_each_row_remembers_its_model():
    log = Log()
    for step, model in [(0, "a" * 64), (1, "b" * 64)]:
        x, proba = rows(step)
        log.served(step, x, model, proba)
    assert log.models == ["a" * 64, "b" * 64]
    assert np.array_equal(log.as_of(1)["model"], np.repeat([0, 1], 100))


def test_impossible_entries_are_refused():
    log = filled(3)
    x, proba = rows(5)
    with pytest.raises(ValueError):
        log.served(2, x, "a" * 64, proba)
    with pytest.raises(ValueError):
        log.arrived(1, 1, [0], [0])
    with pytest.raises(ValueError):
        log.arrived(3, 1, [100], [0])
    log.arrived(3, 1, [0], [0])
    with pytest.raises(ValueError):
        log.arrived(4, 1, [0], [2])


def test_save_writes_every_row():
    log = filled(3)
    log.arrived(2, 0, [5], [1])
    db = sqlite3.connect(":memory:")
    log.save(db)
    assert db.execute("select count(*) from predictions").fetchone()[0] == 300
    row = db.execute("select * from predictions limit 1 offset 5").fetchone()
    assert row[0] == 0 and row[-2:] == (1, 2)
    assert np.array_equal(row[1:1 + FEATURES], rows(0)[0][5])
    assert db.execute("select * from served_by").fetchall() == [(0, "a" * 64)]


def test_a_step_reads_back_as_served():
    log = filled(5)
    got = log.batch(3)
    assert np.array_equal(got["x"], rows(3)[0])
    assert np.array_equal(got["proba"], rows(3)[1])
    assert (got["model"] == 0).all()
    with pytest.raises(KeyError):
        log.batch(9)


def test_landed_gives_what_arrived_at_that_step_only():
    log = filled(5)
    log.arrived(3, 0, [2, 7], [1, 3])
    log.arrived(3, 1, [4], [0])
    log.arrived(4, 1, [5], [2])
    got = log.landed(3)
    assert got["step"].tolist() == [0, 0, 1]
    assert got["label"].tolist() == [1, 3, 0]
    assert np.array_equal(got["x"], np.stack([rows(0)[0][2], rows(0)[0][7], rows(1)[0][4]]))
    assert log.landed(2)["x"].shape == (0, FEATURES)
