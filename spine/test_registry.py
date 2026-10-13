import pytest

from spine.provenance import canonical
from spine.registry import Registry, fresh
from spine.store import Store

CODE = {"commit": "0" * 40, "dirty": False, "changes": None}
VERSIONS = {"python": "3.14.8", "numpy": "2.5.3", "scikit-learn": "1.9.1"}


def record(store, name, parent=None, step=0, accuracy=0.8):
    return {"model": store.put(name.encode()),
            "trained_on": store.put(canonical({"recipe": name, "data": "train"})),
            "evaluated_on": store.put(canonical({"recipe": name, "data": "test"})),
            "config": store.put(canonical({"C": 1.0})),
            "code": CODE, "versions": VERSIONS, "metrics": {"accuracy": accuracy},
            "parent": parent, "step": step}


def test_a_record_comes_back_as_registered():
    r = Registry(Store(":memory:"))
    rec = record(r.store, "a")
    assert r.register(rec) == rec["model"]
    assert r.get(rec["model"]) == rec
    assert r.get("f" * 64) is None


def test_the_same_record_twice_is_fine_another_one_is_not():
    r = Registry(Store(":memory:"))
    r.register(record(r.store, "a"))
    r.register(record(r.store, "a"))
    with pytest.raises(ValueError):
        r.register(record(r.store, "a", accuracy=0.9))


def test_incomplete_records_are_refused():
    r = Registry(Store(":memory:"))
    rec = record(r.store, "a")
    with pytest.raises(ValueError):
        r.register({**rec, "trained_on": "0" * 64})
    with pytest.raises(ValueError):
        r.register({**rec, "parent": "1" * 64})
    with pytest.raises(ValueError):
        r.register({k: v for k, v in rec.items() if k != "step"})
    assert r.dump() == ""


def test_lineage_walks_back_to_the_first_model():
    r = Registry(Store(":memory:"))
    a = r.register(record(r.store, "a"))
    b = r.register(record(r.store, "b", parent=a, step=400))
    c = r.register(record(r.store, "c", parent=b, step=700))
    assert [rec["model"] for rec in r.lineage(c)] == [c, b, a]
    with pytest.raises(KeyError):
        r.lineage("2" * 64)


def test_identical_runs_give_identical_registries(tmp_path):
    dumps = []
    for run in ("one", "two"):
        r = Registry(Store(tmp_path / run / "spine.db"))
        a = r.register(record(r.store, "a"))
        r.register(record(r.store, "b", parent=a, step=400))
        dumps.append(r.dump())
    assert dumps[0] == dumps[1] != ""


def test_fresh_starts_each_run_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("spine.registry.RUNS", tmp_path)
    r = fresh("run")
    r.register(record(r.store, "a"))
    assert r.dump() != ""
    assert fresh("run").dump() == ""
