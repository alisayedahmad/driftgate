import hashlib

import pytest

from spine.store import Store


def test_bytes_come_back_under_their_sha256():
    s = Store(":memory:")
    h = s.put(b"some model")
    assert h == hashlib.sha256(b"some model").hexdigest()
    assert s.get(h) == b"some model"


def test_storing_twice_keeps_one_copy():
    s = Store(":memory:")
    assert s.put(b"x") == s.put(b"x")
    assert s.db.execute("select count(*) from blobs").fetchone()[0] == 1


def test_a_file_store_survives_reopening(tmp_path):
    path = tmp_path / "deeper" / "spine.db"
    h = Store(path).put(b"kept")
    assert Store(path).get(h) == b"kept"


def test_missing_and_corrupt_blobs_are_refused():
    s = Store(":memory:")
    h = s.put(b"original")
    with s.db:
        s.db.execute("update blobs set data = ? where hash = ?", (b"tampered", h))
    with pytest.raises(ValueError):
        s.get(h)
    with pytest.raises(KeyError):
        s.get("0" * 64)
