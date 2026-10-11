import hashlib
import sqlite3
from pathlib import Path


class Store:
    """bytes kept under the sha256 of their content.

    path is a SQLite file, created with its folder if needed, or ":memory:"
    for a store that only lives as long as the object.
    """

    def __init__(self, path):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("create table if not exists blobs (hash text primary key, data blob not null)")

    def put(self, data):
        """store bytes and return their sha256 as 64 hex characters. storing twice keeps one copy."""
        h = hashlib.sha256(data).hexdigest()
        with self.db:
            self.db.execute("insert or ignore into blobs values (?, ?)", (h, data))
        return h

    def get(self, h):
        """the bytes stored under h. KeyError if there are none, ValueError if they don't hash to h."""
        row = self.db.execute("select data from blobs where hash = ?", (h,)).fetchone()
        if row is None:
            raise KeyError(h)
        if hashlib.sha256(row[0]).hexdigest() != h:
            raise ValueError(f"blob {h} doesn't match its hash")
        return row[0]
