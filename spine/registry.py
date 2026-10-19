import json
from pathlib import Path

from spine.provenance import canonical, stream
from spine.store import Store

FIELDS = ("model", "trained_on", "evaluated_on", "config", "code", "versions", "metrics",
          "parent", "step")
BLOBS = ("model", "trained_on", "evaluated_on", "config")
RUNS = Path.home() / ".local/share/driftgate"


def fresh(name):
    """an empty registry for one run, in RUNS/name.db, wiping what a previous run left.

    one registry per run: a rerun after an edit gives the same model with another
    code record, which a registry refuses.
    """
    path = RUNS / f"{name}.db"
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    return Registry(Store(path))


class Registry:
    """what made each model, kept next to the artifacts in a Store.

    a record is a dict with exactly FIELDS. model, trained_on, evaluated_on and
    config are hashes of blobs in the store (the model bytes, two dataset
    documents, the config as canonical JSON). code and versions come from
    spine.provenance, metrics were measured on evaluated_on. parent is the
    model that was serving when this one was trained, or None for the first,
    and step is the simulation step it was trained at. no wall clock anywhere,
    so two identical runs give identical registries.
    """

    def __init__(self, store):
        self.store = store
        self.store.db.execute("create table if not exists models "
                              "(model text primary key, record text not null)")

    def register(self, record):
        """add a record and return its model hash.

        ValueError if the fields aren't FIELDS, if a blob is missing from the
        store, if the parent isn't registered, or if the model already has a
        different record. registering the same record again does nothing.
        """
        if set(record) != set(FIELDS):
            raise ValueError(f"a record has exactly {FIELDS}, got {sorted(record)}")
        missing = [field for field in BLOBS if record[field] not in self.store]
        if missing:
            raise ValueError(f"not in the store: {missing}")
        if record["parent"] is not None and self.get(record["parent"]) is None:
            raise ValueError(f"parent {record['parent']} isn't registered")
        known = self.get(record["model"])
        if known is not None and known != json.loads(canonical(record)):
            raise ValueError(f"model {record['model']} is already registered with another record")
        with self.store.db:
            self.store.db.execute("insert or ignore into models values (?, ?)",
                                  (record["model"], canonical(record).decode()))
        return record["model"]

    def get(self, model):
        """the record of a model, or None if it isn't registered."""
        row = self.store.db.execute("select record from models where model = ?",
                                    (model,)).fetchone()
        return None if row is None else json.loads(row[0])

    def lineage(self, model):
        """records from model back to the first one, following parents. KeyError if unknown."""
        chain = []
        while model is not None:
            record = self.get(model)
            if record is None:
                raise KeyError(model)
            chain.append(record)
            model = record["parent"]
        return chain

    def stream(self, model):
        """the stream a model was trained on, rebuilt from the recipe of its training rows."""
        record = self.get(model)
        return stream(json.loads(self.store.get(record["trained_on"]))["recipe"]["scenario"])

    def records(self):
        """every record, by the step its model was trained at, then by model."""
        rows = self.store.db.execute("select record from models").fetchall()
        return sorted((json.loads(row[0]) for row in rows), key=lambda r: (r["step"], r["model"]))

    def dump(self):
        """every record as canonical JSON, one per line, sorted by model."""
        rows = self.store.db.execute("select record from models order by model").fetchall()
        return "".join(row[0] + "\n" for row in rows)
