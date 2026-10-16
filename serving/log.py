import numpy as np

from world.stream import CLASSES, FEATURES

UNKNOWN = -1


class Log:
    """every prediction served, and the labels as they come back.

    steps are served in increasing order and rows keep their serving order.
    a label only enters the log at the step it arrives, and as_of(t) hides any
    that arrived after t, so nothing reads a label before it was known.
    """

    def __init__(self):
        self.n = 0
        self.last = None
        self.models = []
        self.span = {}
        self.landings = {}
        self.columns = {"step": np.empty(0, np.int64), "x": np.empty((0, FEATURES)),
                        "proba": np.empty((0, CLASSES + 1)), "model": np.empty(0, np.int64),
                        "label": np.empty(0, np.int64), "arrived": np.empty(0, np.int64)}

    def served(self, step, x, model, proba):
        """log the rows of `step`, served by model (a hash).

        x is (n, FEATURES), proba (n, CLASSES + 1) with zeros for classes the
        model doesn't know.
        """
        if self.last is not None and step <= self.last:
            raise ValueError(f"step {step} comes after {self.last}")
        if model not in self.models:
            self.models.append(model)
        self._grow(len(x))
        at = slice(self.n, self.n + len(x))
        c = self.columns
        c["step"][at], c["x"][at], c["proba"][at] = step, x, proba
        c["model"][at], c["label"][at], c["arrived"][at] = self.models.index(model), UNKNOWN, UNKNOWN
        self.span[step] = at
        self.last = step
        self.n += len(x)

    def arrived(self, step, served_at, rows, y):
        """labels y for rows (indices within step served_at), known from `step` on."""
        if step <= served_at:
            raise ValueError(f"a label can't arrive at {step} for a row served at {served_at}")
        at = self.span[served_at]
        rows = np.asarray(rows)
        if rows.size and (rows.min() < 0 or rows.max() >= at.stop - at.start):
            raise ValueError(f"step {served_at} has {at.stop - at.start} rows")
        rows = at.start + rows
        if (self.columns["label"][rows] != UNKNOWN).any():
            raise ValueError("a label arrived twice")
        self.columns["label"][rows] = y
        self.columns["arrived"][rows] = step
        self.landings.setdefault(step, []).append(rows)

    def batch(self, step):
        """the rows served at `step`, as copies of x, proba and model (an index into models)."""
        at = self.span[step]
        return {name: self.columns[name][at].copy() for name in ("x", "proba", "model")}

    def landed(self, step):
        """the labels that arrived at `step` with their rows: step served, x, proba, model, label.

        model is an index into models, the one that served the row. arrays are
        empty when nothing arrived.
        """
        at = np.concatenate(self.landings.get(step, [np.empty(0, np.int64)]))
        return {name: self.columns[name][at] for name in ("step", "x", "proba", "model", "label")}

    def as_of(self, t):
        """what was known at step t: the rows served up to t, with labels arrived by t.

        a dict of arrays step, x, proba, model (an index into self.models), label
        and arrived, the last two UNKNOWN where the label hadn't come back.
        """
        end = int(np.searchsorted(self.columns["step"][:self.n], t, side="right"))
        out = {name: column[:end].copy() for name, column in self.columns.items()}
        late = out["arrived"] > t
        out["label"][late] = UNKNOWN
        out["arrived"][late] = UNKNOWN
        return out

    def save(self, db):
        """write the log into an open SQLite connection, as tables predictions and served_by.

        predictions has one row per row served: step, x0.., p0.., model (the
        served_by index of its hash), label and arrived, UNKNOWN where it never
        came back.
        """
        c = {name: column[:self.n] for name, column in self.columns.items()}
        names = (["step"] + [f"x{i}" for i in range(FEATURES)]
                 + [f"p{k}" for k in range(CLASSES + 1)] + ["model", "label", "arrived"])
        rows = zip(c["step"].tolist(), *c["x"].T.tolist(), *c["proba"].T.tolist(),
                   c["model"].tolist(), c["label"].tolist(), c["arrived"].tolist())
        with db:
            db.execute(f"create table predictions ({', '.join(names)})")
            db.executemany(f"insert into predictions values ({', '.join('?' * len(names))})", rows)
            db.execute("create table served_by (model integer primary key, hash text)")
            db.executemany("insert into served_by values (?, ?)", enumerate(self.models))

    def _grow(self, k):
        size = len(self.columns["step"])
        if self.n + k <= size:
            return
        size = max(2 * size, self.n + k, 1024)
        for name, column in self.columns.items():
            bigger = np.empty((size, *column.shape[1:]), column.dtype)
            bigger[:self.n] = column[:self.n]
            self.columns[name] = bigger
