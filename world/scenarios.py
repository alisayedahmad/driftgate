from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np

from world.stream import (CALIBRATION_ROWS, CHANGE, CLASSES, FEATURES, INFORMATIVE, MEMBER,
                          NOISE, ROWS, SCRIPT, World, label)

KINDS = ("none", "covariate", "concept", "noise", "new")
AT = (300, 700)
SIZES = {"covariate": (0.25, 1.5), "concept": (0.03, 0.25), "noise": (0.1, 0.4),
         "new": (0.02, 0.15)}
ANGLES = np.linspace(0, np.pi, 17)
PLANES = 10
NEW_DISTANCE = 3.0
NEW_SPREAD = 0.5


@cache
def _world(seed):
    return World(seed)


@dataclass(frozen=True)
class Scenario:
    """a world with one abrupt change at step `at`.

    size is in standard deviations for covariate, in accuracy lost by the old
    Bayes classifier for concept (0.1 is 10 points), and is the share of the
    source class relabeled for noise. for new it is the share of rows from the
    new class, rows the old classifier always gets wrong since it can't predict
    that class. what the change applies to comes from the seed, so two
    scenarios that differ only in size change the same thing by a different
    amount.
    """

    seed: int
    kind: str = "none"
    at: int | None = None
    size: float = 0.0

    def __post_init__(self):
        if not 0 <= self.seed < 2**32:
            raise ValueError(f"seed must be in [0, 2**32), got {self.seed}")
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}, expected one of {KINDS}")
        if (self.at is None) != (self.kind == "none"):
            raise ValueError(f"{self.kind} with at={self.at}: only none goes without a step")

    @classmethod
    def draw(cls, seed, kind):
        """step in AT (both ends included) and size in SIZES[kind] (top excluded).

        both come from the seed only, so a seed gives the same step for every
        kind, and a size at the same point of each kind's range.
        """
        if kind == "none":
            return cls(seed)
        if kind not in SIZES:
            raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")
        g = np.random.default_rng([seed, SCRIPT])
        return cls(seed, kind, int(g.integers(AT[0], AT[1] + 1)), float(g.uniform(*SIZES[kind])))

    @property
    def world(self):
        return _world(self.seed)

    def twin(self):
        return Scenario(self.seed)

    @cached_property
    def shift(self):
        """added to x from step `at` on. one column, either sign."""
        g = np.random.default_rng([self.seed, CHANGE])
        d = np.zeros(FEATURES)
        d[g.integers(FEATURES)] = g.choice([-1.0, 1.0]) * self.size
        return d

    @cached_property
    def rotation(self):
        """applied to the informative columns before scoring, from step `at` on.

        the plane is the first drawn from the seed that can cost the old Bayes
        classifier the largest concept size in SIZES, so every size in that
        range uses the same plane. the angle is the smallest that costs `size`.
        ValueError if no plane out of PLANES gets there.
        """
        g = np.random.default_rng([self.seed, CHANGE])
        v = g.standard_normal((CALIBRATION_ROWS, INFORMATIVE))
        p = self.world.proba(v)
        best, old, rows = p.max(axis=1), p.argmax(axis=1), np.arange(len(v))
        reach = max(self.size, SIZES["concept"][1])
        if reach >= best.mean():
            raise ValueError(f"a concept break can't cost {reach}, seed {self.seed} "
                             f"is only {best.mean():.3f} accurate")

        def harm(r):
            return (best - self.world.proba(v @ r.T)[rows, old]).mean()

        for _ in range(PLANES):
            q = np.linalg.qr(g.standard_normal((INFORMATIVE, INFORMATIVE)))[0]
            # harm rose monotonically up to 0.25 on every plane tried, 1000 seeds
            harms = [0.0]
            for angle in ANGLES[1:]:
                harms.append(harm(_rotate(q, angle)))
                if harms[-1] >= reach:
                    break
            else:
                continue
            hit = next(i for i in range(1, len(harms)) if harms[i] >= self.size)
            lo, hi = ANGLES[hit - 1], ANGLES[hit]
            for _ in range(11):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if harm(_rotate(q, mid)) < self.size else (lo, mid)
            return _rotate(q, hi)
        raise ValueError(f"no plane out of {PLANES} costs {reach} accuracy in seed {self.seed}")

    @cached_property
    def flip(self):
        """(source, target): noise relabels some of the source class as target."""
        g = np.random.default_rng([self.seed, CHANGE])
        source = int(g.integers(CLASSES))
        return source, (source + 1 + int(g.integers(CLASSES - 1))) % CLASSES

    @cached_property
    def center(self):
        """where the new class sits in the informative features, NEW_DISTANCE from 0."""
        d = np.random.default_rng([self.seed, CHANGE]).standard_normal(INFORMATIVE)
        return NEW_DISTANCE * d / np.linalg.norm(d)

    def batch(self, step, n=ROWS):
        """(x, y) for one step, from the same random numbers as the twin.

        y is what the pipeline gets to see. under noise it differs from the
        true label, which is the twin's. under new, y can be CLASSES.
        """
        x, u = self.world.draws(step, n)
        if self.kind == "covariate" and step >= self.at:
            x = x + self.shift
        if self.kind == "new" and step >= self.at:
            new = np.random.default_rng([self.seed, MEMBER, step]).random(n) < self.size
            x = x.copy()
            x[new, :INFORMATIVE] = self.center + NEW_SPREAD * x[new, :INFORMATIVE]
            # rows are labeled by the group they came from, proba is the posterior
            y = label(self.world.proba(x), u)
            y[new] = CLASSES
            return x, y
        y = label(self.proba(x, step), u)
        if self.kind == "noise" and step >= self.at:
            source, target = self.flip
            hit = np.random.default_rng([self.seed, NOISE, step]).random(n) < self.size
            y = np.where(hit & (y == source), target, y)
        return x, y

    def proba(self, x, step):
        """true P(y|x) at `step`.

        under new, from step `at` on, the new class is column CLASSES and rows
        have CLASSES + 1 columns.
        """
        if self.kind == "concept" and step >= self.at:
            x = x[:, :INFORMATIVE] @ self.rotation.T
        p = self.world.proba(x)
        if self.kind == "new" and step >= self.at:
            v = x[:, :INFORMATIVE]
            # log odds of the new group against the old one, the other columns cancel
            odds = (np.log(self.size / (1 - self.size)) - INFORMATIVE * np.log(NEW_SPREAD)
                    - ((v - self.center) ** 2).sum(axis=1) / (2 * NEW_SPREAD**2)
                    + (v * v).sum(axis=1) / 2)
            novel = np.exp(-np.logaddexp(0, -odds))
            p = np.hstack([p * (1 - novel)[:, None], novel[:, None]])
        return p


def _rotate(q, angle):
    # by `angle` radians in the plane of the first two columns of q, which must be orthonormal
    a, b = q[:, 0], q[:, 1]
    return (np.eye(len(a)) + (np.cos(angle) - 1) * (np.outer(a, a) + np.outer(b, b))
            + np.sin(angle) * (np.outer(b, a) - np.outer(a, b)))
