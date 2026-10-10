from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np

from world.stream import (CALIBRATION_ROWS, CHANGE, FEATURES, INFORMATIVE, ROWS, SCRIPT,
                          World, label)

KINDS = ("none", "covariate", "concept")
AT = (300, 700)
SIZES = {"covariate": (0.25, 1.5), "concept": (0.03, 0.25)}
ANGLES = np.linspace(0, np.pi, 33)
PLANES = 10


@cache
def world(seed):
    return World(seed)


@dataclass(frozen=True)
class Scenario:
    """a world with one abrupt change at step `at`.

    size is how strong the change is: standard deviations for covariate,
    accuracy lost by the old Bayes classifier for concept (0.1 is 10 points).
    what it applies to comes from the seed, so two scenarios that differ only
    in size change the same thing by a different amount.
    """

    seed: int
    kind: str = "none"
    at: int | None = None
    size: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}, expected one of {KINDS}")
        if (self.at is None) != (self.kind == "none"):
            raise ValueError(f"{self.kind} with at={self.at}: only none goes without a step")

    @classmethod
    def draw(cls, seed, kind):
        """step in AT and size in SIZES[kind], both drawn from the seed (inclusive)."""
        if kind == "none":
            return cls(seed)
        g = np.random.default_rng([seed, SCRIPT])
        return cls(seed, kind, int(g.integers(AT[0], AT[1] + 1)), float(g.uniform(*SIZES[kind])))

    @property
    def world(self):
        return world(self.seed)

    def twin(self):
        return Scenario(self.seed)

    @cached_property
    def shift(self):
        """added to x from step `at` on: one column, either direction."""
        g = np.random.default_rng([self.seed, CHANGE])
        d = np.zeros(FEATURES)
        d[g.integers(FEATURES)] = g.choice([-1.0, 1.0]) * self.size
        return d

    @cached_property
    def rotation(self):
        """applied to the informative columns before scoring, from step `at` on.

        the plane comes from the seed, the angle is the smallest that costs the
        old Bayes classifier `size` accuracy. if the plane can't get there the
        next one is tried, ValueError after PLANES.
        """
        g = np.random.default_rng([self.seed, CHANGE])
        v = g.standard_normal((CALIBRATION_ROWS, INFORMATIVE))
        p = self.world.proba(v)
        best, old, rows = p.max(axis=1), p.argmax(axis=1), np.arange(len(v))

        def harm(r):
            return (best - self.world.proba(v @ r.T)[rows, old]).mean()

        for _ in range(PLANES):
            q = np.linalg.qr(g.standard_normal((INFORMATIVE, INFORMATIVE)))[0]
            # harm rises monotonically up to its peak on all of 1000 seeds tried
            hit = next((i for i in range(1, len(ANGLES))
                        if harm(rotate(q, ANGLES[i])) >= self.size), None)
            if hit is None:
                continue
            lo, hi = ANGLES[hit - 1], ANGLES[hit]
            for _ in range(10):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if harm(rotate(q, mid)) < self.size else (lo, mid)
            return rotate(q, hi)
        raise ValueError(f"no plane out of {PLANES} costs {self.size} accuracy in seed {self.seed}")

    def batch(self, step, n=ROWS):
        """(x, y) for one step, from the same random numbers as the twin."""
        x, u = self.world.draws(step, n)
        if self.kind == "covariate" and step >= self.at:
            x = x + self.shift
        return x, label(self.proba(x, step), u)

    def proba(self, x, step):
        """true P(y|x) at `step`."""
        if self.kind == "concept" and step >= self.at:
            x = x[:, :INFORMATIVE] @ self.rotation.T
        return self.world.proba(x)


def rotate(q, angle):
    # rotation by `angle` in the plane of q's first two columns
    a, b = q[:, 0], q[:, 1]
    return (np.eye(len(a)) + (np.cos(angle) - 1) * (np.outer(a, a) + np.outer(b, b))
            + np.sin(angle) * (np.outer(b, a) - np.outer(a, b)))
