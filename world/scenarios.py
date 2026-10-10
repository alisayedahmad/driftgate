from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np

from world.stream import CHANGE, FEATURES, ROWS, SCRIPT, World, label

KINDS = ("none", "covariate")
AT = (300, 700)
SHIFT = (0.25, 1.5)


@cache
def world(seed):
    return World(seed)


@dataclass(frozen=True)
class Scenario:
    """a world with one abrupt change at step `at`.

    size is how strong the change is, in standard deviations for covariate.
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
        """step in AT and size in SHIFT, both drawn from the seed (inclusive)."""
        if kind == "none":
            return cls(seed)
        g = np.random.default_rng([seed, SCRIPT])
        return cls(seed, kind, int(g.integers(AT[0], AT[1] + 1)), float(g.uniform(*SHIFT)))

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

    def batch(self, step, n=ROWS):
        """(x, y) for one step, from the same random numbers as the twin."""
        x, u = self.world.draws(step, n)
        if self.kind == "covariate" and step >= self.at:
            x = x + self.shift
        return x, label(self.proba(x, step), u)

    def proba(self, x, step):
        """true P(y|x) at `step`."""
        return self.world.proba(x)
