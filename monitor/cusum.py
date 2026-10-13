import numpy as np


class Cusum:
    """two-sided CUSUM over several standardized streams, one value each per step.

    with no shift each stream is N(0, 1). k is the allowance in standard units,
    about half the shift the detector is meant for. update returns the largest
    upper or lower sum over all streams: an alarm is that going over a
    threshold h, which is chosen for the false alarm rate it gives.
    """

    def __init__(self, streams, k=0.5):
        self.k = k
        self.up = np.zeros(streams)
        self.down = np.zeros(streams)

    def update(self, z):
        """add one step, z of shape (streams,). a nan leaves that stream's sums as they were."""
        z = np.asarray(z, dtype=float)
        seen = ~np.isnan(z)
        z = np.where(seen, z, 0.0)
        self.up = np.where(seen, np.maximum(0.0, self.up + z - self.k), self.up)
        self.down = np.where(seen, np.maximum(0.0, self.down - z - self.k), self.down)
        return float(max(self.up.max(), self.down.max()))


def first_alarm(path, h):
    """index of the first value of path over h, None if it never gets there."""
    over = np.flatnonzero(np.asarray(path) > h)
    return int(over[0]) if len(over) else None
