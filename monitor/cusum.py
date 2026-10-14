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
        self.updates = 0
        # the update each sum last rose from zero at
        self.rose_up = np.zeros(streams, dtype=int)
        self.rose_down = np.zeros(streams, dtype=int)

    def update(self, z):
        """add one step, z of shape (streams,). a nan leaves that stream's sums as they were."""
        z = np.asarray(z, dtype=float)
        seen = ~np.isnan(z)
        z = np.where(seen, z, 0.0)
        up = np.where(seen, np.maximum(0.0, self.up + z - self.k), self.up)
        down = np.where(seen, np.maximum(0.0, self.down - z - self.k), self.down)
        self.rose_up[(self.up == 0) & (up > 0)] = self.updates
        self.rose_down[(self.down == 0) & (down > 0)] = self.updates
        self.up, self.down = up, down
        self.updates += 1
        return float(max(up.max(), down.max()))

    def start(self):
        """index of the update the largest sum last rose from zero at, counting from 0.

        at an alarm this is Page's estimate of where the change began: the
        first update after that sum's last zero, skipping updates where its
        stream had no value. None while every sum is zero.
        """
        sums = np.concatenate([self.up, self.down])
        if sums.max() == 0:
            return None
        return int(np.concatenate([self.rose_up, self.rose_down])[sums.argmax()])


class Monitor:
    """a CUSUM fed once a step, keeping what it said after each step.

    after step t, steps holds the t, path the CUSUM statistic and starts the
    step where the excursion behind it began, None when there was none. at an
    alarm, that is where the change is estimated to have started.
    """

    def __init__(self, streams, k):
        self.cusum = Cusum(streams, k)
        self.steps, self.path, self.starts = [], [], []

    def record(self, step, z):
        self.steps.append(step)
        self.path.append(self.cusum.update(z))
        i = self.cusum.start()
        self.starts.append(None if i is None else self.steps[i])


def first_alarm(path, h):
    """index of the first value of path over h, None if it never gets there."""
    over = np.flatnonzero(np.asarray(path) > h)
    return int(over[0]) if len(over) else None
