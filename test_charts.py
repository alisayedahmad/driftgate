import numpy as np

from charts import rolling


def test_rolling_is_a_trailing_mean_over_full_windows():
    a = np.arange(100.0)
    out = rolling(a, 25)
    assert len(out) == len(a) - 25 + 1
    assert out[0] == a[:25].mean()
    assert out[-1] == a[-25:].mean()
