import numpy as np

from monitor.cusum import first_alarm
from monitor.tradeoff import SIGNALS, WINDOW, alarms, operating, tradeoff


def test_alarms_agree_with_first_alarm_for_every_threshold():
    path = np.array([0.0, 3.0, 2.0, 7.0, 5.0, np.inf])
    thresholds = np.array([1.0, 2.5, 6.0, 100.0])
    expected = [first_alarm(path, h) for h in thresholds]
    assert alarms(path, thresholds).tolist() == [len(path) if e is None else e for e in expected]


def test_false_alarms_and_detections_are_counted_apart():
    flat = np.zeros(800)
    noisy = flat.copy()
    noisy[10:] = 5.0
    caught = flat.copy()
    caught[25:] = 20.0
    early = flat.copy()
    early[5:] = 20.0
    runs = [("none", 0, None, dict.fromkeys(SIGNALS, flat)),
            ("none", 1, None, dict.fromkeys(SIGNALS, noisy)),
            ("concept", 2, WINDOW + 20, dict.fromkeys(SIGNALS, caught)),
            ("concept", 3, WINDOW + 20, dict.fromkeys(SIGNALS, early))]
    row = tradeoff(runs, np.array([4.0, 10.0]))["errors"]
    assert row["false"].tolist() == [0.5, 0.0]
    assert row["concept detected"].tolist() == [0.5, 0.5]
    assert row["concept delay"].tolist() == [5.0, 5.0]
    assert np.isnan(row["covariate delay"]).all()
    assert operating({"errors": row}, 0.05) == {"errors": 1}
