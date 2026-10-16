from statistics import NormalDist

import numpy as np


def given_up(result, policy):
    """the classes the candidate gives up, as (class, recall change, rows), worst first.

    from compare()'s result. a class counts when it has at least
    policy["class_rows"] held-out labels and its recall falls by more than
    policy["tolerance"] with policy["class_confidence"] at least: the measured
    change plus that many standard errors is still below -tolerance. a small
    class needs a large drop to count, and one with too few rows never does.
    """
    z = NormalDist().inv_cdf(policy["class_confidence"])
    judged = result["class rows"] >= policy["class_rows"]
    sure = result["class gain"] + z * result["class se"] < -policy["tolerance"]
    lost = np.flatnonzero(judged & sure)
    return sorted(((int(c), float(result["class gain"][c]), int(result["class rows"][c]))
                   for c in lost), key=lambda t: t[1])
