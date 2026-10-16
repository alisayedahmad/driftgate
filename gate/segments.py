import numpy as np


def given_up(result, policy):
    """the classes the candidate gives up, as (class, recall change, rows), worst first.

    from compare()'s result. a class counts when it has at least
    policy["class_rows"] held-out labels and its recall falls by more than
    policy["tolerance"]. smaller classes are too noisy to judge and never count.
    """
    judged = result["class rows"] >= policy["class_rows"]
    lost = np.flatnonzero(judged & (result["class gain"] < -policy["tolerance"]))
    return sorted(((int(c), float(result["class gain"][c]), int(result["class rows"][c]))
                   for c in lost), key=lambda t: t[1])
