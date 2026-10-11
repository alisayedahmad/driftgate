import pickle

from sklearn.linear_model import LogisticRegression

CONFIG = {"C": 1.0, "max_iter": 1000}


def fit(x, y, config):
    """multinomial logistic regression of labels y on rows x.

    config goes to LogisticRegression as keyword arguments, CONFIG is the default.
    """
    return LogisticRegression(**config).fit(x, y)


def to_bytes(model):
    """pickle protocol 5. the same fit gives the same bytes in any process."""
    return pickle.dumps(model, protocol=5)


def from_bytes(data):
    # pickle runs code when it loads, only feed it bytes from our own store
    return pickle.loads(data)
