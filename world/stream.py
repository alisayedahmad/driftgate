import numpy as np

FEATURES = 6
INFORMATIVE = 4
CLASSES = 4
PRIORS = np.array([0.55, 0.25, 0.15, 0.05])
BAYES = 0.85
CURVATURE = 0.3
ROWS = 100
CALIBRATION_ROWS = 50_000

# don't renumber, every seed would give different data
GEOMETRY, CALIBRATION, X, Y, SCRIPT, CHANGE, NOISE = range(7)


def softmax(s, axis=1):
    e = np.exp(s - s.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class World:
    """fixed world for one seed (seed must fit in 32 bits).

    x is standard normal over FEATURES columns, only the first INFORMATIVE
    ones affect y. P(y|x) is a softmax of a linear + quadratic score,
    calibrated to Bayes accuracy BAYES and class priors PRIORS, up to the
    sampling error of CALIBRATION_ROWS rows.
    """

    def __init__(self, seed):
        if not 0 <= seed < 2**32:
            raise ValueError(f"seed must be in [0, 2**32), got {seed}")
        self.seed = seed
        g = np.random.default_rng([seed, GEOMETRY])
        w = g.standard_normal((INFORMATIVE, CLASSES))
        q = CURVATURE * g.standard_normal((INFORMATIVE, CLASSES))
        v = np.random.default_rng([seed, CALIBRATION]).standard_normal(
            (CALIBRATION_ROWS, INFORMATIVE))
        scale, self.b = _calibrate(v @ w + (v * v) @ q)
        self.w, self.q = scale * w, scale * q

    def proba(self, x):
        """P(y|x) as an (n, CLASSES) array."""
        v = x[:, :INFORMATIVE]
        return softmax(v @ self.w + (v * v) @ self.q + self.b)

    def draws(self, step, n=ROWS):
        """the random numbers behind one step: z of shape (n, FEATURES), u of shape (n,)."""
        z = np.random.default_rng([self.seed, X, step]).standard_normal((n, FEATURES))
        u = np.random.default_rng([self.seed, Y, step]).random(n)
        return z, u

    def batch(self, step, n=ROWS):
        """(x, y) for one step. only depends on the seed and the step."""
        x, u = self.draws(step, n)
        return x, label(self.proba(x), u)


def label(p, u):
    # inverse cdf instead of rng.choice, a row keeps its label when p moves a little
    return (u[:, None] >= p[:, :-1].cumsum(axis=1)).sum(axis=1)


def _calibrate(r):
    # (K, n) layout, reducing over a last axis of length 4 is about 2x slower
    r = np.ascontiguousarray(r.T)
    b = np.log(PRIORS)[:, None]

    def gap(scale):
        nonlocal b
        for _ in range(100):
            p = softmax(scale * r + b, axis=0)
            m = p.mean(axis=1, keepdims=True)
            if np.abs(m / PRIORS[:, None] - 1).max() < 1e-4:
                return p.max(axis=0).mean() - BAYES
            # clipped newton step, unclipped it diverged on 10 seeds out of 200
            j = np.diagflat(m) - p @ p.T / p.shape[1]
            step = np.linalg.lstsq(j, PRIORS[:, None] - m, rcond=None)[0]
            # j is singular along a constant shift of b, lstsq sometimes keeps a big one
            step -= step.mean()
            b = b + np.clip(step, -3, 3)
        raise RuntimeError(f"priors did not converge at scale {scale}")

    lo, flo = 0.0, PRIORS.max() - BAYES
    hi, fhi = 1.0, gap(1.0)
    while fhi < 0:
        lo, flo, hi = hi, fhi, 2 * hi
        fhi = gap(hi)
    # illinois method
    side = 0
    for _ in range(50):
        mid = (lo * fhi - hi * flo) / (fhi - flo)
        f = gap(mid)
        if abs(f) < 1e-5:
            break
        if f < 0:
            lo, flo = mid, f
            fhi = fhi / 2 if side < 0 else fhi
            side = -1
        else:
            hi, fhi = mid, f
            flo = flo / 2 if side > 0 else flo
            side = 1
    else:
        raise RuntimeError("bayes accuracy did not converge")
    return mid, b[:, 0]
