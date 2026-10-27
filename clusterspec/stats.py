"""Robust statistics used throughout: biweight location/scale (Beers, Flynn & Gebhardt 1990)
and the jackknife.

These follow the astropy.stats definitions and defaults (c = 6 for location, c = 9 for scale,
median as the initial guess), so results match astropy to machine precision. They are written
out here so the package needs only numpy/scipy, and the ``_v`` versions work along the last
axis of an array, which makes Monte Carlo tests fast.
"""
import numpy as np


def biweight_location(x, c=6.0):
    """Biweight location of a 1D sample."""
    x = np.asarray(x, float)
    M = np.median(x)
    d = x - M
    mad = np.median(np.abs(d))
    if mad == 0:
        return float(M)
    u = d / (c * mad)
    m = np.abs(u) < 1
    w = (1 - u ** 2) ** 2
    return float(M + np.sum((d * w)[m]) / np.sum(w[m]))


def biweight_scale(x, c=9.0):
    """Biweight scale ("biweight midvariance" square root) of a 1D sample."""
    x = np.asarray(x, float)
    M = np.median(x)
    d = x - M
    mad = np.median(np.abs(d))
    if mad == 0:
        return 0.0
    u = d / (c * mad)
    m = np.abs(u) < 1
    u2 = u ** 2
    num = x.size * np.sum((d ** 2 * (1 - u2) ** 4)[m])
    den = np.sum(((1 - u2) * (1 - 5 * u2))[m]) ** 2
    return float(np.sqrt(num / den))


def biweight_location_v(x, c=6.0):
    """Biweight location along the last axis (vectorised)."""
    x = np.asarray(x, float)
    M = np.median(x, -1, keepdims=True)
    d = x - M
    mad = np.median(np.abs(d), -1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = d / (c * mad)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        out = M[..., 0] + (d * w).sum(-1) / w.sum(-1)
    return np.where(mad[..., 0] == 0, M[..., 0], out)


def biweight_scale_v(x, c=9.0):
    """Biweight scale along the last axis (vectorised)."""
    x = np.asarray(x, float)
    M = np.median(x, -1, keepdims=True)
    d = x - M
    n = x.shape[-1]
    mad = np.median(np.abs(d), -1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = d / (c * mad)
        m = np.abs(u) < 1
        u2 = u ** 2
        num = n * np.where(m, d ** 2 * (1 - u2) ** 4, 0).sum(-1)
        den = np.where(m, (1 - u2) * (1 - 5 * u2), 0).sum(-1) ** 2
        out = np.sqrt(num / den)
    return np.where(mad[..., 0] == 0, 0.0, out)


def jackknife(x, statistic):
    """Leave-one-out jackknife. Returns (bias-corrected estimate, bias, standard error)."""
    x = np.asarray(x, float)
    n = x.size
    idx = np.arange(n)
    jk = np.array([statistic(x[idx != i]) for i in range(n)])
    stat = statistic(x)
    mean_jk = jk.mean()
    bias = (n - 1) * (mean_jk - stat)
    se = np.sqrt((n - 1) * np.mean((jk - mean_jk) ** 2))
    return stat - bias, bias, se
