"""
Minimal stand-in for the ``dcor`` package (not installed in this environment), providing the one function
tigramite's GPDC test calls: ``distance_correlation(x, y, method=...)`` = the (biased, V-statistic) distance
correlation of Szekely, Rizzo & Bakirov (2007) for one-dimensional samples, computed with double-centred
distance matrices (O(n^2); tigramite calls it on <= a few thousand residuals). ``method`` is accepted and
ignored (the 'AVL' fast algorithm of dcor gives the same number). Registered as ``sys.modules['dcor']`` by
``clbench.extended.nonlinear.install_shims`` only when the real package is absent.
"""

from __future__ import annotations

import numpy as np

__version__ = "0.0-shim"


def _double_centred(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim == 1:
        A = np.abs(np.subtract.outer(x, x))
    else:
        A = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    rm = A.mean(1)
    A -= rm[:, None]
    A -= rm[None, :]
    A += rm.mean()
    return A


def distance_covariance_sqr(x, y) -> float:
    A, B = _double_centred(x), _double_centred(y)
    return float((A * B).mean())


def distance_correlation_sqr(x, y) -> float:
    A, B = _double_centred(x), _double_centred(y)
    dcov2 = (A * B).mean()
    den = np.sqrt((A * A).mean() * (B * B).mean())
    return float(dcov2 / den) if den > 0 else 0.0


def distance_correlation(x, y, method=None, **_) -> float:
    return float(np.sqrt(max(distance_correlation_sqr(x, y), 0.0)))
