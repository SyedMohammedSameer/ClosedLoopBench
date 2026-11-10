"""
Temporal edge typing: ``precedes`` / ``follows``.

Pair of concept activation sequences on a shared window axis ⇒ a directed
edge from concept A to concept B when A's onset windows systematically
appear before B's onset windows with a consistent lag.

The method is intentionally simple — lagged cross-correlation of the binary
activation sequences, restricted to a physiologically plausible lag window
(default ±30 min) — which is enough to defend the "temporal" edge type
without overfitting to a single statistical test.  We also report the
median onset-difference as a sanity diagnostic.

A pair (A, B) emits up to two edges: a ``PRECEDES`` edge from A→B with
weight = normalised peak cross-correlation, and the mirror ``FOLLOWS``
edge from B→A.  Both are emitted only when the lag is non-zero and the
peak cross-correlation crosses a configurable threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from clbench.persistence import WINDOW_STRIDE_SEC_DEFAULT

log = logging.getLogger(__name__)

# Default search window for the lag.  ±30 min at 15 s stride = ±120 windows.
DEFAULT_MAX_LAG_SEC: float = 30.0 * 60.0
# Minimum normalised cross-correlation peak to emit an edge.
DEFAULT_MIN_XCORR: float = 0.15
# Minimum lag magnitude (seconds) to treat the relationship as ordered.
# Pairs with near-zero peak lag should become CO_OCCURS edges instead.
DEFAULT_MIN_LAG_SEC: float = 15.0


@dataclass
class TemporalRelation:
    """Result of one (A, B) temporal-edge test."""

    lag_sec: float  # +ve ⇒ A precedes B; -ve ⇒ A follows B
    peak_xcorr: float  # normalised peak cross-correlation in [-1, 1]
    median_onset_diff_sec: float  # median(B_onsets) - median(A_onsets)
    n_runs_a: int
    n_runs_b: int

    @property
    def a_precedes_b(self) -> bool:
        return self.lag_sec > 0.0


def _binary(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (np.nan_to_num(np.asarray(probs, dtype=np.float64), nan=-np.inf) >= threshold).astype(
        np.float64
    )


def _onset_indices(active: np.ndarray) -> np.ndarray:
    """Return the indices where ``active`` transitions 0 → 1."""
    if len(active) == 0:
        return np.array([], dtype=np.int64)
    prev = np.concatenate(([0.0], active[:-1]))
    return np.where((active == 1.0) & (prev == 0.0))[0]


def temporal_relation(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
    window_stride_sec: float = WINDOW_STRIDE_SEC_DEFAULT,
    max_lag_sec: float = DEFAULT_MAX_LAG_SEC,
) -> TemporalRelation:
    """
    Estimate the lag at which concept A best precedes concept B.

    Uses normalised cross-correlation of the binary activation series,
    restricted to ``|lag| <= max_lag_sec``.

    The reported lag is in seconds, positive when A→B (A precedes B).
    """
    a = _binary(probs_a, threshold_a)
    b = _binary(probs_b, threshold_b)
    if len(a) != len(b):
        raise ValueError(f"probs length mismatch: a={len(a)} b={len(b)}")
    if len(a) < 2:
        return TemporalRelation(0.0, 0.0, 0.0, 0, 0)

    # Centre to zero-mean so cross-correlation is unbiased by base-rate.
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.sqrt((a_c**2).sum() * (b_c**2).sum())) or 1.0

    max_lag = max(1, int(round(max_lag_sec / window_stride_sec)))
    max_lag = min(max_lag, len(a) - 1)

    # xcorr[k] = sum_t a_c[t] * b_c[t + k]  (positive k ⇒ A leads B)
    lags = np.arange(-max_lag, max_lag + 1)
    xc = np.empty_like(lags, dtype=np.float64)
    for i, k in enumerate(lags):
        if k >= 0:
            xc[i] = float((a_c[: len(a) - k] * b_c[k:]).sum())
        else:
            xc[i] = float((a_c[-k:] * b_c[: len(b) + k]).sum())
    xc /= denom

    peak_idx = int(np.argmax(xc))
    peak_lag_windows = int(lags[peak_idx])
    peak_xcorr = float(xc[peak_idx])

    # Diagnostic: median onset difference
    a_onsets = _onset_indices(a)
    b_onsets = _onset_indices(b)
    if len(a_onsets) and len(b_onsets):
        median_diff = float(np.median(b_onsets) - np.median(a_onsets)) * window_stride_sec
    else:
        median_diff = 0.0

    return TemporalRelation(
        lag_sec=peak_lag_windows * window_stride_sec,
        peak_xcorr=peak_xcorr,
        median_onset_diff_sec=median_diff,
        n_runs_a=len(a_onsets),
        n_runs_b=len(b_onsets),
    )


def is_directed_temporal_edge(
    rel: TemporalRelation,
    min_xcorr: float = DEFAULT_MIN_XCORR,
    min_lag_sec: float = DEFAULT_MIN_LAG_SEC,
) -> bool:
    """True when ``rel`` qualifies as a directed precedes/follows edge."""
    return rel.peak_xcorr >= min_xcorr and abs(rel.lag_sec) >= min_lag_sec
