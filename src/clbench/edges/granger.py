"""
Granger-precedes edge typing: ``granger_precedes``.

Bivariate Granger causality test on two activation sequences.  Per the
locked design, this edge is framed as **suggestive**, not as proof of
causation — the paper text and any downstream consumers must call it
"Granger-precedes" rather than "causes" to stay honest.

Implementation
--------------
We use ``statsmodels.tsa.stattools.grangercausalitytests`` (Seth 2010;
Barnett & Seth 2014) so the published description can cite the
canonical references.  The test is wrapped so that:

* statsmodels is imported lazily — it's a relatively heavy dependency
  and not required by the rest of the KG construction pipeline;
* time series with insufficient variance (e.g. a flat zero activation
  sequence) return a sentinel "insufficient_signal" result instead of
  raising.

Edge interpretation
-------------------
``granger_precedes_score(A, B)`` tests whether A's past helps predict
B beyond B's own past.  When the p-value is below ``alpha`` for at
least one lag in ``lags_to_test``, we emit a directed
``GRANGER_PRECEDES`` edge from A→B.  The edge weight is
``1 - min_p_value`` clamped to [0, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Default lags to test, in windows.  At 15 s stride: 1, 4, 12 windows ≈
# 15 s, 1 min, 3 min — physiologically meaningful short / medium / longer
# causal windows for ICU vital-sign concepts.
DEFAULT_LAGS_TO_TEST: tuple[int, ...] = (1, 4, 12)
DEFAULT_ALPHA: float = 0.05
DEFAULT_MIN_SAMPLES: int = 60  # minimum aligned samples to run the test


@dataclass
class GrangerRelation:
    """Result of one (A → B) Granger test."""

    direction: str  # "A->B"
    min_p_value: float
    best_lag: int
    f_statistic: float
    n_samples: int
    status: str  # "ok" | "insufficient_signal" | "insufficient_samples" | "test_failed"


def _series_ok(x: np.ndarray, min_samples: int) -> bool:
    if len(x) < min_samples:
        return False
    return float(np.nanstd(x)) > 1e-8


def granger_precedes_score(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    lags_to_test: tuple[int, ...] = DEFAULT_LAGS_TO_TEST,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> GrangerRelation:
    """
    Test whether A Granger-causes B.

    Operates on the raw (continuous) probability sequences rather than the
    binary activations — the test is more powerful with the extra dynamic
    range, and the bivariate VAR assumed by the underlying implementation
    is more meaningful on real-valued series.
    """
    a = np.nan_to_num(np.asarray(probs_a, dtype=np.float64))
    b = np.nan_to_num(np.asarray(probs_b, dtype=np.float64))
    if len(a) != len(b):
        raise ValueError(f"probs length mismatch: a={len(a)} b={len(b)}")

    if len(a) < min_samples:
        return GrangerRelation(
            direction="A->B",
            min_p_value=1.0,
            best_lag=0,
            f_statistic=0.0,
            n_samples=len(a),
            status="insufficient_samples",
        )
    if not (_series_ok(a, min_samples) and _series_ok(b, min_samples)):
        return GrangerRelation(
            direction="A->B",
            min_p_value=1.0,
            best_lag=0,
            f_statistic=0.0,
            n_samples=len(a),
            status="insufficient_signal",
        )

    # Lazy import so unrelated KG code doesn't pay for statsmodels' load time.
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        log.warning("statsmodels not installed — Granger edges disabled")
        return GrangerRelation(
            direction="A->B",
            min_p_value=1.0,
            best_lag=0,
            f_statistic=0.0,
            n_samples=len(a),
            status="test_failed",
        )

    # statsmodels expects shape (T, 2); column 0 is the one being PREDICTED.
    # To test "does A help predict B?" we pass [[B_t, A_t], ...].
    data = np.column_stack([b, a])

    try:
        # verbose=False isn't accepted on older statsmodels versions; gracefully
        # fall back if the kwarg name changed.
        try:
            results = grangercausalitytests(data, maxlag=list(lags_to_test), verbose=False)
        except TypeError:
            results = grangercausalitytests(data, maxlag=list(lags_to_test))
    except Exception as exc:  # noqa: BLE001 — statsmodels can raise many types
        log.debug("Granger test failed: %s", exc)
        return GrangerRelation(
            direction="A->B",
            min_p_value=1.0,
            best_lag=0,
            f_statistic=0.0,
            n_samples=len(a),
            status="test_failed",
        )

    best_p = 1.0
    best_lag = 0
    best_f = 0.0
    for lag, lag_result in results.items():
        test_stats = lag_result[0]
        if "ssr_ftest" not in test_stats:
            continue
        f_stat, p_value, _df_num, _df_den = test_stats["ssr_ftest"]
        if p_value < best_p:
            best_p = float(p_value)
            best_lag = int(lag)
            best_f = float(f_stat)

    return GrangerRelation(
        direction="A->B",
        min_p_value=best_p,
        best_lag=best_lag,
        f_statistic=best_f,
        n_samples=len(a),
        status="ok",
    )


def is_granger_edge(
    rel: GrangerRelation,
    alpha: float = DEFAULT_ALPHA,
) -> bool:
    """True when the Granger test is significant at ``alpha``."""
    return rel.status == "ok" and rel.min_p_value < alpha


def granger_edge_weight(rel: GrangerRelation) -> float:
    """Map a Granger result to an edge weight in [0, 1]."""
    if rel.status != "ok":
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - rel.min_p_value)))
