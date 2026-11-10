"""
Statistical utilities for CLBench evaluation.

Bootstrap confidence intervals (Efron & Tibshirani 1993) are the workhorse
for every reported metric: resample the per-unit values (or paired
(y_true, y_score) records) with replacement, recompute the statistic, and
take percentile bounds.  This avoids parametric assumptions about the
sampling distribution of AUROC/AUPRC/F1, which are not normally distributed
in small samples.

We also provide a paired bootstrap for metric *differences* (used by the
ablation suite in later phases) so deltas come with their own CIs rather
than relying on a separate significance test.

All functions are pure-numpy and deterministic given ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_N_RESAMPLES: int = 1000
DEFAULT_ALPHA: float = 0.05


@dataclass(frozen=True)
class CI:
    """A point estimate with a (1 - alpha) confidence interval."""

    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> CI:
    """
    Percentile bootstrap CI for a statistic of a 1-D sample.

    Parameters
    ----------
    values : (N,) ndarray of per-unit values (e.g. per-patient Jaccards).
    statistic : reduction applied to each resample (default mean).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        v = float(statistic(values))
        return CI(v, v, v, 1)

    rng = np.random.default_rng(seed)
    point = float(statistic(values))
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        boot[i] = statistic(sample)
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return CI(point, lo, hi, n)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> CI:
    """
    Percentile bootstrap CI for a paired metric like AUROC/AUPRC.

    Resamples *record indices* with replacement so each resample keeps the
    (label, score) pairing.  Resamples that end up single-class (no positive
    or no negative) are skipped, since ranking metrics are undefined there.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return CI(float("nan"), float("nan"), float("nan"), n)

    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_score))
    boot: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot.append(metric_fn(yt, y_score[idx]))
    if not boot:
        return CI(point, float("nan"), float("nan"), n)
    arr = np.asarray(boot)
    lo = float(np.percentile(arr, 100 * (alpha / 2)))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return CI(point, lo, hi, n)


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> CI:
    """
    Bootstrap CI for ``statistic(a) - statistic(b)`` on paired samples.

    ``values_a`` and ``values_b`` must be aligned per unit (e.g. per-patient
    metric under model A vs B).  A CI excluding 0 indicates a significant
    difference at level ``alpha``.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    point = float(statistic(a) - statistic(b))
    boot = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(a[idx]) - statistic(b[idx])
    lo = float(np.percentile(boot, 100 * (alpha / 2)))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return CI(point, lo, hi, n)


# ---------------------------------------------------------------------------
# Ranking metrics (numpy-only so eval has no sklearn dependency)
# ---------------------------------------------------------------------------


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under ROC via the Mann-Whitney U statistic (handles ties)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average ranks for ties.
    s = y_score[order]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_pos = ranks[y_true == 1].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under precision-recall, step interpolation)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / yt.sum()
    rec_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - rec_prev) * precision))
