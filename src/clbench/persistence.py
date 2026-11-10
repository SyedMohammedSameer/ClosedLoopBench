"""
Phase 2.3 — Persistence-tier calibration and classification.

This module implements the {transient, episodic, persistent} tier typing
that turns raw CBM activation sequences into typed KG nodes.  The
locked design from the project memory is:

    node = (SNOMED concept, persistence-tier)
    persistence-tier ∈ {transient, episodic, persistent}
    derived from activation duration in source signal windows.

Concretely:

1. The CBM produces a probability per (window, concept).  Each patient's
   activation sequence is a 1-D series of probs over chronologically
   ordered windows.
2. Threshold-gate the probs into binary activations.  Threshold is chosen
   per concept by ``calibrate_activation_threshold`` (best-F1 against the
   rule-derived labels we already have from Phase 2.1) or fixed at 0.5.
3. Find contiguous runs of activations within a patient — each run has a
   duration in seconds.
4. Classify each run by duration into a :class:`PersistenceTier`.
   Cutoffs (``transient_max_sec``, ``episodic_max_sec``) come from
   ``calibrate_tier_thresholds`` on the per-concept distribution of run
   durations.

The calibration is data-driven and per-concept, as required by the
locked plan ("data-driven calibration study, not hand-picked").  The
output is a JSON config that 3.1 KG construction consumes when emitting
(concept, tier) nodes.

This module is intentionally CBM-agnostic — it operates on plain numpy
arrays of probabilities so it can be unit-tested on synthetic activation
sequences without any encoder / CBM machinery loaded.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Re-export the canonical enum from schema.py so that classification code and
# the KG data model agree on instance identity.  Defining a parallel Enum here
# would silently break ``classify_run(...) == schema.PersistenceTier.TRANSIENT``
# equality comparisons.
from clbench.schema import PersistenceTier

log = logging.getLogger(__name__)

__all__ = [
    "PersistenceTier",
    "ActivationRun",
    "TierThresholds",
    "CalibrationConfig",
    "find_activation_runs",
    "calibrate_activation_threshold",
    "calibrate_tier_thresholds",
    "calibrate_concept",
    "WINDOW_STRIDE_SEC_DEFAULT",
    "TIER_PERCENTILES_DEFAULT",
    "DEFAULT_ACTIVATION_THRESHOLD",
]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Window cadence — 30 s window @ 15 s stride (50 % overlap) from
# clbench.data.preprocess.  Each new window adds ``WINDOW_STRIDE_SEC``
# of fresh time to a run.
WINDOW_STRIDE_SEC_DEFAULT: float = 15.0

# Percentile cutoffs used when calibrating tier boundaries from a run-duration
# distribution.  Lower tercile → transient, upper tercile → persistent.
TIER_PERCENTILES_DEFAULT: tuple[float, float] = (33.0, 67.0)

# Default activation threshold when calibration data isn't available.
DEFAULT_ACTIVATION_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Enums + dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActivationRun:
    """One contiguous span of above-threshold activation for one concept."""

    start_window_idx: int  # inclusive
    end_window_idx: int  # inclusive
    duration_sec: float
    mean_activation: float
    n_windows: int


@dataclass
class TierThresholds:
    """Per-concept tier-classification thresholds (durations in seconds)."""

    concept: str
    transient_max_sec: float  # runs <= this → TRANSIENT
    episodic_max_sec: float  # runs <= this → EPISODIC; longer → PERSISTENT
    activation_threshold: float
    n_runs_observed: int
    source: str  # "calibrated" | "default" | "fallback"

    def classify(self, duration_sec: float) -> PersistenceTier:
        if duration_sec <= self.transient_max_sec:
            return PersistenceTier.TRANSIENT
        if duration_sec <= self.episodic_max_sec:
            return PersistenceTier.EPISODIC
        return PersistenceTier.PERSISTENT


@dataclass
class CalibrationConfig:
    """One full per-concept persistence-tier calibration record."""

    concepts: dict[str, TierThresholds]
    window_stride_sec: float
    tier_percentiles: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_stride_sec": self.window_stride_sec,
            "tier_percentiles": list(self.tier_percentiles),
            "concepts": {k: dataclasses.asdict(v) for k, v in self.concepts.items()},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("Persistence calibration → %s", path)

    @classmethod
    def load(cls, path: Path) -> CalibrationConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        concepts = {k: TierThresholds(**v) for k, v in raw["concepts"].items()}
        return cls(
            concepts=concepts,
            window_stride_sec=float(raw["window_stride_sec"]),
            tier_percentiles=tuple(raw["tier_percentiles"]),
        )


# ---------------------------------------------------------------------------
# Run extraction
# ---------------------------------------------------------------------------


def find_activation_runs(
    probs: np.ndarray,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
    window_stride_sec: float = WINDOW_STRIDE_SEC_DEFAULT,
    min_run_length: int = 1,
) -> list[ActivationRun]:
    """
    Find contiguous spans where the activation exceeds ``threshold``.

    Parameters
    ----------
    probs : (N,) float ndarray
        Per-window concept probabilities, chronologically ordered.  NaNs are
        treated as below threshold.
    threshold : float
        Activation cutoff in [0, 1].
    window_stride_sec : float
        Seconds added per consecutive window (see :data:`WINDOW_STRIDE_SEC_DEFAULT`).
        A run of ``n`` windows has duration ``(n - 1) * stride + window_len``;
        because we don't carry the window length here, we approximate as
        ``n * window_stride_sec`` (correct to within one stride).
    min_run_length : int
        Drop runs shorter than this many windows.  Useful for ignoring single
        flicker frames at the activation boundary.

    Returns
    -------
    list[ActivationRun] — possibly empty.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1-D, got shape {probs.shape}")
    if len(probs) == 0:
        return []

    active = np.nan_to_num(probs, nan=-np.inf) >= threshold

    runs: list[ActivationRun] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        run_len = j - i
        if run_len >= min_run_length:
            seg = probs[i:j]
            # Mean over the active windows; NaNs treated as 0 so the mean is
            # always finite when run_len > 0.
            seg_mean = float(np.nan_to_num(seg).mean())
            runs.append(
                ActivationRun(
                    start_window_idx=i,
                    end_window_idx=j - 1,
                    duration_sec=run_len * window_stride_sec,
                    mean_activation=seg_mean,
                    n_windows=run_len,
                )
            )
        i = j
    return runs


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_activation_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    candidate_thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Pick the activation threshold that maximises F1 against rule-labels.

    Used when we have ground truth (e.g. from the Phase 2.1 detectors)
    aligned to the same window order as ``probs``.

    Returns
    -------
    (threshold, best_f1)
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if probs.shape != labels.shape:
        raise ValueError(f"probs shape {probs.shape} != labels shape {labels.shape}")
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.05, 0.95, 19)

    best_t = DEFAULT_ACTIVATION_THRESHOLD
    best_f1 = -1.0
    for t in candidate_thresholds:
        pred = (probs >= t).astype(np.int32)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        denom = 2 * tp + fp + fn
        if denom == 0:
            f1 = 0.0
        else:
            f1 = 2 * tp / denom
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def calibrate_tier_thresholds(
    runs: list[ActivationRun],
    concept: str,
    activation_threshold: float,
    percentiles: tuple[float, float] = TIER_PERCENTILES_DEFAULT,
    fallback_transient_sec: float = 60.0,
    fallback_episodic_sec: float = 600.0,
) -> TierThresholds:
    """
    Pick (transient_max, episodic_max) duration cutoffs from a run-duration
    distribution.  Uses lower/upper percentiles of observed durations.

    When too few runs are observed (< 5) the function falls back to clinically
    sane defaults (60 s / 600 s) and marks the source as "fallback" so the
    paper can disclose where calibration was data-driven vs prior-driven.
    """
    if percentiles[0] >= percentiles[1]:
        raise ValueError(f"percentiles must be (lower, upper); got {percentiles}")

    if len(runs) < 5:
        return TierThresholds(
            concept=concept,
            transient_max_sec=fallback_transient_sec,
            episodic_max_sec=fallback_episodic_sec,
            activation_threshold=activation_threshold,
            n_runs_observed=len(runs),
            source="fallback",
        )

    durations = np.array([r.duration_sec for r in runs], dtype=np.float64)
    low_p, high_p = percentiles
    t_max, e_max = np.percentile(durations, [low_p, high_p])
    # Guarantee strict ordering even on degenerate distributions.
    if t_max >= e_max:
        e_max = t_max + 1.0
    return TierThresholds(
        concept=concept,
        transient_max_sec=float(t_max),
        episodic_max_sec=float(e_max),
        activation_threshold=float(activation_threshold),
        n_runs_observed=len(runs),
        source="calibrated",
    )


# ---------------------------------------------------------------------------
# Full per-concept driver
# ---------------------------------------------------------------------------


def calibrate_concept(
    concept: str,
    probs_per_patient: dict[str, np.ndarray],
    labels_per_patient: dict[str, np.ndarray] | None = None,
    window_stride_sec: float = WINDOW_STRIDE_SEC_DEFAULT,
    percentiles: tuple[float, float] = TIER_PERCENTILES_DEFAULT,
) -> TierThresholds:
    """
    Calibrate the tier thresholds for one concept across all patients.

    Steps
    -----
    1. If labels are provided, pick activation threshold by best-F1.
       Otherwise fall back to 0.5.
    2. Pool runs across all patients using the chosen threshold.
    3. Calibrate tier cutoffs from the pooled run-duration distribution.

    Parameters
    ----------
    probs_per_patient :
        ``{patient_id: (N_pid,) float ndarray}`` of per-window probabilities.
    labels_per_patient :
        Optional ``{patient_id: (N_pid,) int ndarray}``.  Must align with
        probs by patient.  When present, used to F1-calibrate the threshold.
    """
    if labels_per_patient is not None:
        pooled_probs = np.concatenate(
            [probs_per_patient[p] for p in probs_per_patient if p in labels_per_patient]
        )
        pooled_labels = np.concatenate(
            [labels_per_patient[p] for p in probs_per_patient if p in labels_per_patient]
        )
        if len(pooled_probs) >= 100:
            threshold, _f1 = calibrate_activation_threshold(pooled_probs, pooled_labels)
        else:
            threshold = DEFAULT_ACTIVATION_THRESHOLD
    else:
        threshold = DEFAULT_ACTIVATION_THRESHOLD

    all_runs: list[ActivationRun] = []
    for probs in probs_per_patient.values():
        all_runs.extend(
            find_activation_runs(
                probs,
                threshold=threshold,
                window_stride_sec=window_stride_sec,
            )
        )

    return calibrate_tier_thresholds(all_runs, concept, threshold, percentiles=percentiles)


# ---------------------------------------------------------------------------
# Self-test on synthetic data (run via ``python -m clbench.persistence``)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Sanity check on synthetic activations.  Raises on assertion failure."""
    rng = np.random.default_rng(0)

    # Synthetic patient: one short run (~30 s), one long run (~600 s).
    n_windows = 100
    probs = np.zeros(n_windows)
    probs[5:7] = 0.9  # short — 2 windows × 15 s = 30 s
    probs[20:60] = 0.95  # long — 40 windows × 15 s = 600 s

    runs = find_activation_runs(probs, threshold=0.5, window_stride_sec=15.0)
    assert len(runs) == 2, f"expected 2 runs, got {len(runs)}"
    assert runs[0].n_windows == 2
    assert runs[1].n_windows == 40
    assert abs(runs[0].duration_sec - 30.0) < 1e-6
    assert abs(runs[1].duration_sec - 600.0) < 1e-6

    # Threshold calibration: noisy probs around a true class boundary at 0.5.
    n = 500
    true_labels = rng.integers(0, 2, size=n).astype(np.int32)
    # Positive class probs centred at 0.75, negatives at 0.25, with overlap.
    noisy_probs = np.where(
        true_labels == 1,
        rng.normal(0.75, 0.15, size=n),
        rng.normal(0.25, 0.15, size=n),
    ).clip(0.0, 1.0)
    t, f1 = calibrate_activation_threshold(noisy_probs, true_labels)
    assert 0.3 < t < 0.7, f"expected threshold near 0.5, got {t}"
    assert f1 > 0.7, f"f1 too low ({f1}) on Gaussian-overlap labels"

    # Tier calibration: with 2 runs we should hit the fallback branch.
    th = calibrate_tier_thresholds(runs, "fake_concept", activation_threshold=0.5)
    assert th.source == "fallback", "expected fallback with <5 runs"

    # Tier calibration with many runs: synthesize a clean distribution.
    big_runs = [
        ActivationRun(
            start_window_idx=i,
            end_window_idx=i,
            duration_sec=float(d),
            mean_activation=0.9,
            n_windows=1,
        )
        for i, d in enumerate(rng.uniform(10, 1000, size=200))
    ]
    th = calibrate_tier_thresholds(big_runs, "fake_concept", activation_threshold=0.5)
    assert th.source == "calibrated"
    assert th.transient_max_sec < th.episodic_max_sec

    # Classification sanity
    assert th.classify(5.0) == PersistenceTier.TRANSIENT
    assert th.classify(th.transient_max_sec + 1) == PersistenceTier.EPISODIC
    assert th.classify(th.episodic_max_sec + 1) == PersistenceTier.PERSISTENT

    # Round-trip the config through disk.
    cfg = CalibrationConfig(
        concepts={"fake_concept": th},
        window_stride_sec=15.0,
        tier_percentiles=TIER_PERCENTILES_DEFAULT,
    )
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        cfg.save(Path(f.name))
        loaded = CalibrationConfig.load(Path(f.name))
    assert loaded.concepts["fake_concept"].source == "calibrated"

    print("persistence._self_test: all assertions passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _self_test()
