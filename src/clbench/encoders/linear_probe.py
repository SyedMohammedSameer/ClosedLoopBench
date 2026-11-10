"""
Linear probe evaluation for CLBench encoder go/no-go gate (Phase 2.1).

Purpose
-------
After MAE pre-training, freeze the encoder and train a single linear
classification head on rule-derived concept labels.  If the encoder has
learned physiologically meaningful representations, a simple linear head
should achieve AUROC > 0.70 on well-defined concepts (e.g. tachycardia).

This is the cheapest sanity check: if a frozen linear layer can't separate
tachycardia from non-tachycardia using the encoder's embeddings, the
pre-training is not producing useful representations → abort and re-examine
the encoder before continuing to Phase 2.2 (fusion).

Workflow
--------
1. Derive binary labels from raw signal windows using rule-based detectors
   (see ``RuleLabeler``).
2. Extract frozen encoder embeddings for all labelled windows.
3. Train a logistic regression head on the embeddings.
4. Evaluate on a held-out subset; compute AUROC + AUPRC.
5. Return a ``LinearProbeResult`` with pass/fail gate status.

Rule detectors implemented (Tier-1 subset)
-------------------------------------------
- ``tachycardia``  HR > 100 bpm (ECG or PPG)
- ``bradycardia``  HR < 60 bpm (ECG or PPG)
- ``tachypnea``    RR > 20 breaths/min (RESP)
- ``hypotension``  SBP < 90 mmHg (ABP)
- ``hypertension`` SBP >= 140 mmHg (ABP)

Usage
-----
From Python::

    from clbench.encoders.linear_probe import run_linear_probe, ProbeConfig

    result = run_linear_probe(
        encoder=ecg_encoder,          # frozen SignalEncoder
        processed_dir=Path("data/processed"),
        patient_ids=train_ids,
        modality="ECG",
        concept="tachycardia",
        device=torch.device("mps"),
    )
    print(result)  # LinearProbeResult(auroc=0.82, auprc=0.71, pass=True)
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from clbench.encoders.base import SignalEncoder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    """Hyperparameters for the linear probe training loop."""

    concept: str = "tachycardia"
    modality: str = "ECG"
    min_auroc: float = 0.70
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 128
    val_frac: float = 0.20  # fraction of labelled windows held out for eval
    min_positive_frac: float = 0.05  # skip probe if fewer than this fraction are positive
    seed: int = 42
    max_windows: int | None = 50_000  # random subsample cap; None = use all
    # Probe-time pooling over the post-transformer token sequence.
    # 'mean'         (default) — matches encoder.forward(), back-compatible with CBM
    # 'mean+std'     — concat mean & std; recovers temporal dispersion mean-pool destroys
    # 'mean+std+max' — adds max-pool channel (peak token activation)
    # Use the richer pools only as a probe-time diagnostic; if a richer pool
    # rescues a concept the proper fix is in SignalEncoder.forward, not here.
    pool: str = "mean"


# ---------------------------------------------------------------------------
# Rule-based label derivation
# ---------------------------------------------------------------------------

# Each detector receives a signal array of shape (n_channels, n_samples) and
# the sampling frequency, and returns True if the concept is active in that
# window.  They must be fast (no heavy scipy calls in the hot path).


def _detect_heart_rate(
    signal: np.ndarray,
    fs: float,
    low_bpm: float | None = None,
    high_bpm: float | None = None,
    **_unused: Any,
) -> bool:
    """
    Estimate heart rate from ECG or PPG via autocorrelation peak detection.

    Returns True if the estimated HR falls in (low_bpm, high_bpm).
    Uses the first channel only.
    """
    x = signal[0].astype(np.float64)
    if np.isnan(x).mean() > 0.2:
        return False  # too many NaNs

    # Fill NaN with mean for autocorrelation
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = x - x.mean()
    if x.std() < 1e-6:
        return False

    # Autocorrelation in the physiological HR range [40, 200] bpm
    min_lag = int(fs * 60 / 200)
    max_lag = int(fs * 60 / 40)
    max_lag = min(max_lag, len(x) // 2)
    if min_lag >= max_lag:
        return False

    corr = np.correlate(x, x, mode="full")
    half = corr[len(corr) // 2 :]
    peak_lag = int(np.argmax(half[min_lag:max_lag]) + min_lag)
    if peak_lag <= 0:
        return False

    hr = 60.0 * fs / peak_lag
    if low_bpm is not None and hr <= low_bpm:
        return False
    if high_bpm is not None and hr >= high_bpm:
        return False
    return True


def _detect_resp_rate(
    signal: np.ndarray,
    fs: float,
    low_brpm: float | None = None,
    high_brpm: float | None = None,
    **_unused: Any,
) -> bool:
    """Estimate respiratory rate via autocorrelation. Returns True if in range."""
    x = signal[0].astype(np.float64)
    if np.isnan(x).mean() > 0.2:
        return False
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = x - x.mean()
    if x.std() < 1e-6:
        return False

    # Autocorrelation in the physiological RR range [4, 40] brpm
    min_lag = int(fs * 60 / 40)
    max_lag = int(fs * 60 / 4)
    max_lag = min(max_lag, len(x) // 2)
    if min_lag >= max_lag:
        return False

    corr = np.correlate(x, x, mode="full")
    half = corr[len(corr) // 2 :]
    peak_lag = int(np.argmax(half[min_lag:max_lag]) + min_lag)
    if peak_lag <= 0:
        return False

    rr = 60.0 * fs / peak_lag
    if low_brpm is not None and rr <= low_brpm:
        return False
    if high_brpm is not None and rr >= high_brpm:
        return False
    return True


def _detect_abp_threshold(
    signal: np.ndarray,
    fs: float,
    *,
    mu_raw: np.ndarray | None = None,
    sigma_raw: np.ndarray | None = None,
    low_mmhg: float | None = None,
    high_mmhg: float | None = None,
    sustained_sec: float = 10.0,
    z_threshold: float = 1.0,
) -> bool:
    """
    Detect a sustained ABP threshold crossing.

    Two operating modes
    -------------------
    1. **Absolute-mmHg mode** (preferred) — when ``mu_raw`` and ``sigma_raw``
       are provided by the caller (typically from the parquet columns
       ``signal_mu_raw`` / ``signal_sigma_raw``), the z-scored window is
       de-normalised back to native mmHg via ``mmhg = z * sigma + mu``
       and the rolling sub-window mean is compared against the
       physiological thresholds ``low_mmhg`` / ``high_mmhg`` (e.g. MAP < 65
       for hypotension, MAP > 105 for hypertension).
    2. **Z-score fallback** — when either scale array is ``None`` (legacy
       parquet predating Step 2 of the Phase 2.1 fix-up), the detector
       falls back to within-segment dispersion: ``low_mmhg``-not-None means
       "check whether any sustained sub-window mean is below ``-z_threshold``
       SD"; ``high_mmhg``-not-None checks the symmetric upper deviation.
       Labels under the fallback are not physiologically interpretable as
       SBP < 90 / >= 140; they reflect intra-segment dispersion only.

    Bug history
    -----------
    Previously defaulted to ``sustained_sec=30.0`` (== window length),
    so ``range(0, len(x) - n_sustained, ...)`` was empty and the function
    always returned False (0/297k positives across the train split).
    Now defaults to ``sustained_sec=10.0`` and guards the empty-range
    case explicitly.
    """
    x = signal[0].astype(np.float64)
    if np.isnan(x).mean() > 0.3:
        return False
    x = np.where(np.isnan(x), np.nanmean(x), x)

    if mu_raw is not None and sigma_raw is not None:
        # De-normalise channel 0 (ABP is single-channel by construction).
        sigma0 = float(sigma_raw[0])
        mu0 = float(mu_raw[0])
        x = x * sigma0 + mu0
        low_thr = low_mmhg
        high_thr = high_mmhg
    else:
        # Treat ``low_mmhg`` / ``high_mmhg`` purely as gating flags; the
        # real threshold is ``±z_threshold`` on the z-scored signal.
        low_thr = -z_threshold if low_mmhg is not None else None
        high_thr = z_threshold if high_mmhg is not None else None

    n_sustained = max(1, int(sustained_sec * fs))
    n_sustained = min(n_sustained, len(x))

    if n_sustained >= len(x):
        win_means = np.array([x.mean()])
    else:
        step = max(1, n_sustained // 2)
        win_means = np.array(
            [x[i : i + n_sustained].mean() for i in range(0, len(x) - n_sustained + 1, step)]
        )

    if low_thr is not None and not np.any(win_means < low_thr):
        return False
    if high_thr is not None and not np.any(win_means > high_thr):
        return False
    return True


# Map concept name → (modality, detector callable)
#
# _detect_heart_rate semantics:
#   low_bpm  → return False if hr <= low_bpm  (i.e. True only when hr > low_bpm)
#   high_bpm → return False if hr >= high_bpm (i.e. True only when hr < high_bpm)
#
# So tachycardia (HR > 100) uses low_bpm=100, bradycardia (HR < 60) uses high_bpm=60.
# Same logic applies to resp rate detectors.
#
# ABP thresholds are absolute MAP in mmHg (since window mean ≈ MAP on a 30 s
# arterial waveform).  They take effect only when the parquet carries the
# ``signal_mu_raw`` / ``signal_sigma_raw`` columns added in Step 2 of the
# Phase 2.1 fix-up; on legacy parquets the detector falls back to a z-score
# proxy (see :func:`_detect_abp_threshold`).
#
# Each detector accepts ``mu_raw`` / ``sigma_raw`` kwargs so ``label_windows``
# can forward the per-window scale uniformly; rate detectors ignore them.
_CONCEPT_DETECTORS: dict[str, tuple[str, Callable[..., bool]]] = {
    "tachycardia": ("ECG", lambda s, f, **kw: _detect_heart_rate(s, f, low_bpm=100, **kw)),
    "bradycardia": ("ECG", lambda s, f, **kw: _detect_heart_rate(s, f, high_bpm=60, **kw)),
    "tachycardia_ppg": ("PPG", lambda s, f, **kw: _detect_heart_rate(s, f, low_bpm=100, **kw)),
    "tachypnea": ("RESP", lambda s, f, **kw: _detect_resp_rate(s, f, low_brpm=20, **kw)),
    "bradypnea": ("RESP", lambda s, f, **kw: _detect_resp_rate(s, f, high_brpm=12, **kw)),
    # MAP < 65 mmHg ≈ "SBP < 90 mmHg" hypotension proxy (per Sepsis-3 MAP target).
    "hypotension": ("ABP", lambda s, f, **kw: _detect_abp_threshold(s, f, low_mmhg=65.0, **kw)),
    # MAP > 105 mmHg ≈ "SBP >= 140 mmHg" hypertension proxy.
    "hypertension": ("ABP", lambda s, f, **kw: _detect_abp_threshold(s, f, high_mmhg=105.0, **kw)),
}


def label_windows(
    df: pd.DataFrame,
    concept: str,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Apply the rule-based detector for ``concept`` to all windows in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed parquet for one patient, filtered to the correct modality.
    concept : str
        Key in ``_CONCEPT_DETECTORS``.

    Returns
    -------
    signals : list of (n_channels, n_samples) float32 arrays
    labels  : list of 0/1 ints
    """
    if concept not in _CONCEPT_DETECTORS:
        raise ValueError(
            f"Unknown concept '{concept}'. " f"Available: {list(_CONCEPT_DETECTORS.keys())}"
        )
    _, detector = _CONCEPT_DETECTORS[concept]

    # Detect whether the parquet carries the Step-2 raw-scale columns.  If
    # absent (legacy parquet), ABP detectors fall back to z-score proxy.
    has_scale = "signal_mu_raw" in df.columns and "signal_sigma_raw" in df.columns
    if not has_scale:
        log.warning(
            "Parquet missing signal_mu_raw / signal_sigma_raw — ABP detectors "
            "will fall back to z-score proxy.  Re-run 02_preprocess.py "
            "(--no-skip-existing) to populate the scale columns."
        )

    signals: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        shape: list[int] = json.loads(row["signal_shape"])
        arr = np.frombuffer(row["signal_bytes"], dtype="float32").reshape(shape)
        sig = arr.T.copy().astype(np.float64)  # (n_channels, n_samples)
        fs = float(row["fs"])

        mu_raw: np.ndarray | None = None
        sigma_raw: np.ndarray | None = None
        if has_scale:
            mu_json = row["signal_mu_raw"]
            sigma_json = row["signal_sigma_raw"]
            # NB: parquet stores ``None`` as NaN for object columns, so guard
            # both ``None`` and ``pd.isna`` here.
            if mu_json is not None and not (isinstance(mu_json, float) and np.isnan(mu_json)):
                mu_raw = np.asarray(json.loads(mu_json), dtype=np.float64)
            if sigma_json is not None and not (
                isinstance(sigma_json, float) and np.isnan(sigma_json)
            ):
                sigma_raw = np.asarray(json.loads(sigma_json), dtype=np.float64)

        label = int(detector(sig, fs, mu_raw=mu_raw, sigma_raw=sigma_raw))
        signals.append(sig.astype(np.float32))
        labels.append(label)

    return signals, labels


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()  # type: ignore[misc]
def extract_embeddings(
    encoder: SignalEncoder,
    signals: list[np.ndarray],
    device: torch.device,
    batch_size: int = 128,
    pool: str = "mean",
) -> np.ndarray:
    """
    Run the frozen encoder over a list of signal arrays.

    Parameters
    ----------
    encoder : SignalEncoder
        Must be in eval mode and on ``device``.
    signals : list of (n_channels, n_samples) float32 arrays
        All arrays must have the same shape.
    device : torch.device
    batch_size : int
    pool : {'mean', 'mean+std', 'mean+std+max'}
        Pooling applied to the post-transformer token sequence.
        'mean' calls ``encoder.forward()`` exactly (the path the CBM uses).
        The other modes are diagnostic — they reach into
        ``extract_tokens`` and ``transformer`` directly to recover
        temporal-dispersion features that mean-pooling erases (relevant
        for rate-like concepts on ECG).

    Returns
    -------
    np.ndarray of shape (N, d_out) where d_out depends on ``pool``:
      mean         -> d_model
      mean+std     -> 2 * d_model
      mean+std+max -> 3 * d_model
    """
    encoder.eval()
    embs: list[np.ndarray] = []

    for start in range(0, len(signals), batch_size):
        batch_np = np.stack(signals[start : start + batch_size])  # (B, C, T)
        x: Tensor = torch.from_numpy(batch_np).to(device)
        # Sanitize NaN/Inf in the INPUT.  Edge windows (~0.5-0.8% of the probe
        # set) carry NaNs; the encoder propagates them to NaN embeddings, which
        # then poison the entire linear-probe fit (a single NaN -> NaN loss ->
        # NaN weights -> all logits NaN -> AUROC pinned at a fixed ~0.5
        # regardless of the encoder).  ``MaskedSignalEncoder.forward`` already
        # does this at train time; the probe path must match it.
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        if pool == "mean":
            emb: Tensor = encoder(x)  # (B, d_model) — matches CBM path
        else:
            tokens = encoder.extract_tokens(x)  # (B, T'', d_model)
            tokens = encoder.transformer(tokens)  # (B, T'', d_model)
            # Exclude the CLS slot (position 0) when present so the diagnostic
            # pool reflects content tokens only.
            if getattr(encoder, "use_cls_token", False):
                tokens = tokens[:, 1:]
            if pool == "mean+std":
                m = tokens.mean(dim=1)
                s = tokens.std(dim=1)
                emb = torch.cat([m, s], dim=-1)  # (B, 2 d_model)
            elif pool == "mean+std+max":
                m = tokens.mean(dim=1)
                s = tokens.std(dim=1)
                mx, _ = tokens.max(dim=1)
                emb = torch.cat([m, s, mx], dim=-1)  # (B, 3 d_model)
            else:
                raise ValueError(
                    f"Unknown pool='{pool}'. " "Expected 'mean' | 'mean+std' | 'mean+std+max'."
                )
        embs.append(emb.cpu().numpy())

    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Linear head training
# ---------------------------------------------------------------------------


class _LinearHead(nn.Module):  # type: ignore[misc]
    """Single linear layer: d_model → 1 (binary classification)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.fc(x).squeeze(-1)
        return result


def _train_linear_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg: ProbeConfig,
    device: torch.device,
) -> tuple[_LinearHead, np.ndarray, np.ndarray]:
    """
    Train a logistic regression head on ``embeddings``.

    Returns
    -------
    head : trained _LinearHead (in eval mode)
    val_logits : (N_val,) float32 array
    val_labels : (N_val,) int array
    """
    rng = np.random.default_rng(cfg.seed)
    n_total = len(labels)
    val_n = max(1, int(n_total * cfg.val_frac))
    idx = rng.permutation(n_total)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    x_train = torch.from_numpy(embeddings[train_idx]).float()
    y_train = torch.from_numpy(labels[train_idx]).float()
    x_val = torch.from_numpy(embeddings[val_idx]).float()
    y_val = labels[val_idx]

    ds = TensorDataset(x_train, y_train)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    d_model = embeddings.shape[1]
    head = _LinearHead(d_model).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _epoch in range(cfg.epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = head(xb)
            loss_fn(logits, yb).backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        val_logits = head(x_val.to(device)).cpu().numpy()

    return head, val_logits, y_val


# ---------------------------------------------------------------------------
# AUROC / AUPRC (no sklearn dependency)
# ---------------------------------------------------------------------------


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUROC via rank-based U-statistic.

    O(n log n) time, O(n) memory — safe for millions of windows.
    The naive outer-product approach allocates O(n_pos × n_neg) memory
    which exceeds available RAM for large datasets.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Sort scores ASCENDING: rank 1 = lowest score, rank N = highest score.
    # With ascending sort, a good classifier gives positives HIGH ranks →
    # large u_stat → AUROC near 1.  (Descending sort inverts the result.)
    order = np.argsort(scores, kind="stable")
    sorted_labels = labels[order]
    # 1-based ranks
    ranks = np.arange(1, len(labels) + 1, dtype=np.float64)
    # U-statistic: sum of ranks of positives minus triangular number
    u_stat = float(ranks[sorted_labels == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    return u_stat / (float(n_pos) * float(n_neg))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUPRC via the trapezoidal rule over sorted thresholds."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / (tp + fp)
    recall = tp / labels.sum()
    # Prepend (recall=0, precision=1) sentinel
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LinearProbeResult:
    concept: str
    modality: str
    n_windows: int
    n_positive: int
    positive_frac: float
    auroc: float
    auprc: float
    min_auroc_gate: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS ✅" if self.passed else "FAIL ❌"
        return (
            f"LinearProbeResult [{status}]\n"
            f"  concept      : {self.concept} ({self.modality})\n"
            f"  windows      : {self.n_windows}  positive={self.n_positive}"
            f" ({self.positive_frac:.1%})\n"
            f"  AUROC        : {self.auroc:.3f}  (gate >= {self.min_auroc_gate:.2f})\n"
            f"  AUPRC        : {self.auprc:.3f}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_linear_probe(
    encoder: SignalEncoder,
    processed_dir: Path,
    patient_ids: list[str],
    modality: str,
    concept: str,
    device: torch.device,
    cfg: ProbeConfig | None = None,
) -> LinearProbeResult:
    """
    Full linear probe pipeline: label → embed → train head → evaluate.

    Parameters
    ----------
    encoder : SignalEncoder
        Frozen (pre-trained) encoder.  Will be set to eval() internally.
    processed_dir : Path
        Directory of ``{patient_id}.parquet`` files.
    patient_ids : list[str]
        Patients to use (typically the training split).
    modality : str
        Signal modality matching ``concept``.
    concept : str
        One of the keys in ``_CONCEPT_DETECTORS``.
    device : torch.device
    cfg : ProbeConfig | None
        If None, defaults are used.

    Returns
    -------
    LinearProbeResult
    """
    if cfg is None:
        cfg = ProbeConfig(concept=concept, modality=modality)

    encoder.eval()

    # If the encoder declares in_channels, filter windows to that channel count.
    # Without this, mixed-channel patients (e.g. 1-lead ECG among 3-lead) cause
    # np.stack to raise ValueError in extract_embeddings.
    n_channels_filter: int | None = getattr(encoder, "in_channels", None)

    # 1. Load parquet rows for all patients (fast — no signal processing yet).
    #    We collect into a combined DataFrame so we can subsample BEFORE running
    #    the expensive autocorrelation detectors.
    dfs: list[pd.DataFrame] = []
    for pid in patient_ids:
        fpath = processed_dir / f"{pid}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        df = df[(df["modality"] == modality) & df["quality_ok"].astype(bool)]
        if n_channels_filter is not None and "n_channels" in df.columns:
            df = df[df["n_channels"] == n_channels_filter]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        log.warning("No usable %s windows for concept '%s'", modality, concept)
        return LinearProbeResult(
            concept=concept,
            modality=modality,
            n_windows=0,
            n_positive=0,
            positive_frac=0.0,
            auroc=float("nan"),
            auprc=float("nan"),
            min_auroc_gate=cfg.min_auroc,
            passed=False,
        )

    combined = pd.concat(dfs, ignore_index=True)
    n_windows_total = len(combined)

    # 2. Subsample rows BEFORE labelling — avoids running autocorrelation on
    #    millions of windows.  50K rows takes ~15 s; 1.3M takes ~7 min.
    if cfg.max_windows is not None and n_windows_total > cfg.max_windows:
        combined = combined.sample(n=cfg.max_windows, random_state=cfg.seed)
        log.info(
            "Pre-subsampled %d → %d rows for [%s / %s] (detectors will run on subset only)",
            n_windows_total,
            cfg.max_windows,
            concept,
            modality,
        )

    # 3. Run detectors only on the (subsampled) rows
    all_signals, all_labels_list = label_windows(combined, concept)

    n_windows = n_windows_total  # report full dataset size for the paper
    n_positive = sum(all_labels_list)
    positive_frac = n_positive / len(all_labels_list) if all_labels_list else 0.0

    log.info(
        "Linear probe [%s / %s]: %d total windows (labelled %d), %d positive (%.1f%%)",
        concept,
        modality,
        n_windows_total,
        len(all_labels_list),
        n_positive,
        positive_frac * 100,
    )

    if len(all_labels_list) < 10:
        log.warning("Too few windows (%d) for probe — returning NaN result", len(all_labels_list))
        return LinearProbeResult(
            concept=concept,
            modality=modality,
            n_windows=n_windows,
            n_positive=n_positive,
            positive_frac=positive_frac,
            auroc=float("nan"),
            auprc=float("nan"),
            min_auroc_gate=cfg.min_auroc,
            passed=False,
        )

    if positive_frac < cfg.min_positive_frac:
        log.warning(
            "Positive fraction %.1f%% < %.0f%% minimum — concept may be too rare",
            positive_frac * 100,
            cfg.min_positive_frac * 100,
        )

    # 4. Extract embeddings (encoder is frozen).  ``cfg.pool`` defaults to
    #    'mean' (== encoder.forward); richer pools are opt-in diagnostics.
    embeddings = extract_embeddings(encoder, all_signals, device, cfg.batch_size, pool=cfg.pool)
    labels = np.array(all_labels_list, dtype=np.int32)

    # 5. Train linear head + evaluate
    _, val_logits, val_labels = _train_linear_head(embeddings, labels, cfg, device)

    # 6. Compute metrics
    auroc = _auroc(val_logits, val_labels)
    auprc = _auprc(val_logits, val_labels)
    passed = (not np.isnan(auroc)) and (auroc >= cfg.min_auroc)

    result = LinearProbeResult(
        concept=concept,
        modality=modality,
        n_windows=n_windows,
        n_positive=n_positive,
        positive_frac=positive_frac,
        auroc=float(auroc),
        auprc=float(auprc),
        min_auroc_gate=cfg.min_auroc,
        passed=passed,
    )
    log.info("%s", result)
    return result
