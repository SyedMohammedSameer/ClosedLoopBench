"""
Phase 1.4 — Per-modality signal preprocessing for CLBench.

Pipeline for each patient
--------------------------
1. Load all waveform segments via wfdb (skipping gap / layout segments).
2. Map raw channel names → canonical modalities (ECG / PPG / ABP / RESP).
3. Apply per-modality bandpass filter (4th-order Butterworth, SOS):
     ECG  : 0.5 – 40.0 Hz
     PPG  : 0.5 –  8.0 Hz
     RESP : 0.1 –  1.0 Hz
     ABP  : no bandpass — DC component encodes mean arterial pressure.
4. Normalise:
     ECG / PPG / RESP : per-segment z-score (each channel independently).
     ABP              : population-scale ``(x - 85) / 20`` mmHg.  Per-segment
                        z-score destroys the DC component that *is* MAP, so
                        ABP uses a fixed population shift+scale that
                        preserves absolute pressure level across patients.
5. Slice into fixed-length windows (default 30 s, 15 s stride = 50% overlap).
6. Flag windows with > 20% NaN as ``quality_ok=False``.
7. Serialise to per-patient parquet under ``data/processed/``.

Output schema (one parquet row per window × modality)
------------------------------------------------------
patient_id        str
segment_name      str        wfdb segment filename stem
window_idx        int        zero-based, within this patient
modality          str        ECG | PPG | ABP | RESP
channel_names     str        JSON list, e.g. ``'["II", "V", "aVR"]'``
fs                float      sampling frequency (Hz)
window_start_sec  float      relative to segment start
window_end_sec    float
n_samples         int        samples per window
n_channels        int        >1 for multi-lead ECG, 1 for others
signal_bytes      bytes      float32 C-order tobytes()
signal_shape      str        JSON list, e.g. ``'[1874, 3]'``
signal_dtype      str        ``'float32'``
quality_ok        bool       False when > MAX_NAN_FRAC of raw window is NaN
signal_mu_raw     str|None   JSON list of per-channel raw means (pre-zscore)
signal_sigma_raw  str|None   JSON list of per-channel raw stds  (pre-zscore)

The mu/sigma fields let downstream consumers recover original-unit values
(notably absolute mmHg for ABP) via
``raw = signal_bytes_array * signal_sigma_raw + signal_mu_raw``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR_DEFAULT = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "raw"
    / "physionet.org"
    / "files"
    / "mimic4wdb"
    / "0.1.0"
)
PROCESSED_DIR_DEFAULT = Path(__file__).parent.parent.parent.parent / "data" / "processed"
MANIFEST_PATH_DEFAULT = Path(__file__).parent.parent.parent.parent / "data" / "manifest.csv"

WINDOW_SEC_DEFAULT: float = 30.0
STRIDE_SEC_DEFAULT: float = 15.0
MAX_NAN_FRAC: float = 0.20  # windows with more NaN are flagged quality_ok=False
ECG_MAX_LEADS: int = 3  # keep at most this many ECG leads (II, V, aVR priority)
FILTER_ORDER: int = 4  # Butterworth order

# ABP normalization is population-scale rather than per-segment z-score.
# Per-segment z-score erases the DC component (= mean arterial pressure),
# which is exactly the signal hypotension / hypertension detectors need.
# These constants come from typical adult ICU ABP distributions
# (MAP centred ~85 mmHg, SD ~20 mmHg).  With this scheme, a chronically
# hypotensive patient (raw MAP ~60) has window-mean ≈ -1.25 SD and a
# chronically hypertensive one (raw MAP ~110) has ≈ +1.25 SD — both
# now visible to the encoder.
ABP_POPULATION_MU_MMHG: float = 85.0
ABP_POPULATION_SIGMA_MMHG: float = 20.0

# ---------------------------------------------------------------------------
# Channel name → Modality mapping
# ---------------------------------------------------------------------------

# Modality string constants (mirrors clbench.ontology.concepts.Modality values)
_ECG = "ECG"
_PPG = "PPG"
_ABP = "ABP"
_RESP = "RESP"

# Only SYSTEMIC arterial pressure channels map to ABP.
# PAP (pulmonary arterial) and CVP (central venous) are intentionally excluded.
CHANNEL_MAP: dict[str, str] = {
    # ── ECG leads ───────────────────────────────────────────────────────────
    "I": _ECG,
    "II": _ECG,
    "III": _ECG,
    "V": _ECG,
    "aVR": _ECG,
    "aVL": _ECG,
    "aVF": _ECG,
    "MCL": _ECG,
    "MCL1": _ECG,
    # MIMIC variant label strings
    "ECG": _ECG,
    "ECG #0": _ECG,
    "ECG #0+": _ECG,
    "ECG #0++": _ECG,
    "ECG #0+++": _ECG,
    # ── PPG ─────────────────────────────────────────────────────────────────
    "Pleth": _PPG,
    "PPG": _PPG,
    "SpO2": _PPG,
    # ── Systemic arterial blood pressure ────────────────────────────────────
    "ABP": _ABP,
    "ART": _ABP,
    "Art": _ABP,
    "Art.": _ABP,
    "Arterial": _ABP,
    # ── Respiration ─────────────────────────────────────────────────────────
    "Resp": _RESP,
    "RESP": _RESP,
    "CO2": _RESP,  # end-tidal CO2 correlates with respiratory cycle
}

# ---------------------------------------------------------------------------
# Per-modality bandpass spec: (low_hz, high_hz) or None (no filter)
# ---------------------------------------------------------------------------

BANDPASS_SPECS: dict[str, tuple[float, float] | None] = {
    _ECG: (0.5, 40.0),
    _PPG: (0.5, 8.0),
    _RESP: (0.1, 1.0),
    _ABP: None,  # preserve DC (mean arterial pressure)
}

# Preferred ECG lead ordering — channels earlier in this list have priority.
_ECG_LEAD_PRIORITY: list[str] = ["II", "V", "aVR", "I", "III", "aVL", "aVF", "MCL", "MCL1"]


# ---------------------------------------------------------------------------
# WindowRecord
# ---------------------------------------------------------------------------


@dataclass
class WindowRecord:
    """One preprocessed signal window for a single patient × modality."""

    patient_id: str
    segment_name: str
    window_idx: int
    modality: str  # ECG | PPG | ABP | RESP
    channel_names: list[str]  # raw channel labels, e.g. ["II", "V", "aVR"]
    fs: float
    window_start_sec: float
    window_end_sec: float
    signal: np.ndarray  # shape (n_samples,) or (n_samples, n_channels), float32
    quality_ok: bool  # False when > MAX_NAN_FRAC NaN in raw window
    # Per-channel raw-unit scale recorded BEFORE z-score normalisation, so
    # downstream code can recover original-unit values (e.g. mmHg for ABP)
    # via ``raw = normalised * signal_sigma_raw + signal_mu_raw``.
    # Shape (n_channels,) each; ``None`` only when loaded from a legacy
    # parquet that predates this field.
    signal_mu_raw: np.ndarray | None = None
    signal_sigma_raw: np.ndarray | None = None

    # Derived shape properties ---------------------------------------------------

    @property
    def _signal_2d(self) -> np.ndarray:
        """Always return signal as 2-D (n_samples, n_channels)."""
        if self.signal.ndim == 1:
            return self.signal[:, np.newaxis]
        return self.signal

    @property
    def n_samples(self) -> int:
        return int(self._signal_2d.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self._signal_2d.shape[1])

    # Serialisation --------------------------------------------------------------

    def to_parquet_row(self) -> dict[str, Any]:
        """Return a flat dict suitable for a single DataFrame row."""
        sig2d = self._signal_2d.astype(np.float32, copy=False)
        mu_json = (
            json.dumps([float(v) for v in self.signal_mu_raw])
            if self.signal_mu_raw is not None
            else None
        )
        sigma_json = (
            json.dumps([float(v) for v in self.signal_sigma_raw])
            if self.signal_sigma_raw is not None
            else None
        )
        return {
            "patient_id": self.patient_id,
            "segment_name": self.segment_name,
            "window_idx": self.window_idx,
            "modality": self.modality,
            "channel_names": json.dumps(self.channel_names),
            "fs": float(self.fs),
            "window_start_sec": float(self.window_start_sec),
            "window_end_sec": float(self.window_end_sec),
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "signal_bytes": sig2d.tobytes(),
            "signal_shape": json.dumps(list(sig2d.shape)),
            "signal_dtype": "float32",
            "quality_ok": bool(self.quality_ok),
            # JSON-encoded per-channel raw scale (None on legacy parquets).
            "signal_mu_raw": mu_json,
            "signal_sigma_raw": sigma_json,
        }

    @staticmethod
    def signal_from_row(row: pd.Series[Any]) -> np.ndarray:
        """Reconstruct signal ndarray from a parquet row."""
        shape: list[int] = json.loads(row["signal_shape"])
        return np.frombuffer(row["signal_bytes"], dtype=np.float32).reshape(shape)


# ---------------------------------------------------------------------------
# Low-level DSP helpers
# ---------------------------------------------------------------------------


def _apply_bandpass(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to each channel.

    NaN values are interpolated before filtering and restored afterwards.
    Returns float32 array of the same shape as *signal*.

    Parameters
    ----------
    signal:
        2-D array of shape ``(n_samples, n_channels)``, dtype float64 or float32.
    fs:
        Sampling frequency in Hz.
    low_hz, high_hz:
        Bandpass corner frequencies in Hz.
    order:
        Butterworth filter order.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = fs / 2.0
    low_norm = low_hz / nyq
    high_norm = high_hz / nyq

    # Guard: frequencies must be strictly inside (0, 1)
    if low_norm <= 0.0 or high_norm >= 1.0 or low_norm >= high_norm:
        log.debug(
            "Bandpass [%.2f, %.2f] Hz invalid for fs=%.2f; skipping filter",
            low_hz,
            high_hz,
            fs,
        )
        return signal.astype(np.float32)

    sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    out = signal.copy().astype(np.float64)

    for ch in range(out.shape[1]):
        col = out[:, ch]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            continue
        # Linearly interpolate NaN spans so the filter has a continuous signal
        col_filled = col.copy()
        if nan_mask.any():
            valid_idx = np.where(~nan_mask)[0]
            col_filled[nan_mask] = np.interp(
                np.where(nan_mask)[0],
                valid_idx,
                col[valid_idx],
            )
        col_filtered = sosfiltfilt(sos, col_filled)
        # Restore NaN at original positions
        col_filtered[nan_mask] = np.nan
        out[:, ch] = col_filtered

    return out.astype(np.float32)


def _zscore_normalize(
    signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise each channel independently, ignoring NaN.

    If a channel has zero variance, only mean-centres it.

    Returns
    -------
    out : float32 array, same shape as ``signal`` — the normalised signal.
    mu  : float64 array of shape ``(n_channels,)`` — per-channel mean of the
          raw (pre-zscore) signal in its native unit.
    sigma : float64 array of shape ``(n_channels,)`` — per-channel std of the
            raw signal.  Zero-variance channels carry ``sigma = 1.0`` so that
            ``raw_value = normalised_value * sigma + mu`` still recovers the
            centred signal.

    The (mu, sigma) arrays are persisted on each window so downstream code
    can recover the original-unit values — notably absolute mmHg for ABP,
    which is required by the hypotension / hypertension concept detectors.
    Per-segment scope is intentional: it matches the scope of the z-score
    itself, which is applied once per segment-modality pair upstream.
    """
    out = signal.astype(np.float64)
    n_ch = out.shape[1]
    mus = np.zeros(n_ch, dtype=np.float64)
    sigmas = np.ones(n_ch, dtype=np.float64)
    for ch in range(n_ch):
        col = out[:, ch]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        mu = float(valid.mean())
        sigma = float(valid.std())
        mus[ch] = mu
        if sigma > 1e-8:
            sigmas[ch] = sigma
            out[:, ch] = (col - mu) / sigma
        else:
            sigmas[ch] = 1.0  # keep raw = z*1 + mu invertible
            out[:, ch] = col - mu
    return out.astype(np.float32), mus, sigmas


def _make_windows(
    signal: np.ndarray,
    fs: float,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
    max_nan_frac: float = MAX_NAN_FRAC,
) -> list[tuple[np.ndarray, float, float, bool]]:
    """
    Slice a 2-D signal ``(n_total_samples, n_channels)`` into fixed-length windows.

    Returns
    -------
    list of ``(window_array, start_sec, end_sec, quality_ok)`` tuples.
    An empty list is returned when the signal is shorter than one window.
    """
    window_samples = int(round(window_sec * fs))
    stride_samples = int(round(stride_sec * fs))
    n_total = signal.shape[0]

    if n_total < window_samples or window_samples == 0:
        return []

    windows: list[tuple[np.ndarray, float, float, bool]] = []
    start = 0
    while start + window_samples <= n_total:
        end = start + window_samples
        window = signal[start:end].copy()
        quality_ok = float(np.isnan(window).mean()) <= max_nan_frac
        windows.append((window, start / fs, end / fs, quality_ok))
        start += stride_samples

    return windows


# ---------------------------------------------------------------------------
# wfdb loading helpers
# ---------------------------------------------------------------------------


def _iter_segments(record_path: str, raw_dir: Path) -> Iterator[tuple[str, Any]]:
    """
    Yield ``(segment_name, wfdb.Record)`` for every non-gap segment.

    Handles both single-segment (``wfdb.Record``) and multi-segment
    (``wfdb.MultiRecord``) layout headers.  Gap / layout segments
    (those with ``seg_len == 0`` or name == ``'~'``) are skipped.
    """
    import wfdb

    header_path = str(raw_dir / record_path)
    seg_dir = str(raw_dir / Path(record_path).parent)

    try:
        hdr = wfdb.rdheader(header_path, rd_segments=False)
    except Exception as exc:
        log.warning("Cannot read header %s: %s", header_path, exc)
        return

    seg_names: list[str] | None = getattr(hdr, "seg_name", None)
    seg_lens: list[int] | None = getattr(hdr, "seg_len", None)

    if seg_names:
        # Multi-segment layout record
        for idx, seg_name in enumerate(seg_names):
            if not seg_name or seg_name == "~":
                continue
            # Skip zero-length segments (layout / gap markers)
            if seg_lens is not None and idx < len(seg_lens) and seg_lens[idx] == 0:
                continue
            seg_path = f"{seg_dir}/{seg_name}"
            try:
                rec = wfdb.rdrecord(seg_path)
                yield seg_name, rec
            except Exception as exc:
                log.debug("Skipping segment %s: %s", seg_name, exc)
    else:
        # Single-segment record
        try:
            rec = wfdb.rdrecord(header_path)
            yield getattr(hdr, "record_name", Path(record_path).stem), rec
        except Exception as exc:
            log.warning("Cannot read record %s: %s", header_path, exc)


# ---------------------------------------------------------------------------
# Per-patient preprocessing
# ---------------------------------------------------------------------------


def _sort_ecg_channels(channel_indices: list[int], channel_names: list[str]) -> list[int]:
    """Return ECG channel indices sorted by lead priority, then limited to ECG_MAX_LEADS."""
    priority = {name: i for i, name in enumerate(_ECG_LEAD_PRIORITY)}
    sorted_indices = sorted(
        channel_indices,
        key=lambda i: priority.get(channel_names[i], len(_ECG_LEAD_PRIORITY)),
    )
    return sorted_indices[:ECG_MAX_LEADS]


def windows_from_signal(
    raw_sig: np.ndarray,
    fs: float,
    modality: str,
    channel_names: list[str],
    patient_id: str,
    segment_name: str,
    start_idx: int = 0,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
) -> list[WindowRecord]:
    """
    Filter, normalise and window one (segment, modality) signal.

    Shared by MIMIC-IV-WDB (:func:`preprocess_patient`) and VitalDB
    (:mod:`clbench.data.vitaldb`) so both datasets produce identical windows.

    Parameters
    ----------
    raw_sig:
        Raw-unit signal of shape ``(n_samples, n_channels)`` (1-D is treated as one channel).
    fs:
        Sampling frequency in Hz.
    modality:
        ``ECG`` | ``PPG`` | ``ABP`` | ``RESP``.
    channel_names:
        Channel labels in column order.
    patient_id, segment_name:
        Identifiers stamped on every window.
    start_idx:
        ``window_idx`` of the first window, so indices continue across segments.
    window_sec, stride_sec:
        Window length and hop in seconds.
    """
    raw_sig = np.asarray(raw_sig, dtype=np.float64)
    if raw_sig.ndim == 1:
        raw_sig = raw_sig[:, np.newaxis]

    # ── Bandpass filter ────────────────────────────────────────────────────
    spec = BANDPASS_SPECS[modality]
    if spec is not None and fs > 2.0 * spec[1]:
        filtered = _apply_bandpass(raw_sig, fs, spec[0], spec[1])
    else:
        filtered = raw_sig.astype(np.float32)

    # ── Normalise ──────────────────────────────────────────────────────────
    # Per-segment z-score for ECG/PPG/RESP (preserves relative shape,
    # erases DC).  Population-scale shift+scale for ABP (preserves
    # absolute MAP so hypotension/hypertension concepts are learnable).
    if modality == _ABP:
        filt64 = filtered.astype(np.float64)
        normalised = ((filt64 - ABP_POPULATION_MU_MMHG) / ABP_POPULATION_SIGMA_MMHG).astype(
            np.float32
        )
        n_ch = normalised.shape[1]
        mu_raw = np.full(n_ch, ABP_POPULATION_MU_MMHG, dtype=np.float64)
        sigma_raw = np.full(n_ch, ABP_POPULATION_SIGMA_MMHG, dtype=np.float64)
    else:
        normalised, mu_raw, sigma_raw = _zscore_normalize(filtered)
    mu_raw_f32 = mu_raw.astype(np.float32, copy=False)
    sigma_raw_f32 = sigma_raw.astype(np.float32, copy=False)

    # ── Window ─────────────────────────────────────────────────────────────
    raw_windows = _make_windows(raw_sig, fs, window_sec, stride_sec)
    proc_windows = _make_windows(normalised, fs, window_sec, stride_sec)

    windows: list[WindowRecord] = []
    for (_, start_sec, end_sec, quality_ok), (proc_arr, _, _, _) in zip(raw_windows, proc_windows):
        windows.append(
            WindowRecord(
                patient_id=patient_id,
                segment_name=segment_name,
                window_idx=start_idx + len(windows),
                modality=modality,
                channel_names=list(channel_names),
                fs=fs,
                window_start_sec=start_sec,
                window_end_sec=end_sec,
                signal=proc_arr,
                quality_ok=quality_ok,
                # Per-segment scale shared by every window from this
                # (segment, modality) — matches the scope of the
                # z-score that produced ``proc_arr``.
                signal_mu_raw=mu_raw_f32,
                signal_sigma_raw=sigma_raw_f32,
            )
        )
    return windows


def preprocess_patient(
    patient_id: str,
    record_path: str,
    raw_dir: Path,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
    max_segments: int | None = None,
) -> list[WindowRecord]:
    """
    Preprocess all waveform segments for one patient.

    Returns a (possibly empty) list of :class:`WindowRecord` objects.
    One record is produced per (window, modality) combination.

    Parameters
    ----------
    patient_id:
        Unique patient identifier from the manifest.
    record_path:
        Path relative to *raw_dir* as stored in ``manifest.csv``,
        e.g. ``"waves/p143/p14328567/80030009/80030009"``.
    raw_dir:
        Absolute path to the raw waveform directory.
    window_sec:
        Length of each window in seconds (default 30).
    stride_sec:
        Hop size in seconds (default 15 — 50% overlap).
    max_segments:
        If set, process at most this many segments per patient (useful for
        smoke-tests and memory-constrained environments).
    """
    all_windows: list[WindowRecord] = []
    global_idx = 0
    n_segs_processed = 0

    for seg_name, rec in _iter_segments(record_path, raw_dir):
        if max_segments is not None and n_segs_processed >= max_segments:
            break
        p_signal: np.ndarray | None = rec.p_signal
        if p_signal is None or p_signal.shape[0] == 0:
            continue

        fs: float = float(rec.fs)
        sig_names: list[str] = list(rec.sig_name)

        # ── Group channel indices by modality ──────────────────────────────
        mod_ch_indices: dict[str, list[int]] = defaultdict(list)
        for ch_idx, ch_name in enumerate(sig_names):
            modality = CHANNEL_MAP.get(ch_name)
            if modality is not None:
                mod_ch_indices[modality].append(ch_idx)

        n_segs_processed += 1
        if not mod_ch_indices:
            log.debug("Segment %s has no recognised channels; skipping", seg_name)
            continue

        # ── Process each modality ──────────────────────────────────────────
        for modality, ch_indices in mod_ch_indices.items():
            # Sort ECG leads by priority and cap at ECG_MAX_LEADS
            if modality == _ECG:
                ch_indices = _sort_ecg_channels(ch_indices, sig_names)

            ch_names = [sig_names[i] for i in ch_indices]
            raw_sig = p_signal[:, ch_indices].astype(np.float64)  # (n_samp, n_ch)
            seg_windows = windows_from_signal(
                raw_sig,
                fs,
                modality,
                ch_names,
                patient_id,
                seg_name,
                start_idx=global_idx,
                window_sec=window_sec,
                stride_sec=stride_sec,
            )
            all_windows.extend(seg_windows)
            global_idx += len(seg_windows)

    log.info(
        "Patient %s: %d windows across %d modalities",
        patient_id,
        len(all_windows),
        len({w.modality for w in all_windows}),
    )
    return all_windows


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def save_patient_windows(windows: list[WindowRecord], out_dir: Path) -> Path:
    """
    Write *windows* to a per-patient parquet file.

    Parameters
    ----------
    windows:
        Non-empty list of :class:`WindowRecord` objects for one patient.
    out_dir:
        Directory where ``{patient_id}.parquet`` will be written.

    Returns
    -------
    Path
        Absolute path to the written parquet file.
    """
    if not windows:
        raise ValueError("Cannot save empty window list")

    patient_id = windows[0].patient_id
    rows = [w.to_parquet_row() for w in windows]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patient_id}.parquet"
    df.to_parquet(out_path, index=False)
    log.debug("Saved %d windows → %s", len(df), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Batch preprocessing
# ---------------------------------------------------------------------------


def preprocess_all(
    manifest_path: Path,
    raw_dir: Path,
    out_dir: Path,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
    limit: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """
    Preprocess all usable patients listed in *manifest_path*.

    Already-processed patients (``{patient_id}.parquet`` present in
    *out_dir*) are skipped by default (``skip_existing=True``).

    Parameters
    ----------
    manifest_path:
        Path to ``data/manifest.csv``.
    raw_dir:
        Root of the waveform data tree (contains ``physionet.org/``).
    out_dir:
        Output directory for per-patient parquet files.
    window_sec:
        Window length in seconds.
    stride_sec:
        Stride in seconds.
    limit:
        If set, process at most this many patients (useful for dry-runs).
    skip_existing:
        When True, patients with an existing parquet are counted but
        not re-processed.

    Returns
    -------
    dict
        Summary statistics: counts, modalities present, failed IDs, etc.
    """
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(it: Any, **_kw: Any) -> Any:  # type: ignore[no-redef]
            return it

    manifest = pd.read_csv(manifest_path)
    usable = manifest[manifest["usable"].astype(bool)].copy()
    if limit is not None:
        usable = usable.head(limit)

    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "n_attempted": 0,
        "n_ok": 0,
        "n_skipped_existing": 0,
        "n_failed": 0,
        "n_windows_total": 0,
        "n_windows_quality_ok": 0,
        "modalities_present": set(),
        "failed_patient_ids": [],
    }

    for _, row in tqdm(usable.iterrows(), total=len(usable), desc="Preprocessing patients"):
        patient_id = str(row["patient_id"])
        record_path = str(row["record_path"])
        summary["n_attempted"] += 1

        out_path = out_dir / f"{patient_id}.parquet"
        if skip_existing and out_path.exists():
            # Tally existing parquet without reprocessing
            try:
                df_ex = pd.read_parquet(out_path)
                summary["n_skipped_existing"] += 1
                summary["n_ok"] += 1
                summary["n_windows_total"] += len(df_ex)
                summary["n_windows_quality_ok"] += int(df_ex["quality_ok"].sum())
                summary["modalities_present"].update(df_ex["modality"].unique())
            except Exception as exc:
                log.warning("Could not read existing parquet %s: %s", out_path, exc)
            continue

        try:
            windows = preprocess_patient(patient_id, record_path, raw_dir, window_sec, stride_sec)
            if not windows:
                log.warning("No windows produced for patient %s", patient_id)
                summary["n_failed"] += 1
                summary["failed_patient_ids"].append(patient_id)
                continue

            save_patient_windows(windows, out_dir)

            summary["n_ok"] += 1
            summary["n_windows_total"] += len(windows)
            summary["n_windows_quality_ok"] += sum(1 for w in windows if w.quality_ok)
            summary["modalities_present"].update(w.modality for w in windows)

        except Exception as exc:
            log.error("Failed patient %s: %s", patient_id, exc)
            summary["n_failed"] += 1
            summary["failed_patient_ids"].append(patient_id)

    summary["modalities_present"] = sorted(summary["modalities_present"])
    return summary
