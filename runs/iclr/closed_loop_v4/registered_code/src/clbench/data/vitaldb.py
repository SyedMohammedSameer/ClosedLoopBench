"""
VitalDB v1.0.0 loading: one ``.vital`` case → CLBench windows, window numerics, interventions.

VitalDB (https://physionet.org/content/vitaldb/1.0.0/) records surgical anaesthesia cases. One
case is one continuous recording, so it maps to one CLBench patient with a single segment.

Every product below shares one clock: seconds from the file's ``dtstart``.

1. **Signal windows** — ECG (lead II, plus V5 when recorded), PPG, ABP and RESP (capnography)
   resampled to 62.5 Hz and passed through the same filter / normalise / window path as
   MIMIC-IV-WDB (:func:`clbench.data.preprocess.windows_from_signal`). Encoders,
   ``fused_inference`` and the concept detectors read them unchanged.
2. **Window numerics** — the monitor's own numbers (HR, MAP, RR, SpO2 …), TCI and infusion
   state, and ventilator settings aggregated over each window. They give monitor-derived concept
   labels and the treatment context of every window.
3. **Intervention events** — timestamped clinician actions: TCI target changes (propofol,
   remifentanil), manually titrated infusion rate changes (vasopressors, vasodilators, inotropes)
   and ventilator set-rate changes. These become intervention nodes in the patient KG.

TCI pump *rates* are not interventions: the pump adjusts them itself to reach the clinician's
target, so only target (``_CT``) changes are events for TCI drugs.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clbench.data.preprocess import (
    STRIDE_SEC_DEFAULT,
    WINDOW_SEC_DEFAULT,
    WindowRecord,
    save_patient_windows,
    windows_from_signal,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_FS: float = 62.5  # MIMIC-IV-WDB rate the encoders were trained on

# modality → [(VitalDB track, channel label)]. Labels follow MIMIC naming so CHANNEL_MAP and the
# ECG lead priority apply unchanged ("V" = precordial lead, here V5).
WAVE_TRACKS: dict[str, list[tuple[str, str]]] = {
    "ECG": [("SNUADC/ECG_II", "II"), ("SNUADC/ECG_V5", "V")],
    "PPG": [("SNUADC/PLETH", "Pleth")],
    "ABP": [("SNUADC/ART", "ART")],
    "RESP": [("Primus/CO2", "CO2")],
}
MODALITIES: tuple[str, ...] = tuple(WAVE_TRACKS)

# Monitor numerics aggregated by window median. Zero means "disconnected" on these tracks.
VITAL_TRACKS: dict[str, str] = {
    "hr": "Solar8000/HR",
    "pleth_hr": "Solar8000/PLETH_HR",
    "map": "Solar8000/ART_MBP",
    "sbp": "Solar8000/ART_SBP",
    "dbp": "Solar8000/ART_DBP",
    "spo2": "Solar8000/PLETH_SPO2",
    "rr": "Solar8000/RR_CO2",
    "etco2": "Solar8000/ETCO2",
    "bt": "Solar8000/BT",
    "vent_rr": "Solar8000/VENT_RR",
}
VENT_SET_RR_TRACK: str = "Primus/SET_RR_IPPV"
VENT_RR_RANGE: tuple[float, float] = (4.0, 60.0)

# TCI pumps: the clinician sets a target concentration (CT); effect-site concentration (CE) follows.
# prefix → (agent, largest plausible target). Propofol in µg/mL, remifentanil in ng/mL.
TCI_DRUGS: dict[str, tuple[str, float]] = {
    "PPF20": ("propofol", 12.0),
    "RFTN20": ("remifentanil", 20.0),
    "RFTN50": ("remifentanil", 20.0),
}
# Manually titrated infusions: the clinician sets the rate (mL/h at the track's fixed concentration).
INFUSION_DRUGS: dict[str, str] = {
    "PHEN": "phenylephrine",
    "NEPI": "norepinephrine",
    "EPI": "epinephrine",
    "VASO": "vasopressin",
    "DOPA": "dopamine",
    "DOBU": "dobutamine",
    "MRN": "milrinone",
    "NTG": "nitroglycerin",
    "NPS": "nitroprusside",
    "DTZ": "diltiazem",
}
MAX_PUMP_RATE_ML_H: float = 500.0

EVENT_KINDS: tuple[str, ...] = ("tci_target", "infusion_rate", "vent_set_rr")
EVENT_DEBOUNCE_SEC: float = 10.0  # adjustments closer than this are one clinician action
SETPOINT_HOLD_SEC: float = 60.0  # a set-point value stays current this long without a new record

# Monitor-numeric versions of the V1 concept rules, with the same cut-offs and strict inequalities
# as ``clbench.encoders.linear_probe._CONCEPT_DETECTORS``.
NUMERIC_CONCEPT_RULES: dict[str, tuple[str, str, float]] = {
    "tachycardia": ("hr", ">", 100.0),
    "bradycardia": ("hr", "<", 60.0),
    "tachycardia_ppg": ("pleth_hr", ">", 100.0),
    "tachypnea": ("rr", ">", 20.0),
    "bradypnea": ("rr", "<", 12.0),
    "hypotension": ("map", "<", 65.0),
    "hypertension": ("map", ">", 105.0),
}


def patient_id_for(caseid: int) -> str:
    """VitalDB case → CLBench patient id; the prefix keeps it disjoint from MIMIC record ids."""
    return f"vdb{int(caseid):04d}"


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def resample_to_target(signal: np.ndarray, fs_in: float, fs_out: float = TARGET_FS) -> np.ndarray:
    """
    Anti-aliased rational resampling of ``(n_samples, n_channels)`` to ``fs_out``.

    NaN spans are interpolated for the filter and restored afterwards, so a gap in the input
    stays a gap in the output. Returns float32 of shape ``(ceil(n * fs_out / fs_in), n_channels)``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    if abs(fs_in - fs_out) < 1e-9:
        return signal.astype(np.float32)

    from scipy.signal import resample_poly

    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    n_in = signal.shape[0]
    n_out = int(np.ceil(n_in * up / down))
    src_idx = np.minimum((np.arange(n_out) * down / up).astype(np.int64), n_in - 1)

    out = np.full((n_out, signal.shape[1]), np.nan, dtype=np.float64)
    for ch in range(signal.shape[1]):
        col = signal[:, ch]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            continue
        if nan_mask.any():
            valid = np.flatnonzero(~nan_mask)
            col = col.copy()
            col[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid, col[valid])
        res = resample_poly(col, up, down)[:n_out]
        res[nan_mask[src_idx[: len(res)]]] = np.nan
        out[: len(res), ch] = res
    return out.astype(np.float32)


def window_grid(
    duration_sec: float,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
) -> np.ndarray:
    """Window start times covering a recording, matching ``_make_windows`` placement."""
    if duration_sec < window_sec:
        return np.empty(0, dtype=np.float64)
    n = int(np.floor((duration_sec - window_sec) / stride_sec + 1e-9)) + 1
    return np.arange(n, dtype=np.float64) * stride_sec


# ---------------------------------------------------------------------------
# Track records → window aggregates
# ---------------------------------------------------------------------------


def _clean(t: np.ndarray, v: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    ok = ~np.isnan(v) & (v >= lo) & (v <= hi)
    return t[ok], v[ok]


def _window_median(
    t: np.ndarray, v: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Median of the records inside each ``[start, end)``; NaN where a window has none."""
    out = np.full(len(starts), np.nan)
    if len(t) == 0:
        return out
    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, ends, side="left")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = float(np.median(v[a:b]))
    return out


def _window_last(
    t: np.ndarray, v: np.ndarray, ends: np.ndarray, hold_sec: float = SETPOINT_HOLD_SEC
) -> np.ndarray:
    """Set-point value current at each window end: the last record no older than ``hold_sec``."""
    out = np.full(len(ends), np.nan)
    if len(t) == 0:
        return out
    idx = np.searchsorted(t, ends, side="left") - 1
    ok = idx >= 0
    safe = np.clip(idx, 0, len(t) - 1)
    ok &= (ends - t[safe]) <= hold_sec
    out[ok] = v[safe[ok]]
    return out


def numeric_concept_labels(numerics: pd.DataFrame) -> pd.DataFrame:
    """``label_<concept>`` columns (nullable Int8): 1/0 from monitor numerics, NA where missing."""
    labels: dict[str, pd.Series] = {}
    for concept, (col, op, threshold) in NUMERIC_CONCEPT_RULES.items():
        x = numerics[col]
        hit = (x > threshold) if op == ">" else (x < threshold)
        labels[f"label_{concept}"] = hit.astype("Int8").where(x.notna())
    return pd.DataFrame(labels, index=numerics.index)


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


@dataclass
class InterventionEvent:
    """One clinician action on a set-point, timestamped on the case clock."""

    patient_id: str
    t_sec: float
    agent: str  # propofol | remifentanil | norepinephrine | … | ventilator
    kind: str  # tci_target | infusion_rate | vent_set_rr
    track: str
    value_before: float
    value_after: float


def step_changes(
    t: np.ndarray,
    v: np.ndarray,
    *,
    tol: float,
    lo: float,
    hi: float,
    debounce_sec: float = EVENT_DEBOUNCE_SEC,
) -> list[tuple[float, float, float]]:
    """
    Settled changes of a set-point track as ``(t_sec, value_before, value_after)``.

    Values outside ``[lo, hi]`` and NaN are dropped. The set-point starts "off" (0), so the first
    setting is itself an event. Adjustments within ``debounce_sec`` of the previous one merge into a
    single action; a merged action that returns to its starting value is discarded.
    """
    t, v = _clean(np.asarray(t, dtype=np.float64), np.asarray(v, dtype=np.float64), lo, hi)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]

    events: list[list[float]] = []
    prev = 0.0
    last_change_t = -np.inf
    for ti, vi in zip(t, v):
        if abs(vi - prev) <= tol:
            continue
        if events and ti - last_change_t <= debounce_sec:
            events[-1][2] = float(vi)
        else:
            events.append([float(ti), float(prev), float(vi)])
        prev = float(vi)
        last_change_t = float(ti)
    return [(e[0], e[1], e[2]) for e in events if abs(e[2] - e[1]) > tol]


# ---------------------------------------------------------------------------
# Case-level loading (needs the ``vitaldb`` package)
# ---------------------------------------------------------------------------


def _track_records(vf: Any, track: str) -> tuple[np.ndarray, np.ndarray]:
    """Numeric track records as sorted ``(t_sec, value)``, seconds from ``vf.dtstart``."""
    trk = vf.trks.get(track)
    if trk is None or not trk.recs:
        return np.empty(0), np.empty(0)
    t = np.array([r["dt"] for r in trk.recs], dtype=np.float64) - float(vf.dtstart)
    v = np.array([r["val"] for r in trk.recs], dtype=np.float64)
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def _merged_records(vf: Any, tracks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Records of several equivalent tracks (e.g. remifentanil 20 and 50) on one timeline."""
    parts = [_track_records(vf, trk) for trk in tracks]
    t = np.concatenate([p[0] for p in parts]) if parts else np.empty(0)
    v = np.concatenate([p[1] for p in parts]) if parts else np.empty(0)
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def load_case_windows(
    vf: Any,
    caseid: int,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
) -> list[WindowRecord]:
    """Signal windows for every modality recorded in the case."""
    pid = patient_id_for(caseid)
    segment = f"{pid}_0000"
    windows: list[WindowRecord] = []
    for modality, tracks in WAVE_TRACKS.items():
        present = [(trk, label) for trk, label in tracks if trk in vf.trks and vf.trks[trk].recs]
        if not present:
            continue
        fs_in = float(vf.trks[present[0][0]].srate)
        raw = vf.to_numpy([trk for trk, _ in present], 1.0 / fs_in)
        if raw is None or np.size(raw) == 0:
            continue
        raw = np.asarray(raw, dtype=np.float64).reshape(len(raw), -1)
        sig = resample_to_target(raw, fs_in)
        keep = [i for i in range(sig.shape[1]) if not np.isnan(sig[:, i]).all()]
        if not keep:
            continue
        windows.extend(
            windows_from_signal(
                sig[:, keep],
                TARGET_FS,
                modality,
                [present[i][1] for i in keep],
                pid,
                segment,
                start_idx=len(windows),
                window_sec=window_sec,
                stride_sec=stride_sec,
            )
        )
    return windows


def window_numerics(
    vf: Any,
    caseid: int,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
) -> pd.DataFrame:
    """Monitor numerics, TCI / infusion state and ventilator setting for every window."""
    starts = window_grid(float(vf.dtend - vf.dtstart), window_sec, stride_sec)
    ends = starts + window_sec
    cols: dict[str, np.ndarray] = {"window_start_sec": starts, "window_end_sec": ends}

    for name, track in VITAL_TRACKS.items():
        t, v = _clean(*_track_records(vf, track), lo=1e-6, hi=np.inf)
        cols[name] = _window_median(t, v, starts, ends)

    t, v = _clean(*_track_records(vf, VENT_SET_RR_TRACK), *VENT_RR_RANGE)
    cols["vent_set_rr"] = _window_last(t, v, ends)

    agents = sorted({agent for agent, _ in TCI_DRUGS.values()})
    for agent in agents:
        prefixes = [p for p, (a, _) in TCI_DRUGS.items() if a == agent]
        max_ct = max(TCI_DRUGS[p][1] for p in prefixes)
        t, v = _clean(*_merged_records(vf, [f"Orchestra/{p}_CT" for p in prefixes]), 0.0, max_ct)
        cols[f"tci_{agent}_ct"] = _window_last(t, v, ends)
        t, v = _clean(*_merged_records(vf, [f"Orchestra/{p}_CE" for p in prefixes]), 0.0, max_ct)
        cols[f"tci_{agent}_ce"] = _window_median(t, v, starts, ends)

    for prefix, agent in INFUSION_DRUGS.items():
        t, v = _clean(*_track_records(vf, f"Orchestra/{prefix}_RATE"), 0.0, MAX_PUMP_RATE_ML_H)
        cols[f"inf_{agent}_rate_ml_h"] = _window_last(t, v, ends)

    df = pd.DataFrame(cols)
    df.insert(0, "patient_id", patient_id_for(caseid))
    df["controlled_ventilation"] = df["vent_set_rr"].notna()
    return pd.concat([df, numeric_concept_labels(df)], axis=1)


def extract_events(vf: Any, caseid: int) -> list[InterventionEvent]:
    """Timestamped clinician interventions in the case, in time order."""
    pid = patient_id_for(caseid)
    events: list[InterventionEvent] = []

    for prefix, (agent, max_ct) in TCI_DRUGS.items():
        track = f"Orchestra/{prefix}_CT"
        t, v = _track_records(vf, track)
        for ti, before, after in step_changes(t, v, tol=0.05, lo=0.0, hi=max_ct):
            events.append(InterventionEvent(pid, ti, agent, "tci_target", track, before, after))

    for prefix, agent in INFUSION_DRUGS.items():
        track = f"Orchestra/{prefix}_RATE"
        t, v = _track_records(vf, track)
        for ti, before, after in step_changes(t, v, tol=0.05, lo=0.0, hi=MAX_PUMP_RATE_ML_H):
            events.append(InterventionEvent(pid, ti, agent, "infusion_rate", track, before, after))

    t, v = _track_records(vf, VENT_SET_RR_TRACK)
    for ti, before, after in step_changes(t, v, tol=0.5, lo=VENT_RR_RANGE[0], hi=VENT_RR_RANGE[1]):
        events.append(
            InterventionEvent(pid, ti, "ventilator", "vent_set_rr", VENT_SET_RR_TRACK, before, after)
        )

    events.sort(key=lambda e: e.t_sec)
    return events


def ingest_case(
    vital_path: Path,
    caseid: int,
    out_dir: Path,
    window_sec: float = WINDOW_SEC_DEFAULT,
    stride_sec: float = STRIDE_SEC_DEFAULT,
) -> dict[str, Any]:
    """
    Ingest one case and write its products under ``out_dir``.

    Writes ``{pid}.parquet`` (signal windows), ``numerics/{pid}.parquet``, ``events/{pid}.parquet``
    and, last, ``summaries/{pid}.json`` — whose presence marks the case as done.
    """
    import vitaldb

    vf = vitaldb.VitalFile(str(vital_path))
    pid = patient_id_for(caseid)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = load_case_windows(vf, caseid, window_sec, stride_sec)
    window_bytes = 0
    if windows:
        window_bytes = save_patient_windows(windows, out_dir).stat().st_size

    numerics = window_numerics(vf, caseid, window_sec, stride_sec)
    (out_dir / "numerics").mkdir(exist_ok=True)
    numerics.to_parquet(out_dir / "numerics" / f"{pid}.parquet", index=False)

    events = extract_events(vf, caseid)
    events_df = pd.DataFrame(
        [asdict(e) for e in events], columns=[f.name for f in fields(InterventionEvent)]
    )
    (out_dir / "events").mkdir(exist_ok=True)
    events_df.to_parquet(out_dir / "events" / f"{pid}.parquet", index=False)

    n_windows = Counter(w.modality for w in windows)
    n_ok = Counter(w.modality for w in windows if w.quality_ok)
    n_events = Counter(e.kind for e in events)
    ecg_leads = next((w.channel_names for w in windows if w.modality == "ECG"), [])
    summary: dict[str, Any] = {
        "patient_id": pid,
        "caseid": int(caseid),
        "duration_h": round(float(vf.dtend - vf.dtstart) / 3600.0, 3),
        "n_windows": {m: n_windows.get(m, 0) for m in MODALITIES},
        "n_windows_ok": {m: n_ok.get(m, 0) for m in MODALITIES},
        "ecg_leads": ";".join(ecg_leads),
        "n_numeric_windows": int(len(numerics)),
        "controlled_ventilation_frac": (
            round(float(numerics["controlled_ventilation"].mean()), 4) if len(numerics) else 0.0
        ),
        "n_events": {k: n_events.get(k, 0) for k in EVENT_KINDS},
        "window_parquet_bytes": int(window_bytes),
    }
    (out_dir / "summaries").mkdir(exist_ok=True)
    (out_dir / "summaries" / f"{pid}.json").write_text(json.dumps(summary, indent=2))
    return summary
