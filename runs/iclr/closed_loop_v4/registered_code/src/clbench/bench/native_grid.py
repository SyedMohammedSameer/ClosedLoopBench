"""
ClosedLoopBench v4 data: a 2 s grid for every VitalDB case from the native 1 s numerics, with all action
streams on ONE clock (seconds from the clinical casestart).

Inputs (``data/processed_vitaldb``):
    native/numerics_1s/<pid>.parquet   t_sec (casestart clock) + monitor / ventilator / pump numerics, value at
                                       the second recorded, NaN between (no forward fill)
    native/summaries/<pid>.json        loader_origin_minus_casestart_s (the older tables' clock offset)
    events/<pid>.parquet               v1-v3 action events (propofol, remifentanil, ventilator RR, pressors) on the
                                       VitalFile loader clock
    numerics/<pid>.parquet             30 s tables (loader clock), used only for the v3 track-stop times

Clock. The data notes say the older tables use the VitalFile loader origin, loader_origin_minus_casestart_s (0-6.5 s)
later than the casestart clock of the native tables. EMPIRICALLY this does not hold between the events file and the
native 1 s numerics: the events-file ventilator steps coincide with the native Primus SET_RR record changes WITHOUT a
shift (median |dt| 0.2-0.6 s = the native table's 1 s rounding) and are off by exactly the offset WITH it (cases with
offsets 2.5-6.5 s: shifted median |dt| 3.0-7.1 s). So ``EVENTS_SHIFT = False``: the events file and the 30 s
numerics (stop times) are joined without a shift. ``qc.csv`` keeps both timing checks for every case.

Outputs (``out_dir``):
    grid/<pid>.parquet    window_start_sec (2 k), map, hr, etco2, spo2, bis, mac: median of the valid records in
                          [2k, 2k + 2) (zero = disconnected; plausibility ranges below; BIS only with SQI >= 50)
    events/<pid>.parquet  patient_id, t_sec, agent, value_before, value_after, source
                          source = "events_file" (v3 agents, stored clock; see Clock) | "native_setpoint"
                          (set TV, set PEEP, set FiO2 from Primus set-point records, ``step_changes`` as the v1-v3
                          ventilator events) | "derived_mac" (steps of the measured Primus MAC, see below)
    stops.parquet         pid, agent, t_stop (casestart clock): last recorded value of the action's track
    qc.csv                per case: clock offset, n rows, vital coverage, event counts per agent, and the timing
                          agreement between the shifted events-file ventilator events and the native SET_RR record

MAC events are NOT recorded clinician actions: the vaporiser dial is not recorded in VitalDB. A MAC "event" is a
step of >= 0.2 MAC (hysteresis on the 30 s rolling median of the measured end-tidal MAC, levels < 0.2 MAC = off) and
lags the dial change by the wash-in time. Relations on MAC are marked ``derived exposure`` in the ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STEP_SEC: float = 2.0
DEFAULT_DATA = Path(__file__).resolve().parents[3] / "data" / "processed_vitaldb"

# grid vital -> (native column, lo, hi). Zero / non-positive = disconnected (as v1-v3 VITAL_TRACKS medians).
GRID_VITALS: dict[str, tuple[str, float, float]] = {
    "map": ("art_mbp", 20.0, 200.0),
    "hr": ("hr", 20.0, 250.0),
    "etco2": ("etco2", 5.0, 100.0),
    "spo2": ("spo2", 50.0, 100.0),
    "bis": ("bis", 1.0, 100.0),
    "mac": ("mac", 0.0, 4.0),  # 0 is a real value (agent off); kept
}
BIS_MIN_SQI: float = 50.0
# native set-point column -> (agent, tol, lo, hi); step_changes as the v1-v3 ventilator RR events
NATIVE_SETPOINTS: dict[str, tuple[str, float, float, float]] = {
    "set_tv": ("vent_tv", 0.005, 0.05, 2.0),  # litres
    "set_peep": ("peep", 0.5, 0.0, 30.0),  # cmH2O / mbar
    "set_fio2": ("fio2", 0.5, 21.0, 100.0),  # %
}
QC_SETPOINTS: dict[str, tuple[str, float, float, float]] = {"set_rr": ("ventilator", 0.5, 4.0, 60.0)}
MAC_STEP: float = 0.2
MAC_ON: float = 0.2
MAC_MEDIAN_SEC: float = 30.0
MAC_GAP_RESET_SEC: float = 120.0
EVENTS_SHIFT: bool = False  # see the module docstring (clock QC)
# 30 s numerics track columns of the v3 agents (stop times, loader clock)
V3_TRACK_COLUMNS: dict[str, str] = {"propofol": "tci_propofol_ct", "remifentanil": "tci_remifentanil_ct",
                                    "ventilator": "vent_set_rr", "phenylephrine": "inf_phenylephrine_rate_ml_h",
                                    "norepinephrine": "inf_norepinephrine_rate_ml_h",
                                    "nitroglycerin": "inf_nitroglycerin_rate_ml_h"}
V3_AGENTS: tuple[str, ...] = tuple(V3_TRACK_COLUMNS)


def _bin_median(t: np.ndarray, v: np.ndarray, n: int, step: float = STEP_SEC) -> np.ndarray:
    """Median of the records in each [k step, (k + 1) step) (NaN where none). Records are 1 s apart at most, so a
    2 s bin holds <= 2 records and the median is their mean."""
    out = np.full(n, np.nan)
    ok = np.isfinite(v) & np.isfinite(t) & (t >= 0)
    if not ok.any():
        return out
    k = np.floor(t[ok] / step).astype(np.int64)
    keep = k < n
    k, vv = k[keep], v[ok][keep]
    s = np.bincount(k, weights=vv, minlength=n)
    c = np.bincount(k, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def mac_events(t_grid: np.ndarray, mac: np.ndarray) -> list[tuple[float, float, float]]:
    """Steps of the measured MAC: 30 s rolling median, hysteresis of ``MAC_STEP``; levels below ``MAC_ON`` are 0
    (off). A gap of more than ``MAC_GAP_RESET_SEC`` without a record resets the level (no event across it)."""
    import pandas as pd

    s = pd.Series(mac)
    w = max(1, int(round(MAC_MEDIAN_SEC / STEP_SEC)))
    med = s.rolling(w, min_periods=2).median().to_numpy()
    ok = np.isfinite(med)
    out: list[tuple[float, float, float]] = []
    level = None
    last_t = -np.inf
    for ti, m in zip(t_grid[ok], med[ok]):
        m = float(np.round(m, 1))
        m = 0.0 if m < MAC_ON else m
        if level is None or ti - last_t > MAC_GAP_RESET_SEC:
            level = m
        elif abs(m - level) >= MAC_STEP - 1e-9:
            out.append((float(ti), float(level), float(m)))
            level = m
        last_t = ti
    return out


def _nearest_dt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a_i - nearest b| for each a_i."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0)
    b = np.sort(b)
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    if len(b) == 1:
        return np.abs(a - b[0])
    return np.minimum(np.abs(a - b[i - 1]), np.abs(a - b[i]))


def build_case(pid: str, data: Path = DEFAULT_DATA, shift_events: bool = EVENTS_SHIFT) -> tuple[dict, "object", "object", dict]:
    """(grid columns, events DataFrame, stops dict agent -> t_stop, qc dict) for one case. ``shift_events``: add
    loader_origin_minus_casestart_s to the events-file and 30 s-table times (False by default, see the clock QC)."""
    import pandas as pd

    from clbench.data.vitaldb import step_changes

    data = Path(data)
    summ = json.loads((data / "native" / "summaries" / f"{pid}.json").read_text())
    off_raw = float(summ.get("loader_origin_minus_casestart_s", 0.0) or 0.0)
    off = off_raw if shift_events else 0.0
    d = pd.read_parquet(data / "native" / "numerics_1s" / f"{pid}.parquet")
    t = d["t_sec"].to_numpy(float)
    n = int(np.floor(np.nanmax(t) / STEP_SEC)) + 1 if len(t) else 0
    grid = {"window_start_sec": np.arange(n) * STEP_SEC}
    for g, (col, lo, hi) in GRID_VITALS.items():
        if col not in d:
            grid[g] = np.full(n, np.nan)
            continue
        v = d[col].to_numpy(float).copy()
        if g == "mac":
            v[(v < lo) | (v > hi)] = np.nan
        else:
            v[(v <= 0) | (v < lo) | (v > hi)] = np.nan
        if g == "bis" and "bis_sqi" in d:
            sqi = d["bis_sqi"].to_numpy(float)
            v[~(sqi >= BIS_MIN_SQI)] = np.nan
        grid[g] = _bin_median(t, v, n)
    if "etco2" in d and not np.isfinite(grid["etco2"]).any() and "etco2_primus" in d:
        v = d["etco2_primus"].to_numpy(float).copy()
        v[(v <= 0) | (v < 5) | (v > 100)] = np.nan
        grid["etco2"] = _bin_median(t, v, n)
    rows: list[tuple] = []
    stops: dict[str, float] = {}
    # (1) v1-v3 agents from the events file, shifted to the casestart clock
    ef = data / "events" / f"{pid}.parquet"
    if ef.exists():
        e = pd.read_parquet(ef, columns=["t_sec", "agent", "value_before", "value_after"])
        e = e[e.agent.isin(V3_AGENTS)]
        for r in e.itertuples(index=False):
            rows.append((pid, float(r.t_sec) + off, r.agent, float(r.value_before), float(r.value_after), "events_file"))
    nf = data / "numerics" / f"{pid}.parquet"
    if nf.exists():
        cols = ["window_end_sec", *V3_TRACK_COLUMNS.values()]
        try:
            n30 = pd.read_parquet(nf, columns=cols)
        except Exception:  # noqa: BLE001 - a missing optional column
            n30 = pd.read_parquet(nf)
        te = n30["window_end_sec"].to_numpy(float)
        for agent, col in V3_TRACK_COLUMNS.items():
            if col in n30:
                ok = np.isfinite(n30[col].to_numpy(float))
                stops[agent] = float(te[ok].max()) + off if ok.any() else np.inf
    # (2) native set-points
    for col, (agent, tol, lo, hi) in NATIVE_SETPOINTS.items():
        if col not in d:
            continue
        v = d[col].to_numpy(float)
        ok = np.isfinite(v)
        if not ok.any():
            continue
        okr = ok & (v >= lo) & (v <= hi)
        t_first = float(t[okr].min()) if okr.any() else np.inf
        for ti, b, a in step_changes(t[ok], v[ok], tol=tol, lo=lo, hi=hi):
            if ti <= t_first:  # step_changes starts from 0: the first record is initialisation, not an action
                continue
            rows.append((pid, float(ti), agent, float(b), float(a), "native_setpoint"))
        stops[agent] = float(t[okr].max()) if okr.any() else np.inf
    # (3) derived MAC steps
    if np.isfinite(grid["mac"]).any():
        tg = grid["window_start_sec"] + STEP_SEC / 2
        for ti, b, a in mac_events(tg, grid["mac"]):
            rows.append((pid, ti, "mac", b, a, "derived_mac"))
        okm = np.isfinite(grid["mac"])
        stops["mac"] = float(tg[okm].max())
    ev = pd.DataFrame(rows, columns=["patient_id", "t_sec", "agent", "value_before", "value_after", "source"])
    # QC: events-file ventilator events (shifted) vs native SET_RR steps
    qc = dict(pid=pid, offset_s=off_raw, shift_applied_s=off, n_rows=n, hr_bias_beat_minus_solar_bpm=summ.get("hr_bias_beat_minus_solar_bpm"))
    for g in GRID_VITALS:
        qc[f"cov_{g}"] = float(np.isfinite(grid[g]).mean()) if n else 0.0
    for agent, g_ in ev.groupby("agent"):
        qc[f"n_{agent}"] = int(len(g_))
    if "set_rr" in d:
        v = d["set_rr"].to_numpy(float)
        ok = np.isfinite(v)
        nat = np.array([x[0] for x in step_changes(t[ok], v[ok], tol=0.5, lo=4.0, hi=60.0)]) if ok.any() else np.zeros(0)
        evv = ev[(ev.agent == "ventilator")].t_sec.to_numpy(float)
        dt_raw = _nearest_dt(evv - off, nat)  # events-file times as stored
        dt_sh = _nearest_dt(evv - off + off_raw, nat)  # with the documented loader -> casestart shift
        qc.update(n_setrr_native=int(len(nat)), n_vent_eventsfile=int(len(evv)),
                  vent_dt_unshifted_median_s=float(np.median(dt_raw)) if len(dt_raw) else np.nan,
                  vent_dt_unshifted_le1s_frac=float((dt_raw <= 1.0).mean()) if len(dt_raw) else np.nan,
                  vent_dt_shifted_median_s=float(np.median(dt_sh)) if len(dt_sh) else np.nan)
    return grid, ev, stops, qc


def build_job(args: tuple[str, str, str]) -> tuple[str, dict, dict, str | None]:
    pid, data, out = args
    import pandas as pd

    try:
        grid, ev, stops, qc = build_case(pid, Path(data))
        pd.DataFrame(grid).astype({k: "float32" for k in grid if k != "window_start_sec"}).to_parquet(
            Path(out) / "grid" / f"{pid}.parquet", index=False)
        ev.to_parquet(Path(out) / "events" / f"{pid}.parquet", index=False)
        return pid, stops, qc, None
    except Exception as e:  # noqa: BLE001 - report, do not stop the build
        return pid, {}, {"pid": pid}, repr(e)


def build_all(out: Path, data: Path = DEFAULT_DATA, workers: int = 4, limit: int = 0) -> dict:
    """Build the grid, events, stops and QC for every case with native numerics. Returns a summary dict."""
    import time
    from multiprocessing import get_context

    import pandas as pd

    out = Path(out)
    (out / "grid").mkdir(parents=True, exist_ok=True)
    (out / "events").mkdir(parents=True, exist_ok=True)
    pids = sorted(p.stem for p in (Path(data) / "native" / "numerics_1s").glob("*.parquet"))
    if limit:
        pids = pids[:limit]
    t0 = time.time()
    stops_rows, qcs, errs = [], [], []
    with get_context("spawn").Pool(workers) as pool:
        for pid, st, qc, err in pool.imap(build_job, [(p, str(data), str(out)) for p in pids], chunksize=8):
            if err:
                errs.append((pid, err))
                continue
            qcs.append(qc)
            stops_rows += [(pid, a, v) for a, v in st.items()]
    pd.DataFrame(stops_rows, columns=["pid", "agent", "t_stop"]).to_parquet(out / "stops.parquet", index=False)
    Q = pd.DataFrame(qcs)
    Q.to_csv(out / "qc.csv", index=False)
    summ = dict(n_cases=len(pids), n_built=len(qcs), n_errors=len(errs), errors=errs[:20], seconds=time.time() - t0)
    (out / "build_summary.json").write_text(json.dumps(summ, indent=2, default=str))
    return summ


__all__ = ["GRID_VITALS", "NATIVE_SETPOINTS", "STEP_SEC", "build_all", "build_case", "mac_events"]
