"""
Native-rate VitalDB extraction: beat table, breath table, 1 s numerics (checklist tracks 5, 7, 10).

Promoted from the wave-2 inventory prototype (runs/iclr/vitaldb_inventory/tools/extract_native.py,
validated on 3 local cases: beat HR vs Solar8000 r = 0.98-0.99). It reads the original
.vital files (the Drive copy on Colab, or local files) and writes compact per-case tables, so the
500 Hz waveforms never have to leave the machine that holds the raw files.

Per case (patient id vdbNNNN), under --out-dir:
    beats/{pid}.parquet        one row per R-peak: clbench.beats.build_beat_table + gate_beats on
                               SNUADC/ECG_II, SNUADC/ART, SNUADC/PLETH at 500 Hz, Primus/CO2 as the
                               respiration reference (phase). ~0.23 MB per recorded hour (1 case measured).
    breaths/{pid}.parquet      one row per breath from the Primus/CO2 capnogram (62.5 Hz) and
                               Primus/AWP: t_exp_start, t_insp_start, t_next, etco2_breath, inco2_breath,
                               ttot, te, ti, pip_breath, peep_breath.
    numerics_1s/{pid}.parquet  1 s grid of the core monitor vitals (HR, SpO2, arterial pressures, EtCO2, RR,
                               BT), every Orchestra TCI target / effect-site / pump-rate track, and tracks
                               NOT in the 30 s / 2 s numerics (BIS, EMG, SQI, SR, SEF, MAC, agents,
                               ventilator measured + set values, ST, NIBP, CVP, SVV/CO).
                               Values sit at the second they were recorded; slower tracks (Primus ~7 s,
                               NIBP) are NaN in between -- no forward fill is applied here.
    summaries/{pid}.json       counts and timings; written last (marks the case done; resumable).

Times are seconds from clinical casestart (absolute 2100-01-01 00:00:00, the 'Case started' EVENT), the
origin of clinical_data.csv (opstart, anestart, ...) and lab_data.csv. NOTE: the existing
data/processed_vitaldb products use VitalFile's loader origin instead (earliest loaded record; measured
0.0-0.94 s after casestart on 10 files); the summary records that offset per case.

USAGE
  # local smoke test on the .vital files present in data/raw
  PYTHONPATH=src .venv/bin/python scripts/48_vitaldb_native_extract.py --limit 3 --workers 3 \
      --out-dir runs/iclr/native_smoke

  # Colab (see colab/vitaldb_native_colab.py), reading the Drive copy
  NG=/content/drive/MyDrive/CLBench
  cd /content/clbench && PYTHONPATH=src python scripts/48_vitaldb_native_extract.py \
      --raw-dir $NG/data/raw/physionet.org/files/vitaldb/1.0.0 \
      --out-dir $NG/data/processed_vitaldb/native --manifest $NG/data/vitaldb_manifest.csv --workers 12

Depends on clbench.beats (build_beat_table, gate_beats); check its signature before a long run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

log = logging.getLogger("extract_native")

FS_WAVE = 500.0
FS_CO2 = 62.5
CHUNK_SEC = 1800.0   # beat extraction in 30 min chunks
PAD_SEC = 10.0       # overlap either side; beats are kept only inside their own chunk

WAVES = ["SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH"]
RESP = ["Primus/CO2", "Primus/AWP"]
NUMERICS_1S = {
    "bis": "BIS/BIS", "bis_sqi": "BIS/SQI", "bis_emg": "BIS/EMG", "bis_sr": "BIS/SR", "bis_sef": "BIS/SEF",
    "bis_totpow": "BIS/TOTPOW",
    "mac": "Primus/MAC", "exp_sevo": "Primus/EXP_SEVO", "insp_sevo": "Primus/INSP_SEVO",
    "exp_des": "Primus/EXP_DES", "insp_des": "Primus/INSP_DES", "fen2o": "Primus/FEN2O",
    "fio2": "Primus/FIO2", "feo2": "Primus/FEO2", "etco2_primus": "Primus/ETCO2", "inco2": "Primus/INCO2",
    "tv": "Primus/TV", "mv": "Primus/MV", "pip": "Primus/PIP_MBAR", "pplat": "Primus/PPLAT_MBAR",
    "peep": "Primus/PEEP_MBAR", "mawp": "Primus/MAWP_MBAR", "compliance": "Primus/COMPLIANCE",
    "set_tv": "Primus/SET_TV_L", "set_peep": "Primus/SET_INTER_PEEP", "set_fio2": "Primus/SET_FIO2",
    "set_fresh_flow": "Primus/SET_FRESH_FLOW", "set_insp_tm": "Primus/SET_INSP_TM", "set_pip": "Primus/SET_PIP",
    "set_rr": "Primus/SET_RR_IPPV",
    "st_ii": "Solar8000/ST_II", "st_i": "Solar8000/ST_I", "st_iii": "Solar8000/ST_III", "st_v5": "Solar8000/ST_V5",
    "nibp_sbp": "Solar8000/NIBP_SBP", "nibp_mbp": "Solar8000/NIBP_MBP", "nibp_dbp": "Solar8000/NIBP_DBP",
    "cvp": "Solar8000/CVP", "gas2_exp": "Solar8000/GAS2_EXPIRED",
    "ev_svv": "EV1000/SVV", "ev_co": "EV1000/CO", "ev_sv": "EV1000/SV", "ev_svr": "EV1000/SVR",
    "vg_svv": "Vigileo/SVV", "vg_co": "Vigileo/CO",
    "roc_rate": "Orchestra/ROC_RATE",
    # core monitor vitals at 1 s (the 2 s export covers only 500 cases)
    "hr": "Solar8000/HR", "pleth_hr": "Solar8000/PLETH_HR", "spo2": "Solar8000/PLETH_SPO2",
    "art_sbp": "Solar8000/ART_SBP", "art_mbp": "Solar8000/ART_MBP", "art_dbp": "Solar8000/ART_DBP",
    "etco2": "Solar8000/ETCO2", "rr_co2": "Solar8000/RR_CO2", "vent_rr": "Solar8000/VENT_RR",
    "bt": "Solar8000/BT",
}
# every TCI target / effect-site track and every titrated pump rate (see clbench.data.vitaldb)
for _p in ("PPF20", "RFTN20", "RFTN50"):
    NUMERICS_1S[f"{_p.lower()}_ct"] = f"Orchestra/{_p}_CT"
    NUMERICS_1S[f"{_p.lower()}_ce"] = f"Orchestra/{_p}_CE"
    NUMERICS_1S[f"{_p.lower()}_rate"] = f"Orchestra/{_p}_RATE"
for _p in ("PHEN", "NEPI", "EPI", "VASO", "DOPA", "DOBU", "MRN", "NTG", "NPS", "DTZ"):
    NUMERICS_1S[f"{_p.lower()}_rate"] = f"Orchestra/{_p}_RATE"
AGENT_TRACK = "Solar8000/GAS2_AGENT"  # string track: 'Sevoflurane', 'Desflurane', ...
CASESTART_ABS = 4102444800.0  # VitalDB anonymises every case to start at 2100-01-01 00:00:00 UTC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/physionet.org/files/vitaldb/1.0.0")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/processed_vitaldb/native")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/vitaldb_manifest.csv")
    p.add_argument("--caseids", type=int, nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--skip", nargs="*", default=[], choices=["beats", "breaths", "numerics"])
    p.add_argument("--from-api", action="store_true",
                   help="stream from api.vitaldb.net instead of --raw-dir (use with --skip beats breaths: ~8%% of each file)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# breaths from the capnogram
# ---------------------------------------------------------------------------


def _rolling(x: np.ndarray, win: int, q: float) -> np.ndarray:
    return pd.Series(x).rolling(win, center=True, min_periods=max(1, win // 4)).quantile(q).to_numpy()


def breath_table(co2: np.ndarray, awp: np.ndarray | None, fs: float = FS_CO2) -> pd.DataFrame:
    """
    Breath segmentation of a capnogram. Exhalation starts where CO2 rises through the midpoint of its
    local (20 s) 5th-95th percentile range; inspiration starts where it falls back through it. Breaths
    need a local CO2 swing >= 10 mmHg and a total duration of 1-20 s.
    """
    cols = ["t_exp_start", "t_insp_start", "t_next", "etco2_breath", "inco2_breath", "ttot", "te", "ti",
            "pip_breath", "peep_breath"]
    x = np.asarray(co2, float)
    if len(x) < fs * 30 or np.isfinite(x).sum() < fs * 30:
        return pd.DataFrame(columns=cols)
    win = int(20 * fs)
    lo, hi = _rolling(x, win, 0.05), _rolling(x, win, 0.95)
    mid = lo + 0.5 * (hi - lo)
    ok = np.isfinite(x) & np.isfinite(mid) & ((hi - lo) >= 10.0)
    above = (x > mid) & ok
    d = np.diff(above.astype(np.int8))
    rise = np.where(d == 1)[0] + 1
    fall = np.where(d == -1)[0] + 1
    rows = []
    fi = np.searchsorted(fall, rise)
    for k in range(len(rise) - 1):
        r, r2 = rise[k], rise[k + 1]
        if fi[k] >= len(fall):
            break
        f = fall[fi[k]]
        if not (r < f < r2):
            continue
        ttot = (r2 - r) / fs
        if not (1.0 <= ttot <= 20.0):
            continue
        seg_e = x[r:f]
        seg_i = x[f:r2]
        if not (np.isfinite(seg_e).any() and np.isfinite(seg_i).any()):
            continue
        row = [r / fs, f / fs, r2 / fs, float(np.nanpercentile(seg_e, 95)), float(np.nanmin(seg_i)), ttot,
               (f - r) / fs, (r2 - f) / fs, np.nan, np.nan]
        if awp is not None:
            a_i = awp[f:r2]
            a_e = awp[r:f]
            if np.isfinite(a_i).any():
                row[8] = float(np.nanmax(a_i))
            tail = a_e[int(0.8 * len(a_e)):] if len(a_e) >= 5 else a_e
            if np.isfinite(tail).any():
                row[9] = float(np.nanmedian(tail))
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# per case
# ---------------------------------------------------------------------------


def beats_chunked(ecg, abp, ppg, co2_on_wave_grid):
    from clbench.beats import build_beat_table, gate_beats

    n = len(ecg)
    step, pad = int(CHUNK_SEC * FS_WAVE), int(PAD_SEC * FS_WAVE)
    parts = []
    for s in range(0, n, step):
        a, b = max(0, s - pad), min(n, s + step + pad)
        e = ecg[a:b]
        if np.isfinite(e).sum() < FS_WAVE * 20:
            continue
        bt = build_beat_table(e, FS_WAVE, abp[a:b], FS_WAVE, ppg=ppg[a:b], fs_ppg=FS_WAVE,
                              resp=co2_on_wave_grid[a:b] if co2_on_wave_grid is not None else None,
                              fs_resp=FS_WAVE if co2_on_wave_grid is not None else None, t0=a / FS_WAVE)
        if len(bt) == 0:
            continue
        bt = gate_beats(bt)
        keep = (bt.t_r >= s / FS_WAVE) & (bt.t_r < min(n, s + step) / FS_WAVE)
        parts.append(bt[keep])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).sort_values("t_r").reset_index(drop=True)
    return out


def process_case(caseid: int, vital_path: str, out_dir: str, skip: tuple[str, ...]) -> dict:
    import vitaldb

    pid = f"vdb{int(caseid):04d}"
    out = Path(out_dir)
    t0 = time.time()
    summ: dict = {"caseid": int(caseid), "patient_id": pid, "vital": vital_path}
    want = list(NUMERICS_1S.values()) + [AGENT_TRACK]
    if "numerics" in skip:
        want = []
    if "breaths" not in skip:
        want += RESP + ["Primus/ETCO2", "Primus/PIP_MBAR"]
    if "beats" not in skip:
        want += WAVES + ["Primus/CO2", "Solar8000/ART_MBP", "Solar8000/HR"]
    want = list(dict.fromkeys(want))
    # vital_path may be an int caseid: VitalFile then streams api.vitaldb.net/1.0.1/<caseid>.vital
    # (packed; with numerics only, the stream stops after ~8% of the file)
    src = int(vital_path) if str(vital_path).isdigit() else vital_path
    vf = vitaldb.VitalFile(src, want)
    # Clock. PhysioNet 1.0.0 files carry no dtstart in the header, so VitalFile sets it to the earliest
    # record of the LOADED tracks (0-3 s after case start, depending on the track set); API 1.0.1 files
    # carry dtstart = 2100-01-01 00:00:00 = clinical casestart. Pin every output to casestart so the two
    # sources and clinical_data.csv (opstart, anestart, lab dt) share one clock.
    summ["loader_origin_minus_casestart_s"] = round(float(vf.dtstart - CASESTART_ABS), 3)
    if abs(vf.dtstart - CASESTART_ABS) < 60:
        vf.dtstart = CASESTART_ABS
    else:
        summ["clock_warning"] = "dtstart not within 60 s of 2100-01-01; kept loader origin"
    summ["load_s"] = round(time.time() - t0, 2)
    summ["duration_s"] = float(vf.dtend - vf.dtstart)
    present = set(vf.get_track_names())
    summ["tracks_present"] = sorted(present & set(want))

    if "breaths" not in skip and "Primus/CO2" in present:
        t = time.time()
        R = vf.to_numpy(RESP, 1 / FS_CO2)
        br = breath_table(R[:, 0], R[:, 1] if "Primus/AWP" in present else None)
        # The Primus/CO2 wave's ADC gain is 0.1 in most files but 0.075 in some; with gain 0.1 the wave
        # reads ~1.33x the Primus/Solar8000 EtCO2 numerics (and above simultaneous PaCO2). Calibrate each
        # case to the Primus/ETCO2 numeric and keep the raw values.
        summ["co2_wave_gain"] = float(vf.trks["Primus/CO2"].gain)
        summ["awp_wave_gain"] = float(vf.trks["Primus/AWP"].gain) if "Primus/AWP" in present else None
        br["etco2_breath_raw"] = br["etco2_breath"]
        br["inco2_breath_raw"] = br["inco2_breath"]
        scale = np.nan
        if len(br) and "Primus/ETCO2" in present:
            ref = vf.to_numpy(["Primus/ETCO2"], 1)[:, 0]
            i = np.clip(br["t_insp_start"].to_numpy().astype(int), 0, len(ref) - 1)
            # Primus numerics update every ~7 s: take the next available value within 10 s
            refv = pd.Series(ref).bfill(limit=10).to_numpy()[i]
            ok = np.isfinite(refv) & (refv > 15) & np.isfinite(br["etco2_breath_raw"].to_numpy())
            if ok.sum() >= 30:
                scale = float(np.median(refv[ok]) / np.median(br["etco2_breath_raw"].to_numpy()[ok]))
        summ["co2_scale_to_primus_etco2"] = None if not np.isfinite(scale) else round(scale, 4)
        if np.isfinite(scale):
            br["etco2_breath"] = br["etco2_breath_raw"] * scale
            br["inco2_breath"] = br["inco2_breath_raw"] * scale
        else:
            br["etco2_breath"] = np.nan
            br["inco2_breath"] = np.nan
        if len(br) and "Primus/PIP_MBAR" in present:
            ref = pd.Series(vf.to_numpy(["Primus/PIP_MBAR"], 1)[:, 0]).bfill(limit=10).to_numpy()
            i = np.clip(br["t_next"].to_numpy().astype(int), 0, len(ref) - 1)
            ok = np.isfinite(ref[i]) & (ref[i] > 3) & np.isfinite(br["pip_breath"].to_numpy())
            summ["pip_ratio_primus_over_wave"] = (round(float(np.median(ref[i][ok]) / np.median(br["pip_breath"].to_numpy()[ok])), 4)
                                                  if ok.sum() >= 30 else None)
        (out / "breaths").mkdir(parents=True, exist_ok=True)
        br.astype("float32").to_parquet(out / "breaths" / f"{pid}.parquet", compression="zstd")
        summ["breaths"] = int(len(br))
        summ["breaths_s"] = round(time.time() - t, 2)

    if "beats" not in skip and all(w in present for w in WAVES):
        t = time.time()
        X = vf.to_numpy(WAVES + ["Primus/CO2"], 1 / FS_WAVE)  # CO2 is sample-and-held onto the 500 Hz grid
        co2 = X[:, 3] if "Primus/CO2" in present else None
        bt = beats_chunked(X[:, 0], X[:, 1], X[:, 2], co2)
        del X
        (out / "beats").mkdir(parents=True, exist_ok=True)
        if len(bt):
            f32 = {c: "float32" for c in bt.columns if bt[c].dtype == np.float64 and not c.startswith("t_")}
            bt = bt.astype(f32)
        bt.to_parquet(out / "beats" / f"{pid}.parquet", compression="zstd")
        summ["beats"] = int(len(bt))
        summ["beats_valid_frac"] = round(float(bt["valid"].mean()), 4) if len(bt) else None
        # SNUADC/ART vs Solar8000 numerics: a per-case additive offset of several mmHg was seen on the
        # 9 local cases (-7.3 to +5.8 mmHg, SBP and DBP shifted together). Record it; do not apply it.
        if len(bt) and bt["valid"].any() and "Solar8000/ART_MBP" in present:
            v = bt[bt["valid"]]
            ref = vf.to_numpy(["Solar8000/ART_MBP", "Solar8000/HR"], 1)
            refb = pd.DataFrame(ref, columns=["map", "hr"]).where(lambda d: d > 0)
            refb = refb.groupby(np.arange(len(refb)) // 30).median()
            g = (v["t_r"] // 30).astype(int)
            bb = pd.DataFrame({"map_w": v["map_beat"].to_numpy(), "hr_w": 60000.0 / v["rri"].to_numpy(), "g": g})
            bb = bb.groupby("g").median().join(refb, how="inner").dropna()
            if len(bb) >= 20:
                summ["art_map_offset_solar_minus_wave_mmHg"] = round(float(np.median(bb["map"] - bb["map_w"])), 2)
                summ["hr_bias_beat_minus_solar_bpm"] = round(float(np.median(bb["hr_w"] - bb["hr"])), 2)
                summ["n_30s_bins_compared"] = int(len(bb))
        summ["beats_s"] = round(time.time() - t, 2)

    if "numerics" not in skip:
        t = time.time()
        cols = {k: v for k, v in NUMERICS_1S.items() if v in present}
        num = pd.DataFrame({"t_sec": np.arange(int(np.ceil(summ["duration_s"])) , dtype=np.float32)})
        if cols:
            A = vf.to_numpy(list(cols.values()), 1)
            m = min(len(A), len(num))
            num = num.iloc[:m].copy()
            for j, k in enumerate(cols):
                num[k] = A[:m, j].astype(np.float32)
        if AGENT_TRACK in present:
            recs = vf.trks[AGENT_TRACK].recs
            if recs:
                ag = pd.Series([r["val"] for r in recs]).value_counts()
                summ["gas2_agent_counts"] = {str(k): int(v) for k, v in ag.items()}
        (out / "numerics_1s").mkdir(parents=True, exist_ok=True)
        num.to_parquet(out / "numerics_1s" / f"{pid}.parquet", compression="zstd")
        summ["numerics_cols"] = sorted(cols)
        summ["numerics_s"] = round(time.time() - t, 2)

    summ["total_s"] = round(time.time() - t0, 2)
    (out / "summaries").mkdir(parents=True, exist_ok=True)
    (out / "summaries" / f"{pid}.json").write_text(json.dumps(summ, indent=1))
    return summ


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    a = parse_args()
    man = pd.read_csv(a.manifest)
    col = "caseid" if "caseid" in man.columns else None
    if a.caseids is not None:
        ids = a.caseids
    elif col:
        ids = man[col].tolist()
    else:  # manifest keyed by patient_id vdbNNNN
        ids = [int(str(p).replace("vdb", "")) for p in man["patient_id"]]
    vdir = a.raw_dir / "vital_files"
    todo = []
    for cid in ids:
        p = str(int(cid)) if a.from_api else vdir / f"{int(cid):04d}.vital"
        if not a.from_api and not Path(p).exists():
            continue
        if not a.no_skip_existing and (a.out_dir / "summaries" / f"vdb{int(cid):04d}.json").exists():
            continue
        todo.append((cid, str(p)))
    if a.limit:
        todo = todo[: a.limit]
    log.info("%d cases to process (%d requested)", len(todo), len(ids))
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_case, c, p, str(a.out_dir), tuple(a.skip)): c for c, p in todo}
        for f in as_completed(futs):
            try:
                s = f.result()
                done += 1
                log.info("case %s: %s beats, %s breaths, %.1fs (%d/%d)", s["caseid"], s.get("beats"),
                         s.get("breaths"), s["total_s"], done, len(todo))
            except Exception as e:  # keep going; the case is retried on the next run
                log.error("case %s failed: %r", futs[f], e)
    log.info("finished %d cases in %.1f min", done, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
