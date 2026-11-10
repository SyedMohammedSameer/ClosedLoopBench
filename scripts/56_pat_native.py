"""
Delay-aware pulse arrival time (PAT) for the VitalDB native beat tables.

Script 48 matched each R-peak to the first PLETH foot 80–600 ms later. On VitalDB the SNUADC/PLETH
foot trails the R-peak by 0.60–0.72 s (true PAT plus a constant device delay), so almost every
match fell outside the window and native PAT covers a median 3.1% of beats (final-wave track B).
This script re-derives PAT with the method that recovered the coupling on 9 local files
(runs/iclr/fast_vitaldb/pat_rescue_exploratory.csv, coverage 88–99%):

  1. read ONLY the SNUADC/PLETH track (500 Hz) — R-peaks come from the existing beats table;
  2. detect PLETH feet (clbench.beats.detect_ppg_beats, 10-min chunks);
  3. per case, D = mode of all (foot − R) lags in (0, 1.5] s over valid beats (10 ms bins);
  4. assign to each R-peak the foot nearest R + D, kept if within ±min(0.15 s, 0.4 × median RR).

PAT therefore includes the device delay; only its beat-to-beat variation is interpretable.

Output (rows aligned 1:1 with native/beats/{pid}.parquet, same order):
    <out-dir>/pat/{pid}.parquet         t_r, pat_ms, t_ppg_foot, ppg_amp
    <out-dir>/pat_summaries/{pid}.json  delay_mode_s, half_width_s, n_feet, coverage (written last)

USAGE
  # local (the 9 raw .vital files in data/raw)
  PYTHONPATH=src .venv/bin/python scripts/56_pat_native.py --limit 9 --out-dir runs/iclr/pat_native_local
  # Colab: see colab/pat_colab.py
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

log = logging.getLogger("56_pat_native")
FS = 500.0
CHUNK_SEC, PAD_SEC = 600.0, 10.0
CASESTART_ABS = 4102444800.0  # every VitalDB case starts at 2100-01-01 00:00:00 (as in script 48)


def process_case(pid: str, vital_path: str, beats_path: str, out_dir: str) -> dict:
    import vitaldb

    from clbench.beats import detect_ppg_beats

    t0 = time.time()
    out = Path(out_dir)
    summ: dict = {"patient_id": pid}
    beats = pd.read_parquet(beats_path, columns=["t_r", "rri", "valid"])
    tr = beats["t_r"].to_numpy(float)
    valid = beats["valid"].to_numpy(bool)
    pat = np.full(len(tr), np.nan)
    foot = np.full(len(tr), np.nan)
    amp = np.full(len(tr), np.nan)

    vf = vitaldb.VitalFile(vital_path, ["SNUADC/PLETH"])
    # same clock as the beats table: pin the loader origin to clinical casestart (script 48)
    if abs(vf.dtstart - CASESTART_ABS) < 60:
        vf.dtstart = CASESTART_ABS
    else:
        summ["clock_warning"] = "dtstart not within 60 s of casestart"
    if "SNUADC/PLETH" in vf.get_track_names() and len(tr):
        x = vf.to_numpy(["SNUADC/PLETH"], 1 / FS)[:, 0]
        del vf
        n = len(x)
        step, pad = int(CHUNK_SEC * FS), int(PAD_SEC * FS)
        feet, amps = [], []
        for s in range(0, n, step):
            a, b = max(0, s - pad), min(n, s + step + pad)
            if np.isfinite(x[a:b]).sum() < FS * 20:
                continue
            pg = detect_ppg_beats(x[a:b], FS)
            tf = pg.t_foot + a / FS
            keep = np.isfinite(tf) & (tf >= s / FS) & (tf < min(n, s + step) / FS)
            feet.append(tf[keep])
            amps.append(pg.amp[keep])
        del x
        tf = np.concatenate(feet) if feet else np.empty(0)
        af = np.concatenate(amps) if amps else np.empty(0)
        order = np.argsort(tf)
        tf, af = tf[order], af[order]
        summ["n_feet"] = int(len(tf))
        if len(tf) >= 50 and valid.sum() >= 50:
            tv = tr[valid]
            lo = np.searchsorted(tf, tv)
            hi = np.searchsorted(tf, tv + 1.5)
            d = np.concatenate([tf[l_:h_] - t for l_, h_, t in zip(lo, hi, tv)])
            h, edges = np.histogram(d, bins=np.arange(0.0, 1.51, 0.01))
            D = float(edges[np.argmax(h)] + 0.005)
            rr = np.nanmedian(beats["rri"].to_numpy(float)[valid]) / 1000.0
            hw = float(min(0.15, 0.4 * rr)) if np.isfinite(rr) else 0.15
            j = np.clip(np.searchsorted(tf, tr + D), 1, len(tf) - 1)
            cand = np.stack([tf[j - 1], tf[j]], axis=1)
            k = np.argmin(np.abs(cand - (tr + D)[:, None]), axis=1)
            jj = np.where(k == 0, j - 1, j)
            f = tf[jj]
            ok = np.abs(f - tr - D) <= hw
            pat = np.where(ok, (f - tr) * 1000.0, np.nan)
            foot = np.where(ok, f, np.nan)
            amp = np.where(ok, af[jj], np.nan)
            summ.update({"delay_mode_s": D, "half_width_s": hw,
                         "coverage_valid": float(np.isfinite(pat[valid]).mean()),
                         "median_pat_ms": float(np.nanmedian(pat[valid])) if np.isfinite(pat[valid]).any() else None})
    else:
        summ["error"] = "no PLETH track or no beats"
    (out / "pat").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_r": tr, "pat_ms": pat.astype("float32"), "t_ppg_foot": foot,
                  "ppg_amp": amp.astype("float32")}).to_parquet(out / "pat" / f"{pid}.parquet", compression="zstd")
    summ["seconds"] = round(time.time() - t0, 2)
    (out / "pat_summaries").mkdir(parents=True, exist_ok=True)
    (out / "pat_summaries" / f"{pid}.json").write_text(json.dumps(summ))
    return summ


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/physionet.org/files/vitaldb/1.0.0")
    p.add_argument("--beats-dir", type=Path, default=ROOT / "data/processed_vitaldb/native/beats")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/processed_vitaldb/native")
    p.add_argument("--cases-csv", type=Path, default=ROOT / "data/pat_cases.csv",
                   help="patient_id column, processed in file order (longest first)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    pids = pd.read_csv(a.cases_csv)["patient_id"].tolist()
    todo = []
    for pid in pids:
        vital = a.raw_dir / "vital_files" / f"{int(pid[3:]):04d}.vital"
        beats = a.beats_dir / f"{pid}.parquet"
        if not (vital.exists() and beats.exists()):
            continue
        if (a.out_dir / "pat_summaries" / f"{pid}.json").exists():
            continue
        todo.append((pid, str(vital), str(beats)))
    if a.limit:
        todo = todo[: a.limit]
    log.info("%d cases to process (%d listed)", len(todo), len(pids))
    t0, done = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_case, pid, v, b, str(a.out_dir)): pid for pid, v, b in todo}
        for f in as_completed(futs):
            try:
                s = f.result()
                done += 1
                if done % 50 == 0 or done == len(todo):
                    el = (time.time() - t0) / 60
                    log.info("%d/%d done, %.1f min, ~%.0f min left (last %s coverage %s)", done, len(todo), el,
                             el / done * (len(todo) - done), s["patient_id"], s.get("coverage_valid"))
            except Exception as e:  # keep going; a failed case is retried on the next run
                log.error("case %s failed: %r", futs[f], e)
    log.info("finished %d cases in %.1f min", done, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
