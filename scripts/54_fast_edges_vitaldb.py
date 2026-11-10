"""
Track B (final wave): fast coupling edges at VitalDB scale, monitor display latency, and the wave-4
patient-specificity test on all QC-passing VitalDB cases.

Pre-registered protocol: runs/iclr/fast_vitaldb/protocol.md (sha256 in protocol.sha256). The analysis stages
refuse to run if the protocol no longer matches the registered hash.

Data: VitalDB native tables (scripts/48_vitaldb_native_extract.py, run on Colab):
data/processed_vitaldb/native/{beats,breaths,numerics_1s,summaries}; all on the clinical-casestart clock.

Stages (PYTHONPATH=src:.):
  python scripts/54_fast_edges_vitaldb.py --stage extract --workers 4      # QC, epochs, lag scans
  python scripts/54_fast_edges_vitaldb.py --stage fits --workers 4         # 2 s MAP -> HR fits (a, b, b2, c)
  python scripts/54_fast_edges_vitaldb.py --stage specificity --workers 4  # wave-4 registered test at scale
  python scripts/54_fast_edges_vitaldb.py --stage report

clbench.beats and clbench.edges.fast_coupling are imported, not modified; the wave-4 epoch statistics and
split-half machinery are imported unchanged from scripts/52_fast_edge_specificity.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clbench.edges.fast_coupling import bin_means, lagged_timescale_fit, pat_coupling  # noqa: E402

_spec = importlib.util.spec_from_file_location("s52", ROOT / "scripts" / "52_fast_edge_specificity.py")
s52 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s52)

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("54_fast_vdb")

OUT = ROOT / "runs" / "iclr" / "fast_vitaldb"
NATIVE = ROOT / "data" / "processed_vitaldb" / "native"
MANIFEST = ROOT / "data" / "vitaldb_manifest.csv"
CLINICAL = ROOT / "data" / "raw" / "physionet.org" / "files" / "vitaldb" / "1.0.0" / "clinical_data.csv"
PROTOCOL = OUT / "protocol.md"
PROTOCOL_SHA_FILE = OUT / "protocol.sha256"

SEED = 20260925
EPOCH_SEC = 300.0
LAGS = np.arange(-30, 31)
N_BOOT = 2000
NUM_COLS = ["t_sec", "hr", "art_mbp", "bis", "bis_sqi", "ppf20_ce", "rftn20_ce", "rftn50_ce", "set_rr", "tv"]


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_protocol() -> str:
    h = sha256(PROTOCOL)
    reg = PROTOCOL_SHA_FILE.read_text().split()[0]
    if h != reg:
        raise SystemExit(f"protocol.md sha256 {h} != registered {reg}; refusing to run the analysis")
    return h


def blocks_from_beats(beats: pd.DataFrame, gap_sec: float = 60.0, min_sec: float = 600.0) -> list[tuple[float, float]]:
    """Wave-2 blocks (scripts/45_fast_coupling.py blocks_from_beats), re-implemented verbatim."""
    v = beats["valid"].to_numpy(bool)
    t = beats["t_r"].to_numpy(float)[v]
    if len(t) < 100:
        return []
    cut = np.where(np.diff(t) > gap_sec)[0]
    starts = np.concatenate([[t[0]], t[cut + 1]])
    ends = np.concatenate([t[cut], [t[-1]]])
    return [(float(a), float(b)) for a, b in zip(starts, ends) if b - a >= min_sec]


def highpass_bins(x: np.ndarray, step: float, width_sec: float = 30.0) -> np.ndarray:
    """Wave-2 30 s high-pass on binned series (scripts/45 highpass_bins), re-implemented verbatim."""
    k = max(int(round(width_sec / step)), 3)
    if k % 2 == 0:
        k += 1
    v = np.isfinite(x)
    xv = np.where(v, x, 0.0)
    c = np.convolve(xv, np.ones(k), mode="same")
    n = np.convolve(v.astype(float), np.ones(k), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = c / n
    return np.where(v & (n >= k // 2), x - m, np.nan)


def hp_1s(x: np.ndarray, k: int = 61, min_n: int = 20) -> np.ndarray:
    """Protocol section 5: minus a centred k-s moving mean over finite values, kept where >= min_n finite."""
    v = np.isfinite(x)
    c = np.convolve(np.where(v, x, 0.0), np.ones(k), mode="same")
    n = np.convolve(v.astype(float), np.ones(k), mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = c / n
    return np.where(v & (n >= min_n), x - m, np.nan)


def xcorr_lags(x: np.ndarray, y: np.ndarray, lags=LAGS, min_pairs: int = 300) -> np.ndarray:
    """Pearson r of x(t) with y(t + L) for each L (positive L = y later)."""
    out = np.full(len(lags), np.nan)
    n = len(x)
    for i, L in enumerate(lags):
        if L >= 0:
            a, b = x[: n - L], y[L:]
        else:
            a, b = x[-L:], y[: n + L]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_pairs:
            continue
        aa, bb = a[ok], b[ok]
        if aa.std() == 0 or bb.std() == 0:
            continue
        out[i] = float(np.corrcoef(aa, bb)[0, 1])
    return out


def peak_lag(c: np.ndarray, lags=LAGS) -> tuple[float, float]:
    if not np.isfinite(c).any():
        return np.nan, np.nan
    i = int(np.nanargmax(c))
    lag = float(lags[i])
    if 0 < i < len(c) - 1 and np.isfinite(c[i - 1]) and np.isfinite(c[i + 1]):
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        if den < 0:
            lag += float(np.clip(0.5 * (c[i - 1] - c[i + 1]) / den, -0.5, 0.5))
    return lag, float(c[i])


def sec_bins(t: np.ndarray, v: np.ndarray, T0: int, n: int) -> np.ndarray:
    """Mean of values in 1-s bins [T0 + k, T0 + k + 1)."""
    ok = np.isfinite(t) & np.isfinite(v)
    k = np.floor(t[ok]).astype(np.int64) - T0
    keep = (k >= 0) & (k < n)
    s = np.bincount(k[keep], weights=v[ok][keep], minlength=n)
    c = np.bincount(k[keep], minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out


def epoch_means(t: np.ndarray, x: np.ndarray, e0: np.ndarray, e1: np.ndarray) -> np.ndarray:
    """Mean of finite x at times t within each [e0, e1) (epochs sorted, non-overlapping)."""
    out = np.full(len(e0), np.nan)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if not len(t) or not len(e0):
        return out
    i = np.searchsorted(e0, t, side="right") - 1
    inside = (i >= 0) & (t < e1[np.clip(i, 0, len(e1) - 1)])
    i, x = i[inside], x[inside]
    s = np.bincount(i, weights=x, minlength=len(e0))
    c = np.bincount(i, minlength=len(e0)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out


def load_numerics(pid: str) -> pd.DataFrame:
    f = NATIVE / "numerics_1s" / f"{pid}.parquet"
    try:
        import pyarrow.parquet as pq
        have = set(pq.read_schema(f).names)
    except Exception:  # noqa: BLE001
        have = set(pd.read_parquet(f).columns)
    cols = [c for c in NUM_COLS if c in have]
    d = pd.read_parquet(f, columns=cols)
    for c in NUM_COLS:
        if c not in d:
            d[c] = np.nan
    return d


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage extract: QC, epochs (wave-4 epoch_stats + extras), monitor lag scan
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def qc_table() -> pd.DataFrame:
    rows = []
    for f in sorted((NATIVE / "summaries").glob("*.json")):
        d = json.loads(f.read_text())
        pid = d["patient_id"]
        rows.append({"patient_id": pid, "beats": d.get("beats"), "beats_valid_frac": d.get("beats_valid_frac"),
                     "hr_bias": d.get("hr_bias_beat_minus_solar_bpm"),
                     "art_map_offset": d.get("art_map_offset_solar_minus_wave_mmHg"),
                     "clock_warning": d.get("clock_warning"),
                     "beats_file": (NATIVE / "beats" / f"{pid}.parquet").exists()})
    q = pd.DataFrame(rows)
    q["g1_beats"] = q["beats_file"] & (q["beats"].fillna(0) > 0)
    q["g2_valid"] = q["g1_beats"] & (q["beats_valid_frac"].fillna(0) >= 0.5)
    q["g3_hrbias"] = q["g2_valid"] & (q["hr_bias"].abs() <= 5)
    return q


def process_case(args) -> dict:
    ci, pid = args
    t_start = time.time()
    meta: dict = {"patient_id": pid, "case_index": ci}
    try:
        beats = pd.read_parquet(NATIVE / "beats" / f"{pid}.parquet")
        v = beats["valid"].to_numpy(bool)
        rri = beats["rri"].to_numpy(float)
        meta["valid_sec"] = float(np.nansum(rri[v]) / 1000.0)
        if meta["valid_sec"] < 1200:
            meta["g4_20min"] = False
            return {"meta": meta, "epochs": [], "curves": []}
        meta["g4_20min"] = True
        blocks = blocks_from_beats(beats)
        meta["n_blocks"] = len(blocks)
        meta["block_sec"] = float(sum(b - a for a, b in blocks))
        num = load_numerics(pid)
        br = pd.read_parquet(NATIVE / "breaths" / f"{pid}.parquet") if (NATIVE / "breaths" / f"{pid}.parquet").exists() \
            else pd.DataFrame(columns=["t_exp_start", "ttot"])
        tr = beats["t_r"].to_numpy(float)
        # ---- epochs (wave-2 grid) ----
        grid = []
        for bi, (a, b) in enumerate(blocks):
            t = a
            j = 0
            while t + EPOCH_SEC <= b + 1e-6:
                grid.append((bi, j, t, t + EPOCH_SEC))
                t += EPOCH_SEC
                j += 1
        rows = []
        for ei, (bi, j, e0, e1) in enumerate(grid):
            lo, hi = np.searchsorted(tr, e0, "left"), np.searchsorted(tr, e1, "left")
            vv = v[lo:hi]
            base = {"patient_id": pid, "block": bi, "j": j, "t0": e0, "t1": e1, "epoch_index": ei,
                    "n_beats_all": int(hi - lo)}
            if len(vv) < 100 or vv.mean() < 0.8 or vv.sum() < 150:
                rows.append({**base, "usable": False})
                continue
            b = beats.iloc[lo:hi]
            rng = np.random.default_rng([SEED, ci, ei])
            st = s52.epoch_stats(b, rng)
            pc = pat_coupling(b["pat"].to_numpy(float), b["sbp"].to_numpy(float), b["valid"].to_numpy(bool))
            st["pat_median"] = pc["pat_median"]
            st["map_mean"] = float(np.nanmean(b["map_beat"].to_numpy(float)[b["valid"].to_numpy(bool)]))
            rows.append({**base, "usable": True, **st})
        ep = pd.DataFrame(rows)
        if len(ep):
            e0 = ep["t0"].to_numpy(float)
            e1 = ep["t1"].to_numpy(float)
            ts = num["t_sec"].to_numpy(float) + 0.5
            bis = num["bis"].to_numpy(float)
            bis = np.where((num["bis_sqi"].to_numpy(float) >= 50) & (bis > 0), bis, np.nan)
            ep["bis"] = epoch_means(ts, bis, e0, e1)
            ep["propofol_ce"] = epoch_means(ts, num["ppf20_ce"].to_numpy(float), e0, e1)
            r20 = epoch_means(ts, num["rftn20_ce"].to_numpy(float), e0, e1)
            r50 = epoch_means(ts, num["rftn50_ce"].to_numpy(float), e0, e1)
            ep["remifentanil_ce"] = np.where(np.isfinite(r20), r20, r50)
            ep["set_rr"] = epoch_means(ts, np.where(num["set_rr"].to_numpy(float) > 0, num["set_rr"].to_numpy(float),
                                                    np.nan), e0, e1)
            ep["tv"] = epoch_means(ts, np.where(num["tv"].to_numpy(float) > 0, num["tv"].to_numpy(float), np.nan), e0, e1)
            # breath-table ventilation descriptors
            tb = br["t_exp_start"].to_numpy(float) if len(br) else np.empty(0)
            tt = br["ttot"].to_numpy(float) if len(br) else np.empty(0)
            nbr, medt, cvt = [], [], []
            for a0, a1 in zip(e0, e1):
                lo, hi = np.searchsorted(tb, a0, "left"), np.searchsorted(tb, a1, "left")
                x = tt[lo:hi]
                x = x[np.isfinite(x)]
                nbr.append(len(x))
                medt.append(float(np.median(x)) if len(x) else np.nan)
                cvt.append(float(np.std(x) / np.mean(x)) if len(x) >= 2 and np.mean(x) > 0 else np.nan)
            ep["n_breaths_tab"] = nbr
            ep["ttot_median"] = medt
            ep["ttot_cv"] = cvt
            ep["br_rate"] = 60.0 / ep["ttot_median"]
            ep["controlled_vent"] = ((ep["n_breaths_tab"] >= 30) & (ep["ttot_cv"] <= 0.15) & (ep["set_rr"] > 0)
                                     & ((ep["br_rate"] - ep["set_rr"]).abs() <= 2))
        meta["n_epochs"] = int(len(ep))
        meta["n_usable"] = int(ep["usable"].sum()) if len(ep) else 0
        # ---- monitor lag scan (protocol section 5) ----
        curves = []
        if blocks:
            T0 = int(np.floor(blocks[0][0]))
            T1 = int(np.ceil(blocks[-1][1])) + 1
            n = T1 - T0
            inblk = np.zeros(n, dtype=bool)
            for a, b in blocks:
                inblk[max(int(np.floor(a)) - T0, 0): min(int(np.ceil(b)) - T0, n)] = True
            t_hr = tr + rri / 1000.0
            tsys = beats["t_sys"].to_numpy(float)
            bh = sec_bins(t_hr[v], 60000.0 / rri[v], T0, n)
            bm = sec_bins(tsys[v], beats["map_beat"].to_numpy(float)[v], T0, n)
            ts_i = num["t_sec"].to_numpy(float)
            mh = sec_bins(ts_i + 0.5, np.where(num["hr"].to_numpy(float) > 0, num["hr"].to_numpy(float), np.nan), T0, n)
            mm = sec_bins(ts_i + 0.5, np.where(num["art_mbp"].to_numpy(float) > 0, num["art_mbp"].to_numpy(float), np.nan),
                          T0, n)
            for arr in (bh, bm, mh, mm):
                arr[~inblk] = np.nan
            meta["mon_hr_cover"] = float(np.isfinite(mh[inblk]).mean())
            meta["mon_map_cover"] = float(np.isfinite(mm[inblk]).mean())
            for sig, xb, xm in (("hr", bh, mh), ("map", bm, mm)):
                for var, f in (("hp", hp_1s), ("raw", lambda z: z)):
                    c = xcorr_lags(f(xb), f(xm))
                    lag, pk = peak_lag(c)
                    meta[f"lag_{sig}_{var}"] = lag
                    meta[f"peak_{sig}_{var}"] = pk
                    meta[f"r0_{sig}_{var}"] = float(c[LAGS == 0][0]) if np.isfinite(c[LAGS == 0][0]) else np.nan
                    curves.append({"patient_id": pid, "signal": sig, "variant": var, **{f"L{L}": c[i] for i, L in enumerate(LAGS)}})
        meta["seconds"] = time.time() - t_start
        return {"meta": meta, "epochs": ep.to_dict("records") if len(ep) else [], "curves": curves}
    except Exception as ex:  # noqa: BLE001
        import traceback
        meta["error"] = f"{type(ex).__name__}: {ex} | {traceback.format_exc()[-400:]}"
        return {"meta": meta, "epochs": [], "curves": []}


def stage_extract(workers: int, limit: int | None = None) -> None:
    psha = check_protocol()
    t0 = time.time()
    q = qc_table()
    pids = q.loc[q["g3_hrbias"], "patient_id"].tolist()
    if limit:
        pids = pids[:limit]
    jobs = list(enumerate(pids))
    metas, eps, curves = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(process_case, jobs, chunksize=4)):
            metas.append(r["meta"])
            eps.extend(r["epochs"])
            curves.extend(r["curves"])
            if (i + 1) % 200 == 0:
                log.info("extract %d/%d cases (%.0f s)", i + 1, len(jobs), time.time() - t0)
    M = pd.DataFrame(metas)
    Q = q.merge(M, on="patient_id", how="left")
    sfx = "" if not limit else f"_limit{limit}"
    Q.to_parquet(OUT / f"cases{sfx}.parquet", index=False)
    pd.DataFrame(eps).to_parquet(OUT / f"epochs{sfx}.parquet", index=False)
    pd.DataFrame(curves).to_parquet(OUT / f"lag_curves{sfx}.parquet", index=False)
    (OUT / f"extract_meta{sfx}.json").write_text(json.dumps({
        "protocol_sha256": psha, "script_sha256": sha256(Path(__file__)),
        "s52_sha256": sha256(ROOT / "scripts" / "52_fast_edge_specificity.py"),
        "fast_coupling_sha256": sha256(ROOT / "src" / "clbench" / "edges" / "fast_coupling.py"),
        "beats_sha256": sha256(ROOT / "src" / "clbench" / "beats.py"),
        "seconds": time.time() - t0, "n_cases": len(jobs), "n_errors": int(M.get("error", pd.Series(dtype=str)).notna().sum()),
        "workers": workers, "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    log.info("extract done: %d cases, %d epoch rows, %.0f s", len(jobs), len(eps), time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage fits: 2 s lags-only MAP -> HR (a) monitor, (b) monitor HR shifted by the case's relative lag,
# (b2) shifted by the population median, (c) beat-derived
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def fit_case(args) -> dict:
    pid, rel_case, rel_pop = args
    out = {"patient_id": pid, "rel_used": np.nan}
    try:
        beats = pd.read_parquet(NATIVE / "beats" / f"{pid}.parquet")
        blocks = blocks_from_beats(beats)
        if not blocks:
            return out
        num = load_numerics(pid)
        v = beats["valid"].to_numpy(bool)
        rri = beats["rri"].to_numpy(float)
        tr = beats["t_r"].to_numpy(float)
        tsys = beats["t_sys"].to_numpy(float)
        t_mp = np.where(np.isfinite(tsys), tsys, tr + 0.3)
        beat_sec = float(np.nanmedian(rri[v])) / 1000.0
        tm = num["t_sec"].to_numpy(float) + 0.5
        fh = np.where(num["hr"].to_numpy(float) > 0, num["hr"].to_numpy(float), np.nan)
        fm = np.where(num["art_mbp"].to_numpy(float) > 0, num["art_mbp"].to_numpy(float), np.nan)
        rel = rel_case if np.isfinite(rel_case) else rel_pop
        out["rel_used"] = rel
        out["rel_is_case"] = bool(np.isfinite(rel_case))

        def mon(a, b, shift):
            return (bin_means(tm - shift, fh, np.isfinite(fh), a, b, 2.0, min_frac=0.0, beat_sec=2.0),
                    bin_means(tm, fm, np.isfinite(fm), a, b, 2.0, min_frac=0.0, beat_sec=2.0))

        variants = {
            "a_monitor": [mon(a, b, 0.0) for a, b in blocks],
            "b_monitor_shift_case": [mon(a, b, rel) for a, b in blocks] if np.isfinite(rel) else None,
            "b2_monitor_shift_pop": [mon(a, b, rel_pop) for a, b in blocks] if np.isfinite(rel_pop) else None,
            "c_beat": [(bin_means(tr + rri / 1000.0, 60000.0 / rri, v, a, b, 2.0, beat_sec=beat_sec),
                        bin_means(t_mp, beats["map_beat"].to_numpy(float), v, a, b, 2.0, beat_sec=beat_sec))
                       for a, b in blocks],
        }
        for name, bl in variants.items():
            if bl is None:
                out[f"{name}_raw"] = np.nan
                out[f"{name}_hp"] = np.nan
                continue
            r = lagged_timescale_fit(bl, n_lags=5)
            out[f"{name}_raw"] = r.total
            out[f"{name}_raw_se"] = r.se_total
            out[f"{name}_raw_n"] = r.n
            hp = [(highpass_bins(h, 2.0), highpass_bins(m, 2.0)) for h, m in bl]
            r = lagged_timescale_fit(hp, n_lags=5)
            out[f"{name}_hp"] = r.total
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def lag_inclusion(C: pd.DataFrame) -> pd.DataFrame:
    C = C.copy()
    C["inc_hr"] = C["peak_hr_hp"] >= 0.5
    C["inc_map"] = C["peak_map_hp"] >= 0.5
    C["rel_lag"] = np.where(C["inc_hr"] & C["inc_map"], C["lag_hr_hp"] - C["lag_map_hp"], np.nan)
    return C


def stage_fits(workers: int) -> None:
    check_protocol()
    t0 = time.time()
    C = pd.read_parquet(OUT / "cases.parquet")
    C = C[C["g4_20min"] == True].copy()  # noqa: E712
    C = lag_inclusion(C)
    rel_pop = float(np.nanmedian(C["rel_lag"]))
    jobs = [(p, r, rel_pop) for p, r in zip(C["patient_id"], C["rel_lag"])]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(fit_case, jobs, chunksize=8))
    F = pd.DataFrame(rows)
    F.to_parquet(OUT / "fits_2s.parquet", index=False)
    (OUT / "fits_meta.json").write_text(json.dumps({"rel_pop_median_s": rel_pop, "seconds": time.time() - t0,
                                                    "n_cases": len(jobs)}, indent=2))
    log.info("fits done: %d cases, rel_pop %.2f s, %.0f s", len(jobs), rel_pop, time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage specificity: the wave-4 registered test on all QC cases
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

PRIMARY = s52.PRIMARY
SECONDARY = s52.SECONDARY
TWIN = s52.TWIN
COV_HS = s52.COV_HS
COV_DRUG = s52.COV_DRUG
OPERATING = {"O-hr": ("hr_mean", "mean HR (operating point)", "bpm"),
             "O-sbp": ("sbp_mean", "mean SBP (operating point)", "mmHg")}


def prepare_vdb(controlled_only: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(OUT / "epochs.parquet")
    cases = pd.read_parquet(OUT / "cases.parquet")
    keep = set(cases.loc[cases["g4_20min"] == True, "patient_id"])  # noqa: E712
    df = df[df["usable"] & df["patient_id"].isin(keep)].copy()
    if controlled_only:
        df = df[df["controlled_vent"] == True]  # noqa: E712
    df["dataset"] = "vitaldb"
    df["remifentanil_ce0"] = df["remifentanil_ce"].fillna(0.0).astype(float)
    df["propofol_rec"] = df["propofol_ce"].notna().astype(float)
    df["propofol_ce0"] = df["propofol_ce"].fillna(0.0).astype(float)
    cols = [c for c, *_ in PRIMARY.values()] + [c for c, *_ in SECONDARY.values()] + list(TWIN.values()) + COV_HS + \
        ["remifentanil_ce0", "propofol_ce0"]
    return s52.winsorise(df, cols)


def fast_groups(df: pd.DataFrame, col: str, covs: list[str], split: str, rng: np.random.Generator | None = None):
    """Vectorised equivalent of s52.build_groups (same eligibility, halves and early-fitted residualisation)."""
    need = [col] + covs
    d = df[np.all(np.isfinite(df[need].to_numpy(float)), axis=1)].sort_values(["patient_id", "t0"])
    pid = d["patient_id"].to_numpy()
    n = len(d)
    if n == 0:
        return [], []
    g = pd.Series(pid)
    cnt = g.map(g.value_counts()).to_numpy()
    pos = g.groupby(g).cumcount().to_numpy()
    if split == "S3":
        key = rng.random(n)
        pos = pd.DataFrame({"p": pid, "k": key}).groupby("p")["k"].rank(method="first").to_numpy().astype(int) - 1
    half = np.where(pos < cnt // 2, 0, 1)
    if split == "S4":
        pos_in_half = np.where(half == 0, pos, pos - cnt // 2)
        half = np.where(pos_in_half < 3, half, -1)
    if split == "S2":
        j = d["j"].to_numpy(int)
        half = np.full(n, -1)
        half[j % 4 == 0] = 0
        half[j % 4 == 2] = 1
    ne = pd.Series(half == 0).groupby(pid).transform("sum").to_numpy()
    nl = pd.Series(half == 1).groupby(pid).transform("sum").to_numpy()
    ok = (half >= 0) & (ne >= s52.MIN_PER_HALF) & (nl >= s52.MIN_PER_HALF)
    d = d[ok]
    half = half[ok]
    y = d[col].to_numpy(float)
    if covs:
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in covs])
        early = half == 0
        beta, *_ = np.linalg.lstsq(X[early], y[early], rcond=None)
        y = y - X @ beta
    p = d["patient_id"].to_numpy()
    order = np.lexsort((half, p))  # stable within half keeps time order for S1/S2/S4
    p, half, y = p[order], half[order], y[order]
    upids, starts = np.unique(p, return_index=True)
    ends = np.append(starts[1:], len(p))
    groups = []
    for s, e in zip(starts, ends):
        h = half[s:e]
        yy = y[s:e]
        groups.append((yy[h == 0], yy[h == 1]))
    return groups, list(upids)


def _spec_task(args) -> list[dict]:
    kind, eid, col, lab, unit, adj, covs, splits, controlled_only = args
    df = prepare_vdb(controlled_only)
    rows, pat_rows = [], []
    for split in splits:
        if split == "S3":
            rng = np.random.default_rng(7)
            draws = [s52.split_stats(fast_groups(df, col, covs, "S3", rng)[0], full=False) for _ in range(s52.N_SHUFFLE)]
            D = pd.DataFrame(draws)
            r = {"edge_id": eid, "column": col, "label": lab, "unit": unit, "kind": kind, "adjust": adj, "split": "S3",
                 "controlled_only": controlled_only, "n_patients": int(D["n_patients"].median())}
            for k in ("r_early_late", "rel_shrunk_vs_pooled", "d_shrunk_minus_pooled", "icc1", "half_mse_ratio"):
                if k in D:
                    r[k] = float(D[k].median())
                    r[k + "_lo"] = float(D[k].quantile(0.025))
                    r[k + "_hi"] = float(D[k].quantile(0.975))
            rows.append(r)
            continue
        groups, pids = s52.build_groups(df, col, covs, split)
        res = s52.split_stats(groups, seed=0)
        rows.append({"edge_id": eid, "column": col, "label": lab, "unit": unit, "kind": kind, "adjust": adj,
                     "split": split, "controlled_only": controlled_only, **res})
        if split == "S1":
            fg, fp = fast_groups(df, col, covs, "S1")
            same = (fp == pids) and all(np.allclose(a, c) and np.allclose(b, d_) for (a, b), (c, d_) in zip(groups, fg))
            rows[-1]["fast_groups_equal_s52"] = bool(same)
            for pp, (a, b) in zip(pids, groups):
                pat_rows.append({"edge_id": eid, "column": col, "adjust": adj, "controlled_only": controlled_only,
                                 "patient_id": pp, "n_early": len(a), "n_late": len(b), "early_mean": a.mean(),
                                 "late_mean": b.mean()})
    return [{"_rows": rows, "_pat": pat_rows}]


def stage_specificity(workers: int) -> None:
    psha = check_protocol()
    t0 = time.time()
    tasks = []
    for eid, (col, lab, unit) in PRIMARY.items():
        adjs = {"raw": [], "adj_hs": COV_HS, "adj_hsd": COV_HS + COV_DRUG}
        if col in TWIN:
            adjs["adj_hst"] = COV_HS + [TWIN[col]]
        for adj, covs in adjs.items():
            tasks.append(("primary", eid, col, lab, unit, adj, covs, ("S1", "S2", "S4", "S3"), False))
    for eid, (col, lab, unit) in SECONDARY.items():
        adjs = {"raw": [], "adj_hs": COV_HS, "adj_hsd": COV_HS + COV_DRUG}
        for adj, covs in adjs.items():
            tasks.append(("secondary", eid, col, lab, unit, adj, covs, ("S1",), False))
    for col, tw in TWIN.items():
        tasks.append(("twin", f"T-{col}", tw, f"twin of {col}", "", "raw", [], ("S1",), False))
    for eid, (col, lab, unit) in OPERATING.items():
        tasks.append(("operating", eid, col, lab, unit, "raw", [], ("S1",), False))
    for eid in ("E2", "E3"):  # secondary: controlled-ventilation epochs only
        col, lab, unit = PRIMARY[eid]
        for adj, covs in {"raw": [], "adj_hs": COV_HS}.items():
            tasks.append(("primary_cv", eid, col, lab + " [controlled ventilation]", unit, adj, covs, ("S1",), True))
    # longest first
    tasks.sort(key=lambda t: -len(t[7]))
    rows, pats = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_spec_task, tasks)):
            rows.extend(res[0]["_rows"])
            pats.extend(res[0]["_pat"])
            log.info("spec task %d/%d done (%.0f s)", i + 1, len(tasks), time.time() - t0)
    R = pd.DataFrame(rows)
    R["ci_d_below0"] = R["d_shrunk_minus_pooled_hi"] < 0
    m = (R["kind"] == "primary") & (R["adjust"] == "adj_hs") & (R["split"] == "S1")
    p = R.loc[m, "perm_p_r"].to_numpy(float)
    order = np.argsort(p)
    holm = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        holm[i] = running
    R["perm_p_r_holm_adj_hs"] = np.nan
    R.loc[m, "perm_p_r_holm_adj_hs"] = holm
    R.to_csv(OUT / "specificity_all.csv", index=False)
    P = pd.DataFrame(pats)
    P.to_csv(OUT / "specificity_patient_halves.csv", index=False)
    # paired bootstrap: r(edge, adj_hs) - r(hr_mean, raw) on common patients (and vs sbp_mean)
    comp = []
    hm = P[(P["edge_id"].isin(["O-hr", "O-sbp"])) & (~P["controlled_only"])]
    for eid in list(PRIMARY) + list(SECONDARY):
        for adj in ("raw", "adj_hs"):
            e = P[(P["edge_id"] == eid) & (P["adjust"] == adj) & (~P["controlled_only"])]
            for oid in ("O-hr", "O-sbp"):
                o = hm[hm["edge_id"] == oid]
                j = e.merge(o, on="patient_id", suffixes=("_e", "_o"))
                if len(j) < 10:
                    continue
                E1, L1 = j["early_mean_e"].to_numpy(), j["late_mean_e"].to_numpy()
                E2, L2 = j["early_mean_o"].to_numpy(), j["late_mean_o"].to_numpy()
                est = s52.wpearson(E1, L1, np.ones(len(j))) - s52.wpearson(E2, L2, np.ones(len(j)))
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(N_BOOT):
                    w = np.bincount(rng.integers(0, len(j), len(j)), minlength=len(j)).astype(float)
                    bs.append(s52.wpearson(E1, L1, w) - s52.wpearson(E2, L2, w))
                bs = np.asarray(bs)
                comp.append({"edge_id": eid, "adjust": adj, "vs": oid, "n_common": len(j),
                             "r_edge": s52.wpearson(E1, L1, np.ones(len(j))), "r_oper": s52.wpearson(E2, L2, np.ones(len(j))),
                             "diff": est, "diff_lo": float(np.nanquantile(bs, 0.025)), "diff_hi": float(np.nanquantile(bs, 0.975))})
    pd.DataFrame(comp).to_csv(OUT / "specificity_vs_operating_point.csv", index=False)
    (OUT / "specificity_meta.json").write_text(json.dumps({
        "protocol_sha256": psha, "wave4_protocol_sha256": s52.sha256(s52.PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "s52_sha256": sha256(ROOT / "scripts" / "52_fast_edge_specificity.py"),
        "n_boot": s52.N_BOOT, "n_perm": s52.N_PERM, "n_shuffle": s52.N_SHUFFLE, "seconds": time.time() - t0,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    log.info("specificity done %.0f s", time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage report
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def boot_cases(vals: np.ndarray, stat, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Case bootstrap of a statistic of one value per case."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    est = stat(vals)
    bs = np.array([stat(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)])
    return float(est), float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))


def boot_cluster(groups: list[np.ndarray], stat, n_boot: int = N_BOOT, seed: int = 0):
    """Case-cluster bootstrap: resample cases, pool their epoch values, apply stat."""
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if len(groups) < 3:
        return np.nan, np.nan, np.nan
    lens = np.array([len(g) for g in groups])
    flat = np.concatenate(groups)
    offs = np.concatenate([[0], np.cumsum(lens)])
    est = stat(flat)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in pick])
        bs.append(stat(flat[idx]))
    return float(est), float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))


def fmt(t, pct=False, nd=2):
    e, lo, hi = t
    if not np.isfinite(e):
        return "n/a"
    if pct:
        return f"{100 * e:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
    return f"{e:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def spearman_boot(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return (np.nan, np.nan, np.nan), len(x)
    est = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        k = rng.integers(0, len(x), len(x))
        bs.append(spearmanr(x[k], y[k]).statistic)
    return (float(est), float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))), len(x)


def within_slope(d: pd.DataFrame, xcol: str, ycol: str, min_sd: float, log_y: bool = True):
    d = d[["patient_id", xcol, ycol]].dropna()
    if log_y:
        d = d[d[ycol] > 0]
        d = d.assign(y=np.log(d[ycol]))
    else:
        d = d.assign(y=d[ycol])
    sd = d.groupby("patient_id")[xcol].std()
    d = d[d["patient_id"].isin(sd[sd >= min_sd].index)]
    if d["patient_id"].nunique() < 10:
        return (np.nan, np.nan, np.nan), 0, 0
    d = d.assign(xd=d[xcol] - d.groupby("patient_id")[xcol].transform("mean"),
                 yd=d["y"] - d.groupby("patient_id")["y"].transform("mean"))
    g = d.groupby("patient_id").apply(lambda z: pd.Series({"sxy": float(np.sum(z.xd * z.yd)), "sxx": float(np.sum(z.xd ** 2))}))
    sxy, sxx = g["sxy"].to_numpy(), g["sxx"].to_numpy()
    est = sxy.sum() / sxx.sum()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(sxy), len(sxy))
        bs.append(sxy[k].sum() / sxx[k].sum())
    return (float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))), len(sxy), len(d)


def verdict(t, expected_sign: int) -> str:
    e, lo, hi = t
    if not np.isfinite(e):
        return "not assessable"
    if expected_sign > 0:
        return "holds" if lo > 0 else ("fails (reversed)" if hi < 0 else "not shown")
    return "holds" if hi < 0 else ("fails (reversed)" if lo > 0 else "not shown")


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    from scipy.stats import binomtest
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else np.nan
    return n10, n01, float(p)


def stage_report() -> None:
    psha = check_protocol()
    C = pd.read_parquet(OUT / "cases.parquet")
    E = pd.read_parquet(OUT / "epochs.parquet")
    man = pd.read_csv(MANIFEST).set_index("patient_id")
    clin = pd.read_csv(CLINICAL)
    clin["patient_id"] = clin["caseid"].map(lambda c: f"vdb{int(c):04d}")
    clin = clin.set_index("patient_id")
    lines = [f"# Track B final wave: fast edges at VitalDB scale (protocol sha256 {psha[:8]})", ""]
    summ: dict = {"protocol_sha256": psha}

    # ---- QC gates ----
    n_all = len(C)
    gates = {"summaries": n_all, "beats file present": int(C["g1_beats"].sum()), "valid frac >= 0.5": int(C["g2_valid"].sum()),
             "|HR bias| <= 5": int(C["g3_hrbias"].sum()), ">= 20 min valid beats": int((C["g4_20min"] == True).sum()),  # noqa: E712
             "errors": int(C["error"].notna().sum()) if "error" in C else 0}
    Q = C[C["g4_20min"] == True].copy()  # noqa: E712
    Eu = E[E["usable"] & E["patient_id"].isin(Q["patient_id"])].copy()
    gates["cases with >= 1 usable epoch"] = int(Eu["patient_id"].nunique())
    gates["usable epochs"] = int(len(Eu))
    gates["controlled-ventilation epochs"] = int(Eu["controlled_vent"].sum())
    gates["epochs with PAT slope (>= 30 valid PAT beats)"] = int(np.isfinite(Eu["pat_sbp_slope"]).sum())
    summ["qc"] = gates
    lines += ["## QC gates", "", md(pd.DataFrame({"gate": list(gates), "n": list(gates.values())})), ""]

    # ---- population results ----
    pop = []
    by = {p: g for p, g in Eu.groupby("patient_id")}
    Ecv = Eu[Eu["controlled_vent"] == True]  # noqa: E712
    bycv = {p: g for p, g in Ecv.groupby("patient_id")}

    def signrow(label, col, sign, src):
        groups = [g[col].to_numpy(float) for g in src.values()]
        med = np.array([np.nanmedian(g) if np.isfinite(g).any() else np.nan for g in groups])
        ok = np.isfinite(med)
        pop.append({"quantity": label, "expected": "+" if sign > 0 else "-", "n_cases": int(ok.sum()),
                    "n_epochs": int(sum(np.isfinite(g).sum() for g in groups)),
                    "cases with expected sign": fmt(boot_cases(med[ok], lambda a: np.mean(np.sign(a) == sign)), pct=True),
                    "epochs with expected sign (cluster CI)": fmt(boot_cluster(groups, lambda a: np.mean(np.sign(a) == sign)), pct=True),
                    "median of case medians": fmt(boot_cases(med[ok], np.median), nd=3),
                    "IQR of case medians": f"{np.nanpercentile(med, 25):.3f} to {np.nanpercentile(med, 75):.3f}" if ok.any() else "n/a"})
        return med

    signrow("PAT-SBP slope (ms/mmHg), censored native PAT", "pat_sbp_slope", -1, by)
    signrow("PAT-SBP beat correlation, censored native PAT", "pat_sbp_corr", -1, by)
    signrow("baroreflex ARX SBP(n..n-2) -> RRI(n) (ms/mmHg), wave-2 primary", "reg_brs", +1, by)
    pop_df = pd.DataFrame(pop)
    dist = []
    Eu = Eu.assign(tv_l=np.where(Eu["tv"] > 20, Eu["tv"] / 1000.0, Eu["tv"]))
    Eu["rsa_gain"] = Eu["rsa_h"] / Eu["tv_l"]
    Ecv = Eu[Eu["controlled_vent"] == True]  # noqa: E712
    for col, label, src in (("ppv", "PPV (%), controlled ventilation", Ecv), ("spv", "SPV (mmHg), controlled ventilation", Ecv),
                            ("rsa_h", "RSA harmonic amplitude (ms), controlled ventilation", Ecv),
                            ("rsa_amp", "RSA phase-binned amplitude (ms), controlled ventilation", Ecv),
                            ("rsa_gain", "RSA gain (ms per litre tidal volume), controlled ventilation", Ecv),
                            ("seq_pooled_brs", "sequence BRS (ms/mmHg), all usable epochs", Eu),
                            ("reg_brs", "ARX BRS (ms/mmHg), all usable epochs", Eu),
                            ("pat_median", "PAT R -> PLETH foot (ms), censored at 600 ms, includes device delay", Eu),
                            ("hr_mean", "HR (bpm)", Eu), ("sbp_mean", "SBP (mmHg)", Eu)):
        med = src.groupby("patient_id")[col].median().to_numpy(float)
        ok = np.isfinite(med)
        dist.append({"quantity": label, "n_cases": int(ok.sum()), "n_epochs": int(np.isfinite(src[col]).sum()),
                     "median of case medians": fmt(boot_cases(med[ok], np.median)),
                     "IQR": f"{np.nanpercentile(med, 25):.2f} to {np.nanpercentile(med, 75):.2f}" if ok.any() else "n/a",
                     "5-95%": f"{np.nanpercentile(med, 5):.2f} to {np.nanpercentile(med, 95):.2f}" if ok.any() else "n/a"})
    dist_df = pd.DataFrame(dist)
    pop_df.to_csv(OUT / "population_signs.csv", index=False)
    dist_df.to_csv(OUT / "population_distributions.csv", index=False)
    lines += ["## Population sign shares (case-cluster bootstrap, 2,000 replicates)", "", md(pop_df), "",
              "## Distributions of case medians", "", md(dist_df), ""]

    # ---- construct validity ----
    age = pd.to_numeric(man["age"], errors="coerce")
    cv = []
    case_brs = Eu.groupby("patient_id")["seq_pooled_brs"].median()
    case_rsa = Ecv.groupby("patient_id")["rsa_h"].median()
    case_pat = Eu.groupby("patient_id")["pat_median"].median()
    for cid, label, s, sign in (("CV1", "age vs case-median sequence BRS (adults)", case_brs, -1),
                                ("CV2", "age vs case-median RSA rsa_h, controlled ventilation (adults)", case_rsa, -1),
                                ("CV7", "age vs case-median PAT (adults; censored native PAT, declared not assessable)", case_pat, -1)):
        a = age.reindex(s.index).to_numpy(float)
        ok = a >= 18
        t, n = spearman_boot(a[ok], s.to_numpy(float)[ok])
        cv.append({"id": cid, "relation": label, "n": n, "estimate [95% CI]": f"Spearman rho {fmt(t)}",
                   "expected": "< 0", "verdict": verdict(t, sign) if cid != "CV7" else f"not assessable (censored); CI reads: {verdict(t, sign)}"})
    t, nc, ne = within_slope(Eu, "propofol_ce", "seq_pooled_brs", 0.1)
    cv.append({"id": "CV3", "relation": "within-case log sequence BRS vs propofol Ce (per ug/ml)", "n": f"{nc} cases, {ne} epochs",
               "estimate [95% CI]": f"slope {fmt(t, nd=4)}", "expected": "< 0", "verdict": verdict(t, -1)})
    t, nc, ne = within_slope(Eu, "bis", "seq_pooled_brs", 3.0)
    cv.append({"id": "CV4", "relation": "within-case log sequence BRS vs BIS (per BIS unit)", "n": f"{nc} cases, {ne} epochs",
               "estimate [95% CI]": f"slope {fmt(t, nd=4)}", "expected": "> 0", "verdict": verdict(t, +1)})
    # CV5 vasopressors
    vp = ((clin[["intraop_phe", "intraop_eph", "intraop_epi"]].fillna(0) > 0).any(axis=1))
    pumps = man["vasoactive_pumps"].fillna("").astype(str)
    vp_pump = pumps.str.contains("PHEN|NEPI|EPI|VASO|DOPA")
    case_ppv = Ecv.groupby("patient_id")["ppv"].median()
    vpc = (vp.reindex(case_ppv.index).fillna(False) | vp_pump.reindex(case_ppv.index).fillna(False)).to_numpy(bool)
    x1, x0 = case_ppv.to_numpy(float)[vpc], case_ppv.to_numpy(float)[~vpc]
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    rng = np.random.default_rng(0)
    bs = [np.median(x1[rng.integers(0, len(x1), len(x1))]) - np.median(x0[rng.integers(0, len(x0), len(x0))]) for _ in range(N_BOOT)]
    t = (float(np.median(x1) - np.median(x0)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    cv.append({"id": "CV5", "relation": "median case PPV: vasopressor cases minus none (percentage points)",
               "n": f"{len(x1)} vs {len(x0)} cases", "estimate [95% CI]": f"{fmt(t)} (medians {np.median(x1):.2f} vs {np.median(x0):.2f})",
               "expected": "> 0", "verdict": verdict(t, +1)})
    # CV6 early post-induction vs later
    ane = clin["anestart"].reindex(Ecv["patient_id"].unique())
    d6 = Ecv[["patient_id", "t0", "ppv"]].dropna().copy()
    d6["ane"] = d6["patient_id"].map(ane)
    d6["rel"] = d6["t0"] - d6["ane"]
    early = d6[(d6["rel"] >= 0) & (d6["rel"] < 1800)].groupby("patient_id")["ppv"].median()
    late = d6[d6["rel"] >= 3600].groupby("patient_id")["ppv"].median()
    diff = (early - late).dropna()
    t = boot_cases(diff.to_numpy(float), np.median)
    cv.append({"id": "CV6", "relation": "within-case median PPV, first 30 min after induction minus >= 60 min (pp)",
               "n": f"{len(diff)} cases", "estimate [95% CI]": f"median difference {fmt(t)}; share of cases > 0 "
               f"{100 * np.mean(diff > 0):.0f}%", "expected": "> 0", "verdict": verdict(t, +1)})
    cv_df = pd.DataFrame(cv).sort_values("id")
    cv_df.to_csv(OUT / "construct_validity.csv", index=False)
    lines += ["## Construct validity (pre-registered directions)", "", md(cv_df), ""]
    summ["construct_validity"] = cv_df.to_dict(orient="records")

    # ---- deviations / post hoc (labelled; not part of the registered verdicts) ----
    dev = []
    bad = Eu["propofol_ce"] > 20
    t, nc, ne = within_slope(Eu[~bad], "propofol_ce", "seq_pooled_brs", 0.1)
    dev.append({"item": "CV3 sensitivity (DEVIATION): drop physiologically impossible propofol Ce > 20 ug/ml",
                "n": f"{int(bad.sum())} epoch(s) in {Eu.loc[bad, 'patient_id'].nunique()} case(s) dropped; {nc} cases, {ne} epochs",
                "estimate [95% CI]": f"slope {fmt(t, nd=4)} per ug/ml", "verdict": verdict(t, -1),
                "why": "the registered pooled within-case slope is dominated by one epoch with Ce = "
                       f"{Eu['propofol_ce'].max():.0f} ug/ml (registered estimate {cv_df.set_index('id').loc['CV3', 'estimate [95% CI]']})"})
    t, nc, ne = within_slope(Eu[~bad], "propofol_ce", "reg_brs", 0.1, log_y=False)
    dev.append({"item": "CV3 with the ARX BRS (post hoc, same exclusion)", "n": f"{nc} cases, {ne} epochs",
                "estimate [95% CI]": f"slope {fmt(t, nd=4)} ms/mmHg per ug/ml", "verdict": verdict(t, -1), "why": "secondary BRS estimator"})
    from scipy.stats import rankdata
    g = Eu.groupby("patient_id")[["seq_pooled_brs", "hr_mean", "sbp_mean"]].median()
    g["age"] = age.reindex(g.index)
    g = g[g["age"] >= 18].dropna()

    def partial(df):
        X = np.column_stack([np.ones(len(df)), rankdata(df["hr_mean"]), rankdata(df["sbp_mean"])])
        rx = rankdata(df["age"]) - X @ np.linalg.lstsq(X, rankdata(df["age"]), rcond=None)[0]
        ry = rankdata(df["seq_pooled_brs"]) - X @ np.linalg.lstsq(X, rankdata(df["seq_pooled_brs"]), rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])

    rng = np.random.default_rng(0)
    bs = [partial(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(N_BOOT)]
    t = (partial(g), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    dev.append({"item": "CV1 post hoc: age vs BRS partial Spearman given case HR and SBP", "n": len(g),
                "estimate [95% CI]": fmt(t), "verdict": verdict(t, -1),
                "why": "sequence BRS (ms/mmHg) scales with RRI; tests whether the reversal is an HR artefact"})
    ok = (Q["peak_hr_raw"] >= 0.5) & (Q["peak_map_raw"] >= 0.5)
    rel_raw = (Q["lag_hr_raw"] - Q["lag_map_raw"])[ok].to_numpy(float)
    dev.append({"item": "relative lag HR - MAP on raw series (secondary variant of section 5)", "n": int(ok.sum()),
                "estimate [95% CI]": f"median {fmt(boot_cases(rel_raw, np.median))} s; IQR {np.percentile(rel_raw, 25):.2f} to "
                                     f"{np.percentile(rel_raw, 75):.2f}", "verdict": "-",
                "why": "the primary (high-passed) inclusion rule keeps only 342 cases"})
    dev_df = pd.DataFrame(dev)
    dev_df.to_csv(OUT / "deviations_posthoc.csv", index=False)
    lines += ["## Deviations and post hoc analyses (labelled; registered verdicts above are unchanged)", "", md(dev_df), ""]
    summ["deviations_posthoc"] = dev_df.to_dict(orient="records")
    if (OUT / "pat_rescue_exploratory.csv").exists():
        pr = pd.read_csv(OUT / "pat_rescue_exploratory.csv")
        lines += ["## EXPLORATORY (not pre-registered): PAT with the PLETH delay handled, local complete .vital files", "",
                  "Feet assigned to the R-peak nearest R + D (D = modal foot - R delay per case, +-0.15 s).", "",
                  md(pr.round(3)), ""]
        summ["pat_rescue"] = pr.to_dict(orient="records")

    # ---- monitor latency ----
    L = lag_inclusion(Q)
    lat = []
    for col, label, inc in (("lag_hr_hp", "HR lag (monitor minus beat), s", "inc_hr"),
                            ("lag_map_hp", "MAP lag (monitor minus beat), s", "inc_map"),
                            ("rel_lag", "relative lag HR minus MAP, s", None),
                            ("lag_hr_raw", "HR lag, raw series (secondary), s", None),
                            ("lag_map_raw", "MAP lag, raw series (secondary), s", None)):
        x = L[col].to_numpy(float)
        if inc:
            x = np.where(L[inc], x, np.nan)
        if col.endswith("_raw"):
            pk = L[col.replace("lag_", "peak_")].to_numpy(float)
            x = np.where(pk >= 0.5, x, np.nan)
        ok = np.isfinite(x)
        lat.append({"quantity": label, "n_cases": int(ok.sum()), "median [95% CI]": fmt(boot_cases(x[ok], np.median)),
                    "IQR": f"{np.nanpercentile(x, 25):.2f} to {np.nanpercentile(x, 75):.2f}",
                    "5-95%": f"{np.nanpercentile(x, 5):.2f} to {np.nanpercentile(x, 95):.2f}",
                    "share > 0": f"{100 * np.mean(x[ok] > 0):.0f}%"})
    lat_df = pd.DataFrame(lat)
    extra = {"cases scanned": int(L["peak_hr_hp"].notna().sum()), "HR peak r >= 0.5": int(L["inc_hr"].sum()),
             "MAP peak r >= 0.5": int(L["inc_map"].sum()), "both": int(L["rel_lag"].notna().sum()),
             "median HR peak r (hp)": float(np.nanmedian(L["peak_hr_hp"])), "median MAP peak r (hp)": float(np.nanmedian(L["peak_map_hp"])),
             "median monitor HR coverage (share of in-block seconds)": float(np.nanmedian(L["mon_hr_cover"])),
             "median monitor MAP coverage": float(np.nanmedian(L["mon_map_cover"]))}
    lat_df.to_csv(OUT / "monitor_latency.csv", index=False)
    lines += ["## Monitor display latency (1-s cross-correlation, 61-s high-pass; positive = monitor later)", "",
              md(lat_df), "", "Counts: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in extra.items()}), ""]
    summ["latency"] = {"table": lat_df.to_dict(orient="records"), **extra}

    # ---- 2 s fits ----
    if (OUT / "fits_2s.parquet").exists():
        F = pd.read_parquet(OUT / "fits_2s.parquet")
        fm = json.loads((OUT / "fits_meta.json").read_text())
        fit_rows, mc = [], []
        for var in ("raw", "hp"):
            cols = [f"a_monitor_{var}", f"b_monitor_shift_case_{var}", f"c_beat_{var}"]
            S = F.dropna(subset=cols)
            for c, lab in zip(cols + [f"b2_monitor_shift_pop_{var}"],
                              ["(a) monitor numerics as is", "(b) monitor, HR moved by the case's relative lag",
                               "(c) beat-derived 2 s series", "(b2) monitor, HR moved by the population median lag"]):
                x = S[c].to_numpy(float)
                ok = np.isfinite(x)
                fit_rows.append({"series": var, "variant": lab, "n_cases": int(ok.sum()),
                                 "negative (expected) sign": fmt(boot_cases(x[ok], lambda a: np.mean(a < 0)), pct=True),
                                 "median coefficient (bpm/mmHg)": fmt(boot_cases(x[ok], np.median), nd=4)})
            neg = {c: (S[c].to_numpy(float) < 0) for c in cols}
            for x_, y_ in ((0, 1), (0, 2), (1, 2)):
                n10, n01, p = mcnemar(neg[cols[y_]], neg[cols[x_]])
                mc.append({"series": var, "comparison": f"{cols[y_]} vs {cols[x_]}", "n": len(S),
                           "right only in first / right only in second": f"{n10} / {n01}", "McNemar p": f"{p:.3g}"})
        fit_df, mc_df = pd.DataFrame(fit_rows), pd.DataFrame(mc)
        fit_df.to_csv(OUT / "fits_2s_table.csv", index=False)
        mc_df.to_csv(OUT / "fits_2s_mcnemar.csv", index=False)
        lines += ["## 2 s lags-only MAP -> HR fit (5 lags; expected negative), same cases",
                  f"Population median relative lag used for (b) fallback and (b2): {fm['rel_pop_median_s']:.2f} s", "",
                  md(fit_df), "", md(mc_df), ""]
        summ["fits_2s"] = {"table": fit_df.to_dict(orient="records"), "mcnemar": mc_df.to_dict(orient="records"),
                           "rel_pop_median_s": fm["rel_pop_median_s"]}
    if (OUT / "emulation_exploratory.parquet").exists():
        D = pd.read_parquet(OUT / "emulation_exploratory.parquet")
        em = []
        for c in EMU:
            x = D[c].to_numpy(float)
            x = x[np.isfinite(x)]
            em.append({"scenario (beat-derived series)": c, "n_cases": len(x),
                       "negative (expected) sign": fmt(boot_cases(x, lambda a: np.mean(a < 0)), pct=True),
                       "median coefficient": fmt(boot_cases(x, np.median), nd=4)})
        em_df = pd.DataFrame(em)
        em_df.to_csv(OUT / "emulation_exploratory_table.csv", index=False)
        lines += ["## EXPLORATORY (not pre-registered): monitor-processing emulation on beat-derived series, 2 s fit", "",
                  md(em_df), ""]
        summ["emulation_exploratory"] = em_df.to_dict(orient="records")

    # ---- specificity ----
    if (OUT / "specificity_all.csv").exists():
        R = pd.read_csv(OUT / "specificity_all.csv")
        dec = []
        for eid, (col, lab, unit) in PRIMARY.items():
            g = R[(R["edge_id"] == eid) & (R["kind"] == "primary")]

            def get(adj, split="S1"):
                x = g[(g["adjust"] == adj) & (g["split"] == split)]
                return x.iloc[0] if len(x) else None

            raw, hs, hst, hsd = get("raw"), get("adj_hs"), get("adj_hst"), get("adj_hsd")
            s2 = get("adj_hs", "S2")
            p1 = bool(raw is not None and raw.get("ci_d_below0", False))
            p2 = bool(hs is not None and hs.get("ci_d_below0", False))
            dec.append({"edge": f"{eid} {lab} ({unit})", "n": int(raw["n_patients"]) if raw is not None else 0,
                        "P1 raw": "yes" if p1 else "no", "P2 HR/SBP-adj": "yes" if p2 else "no",
                        "REAL": "yes" if (p1 and p2) else "not shown",
                        "twin-adj (H3)": ("yes" if hst is not None and hst.get("ci_d_below0", False) else ("-" if hst is None else "no")),
                        "drug-adj": "yes" if hsd is not None and hsd.get("ci_d_below0", False) else "no",
                        "S2 adj holds": "yes" if s2 is not None and s2.get("ci_d_below0", False) else "no",
                        "perm p (r, adj)": f"{hs['perm_p_r']:.3g}" if hs is not None else "n/a",
                        "Holm p": f"{hs['perm_p_r_holm_adj_hs']:.3g}" if hs is not None else "n/a",
                        "fast grouping = s52": str(raw.get("fast_groups_equal_s52")) if raw is not None else "n/a"})
        dec_df = pd.DataFrame(dec)
        dec_df.to_csv(OUT / "specificity_decision.csv", index=False)
        S1 = R[(R["split"] == "S1")].copy()
        tab = pd.DataFrame({
            "edge": S1["edge_id"] + " " + S1["label"].astype(str), "kind": S1["kind"], "adjust": S1["adjust"],
            "n": S1["n_patients"],
            "epochs/half": S1.apply(lambda r: f"{r.get('mean_epochs_early', np.nan):.0f}/{r.get('mean_epochs_late', np.nan):.0f}", axis=1),
            "r(early,late)": S1.apply(lambda r: f"{r.r_early_late:.2f} [{r.pearson_lo:.2f}, {r.pearson_hi:.2f}]", axis=1),
            "ICC(1,1)": S1.apply(lambda r: f"{r.icc1:.2f} [{r.icc1_lo:.2f}, {r.icc1_hi:.2f}]", axis=1),
            "rel shrunk vs pooled": S1.apply(lambda r: f"{r.rel_shrunk_vs_pooled:.3f} [{r.rel_shrunk_vs_pooled_lo:.3f}, {r.rel_shrunk_vs_pooled_hi:.3f}]", axis=1),
            "d<0 CI": S1["ci_d_below0"], "perm p(r)": S1["perm_p_r"].map(lambda x: f"{x:.3g}")})
        tab.to_csv(OUT / "specificity_main_table.csv", index=False)
        s3 = R[R["split"] == "S3"][["edge_id", "adjust", "n_patients", "r_early_late", "r_early_late_lo", "r_early_late_hi"]]
        comp = pd.read_csv(OUT / "specificity_vs_operating_point.csv")
        comp["reading"] = np.where(comp["diff_hi"] < 0, "less persistent than operating point",
                                   np.where(comp["diff_lo"] > 0, "more persistent", "as persistent"))
        comp["diff [95% CI]"] = comp.apply(lambda r: f"{r['diff']:.2f} [{r.diff_lo:.2f}, {r.diff_hi:.2f}]", axis=1)
        lines += ["## Patient specificity at scale (wave-4 registered test, VitalDB)", "", md(dec_df), "",
                  "S1 statistics (all edges, adjustments):", "", md(tab), "",
                  "S3 within-patient shuffle (stationary ceiling), median over 200 draws:", "",
                  md(s3.round(3)), "",
                  "Persistence relative to the operating point (paired case bootstrap, r(edge) - r(mean HR or SBP, raw)):", "",
                  md(comp[["edge_id", "adjust", "vs", "n_common", "r_edge", "r_oper", "diff [95% CI]", "reading"]].round(3)), ""]
        summ["specificity_decision"] = dec_df.to_dict(orient="records")
    (OUT / "tables.md").write_text("\n".join(lines))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    log.info("report written")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage pat_rescue (EXPLORATORY, not pre-registered): PAT with the PLETH device delay handled, on the
# complete local .vital files. Diagnoses why native PAT is censored and whether the coupling is recoverable.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

VITAL_DIR = ROOT / "data" / "raw" / "physionet.org" / "files" / "vitaldb" / "1.0.0" / "vital_files"


def pat_rescue_case(path: str) -> dict:
    import vitaldb
    from clbench.beats import build_beat_table, detect_ppg_beats, gate_beats
    pid = "vdb" + Path(path).name.split(".")[0]
    out: dict = {"patient_id": pid}
    try:
        vf = vitaldb.VitalFile(path, ["SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH"])
        if not all(t in vf.get_track_names() for t in ["SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH"]):
            out["error"] = "missing track"
            return out
        X = vf.to_numpy(["SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH"], 1 / 500.0)
        fs = 500.0
        n = len(X)
        step, pad = int(600 * fs), int(10 * fs)
        parts, feet = [], []
        for s in range(0, n, step):
            a, b = max(0, s - pad), min(n, s + step + pad)
            e = X[a:b, 0]
            if np.isfinite(e).sum() < fs * 20:
                continue
            bt = build_beat_table(e, fs, X[a:b, 1], fs, t0=a / fs)
            if len(bt):
                bt = gate_beats(bt)
                parts.append(bt[(bt.t_r >= s / fs) & (bt.t_r < min(n, s + step) / fs)])
            pg = detect_ppg_beats(X[a:b, 2], fs)
            tf = pg.t_foot[np.isfinite(pg.t_foot)] + a / fs
            feet.append(tf[(tf >= s / fs) & (tf < min(n, s + step) / fs)])
        del X
        beats = pd.concat(parts, ignore_index=True).sort_values("t_r").reset_index(drop=True)
        tf = np.sort(np.concatenate(feet)) if feet else np.empty(0)
        tr = beats["t_r"].to_numpy(float)
        v = beats["valid"].to_numpy(bool)
        # delay histogram: all (foot - R) in (0, 1.5] s for valid beats
        lo = np.searchsorted(tf, tr[v])
        hi = np.searchsorted(tf, tr[v] + 1.5)
        d = np.concatenate([tf[l:h] - t for l, h, t in zip(lo, hi, tr[v])]) if len(tf) else np.empty(0)
        h, edges = np.histogram(d, bins=np.arange(0.0, 1.51, 0.01))
        D = float(edges[np.argmax(h)] + 0.005)
        out["delay_mode_s"] = D
        out["n_feet"] = int(len(tf))
        # assign the foot nearest R + D within +-0.15 s
        j = np.clip(np.searchsorted(tf, tr + D), 1, max(len(tf) - 1, 1))
        cand = np.stack([tf[j - 1], tf[np.minimum(j, len(tf) - 1)]], axis=1)
        k = np.argmin(np.abs(cand - (tr + D)[:, None]), axis=1)
        f = cand[np.arange(len(tr)), k]
        pat = np.where(np.abs(f - tr - D) <= 0.15, (f - tr) * 1000.0, np.nan)
        out["pat_cover_valid"] = float(np.isfinite(pat[v]).mean())
        out["native_window_share"] = float(np.mean((pat[v][np.isfinite(pat[v])] / 1000.0) < 0.6))
        beats["pat_r"] = pat
        blocks = blocks_from_beats(beats)
        rows = []
        for a, b in blocks:
            t = a
            while t + EPOCH_SEC <= b + 1e-6:
                m = (tr >= t) & (tr < t + EPOCH_SEC)
                vv = v[m]
                if len(vv) >= 100 and vv.mean() >= 0.8 and vv.sum() >= 150:
                    pc = pat_coupling(pat[m], beats["sbp"].to_numpy(float)[m], vv)
                    rows.append({"t0": t, **pc})
                t += EPOCH_SEC
        ep = pd.DataFrame(rows)
        out["n_epochs"] = len(ep)
        if len(ep):
            out["epochs_with_slope"] = int(np.isfinite(ep["pat_sbp_slope"]).sum())
            out["median_slope"] = float(np.nanmedian(ep["pat_sbp_slope"]))
            out["share_epochs_negative"] = float(np.mean(ep["pat_sbp_slope"].dropna() < 0))
            out["median_corr"] = float(np.nanmedian(ep["pat_sbp_corr"]))
            out["median_pat_ms"] = float(np.nanmedian(ep["pat_median"]))
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def stage_pat_rescue(workers: int) -> None:
    t0 = time.time()
    files = sorted(str(p) for p in VITAL_DIR.glob("*.vital"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(pat_rescue_case, files))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "pat_rescue_exploratory.csv", index=False)
    log.info("pat rescue done: %d files, %.0f s", len(files), time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage emulate (EXPLORATORY, not pre-registered): which monitor processing reproduces the wrong sign of
# the 2 s fit on monitor numerics -- pure reporting delay or averaging? Beat-derived 1 s series are passed
# through causal moving averages / delays matched to the measured monitor lags, then fitted as in (c).
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def _cma(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    v = np.isfinite(x)
    c = np.convolve(np.where(v, x, 0.0), np.ones(w))[: len(x)]
    n = np.convolve(v.astype(float), np.ones(w))[: len(x)]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = c / n
    return np.where(n >= w / 2, m, np.nan)


def _dly(x: np.ndarray, d: int) -> np.ndarray:
    return x if d <= 0 else np.concatenate([np.full(d, np.nan), x[:-d]])


EMU = {  # name: (HR causal MA width s, MAP width s, HR delay s, MAP delay s)
    "e0 beat 1 s -> 2 s bins, no processing": (1, 1, 0, 0),
    "e1 pure delays HR 7 s, MAP 6 s": (1, 1, 7, 6),
    "e2 causal averaging 8 s, both": (8, 8, 0, 0),
    "e3 lag-matched averaging HR 14 s, MAP 12 s (group delays ~6.5 / 5.5 s)": (14, 12, 0, 0),
    "e4 averaging HR 14 s, MAP 4 s": (14, 4, 0, 0),
}


def emulate_case(pid: str) -> dict:
    out = {"patient_id": pid}
    try:
        beats = pd.read_parquet(NATIVE / "beats" / f"{pid}.parquet")
        blocks = blocks_from_beats(beats)
        if not blocks:
            return out
        v = beats["valid"].to_numpy(bool)
        rri = beats["rri"].to_numpy(float)
        tr = beats["t_r"].to_numpy(float)
        tsys = beats["t_sys"].to_numpy(float)
        for name, (wh, wm, dh, dm) in EMU.items():
            bl = []
            for a, b in blocks:
                T0 = int(np.floor(a))
                n = int(np.ceil(b)) - T0
                H = sec_bins((tr + rri / 1000.0)[v], (60000.0 / rri)[v], T0, n)
                M = sec_bins(tsys[v], beats["map_beat"].to_numpy(float)[v], T0, n)
                H = _dly(_cma(H, wh), dh)
                M = _dly(_cma(M, wm), dm)
                m2 = n // 2
                H2 = np.nanmean(H[: 2 * m2].reshape(m2, 2), axis=1)
                M2 = np.nanmean(M[: 2 * m2].reshape(m2, 2), axis=1)
                bl.append((H2, M2))
            out[name] = lagged_timescale_fit(bl, n_lags=5).total
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def stage_emulate(workers: int) -> None:
    t0 = time.time()
    C = pd.read_parquet(OUT / "cases.parquet")
    pids = C.loc[C["g4_20min"] == True, "patient_id"].tolist()  # noqa: E712
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(emulate_case, pids, chunksize=8))
    pd.DataFrame(rows).to_parquet(OUT / "emulation_exploratory.parquet", index=False)
    log.info("emulation done: %d cases, %.0f s", len(pids), time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "fits", "specificity", "report", "pat_rescue", "emulate"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "extract":
        stage_extract(a.workers, a.limit)
    elif a.stage == "fits":
        stage_fits(a.workers)
    elif a.stage == "specificity":
        stage_specificity(a.workers)
    elif a.stage == "pat_rescue":
        stage_pat_rescue(a.workers)
    elif a.stage == "emulate":
        stage_emulate(a.workers)
    else:
        stage_report()


if __name__ == "__main__":
    main()
