"""
Track D (wave 4): is patient-specific structure real at the FAST (beat) timescale?

Pre-registered protocol: runs/iclr/fast_specificity/protocol.md (sha256 in protocol.sha256). The analysis
stage refuses to run if the protocol file no longer matches the registered hash.

For five fast edges (PAT-SBP slope, PPV, RSA amplitude, sequence BRS, RMSSD) computed per 5-min epoch from
the wave-2 beat tables (MIMIC-IV-WDB ICU, primary; local VitalDB surgical cases, secondary), each patient's
recording is split into early/late halves (and interleaved-with-gap, shuffled and data-limited variants).
The wave-3 patient-specificity estimator (``typed_temporal_pooled.patient_specificity``; epochs as units)
tests whether a patient's own early estimate predicts the late epochs better than the population mean,
on raw values and on residuals after concurrent mean HR / SBP (and a coupling-destroyed "signal-property
twin"). Cross-patient pairing permutations give the null for r. Wave-3 slow drug-response edges are
placed alongside for contrast.

USAGE (PYTHONPATH=src:.)
  python scripts/52_fast_edge_specificity.py --stage extract --workers 4
  python scripts/52_fast_edge_specificity.py --stage synthetic
  python scripts/52_fast_edge_specificity.py --stage analyse
  python scripts/52_fast_edge_specificity.py --stage report
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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

from clbench.edges.fast_coupling import (  # noqa: E402
    beat_regression, pat_coupling, respiratory_pressure_variation, rsa_amplitude, running_mean_detrend,
    sequence_method,
)
from clbench.edges.typed_temporal_pooled import _spec_stats, patient_specificity  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("52_fast_spec")

OUT = ROOT / "runs" / "iclr" / "fast_specificity"
FC = ROOT / "runs" / "iclr" / "fast_coupling"
CL2 = ROOT / "runs" / "iclr" / "closed_loop_v2"
PROTOCOL = OUT / "protocol.md"
PROTOCOL_SHA_FILE = OUT / "protocol.sha256"

DATASETS = ["mimic", "vitaldb"]
SEED = 20260924
K_TWIN = 5
N_BOOT = 2000
N_PERM = 1000
N_SHUFFLE = 200
MIN_PER_HALF = 3

PRIMARY = {  # id -> (column, label, unit)
    "E1": ("pat_sbp_slope", "PAT vs SBP slope", "ms/mmHg"),
    "E2": ("ppv", "pulse-pressure variation (PPV)", "%"),
    "E3": ("rsa_h", "RSA amplitude (harmonic)", "ms"),
    "E4": ("seq_pooled_brs", "sequence BRS (lags 0-2)", "ms/mmHg"),
    "E5": ("rmssd", "RMSSD (RRI variability)", "ms"),
}
SECONDARY = {
    "S-spv": ("spv", "systolic pressure variation (SPV)", "mmHg"),
    "S-rsaamp": ("rsa_amp", "RSA amplitude (wave-2 phase-binned)", "ms"),
    "S-regbrs": ("reg_brs", "ARX BRS SBP(n..n-2)->RRI(n)", "ms/mmHg"),
    "S-sdnn": ("sdnn", "SDNN", "ms"),
    "S-patcorr": ("pat_sbp_corr", "PAT-SBP beat correlation", "r"),
}
TWIN = {"pat_sbp_slope": "twin_pat", "ppv": "twin_ppv", "rsa_h": "twin_rsa", "seq_pooled_brs": "twin_seq"}
COV_HS = ["hr_mean", "sbp_mean"]
COV_DRUG = ["remifentanil_ce0", "propofol_ce0", "propofol_rec"]
CHECK_COLS = ["pat_sbp_slope", "pat_sbp_corr", "ppv", "spv", "rsa_amp", "seq_pooled_brs", "sdnn", "reg_brs",
              "hr_mean", "sbp_mean"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def registered_sha() -> str:
    return PROTOCOL_SHA_FILE.read_text().split()[0]


def check_protocol() -> str:
    h = sha256(PROTOCOL)
    reg = registered_sha()
    if h != reg:
        raise SystemExit(f"protocol.md sha256 {h} != registered {reg}; refusing to run the analysis")
    return h


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage 1: per-epoch edges and signal-property twins from the wave-2 beat tables
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def breath_ids(ph: np.ndarray) -> np.ndarray:
    """Breath index per beat, delimited by respiratory-phase wraps exactly as in
    ``respiratory_pressure_variation``."""
    wraps = np.where(np.diff(np.unwrap(np.where(np.isfinite(ph), ph, 0.0)) // (2 * np.pi)) > 0)[0] + 1
    ind = np.zeros(len(ph), dtype=int)
    ind[wraps] = 1
    return np.cumsum(ind)


def harmonic_rsa(rri: np.ndarray, ph: np.ndarray, v: np.ndarray) -> float:
    """Peak-to-trough (2 sqrt(a^2 + b^2)) of the OLS fit of 15-beat-detrended RRI on [1, cos, sin](phase)."""
    y = running_mean_detrend(rri, v, 15)
    ok = np.isfinite(y) & np.isfinite(ph)
    if ok.sum() < 60:
        return np.nan
    X = np.column_stack([np.ones(ok.sum()), np.cos(ph[ok]), np.sin(ph[ok])])
    c, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    return float(2.0 * np.hypot(c[1], c[2]))


def seq_pooled(sbp: np.ndarray, rri: np.ndarray, v: np.ndarray) -> float:
    slopes = np.concatenate([sequence_method(sbp, rri, v, k).slopes for k in (0, 1, 2)])
    return float(np.mean(slopes)) if len(slopes) else np.nan


def epoch_stats(b: pd.DataFrame, rng: np.random.Generator) -> dict:
    v = b["valid"].to_numpy(bool)
    sbp = b["sbp"].to_numpy(float)
    rri = b["rri"].to_numpy(float)
    pp = b["pp"].to_numpy(float)
    pat = b["pat"].to_numpy(float)
    ph = b["resp_phase"].to_numpy(float)
    n = len(b)
    out: dict = {"n_beats": n, "n_valid": int(v.sum())}
    if v.sum() < 60:  # as in wave-2 analyse_epoch
        return out
    out.update({"hr_mean": float(60000.0 / np.nanmean(rri[v])), "sbp_mean": float(np.nanmean(sbp[v])),
                "sdnn": float(np.nanstd(rri[v]))})
    # E5 RMSSD over consecutive valid pairs
    pr = v[:-1] & v[1:] & np.isfinite(rri[:-1]) & np.isfinite(rri[1:])
    d = np.diff(rri)[pr]
    out["rmssd"] = float(np.sqrt(np.mean(d ** 2))) if len(d) >= 30 else np.nan
    # E4 sequence BRS and its twin (SBP circularly shifted against RRI)
    out["seq_pooled_brs"] = seq_pooled(sbp, rri, v)
    tw = []
    for _ in range(K_TWIN):
        u = int(rng.integers(20, n - 20 + 1)) if n > 60 else n // 2
        tw.append(seq_pooled(np.roll(sbp, u), rri, v & np.roll(v, u)))
    out["twin_seq"] = float(np.nanmean(tw)) if np.isfinite(tw).any() else np.nan
    out["reg_brs"] = float(beat_regression(sbp, rri, v).total)
    # E1 PAT-SBP slope and its twin
    pc = pat_coupling(pat, sbp, v)
    out.update({"pat_sbp_slope": pc["pat_sbp_slope"], "pat_sbp_corr": pc["pat_sbp_corr"], "n_pat": pc["n_pat"]})
    if np.isfinite(pc["pat_sbp_slope"]):
        tw = []
        for _ in range(K_TWIN):
            u = int(rng.integers(30, n - 30 + 1)) if n > 90 else n // 2
            tw.append(pat_coupling(pat, np.roll(sbp, u), v & np.roll(v, u))["pat_sbp_slope"])
        out["twin_pat"] = float(np.nanmean(tw)) if np.isfinite(tw).any() else np.nan
    # respiration-dependent edges (same gate as wave 2)
    if np.isfinite(ph).mean() > 0.8:
        amp, _ = rsa_amplitude(rri, ph, v)
        out["rsa_amp"] = amp
        rv = respiratory_pressure_variation(sbp, pp, ph, v)
        out.update({"spv": rv["spv"], "ppv": rv["ppv"], "n_breaths": rv["n_breaths"]})
        out["rsa_h"] = harmonic_rsa(rri, ph, v)
        if np.isfinite(out["rsa_h"]):
            bid = breath_ids(ph)
            tw = []
            for _ in range(K_TWIN):
                delta = rng.uniform(0.0, 2 * np.pi, bid.max() + 1)
                tw.append(harmonic_rsa(rri, ph + delta[bid], v))
            out["twin_rsa"] = float(np.nanmean(tw)) if np.isfinite(tw).any() else np.nan
        if np.isfinite(rv["ppv"]):
            ok = v & np.isfinite(ph) & np.isfinite(pp)
            if ok.sum() >= 30:
                X = np.column_stack([np.ones(ok.sum()), np.cos(ph[ok]), np.sin(ph[ok]),
                                     np.cos(2 * ph[ok]), np.sin(2 * ph[ok])])
                c, *_ = np.linalg.lstsq(X, pp[ok], rcond=None)
                pp_t = pp.copy()
                pp_t[ok] = pp[ok] - (X @ c - c[0])
                out["twin_ppv"] = respiratory_pressure_variation(sbp, pp_t, ph, v)["ppv"]
    return out


def _extract_patient(args) -> list[dict]:
    ds, di, pid, pi, ep = args
    beats = pd.read_parquet(FC / "cache" / ds / f"{pid}_beats.parquet")
    tr = beats["t_r"].to_numpy(float)
    rows = []
    ep = ep.sort_values("t0").reset_index(drop=True)
    # grid index within block (contiguous 5-min grid incl. unusable epochs)
    ep["j"] = ep.groupby("block")["t0"].rank(method="first").astype(int) - 1
    for ei, r in ep.iterrows():
        base = {"dataset": ds, "patient_id": pid, "block": int(r["block"]), "t0": float(r["t0"]),
                "t1": float(r["t1"]), "j": int(r["j"]), "epoch_index": int(ei), "usable": bool(r["usable"]),
                "propofol_ce": r.get("propofol_ce", np.nan), "remifentanil_ce": r.get("remifentanil_ce", np.nan)}
        if not r["usable"]:
            rows.append(base)
            continue
        rng = np.random.default_rng([SEED, di, pi, int(ei)])
        m = (tr >= r["t0"]) & (tr < r["t1"])
        rows.append({**base, **epoch_stats(beats.loc[m], rng)})
    return rows


def stage_extract(workers: int) -> None:
    t0 = time.time()
    E = pd.read_parquet(FC / "epochs.parquet")
    jobs = []
    for di, ds in enumerate(DATASETS):
        e = E[E["dataset"] == ds]
        for pi, pid in enumerate(sorted(e["patient_id"].unique())):
            jobs.append((ds, di, pid, pi, e[e["patient_id"] == pid][["block", "t0", "t1", "usable", "propofol_ce",
                                                                     "remifentanil_ce"]].copy()))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rr in enumerate(ex.map(_extract_patient, jobs)):
            rows.extend(rr)
            if (i + 1) % 10 == 0:
                log.info("extract %d/%d patients (%.0f s)", i + 1, len(jobs), time.time() - t0)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "epochs_fast.parquet", index=False)
    # integrity: recomputed values vs wave-2 epochs.parquet
    ref = E[E["usable"] == True][["dataset", "patient_id", "t0"] + CHECK_COLS]  # noqa: E712
    mm = df[df["usable"]].merge(ref, on=["dataset", "patient_id", "t0"], suffixes=("", "_w2"))
    chk = []
    for c in CHECK_COLS:
        a, b = mm[c].to_numpy(float), mm[c + "_w2"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        chk.append({"column": c, "n_matched_epochs": len(mm), "n_both_finite": int(both.sum()),
                    "n_finite_mismatch": int((np.isfinite(a) != np.isfinite(b)).sum()),
                    "max_abs_diff": float(np.max(np.abs(a[both] - b[both]))) if both.any() else np.nan})
    pd.DataFrame(chk).to_csv(OUT / "integrity_vs_wave2.csv", index=False)
    log.info("extract done: %d epoch rows, %.0f s", len(df), time.time() - t0)
    (OUT / "extract_meta.json").write_text(json.dumps({
        "seconds": time.time() - t0, "n_rows": len(df), "script_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(PROTOCOL), "workers": workers}, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage 2/3: split-half statistics
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def icc1(e: np.ndarray, l: np.ndarray, w: np.ndarray | None = None) -> float:
    """ICC(1,1), one-way random, absolute agreement, k = 2 (optionally patient-weighted)."""
    w = np.ones(len(e)) if w is None else w
    W = w.sum()
    m = (e + l) / 2.0
    G = np.sum(w * m) / W
    msb = 2.0 * np.sum(w * (m - G) ** 2) / max(W - 1, 1e-12)
    msw = np.sum(w * ((e - m) ** 2 + (l - m) ** 2)) / max(W, 1e-12)
    return float((msb - msw) / (msb + msw)) if (msb + msw) > 0 else np.nan


def wpearson(x, y, w):
    W = w.sum()
    xm, ym = np.sum(w * x) / W, np.sum(w * y) / W
    den = np.sqrt(np.sum(w * (x - xm) ** 2) * np.sum(w * (y - ym) ** 2))
    return float(np.sum(w * (x - xm) * (y - ym)) / den) if den > 0 else np.nan


def _ranks(x):
    return pd.Series(x).rank().to_numpy()


def half_mse_ratio(e, l, w):
    W = w.sum()
    mu = np.sum(w * e) / W
    num = np.sum(w * (e - l) ** 2) / W
    den = np.sum(w * (mu - l) ** 2) / W
    return float(num / den) if den > 0 else np.nan


def own_stats(me: np.ndarray, ml: np.ndarray, w: np.ndarray) -> dict:
    """Stats on half means that need expanding (bootstrap = integer weights)."""
    idx = np.repeat(np.arange(len(me)), w.astype(int)) if not np.all(w == 1) else np.arange(len(me))
    e, l = me[idx], ml[idx]
    return {"pearson": wpearson(e, l, np.ones(len(e))), "spearman": wpearson(_ranks(e), _ranks(l), np.ones(len(e))),
            "icc1": icc1(e, l), "half_mse_ratio": half_mse_ratio(e, l, np.ones(len(e)))}


def suff(groups: list[tuple[np.ndarray, np.ndarray]]):
    ne = np.array([len(a) for a, _ in groups], float)
    se = np.array([a.sum() for a, _ in groups])
    sse = np.array([(a ** 2).sum() for a, _ in groups])
    nl = np.array([len(b) for _, b in groups], float)
    sl = np.array([b.sum() for _, b in groups])
    ssl = np.array([(b ** 2).sum() for _, b in groups])
    return ne, se, sse, nl, sl, ssl


def split_stats(groups, n_boot=N_BOOT, n_perm=N_PERM, seed=0, full=True) -> dict:
    """All pre-registered statistics for one list of per-patient (early, late) epoch-value arrays."""
    P = len(groups)
    if P < 5:
        return {"n_patients": P}
    ne, se, sse, nl, sl, ssl = suff(groups)
    me, ml = se / ne, sl / nl
    if not full:  # point estimates only (S3 shuffle draws)
        s = _spec_stats(ne, se, sse, nl, sl, ssl, np.ones(P))
        o = own_stats(me, ml, np.ones(P))
        return {"n_patients": P, **{k: s[k] for k in ("r_early_late", "rel_shrunk_vs_pooled", "d_shrunk_minus_pooled")},
                "icc1": o["icc1"], "half_mse_ratio": o["half_mse_ratio"]}
    res = patient_specificity(ne, se, sse, nl, sl, ssl, n_boot=n_boot, seed=seed)
    est = own_stats(me, ml, np.ones(P))
    rng = np.random.default_rng(seed)
    reps = {k: np.empty(n_boot) for k in est}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)
        o = own_stats(me, ml, w)
        for k in est:
            reps[k][b] = o[k]
    for k, v in est.items():
        rr = reps[k][np.isfinite(reps[k])]
        res[k] = v
        res[k + "_lo"] = float(np.quantile(rr, 0.025)) if len(rr) > 10 else np.nan
        res[k + "_hi"] = float(np.quantile(rr, 0.975)) if len(rr) > 10 else np.nan
    # cross-patient pairing null: late halves permuted across patients
    prng = np.random.default_rng(seed + 1)
    nr, nd = np.empty(n_perm), np.empty(n_perm)
    for b in range(n_perm):
        p = prng.permutation(P)
        s = _spec_stats(ne, se, sse, nl[p], sl[p], ssl[p], np.ones(P))
        nr[b], nd[b] = s["r_early_late"], s["d_shrunk_minus_pooled"]
    res["perm_p_r"] = float((1 + np.sum(nr >= res["r_early_late"])) / (n_perm + 1))
    res["perm_p_d"] = float((1 + np.sum(nd <= res["d_shrunk_minus_pooled"])) / (n_perm + 1))
    res["perm_null_r_q95"] = float(np.quantile(nr, 0.95))
    res["mean_epochs_early"] = float(ne.mean())
    res["mean_epochs_late"] = float(nl.mean())
    return res


def winsorise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        x = df[c].to_numpy(float)
        f = np.isfinite(x)
        if f.sum() < 10:
            continue
        lo, hi = np.quantile(x[f], [0.01, 0.99])
        df[c] = np.where(f, np.clip(x, lo, hi), np.nan)
    return df


def assign_halves(d: pd.DataFrame, split: str, rng: np.random.Generator | None = None) -> pd.Series:
    """Return a Series in {0 (early/A), 1 (late/B), -1 (drop)} for the rows of one patient's finite epochs
    (already sorted by time)."""
    n = len(d)
    lab = np.full(n, -1)
    if split in ("S1", "S3", "S4"):
        order = np.arange(n) if split != "S3" else rng.permutation(n)
        k = n // 2
        lab[order[:k]] = 0
        lab[order[k:]] = 1
        if split == "S4":
            e_idx = np.where(lab == 0)[0][:3]
            l_idx = np.where(lab == 1)[0][:3]
            lab[:] = -1
            lab[e_idx] = 0
            lab[l_idx] = 1
    elif split == "S2":
        j = d["j"].to_numpy(int)
        lab[j % 4 == 0] = 0
        lab[j % 4 == 2] = 1
    return pd.Series(lab, index=d.index)


def build_groups(df: pd.DataFrame, col: str, covs: list[str], split: str, rng=None):
    """Eligible epochs -> halves -> (OLS residualisation fitted on early epochs) -> per-patient arrays."""
    need = [col] + covs
    d = df[np.all(np.isfinite(df[need].to_numpy(float)), axis=1)].sort_values(["patient_id", "t0"])
    parts = []
    for pid, g in d.groupby("patient_id", sort=True):
        h = assign_halves(g, split, rng)
        g = g.assign(half=h.values)
        g = g[g["half"] >= 0]
        if (g["half"] == 0).sum() >= MIN_PER_HALF and (g["half"] == 1).sum() >= MIN_PER_HALF:
            parts.append(g)
    if not parts:
        return [], []
    d = pd.concat(parts)
    y = d[col].to_numpy(float)
    if covs:
        X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in covs])
        early = d["half"].to_numpy() == 0
        beta, *_ = np.linalg.lstsq(X[early], y[early], rcond=None)
        y = y - X @ beta
    d = d.assign(val=y)
    groups, pids = [], []
    for pid, g in d.groupby("patient_id", sort=True):
        groups.append((g.loc[g["half"] == 0, "val"].to_numpy(float), g.loc[g["half"] == 1, "val"].to_numpy(float)))
        pids.append(pid)
    return groups, pids


def prepare(ds: str) -> pd.DataFrame:
    df = pd.read_parquet(OUT / "epochs_fast.parquet")
    df = df[(df["dataset"] == ds) & (df["usable"])].copy()
    df["remifentanil_ce0"] = df["remifentanil_ce"].fillna(0.0).astype(float)
    df["propofol_rec"] = df["propofol_ce"].notna().astype(float)
    df["propofol_ce0"] = df["propofol_ce"].fillna(0.0).astype(float)
    cols = [c for c, *_ in PRIMARY.values()] + [c for c, *_ in SECONDARY.values()] + list(TWIN.values()) + COV_HS + \
        ["remifentanil_ce0", "propofol_ce0"]
    return winsorise(df, cols)


def adjustments(ds: str, col: str) -> dict[str, list[str]]:
    a = {"raw": [], "adj_hs": COV_HS}
    if col in TWIN:
        a["adj_hst"] = COV_HS + [TWIN[col]]
    if ds == "vitaldb":
        a["adj_hsd"] = COV_HS + COV_DRUG
    return a


def stage_analyse() -> None:
    psha = check_protocol()
    t0 = time.time()
    rows, pat_rows = [], []
    edges = {**PRIMARY, **SECONDARY}
    for ds in DATASETS:
        df = prepare(ds)
        items = [(eid, col, lab, unit, "primary" if eid in PRIMARY else "secondary")
                 for eid, (col, lab, unit) in edges.items()]
        items += [(f"T-{col}", tw, f"twin of {col}", "", "twin") for col, tw in TWIN.items()]
        for eid, col, lab, unit, kind in items:
            adjs = adjustments(ds, col) if kind != "twin" else {"raw": []}
            for adj, covs in adjs.items():
                for split in ("S1", "S2", "S4"):
                    if kind != "primary" and split != "S1":
                        continue
                    groups, pids = build_groups(df, col, covs, split)
                    res = split_stats(groups, seed=0)
                    rows.append({"dataset": ds, "edge_id": eid, "column": col, "label": lab, "unit": unit,
                                 "kind": kind, "adjust": adj, "split": split, **res})
                    if split == "S1":
                        for pid, (a, b) in zip(pids, groups):
                            pat_rows.append({"dataset": ds, "edge_id": eid, "column": col, "adjust": adj,
                                             "patient_id": pid, "n_early": len(a), "n_late": len(b),
                                             "early_mean": a.mean(), "late_mean": b.mean()})
                if kind == "primary":  # S3 within-patient epoch-order shuffle (stationary ceiling)
                    rng = np.random.default_rng(7)
                    draws = []
                    for _ in range(N_SHUFFLE):
                        g, _p = build_groups(df, col, covs, "S3", rng)
                        draws.append(split_stats(g, full=False))
                    D = pd.DataFrame(draws)
                    r = {"dataset": ds, "edge_id": eid, "column": col, "label": lab, "unit": unit, "kind": kind,
                         "adjust": adj, "split": "S3", "n_patients": int(D["n_patients"].median())}
                    for k in ("r_early_late", "rel_shrunk_vs_pooled", "d_shrunk_minus_pooled", "icc1", "half_mse_ratio"):
                        if k in D:
                            r[k] = float(D[k].median())
                            r[k + "_lo"] = float(D[k].quantile(0.025))
                            r[k + "_hi"] = float(D[k].quantile(0.975))
                    rows.append(r)
            log.info("%s %s done (%.0f s)", ds, eid, time.time() - t0)
    R = pd.DataFrame(rows)
    # decision rule (protocol section 6)
    R["ci_d_below0"] = R["d_shrunk_minus_pooled_hi"] < 0
    # Holm over the 10 primary (edge, dataset) cells, permutation p for r on adj_hs, S1
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
    R.to_csv(OUT / "split_half_all.csv", index=False)
    pd.DataFrame(pat_rows).to_csv(OUT / "patient_halves.csv", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "protocol_sha256": psha, "script_sha256": sha256(Path(__file__)),
        "typed_temporal_pooled_sha256": sha256(ROOT / "src" / "clbench" / "edges" / "typed_temporal_pooled.py"),
        "fast_coupling_sha256": sha256(ROOT / "src" / "clbench" / "edges" / "fast_coupling.py"),
        "spec_function_source_sha256": {f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
                                        for f in (_spec_stats, patient_specificity)},
        "n_boot": N_BOOT, "n_perm": N_PERM, "n_shuffle": N_SHUFFLE, "seconds_analyse": time.time() - t0,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    log.info("analyse done %.0f s", time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Synthetic check of the analysis code (protocol section 8)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def stage_synthetic(n_rep: int = 200) -> None:
    rng = np.random.default_rng(99)
    rows = []
    for scen, tau, beta in [("null_tau0", 0.0, 0.0), ("tau0.5", 0.5, 0.0), ("tau1", 1.0, 0.0),
                            ("covariate_only", 0.0, 1.0)]:
        for rep in range(n_rep):
            P = 50
            n_ep = rng.integers(10, 49, P)
            theta = rng.normal(0, tau, P)
            hr_level = rng.normal(0, 1, P)
            recs = []
            for i in range(P):
                hr = hr_level[i] + rng.normal(0, 0.3, n_ep[i])
                y = theta[i] + beta * hr + rng.normal(0, 1.0, n_ep[i])
                for k in range(n_ep[i]):
                    recs.append({"patient_id": f"p{i:02d}", "t0": float(k), "j": k, "y": y[k], "hr_mean": hr[k],
                                 "sbp_mean": rng.normal()})
            df = pd.DataFrame(recs)
            for adj, covs in (("raw", []), ("adj_hs", COV_HS)):
                g, _ = build_groups(df, "y", covs, "S1")
                s = split_stats(g, n_boot=500, n_perm=200, seed=rep)
                # theoretical half-mean ICC (average over patients) for the raw scenario
                half_var = np.mean(1.0 / (n_ep // 2))
                icc_true = tau ** 2 / (tau ** 2 + half_var) if tau > 0 else 0.0
                rows.append({"scenario": scen, "rep": rep, "adjust": adj, "icc1": s["icc1"], "icc_true": icc_true,
                             "r": s["r_early_late"], "ci_d_below0": s["d_shrunk_minus_pooled_hi"] < 0,
                             "perm_p_r": s["perm_p_r"], "rel": s["rel_shrunk_vs_pooled"]})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "synthetic_reps.csv", index=False)
    summ = S.groupby(["scenario", "adjust"]).agg(
        n=("rep", "size"), icc_mean=("icc1", "mean"), icc_true=("icc_true", "mean"), r_mean=("r", "mean"),
        rule_rate=("ci_d_below0", "mean"), perm_p_lt05=("perm_p_r", lambda x: float(np.mean(x < 0.05))),
        rel_median=("rel", "median")).reset_index()
    summ.to_csv(OUT / "synthetic_check.csv", index=False)
    print(summ.to_string())


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage 4: report
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def f3(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def ci(r, k, nd=2):
    return f"{f3(r.get(k), nd)} [{f3(r.get(k + '_lo'), nd)}, {f3(r.get(k + '_hi'), nd)}]"


def pv(x):
    return "n/a" if x is None or not np.isfinite(x) else (f"{x:.3f}" if x >= 0.001 else "<0.001")


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def wave3_rows() -> pd.DataFrame:
    rows = []
    for arm_dir in ("30s_all", "2s_fine"):
        f = CL2 / arm_dir / "patient_specificity.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d["arm"] == "real") & (d["outcome"].isin(["obs", "kink"]))]
        for _, r in d.iterrows():
            rows.append({"source": f"wave3 {arm_dir}", "edge": f"{r['rid']} {r['label']}", "outcome": r["outcome"],
                         "n": int(r["n_patients"]),
                         "units_per_patient": (r["n_events_early"] + r["n_events_late"]) / r["n_patients"],
                         "r": r["r_early_late"], "r_lo": r["r_early_late_lo"], "r_hi": r["r_early_late_hi"],
                         "rel": r["rel_shrunk_vs_pooled"], "rel_lo": r["rel_shrunk_vs_pooled_lo"],
                         "rel_hi": r["rel_shrunk_vs_pooled_hi"],
                         "rule": bool(r["d_shrunk_minus_pooled_hi"] < 0)})
    return pd.DataFrame(rows)


def make_figure(R: pd.DataFrame, W: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fast = R[(R["kind"] == "primary") & (R["split"] == "S1") & (R["adjust"].isin(["raw", "adj_hs"]))]
    w3 = W[(W["outcome"] == "obs") & (W["n"] >= 20)].sort_values(["source", "edge"])
    labels, vals, los, his, colors = [], [], [], [], []
    ink = {"raw": "#2a78d6", "adj_hs": "#eb6834", "slow": "#8a8984"}  # validated categorical slots 1-2 + neutral
    for ds in DATASETS:
        for eid, (col, lab, unit) in PRIMARY.items():
            for adj in ("raw", "adj_hs"):
                r = fast[(fast["dataset"] == ds) & (fast["edge_id"] == eid) & (fast["adjust"] == adj)]
                if r.empty or not np.isfinite(r["r_early_late"].iloc[0]):
                    continue
                r = r.iloc[0]
                labels.append(f"{ds} {lab}{' | HR,SBP-adj' if adj == 'adj_hs' else ''} (n={int(r['n_patients'])})")
                vals.append(r["r_early_late"]); los.append(r["r_early_late_lo"]); his.append(r["r_early_late_hi"])
                colors.append(ink[adj])
    for _, r in w3.iterrows():
        labels.append(f"{r['source']} {r['edge']} (n={r['n']})")
        vals.append(r["r"]); los.append(r["r_lo"]); his.append(r["r_hi"]); colors.append(ink["slow"])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.24 * len(labels) + 1.4))
    for yi, v, lo, hi, c in zip(y, vals, los, his, colors):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
        ax.plot([v], [yi], "o", color=c, ms=4.5)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("early vs late half, correlation of patient means (95% patient-bootstrap CI)", fontsize=8)
    ax.set_xlim(-1, 1)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ink[k], marker="o", lw=1.6, ms=4.5, label=t)
               for k, t in (("raw", "fast edge, raw"), ("adj_hs", "fast edge, adjusted for mean HR and SBP"),
                            ("slow", "slow drug response (wave 3, 'obs', n >= 20)"))]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("Early vs late half of each recording: does the patient's own edge persist?",
                 fontsize=8.5, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fast_vs_slow_specificity.png", dpi=200)
    plt.close(fig)


def stage_report() -> None:
    R = pd.read_csv(OUT / "split_half_all.csv")
    meta = json.loads((OUT / "meta.json").read_text())
    integ = pd.read_csv(OUT / "integrity_vs_wave2.csv")
    syn = pd.read_csv(OUT / "synthetic_check.csv") if (OUT / "synthetic_check.csv").exists() else None
    W = wave3_rows()
    lines = [f"# Track D: patient-specific structure at the fast (beat) timescale",
             "", f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `scripts/52_fast_edge_specificity.py`. "
             f"Protocol `protocol.md`, sha256 `{meta['protocol_sha256']}` (registered before the analysis; "
             f"script sha256 at run time `{meta['script_sha256'][:16]}...`). CIs: 2,000 patient-bootstrap "
             f"replicates; permutation nulls: 1,000 cross-patient pairings; S3: 200 within-patient shuffles.", ""]

    # decision table
    lines += ["## 1. Decision table (S1 early/late; protocol section 6)", ""]
    dec = []
    for ds in DATASETS:
        for eid, (col, lab, unit) in PRIMARY.items():
            def g(adj, split="S1"):
                x = R[(R["dataset"] == ds) & (R["edge_id"] == eid) & (R["adjust"] == adj) & (R["split"] == split)]
                return x.iloc[0].to_dict() if len(x) else {}
            raw, hs, hst, s2 = g("raw"), g("adj_hs"), g("adj_hst"), g("adj_hs", "S2")
            p1 = bool(raw.get("ci_d_below0", False)) if raw.get("n_patients", 0) >= 5 else None
            p2 = bool(hs.get("ci_d_below0", False)) if hs.get("n_patients", 0) >= 5 else None
            real = (p1 and p2) if (p1 is not None and p2 is not None) else None
            disc = bool(real and hs.get("perm_p_r", 1) >= 0.05)
            dec.append({"dataset": ds, "edge": f"{eid} {lab} ({unit})", "n": raw.get("n_patients", 0),
                        "P1 raw": {True: "yes", False: "no", None: "n/a"}[p1],
                        "P2 HR/SBP-adj": {True: "yes", False: "no", None: "n/a"}[p2],
                        "REAL": {True: "**REAL**", False: "not shown", None: "n/a (n<5)"}[real],
                        "twin-adj (H3)": ("yes" if hst.get("ci_d_below0") else "no") if hst.get("n_patients", 0) >= 5 else "-",
                        "S2 adj holds": ("yes" if s2.get("ci_d_below0") else "no") if s2.get("n_patients", 0) >= 5 else "n/a",
                        "perm p (r, adj)": pv(hs.get("perm_p_r", np.nan)),
                        "Holm p": pv(hs.get("perm_p_r_holm_adj_hs", np.nan)),
                        "discordant": "yes" if disc else ""})
    D = pd.DataFrame(dec)
    D.to_csv(OUT / "decision_table.csv", index=False)
    lines += [md(D), ""]

    # main numbers
    lines += ["## 2. Primary edges, S1 early/late: effect sizes", "",
              "rel = MSE(shrunk)/MSE(pooled) - 1 against the late-half epochs (wave-3 metric; < 0 = own early data "
              "helps). Half-MSE ratio = own early mean vs population mean predicting the late-half mean (< 1 = own "
              "early helps).", ""]
    main = []
    for ds in DATASETS:
        for eid, (col, lab, unit) in PRIMARY.items():
            for adj in ("raw", "adj_hs", "adj_hst", "adj_hsd"):
                x = R[(R["dataset"] == ds) & (R["edge_id"] == eid) & (R["adjust"] == adj) & (R["split"] == "S1")]
                if x.empty or x.iloc[0].get("n_patients", 0) < 5:
                    continue
                r = x.iloc[0].to_dict()
                main.append({"dataset": ds, "edge": f"{eid} {lab}", "adjust": adj, "n": int(r["n_patients"]),
                             "epochs/half": f"{r['mean_epochs_early']:.0f}/{r['mean_epochs_late']:.0f}",
                             "r(early,late)": ci(r, "r_early_late"), "Spearman": ci(r, "spearman"),
                             "ICC(1,1)": ci(r, "icc1"), "rel shrunk vs pooled": ci(r, "rel_shrunk_vs_pooled"),
                             "half-MSE ratio": ci(r, "half_mse_ratio"), "mean B": f3(r["mean_B"]),
                             "perm p(r)": pv(r["perm_p_r"]), "null r q95": f3(r["perm_null_r_q95"])})
    M = pd.DataFrame(main)
    M.to_csv(OUT / "main_table.csv", index=False)
    lines += [md(M), ""]

    # splits
    lines += ["## 3. Split variants (primary edges, HR/SBP-adjusted and raw)", "",
              "S1 early/late; S2 interleaved 5-min epochs with a 5-min gap; S3 within-patient shuffled epoch order "
              "(median and 2.5-97.5 % range over 200 draws; stationary ceiling); S4 first 3 epochs of each S1 half "
              "(data-limited, 15 min per half).", ""]
    sp = []
    for ds in DATASETS:
        for eid, (col, lab, unit) in PRIMARY.items():
            for adj in ("raw", "adj_hs"):
                row = {"dataset": ds, "edge": f"{eid} {lab}", "adjust": adj}
                for split in ("S1", "S2", "S3", "S4"):
                    x = R[(R["dataset"] == ds) & (R["edge_id"] == eid) & (R["adjust"] == adj) & (R["split"] == split)]
                    if x.empty or x.iloc[0].get("n_patients", 0) < 5:
                        row[f"{split} r"] = "n/a"
                        row[f"{split} rel"] = "n/a"
                        continue
                    r = x.iloc[0].to_dict()
                    row[f"{split} r"] = f"{ci(r, 'r_early_late')} (n={int(r['n_patients'])})"
                    row[f"{split} rel"] = ci(r, "rel_shrunk_vs_pooled")
                sp.append(row)
    lines += [md(pd.DataFrame(sp)), ""]

    # twins
    lines += ["## 4. Signal-property twins (coupling destroyed, same epochs): how patient-specific are they?", ""]
    tw = []
    for ds in DATASETS:
        for col, t in TWIN.items():
            x = R[(R["dataset"] == ds) & (R["column"] == t) & (R["split"] == "S1")]
            e = R[(R["dataset"] == ds) & (R["column"] == col) & (R["split"] == "S1") & (R["adjust"] == "raw")]
            if x.empty or x.iloc[0].get("n_patients", 0) < 5:
                continue
            r, er = x.iloc[0].to_dict(), e.iloc[0].to_dict()
            tw.append({"dataset": ds, "edge": col, "twin": t, "n": int(r["n_patients"]),
                       "edge ICC": ci(er, "icc1"), "twin ICC": ci(r, "icc1"), "twin r": ci(r, "r_early_late"),
                       "twin rel": ci(r, "rel_shrunk_vs_pooled")})
    lines += [md(pd.DataFrame(tw)) if tw else "(none)", ""]

    # secondary
    lines += ["## 5. Secondary statistics (S1)", ""]
    sc = []
    for ds in DATASETS:
        for eid, (col, lab, unit) in SECONDARY.items():
            for adj in ("raw", "adj_hs"):
                x = R[(R["dataset"] == ds) & (R["edge_id"] == eid) & (R["adjust"] == adj) & (R["split"] == "S1")]
                if x.empty or x.iloc[0].get("n_patients", 0) < 5:
                    continue
                r = x.iloc[0].to_dict()
                sc.append({"dataset": ds, "statistic": f"{lab} ({unit})", "adjust": adj, "n": int(r["n_patients"]),
                           "r(early,late)": ci(r, "r_early_late"), "ICC(1,1)": ci(r, "icc1"),
                           "rel shrunk vs pooled": ci(r, "rel_shrunk_vs_pooled"),
                           "rule (CI d<0)": "yes" if r["ci_d_below0"] else "no"})
    lines += [md(pd.DataFrame(sc)), ""]

    # contrast with wave 3
    lines += ["## 6. Contrast with wave-3 slow drug-response edges (real arm; descriptive)", ""]
    ct = []
    for ds in DATASETS:
        for eid, (col, lab, unit) in PRIMARY.items():
            for adj in ("raw", "adj_hs"):
                x = R[(R["dataset"] == ds) & (R["edge_id"] == eid) & (R["adjust"] == adj) & (R["split"] == "S1")]
                if x.empty or x.iloc[0].get("n_patients", 0) < 5:
                    continue
                r = x.iloc[0].to_dict()
                ct.append({"source": f"fast {ds} ({adj})", "edge": f"{eid} {lab}", "outcome": "5-min epochs",
                           "n": int(r["n_patients"]),
                           "units/patient": f"{r['mean_epochs_early'] + r['mean_epochs_late']:.0f}",
                           "r(early,late)": ci(r, "r_early_late"), "rel shrunk vs pooled": ci(r, "rel_shrunk_vs_pooled"),
                           "wave-3 rule holds": "yes" if r["ci_d_below0"] else "no"})
    for _, r in W.iterrows():
        ct.append({"source": r["source"], "edge": r["edge"], "outcome": r["outcome"], "n": r["n"],
                   "units/patient": f"{r['units_per_patient']:.0f}",
                   "r(early,late)": f"{f3(r['r'])} [{f3(r['r_lo'])}, {f3(r['r_hi'])}]",
                   "rel shrunk vs pooled": f"{f3(r['rel'], 3)} [{f3(r['rel_lo'], 3)}, {f3(r['rel_hi'], 3)}]",
                   "wave-3 rule holds": "yes" if r["rule"] else "no"})
    C = pd.DataFrame(ct)
    C.to_csv(OUT / "contrast_fast_vs_slow.csv", index=False)
    lines += [md(C), ""]

    # integrity and synthetic
    lines += ["## 7. Integrity and synthetic check", "",
              "Recomputed epoch values vs wave-2 `epochs.parquet` (same beats, same functions):", "",
              md(integ.round(6)), ""]
    if syn is not None:
        lines += ["Synthetic random-effects check (50 patients, 10-48 epochs each, 200 datasets per scenario; "
                  "`rule_rate` = fraction where the wave-3 rule CI(d) < 0 fires; `perm_p_lt05` = fraction with "
                  "cross-patient-pairing p < 0.05):", "", md(syn.round(3)), ""]
    try:
        em = json.loads((OUT / "extract_meta.json").read_text())
        lines += [f"Runtime: extraction {em['seconds']:.0f} s ({em['workers']} workers); analysis "
                  f"{meta['seconds_analyse']:.0f} s.", ""]
    except Exception:  # noqa: BLE001
        pass
    dev = OUT / "deviations.md"
    lines += ["## 8. Deviations from the protocol", "",
              dev.read_text() if dev.exists() else "None.", ""]
    vd = OUT / "verify"
    if (vd / "v1_independent_recompute.csv").exists():
        v1 = pd.read_csv(vd / "v1_independent_recompute.csv")
        lines += ["## 9. Verification and POST HOC diagnostics (not pre-registered; `verify/`)", "",
                  f"V1 independent re-implementation (plain pandas, S1, raw and adj_hs, {len(v1)} cells): max |diff| "
                  f"in r and ICC = {v1['max_abs_diff'].max():.2g}; max |diff| in the point d = "
                  f"{(v1['d_indep'] - v1['d_pipeline']).abs().max():.2g}.", ""]
    if (vd / "v2_half_gap_hours.csv").exists():
        v2 = pd.read_csv(vd / "v2_half_gap_hours.csv")
        v2 = v2.assign(**{"gap h median [IQR]": v2.apply(
            lambda r: f"{r.gap_h_median:.1f} [{r.gap_h_q25:.1f}, {r.gap_h_q75:.1f}]", axis=1)})
        lines += ["V2 (descriptive) hours between the midpoints of the early and late halves (S1, raw):", "",
                  md(v2[["dataset", "edge", "n", "gap h median [IQR]"]]), ""]
    if (vd / "v3_posthoc_influence_lopo.csv").exists():
        v3 = pd.read_csv(vd / "v3_posthoc_influence_lopo.csv")
        v3 = v3.assign(**{
            "LOPO calibrated MSE ratio [95% CI]": v3.apply(lambda r: f"{r.lopo_mse_ratio:.2f} [{r.lopo_lo:.2f}, {r.lopo_hi:.2f}]", axis=1),
            "d sum: all / 3 largest drifters": v3.apply(lambda r: f"{r.d_all_sum:.3g} / {r.d_top3_drifters_sum:+.3g}", axis=1),
            "d mean without them": v3["d_mean_without_top3"].map(lambda x: f"{x:.3g}")})
        lines += ["V3 POST HOC. (a) Contribution to d = MSE(shrunk) - MSE(pooled) of the 3 patients whose early "
                  "and late means differ most. (b) Leave-one-patient-out calibrated predictor of the late-half mean, "
                  "late_hat = a + b early (a, b fitted without the patient), MSE relative to the leave-one-out "
                  "population mean (< 1 = own early data helps; 2,000 patient-bootstrap CI with the LOPO refit):", "",
                  md(v3[["dataset", "edge", "adjust", "n", "d sum: all / 3 largest drifters", "d mean without them",
                         "LOPO calibrated MSE ratio [95% CI]"]]), ""]
    (OUT / "tables.md").write_text("\n".join(lines))
    make_figure(R, W)
    # summary json
    summ = {"protocol_sha256": meta["protocol_sha256"], "decision": D.to_dict(orient="records")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    log.info("report written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "synthetic", "analyse", "report"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--syn-rep", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "extract":
        stage_extract(a.workers)
    elif a.stage == "synthetic":
        stage_synthetic(a.syn_rep)
    elif a.stage == "analyse":
        stage_analyse()
    else:
        stage_report()


if __name__ == "__main__":
    main()
