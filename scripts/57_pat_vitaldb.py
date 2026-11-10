"""
PAT analyses at VitalDB scale with the delay-aware PAT of scripts/56_pat_native.py.

Pre-registered protocol: runs/iclr/pat_vitaldb/protocol.md (addendum A1 to track-B protocol af14df06; sha256 in
protocol.sha256). Every stage refuses to run if the protocol no longer matches the registered hash.

Stages (PYTHONPATH=src:.):
  python scripts/57_pat_vitaldb.py --stage extract --workers 8      # per-epoch PAT stats (s52.epoch_stats) + nulls
  python scripts/57_pat_vitaldb.py --stage specificity --workers 8  # wave-4 registered test for E1 (PAT-SBP slope)
  python scripts/57_pat_vitaldb.py --stage report

Unchanged imports: clbench.edges.fast_coupling.pat_coupling; script 52 (epoch_stats, build_groups, split_stats,
winsorise, wpearson); script 54 (boot_cases, boot_cluster, spearman_boot, within_slope, verdict, fast_groups, md).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clbench.edges.fast_coupling import pat_coupling  # noqa: E402


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s52 = _load("s52", "52_fast_edge_specificity.py")
s54 = _load("s54", "54_fast_edges_vitaldb.py")

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("57_pat_vdb")

OUT = ROOT / "runs" / "iclr" / "pat_vitaldb"
FB = ROOT / "runs" / "iclr" / "fast_vitaldb"
NATIVE = ROOT / "data" / "processed_vitaldb" / "native"
MANIFEST = ROOT / "data" / "vitaldb_manifest.csv"
PROTOCOL = OUT / "protocol.md"
PROTOCOL_SHA_FILE = OUT / "protocol.sha256"

SEED = 20260925  # af14df06 epoch RNG seed
K_NULL = 5
N_BOOT = 2000
PAT_COLS = ["pat_sbp_slope", "pat_sbp_corr", "n_pat", "twin_pat"]
INTEGRITY_COLS = ["hr_mean", "sbp_mean", "sdnn", "rmssd", "seq_pooled_brs", "twin_seq", "reg_brs", "rsa_h", "ppv", "spv"]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_protocol() -> str:
    h = sha256(PROTOCOL)
    reg = PROTOCOL_SHA_FILE.read_text().split()[0]
    if h != reg:
        raise SystemExit(f"protocol.md sha256 {h} != registered {reg}; refusing to run the analysis")
    return h


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# PAT QC (protocol section 1)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def pat_qc() -> pd.DataFrame:
    C = pd.read_parquet(FB / "cases.parquet")
    C = C[C["g4_20min"] == True][["patient_id", "case_index"]].copy()  # noqa: E712
    C["case_index"] = C["case_index"].astype(int)
    rows = []
    for pid in C["patient_id"]:
        f = NATIVE / "pat_summaries" / f"{pid}.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        rows.append({"patient_id": pid, "has_summary": f.exists(),
                     "has_pat_file": (NATIVE / "pat" / f"{pid}.parquet").exists(),
                     "clock_warning": d.get("clock_warning"), "delay_mode_s": d.get("delay_mode_s", np.nan),
                     "coverage_valid": d.get("coverage_valid", np.nan), "half_width_s": d.get("half_width_s", np.nan)})
    Q = C.merge(pd.DataFrame(rows), on="patient_id")
    Q["q0_file"] = Q["has_summary"] & Q["has_pat_file"]
    Q["q1_clock"] = Q["q0_file"] & Q["clock_warning"].isna()
    Q["q2_delay"] = Q["q1_clock"] & (Q["delay_mode_s"] >= 0.45) & (Q["delay_mode_s"] <= 0.90)
    Q["q3_cover"] = Q["q2_delay"] & (Q["coverage_valid"] >= 0.80)
    return Q


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage extract: per-epoch PAT statistics (s52.epoch_stats with script-56 PAT) and within-case nulls
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def _slope(pat, sbp, v) -> float:
    return pat_coupling(pat, sbp, v)["pat_sbp_slope"]


def process_case(args) -> dict:
    ci, pid, eps = args  # eps: list of (epoch_index, t0, t1) of the case's usable epochs (track B)
    out = {"patient_id": pid, "case_index": ci, "rows": [], "error": None, "aligned": None}
    try:
        beats = pd.read_parquet(NATIVE / "beats" / f"{pid}.parquet")
        pf = pd.read_parquet(NATIVE / "pat" / f"{pid}.parquet")
        aligned = len(pf) == len(beats) and np.array_equal(pf["t_r"].to_numpy(float), beats["t_r"].to_numpy(float))
        out["aligned"] = bool(aligned)
        if not aligned:
            return out
        beats = beats.copy()
        beats["pat"] = pf["pat_ms"].to_numpy(float)
        tr = beats["t_r"].to_numpy(float)
        seg = {}
        for ei, t0, t1 in eps:
            lo, hi = np.searchsorted(tr, t0, "left"), np.searchsorted(tr, t1, "left")
            seg[ei] = (lo, hi)
        ei_list = [e for e, *_ in eps]
        pat_all = beats["pat"].to_numpy(float)
        sbp_all = beats["sbp"].to_numpy(float)
        v_all = beats["valid"].to_numpy(bool)
        for ei, t0, t1 in eps:
            lo, hi = seg[ei]
            b = beats.iloc[lo:hi]
            rng = np.random.default_rng([SEED, ci, ei])
            st = s52.epoch_stats(b, rng)
            pc = pat_coupling(b["pat"].to_numpy(float), b["sbp"].to_numpy(float), b["valid"].to_numpy(bool))
            row = {"patient_id": pid, "epoch_index": ei, "t0": t0, "t1": t1,
                   **{c: st.get(c, np.nan) for c in PAT_COLS + INTEGRITY_COLS},
                   "pat_median": pc["pat_median"],
                   "pat_cover": float(np.isfinite(pat_all[lo:hi][v_all[lo:hi]]).mean()) if v_all[lo:hi].any() else np.nan}
            if np.isfinite(row["pat_sbp_slope"]):
                pat, sbp, v = pat_all[lo:hi], sbp_all[lo:hi], v_all[lo:hi]
                trr = tr[lo:hi] - tr[lo]
                # N-a: within-epoch circular shift of SBP (with validity) by s ~ U(60, 240) s
                ra = np.random.default_rng([SEED, ci, ei, 57, 1])
                for k in range(K_NULL):
                    s = ra.uniform(60.0, 240.0)
                    u = int(np.searchsorted(trr, s, "left"))
                    row[f"na_{k}"] = _slope(pat, np.roll(sbp, u), v & np.roll(v, u))
                # N-b: PAT of this epoch vs SBP of another usable epoch of the same case
                others = [e for e in ei_list if e != ei]
                rb = np.random.default_rng([SEED, ci, ei, 57, 2])
                for k in range(K_NULL):
                    if not others:
                        row[f"nb_{k}"] = np.nan
                        continue
                    e2 = others[int(rb.integers(0, len(others)))]
                    lo2, hi2 = seg[e2]
                    m = min(hi - lo, hi2 - lo2)
                    row[f"nb_{k}"] = _slope(pat[:m], sbp_all[lo2:lo2 + m], v[:m] & v_all[lo2:lo2 + m])
            out["rows"].append(row)
    except Exception as ex:  # noqa: BLE001
        import traceback
        out["error"] = f"{type(ex).__name__}: {ex} | {traceback.format_exc()[-400:]}"
    return out


def stage_extract(workers: int, limit: int | None = None) -> None:
    psha = check_protocol()
    t0 = time.time()
    Q = pat_qc()
    E = pd.read_parquet(FB / "epochs.parquet", columns=["patient_id", "epoch_index", "t0", "t1", "usable"])
    E = E[E["usable"]]
    by = {p: list(zip(g["epoch_index"].astype(int), g["t0"].astype(float), g["t1"].astype(float)))
          for p, g in E.groupby("patient_id")}
    jobs = [(int(ci), pid, by.get(pid, [])) for ci, pid, ok in zip(Q["case_index"], Q["patient_id"], Q["q0_file"]) if ok]
    if limit:
        jobs = jobs[:limit]
    rows, metas = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(process_case, jobs, chunksize=4)):
            rows.extend(r.pop("rows"))
            metas.append(r)
            if (i + 1) % 250 == 0:
                log.info("extract %d/%d cases (%.0f s)", i + 1, len(jobs), time.time() - t0)
    sfx = "" if not limit else f"_limit{limit}"
    M = pd.DataFrame(metas)
    Q = Q.merge(M[["patient_id", "aligned", "error"]], on="patient_id", how="left")
    Q["q0_file"] = Q["q0_file"] & (Q["aligned"] == True)  # noqa: E712  (misaligned files excluded, counted)
    for g in ("q1_clock", "q2_delay", "q3_cover"):
        Q[g] = Q[g] & Q["q0_file"]
    Q.to_parquet(OUT / f"cases{sfx}.parquet", index=False)
    pd.DataFrame(rows).to_parquet(OUT / f"epochs_pat{sfx}.parquet", index=False)
    (OUT / f"extract_meta{sfx}.json").write_text(json.dumps({
        "protocol_sha256": psha, "script_sha256": sha256(Path(__file__)),
        "s52_sha256": sha256(ROOT / "scripts" / "52_fast_edge_specificity.py"),
        "s54_sha256": sha256(ROOT / "scripts" / "54_fast_edges_vitaldb.py"),
        "s56_sha256": sha256(ROOT / "scripts" / "56_pat_native.py"),
        "fast_coupling_sha256": sha256(ROOT / "src" / "clbench" / "edges" / "fast_coupling.py"),
        "n_cases": len(jobs), "n_epoch_rows": len(rows), "n_misaligned": int((M["aligned"] == False).sum()),  # noqa: E712
        "n_errors": int(M["error"].notna().sum()), "seconds": time.time() - t0, "workers": workers,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    log.info("extract done: %d cases, %d epochs, %.0f s", len(jobs), len(rows), time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage specificity: the wave-4 registered test for E1 on the track-B frame with the script-56 PAT columns
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

REPL = ["pat_sbp_slope", "pat_sbp_corr", "twin_pat", "n_pat"]


def spec_frame() -> pd.DataFrame:
    """s54.prepare_vdb with the PAT columns replaced before winsorisation (protocol section 5)."""
    df = pd.read_parquet(FB / "epochs.parquet")
    cases = pd.read_parquet(FB / "cases.parquet")
    keep = set(cases.loc[cases["g4_20min"] == True, "patient_id"])  # noqa: E712
    df = df[df["usable"] & df["patient_id"].isin(keep)].copy()
    df["dataset"] = "vitaldb"
    df["remifentanil_ce0"] = df["remifentanil_ce"].fillna(0.0).astype(float)
    df["propofol_rec"] = df["propofol_ce"].notna().astype(float)
    df["propofol_ce0"] = df["propofol_ce"].fillna(0.0).astype(float)
    P = pd.read_parquet(OUT / "epochs_pat.parquet", columns=["patient_id", "epoch_index"] + REPL)
    Q = pd.read_parquet(OUT / "cases.parquet")
    P = P[P["patient_id"].isin(set(Q.loc[Q["q3_cover"], "patient_id"]))]
    df = df.drop(columns=REPL).merge(P, on=["patient_id", "epoch_index"], how="left")
    cols = [c for c, *_ in s52.PRIMARY.values()] + [c for c, *_ in s52.SECONDARY.values()] + list(s52.TWIN.values()) + \
        s52.COV_HS + ["remifentanil_ce0", "propofol_ce0"]
    return s52.winsorise(df, cols)


def _spec_task(args) -> dict:
    kind, eid, col, lab, unit, adj, covs, splits = args
    df = pd.read_parquet(OUT / "spec_frame.parquet")
    rows, pat_rows = [], []
    for split in splits:
        if split == "S3":
            rng = np.random.default_rng(7)
            draws = [s52.split_stats(s54.fast_groups(df, col, covs, "S3", rng)[0], full=False) for _ in range(s52.N_SHUFFLE)]
            D = pd.DataFrame(draws)
            r = {"edge_id": eid, "column": col, "label": lab, "unit": unit, "kind": kind, "adjust": adj, "split": "S3",
                 "n_patients": int(D["n_patients"].median())}
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
                     "split": split, **res})
        if split == "S1":
            fg, fp = s54.fast_groups(df, col, covs, "S1")
            same = (fp == pids) and all(np.allclose(a, c) and np.allclose(b, d_) for (a, b), (c, d_) in zip(groups, fg))
            rows[-1]["fast_groups_equal_s52"] = bool(same)
            for pp, (a, b) in zip(pids, groups):
                pat_rows.append({"edge_id": eid, "column": col, "adjust": adj, "patient_id": pp, "n_early": len(a),
                                 "n_late": len(b), "early_mean": a.mean(), "late_mean": b.mean()})
    return {"rows": rows, "pat": pat_rows}


def stage_specificity(workers: int) -> None:
    psha = check_protocol()
    t0 = time.time()
    df = spec_frame()
    # integrity: every non-PAT column equals track B's prepare_vdb frame
    ref = s54.prepare_vdb().sort_values(["patient_id", "epoch_index"]).reset_index(drop=True)
    d2 = df.sort_values(["patient_id", "epoch_index"]).reset_index(drop=True)
    check_cols = [c for c in ref.columns if c not in REPL and c in d2.columns and pd.api.types.is_numeric_dtype(ref[c])]
    diffs = {c: int((~np.isclose(ref[c].to_numpy(float), d2[c].to_numpy(float), equal_nan=True)).sum()) for c in check_cols}
    df.to_parquet(OUT / "spec_frame.parquet", index=False)
    col, lab, unit = s52.PRIMARY["E1"]
    tasks = []
    adjs = {"raw": [], "adj_hs": s52.COV_HS, "adj_hst": s52.COV_HS + [s52.TWIN[col]], "adj_hsd": s52.COV_HS + s52.COV_DRUG}
    for adj, covs in adjs.items():
        tasks.append(("primary", "E1", col, lab + " (script-56 delay-aware PAT)", unit, adj, covs, ("S1", "S2", "S4", "S3")))
    c2, l2, u2 = s52.SECONDARY["S-patcorr"]
    for adj, covs in {"raw": [], "adj_hs": s52.COV_HS, "adj_hsd": s52.COV_HS + s52.COV_DRUG}.items():
        tasks.append(("secondary", "S-patcorr", c2, l2 + " (script-56 PAT)", u2, adj, covs, ("S1",)))
    tasks.append(("twin", f"T-{col}", s52.TWIN[col], f"twin of {col}", "", "raw", [], ("S1",)))
    for eid, (c, l_, u_) in s54.OPERATING.items():
        tasks.append(("operating", eid, c, l_, u_, "raw", [], ("S1",)))
    tasks.sort(key=lambda t: -len(t[7]))
    rows, pats = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_spec_task, tasks)):
            rows.extend(res["rows"])
            pats.extend(res["pat"])
            log.info("spec task %d/%d done (%.0f s)", i + 1, len(tasks), time.time() - t0)
    R = pd.DataFrame(rows)
    R["ci_d_below0"] = R["d_shrunk_minus_pooled_hi"] < 0
    # Holm over the 5 primary edges (adj_hs, S1): E2-E5 from track B unchanged, E1 replaced
    TB = pd.read_csv(FB / "specificity_all.csv")
    tb = TB[(TB["kind"] == "primary") & (TB["adjust"] == "adj_hs") & (TB["split"] == "S1") & (TB["edge_id"] != "E1")]
    e1 = R[(R["edge_id"] == "E1") & (R["adjust"] == "adj_hs") & (R["split"] == "S1")]
    ids = list(tb["edge_id"]) + ["E1"]
    p = np.concatenate([tb["perm_p_r"].to_numpy(float), e1["perm_p_r"].to_numpy(float)])
    order = np.argsort(p)
    holm = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        holm[i] = running
    holm_tab = pd.DataFrame({"edge_id": ids, "perm_p_r_adj_hs": p, "holm": holm,
                             "source": ["track B specificity_all.csv"] * len(tb) + ["this run"]})
    R["perm_p_r_holm_adj_hs"] = np.nan
    R.loc[(R["edge_id"] == "E1") & (R["adjust"] == "adj_hs") & (R["split"] == "S1"), "perm_p_r_holm_adj_hs"] = holm[-1]
    R.to_csv(OUT / "specificity_all.csv", index=False)
    holm_tab.to_csv(OUT / "specificity_holm.csv", index=False)
    P = pd.DataFrame(pats)
    P.to_csv(OUT / "specificity_patient_halves.csv", index=False)
    comp = []
    hm = P[P["edge_id"].isin(["O-hr", "O-sbp"])]
    for eid in ("E1", "S-patcorr"):
        for adj in ("raw", "adj_hs"):
            e = P[(P["edge_id"] == eid) & (P["adjust"] == adj)]
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
        "protocol_sha256": psha, "parent_protocol_sha256": sha256(FB / "protocol.md"),
        "wave4_protocol_sha256": s52.sha256(s52.PROTOCOL), "script_sha256": sha256(Path(__file__)),
        "frame_nonpat_columns_differing_from_trackB": diffs, "n_frame_rows": len(df), "n_ref_rows": len(ref),
        "n_boot": s52.N_BOOT, "n_perm": s52.N_PERM, "n_shuffle": s52.N_SHUFFLE, "seconds": time.time() - t0,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=2))
    log.info("specificity done %.0f s", time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage report
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def case_medians(E: pd.DataFrame, col: str) -> pd.Series:
    return E.groupby("patient_id")[col].median()


def pop_rows(E: pd.DataFrame, label: str) -> list[dict]:
    out = []
    by = {p: g for p, g in E.groupby("patient_id")}
    for col, name in (("pat_sbp_slope", "PAT-SBP slope (ms/mmHg)"), ("pat_sbp_corr", "PAT-SBP beat correlation (r)")):
        groups = [g[col].to_numpy(float) for g in by.values()]
        med = np.array([np.nanmedian(g) if np.isfinite(g).any() else np.nan for g in groups])
        ok = np.isfinite(med)
        cs = s54.boot_cases(med[ok], lambda a: np.mean(np.sign(a) == -1))
        out.append({"case set": label, "quantity": name, "expected": "-", "n_cases": int(ok.sum()),
                    "n_epochs": int(sum(np.isfinite(g).sum() for g in groups)),
                    "cases negative": s54.fmt(cs, pct=True), "_cs": cs,
                    "epochs negative (cluster CI)": s54.fmt(s54.boot_cluster(groups, lambda a: np.mean(np.sign(a) == -1)), pct=True),
                    "median of case medians": s54.fmt(s54.boot_cases(med[ok], np.median), nd=3),
                    "IQR of case medians": f"{np.nanpercentile(med, 25):.3f} to {np.nanpercentile(med, 75):.3f}" if ok.any() else "n/a"})
    return out


def null_rows(E: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Protocol section 3: observed vs null case share negative, paired case bootstrap."""
    res, summ = [], {}
    E = E[np.isfinite(E["pat_sbp_slope"])]
    for nid, pref, label in (("N-a", "na", "within-epoch circular shift of SBP by 60-240 s"),
                             ("N-b", "nb", "PAT paired with SBP of another usable epoch of the same case"),
                             ("T1", None, "registered T1 twin (shift 30..n-30 beats, mean of 5; descriptive)")):
        if pref:
            cols = [f"{pref}_{k}" for k in range(K_NULL)]
        else:
            cols = ["twin_pat"]
        obs = E.groupby("patient_id")["pat_sbp_slope"].median()
        nulls = []
        for c in cols:
            d = E[np.isfinite(E[c])]
            nulls.append(d.groupby("patient_id")[c].median())
        common = obs.index
        for n_ in nulls:
            common = common.intersection(n_.index)
        o = (obs.reindex(common).to_numpy(float) < 0).astype(float)
        nm = np.column_stack([(n_.reindex(common).to_numpy(float) < 0).astype(float) for n_ in nulls]).mean(axis=1)
        diff = o - nm
        rng = np.random.default_rng(0)
        bs_d, bs_n = [], []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(diff), len(diff))
            bs_d.append(diff[k].mean())
            bs_n.append(nm[k].mean())
        ep_null = np.mean([np.mean(E[c].dropna() < 0) for c in cols])
        t = (float(diff.mean()), float(np.quantile(bs_d, 0.025)), float(np.quantile(bs_d, 0.975)))
        res.append({"null": nid, "construction": label, "n_cases (common)": len(common),
                    "observed cases negative": f"{100 * o.mean():.1f}%",
                    "null cases negative (mean over draws) [95% CI]": f"{100 * nm.mean():.1f}% [{100 * np.quantile(bs_n, 0.025):.1f}, "
                                                                       f"{100 * np.quantile(bs_n, 0.975):.1f}]",
                    "null epochs negative": f"{100 * ep_null:.1f}%",
                    "observed - null (pp) [95% CI]": f"{100 * t[0]:.1f} [{100 * t[1]:.1f}, {100 * t[2]:.1f}]",
                    "difference CI above 0": bool(t[1] > 0),
                    "null median slope (median of case medians)": f"{np.nanmedian(np.column_stack([n_.reindex(common).to_numpy(float) for n_ in nulls]).mean(axis=1)):.3f}"})
        summ[nid] = {"diff": t, "null_case_share": float(nm.mean()), "n": len(common)}
    return pd.DataFrame(res), summ


def stage_report() -> None:
    psha = check_protocol()
    Q = pd.read_parquet(OUT / "cases.parquet")
    E = pd.read_parquet(OUT / "epochs_pat.parquet")
    TB = pd.read_parquet(FB / "epochs.parquet")
    man = pd.read_csv(MANIFEST).set_index("patient_id")
    meta = json.loads((OUT / "extract_meta.json").read_text())
    lines = [f"# PAT at VitalDB scale, delay-aware PAT (addendum A1 to af14df06; protocol sha256 {psha[:8]})", ""]
    summ: dict = {"protocol_sha256": psha, "parent_protocol_sha256": sha256(FB / "protocol.md")}

    # ---- QC gates ----
    gates = {"track-B QC cases (g4_20min)": len(Q), "PAT summary + file present, rows aligned": int(Q["q0_file"].sum()),
             "misaligned PAT files": meta["n_misaligned"], "no clock_warning": int(Q["q1_clock"].sum()),
             "delay_mode_s in [0.45, 0.90] s": int(Q["q2_delay"].sum()), "coverage_valid >= 0.80 (PAT-QC set)": int(Q["q3_cover"].sum()),
             "extract errors": meta["n_errors"]}
    sets = {"PAT-QC (primary)": set(Q.loc[Q["q3_cover"], "patient_id"]),
            "R2: clock_warning excluded only": set(Q.loc[Q["q1_clock"], "patient_id"]),
            "R1: no PAT QC (all track-B QC cases)": set(Q.loc[Q["q0_file"], "patient_id"])}
    Eq = E[E["patient_id"].isin(sets["PAT-QC (primary)"])]
    gates["usable epochs, PAT-QC cases"] = int(len(Eq))
    gates["epochs with a PAT-SBP slope (>= 30 valid PAT beats), PAT-QC"] = int(np.isfinite(Eq["pat_sbp_slope"]).sum())
    gates["cases with >= 1 slope, PAT-QC"] = int(Eq.loc[np.isfinite(Eq["pat_sbp_slope"]), "patient_id"].nunique())
    gates["median per-epoch PAT coverage of valid beats, PAT-QC"] = round(float(np.nanmedian(Eq["pat_cover"])), 4)
    summ["qc"] = gates
    lines += ["## QC gates (protocol section 1)", "", s54.md(pd.DataFrame({"gate": list(gates), "n": list(gates.values())})), ""]

    # ---- integrity: non-PAT epoch_stats columns equal track B ----
    J = E.merge(TB[["patient_id", "epoch_index"] + INTEGRITY_COLS], on=["patient_id", "epoch_index"], suffixes=("", "_tb"))
    integ = {c: int((~np.isclose(J[c].to_numpy(float), J[c + "_tb"].to_numpy(float), equal_nan=True)).sum()) for c in INTEGRITY_COLS}
    summ["integrity_mismatches_vs_trackB"] = integ
    summ["integrity_rows_compared"] = len(J)
    lines += ["## Integrity", "", f"epoch_stats non-PAT columns vs track B on {len(J)} epochs, mismatching epochs per column: "
              + json.dumps(integ), ""]

    # ---- population (section 2) ----
    pop = []
    for lab, s in sets.items():
        pop += pop_rows(E[E["patient_id"].isin(s)], lab)
    pop_df = pd.DataFrame(pop)
    prim = pop_df[(pop_df["case set"] == "PAT-QC (primary)") & (pop_df["quantity"].str.startswith("PAT-SBP slope"))].iloc[0]
    cs = prim["_cs"]
    reading = {"case_share_negative": cs, "above_chance (CI > 0.5)": bool(cs[1] > 0.5),
               "positive_control (CI lower >= 0.90)": bool(cs[1] >= 0.90)}
    summ["population"] = pop_df.drop(columns="_cs").to_dict(orient="records")
    summ["reading"] = reading
    pop_df.drop(columns="_cs").to_csv(OUT / "population_signs.csv", index=False)
    # distributions (descriptive; PAT level includes the device delay)
    dist = []
    for col, label in (("pat_median", "epoch median PAT (ms), INCLUDES the per-case PLETH device delay"),
                       ("pat_sbp_slope", "PAT-SBP slope (ms/mmHg)"), ("pat_sbp_corr", "PAT-SBP correlation (r)")):
        med = case_medians(Eq, col).to_numpy(float)
        ok = np.isfinite(med)
        dist.append({"quantity (PAT-QC cases)": label, "n_cases": int(ok.sum()),
                     "median of case medians [95% CI]": s54.fmt(s54.boot_cases(med[ok], np.median)),
                     "IQR": f"{np.nanpercentile(med, 25):.3f} to {np.nanpercentile(med, 75):.3f}",
                     "5-95%": f"{np.nanpercentile(med, 5):.3f} to {np.nanpercentile(med, 95):.3f}"})
    dist_df = pd.DataFrame(dist)
    dist_df.to_csv(OUT / "population_distributions.csv", index=False)
    lines += ["## Population sign (af14df06 section 3 estimators; case-cluster bootstrap, 2,000 replicates)", "",
              s54.md(pop_df.drop(columns="_cs")), "",
              f"Reading (registered in A1): case share negative {s54.fmt(cs, pct=True)}; above chance (CI > 50%): "
              f"{reading['above_chance (CI > 0.5)']}; positive-control bar (CI lower bound >= 90%): "
              f"{reading['positive_control (CI lower >= 0.90)']}", "",
              "Distributions of case medians (PAT-QC):", "", s54.md(dist_df), ""]

    # ---- nulls (section 3) ----
    null_df, null_s = null_rows(Eq)
    null_df.to_csv(OUT / "nulls.csv", index=False)
    art_ok = bool(null_s["N-a"]["diff"][1] > 0 and null_s["N-b"]["diff"][1] > 0)
    summ["nulls"] = null_df.to_dict(orient="records")
    summ["not_mechanical_artefact"] = art_ok
    lines += ["## Within-case nulls (section 3; PAT-QC cases, K = 5 draws per epoch)", "", s54.md(null_df), "",
              f"Reading: not a mechanical artefact (difference CI above 0 under both N-a and N-b): {art_ok}", ""]

    # ---- construct validity (section 4) ----
    age = pd.to_numeric(man["age"], errors="coerce")
    cv = []
    cp = case_medians(Eq, "pat_median")
    a = age.reindex(cp.index).to_numpy(float)
    ok = a >= 18
    t, n = s54.spearman_boot(a[ok], cp.to_numpy(float)[ok])
    cv.append({"id": "CV7", "relation": "age vs case-median PAT (adults), Spearman rho", "n": n, "estimate [95% CI]": s54.fmt(t, nd=3),
               "expected": "< 0", "rule reads": s54.verdict(t, -1),
               "status": "NOT physiologically interpretable: between-case PAT includes the per-case device delay; not counted in the CV tally"})
    for cid, xcol, min_sd, exp, lab in (("CV7w", "map_mean", 3.0, -1, "within-case epoch PAT (ms) vs MAP (mmHg), case FE"),
                                        ("CV7w-prop", "propofol_ce", 0.1, 0, "within-case epoch PAT (ms) vs propofol Ce (ug/ml), case FE"),
                                        ("CV7w-bis", "bis", 3.0, 0, "within-case epoch PAT (ms) vs BIS, case FE")):
        d = Eq[["patient_id", "epoch_index", "pat_median"]].merge(TB[["patient_id", "epoch_index", xcol]], on=["patient_id", "epoch_index"])
        t, nc, ne = s54.within_slope(d, xcol, "pat_median", min_sd, log_y=False)
        cv.append({"id": cid, "relation": lab, "n": f"{nc} cases, {ne} epochs", "estimate [95% CI]": f"slope {s54.fmt(t, nd=4)}",
                   "expected": "< 0" if exp < 0 else "none registered", "rule reads": s54.verdict(t, exp) if exp else "descriptive",
                   "status": "registered in A1 (consistency check; truncation biases toward 0)" if exp else "descriptive only"})
    cv_df = pd.DataFrame(cv)
    cv_df.to_csv(OUT / "construct_validity.csv", index=False)
    summ["construct_validity"] = cv_df.to_dict(orient="records")
    lines += ["## Construct validity (section 4)", "", s54.md(cv_df), ""]

    # ---- specificity (section 5) ----
    spec_ok = (OUT / "specificity_all.csv").exists()
    if spec_ok:
        R = pd.read_csv(OUT / "specificity_all.csv")
        sm = json.loads((OUT / "specificity_meta.json").read_text())
        H = pd.read_csv(OUT / "specificity_holm.csv")

        def get(eid, adj, split="S1"):
            x = R[(R["edge_id"] == eid) & (R["adjust"] == adj) & (R["split"] == split)]
            return x.iloc[0] if len(x) else None

        raw, hs, hst, hsd, s2, s4 = get("E1", "raw"), get("E1", "adj_hs"), get("E1", "adj_hst"), get("E1", "adj_hsd"), \
            get("E1", "adj_hs", "S2"), get("E1", "adj_hs", "S4")
        p1, p2 = bool(raw["ci_d_below0"]), bool(hs["ci_d_below0"])
        dec = {"edge": "E1 PAT-SBP slope (script-56 PAT)", "n": int(raw["n_patients"]), "P1 raw": p1, "P2 HR/SBP-adj": p2,
               "REAL": "yes" if (p1 and p2) else "not shown", "twin-adj (H3)": bool(hst["ci_d_below0"]),
               "drug-adj": bool(hsd["ci_d_below0"]), "S2 adj holds": bool(s2["ci_d_below0"]) if s2 is not None else None,
               "S4 adj holds": bool(s4["ci_d_below0"]) if s4 is not None else None,
               "perm p (r, adj_hs)": float(hs["perm_p_r"]), "Holm p (5 primary edges)": float(H.loc[H["edge_id"] == "E1", "holm"].iloc[0]),
               "fast grouping = s52": bool(raw["fast_groups_equal_s52"])}
        S1 = R[R["split"] == "S1"].copy()
        tab = pd.DataFrame({
            "edge": S1["edge_id"], "kind": S1["kind"], "adjust": S1["adjust"], "n": S1["n_patients"],
            "epochs/half": S1.apply(lambda r: f"{r.get('mean_epochs_early', np.nan):.0f}/{r.get('mean_epochs_late', np.nan):.0f}", axis=1),
            "r(early,late)": S1.apply(lambda r: f"{r.r_early_late:.2f} [{r.pearson_lo:.2f}, {r.pearson_hi:.2f}]", axis=1),
            "ICC(1,1)": S1.apply(lambda r: f"{r.icc1:.2f} [{r.icc1_lo:.2f}, {r.icc1_hi:.2f}]", axis=1),
            "MSE shrunk - pooled": S1.apply(lambda r: f"{r.d_shrunk_minus_pooled:.3g} [{r.d_shrunk_minus_pooled_lo:.3g}, {r.d_shrunk_minus_pooled_hi:.3g}]", axis=1),
            "rel shrunk vs pooled": S1.apply(lambda r: f"{r.rel_shrunk_vs_pooled:.3f} [{r.rel_shrunk_vs_pooled_lo:.3f}, {r.rel_shrunk_vs_pooled_hi:.3f}]", axis=1),
            "d<0 CI": S1["ci_d_below0"], "perm p(r)": S1["perm_p_r"].map(lambda x: f"{x:.3g}")})
        tab.to_csv(OUT / "specificity_main_table.csv", index=False)
        oth = R[R["split"] != "S1"][["edge_id", "adjust", "split", "n_patients", "r_early_late", "r_early_late_lo", "r_early_late_hi"]
                                    + [c for c in ("d_shrunk_minus_pooled", "d_shrunk_minus_pooled_lo", "d_shrunk_minus_pooled_hi") if c in R]]
        comp = pd.read_csv(OUT / "specificity_vs_operating_point.csv")
        comp["reading"] = np.where(comp["diff_hi"] < 0, "less persistent than operating point",
                                   np.where(comp["diff_lo"] > 0, "more persistent", "as persistent"))
        comp["diff [95% CI]"] = comp.apply(lambda r: f"{r['diff']:.2f} [{r.diff_lo:.2f}, {r.diff_hi:.2f}]", axis=1)
        pd.DataFrame([dec]).to_csv(OUT / "specificity_decision.csv", index=False)
        summ["specificity_decision"] = dec
        summ["specificity_frame_integrity"] = sm["frame_nonpat_columns_differing_from_trackB"]
        lines += ["## Patient specificity of the PAT-SBP slope (section 5; wave-4 registered test)", "",
                  s54.md(pd.DataFrame([dec])), "", "Holm over the 5 primary edges (adj_hs, S1):", "", s54.md(H.round(4)), "",
                  "S1 statistics:", "", s54.md(tab), "", "S2 / S3 / S4:", "", s54.md(oth.round(3)), "",
                  "Persistence relative to the operating point (paired case bootstrap):", "",
                  s54.md(comp[["edge_id", "adjust", "vs", "n_common", "r_edge", "r_oper", "diff [95% CI]", "reading"]].round(3)), "",
                  "Frame integrity (non-PAT columns differing from track B's prepare_vdb frame): "
                  + json.dumps(sm["frame_nonpat_columns_differing_from_trackB"]), ""]

    # ---- F4 admission reading (section 6) ----
    a_ok = reading["positive_control (CI lower >= 0.90)"]
    f4 = {"(a) case-share CI lower bound >= 0.90 (F3 rule)": a_ok, "(b) not a mechanical artefact (N-a and N-b)": art_ok,
          "population sign meets the F4 admission rule": bool(a_ok and art_ok)}
    summ["f4_admission_reading"] = f4
    lines += ["## F4 admission reading (section 6; input to the next stage)", "", json.dumps(f4, indent=1), ""]

    # ---- exploratory post hoc (labelled; registered readings above are unchanged) ----
    if (OUT / "exploratory_posthoc.csv").exists():
        X = pd.read_csv(OUT / "exploratory_posthoc.csv")
        summ["exploratory_posthoc"] = X.to_dict(orient="records")
        lines += ["## EXPLORATORY / post hoc (not in addendum A1; registered readings above are unchanged)", "",
                  "Within-case nulls break beat alignment, so they cannot detect a beat-aligned measurement coupling. "
                  "The PLETH foot (intersecting tangent) could shift with pulse amplitude, which co-varies with SBP. "
                  "Check: per epoch, PAT regressed on detrended SBP and detrended PPG amplitude (31-beat detrend).", "",
                  s54.md(X), ""]
    (OUT / "tables.md").write_text("\n".join(lines))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    log.info("report written")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Stage posthoc (EXPLORATORY, not in addendum A1): is the beat-aligned PAT-SBP slope carried by the PLETH pulse
# amplitude (a foot-timing measurement coupling that the within-case nulls cannot detect)?
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def amp_case(args) -> list[dict]:
    from clbench.edges.fast_coupling import running_mean_detrend
    pid, eps = args
    rows = []
    beats = pd.read_parquet(NATIVE / "beats" / f"{pid}.parquet", columns=["t_r", "sbp", "valid"])
    pf = pd.read_parquet(NATIVE / "pat" / f"{pid}.parquet")
    tr = beats["t_r"].to_numpy(float)
    pat_all, amp_all = pf["pat_ms"].to_numpy(float), pf["ppg_amp"].to_numpy(float)
    sbp_all, v_all = beats["sbp"].to_numpy(float), beats["valid"].to_numpy(bool)
    for ei, t0, t1 in eps:
        lo, hi = np.searchsorted(tr, t0, "left"), np.searchsorted(tr, t1, "left")
        pat, sbp, amp, v = pat_all[lo:hi], sbp_all[lo:hi], amp_all[lo:hi], v_all[lo:hi]
        ok = v & np.isfinite(pat) & np.isfinite(sbp) & np.isfinite(amp) & (amp > 0)
        r = {"patient_id": pid, "epoch_index": ei, "slope_same_beats": np.nan, "slope_sbp_given_amp": np.nan,
             "slope_pat_on_amp": np.nan, "corr_sbp_amp": np.nan}
        if ok.sum() >= 30:
            a, b, c = (running_mean_detrend(x, ok, 31) for x in (pat, sbp, amp))
            m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            if m.sum() >= 30 and a[m].std() > 0 and b[m].std() > 0 and c[m].std() > 0:
                cs = c[m] / c[m].std()  # amplitude in SD units (PLETH amplitude is in arbitrary units)
                r["slope_same_beats"] = float(np.polyfit(b[m], a[m], 1)[0])
                X = np.column_stack([np.ones(m.sum()), b[m], cs])
                beta, *_ = np.linalg.lstsq(X, a[m], rcond=None)
                r["slope_sbp_given_amp"] = float(beta[1])
                r["slope_pat_on_amp"] = float(np.polyfit(cs, a[m], 1)[0])
                r["corr_sbp_amp"] = float(np.corrcoef(b[m], c[m])[0, 1])
        rows.append(r)
    return rows


def stage_posthoc(workers: int) -> None:
    psha = check_protocol()
    t0 = time.time()
    Q = pd.read_parquet(OUT / "cases.parquet")
    qc = set(Q.loc[Q["q3_cover"], "patient_id"])
    E = pd.read_parquet(OUT / "epochs_pat.parquet", columns=["patient_id", "epoch_index", "t0", "t1", "pat_sbp_slope"])
    E = E[E["patient_id"].isin(qc) & np.isfinite(E["pat_sbp_slope"])]
    jobs = [(p, list(zip(g["epoch_index"].astype(int), g["t0"], g["t1"]))) for p, g in E.groupby("patient_id")]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(amp_case, jobs, chunksize=8):
            rows.extend(r)
    A = pd.DataFrame(rows)
    A.to_parquet(OUT / "posthoc_amp_epochs.parquet", index=False)
    out = []
    for col, lab in (("slope_same_beats", "PAT-SBP slope on the beats with a PPG amplitude (ms/mmHg)"),
                     ("slope_sbp_given_amp", "PAT-SBP slope adjusted for detrended PPG amplitude (ms/mmHg)"),
                     ("slope_pat_on_amp", "PAT on PPG amplitude (ms per SD)"), ("corr_sbp_amp", "corr(SBP, PPG amplitude)")):
        med = A.groupby("patient_id")[col].median().to_numpy(float)
        med = med[np.isfinite(med)]
        out.append({"item": f"EXPLORATORY post hoc: {lab}", "n_cases": len(med),
                    "cases negative": s54.fmt(s54.boot_cases(med, lambda a: np.mean(a < 0)), pct=True),
                    "median of case medians": s54.fmt(s54.boot_cases(med, np.median), nd=3)})
    # CV7w-prop with track B's CV3 exclusion (propofol Ce > 20 ug/ml is physiologically impossible)
    TB = pd.read_parquet(FB / "epochs.parquet", columns=["patient_id", "epoch_index", "propofol_ce"])
    P = pd.read_parquet(OUT / "epochs_pat.parquet", columns=["patient_id", "epoch_index", "pat_median"])
    d = P[P["patient_id"].isin(qc)].merge(TB, on=["patient_id", "epoch_index"])
    bad = d["propofol_ce"] > 20
    t, nc, ne = s54.within_slope(d[~bad], "propofol_ce", "pat_median", 0.1, log_y=False)
    out.append({"item": f"post hoc: CV7w-prop dropping propofol Ce > 20 ug/ml ({int(bad.sum())} epochs; track-B CV3 exclusion)",
                "n_cases": nc, "cases negative": "-", "median of case medians": f"within-case slope {s54.fmt(t, nd=3)} ms per ug/ml ({ne} epochs)"})
    pd.DataFrame(out).to_csv(OUT / "exploratory_posthoc.csv", index=False)
    (OUT / "posthoc_meta.json").write_text(json.dumps({"protocol_sha256": psha, "script_sha256": sha256(Path(__file__)),
                                                       "seconds": time.time() - t0, "label": "EXPLORATORY, not in A1"}, indent=2))
    log.info("posthoc done %.0f s", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["extract", "specificity", "report", "posthoc"], required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "extract":
        stage_extract(a.workers, a.limit)
    elif a.stage == "specificity":
        stage_specificity(a.workers)
    elif a.stage == "posthoc":
        stage_posthoc(a.workers)
    else:
        stage_report()


if __name__ == "__main__":
    main()
