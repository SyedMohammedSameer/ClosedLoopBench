#!/usr/bin/env python
"""
Extended analysis E: sensitivity of admissions to the fixed thresholds.

Part 1  ClosedLoopBench (2 s, 3,442 cases). Re-admission from the raw columns of
        runs/iclr/closed_loop_v4/2s_all/calibrated.csv (p, lo, hi, stable_frac, n_cases) with the paper's rule
        (closed_loop_bench._admit_family / typed_temporal_pooled.admit_multi) and a sweep over
        q, BY vs BH, min_stable, min_cases and the CI condition. No estimator is rerun.
Part 2  Concept-level relations and graphs. Re-admission from the cached pooled fits
        runs/iclr/typed_edges/confirmatory/cache/fits.jsonl (omega, se_omega, exposed events, patients per
        edge, real and 38 null replicates) with the paper's rule (script 47 admissions() + beats-nulls, script 50
        null_admissions()) and a sweep over q, the null rule and the testability thresholds; false relation
        instances per graph are re-instantiated with the paper's pointer rule (script 50 null_counts logic) on
        the cohort arrays. No refit is needed: testability only gates z/p, not the fit.
Part 3  Lag windows: constants only (written in the results note).

Run from the repo root:
    PYTHONPATH=src .venv/bin/python scripts/extended/E_sensitivity.py [--part 1|2|all] [--smoke]
Outputs: runs/extended/E_sensitivity/. Never writes into runs/iclr or data.
Single process; BLAS threads pinned to 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from clbench.edges import typed_temporal as tt  # noqa: E402
from clbench.edges import typed_temporal_pooled as tp  # noqa: E402
from clbench.eval import closed_loop_bench as clb  # noqa: E402

OUT = ROOT / "runs/extended/E_sensitivity"
CALIB_CSV = ROOT / "runs/iclr/closed_loop_v4/2s_all/calibrated.csv"
DOMAIN_CSV = ROOT / "runs/iclr/closed_loop_v4/2s_all/domain_summary.csv"
TE = ROOT / "runs/iclr/typed_edges/confirmatory"
NULL_CSV = ROOT / "runs/iclr/graphs_v2/null_replicates_vitaldb.csv"
NULL_JSON = ROOT / "runs/iclr/graphs_v2/results.json"
CALIB_JSON = ROOT / "checkpoints/cbm/persistence_calibration.json"

# ── estimators ───────────────────────────────────────────────────────────────────────────────────────
NAMES = {
    "xcorr_pos": "Positive-peak cross-corr.", "xcorr_v1": "Signed-peak cross-corr.",
    "xcorr_dir": "Direction-aware cross-corr.", "granger": "Pairwise Granger", "pcmciplus": "PCMCI+",
    "varlingam": "VAR-LiNGAM", "varx": "VARX (typed pooled)", "naive": "Naive event contrast",
    "naive_lvl": "Naive level contrast", "matched": "Matched event design", "kink": "Interrupted trend (kink)",
    "seqtrial": "Sequential-trial contrast", "typed_pooled": "typed_pooled (= VARX base)",
}
LAG = ("xcorr_pos", "xcorr_v1", "xcorr_dir", "granger", "pcmciplus", "varlingam", "varx")
NAIVE = ("naive", "naive_lvl")
DESIGN = ("matched", "kink", "seqtrial")
ORDER = LAG + NAIVE + DESIGN
GROUPS = {"design": DESIGN, "lag": LAG, "naive": NAIVE}
DOMAINS = ("gas_exchange", "drug", "drug_derived", "policy", "cardiopulmonary")
# v4 relation set = v3's 18 + RELATIONS_V4_EXTRA (11); domain of the v3 ones comes from DOMAIN_V3
DOMAIN_OF = {r.rid: clb.relation_domain(r) for r in clb.RELATIONS + clb.RELATIONS_V4_EXTRA}

# default thresholds (ClosedLoopBench protocol 418281f1 / 5e578794; typed_temporal_pooled.MIN_STABLE,
# closed_loop_bench.MIN_CASES_FOR_CI)
P1_DEFAULT = dict(q=0.05, adjust="BY", min_stable=tp.MIN_STABLE, min_cases=clb.MIN_CASES_FOR_CI, ci=True)
P1_RANGES = dict(q=(0.01, 0.025, 0.05, 0.10), adjust=("BY", "BH"), min_stable=(0.5, 0.625, 0.75, 0.875, 1.0),
                 min_cases=(3, 5, 10, 20, 50), ci=(True, False))
# concept level (protocol 43947eaf; typed_temporal.MIN_EXPOSED_EVENTS / MIN_EVENT_PATIENTS)
P2_DEFAULT = dict(q=0.05, rule="max", min_ev=tt.MIN_EXPOSED_EVENTS, min_pat=tt.MIN_EVENT_PATIENTS)
P2_RANGES = dict(q=(0.01, 0.025, 0.05, 0.10), rule=("max", "p95", "bh"), min_ev=(10, 20, 40), min_pat=(5, 10, 20))
FAMILIES = ("association", "policy", "response")
N_NULL = 19

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# multiplicity helpers
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (same structure as tp.benjamini_yekutieli without the c_m factor;
    NaN p-values are not hypotheses). adj <= q  <=>  rejected by controlled.benjamini_hochberg(p, q)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    o = np.argsort(pv, kind="mergesort")
    ranked = pv[o] * m / np.arange(1, m + 1)
    adj = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    res = np.empty(m)
    res[o] = adj
    out[ok] = res
    return out


ADJUST = {"BY": tp.benjamini_yekutieli, "BH": bh_adjust}


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Part 1: ClosedLoopBench
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def p1_admit(C: pd.DataFrame, q: float, adjust: str, min_stable: float, min_cases: float, ci: bool) -> pd.DataFrame:
    """The paper's rule (admit_multi + eligible = n_cases >= MIN_CASES_FOR_CI) re-applied to the raw columns,
    family = the rows of one (input, method); returns C with columns adm / call2 / p_adj."""
    C = C.copy()
    C["p_adj"] = np.nan
    f = ADJUST[adjust]
    for _, idx in C.groupby(["input", "method"], sort=False).groups.items():
        C.loc[idx, "p_adj"] = f(C.loc[idx, "p"].to_numpy(float))
    ci_ex = np.isfinite(C.lo) & np.isfinite(C.hi) & ((C.lo > 0) | (C.hi < 0))
    stable = np.isfinite(C.stable_frac) & (C.stable_frac >= min_stable - 1e-12)
    enough = C.n_cases.to_numpy(float) >= min_cases  # NaN -> False, as in _admit_family
    adm = np.isfinite(C.p_adj) & (C.p_adj <= q) & stable & enough
    if ci:
        adm &= ci_ex
    right = np.sign(C.est) == C.expected
    call = np.where(~np.isfinite(C.est), "na",
                    np.where(adm, np.where(right, "correct", "wrong"), np.where(right, "ns_right", "ns_wrong")))
    C["adm"] = adm.to_numpy(bool)
    C["call2"] = call
    return C


def p1_summary(A: pd.DataFrame) -> pd.DataFrame:
    """Per estimator: expected / wrong admissions on real data (by domain too), held-out-transplant (LOO)
    false admissions over the 8 draws, random-time false admissions, and the R01/R02 calls."""
    rows = []
    for m in ORDER + ("typed_pooled",):
        M = A[A.method == m]
        if not len(M):
            continue
        R = M[M.input == "real"]
        L = M[M.input.str.startswith("loo:")]
        N = M[M.input == "nc_random"]
        per_draw = L.groupby("input").adm.sum()
        d = dict(method=m, expected=int((R.call2 == "correct").sum()), wrong=int((R.call2 == "wrong").sum()),
                 loo_total=int(L.adm.sum()), loo_per_draw=float(per_draw.mean()) if len(per_draw) else np.nan,
                 loo_max=int(per_draw.max()) if len(per_draw) else 0, rt=int(N.adm.sum()))
        for dom in DOMAINS:
            Rd = R[R.rid.map(DOMAIN_OF) == dom]
            d[f"{dom}_E"] = int((Rd.call2 == "correct").sum())
            d[f"{dom}_W"] = int((Rd.call2 == "wrong").sum())
        for rid in ("R01", "R02", "R13", "R14"):
            x = R[R.rid == rid]
            d[f"call_{rid}"] = x.call2.iloc[0] if len(x) else "na"
        rows.append(d)
    return pd.DataFrame(rows)


def p1_groups(S: pd.DataFrame) -> dict:
    d = {}
    for g, ms in GROUPS.items():
        X = S[S.method.isin(ms)]
        for c in ("expected", "wrong", "loo_total", "rt"):
            d[f"{g}_{c}"] = int(X[c].sum())
    # sets of estimators with any wrong-sign / null false admission
    X = S[S.method != "typed_pooled"]
    d["set_wrong"] = ",".join(X[X.wrong > 0].method)
    d["set_loo"] = ",".join(X[X.loo_total > 0].method)
    d["set_rt"] = ",".join(X[X.rt > 0].method)
    # set-rate -> EtCO2 (R01, R02) sign conclusion
    ok_design = sum((X[X.method == m].call_R01.iloc[0] == "correct") and (X[X.method == m].call_R02.iloc[0] == "correct")
                    for m in DESIGN)
    xc = ("xcorr_pos", "xcorr_v1", "xcorr_dir", "granger")
    wrong_lag = sum((X[X.method == m].call_R01.iloc[0] == "wrong") and (X[X.method == m].call_R02.iloc[0] == "wrong")
                    for m in xc)
    d["setrate_design_correct_both"] = int(ok_design)  # of 3
    d["setrate_xcorr_granger_wrong_both"] = int(wrong_lag)  # of 4
    return d


def part1(smoke: bool = False) -> dict:
    t0 = time.time()
    C = pd.read_csv(CALIB_CSV)
    log(f"Part 1: {len(C)} rows, {C.method.nunique()} methods, inputs {sorted(C.input.unique())}")
    res: dict = {"n_rows": int(len(C))}

    # ---- validation at the defaults: recomputed admitted / call / p_by equal the stored columns
    A = p1_admit(C, **P1_DEFAULT)
    res["val_admitted_match"] = int((A.adm == A.admitted.astype(bool)).sum())
    res["val_call_match"] = int((A.call2 == A.call).sum())
    res["val_p_by_max_abs_diff"] = float(np.nanmax(np.abs(A.p_adj - A.p_by)))
    fam = A[A.input == "real"].groupby("method").p.apply(lambda s: int(np.isfinite(s).sum()))
    res["val_family_size_real"] = fam.to_dict()
    # BY over the 29 rows (not over distinct estimates): check the rank-1 factor m*c_m on each method's family
    g = A[(A.input == "real") & (A.method == "matched") & (A.p > 0)].sort_values("p")
    r1 = g.iloc[0]
    res["val_by_rank1_factor_matched"] = float(r1.p_by / r1.p)
    res["val_by_rank1_factor_expected_m29"] = float(29 * np.sum(1.0 / np.arange(1, 30)))
    dup = A[(A.input == "real") & (A.method == "xcorr_v1")]
    res["val_xcorr_v1_distinct_estimates_real"] = int(dup.est.round(12).nunique())
    # bh_adjust <= q  <=> controlled.benjamini_hochberg
    rng = np.random.default_rng(0)
    pp = rng.uniform(size=(200, 29)) ** 3
    res["val_bh_adjust_equals_bh_stepup"] = bool(all((bh_adjust(x) <= 0.05).tolist() == tt.bh(x, 0.05).tolist() for x in pp))
    S0 = p1_summary(A)
    # Table 1 reproduction (domain columns + RT), compared with domain_summary.csv
    D = pd.read_csv(DOMAIN_CSV)
    t1 = []
    ok_all = True
    for m in ORDER:
        s = S0[S0.method == m].iloc[0]
        row = dict(estimator=NAMES[m], method=m)
        for dom, lab in (("gas_exchange", "Gas (5)"), ("drug", "Drug (11)"), ("drug_derived", "Vol. (2)"), ("policy", "Policy (10)")):
            row[lab] = f"{s[f'{dom}_E']}/{s[f'{dom}_W']}"
            dd = D[(D.method == m) & (D.domain == dom)]
            if len(dd):
                ok_all &= (int(dd.correct.iloc[0]) == s[f"{dom}_E"]) and (int(dd.wrong.iloc[0]) == s[f"{dom}_W"])
        row["RT"] = int(s.rt)
        row["LOO (8 draws)"] = int(s.loo_total)
        t1.append(row)
    res["val_table1_domain_match"] = bool(ok_all)
    T1 = pd.DataFrame(t1)
    T1.to_csv(OUT / "part1_table1_reproduced.csv", index=False)
    log("Part 1 validation:", {k: v for k, v in res.items() if k.startswith("val") and not isinstance(v, dict)})

    # ---- one-factor-at-a-time sweep
    oat_rows, oat_groups = [], []
    for factor, values in P1_RANGES.items():
        for v in values:
            cfg = dict(P1_DEFAULT)
            cfg[factor] = v
            S = p1_summary(p1_admit(C, **cfg))
            S.insert(0, "value", str(v))
            S.insert(0, "factor", factor)
            oat_rows.append(S)
            g = p1_groups(S)
            oat_groups.append(dict(factor=factor, value=str(v), is_default=(v == P1_DEFAULT[factor]), **g))
    OAT = pd.concat(oat_rows, ignore_index=True)
    OAT.to_csv(OUT / "part1_oat_by_estimator.csv", index=False)
    OATG = pd.DataFrame(oat_groups)
    OATG.to_csv(OUT / "part1_oat_groups.csv", index=False)

    # ---- full grid
    grid_rows, grid_est = [], []
    combos = list(itertools.product(*P1_RANGES.values()))
    if smoke:
        combos = combos[:6]
    for combo in combos:
        cfg = dict(zip(P1_RANGES.keys(), combo))
        S = p1_summary(p1_admit(C, **cfg))
        g = p1_groups(S)
        grid_rows.append({**{k: (str(v) if isinstance(v, bool) else v) for k, v in cfg.items()}, **g})
        for k, v in cfg.items():
            S[k] = str(v) if isinstance(v, bool) else v
        grid_est.append(S)
    GRID = pd.DataFrame(grid_rows)
    GRID.to_csv(OUT / "part1_grid_groups.csv", index=False)
    pd.concat(grid_est, ignore_index=True).to_csv(OUT / "part1_grid_by_estimator.csv", index=False)
    log(f"Part 1 grid: {len(GRID)} settings")

    # ---- what changes relative to the defaults
    d0 = GRID[(GRID.q == 0.05) & (GRID.adjust == "BY") & (GRID.min_stable == 0.75) & (GRID.min_cases == 5) & (GRID.ci == "True")]
    if len(d0):
        d0 = d0.iloc[0]
        chg = {}
        for col in ("set_wrong", "set_loo", "set_rt"):
            diff = GRID[GRID[col] != d0[col]]
            chg[col] = {"default": d0[col], "n_settings_differ": int(len(diff)),
                        "distinct_sets": sorted(diff[col].unique().tolist())}
        res["grid_changes_vs_default"] = chg
        res["grid_setrate"] = {"design_correct_both_min": int(GRID.setrate_design_correct_both.min()),
                               "xcorr_granger_wrong_both_min": int(GRID.setrate_xcorr_granger_wrong_both.min()),
                               "n_settings": int(len(GRID))}
        res["grid_design_loo_rt_max"] = {"design_loo_total_max": int(GRID.design_loo_total.max()),
                                         "design_rt_max": int(GRID.design_rt.max()),
                                         "design_wrong_min": int(GRID.design_wrong.min()),
                                         "design_wrong_max": int(GRID.design_wrong.max())}
    # per-estimator table at a few key settings
    key_settings = [("default", dict(P1_DEFAULT)), ("q=0.10", {**P1_DEFAULT, "q": 0.10}), ("q=0.01", {**P1_DEFAULT, "q": 0.01}),
                    ("BH", {**P1_DEFAULT, "adjust": "BH"}), ("stable>=0.5", {**P1_DEFAULT, "min_stable": 0.5}),
                    ("stable=1.0", {**P1_DEFAULT, "min_stable": 1.0}), ("cases>=3", {**P1_DEFAULT, "min_cases": 3}),
                    ("cases>=50", {**P1_DEFAULT, "min_cases": 50}), ("CI off", {**P1_DEFAULT, "ci": False}),
                    ("loosest: q=0.10,BH,stable>=0.5,cases>=3,CI off", dict(q=0.10, adjust="BH", min_stable=0.5, min_cases=3, ci=False)),
                    ("strictest: q=0.01,BY,stable=1.0,cases>=50,CI on", dict(q=0.01, adjust="BY", min_stable=1.0, min_cases=50, ci=True))]
    KT = []
    for lab, cfg in key_settings:
        S = p1_summary(p1_admit(C, **cfg))
        S.insert(0, "setting", lab)
        KT.append(S)
    KT = pd.concat(KT, ignore_index=True)
    KT.to_csv(OUT / "part1_key_settings_by_estimator.csv", index=False)

    # ---- figure
    p1_figure(OATG)
    # ---- markdown tables
    p1_markdown(T1, OATG, KT, GRID, res)
    res["seconds"] = time.time() - t0
    (OUT / "part1_results.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"Part 1 done in {res['seconds']:.0f}s")
    return res


def p1_figure(OATG: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = [("q", "FDR level q", "(BY)"), ("adjust", "adjustment", ""), ("min_stable", "min stable fraction", "(K = 8)"),
               ("min_cases", "min contributing cases", ""), ("ci", "CI-excludes-0 condition", "")]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharey="row")
    series_top = [("design_expected", BLUE, "-", "o", "design-based (3): expected sign"),
                  ("design_wrong", BLUE, "--", "s", "design-based (3): wrong sign"),
                  ("lag_expected", ORANGE, "-", "o", "lag-based (7): expected sign"),
                  ("lag_wrong", ORANGE, "--", "s", "lag-based (7): wrong sign")]
    series_bot = [("design_loo_total", BLUE, "-", "o", "design-based (3): held-out transplant (8 draws)"),
                  ("design_rt", BLUE, "--", "s", "design-based (3): random-time input"),
                  ("lag_loo_total", ORANGE, "-", "o", "lag-based (7): held-out transplant (8 draws)"),
                  ("lag_rt", ORANGE, "--", "s", "lag-based (7): random-time input")]
    for j, (factor, xlabel, note) in enumerate(factors):
        G = OATG[OATG.factor == factor]
        xs = list(range(len(G)))
        labels = [str(v) for v in G.value]
        if factor == "ci":
            labels = ["on" if v == "True" else "off" for v in G.value]
        for i, series in enumerate((series_top, series_bot)):
            ax = axes[i, j]
            for col, color, ls, mk, lab in series:
                ax.plot(xs, G[col].to_numpy(), color=color, ls=ls, marker=mk, ms=5, lw=2, label=lab)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            k0 = int(np.nonzero(G.is_default.to_numpy())[0][0])
            ax.axvline(k0, color=MUTED, lw=1, ls=":", zorder=0)
            ax.grid(axis="y", color="#e6e5e1", lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=9)
            if i == 1:
                ax.set_xlabel(f"{xlabel} {note}".strip(), color=INK, fontsize=10)
        axes[0, j].set_title(f"vary {factor}", color=INK, fontsize=10)
    axes[0, 0].set_ylabel("admissions on real data\n(sum over the group's estimators)", color=INK, fontsize=10)
    axes[1, 0].set_ylabel("false admissions under null input\n(sum over the group's estimators)", color=INK, fontsize=10)
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0, l0, loc="upper center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.legend(h1, l1, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("ClosedLoopBench (2 s, 3,442 cases): admissions vs. threshold, one factor at a time (dotted = paper default)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUT / "fig_part1_thresholds.png", dpi=160)
    fig.savefig(OUT / "fig_part1_thresholds.pdf")
    plt.close(fig)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def p1_markdown(T1, OATG, KT, GRID, res) -> None:
    L = ["## Part 1 tables (auto-generated)", "", "### Table 1 reproduced at the defaults (expected/wrong; RT = random-time false admissions; LOO = held-out-transplant false admissions over 8 draws)", "",
         _md(T1.drop(columns=["method"])), ""]
    G = OATG.copy()
    G["setting"] = G.factor + "=" + G.value + np.where(G.is_default, " (default)", "")
    cols = ["setting", "design_expected", "design_wrong", "design_loo_total", "design_rt", "lag_expected", "lag_wrong",
            "lag_loo_total", "lag_rt", "naive_expected", "naive_wrong", "naive_loo_total", "naive_rt",
            "setrate_design_correct_both", "setrate_xcorr_granger_wrong_both", "set_wrong", "set_loo", "set_rt"]
    L += ["### One factor at a time: group totals (design-based = matched+kink+seqtrial; lag-based = 7 lag estimators; naive = 2 naive contrasts)", "",
          _md(G[cols]), ""]
    K = KT[["setting", "method", "expected", "wrong", "loo_total", "rt", "call_R01", "call_R02"]].copy()
    K["method"] = K.method.map(NAMES)
    L += ["### Key settings by estimator", "", _md(K), ""]
    L += ["### Full grid summary", "", f"{len(GRID)} settings. ", json.dumps(res.get("grid_changes_vs_default"), indent=1),
          json.dumps(res.get("grid_setrate")), json.dumps(res.get("grid_design_loo_rt_max")), ""]
    (OUT / "part1_tables.md").write_text("\n".join(L))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Part 2: concept-level relations and graphs
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def load_s50():
    spec = importlib.util.spec_from_file_location("s50", ROOT / "scripts/50_build_kgs_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_edges() -> pd.DataFrame:
    recs = [json.loads(x) for x in (TE / "cache/fits.jsonl").read_text().splitlines() if x.strip()]
    rows = []
    for r in recs:
        for e in r["edges"]:
            rows.append(dict(family=r["family"], outcome=r["outcome"], variant=r["variant"], rep=int(r["rep"]),
                             source=e["source"], omega=e["omega"], omega_lo=e["omega_lo"], omega_hi=e["omega_hi"],
                             se_omega=e["se_omega"], n_exposed_events=e["n_exposed_events"],
                             n_event_patients=e["n_event_patients"], z_omega=e["z_omega"], p_omega=e["p_omega"],
                             testable=bool(e["testable"])))
    E = pd.DataFrame(rows)
    E["key"] = E.family + "|" + E.source + "->" + E.outcome
    return E


def stats_under(E: pd.DataFrame, min_ev: int, min_pat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """typed_temporal.edge_stats testability gate re-applied: z = omega/se if testable else 0, p = 2 Phi(-|z|) else 1."""
    se = E.se_omega.to_numpy(float)
    ok = (E.n_exposed_events.to_numpy(float) >= min_ev) & (E.n_event_patients.to_numpy(float) >= min_pat) & (se > 0)
    z = np.where(ok, E.omega.to_numpy(float) / np.where(se > 0, se, 1.0), 0.0)
    p = np.where(ok, 2 * stats.norm.sf(np.abs(z)), 1.0)
    return ok, z, p


class NullIndex:
    """z^2 of every null replicate per relation key: arrays (19,) for nullT and nullS."""

    def __init__(self, E: pd.DataFrame, z: np.ndarray):
        self.T: dict[str, np.ndarray] = {}
        self.S: dict[str, np.ndarray] = {}
        z2 = z**2
        for kind, store in (("nullT", self.T), ("nullS", self.S)):
            M = E.variant.to_numpy() == kind
            for key, grp in E[M].groupby("key").groups.items():
                a = np.zeros(N_NULL)
                sub = E.loc[grp]
                a[sub.rep.to_numpy()] = z2[grp]
                store[key] = a

    def others(self, key: str, kind: str | None, rep: int | None) -> tuple[np.ndarray, np.ndarray]:
        oT = self.T.get(key, np.zeros(0)).copy()
        oS = self.S.get(key, np.zeros(0)).copy()
        if kind == "nullT" and len(oT):
            oT = np.delete(oT, rep)
        if kind == "nullS" and len(oS):
            oS = np.delete(oS, rep)
        return oT, oS

    def passes(self, key: str, z2: float, rule: str, kind: str | None = None, rep: int | None = None) -> bool:
        if rule == "bh":
            return True
        oT, oS = self.others(key, kind, rep)
        if rule == "max":  # script 47 / 50: beat the maximum of each kind (max(o + [0]))
            return z2 > max(oT.max() if len(oT) else 0.0, 0.0) and z2 > max(oS.max() if len(oS) else 0.0, 0.0)
        if rule == "p95":  # 95th percentile (linear interpolation) of the union of the other replicates
            o = np.concatenate([oT, oS])
            return z2 > (float(np.quantile(o, 0.95)) if len(o) else 0.0)
        raise ValueError(rule)


def admit_variant(E: pd.DataFrame, p: np.ndarray, z: np.ndarray, NI: NullIndex, q: float, rule: str,
                  variant: str, rep: int) -> tuple[list[dict], dict]:
    """BH at q within family on one variant/rep (family = all candidate edges, untestable p = 1), then the null
    rule (leave-one-out when the variant is itself a null replicate). Returns admitted edges and per-family
    counts of rule (i) and admitted."""
    out, cnt = [], {}
    kind = variant if variant in ("nullT", "nullS") else None
    for fam in FAMILIES:
        M = (E.family.to_numpy() == fam) & (E.variant.to_numpy() == variant) & (E.rep.to_numpy() == rep)
        idx = np.nonzero(M)[0]
        if not len(idx):
            continue
        rej = tt.bh(p[idx], q)
        n_adm = 0
        for i, k in zip(idx, rej):
            if not k:
                continue
            key = E.key.iat[i]
            if NI.passes(key, float(z[i] ** 2), rule, kind, rep):
                out.append(dict(family=fam, source=E.source.iat[i], outcome=E.outcome.iat[i], key=key,
                                omega=float(E.omega.iat[i]), omega_lo=float(E.omega_lo.iat[i]), omega_hi=float(E.omega_hi.iat[i])))
                n_adm += 1
        cnt[fam] = dict(n_candidate=int(len(idx)), n_testable=int((p[idx] < 1).sum()), n_bh=int(rej.sum()), n_admitted=n_adm)
    return out, cnt


def instantiate_keys(co, s50, adm: list[dict], kind: str, rep: int, spacing, calib) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per admitted null relation: per-case instance indicator and node-level edge count under null replicate
    (kind, rep). Mirrors scripts/50_build_kgs_v3.null_counts, which sums these over the admitted set."""
    from clbench import kg_v2 as kv

    n = co.n_cases
    rels = [s50.null_rel(a, "E") for a in adm]
    keys = [a["key"] for a in adm]
    rel_vec = np.zeros((len(rels), n), bool)
    node_vec = np.zeros((len(rels), n))
    if kind == "nullT":
        m = tt.transplant_map(co, np.random.default_rng(10_000 + rep))
    else:
        m = tt.shift_map(co, np.random.default_rng(20_000 + rep), spacing)
    Sn = tt.apply_map_S(co.S, m)
    chgn = tt.apply_map_chg(co.chg, m)
    for c in range(n):
        s0, T = int(co.case_start[c]), int(co.T[c])
        S, Snc = co.S[s0:s0 + T], Sn[s0:s0 + T]
        chg, chgnc = co.chg[s0:s0 + T], chgn[s0:s0 + T]
        tiers_null = None
        tiers_real = None
        for j, r in enumerate(rels):
            if r.edge_type == "association":
                a, b = tt.CONCEPTS.index(r.source), tt.CONCEPTS.index(r.target)
                p = kv.association_pairs(Snc[:, a], S[:, b])
                if len(p):
                    tiers_null = tiers_null or kv.case_tiers(Snc, calib)
                    tiers_real = tiers_real or kv.case_tiers(S, calib)
                    ta, tb = tiers_null[r.source], tiers_real[r.target]
                    node_vec[j, c] += len({(ta[s], tb[t]) for s, t in zip(p.src, p.tgt)})
            elif r.edge_type == "policy":
                ai, di = tt.ACTIONS.index(r.action), tt.DIRS.index(r.direction)
                p = kv.policy_pairs(Snc[:, tt.CONCEPTS.index(r.source)], np.nonzero(chg[:, ai, di])[0])
                if len(p):
                    tiers_null = tiers_null or kv.case_tiers(Snc, calib)
                    node_vec[j, c] += len({tiers_null[r.source][s] for s in p.src})
            else:
                ai, di = tt.ACTIONS.index(r.action), tt.DIRS.index(r.direction)
                p = kv.response_pairs(np.nonzero(chgnc[:, ai, di])[0], S[:, tt.CONCEPTS.index(r.target)])
                if len(p):
                    tiers_real = tiers_real or kv.case_tiers(S, calib)
                    node_vec[j, c] += len({tiers_real[r.target][t] for t in p.tgt})
            rel_vec[j, c] = len(p) > 0
    return {k: (rel_vec[j], node_vec[j]) for j, k in enumerate(keys)}


def boot_reps(rep_means: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over the 19 replicate means (the paper's Table 2 CI)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(rep_means, float)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_two_stage(M: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Two-stage bootstrap: replicates with replacement, then patients with replacement (App. graphs-fe)."""
    rng = np.random.default_rng(seed)
    R, P = M.shape
    out = np.empty(n)
    for b in range(n):
        ri = rng.integers(0, R, R)
        pi = rng.integers(0, P, P)
        out[b] = M[ri][:, pi].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def part2(smoke: bool = False) -> dict:
    from clbench.persistence import CalibrationConfig

    t0 = time.time()
    s50 = load_s50()
    s47 = s50.s47()
    E = load_edges()
    log(f"Part 2: {len(E)} edge records ({E.key.nunique()} relation keys); variants {sorted(E.variant.unique())}")
    res: dict = {"n_edge_records": int(len(E)), "n_keys": int(E.key.nunique())}

    # ---- validation of the cached statistics
    ok, z, p = stats_under(E, tt.MIN_EXPOSED_EVENTS, tt.MIN_EVENT_PATIENTS)
    res["val_testable_match"] = bool((ok == E.testable.to_numpy()).all())
    res["val_z_max_abs_diff"] = float(np.max(np.abs(z - E.z_omega.to_numpy())))
    res["val_p_max_abs_diff"] = float(np.max(np.abs(p - E.p_omega.to_numpy())))
    # loosening is possible from the cache iff se_omega is finite and > 0 for the currently untestable edges
    unt = E[~E.testable]
    res["val_untestable_edges"] = int(len(unt))
    res["val_untestable_with_se"] = int((np.isfinite(unt.se_omega) & (unt.se_omega > 0)).sum())
    res["val_untestable_events_ge10_pat_ge5"] = int(((unt.n_exposed_events >= 10) & (unt.n_event_patients >= 5)).sum())
    NI = NullIndex(E, z)
    adm_real, cnt_real = admit_variant(E, p, z, NI, 0.05, "max", "full", 0)
    res["val_real_counts_default"] = cnt_real  # paper: rule (i) 31/36/25, admitted 28/33/20
    null_csv = pd.read_csv(NULL_CSV)
    null_adm = {}
    match = True
    for kind in ("nullT", "nullS"):
        for rep in range(N_NULL):
            a, _ = admit_variant(E, p, z, NI, 0.05, "max", kind, rep)
            null_adm[(kind, rep)] = a
            ref = null_csv[(null_csv["null"] == kind) & (null_csv.rep == rep)].iloc[0]
            mine = ";".join(f"{x['family']}:{x['source']}->{x['outcome']}" for x in a)
            match &= (int(ref.n_admitted) == len(a)) and (mine == (ref.admitted if isinstance(ref.admitted, str) else ""))
    res["val_null_admissions_match_csv"] = bool(match)
    log("Part 2 admission validation:", {k: v for k, v in res.items() if k.startswith("val")})

    # ---- cohort for instantiation
    co = s47.load_cohort(str(TE / "cache/cohort.npz"))
    calib = CalibrationConfig.load(CALIB_JSON)
    log(f"cohort: {co.n_cases} cases, {co.n} rows; computing shift spacing")
    spacing = s47.case_spacing(co)
    # validation of the per-relation instantiation against script 50 null_counts (default admissions)
    t1 = time.time()
    cache: dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    fe_default = {}
    for (kind, rep), a in null_adm.items():
        if not a:
            continue
        cache[(kind, rep)] = instantiate_keys(co, s50, a, kind, rep, spacing, calib)
        rc, nc = s50.null_counts(co, a, kind, rep, spacing, calib, "E")
        mine_r = np.sum([cache[(kind, rep)][x["key"]][0] for x in a], axis=0).astype(float)
        mine_n = np.sum([cache[(kind, rep)][x["key"]][1] for x in a], axis=0)
        fe_default[(kind, rep)] = (float(rc.mean()), float(np.abs(mine_r - rc).max()), float(np.abs(mine_n - nc).max()))
    res["val_instantiation_max_abs_diff"] = float(max(max(v[1], v[2]) for v in fe_default.values()))
    ref_rows = null_csv.set_index(["null", "rep"]).false_rel_per_graph
    res["val_fe_per_rep_max_abs_diff_vs_csv"] = float(max(abs(v[0] - ref_rows[(k, r)]) for (k, r), v in fe_default.items()))
    log(f"instantiation validation ({time.time() - t1:.0f}s): max diff vs null_counts {res['val_instantiation_max_abs_diff']:.2e}, "
        f"vs csv {res['val_fe_per_rep_max_abs_diff_vs_csv']:.2e}")

    # ---- the sweep: admissions for every setting, then instantiate the union of admitted null relations
    settings = []
    for q, rule, (min_ev, min_pat) in itertools.product(P2_RANGES["q"], P2_RANGES["rule"],
                                                        [(10, 5), (20, 10), (40, 20), (10, 10), (40, 10), (20, 5), (20, 20)]):
        settings.append(dict(q=q, rule=rule, min_ev=min_ev, min_pat=min_pat))
    if smoke:
        settings = [dict(P2_DEFAULT), dict(P2_DEFAULT, q=0.10, rule="bh")]
    stat_cache = {}
    per_setting = []
    null_rows = []
    real_rows = []
    for st in settings:
        tk = (st["min_ev"], st["min_pat"])
        if tk not in stat_cache:
            _, zz, pp = stats_under(E, *tk)
            stat_cache[tk] = (zz, pp, NullIndex(E, zz))
        zz, pp, NIk = stat_cache[tk]
        a_real, c_real = admit_variant(E, pp, zz, NIk, st["q"], st["rule"], "full", 0)
        nulls = {}
        for kind in ("nullT", "nullS"):
            for rep in range(N_NULL):
                nulls[(kind, rep)], _ = admit_variant(E, pp, zz, NIk, st["q"], st["rule"], kind, rep)
                null_rows.append({**st, "null": kind, "rep": rep, "n_admitted": len(nulls[(kind, rep)]),
                                  "admitted": ";".join(x["key"] for x in nulls[(kind, rep)])})
        per_setting.append((st, a_real, c_real, nulls))
        real_rows.append({**st, **{f"{f}_{k}": c_real.get(f, {}).get(k, 0) for f in FAMILIES for k in ("n_testable", "n_bh", "n_admitted")},
                          "admitted_pos": sum(x["omega"] > 0 for x in a_real), "admitted_neg": sum(x["omega"] < 0 for x in a_real),
                          "admitted": ";".join(f"{x['key']}{'+' if x['omega'] > 0 else '-'}" for x in a_real)})
    log(f"Part 2: admissions computed for {len(settings)} settings")
    # union of null admissions needing instantiation
    need: dict[tuple[str, int], dict[str, dict]] = {}
    for st, _, _, nulls in per_setting:
        for kr, a in nulls.items():
            for x in a:
                if kr not in cache or x["key"] not in cache[kr]:
                    need.setdefault(kr, {})[x["key"]] = x
    n_need = sum(len(v) for v in need.values())
    log(f"instantiating {n_need} additional (replicate, relation) pairs over {len(need)} replicates")
    t1 = time.time()
    for i, (kr, d) in enumerate(sorted(need.items())):
        cache.setdefault(kr, {}).update(instantiate_keys(co, s50, list(d.values()), kr[0], kr[1], spacing, calib))
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(need)} replicates ({time.time() - t1:.0f}s)")
    log(f"instantiation done in {time.time() - t1:.0f}s")

    # ---- assemble FE per setting
    rows = []
    for st, a_real, c_real, nulls in per_setting:
        d = dict(st)
        for fam in FAMILIES:
            for k in ("n_testable", "n_bh", "n_admitted"):
                d[f"{fam}_{k}"] = c_real.get(fam, {}).get(k, 0)
        d["real_admitted_total"] = sum(c_real.get(f, {}).get("n_admitted", 0) for f in FAMILIES)
        for kind, lab in (("nullT", "T"), ("nullS", "S")):
            M = np.zeros((N_NULL, co.n_cases))
            Mn = np.zeros((N_NULL, co.n_cases))
            n_adm = np.zeros(N_NULL, int)
            for rep in range(N_NULL):
                a = nulls[(kind, rep)]
                n_adm[rep] = len(a)
                if a:
                    M[rep] = np.sum([cache[(kind, rep)][x["key"]][0] for x in a], axis=0)
                    Mn[rep] = np.sum([cache[(kind, rep)][x["key"]][1] for x in a], axis=0)
            rep_means = M.mean(1)
            d[f"FE_{lab}"] = float(M.mean())
            d[f"FE_{lab}_lo_reps"], d[f"FE_{lab}_hi_reps"] = boot_reps(rep_means, seed=0)
            d[f"FE_{lab}_lo_2stage"], d[f"FE_{lab}_hi_2stage"] = boot_two_stage(M, seed=0)
            pm = s50.boot_mean(M.mean(0), seed=1)
            d[f"FE_{lab}_lo_patients"], d[f"FE_{lab}_hi_patients"] = pm[1], pm[2]
            d[f"FEnode_{lab}"] = float(Mn.mean())
            d[f"reps_any_{lab}"] = int((n_adm > 0).sum())
            d[f"share_reps_any_{lab}"] = float((n_adm > 0).mean())
            d[f"mean_null_adm_{lab}"] = float(n_adm.mean())
            d[f"max_null_adm_{lab}"] = int(n_adm.max())
            d[f"frac_graphs_any_{lab}"] = float((M > 0).mean())
        rows.append(d)
    SW = pd.DataFrame(rows)
    SW.to_csv(OUT / "part2_sweep.csv", index=False)
    pd.DataFrame(null_rows).to_csv(OUT / "part2_null_admissions.csv", index=False)
    pd.DataFrame(real_rows).to_csv(OUT / "part2_real_admissions.csv", index=False)
    d0 = SW[(SW.q == 0.05) & (SW.rule == "max") & (SW.min_ev == 20) & (SW.min_pat == 10)]
    if len(d0):
        d0 = d0.iloc[0]
        res["val_FE_default"] = {k: float(d0[k]) for k in ("FE_T", "FE_S", "FE_T_lo_reps", "FE_T_hi_reps", "FE_S_lo_reps", "FE_S_hi_reps",
                                                           "FE_T_lo_2stage", "FE_T_hi_2stage", "FE_S_lo_2stage", "FE_S_hi_2stage")}
        res["val_reps_any_default"] = {"T": int(d0.reps_any_T), "S": int(d0.reps_any_S)}
        # seed dependence of the replicate-mean CI (19 values)
        MT = [v[0] for (k, r), v in sorted(fe_default.items()) if k == "nullT"]
        rmT = np.zeros(N_NULL)
        rmS = np.zeros(N_NULL)
        for (k, r), v in fe_default.items():
            (rmT if k == "nullT" else rmS)[r] = v[0]
        res["val_FE_reps_ci_seed_range"] = {
            "T": [[*boot_reps(rmT, seed=s)] for s in range(10)], "S": [[*boot_reps(rmS, seed=s)] for s in range(10)]}
        res["paper"] = {"FE_T": 0.057, "FE_T_ci": [0.014, 0.107], "FE_S": 0.103, "FE_S_ci": [0.035, 0.184],
                        "FE_T_ci_2stage": [0.015, 0.106], "FE_S_ci_2stage": [0.032, 0.190], "reps_any_T": 7, "reps_any_S": 9,
                        "results_json": json.load(open(NULL_JSON))["nulls_vitaldb"]["summary"]}
    p2_figure(SW)
    p2_markdown(SW, res)
    res["seconds"] = time.time() - t0
    (OUT / "part2_results.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"Part 2 done in {res['seconds']:.0f}s")
    return res


def p2_figure(SW: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = SW[(SW.min_ev == 20) & (SW.min_pat == 10)]
    if len(D) < len(P2_RANGES["q"]) * len(P2_RANGES["rule"]):
        log("p2_figure skipped (incomplete sweep)")
        return
    rules = [("max", "-", "o", "exceed max of the other null replicates (paper)"),
             ("p95", "--", "s", "exceed 95th pct of the other null replicates"), ("bh", ":", "^", "BH only")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    qs = list(P2_RANGES["q"])
    xs = list(range(len(qs)))
    # (a) FE per graph, transplant and shift
    ax = axes[0]
    for rule, ls, mk, lab in rules:
        G = D[D.rule == rule].set_index("q").loc[qs]
        ax.plot(xs, G.FE_T, color=BLUE, ls=ls, marker=mk, lw=2, ms=5, label=f"transplant, {lab}")
        ax.plot(xs, G.FE_S, color=ORANGE, ls=ls, marker=mk, lw=2, ms=5, label=f"shift, {lab}")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(0.02, 0.51, "registered prediction < 0.5", color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_ylabel("false relation instances per graph", color=INK)
    ax.set_title("(a) FE per graph vs q (testability 20 / 10)", color=INK, fontsize=10)
    # (b) real admissions
    ax = axes[1]
    for rule, ls, mk, lab in rules:
        G = D[D.rule == rule].set_index("q").loc[qs]
        ax.plot(xs, G.real_admitted_total, color=BLUE, ls=ls, marker=mk, lw=2, ms=5, label=lab)
    ax.set_ylabel("admitted relations on real data (of 176)", color=INK)
    ax.set_title("(b) real-data admissions vs q", color=INK, fontsize=10)
    # (c) share of replicates with >= 1 false population admission
    ax = axes[2]
    for rule, ls, mk, lab in rules:
        G = D[D.rule == rule].set_index("q").loc[qs]
        ax.plot(xs, G.share_reps_any_T, color=BLUE, ls=ls, marker=mk, lw=2, ms=5)
        ax.plot(xs, G.share_reps_any_S, color=ORANGE, ls=ls, marker=mk, lw=2, ms=5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("share of null replicates with >= 1 admission", color=INK)
    ax.set_title("(c) replicates with a false population admission", color=INK, fontsize=10)
    for ax in axes[:3]:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(q) for q in qs])
        ax.set_xlabel("FDR level q (BH within family)", color=INK)
    # (d) testability thresholds at q = 0.05, max rule
    ax = axes[3]
    T = SW[(SW.q == 0.05) & (SW.rule == "max")]
    labs = [(10, 5), (20, 10), (40, 20), (10, 10), (40, 10), (20, 5), (20, 20)]
    T = T.set_index(["min_ev", "min_pat"]).loc[labs]
    xt = list(range(len(labs)))
    ax.plot(xt, T.FE_T, color=BLUE, marker="o", lw=2, ms=5, label="transplant")
    ax.plot(xt, T.FE_S, color=ORANGE, marker="o", lw=2, ms=5, label="shift")
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{a}/{b}" for a, b in labs], fontsize=8)
    ax.set_xlabel("min events / min patients (q = 0.05, max rule)", color=INK)
    ax.set_ylabel("false relation instances per graph", color=INK)
    ax.set_title("(d) FE per graph vs testability thresholds", color=INK, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h[::2] + h[1::2], ["transplant: " + r[3] for r in rules] + ["shift: " + r[3] for r in rules],
               loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Concept-level relations (VitalDB, 176 candidates, 38 null replicates): admission thresholds", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(OUT / "fig_part2_thresholds.png", dpi=160)
    fig.savefig(OUT / "fig_part2_thresholds.pdf")
    plt.close(fig)


def p2_markdown(SW: pd.DataFrame, res: dict) -> None:
    def ci(r, lab):
        return f"{r[f'FE_{lab}']:.3f} [{r[f'FE_{lab}_lo_reps']:.3f}, {r[f'FE_{lab}_hi_reps']:.3f}]"

    T = pd.DataFrame({
        "q": SW.q, "null rule": SW.rule, "events/patients": SW.min_ev.astype(str) + "/" + SW.min_pat.astype(str),
        "testable A/P/R": SW.association_n_testable.astype(str) + "/" + SW.policy_n_testable.astype(str) + "/" + SW.response_n_testable.astype(str),
        "rule (i) A/P/R": SW.association_n_bh.astype(str) + "/" + SW.policy_n_bh.astype(str) + "/" + SW.response_n_bh.astype(str),
        "admitted A/P/R": SW.association_n_admitted.astype(str) + "/" + SW.policy_n_admitted.astype(str) + "/" + SW.response_n_admitted.astype(str),
        "FE transplant [95% CI]": [ci(r, "T") for _, r in SW.iterrows()],
        "FE shift [95% CI]": [ci(r, "S") for _, r in SW.iterrows()],
        "reps with >=1 (T/S, of 19)": SW.reps_any_T.astype(str) + "/" + SW.reps_any_S.astype(str),
        "mean null adm. per rep (T/S)": SW.mean_null_adm_T.map(lambda x: f"{x:.2f}") + "/" + SW.mean_null_adm_S.map(lambda x: f"{x:.2f}"),
    })
    L = ["## Part 2 tables (auto-generated)", "", "CI: percentile bootstrap over the 19 replicate means (1000 resamples, seed 0), as in Table 2.", "",
         "### Sweep at the default testability (20 events / 10 patients)", "",
         _md(T[(SW.min_ev == 20) & (SW.min_pat == 10)].reset_index(drop=True)), "",
         "### Testability sweep at q = 0.05, max rule", "",
         _md(T[(SW.q == 0.05) & (SW.rule == "max")].reset_index(drop=True)), "",
         "### Full sweep", "", _md(T), "",
         "### Validation", "", "```", json.dumps({k: v for k, v in res.items() if k.startswith("val") or k == "paper"}, indent=1, default=str), "```", ""]
    (OUT / "part2_tables.md").write_text("\n".join(L))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=("1", "2", "all"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="re-render the figures from the saved CSVs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.figures_only:
        p1_figure(pd.read_csv(OUT / "part1_oat_groups.csv"))
        p2_figure(pd.read_csv(OUT / "part2_sweep.csv"))
        log("figures re-rendered")
        return
    if args.part in ("1", "all"):
        part1(args.smoke)
    if args.part in ("2", "all"):
        part2(args.smoke)
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
