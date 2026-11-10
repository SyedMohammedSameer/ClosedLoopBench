"""
Summary of extended analysis C (``scripts/extended/C_sim_grid.py``): per cell x estimator x relation metrics
against the ITT truth, per-cell power / wrong-sign / false-positive rates, markdown tables and figures.

Metrics (calibrated admission rule of ClosedLoopBench 1.0, every estimator, primary horizons):
  TP          true response relation admitted with the sign of the ITT truth (active cohort)
  wrong       true response relation admitted with the opposite sign
  FP_inert    response relation admitted in the inert cohort (same seed, drug effects 0; truth exactly 0)
  FP_null     admissions with a held-out transplant draw as input (leave-one-out over the K = 8 draws), sim relations
  FP_random   admissions with the random-time stream as input
  null share  uncalibrated discoveries (single-arm CI excluding 0) per transplant draw / relations tested
  bias, RMSE  calibrated estimate minus ITT truth (event-contrast estimators in the vital's units per titration
              step; VARX converted from per-unit-dose to per-step by the step size; other estimators: signs only)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clbench.eval import closed_loop_bench as clb

METHODS: tuple[str, ...] = ("naive", "naive_lvl", "xcorr_pos", "xcorr_v1", "xcorr_dir", "granger", "varx", "pcmciplus",
                            "varlingam", "matched", "kink", "seqtrial")
TYPED = "typed_pooled"
ALL_METHODS: tuple[str, ...] = METHODS + (TYPED,)
LABELS: dict[str, str] = {
    "naive": "naive event", "naive_lvl": "naive level", "xcorr_pos": "xcorr positive-peak", "xcorr_v1": "xcorr signed-peak",
    "xcorr_dir": "xcorr direction-aware", "granger": "Granger", "varx": "VARX", "pcmciplus": "PCMCI+",
    "varlingam": "VAR-LiNGAM", "matched": "matched event design", "kink": "interrupted trend", "seqtrial": "sequential trial",
    TYPED: "typed composite (VARX base)"}
GROUPS: dict[str, tuple[str, ...]] = {
    "design-based": ("matched", "kink", "seqtrial"),
    "lag-based": ("granger", "varx", "pcmciplus", "varlingam"),
    "naive / correlation": ("naive", "naive_lvl", "xcorr_pos", "xcorr_v1", "xcorr_dir"),
}
EVENT_UNIT: frozenset[str] = frozenset({"naive", "naive_lvl", "matched", "kink", "seqtrial"})
STEP: dict[str, float] = {"vent_rr": 2.0, "remifentanil": 0.5}  # titration step of the simulator (VARX unit conversion)
RESP: tuple[str, ...] = ("R01", "R02", "R05", "R06", "R07", "R08")
POL: tuple[str, ...] = ("P01", "P02", "P05")
SHARED: dict[str, tuple[str, ...]] = {  # direction-agnostic estimators: one estimate for the increase/decrease pair
    "granger": ("R01/R02", "R05/R07", "R06/R08"), "varlingam": ("R01/R02", "R05/R07", "R06/R08"),
    "pcmciplus": ("R01/R02", "R05/R07", "R06/R08"), "xcorr_pos": ("R01/R02/P05", "R05/R07/P02", "R06/R08/P01"),
    "xcorr_v1": ("R01/R02/P05", "R05/R07/P02", "R06/R08/P01"), "xcorr_dir": ("R01/R02", "R05/R07", "R06/R08")}

# palette (dataviz reference instance): categorical slots 1-3, blue sequential ramp, ink and chrome
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
            "#184f95", "#104281", "#0d366b"]
SEQ_ORANGE = ["#fdeee6", "#f9cdb8", "#f5ac8a", "#f08b5f", "#eb6834", "#c9532a", "#a63f1f", "#852c14"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _sgn(x) -> int:
    return int(np.sign(x)) if np.isfinite(x) else 0


def load_cells(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(long table: one row per cell x method x relation at the primary horizon, null table, metas)."""
    design = json.loads((out / "design.json").read_text()) if (out / "design.json").exists() else {}
    rows, nrows, metas = [], [], {}
    for mdir in sorted((out / "cells").glob("*/meta.json")):
        d = mdir.parent
        meta = json.loads(mdir.read_text())
        cell = meta["cell"]
        metas[cell] = meta
        st = meta["settings"]
        cfg = clb.SCALES[meta["scale"]]
        P = pd.read_csv(d / "pooled.csv")
        C = pd.read_csv(d / "calibrated.csv") if (d / "calibrated.csv").exists() else None
        N = pd.read_csv(d / "null_discovery.csv") if (d / "null_discovery.csv").exists() else None
        T = pd.read_csv(d / "truth.csv")
        base = dict(cell=cell, config=meta["config"], drug=meta["drug"], rep=meta["rep"], seed=meta["seed"], n=meta["n"],
                    scale=meta["scale"], gain=st["gain"], w_z=st["w_z"], emax_c50=st["emax_c50"], emax_gamma=st["emax_gamma"],
                    ant=st["ant"], exog_remi=st["exog_remi"],
                    remi_per_patient=meta["events"]["remi"]["per_patient"], vent_per_patient=meta["events"]["vent"]["per_patient"],
                    remi_exog_share=meta["events"]["remi"]["exog_share"], seconds=meta["seconds"]["total"])
        for m in ALL_METHODS:
            for rid in RESP + POL:
                r = clb.REL_BY_ID[rid]
                h = clb.primary_h(r, m, cfg)
                q = P[(P.arm == "real") & (P.method == m) & (P.rid == rid) & (P.h == h)]
                c = C[(C.input == "real") & (C.method == m) & (C.rid == rid) & (C.h == h)] if C is not None else None
                if (q is None or not len(q)) and (c is None or not len(c)):
                    continue
                row = dict(base, method=m, rid=rid, kind=r.kind, h=h, expected=r.expected)
                if q is not None and len(q):
                    q = q.iloc[0]
                    row.update(est=float(q.est), lo=float(q.lo), hi=float(q.hi), call=q.call, n_cases=float(q.n_cases))
                else:
                    row.update(est=np.nan, lo=np.nan, hi=np.nan, call="na", n_cases=np.nan)
                if c is not None and len(c):
                    c = c.iloc[0]
                    row.update(cal_est=float(c.est), cal_lo=float(c.lo), cal_hi=float(c.hi), admitted=bool(c.admitted),
                               cal_call=c.call, stable_frac=float(c.stable_frac), null_mean=float(c.est_null_mean),
                               n_cases=float(c.n_cases) if np.isfinite(c.n_cases) else row["n_cases"])
                else:
                    row.update(cal_est=np.nan, cal_lo=np.nan, cal_hi=np.nan, admitted=False, cal_call="na", stable_frac=np.nan,
                               null_mean=np.nan)
                if r.kind == "response":
                    # truth at the relation's primary horizon (estimators without a horizon, h = 0, are scored on sign only)
                    h_t = h if h > 0 else clb.primary_h(r, "naive", cfg)
                    t = T[(T.rid == rid) & (T.h == h_t)]
                    t = t.iloc[0] if len(t) else None
                    row.update(truth=float(t.truth) if t is not None else np.nan, truth_lo=float(t.truth_lo) if t is not None else np.nan,
                               truth_hi=float(t.truth_hi) if t is not None else np.nan,
                               truth_sign=int(t.truth_sign) if t is not None else 0, truth_n=int(t.truth_n) if t is not None else 0)
                else:  # policy: the recorded clinician acts in the expected direction; no ITT truth (sign only)
                    row.update(truth=np.nan, truth_lo=np.nan, truth_hi=np.nan, truth_sign=r.expected, truth_n=0)
                # unit conversion for the bias columns
                # bias is defined for the event-contrast estimators only: they share the ITT truth's functional (post-window mean
                # at t + h minus the pre-level). VARX's estimand (cumulative response to a sustained unit step) and the correlation /
                # coefficient estimators are scored on sign only. ``STEP`` converts VARX to per-step units for the long table.
                conv = 1.0 if m in EVENT_UNIT else np.nan
                row["conv"] = conv
                row["varx_per_step"] = row["est"] * STEP[r.action] if m in ("varx", TYPED) else np.nan
                row["est_step"] = row["est"] * conv
                row["cal_est_step"] = row["cal_est"] * conv
                rows.append(row)
        if C is not None:
            for m in ALL_METHODS:
                cm = C[(C.method == m) & C.rid.isin(RESP + POL)]
                loo = cm[cm.input.str.startswith("loo:")]
                rnd = cm[cm.input == "nc_random"]
                per = loo.groupby("input").admitted.sum() if len(loo) else pd.Series(dtype=float)
                nd = dict(base, method=m, loo_adm=int(loo.admitted.sum()), loo_tests=int(len(loo)),
                          loo_per_draw=float(per.mean()) if len(per) else np.nan, loo_draws_any=int((per > 0).sum()),
                          loo_adm_resp=int(loo[loo.rid.isin(RESP)].admitted.sum()), loo_tests_resp=int(loo.rid.isin(RESP).sum()),
                          random_adm=int(rnd.admitted.sum()), random_tests=int(len(rnd)))
                if N is not None and m != TYPED:
                    cr = N[(N.method == m) & N.arm.isin(clb.CROSS_ARMS)]
                    rn = N[(N.method == m) & (N.arm == "nc_random")]
                    re = N[(N.method == m) & (N.arm == "real")]
                    nd.update(uncal_null_sig=float(cr.n_sig.mean()) if len(cr) else np.nan,
                              uncal_null_rel=float(cr.n_relations.mean()) if len(cr) else np.nan,
                              uncal_null_distinct_sig=float(cr.n_distinct_sig.mean()) if len(cr) else np.nan,
                              uncal_null_distinct=float(cr.n_distinct.mean()) if len(cr) else np.nan,
                              uncal_random_sig=float(rn.n_sig.iloc[0]) if len(rn) else np.nan,
                              uncal_real_sig=float(re.n_sig.iloc[0]) if len(re) else np.nan)
                nrows.append(nd)
    L = pd.DataFrame(rows)
    if len(L):
        s = np.sign(L.cal_est)
        L["tp"] = L.admitted & (L.kind == "response") & (L.truth_sign != 0) & (s == L.truth_sign) & (L.drug == "active")
        L["wrong"] = L.admitted & (L.kind == "response") & (L.truth_sign != 0) & (s == -L.truth_sign) & (L.drug == "active")
        L["fp_inert"] = L.admitted & (L.kind == "response") & (L.drug == "inert")
        L["pol_correct"] = L.admitted & (L.kind == "policy") & (s == L.expected)
        L["pol_wrong"] = L.admitted & (L.kind == "policy") & (s == -L.expected)
        su = np.sign(L.est)
        sig = (L.lo > 0) | (L.hi < 0)
        L["uncal_tp"] = sig & (L.kind == "response") & (L.truth_sign != 0) & (su == L.truth_sign) & (L.drug == "active")
        L["uncal_wrong"] = sig & (L.kind == "response") & (L.truth_sign != 0) & (su == -L.truth_sign) & (L.drug == "active")
        L["uncal_fp_inert"] = sig & (L.kind == "response") & (L.drug == "inert")
        L["bias"] = L.cal_est_step - L.truth
        L["uncal_bias"] = L.est_step - L.truth
        L["cover"] = (L.cal_lo * L.conv <= L.truth) & (L.cal_hi * L.conv >= L.truth)
    return L, pd.DataFrame(nrows), {"metas": metas, "design": design}


def cell_summary(L: pd.DataFrame, Nt: pd.DataFrame) -> pd.DataFrame:
    """One row per cell x method."""
    rows = []
    for (cell, m), g in L.groupby(["cell", "method"]):
        r = g[g.kind == "response"]
        p = g[g.kind == "policy"]
        b = g.iloc[0]
        n_true = int((r.truth_sign != 0).sum()) if b.drug == "active" else 0
        d = dict(cell=cell, config=b.config, drug=b.drug, rep=b.rep, scale=b.scale, gain=b.gain, w_z=b.w_z, emax_c50=b.emax_c50,
                 emax_gamma=b.emax_gamma, ant=b.ant, exog_remi=b.exog_remi, method=m, n_resp=int(len(r)), n_true=n_true,
                 tp=int(r.tp.sum()), wrong=int(r.wrong.sum()), fp_inert=int(r.fp_inert.sum()),
                 adm_resp=int(r.admitted.sum()), pol_correct=int(p.pol_correct.sum()), pol_wrong=int(p.pol_wrong.sum()),
                 n_pol=int(len(p)), uncal_tp=int(r.uncal_tp.sum()), uncal_wrong=int(r.uncal_wrong.sum()),
                 uncal_fp_inert=int(r.uncal_fp_inert.sum()), remi_per_patient=b.remi_per_patient,
                 vent_per_patient=b.vent_per_patient, mean_n_cases=float(r.n_cases.mean()))
        d["power"] = d["tp"] / n_true if n_true else np.nan
        d["wrong_rate"] = d["wrong"] / n_true if n_true else np.nan
        d["fp_inert_rate"] = d["fp_inert"] / len(r) if (b.drug == "inert" and len(r)) else np.nan
        d["uncal_power"] = d["uncal_tp"] / n_true if n_true else np.nan
        d["uncal_fp_inert_rate"] = d["uncal_fp_inert"] / len(r) if (b.drug == "inert" and len(r)) else np.nan
        if m in EVENT_UNIT:
            ok = r[np.isfinite(r.bias) & (r.truth_sign != 0)] if b.drug == "active" else r[np.isfinite(r.bias)]
            d["bias"] = float(ok.bias.mean()) if len(ok) else np.nan
            d["abs_bias"] = float(ok.bias.abs().mean()) if len(ok) else np.nan
            d["rmse"] = float(np.sqrt((ok.bias**2).mean())) if len(ok) else np.nan
            d["rel_bias"] = float((ok.bias / ok.truth.abs()).mean()) if (len(ok) and b.drug == "active") else np.nan
            d["uncal_rmse"] = float(np.sqrt((ok.uncal_bias**2).mean())) if len(ok) else np.nan
            d["mean_truth"] = float(ok.truth.mean()) if len(ok) else np.nan
        else:
            d.update(bias=np.nan, abs_bias=np.nan, rmse=np.nan, rel_bias=np.nan, uncal_rmse=np.nan, mean_truth=np.nan)
        q = Nt[(Nt.cell == cell) & (Nt.method == m)] if len(Nt) else Nt
        if len(q):
            q = q.iloc[0]
            d.update(fp_null=int(q.loo_adm), fp_null_tests=int(q.loo_tests), fp_null_per_draw=float(q.loo_per_draw),
                     fp_null_rate=q.loo_adm / q.loo_tests if q.loo_tests else np.nan, fp_random=int(q.random_adm),
                     fp_random_tests=int(q.random_tests),
                     null_share=(q.uncal_null_sig / q.uncal_null_rel) if ("uncal_null_sig" in q and np.isfinite(q.uncal_null_sig)
                                                                           and q.uncal_null_rel) else np.nan,
                     null_distinct_share=(q.uncal_null_distinct_sig / q.uncal_null_distinct)
                     if ("uncal_null_distinct_sig" in q and np.isfinite(q.uncal_null_distinct_sig) and q.uncal_null_distinct) else np.nan)
        rows.append(d)
    return pd.DataFrame(rows)


def config_summary(S: pd.DataFrame, design: dict) -> pd.DataFrame:
    """Active and inert cells of a configuration joined (inert of an Emax configuration = the linear inert cell),
    averaged over replicates: one row per configuration x method."""
    configs = design.get("configs", {})

    def alias(name: str) -> str:
        c = configs.get(name)
        if not c or c.get("emax_c50", 0) <= 0:
            return name
        base = {**c, "emax_c50": 0.0, "emax_gamma": 1.0}
        for k, v in configs.items():
            if v == base:
                return k
        return name

    A = S[S.drug == "active"]
    Inert = S[S.drug == "inert"]
    rows = []
    for (cfg_name, m), g in A.groupby(["config", "method"]):
        gi = Inert[(Inert.config == alias(cfg_name)) & (Inert.method == m)]
        b = g.iloc[0]
        d = dict(config=cfg_name, method=m, scale=b.scale, gain=b.gain, w_z=b.w_z, emax_c50=b.emax_c50, emax_gamma=b.emax_gamma,
                 ant=b.ant, exog_remi=b.exog_remi, reps=int(len(g)), inert_reps=int(len(gi)), inert_config=alias(cfg_name),
                 n_true=float(g.n_true.mean()), tp=float(g.tp.mean()), wrong=float(g.wrong.mean()),
                 power=float(g.power.mean()), wrong_rate=float(g.wrong_rate.mean()),
                 fp_inert=float(gi.fp_inert.mean()) if len(gi) else np.nan,
                 fp_inert_rate=float(gi.fp_inert_rate.mean()) if len(gi) else np.nan,
                 fp_null_rate=float(g.fp_null_rate.mean()) if "fp_null_rate" in g else np.nan,
                 fp_null_per_draw=float(g.fp_null_per_draw.mean()) if "fp_null_per_draw" in g else np.nan,
                 fp_null_rate_inert=float(gi.fp_null_rate.mean()) if (len(gi) and "fp_null_rate" in gi) else np.nan,
                 fp_random=float(g.fp_random.mean()) if "fp_random" in g else np.nan,
                 null_share=float(g.null_share.mean()) if "null_share" in g else np.nan,
                 null_share_inert=float(gi.null_share.mean()) if (len(gi) and "null_share" in gi) else np.nan,
                 pol_correct=float(g.pol_correct.mean()), pol_wrong=float(g.pol_wrong.mean()), n_pol=int(b.n_pol),
                 uncal_power=float(g.uncal_power.mean()), uncal_wrong=float(g.uncal_wrong.mean()),
                 uncal_fp_inert_rate=float(gi.uncal_fp_inert_rate.mean()) if len(gi) else np.nan,
                 bias=float(g.bias.mean()), rel_bias=float(g.rel_bias.mean()), rmse=float(g.rmse.mean()),
                 bias_inert=float(gi.bias.mean()) if len(gi) else np.nan, rmse_inert=float(gi.rmse.mean()) if len(gi) else np.nan,
                 uncal_rmse=float(g.uncal_rmse.mean()), mean_truth=float(g.mean_truth.mean()),
                 remi_per_patient=float(g.remi_per_patient.mean()), vent_per_patient=float(g.vent_per_patient.mean()),
                 mean_n_cases=float(g.mean_n_cases.mean()))
        rows.append(d)
    return pd.DataFrame(rows)


# ── tables ───────────────────────────────────────────────────────────────────────────────────────────

def _fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, cols: list[str], names: list[str] | None = None, nd: int = 2) -> str:
    names = names or cols
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def metric_table(G: pd.DataFrame, configs: list[str], metric: str, methods: tuple[str, ...] = ALL_METHODS, nd: int = 2,
                 label: dict[str, str] | None = None) -> str:
    cols = [LABELS.get(m, m) for m in methods]
    out = ["| configuration | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for c in configs:
        g = G[G.config == c].set_index("method")
        vals = [_fmt(float(g.loc[m, metric]), nd) if m in g.index else "—" for m in methods]
        out.append(f"| {label.get(c, c) if label else c} | " + " | ".join(vals) + " |")
    return "\n".join(out)


# ── figures ──────────────────────────────────────────────────────────────────────────────────────────

def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
                         "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    return plt


def _cmap(colors):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", colors)


def _heat(ax, M: np.ndarray, xs, ys, cmap, vmin, vmax, title, fmt="{:.2f}", xlabel="unrecorded confounding $w_z$",
          ylabel="feedback gain"):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", origin="lower")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([f"{y:g}×" for y in ys])
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                dark = (v - vmin) / max(vmax - vmin, 1e-9) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="#ffffff" if dark else INK)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=MUTED)


def grid_matrix(G: pd.DataFrame, methods: tuple[str, ...], metric: str, gains=(0.5, 1.0, 2.0), wzs=(0.0, 0.5, 1.0),
                scale: str = "2s") -> np.ndarray:
    M = np.full((len(gains), len(wzs)), np.nan)
    for i, g in enumerate(gains):
        for j, wz in enumerate(wzs):
            q = G[(G.scale == scale) & (G.gain == g) & (G.w_z == wz) & (G.emax_c50 == 0) & (G.ant == 0) & G.method.isin(methods)
                  & (G.exog_remi == 1.0)]
            if len(q):
                M[i, j] = float(q[metric].mean())
    return M


def fig_grid_groups(G: pd.DataFrame, path: Path) -> None:
    plt = _style()
    metrics = [("power", "power\n(TP / true relations)", SEQ_BLUE),
               ("wrong_rate", "wrong-sign rate\n(wrong / true relations)", SEQ_ORANGE),
               ("fp_inert_rate", "false positives\n(inert drug, truth 0)", SEQ_ORANGE)]
    groups = list(GROUPS.items()) + [("typed composite", (TYPED,))]
    fig, axes = plt.subplots(len(metrics), len(groups), figsize=(2.55 * len(groups), 2.5 * len(metrics)), squeeze=False)
    for i, (metric, rlab, ramp) in enumerate(metrics):
        for j, (gname, ms) in enumerate(groups):
            _heat(axes[i, j], grid_matrix(G, ms, metric), (0.0, 0.5, 1.0), (0.5, 1.0, 2.0), _cmap(ramp), 0, 1,
                  gname if i == 0 else "", xlabel="$w_z$" if i == len(metrics) - 1 else "", ylabel="gain" if j == 0 else "")
        axes[i, 0].text(-0.62, 0.5, rlab, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="center",
                        fontsize=9, color=INK)
    fig.suptitle("Feedback × unrecorded confounding (2 s, linear, reactive policy)\nmean over the estimators in each group and over seeds; "
                 "$w_z$ = weight on the unrecorded surgical field in the clinician's policy", fontsize=9.5, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0.03, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_grid_methods(G: pd.DataFrame, path: Path, metric: str, title: str, ramp) -> None:
    plt = _style()
    ms = ALL_METHODS
    ncol = 5
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.5 * nrow), squeeze=False)
    for k, m in enumerate(ms):
        ax = axes[k // ncol, k % ncol]
        _heat(ax, grid_matrix(G, (m,), metric), (0.0, 0.5, 1.0), (0.5, 1.0, 2.0), _cmap(ramp), 0, 1, LABELS[m],
              xlabel="$w_z$" if k // ncol == nrow - 1 else "", ylabel="gain" if k % ncol == 0 else "")
    for k in range(len(ms), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_contrast(G: pd.DataFrame, path: Path, configs: list[tuple[str, str, str]], title: str,
                 metrics=(("power", "power"), ("wrong_rate", "wrong-sign rate"), ("fp_inert_rate", "FP rate, inert drug"),
                          ("fp_null_rate", "FP rate, held-out transplant")), extra_bias: bool = True) -> None:
    """Dot plot: estimators on the y axis, one dot per configuration (colour = configuration), one panel per metric."""
    plt = _style()
    ms = list(ALL_METHODS)
    mets = list(metrics) + ([("rel_bias", "relative bias (event units)")] if extra_bias else [])
    fig, axes = plt.subplots(1, len(mets), figsize=(2.6 * len(mets) + 1.5, 0.34 * len(ms) + 1.6), sharey=True)
    y = np.arange(len(ms))[::-1]
    marks = ("o", "s", "^")  # secondary encoding: identity never rests on colour alone
    for ax, (metric, mtitle) in zip(axes, mets):
        for k, (cfg_name, lab, col) in enumerate(configs):
            g = G[G.config == cfg_name].set_index("method")
            v = np.array([float(g.loc[m, metric]) if m in g.index else np.nan for m in ms])
            ax.scatter(v, y, s=30, color=col, marker=marks[k % len(marks)], label=lab, zorder=3, edgecolor=SURFACE,
                       linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[m] for m in ms])
        ax.set_title(mtitle, fontsize=9, loc="left")
        ax.grid(axis="x", zorder=0)
        ax.tick_params(length=0)
        if metric == "rel_bias":
            ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=1)
            ax.set_xlim(-3.2, 1.2)  # the naive level contrast sits near -3
        else:
            ax.set_xlim(-0.05, 1.05)
        for yy in y:
            ax.axhline(yy, color=GRID, lw=0.4, zorder=0)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    # legend on the figure, not on an axis: a wide legend attached to axes[0] makes tight_layout squeeze every panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.97), ncol=len(configs), fontsize=8,
               handletextpad=0.3, columnspacing=1.2, frameon=False)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK, y=0.995)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ── driver ───────────────────────────────────────────────────────────────────────────────────────────

def summarize(out: Path) -> dict:
    L, Nt, info = load_cells(out)
    if not len(L):
        raise SystemExit("no finished cells")
    S = cell_summary(L, Nt)
    G = config_summary(S, info["design"])
    L.to_csv(out / "cells_long.csv", index=False)
    S.to_csv(out / "cells_summary.csv", index=False)
    G.to_csv(out / "config_summary.csv", index=False)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    done = set(G.config)
    grid_cfgs = [f"g{g:g}_wz{wz:g}" for g in (0.5, 1.0, 2.0) for wz in (0.0, 0.5, 1.0)]
    if any(c in done for c in grid_cfgs):
        fig_grid_groups(G, figs / "grid_groups.png")
        fig_grid_methods(G, figs / "grid_power_methods.png", "power", "Power per estimator (feedback × confounding, 2 s)", SEQ_BLUE)
        fig_grid_methods(G, figs / "grid_fp_inert_methods.png", "fp_inert_rate",
                         "False-positive rate on the inert drug per estimator (feedback × confounding, 2 s)", SEQ_ORANGE)
        fig_grid_methods(G, figs / "grid_wrong_methods.png", "wrong_rate",
                         "Wrong-sign admission rate per estimator (feedback × confounding, 2 s)", SEQ_ORANGE)
    panels = [
        ("sampling.png", [("g1_wz0", "2 s", C_BLUE), ("scale30", "30 s", C_ORANGE)], "Sampling rate (gain 1, w_z = 0, linear)"),
        ("sampling_g2_wz1.png", [("g2_wz1", "2 s", C_BLUE), ("scale30_g2_wz1", "30 s", C_ORANGE)],
         "Sampling rate under strong feedback and confounding (gain 2, w_z = 1)"),
        ("nonlinearity.png", [("g1_wz0", "linear", C_BLUE), ("emax_g3", "Emax C50 = 3, γ = 3", C_ORANGE),
                              ("emax_strong", "Emax C50 = 1, γ = 3 (strong)", C_AQUA)],
         "Response nonlinearity (gain 1, w_z = 0, 2 s; inert cohort shared with the linear arm)"),
        ("nonlinearity_wz1.png", [("g1_wz1", "linear", C_BLUE), ("emax_g3_wz1", "Emax C50 = 3, γ = 3", C_ORANGE),
                                  ("emax_strong_wz1", "Emax C50 = 1, γ = 3 (strong)", C_AQUA)],
         "Response nonlinearity under confounding (gain 1, w_z = 1, 2 s)"),
        ("nonlinearity_gamma.png", [("g1_wz0", "linear", C_BLUE), ("emax_g1", "Emax C50 = 3, γ = 1", C_ORANGE),
                                    ("emax_g3", "Emax C50 = 3, γ = 3", C_AQUA)],
         "Hill coefficient at the same C50 (gain 1, w_z = 0, 2 s)"),
        ("policy.png", [("g1_wz0", "reactive", C_BLUE), ("anticip", "anticipatory (plan 60 s)", C_ORANGE),
                        ("anticip_field", "anticipatory + field (w_z = 1)", C_AQUA)], "Policy type (gain 1, 2 s, linear)"),
        ("openloop.png", [("openloop", "open loop (exogenous 1/h)", C_BLUE), ("openloop_matched", "open loop, gain-1 event rate", C_ORANGE),
                          ("g1_wz0", "closed loop, gain 1", C_AQUA)], "Open loop vs closed loop (2 s, linear)"),
    ]
    for fname, cfgs, title in panels:
        cfgs = [c for c in cfgs if c[0] in done]
        if len(cfgs) >= 2:
            fig_contrast(G, figs / fname, cfgs, title)
    (out / "tables.md").write_text(tables_md(G, S, L, info, out))
    return dict(L=L, S=S, G=G, info=info)


def estimator_table(G: pd.DataFrame, config: str, metrics: list[tuple[str, str, int]],
                    methods: tuple[str, ...] = METHODS) -> str:
    """Rows = estimators (the 12 benchmark estimators by default), columns = metrics, for one configuration."""
    g = G[G.config == config].set_index("method")
    out = ["| estimator | " + " | ".join(t for _, t, _ in metrics) + " |", "|---|" + "---|" * len(metrics)]
    for m in methods:
        if m not in g.index:
            continue
        out.append(f"| {LABELS[m]} | " + " | ".join(_fmt(float(g.loc[m, k]), nd) for k, _, nd in metrics) + " |")
    return "\n".join(out)


def sweep_table(G: pd.DataFrame, configs: list[tuple[str, str]], metric: str, nd: int = 2,
                methods: tuple[str, ...] = METHODS) -> str:
    """Rows = estimators (the 12 benchmark estimators by default), columns = configurations, one metric."""
    cols = [lab for _, lab in configs if _ in set(G.config)]
    keys = [c for c, _ in configs if c in set(G.config)]
    out = ["| estimator | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in methods:
        vals = []
        for c in keys:
            g = G[(G.config == c) & (G.method == m)]
            vals.append(_fmt(float(g[metric].iloc[0]), nd) if len(g) else "—")
        if all(v == "—" for v in vals):
            continue
        out.append(f"| {LABELS[m]} | " + " | ".join(vals) + " |")
    return "\n".join(out)


DESIGN_BASED: tuple[str, ...] = ("matched", "kink", "seqtrial")
REL_GROUPS: dict[str, tuple[str, ...]] = {"remi up (R05, R06)": ("R05", "R06"), "remi down (R07, R08)": ("R07", "R08"),
                                          "vent (R01, R02)": ("R01", "R02")}


def horizon_bias(out_dir: Path, config: str) -> pd.DataFrame | None:
    """Bias by horizon of the design-based estimators in one configuration: real-arm estimate minus the ITT truth at the
    same horizon, averaged over the relations of each group and over replicates. The calibrated estimate exists only at
    the primary horizon, so the uncalibrated real-arm estimate is used at every horizon."""
    frames = []
    for d in sorted(out_dir.glob(f"cells/{config}__active__r*")):
        if not (d / "pooled.csv").exists():
            continue
        P = pd.read_csv(d / "pooled.csv")
        T = pd.read_csv(d / "truth.csv").set_index(["rid", "h"]).truth
        q = P[(P.arm == "real") & P.method.isin(DESIGN_BASED) & P.rid.isin(RESP)][["method", "rid", "h", "est"]].copy()
        q["truth"] = [T.get((r, h), np.nan) for r, h in zip(q.rid, q.h)]
        q["bias"] = q.est - q.truth
        q["group"] = [next(g for g, rids in REL_GROUPS.items() if r in rids) for r in q.rid]
        frames.append(q)
    if not frames:
        return None
    Q = pd.concat(frames)
    return Q.groupby(["group", "h", "method"]).agg(bias=("bias", "mean"), truth=("truth", "mean")).reset_index()


def horizon_bias_md(out_dir: Path) -> str:
    out = ["Bias = real-arm estimate − ITT truth at the same horizon (vital units per titration step), mean over the "
           "relations of the group and over replicates; the truth is given for the group. The paper's corollary for an "
           "anticipatory clinician predicts no bias for horizons up to the lead and a positive bias beyond it; the plan "
           "window here is (t, t + 60 s], so the lead is spread over 0-60 s.", ""]
    for config, label in (("anticip", "anticipatory (plan 60 s), w_z = 0"), ("anticip_field", "anticipatory + field, w_z = 1"),
                          ("g1_wz0", "reactive (default), for contrast")):
        H = horizon_bias(out_dir, config)
        if H is None:
            continue
        out += [f"**{label}**", "",
                "| relation group | h (s) | ITT truth | matched | interrupted trend | sequential trial |",
                "|---|---|---|---|---|---|"]
        for grp in REL_GROUPS:
            for h in sorted(H.h.unique()):
                sub = H[(H.group == grp) & (H.h == h)].set_index("method")
                if not len(sub):
                    continue
                cells = [_fmt(float(sub.loc[m, "bias"])) if m in sub.index else "—" for m in DESIGN_BASED]
                out.append(f"| {grp} | {h:g} | {_fmt(float(sub.truth.iloc[0]))} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


PUBLISHED_SIM = Path("runs/iclr/closed_loop_v3/sim_validation/sim_validation.csv")


def published_recorded(root: Path, scale: str) -> pd.DataFrame:
    """The paper's simulator validation (``--sim-validate``, n = 800, K = 4 cross draws, 2 seeds) restricted to the
    *recorded* clinician at w_z = 0 — the policy and confounding of this grid's default cell. Read-only.

    Rates are per relation-cell (6 response relations x 2 seeds = 12), so they are directly comparable to this
    grid's per-cell power / FP_inert rates."""
    R = pd.read_csv(root / PUBLISHED_SIM)
    G = R[(R.kind == "response") & R.primary.astype(bool) & (R.clinician == "recorded") & (R.w_z == 0.0)
          & (R.scale == scale)]
    rows = []
    for (m, drug), g in G.groupby(["method", "drug"]):
        adm = g.cal_admitted.fillna(False).astype(bool)
        sig = (g.lo > 0) | (g.hi < 0)
        right = np.sign(g.cal_est) == np.sign(g.truth)
        rows.append(dict(method=m, drug=drug, cells=len(g),
                         admitted=int((adm & right).sum()) if drug == "active" else int(adm.sum()),
                         rate=float((adm & right).mean()) if drug == "active" else float(adm.mean()),
                         uncal=int((sig & (np.sign(g.est) == np.sign(g.truth))).sum()) if drug == "active" else int(sig.sum()),
                         uncal_rate=float((sig & (np.sign(g.est) == np.sign(g.truth))).mean()) if drug == "active" else float(sig.mean())))
    return pd.DataFrame(rows)


def validation_md(G: pd.DataFrame, root: Path) -> str:
    """Compare this grid's default cells with the paper's published simulator validation."""
    out = ["### Default cell vs the paper's published simulator validation", "",
           "Paper: `runs/iclr/closed_loop_v3/sim_validation/` (n = 800, K = 4 transplant draws, 2 seeds), restricted to the "
           "**recorded** clinician at w_z = 0 — this grid's default policy and confounding. This grid: n = 600, K = 8 draws, "
           "2 seeds, different seed values, so agreement is expected in level and ordering, not to the decimal.",
           "Uncal. = single-arm CI excluding 0 (the paper's published simulator claim is stated in these terms); "
           "the paper's run reported only the event-design and VARX methods, so the lag-based rows are blank.", "",
           "| estimator | power paper | power grid | FP inert paper | FP inert grid | uncal. FP inert paper | uncal. FP inert grid |",
           "|---|---|---|---|---|---|---|"]
    for scale, cfg in (("2s", "g1_wz0"), ("30s", "scale30")):
        if cfg not in set(G.config):
            continue
        try:
            P = published_recorded(root, scale)
        except FileNotFoundError:
            continue
        out.append(f"| **scale {scale}** | | | | | | |")
        g = G[G.config == cfg].set_index("method")
        for m in ALL_METHODS:
            if m not in g.index:
                continue
            pa = P[(P.method == m) & (P.drug == "active")]
            pi = P[(P.method == m) & (P.drug == "inert")]
            out.append(f"| {LABELS[m]} | {_fmt(float(pa.rate.iloc[0])) if len(pa) else '—'} | {_fmt(float(g.loc[m, 'power']))} | "
                       f"{_fmt(float(pi.rate.iloc[0])) if len(pi) else '—'} | {_fmt(float(g.loc[m, 'fp_inert_rate']))} | "
                       f"{_fmt(float(pi.uncal_rate.iloc[0])) if len(pi) else '—'} | {_fmt(float(g.loc[m, 'uncal_fp_inert_rate']))} |")
    return "\n".join(out)


def tables_md(G: pd.DataFrame, S: pd.DataFrame, L: pd.DataFrame, info: dict, out_dir: Path | None = None) -> str:
    root = Path(__file__).resolve().parents[3]
    out_dir = out_dir or (root / "runs" / "extended" / "C_sim_grid")
    cfg_order = [c for c in info["design"].get("configs", {}) if c in set(G.config)]
    lab = {c: c for c in cfg_order}
    out = ["# C_sim_grid tables (auto-generated)", "",
           f"Cells finished: {S.cell.nunique()} ({(S.drug == 'active').groupby(S.cell).first().sum()} active, "
           f"{(S.drug == 'inert').groupby(S.cell).first().sum()} inert); configurations: {len(cfg_order)}.", ""]
    for metric, title, nd in (("power", "Power (TP / true response relations; 6 relations; mean over seeds)", 2),
                              ("wrong_rate", "Wrong-sign admission rate (true response relations)", 2),
                              ("fp_inert_rate", "False-positive rate on the inert drug (6 response relations, truth 0)", 2),
                              ("fp_null_rate", "False admissions with a held-out transplant draw as input (active cohort; sim relations, 8 LOO draws)", 3),
                              ("fp_random", "Admissions with the random-time stream as input (count of 9 sim relations, active cohort)", 2),
                              ("null_share", "Uncalibrated null-discovery share (single-arm CI excl. 0 per transplant draw / relations; active cohort)", 2),
                              ("rel_bias", "Relative bias of the calibrated estimate, (est − ITT truth) / |truth| (event-unit estimators and VARX per step)", 2),
                              ("rmse", "RMSE of the calibrated estimate vs ITT truth (vital units per titration step)", 2),
                              ("rmse_inert", "RMSE on the inert drug (truth 0)", 2),
                              ("pol_correct", "Policy relations (P01, P02, P05) admitted with the expected sign (count)", 2),
                              ("pol_wrong", "Policy relations admitted with the wrong sign (count)", 2)):
        out += [f"## {title}", "", metric_table(G, cfg_order, metric, nd=nd, label=lab), ""]
    out += ["## Event counts per simulated patient (2 h) and contributing cases", "",
            md_table(G[G.method == "matched"][["config", "scale", "gain", "w_z", "emax_gamma", "ant", "exog_remi", "remi_per_patient",
                                                "vent_per_patient", "mean_n_cases", "n_true", "mean_truth"]].reset_index(drop=True),
                     ["config", "scale", "gain", "w_z", "emax_gamma", "ant", "exog_remi", "remi_per_patient", "vent_per_patient",
                      "mean_n_cases", "n_true", "mean_truth"]), ""]
    if (root / PUBLISHED_SIM).exists():
        out += ["## Validation", "", validation_md(G, root), ""]
    out += ["## Headline tables (estimators as rows)", "", headline_md(G), ""]
    out += ["## Anticipatory policy: bias by horizon (design-based estimators)", "", horizon_bias_md(out_dir), ""]
    return "\n".join(out)


HEADLINE_METRICS: list[tuple[str, str, int]] = [
    ("power", "power", 2), ("wrong_rate", "wrong-sign", 2), ("fp_inert_rate", "FP inert", 2),
    ("fp_null_rate", "FP transplant", 3), ("fp_random", "FP random (of 9)", 1), ("null_share", "uncal. null share", 2),
    ("rel_bias", "rel. bias", 2), ("rmse", "RMSE", 2)]


def headline_md(G: pd.DataFrame) -> str:
    done = set(G.config)
    out: list[str] = []
    if "g1_wz0" in done:
        out += ["### Default cell: 2 s, gain 1, w_z = 0, linear, reactive (the paper's configuration)", "",
                estimator_table(G, "g1_wz0", HEADLINE_METRICS), ""]
    sweeps = [
        ("Unrecorded confounding at gain 1 (2 s, linear, reactive)",
         [("g1_wz0", "w_z = 0"), ("g1_wz0.5", "w_z = 0.5"), ("g1_wz1", "w_z = 1")]),
        ("Feedback strength at w_z = 0 (2 s, linear, reactive)",
         [("openloop", "gain 0"), ("openloop_matched", "gain 0, matched rate"), ("g0.5_wz0", "gain 0.5"),
          ("g1_wz0", "gain 1"), ("g2_wz0", "gain 2")]),
        ("Sampling rate", [("g1_wz0", "2 s"), ("scale30", "30 s"), ("g2_wz1", "2 s, gain 2 w_z 1"),
                           ("scale30_g2_wz1", "30 s, gain 2 w_z 1")]),
        ("Response nonlinearity (gain 1, 2 s)",
         [("g1_wz0", "linear"), ("emax_g1", "Emax C50 3 γ1"), ("emax_g3", "Emax C50 3 γ3"), ("emax_strong", "Emax C50 1 γ3"),
          ("g1_wz1", "linear, w_z 1"), ("emax_g3_wz1", "Emax C50 3 γ3, w_z 1"), ("emax_strong_wz1", "Emax C50 1 γ3, w_z 1")]),
        ("Policy type (gain 1, 2 s, linear)",
         [("g1_wz0", "reactive"), ("anticip", "anticipatory"), ("g1_wz1", "reactive + field"),
          ("anticip_field", "anticipatory + field")]),
    ]
    for title, cfgs in sweeps:
        if sum(c in done for c, _ in cfgs) < 2:
            continue
        out += [f"### {title}", ""]
        for metric, mlabel, nd in (("power", "power (TP / true relations)", 2),
                                   ("fp_inert_rate", "false-positive rate, inert drug", 2),
                                   ("wrong_rate", "wrong-sign rate", 2),
                                   ("fp_null_rate", "false admissions, held-out transplant", 3),
                                   ("rmse", "RMSE vs ITT truth (event-design estimators)", 2)):
            out += [f"**{mlabel}**", "", sweep_table(G, cfgs, metric, nd), ""]
    out += typed_md(G)
    return "\n".join(out)


def typed_md(G: pd.DataFrame) -> list[str]:
    """The typed composite (CLBench's typed edge = the VARX estimate under the same calibrated rule), kept out of the
    12-estimator headline tables."""
    cfgs = [("g1_wz0", "default"), ("g1_wz0.5", "w_z 0.5"), ("g1_wz1", "w_z 1"), ("openloop_matched", "open loop"),
            ("g2_wz0", "gain 2"), ("scale30", "30 s"), ("emax_strong", "Emax C50 1"), ("anticip", "anticipatory")]
    cfgs = [c for c in cfgs if c[0] in set(G.config)]
    if not cfgs:
        return []
    rows = ["### Typed composite (CLBench typed edge; VARX-based, same calibrated rule; not one of the 12)", "",
            "| metric | " + " | ".join(lab for _, lab in cfgs) + " |", "|---|" + "---|" * len(cfgs)]
    for metric, mlabel, nd in (("power", "power", 2), ("wrong_rate", "wrong-sign rate", 2),
                               ("fp_inert_rate", "FP rate, inert drug", 2), ("fp_null_rate", "FP, held-out transplant", 3),
                               ("fp_random", "FP, random-time input (of 9)", 1)):
        vals = []
        for c, _ in cfgs:
            g = G[(G.config == c) & (G.method == TYPED)]
            vals.append(_fmt(float(g[metric].iloc[0]), nd) if len(g) else "—")
        rows.append(f"| {mlabel} | " + " | ".join(vals) + " |")
    return rows + [""]


__all__ = ["ALL_METHODS", "GROUPS", "LABELS", "METHODS", "cell_summary", "config_summary", "load_cells", "summarize"]
