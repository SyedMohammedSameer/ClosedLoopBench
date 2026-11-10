"""
Extended analysis B: ClosedLoopBench results by evidence tier.

Re-reads the published benchmark outputs (no estimator is re-run) and reports, separately for three
pre-declared tiers of sign evidence, every estimator's calibrated admissions, uncalibrated null discoveries,
held-out transplant false admissions and random-time false admissions, two pre-declared rankings, the
tier-A verdict on the paper's headline (set-rate -> EtCO2 sign) and the per-relation call matrix.

Tiers (fixed before looking at the results):
  A  respiratory physics          R01, R02, R13, R14, R15
  B  pharmacology / physiology    R03-R12, R16, R17, R18, R19
     B1 pharmacology              R03-R09, R11, R12, R17, R18, R19
     B2 baroreflex / cardiopulm.  R10, R16
  C  documented practice (policy) P01-P10

Primary resolution: runs/iclr/closed_loop_v4/2s_all (ClosedLoopBench 1.1, 29 relations, 3,442 cases; the paper's
Table 1). Secondary: runs/iclr/closed_loop_v3/30s_all (ClosedLoopBench 1.0, 18 relations R01-R12, P01-P06; the
paper's App. tab:bench-30s). Tier membership at 30 s is the intersection with the 18 relations.

Run (repo root; pandas only, 1 core, ~10 s):
  PYTHONPATH=src .venv/bin/python scripts/extended/B_tiers.py [--out DIR]

Outputs: runs/extended/B_tiers/{2s,30s}/*.csv|md|tex and extended/results/B_tiers.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from clbench.eval import closed_loop_bench as clb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "extended" / "B_tiers"
NOTE = ROOT / "extended" / "results" / "B_tiers.md"

RUNS = {
    "2s": dict(dir=ROOT / "runs/iclr/closed_loop_v4/2s_all", table="v4", scale="2s_v4",
               title="2 s (primary; ClosedLoopBench 1.1, v4, 29 relations, 3,442 cases)"),
    "30s": dict(dir=ROOT / "runs/iclr/closed_loop_v3/30s_all", table="v3", scale="30s",
                title="30 s (secondary; ClosedLoopBench 1.0, v3, 18 relations, 3,442 cases)"),
}

# ── Tiers (pre-declared) ───────────────────────────────────────────────────────────────────────────────
TIERS: dict[str, tuple[str, ...]] = {
    "A": ("R01", "R02", "R13", "R14", "R15"),
    "B": tuple(f"R{i:02d}" for i in range(3, 13)) + ("R16", "R17", "R18", "R19"),
    "B1": tuple(f"R{i:02d}" for i in range(3, 10)) + ("R11", "R12", "R17", "R18", "R19"),
    "B2": ("R10", "R16"),
    "C": tuple(f"P{i:02d}" for i in range(1, 11)),
}
TIER_NAME = {"A": "respiratory physics", "B": "pharmacology/physiology", "B1": "pharmacology",
             "B2": "baroreflex/cardiopulmonary", "C": "documented practice (policy)"}
MAIN_TIERS = ("A", "B", "C")

# ── Estimators ─────────────────────────────────────────────────────────────────────────────────────────
# Table 1 order: lag-based first, then event designs.
METHODS: tuple[str, ...] = ("xcorr_pos", "xcorr_v1", "xcorr_dir", "granger", "pcmciplus", "varlingam", "varx",
                            "naive", "naive_lvl", "matched", "kink", "seqtrial")
LAG_BASED = ("xcorr_pos", "xcorr_v1", "xcorr_dir", "granger", "pcmciplus", "varlingam", "varx")
EVENT = ("naive", "naive_lvl", "matched", "kink", "seqtrial")
DESIGN = ("matched", "kink", "seqtrial")
NAME = {"xcorr_pos": "Positive-peak cross-corr.", "xcorr_v1": "Signed-peak cross-corr.",
        "xcorr_dir": "Direction-aware cross-corr.", "granger": "Pairwise Granger", "pcmciplus": "PCMCI+",
        "varlingam": "VAR-LiNGAM", "varx": "VARX", "naive": "Naive event contrast",
        "naive_lvl": "Naive level contrast", "matched": "Matched event design",
        "kink": "Interrupted trend (kink)", "seqtrial": "Sequential-trial contrast"}
SHORT = {"xcorr_pos": "XP", "xcorr_v1": "XS", "xcorr_dir": "XD", "granger": "GR", "pcmciplus": "PC",
         "varlingam": "VL", "varx": "VX", "naive": "NE", "naive_lvl": "NL", "matched": "MA", "kink": "KI",
         "seqtrial": "ST"}


def family(m: str) -> str:
    if m in DESIGN:
        return "design-based"
    return "lag-based" if m in LAG_BASED else "event (naive)"


# ── Published numbers to validate against ─────────────────────────────────────────────────────────────
# iclr2027/sections/exp_benchmark.tex, Table 1 (tab:benchmark): null uncal./distinct, Gas(5), Drug(11),
# Vol.(2), Policy(10) expected/wrong, RT false admissions.
TABLE1 = {
    "xcorr_pos": ("11.00", "19", "1/3", "1/6", "0/0", "4/2", 13),
    "xcorr_v1": ("11.38", "19", "2/2", "5/0", "2/0", "1/5", 13),
    "xcorr_dir": ("18.12", "29", "2/2", "6/0", "2/0", "4/4", 10),
    "granger": ("14.75", "24", "2/3", "8/0", "2/0", "8/2", 16),
    "pcmciplus": ("3.88", "28-29", "0/0", "0/0", "1/0", "2/0", 0),
    "varlingam": ("12.12", "24", "5/0", "8/0", "2/0", "7/1", 12),
    "varx": ("7.88", "29", "4/1", "4/1", "2/0", "6/2", 4),
    "naive": ("18.25", "29", "4/0", "2/4", "2/0", "8/1", 7),
    "naive_lvl": ("6.38", "19", "3/0", "5/2", "2/0", "n/a", 2),
    "matched": ("2.75", "29", "4/0", "3/0", "2/0", "5/2", 0),
    "kink": ("3.62", "29", "4/0", "4/0", "2/0", "5/2", 0),
    "seqtrial": ("9.25", "29", "4/0", "3/0", "2/0", "7/1", 4),
}
TABLE1_DOMAINS = {"Gas (5)": "gas_exchange", "Drug (11)": "drug", "Vol. (2)": "drug_derived", "Policy (10)": "policy"}
# iclr2027/sections/appendix_benchmark.tex, tab:bench-30s: null uncal./distinct, Gas(2), Drug(10), Policy(6),
# held-out FA per draw, RT.
TABLE30 = {
    "naive": ("10.12", "18", "2/0", "1/5", "6/0", "0", 6),
    "naive_lvl": ("5.25", "12", "2/0", "4/2", "n/a", "0", 1),
    "xcorr_pos": ("10.00", "12", "0/2", "0/5", "4/0", "0", 9),
    "xcorr_v1": ("7.75", "12", "0/2", "4/0", "1/2", "0", 2),
    "xcorr_dir": ("12.50", "18", "0/2", "5/0", "4/1", "0", 5),
    "granger": ("6.50", "14", "2/0", "6/0", "5/1", "0.125", 6),
    "pcmciplus": ("8.62", "18", "2/0", "6/0", "5/0", "0", 0),
    "varlingam": ("5.00", "14", "2/0", "4/0", "6/0", "0.25", 4),
    "varx": ("4.25", "18", "2/0", "3/2", "2/1", "0", 3),
    "matched": ("1.50", "18", "2/0", "1/0", "3/0", "0", 0),
    "kink": ("2.12", "18", "2/0", "0/1", "3/0", "0", 0),
    "seqtrial": ("1.25", "18", "2/0", "0/2", "4/0", "0", 0),
}
TABLE30_DOMAINS = {"Gas (2)": "gas_exchange", "Drug (10)": "drug", "Policy (6)": "policy"}


# ── Loading ────────────────────────────────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, key: str):
        spec = RUNS[key]
        self.key, self.dir, self.title = key, spec["dir"], spec["title"]
        clb.use_table(spec["table"])
        self.cfg = clb.SCALES[spec["scale"]]
        self.relations = list(clb.RELATIONS)
        self.rel = {r.rid: r for r in self.relations}
        self.domain = {r.rid: clb.relation_domain(r) for r in self.relations}
        self.rids = [r.rid for r in self.relations]
        self.calib = pd.read_csv(self.dir / "calibrated.csv")
        self.pooled = pd.read_csv(self.dir / "pooled.csv")
        self.nulls = pd.read_csv(self.dir / "null_discovery.csv")
        self.leader = pd.read_csv(self.dir / "leaderboard.csv")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.tiers = {t: tuple(x for x in rids if x in self.rel) for t, rids in TIERS.items()}
        # primary-horizon rows of pooled.csv (the rows calibrated.csv is built from)
        ph = {(m, r.rid): clb.primary_h(r, m, self.cfg) for m in METHODS for r in self.relations}
        P = self.pooled[self.pooled.method.isin(METHODS) & self.pooled.rid.isin(self.rids)].copy()
        P["h_primary"] = [ph[(m, r)] for m, r in zip(P.method, P.rid)]
        self.primary = P[P.h == P.h_primary].drop(columns="h_primary")
        # the h used in calibrated.csv must agree with primary_h
        C = self.calib[self.calib.method.isin(METHODS)]
        bad = [(m, r, h) for m, r, h in zip(C.method, C.rid, C.h) if ph[(m, r)] != h]
        assert not bad, f"primary horizon mismatch: {bad[:5]}"
        self.real = self.calib[(self.calib.input == "real") & self.calib.method.isin(METHODS)]
        self.rt = self.calib[(self.calib.input == "nc_random") & self.calib.method.isin(METHODS)]
        self.loo = self.calib[self.calib.input.str.startswith("loo:") & self.calib.method.isin(METHODS)]
        self.n_draws = self.loo.input.nunique()


# ── Metrics ────────────────────────────────────────────────────────────────────────────────────────────
def distinct_key(F: pd.DataFrame) -> np.ndarray:
    return (F.est.astype(float).round(12).astype(str) + "|" + F.lo.astype(float).round(12).astype(str)
            + "|" + F.hi.astype(float).round(12).astype(str)).to_numpy()


def null_discoveries(run: Run, m: str, rids: tuple[str, ...]) -> dict:
    """Uncalibrated discoveries on the transplant draws restricted to ``rids``: relations whose single-arm
    case-bootstrap CI excludes 0 (``call`` in correct/wrong), counted over distinct estimates
    (``null_discovery_table`` logic) and over relations; mean / min / max over the K draws."""
    P = run.primary[(run.primary.method == m) & run.primary.rid.isin(rids)]
    per_draw, per_draw_rel, n_dist = [], [], []
    for arm in clb.CROSS_ARMS:
        F = P[P.arm == arm]
        F = F[np.isfinite(F.est.astype(float))]
        if len(F) == 0:
            continue
        sig = F.call.isin(["correct", "wrong"]).to_numpy()
        key = distinct_key(F)
        per_draw.append(len(set(key[sig])))
        per_draw_rel.append(int(sig.sum()))
        n_dist.append(len(set(key)))
    if not per_draw:
        return dict(null_distinct_mean=np.nan, null_distinct_min=np.nan, null_distinct_max=np.nan,
                    null_rel_mean=np.nan, n_distinct=np.nan, n_distinct_min=np.nan, n_draws_null=0)
    return dict(null_distinct_mean=float(np.mean(per_draw)), null_distinct_min=int(min(per_draw)),
                null_distinct_max=int(max(per_draw)), null_rel_mean=float(np.mean(per_draw_rel)),
                n_distinct=int(max(n_dist)), n_distinct_min=int(min(n_dist)), n_draws_null=len(per_draw))


def admissions(run: Run, m: str, rids: tuple[str, ...]) -> dict:
    R = run.real[(run.real.method == m) & run.real.rid.isin(rids)]
    if len(R) == 0:
        return dict(n=0, expected=np.nan, wrong=np.nan, not_admitted=np.nan, ns_right=np.nan, ns_wrong=np.nan,
                    expected_distinct=np.nan, wrong_distinct=np.nan, admitted_ids="")
    key = distinct_key(R)
    corr, wrong = (R.call == "correct").to_numpy(), (R.call == "wrong").to_numpy()
    ids = [f"{r}{'+' if c == 'correct' else '-'}" for r, c in zip(R.rid, R.call) if c in ("correct", "wrong")]
    return dict(n=len(R), expected=int(corr.sum()), wrong=int(wrong.sum()),
                not_admitted=int((~(corr | wrong)).sum()), ns_right=int((R.call == "ns_right").sum()),
                ns_wrong=int((R.call == "ns_wrong").sum()), expected_distinct=len(set(key[corr])),
                wrong_distinct=len(set(key[wrong])), admitted_ids=";".join(ids))


def false_admissions(run: Run, m: str, rids: tuple[str, ...]) -> dict:
    L = run.loo[(run.loo.method == m) & run.loo.rid.isin(rids)]
    per = L.groupby("input").admitted.sum() if len(L) else pd.Series(dtype=float)
    T = run.rt[(run.rt.method == m) & run.rt.rid.isin(rids)]
    return dict(heldout_fa_mean=float(per.mean()) if len(per) else np.nan,
                heldout_fa_max=int(per.max()) if len(per) else np.nan,
                rt_fa=int(T.admitted.sum()) if len(T) else np.nan,
                rt_fa_ids=";".join(f"{r}{'+' if c == 'correct' else '-'}" for r, c in zip(T.rid, T.call)
                                   if c in ("correct", "wrong")))


def tier_row(run: Run, tier: str, m: str) -> dict:
    rids = run.tiers[tier]
    d = dict(tier=tier, tier_name=TIER_NAME[tier], method=m, estimator=NAME[m], family=family(m))
    d.update(admissions(run, m, rids))
    d.update(null_discoveries(run, m, rids))
    d.update(false_admissions(run, m, rids))
    return d


def tier_table(run: Run) -> pd.DataFrame:
    rows = [tier_row(run, t, m) for t in TIERS for m in METHODS]
    T = pd.DataFrame(rows)
    T["n_tier"] = [len(run.tiers[t]) for t in T.tier]
    return T


# ── Rankings ───────────────────────────────────────────────────────────────────────────────────────────
def rankings(T: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Rule 1 (reliability first): (wrong + RT false) ascending, then expected descending.
    Rule 2 (score): expected - 2 * wrong - RT false, descending. Ties share the minimum rank."""
    S = T[(T.tier == tier) & T.expected.notna()].copy()
    S["errors"] = S.wrong + S.rt_fa
    S["score2"] = S.expected - 2 * S.wrong - S.rt_fa
    S["rule1_key"] = list(zip(S.errors, -S.expected))
    S = S.sort_values(["errors", "expected", "method"], ascending=[True, False, True]).reset_index(drop=True)
    S["rank_rule1"] = S.rule1_key.rank(method="min").astype(int)
    S["rank_rule2"] = (-S.score2).rank(method="min").astype(int)
    return S.drop(columns="rule1_key")


def design_vs_lag(S: pd.DataFrame, rank_col: str) -> dict:
    """Do the three design-based estimators outrank every lag-based estimator under ``rank_col``?"""
    des = S[S.family == "design-based"]
    lag = S[S.family == "lag-based"]
    worst_design = des[rank_col].max()
    better_or_tied = lag[lag[rank_col] <= worst_design]
    strictly_better = lag[lag[rank_col] < des[rank_col].min()]
    return dict(worst_design_rank=int(worst_design), lag_at_or_above=better_or_tied.method.tolist(),
                lag_strictly_above_all=strictly_better.method.tolist(),
                clean=bool(len(better_or_tied) == 0))


# ── Validation ─────────────────────────────────────────────────────────────────────────────────────────
def pair(d: dict) -> str:
    return "n/a" if d["n"] == 0 else f"{d['expected']}/{d['wrong']}"


def validate_2s(run: Run) -> tuple[pd.DataFrame, list[str]]:
    rows, problems = [], []
    dom_rids = {k: tuple(r for r in run.rids if run.domain[r] == v) for k, v in TABLE1_DOMAINS.items()}
    for m in METHODS:
        pub = TABLE1[m]
        nd = null_discoveries(run, m, tuple(run.rids))
        n_distinct = f"{nd['n_distinct_min']}-{nd['n_distinct']}" if nd["n_distinct_min"] != nd["n_distinct"] else str(nd["n_distinct"])
        got = [f"{nd['null_distinct_mean']:.2f}", n_distinct] + [pair(admissions(run, m, dom_rids[k])) for k in TABLE1_DOMAINS]
        got.append(false_admissions(run, m, tuple(run.rids))["rt_fa"])
        # cross-checks against the run's own summary files
        L = run.leader[run.leader.method == m].iloc[0]
        N = run.nulls[(run.nulls.method == m) & run.nulls.arm.isin(clb.CROSS_ARMS)]
        chk = dict(leader_cal=f"{int(L.cal_correct)}/{int(L.cal_wrong)}", leader_rt=int(L.null_random_fa),
                   nullcsv_distinct_mean=f"{N.n_distinct_sig.mean():.2f}")
        tot = admissions(run, m, tuple(run.rids))
        if chk["leader_cal"] != f"{tot['expected']}/{tot['wrong']}" or chk["leader_rt"] != got[-1] \
                or chk["nullcsv_distinct_mean"] != got[0]:
            problems.append(f"{m}: summary-file cross-check failed {chk} vs {tot['expected']}/{tot['wrong']}, {got[-1]}, {got[0]}")
        ok = all(str(a) == str(b) for a, b in zip(pub, got))
        if not ok:
            problems.append(f"{m}: Table 1 {pub} vs recomputed {tuple(got)}")
        rows.append(dict(estimator=NAME[m], method=m, **{f"pub_{k}": v for k, v in zip(
            ["null", "distinct", "gas", "drug", "vol", "policy", "rt"], pub)}, **{f"got_{k}": v for k, v in zip(
            ["null", "distinct", "gas", "drug", "vol", "policy", "rt"], got)}, match=ok))
    return pd.DataFrame(rows), problems


def validate_30s(run: Run) -> tuple[pd.DataFrame, list[str]]:
    rows, problems = [], []
    dom_rids = {k: tuple(r for r in run.rids if run.domain[r] == v) for k, v in TABLE30_DOMAINS.items()}
    for m in METHODS:
        pub = TABLE30[m]
        nd = null_discoveries(run, m, tuple(run.rids))
        n_distinct = f"{nd['n_distinct_min']}-{nd['n_distinct']}" if nd["n_distinct_min"] != nd["n_distinct"] else str(nd["n_distinct"])
        fa = false_admissions(run, m, tuple(run.rids))
        ho = fa["heldout_fa_mean"]
        got = [f"{nd['null_distinct_mean']:.2f}", n_distinct] + [pair(admissions(run, m, dom_rids[k])) for k in TABLE30_DOMAINS]
        got += [f"{ho:g}", fa["rt_fa"]]
        ok = all(str(a) == str(b) for a, b in zip(pub, got))
        if not ok:
            problems.append(f"{m}: tab:bench-30s {pub} vs recomputed {tuple(got)}")
        rows.append(dict(estimator=NAME[m], method=m, **{f"pub_{k}": v for k, v in zip(
            ["null", "distinct", "gas", "drug", "policy", "heldout", "rt"], pub)}, **{f"got_{k}": v for k, v in zip(
            ["null", "distinct", "gas", "drug", "policy", "heldout", "rt"], got)}, match=ok))
    return pd.DataFrame(rows), problems


# ── Headline (tier A) and call matrix ──────────────────────────────────────────────────────────────────
def headline(run: Run) -> pd.DataFrame:
    rows = []
    for rid in ("R01", "R02"):
        if rid not in run.rel:
            continue
        for m in METHODS:
            q = run.real[(run.real.method == m) & (run.real.rid == rid)]
            if len(q) == 0:
                continue
            x = q.iloc[0]
            sign = "physical (-)" if x.est < 0 else "policy (+)"
            rows.append(dict(rid=rid, label=x.label, method=m, estimator=NAME[m], family=family(m), est=x.est,
                             lo=x.lo, hi=x.hi, p_by=x.p_by, stable_frac=x.stable_frac, n_cases=int(x.n_cases),
                             admitted=bool(x.admitted), call=x.call,
                             sign_of_estimate=sign, verdict=(sign if x.admitted else "not admitted")))
    return pd.DataFrame(rows)


def call_matrix(run: Run) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long: one row per (tier, relation, estimator) with sign/call/estimate. Wide: relation x estimator cell
    '+C' / '-C' (admitted, expected sign), '+W' / '-W' (admitted, wrong sign), '0' (not admitted), 'n/a'."""
    long = []
    for t in MAIN_TIERS:
        for rid in run.tiers[t]:
            r = run.rel[rid]
            for m in METHODS:
                q = run.real[(run.real.method == m) & (run.real.rid == rid)]
                if len(q) == 0:
                    long.append(dict(tier=t, rid=rid, label=r.label, expected=r.expected, method=m, cell="n/a",
                                     sign="", call="n/a", est=np.nan, lo=np.nan, hi=np.nan))
                    continue
                x = q.iloc[0]
                sgn = "+" if x.est > 0 else "-"
                if x.call == "correct":
                    cell = f"{sgn}C"
                elif x.call == "wrong":
                    cell = f"{sgn}W"
                else:
                    cell = "0"
                long.append(dict(tier=t, rid=rid, label=r.label, expected=r.expected, method=m, cell=cell, sign=sgn,
                                 call=x.call, est=x.est, lo=x.lo, hi=x.hi))
    L = pd.DataFrame(long)
    W = L.pivot(index=["tier", "rid"], columns="method", values="cell").reindex(columns=list(METHODS))
    W = W.loc[[(t, r) for t in MAIN_TIERS for r in run.tiers[t]]]
    W.insert(0, "expected", [("+" if run.rel[r].expected > 0 else "-") for _, r in W.index])
    W.insert(1, "label", [run.rel[r].label for _, r in W.index])
    W["n_correct"] = (L[L.call == "correct"].groupby(["tier", "rid"]).size()).reindex(W.index).fillna(0).astype(int)
    W["n_wrong"] = (L[L.call == "wrong"].groupby(["tier", "rid"]).size()).reindex(W.index).fillna(0).astype(int)
    return L, W.reset_index()


# ── Formatting ─────────────────────────────────────────────────────────────────────────────────────────
def f2(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def fi(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else str(int(x))


def md_table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def tier_md(T: pd.DataFrame, tier: str, n_draws: int) -> str:
    S = T[T.tier == tier]
    n = int(S.n_tier.iloc[0])
    rows = []
    for _, x in S.iterrows():
        null = "n/a" if not np.isfinite(x.null_distinct_mean) else \
            f"{x.null_distinct_mean:.2f} [{x.null_distinct_min}-{x.null_distinct_max}] / {x.n_distinct}"
        rows.append([x.estimator, x.family, fi(x.n), fi(x.expected), fi(x.wrong), fi(x.not_admitted),
                     f"{fi(x.expected_distinct)}/{fi(x.wrong_distinct)}" if x.n else "n/a", null,
                     f2(x.heldout_fa_mean), fi(x.rt_fa), x.admitted_ids or "-"])
    hdr = ["Estimator", "Family", "n", "Expected", "Wrong", "Not adm.", "Distinct exp/wrong",
           f"Null uncal. (mean [min-max] of {n_draws} draws / distinct)", "Held-out FA", "RT FA", "Admitted (sign)"]
    return f"**Tier {tier} ({TIER_NAME[tier]}, {n} relations)**\n\n" + md_table(hdr, rows)


def ranking_md(S: pd.DataFrame, tier: str) -> str:
    rows = [[x.rank_rule1, x.rank_rule2, x.estimator, x.family, fi(x.expected), fi(x.wrong), fi(x.rt_fa),
             fi(x.errors), fi(x.score2)] for _, x in S.iterrows()]
    return md_table(["Rule 1", "Rule 2", "Estimator", "Family", "Exp.", "Wrong", "RT FA", "Wrong+RT", "Score"], rows)


def latex_tier_table(T: pd.DataFrame, run: Run, label: str) -> str:
    tiers = [t for t in ("A", "B", "B1", "B2", "C") if run.tiers[t]]
    cols = {"A": ["ew", "null", "rt"], "B": ["ew", "null", "rt"], "B1": ["ew"], "B2": ["ew"], "C": ["ew", "null", "rt"]}
    spec = "@{}l" + "".join("c" * len(cols[t]) for t in tiers) + "@{}"
    head1, head2, cmid, col = [""], ["Estimator"], [], 2
    for t in tiers:
        k = len(cols[t])
        head1.append(f"\\multicolumn{{{k}}}{{c}}{{{t} ({len(run.tiers[t])})}}")
        cmid.append(f"\\cmidrule(lr){{{col}-{col + k - 1}}}")
        col += k
        for c in cols[t]:
            head2.append({"ew": "E/W", "null": "Null", "rt": "RT"}[c])
    n_rel = len(run.rids)
    res_label = run.key.replace("s", "\\,s")
    lines = [
        "\\begin{table}[h]",
        f"\\caption{{ClosedLoopBench by evidence tier ({res_label} resolution, {n_rel} relations). "
        "Tiers: A, respiratory physics; B, pharmacology/physiology (B1 pharmacology; B2 baroreflex and PEEP); "
        "C, documented practice (policies). E/W: calibrated admissions with the expected/wrong sign "
        "(counted over relations); Null: uncalibrated significant distinct estimates per transplant draw "
        f"(mean of {run.n_draws}) / distinct estimates among the tier's relations; RT: random-time false admissions "
        "in the tier. Top: lag-based; bottom: event designs. PCMCI+ runs on 30\\,s block means of 60 cases.}",
        f"\\label{{{label}}}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
        " & ".join(head1) + " \\\\", "".join(cmid), " & ".join(head2) + " \\\\", "\\midrule"]
    for m in METHODS:
        if m == "naive":
            lines.append("\\midrule")
        cells = [NAME[m]]
        for t in tiers:
            x = T[(T.tier == t) & (T.method == m)].iloc[0]
            for c in cols[t]:
                if c == "ew":
                    cells.append("n/a" if x.n == 0 else f"{int(x.expected)}/{int(x.wrong)}")
                elif c == "null":
                    cells.append("n/a" if not np.isfinite(x.null_distinct_mean) else f"{x.null_distinct_mean:.2f}/{int(x.n_distinct)}")
                else:
                    cells.append("n/a" if not np.isfinite(x.rt_fa) else str(int(x.rt_fa)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def latex_rank_table(ranks: dict[str, pd.DataFrame], run: Run, label: str) -> str:
    tiers = [t for t in MAIN_TIERS if run.tiers[t]]
    spec = "@{}l" + "cc" * len(tiers) + "@{}"
    head1, cmid, col = [""], [], 2
    for t in tiers:
        head1.append(f"\\multicolumn{{2}}{{c}}{{Tier {t} ({len(run.tiers[t])})}}")
        cmid.append(f"\\cmidrule(lr){{{col}-{col + 1}}}")
        col += 2
    res_label = run.key.replace("s", "\\,s")
    lines = ["\\begin{table}[h]",
             f"\\caption{{Within-tier ranks ({res_label}). Rule 1: wrong-sign plus random-time false "
             "admissions ascending, then expected-sign admissions descending. Rule 2: expected $-$ 2$\\times$wrong $-$ "
             "random-time false admissions, descending. Ties share the lower rank. In bold: the three design-based estimators.}",
             f"\\label{{{label}}}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(head1) + " \\\\", "".join(cmid),
             "Estimator" + " & Rule 1 & Rule 2" * len(tiers) + " \\\\", "\\midrule"]
    for m in METHODS:
        if m == "naive":
            lines.append("\\midrule")
        name = f"\\textbf{{{NAME[m]}}}" if m in DESIGN else NAME[m]
        cells = [name]
        for t in tiers:
            S = ranks[t]
            q = S[S.method == m]
            if len(q) == 0:
                cells += ["n/a", "n/a"]
            else:
                cells += [str(int(q.rank_rule1.iloc[0])), str(int(q.rank_rule2.iloc[0]))]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def latex_call_matrix(W: pd.DataFrame, run: Run, label: str) -> str:
    spec = "@{}ll" + "c" * len(METHODS) + "|cc@{}"
    res_label = run.key.replace("s", "\\,s")
    lines = ["\\begin{table}[h]",
             f"\\caption{{Calibrated calls by tier ({res_label}). $+$C/$-$C: admitted with the expected "
             "sign (sign of the calibrated estimate); $+$W/$-$W: admitted with the wrong sign; 0: not admitted. "
             "Estimator codes as in Table~\\ref{tab:bench-calls}. Last columns: estimators admitting with the expected/wrong sign.}",
             f"\\label{{{label}}}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             "Tier & ID & " + " & ".join(SHORT[m] for m in METHODS) + " & R & X \\\\", "\\midrule"]
    last = None
    for _, x in W.iterrows():
        if last is not None and x.tier != last:
            lines.append("\\midrule")
        last = x.tier
        cells = [x.tier, x.rid]
        for m in METHODS:
            c = x[m]
            c = "n/a" if (not isinstance(c, str)) else c
            c = c.replace("+C", "$+$C").replace("-C", "$-$C").replace("+W", "\\textbf{$+$W}").replace("-W", "\\textbf{$-$W}")
            cells.append(c)
        cells += [str(x.n_correct), str(x.n_wrong)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ── Narrative helpers ──────────────────────────────────────────────────────────────────────────────────
def names(ms: list[str]) -> str:
    return ", ".join(NAME[m] for m in ms) if ms else "none"


def mechanical_verdict(S: pd.DataFrame, tier: str, run: Run) -> str:
    """Rule-derived statement (no judgement): which lag-based estimators rank at or above the lowest-ranked
    design-based estimator under each rule."""
    v1, v2 = design_vs_lag(S, "rank_rule1"), design_vs_lag(S, "rank_rule2")
    des = S[S.family == "design-based"]
    des_txt = "; ".join(f"{NAME[m]} {int(e)}/{int(w)}/{int(r)}" for m, e, w, r in zip(des.method, des.expected, des.wrong, des.rt_fa))
    s = f"Tier {tier} ({len(run.tiers[tier])} relations; design-based expected/wrong/RT: {des_txt}). "
    s += ("Rule 1: no lag-based estimator at or above the lowest-ranked design-based estimator."
          if v1["clean"] else f"Rule 1: at or above the lowest-ranked design-based estimator: {names(v1['lag_at_or_above'])}"
          + (f" (above all three: {names(v1['lag_strictly_above_all'])})." if v1["lag_strictly_above_all"] else "."))
    s += (" Rule 2: none." if v2["clean"] else f" Rule 2: {names(v2['lag_at_or_above'])}"
          + (f" (above all three: {names(v2['lag_strictly_above_all'])})." if v2["lag_strictly_above_all"] else "."))
    return s


# Hand-written one-sentence verdicts (written after the tables were computed; every number in them is asserted
# against the computed tables in ``assert_facts`` so that stale prose cannot be regenerated silently).
VERDICTS: dict[tuple[str, str], str] = {
    ("2s", "A"): "Yes for the matched and interrupted-trend designs, which rank first under both rules (4/0 expected/wrong, "
                 "0 random-time false admissions, 1.00 and 1.50 of 5 estimates significant per transplant draw); the "
                 "sequential-trial contrast is also 4/0 but has 2 random-time false admissions in the tier (R02, R15), so "
                 "PCMCI+ (0 admissions) and VARX (4/1, 0 RT) rank above it under Rule 1; the one lag-based exception is "
                 "VAR-LiNGAM, which admits all 5 with the expected sign but has 3 random-time false admissions (R01, R02, "
                 "R15) and declares all 3 of its distinct tier-A estimates significant on every transplant draw (Rule 1 "
                 "rank 8, Rule 2 rank 4, tied with VARX, seqtrial and the naive contrast).",
    ("2s", "B"): "Yes under Rule 1: kink 6/0, matched 5/0 and seqtrial 5/0 with 0 random-time false admissions rank 1-3 "
                 "ahead of every lag-based estimator (PCMCI+ 4th with 1/0); under Rule 2 VAR-LiNGAM (10/0 but 4 RT false "
                 "admissions) ties kink at rank 1 and the direction-aware rule (8/0, 3 RT) ties matched and seqtrial at "
                 "rank 3; note that seqtrial's uncalibrated null discoveries in this tier (5.62 of 14 per draw) are closer "
                 "to the lag-based estimators than to matched (0.75) and kink (1.62).",
    ("2s", "C"): "No: PCMCI+ ranks first under Rule 1 with 2/0 and 0 RT (zero errors by admitting almost nothing); "
                 "matched and kink (5/2, 0 RT) rank 2 and seqtrial (7/1, 2 RT) rank 4, ahead of every other lag-based "
                 "estimator (VAR-LiNGAM 7/1 with 5 RT, Granger 8/2 with 5 RT, VARX 6/2 with 4 RT); under Rule 2 seqtrial "
                 "ties the naive event contrast at rank 1 and PCMCI+ (rank 3) is above matched and kink (rank 4); the "
                 "wrong-sign admissions are P08 and P10 for every family.",
    ("30s", "A"): "Tie, not outrank (2 relations): Granger, PCMCI+, VAR-LiNGAM and VARX admit both set-rate relations "
                  "with the physical sign with 0 RT false admissions and share rank 1 with the design-based three and the "
                  "naive level contrast; only the three cross-correlation rules admit both with the policy sign (the paper "
                  "notes that PCMCI+'s set-rate admissions here come from the transplant offset).",
    ("30s", "B"): "No: PCMCI+ (1,200 cases at 30 s) ranks first with 6/0 and 0 RT; the design-based estimators admit "
                  "1/0 (matched, rank 2), 0/1 (kink, R06 wrong-signed, rank 4) and 0/2 (seqtrial, R05 and R06 wrong-signed, "
                  "rank 8), so two of the three are outranked by the signed-peak rule, Granger, VAR-LiNGAM and VARX under "
                  "Rule 1.",
    ("30s", "C"): "Partly: PCMCI+ ranks first (5/0, 0 RT); the design-based three rank 2-3 (seqtrial 4/0, matched and "
                  "kink 3/0, all 0 RT) ahead of every other lag-based estimator under Rule 1; under Rule 2 the naive event "
                  "contrast ties PCMCI+ at rank 1 and VAR-LiNGAM (6/0, 2 RT) ties seqtrial at rank 3.",
}


def assert_facts(res: dict[str, dict]) -> None:
    """Every number quoted in VERDICTS and in the summary, checked against the computed tables."""
    def g(key, tier, m, col):
        T = res[key]["T"]
        return T[(T.tier == tier) & (T.method == m)][col].iloc[0]

    def rk(key, tier, m, rule):
        S = res[key]["ranks"][tier]
        return int(S[S.method == m][f"rank_rule{rule}"].iloc[0])

    def ewr(key, tier, m):
        return tuple(int(g(key, tier, m, c)) for c in ("expected", "wrong", "rt_fa"))

    chk = [
        ewr("2s", "A", "matched") == (4, 0, 0), ewr("2s", "A", "kink") == (4, 0, 0), ewr("2s", "A", "seqtrial") == (4, 0, 2),
        g("2s", "A", "seqtrial", "rt_fa_ids") == "R02-;R15-", ewr("2s", "A", "varlingam") == (5, 0, 3),
        g("2s", "A", "varlingam", "rt_fa_ids") == "R01+;R02+;R15-", g("2s", "A", "varlingam", "null_distinct_mean") == 3.0,
        g("2s", "A", "varlingam", "n_distinct") == 3, ewr("2s", "A", "pcmciplus") == (0, 0, 0), ewr("2s", "A", "varx") == (4, 1, 0),
        ewr("2s", "A", "granger") == (2, 3, 3), f"{g('2s', 'A', 'matched', 'null_distinct_mean'):.2f}" == "1.00",
        f"{g('2s', 'A', 'kink', 'null_distinct_mean'):.2f}" == "1.50",
        rk("2s", "A", "matched", 1) == 1, rk("2s", "A", "kink", 1) == 1, rk("2s", "A", "matched", 2) == 1, rk("2s", "A", "kink", 2) == 1,
        rk("2s", "A", "seqtrial", 1) == 6, rk("2s", "A", "pcmciplus", 1) == 4, rk("2s", "A", "varx", 1) == 5,
        rk("2s", "A", "varlingam", 1) == 8, rk("2s", "A", "varlingam", 2) == 4, rk("2s", "A", "seqtrial", 2) == 4,
        rk("2s", "A", "varx", 2) == 4, rk("2s", "A", "naive", 2) == 4,
        all(rk("2s", "A", m, 1) >= 9 for m in ("xcorr_pos", "xcorr_v1", "xcorr_dir", "granger")),
        ewr("2s", "B", "kink") == (6, 0, 0), ewr("2s", "B", "matched") == (5, 0, 0), ewr("2s", "B", "seqtrial") == (5, 0, 0),
        rk("2s", "B", "kink", 1) == 1, rk("2s", "B", "matched", 1) == 2, rk("2s", "B", "seqtrial", 1) == 2, rk("2s", "B", "pcmciplus", 1) == 4,
        ewr("2s", "B", "pcmciplus") == (1, 0, 0), ewr("2s", "B", "varlingam") == (10, 0, 4), ewr("2s", "B", "granger") == (10, 0, 8),
        ewr("2s", "B", "xcorr_dir") == (8, 0, 3), rk("2s", "B", "varlingam", 2) == 1, rk("2s", "B", "kink", 2) == 1,
        rk("2s", "B", "xcorr_dir", 2) == 3, rk("2s", "B", "matched", 2) == 3, rk("2s", "B", "seqtrial", 2) == 3,
        f"{g('2s', 'B', 'seqtrial', 'null_distinct_mean'):.2f}" == "5.62", f"{g('2s', 'B', 'matched', 'null_distinct_mean'):.2f}" == "0.75",
        f"{g('2s', 'B', 'kink', 'null_distinct_mean'):.2f}" == "1.62",
        ewr("2s", "C", "pcmciplus") == (2, 0, 0), rk("2s", "C", "pcmciplus", 1) == 1, ewr("2s", "C", "matched") == (5, 2, 0),
        ewr("2s", "C", "kink") == (5, 2, 0), ewr("2s", "C", "seqtrial") == (7, 1, 2), rk("2s", "C", "matched", 1) == 2,
        rk("2s", "C", "kink", 1) == 2, rk("2s", "C", "seqtrial", 1) == 4, ewr("2s", "C", "varlingam") == (7, 1, 5),
        ewr("2s", "C", "granger") == (8, 2, 5), ewr("2s", "C", "varx") == (6, 2, 4), rk("2s", "C", "seqtrial", 2) == 1,
        rk("2s", "C", "naive", 2) == 1, rk("2s", "C", "pcmciplus", 2) == 3, rk("2s", "C", "matched", 2) == 4, rk("2s", "C", "kink", 2) == 4,
        all(rk("30s", "A", m, 1) == 1 for m in ("granger", "pcmciplus", "varlingam", "varx", "matched", "kink", "seqtrial", "naive_lvl")),
        all(ewr("30s", "A", m) == (0, 2, 0) for m in ("xcorr_pos", "xcorr_v1", "xcorr_dir")),
        ewr("30s", "B", "pcmciplus") == (6, 0, 0), rk("30s", "B", "pcmciplus", 1) == 1, ewr("30s", "B", "matched") == (1, 0, 0),
        ewr("30s", "B", "kink") == (0, 1, 0), ewr("30s", "B", "seqtrial") == (0, 2, 0), rk("30s", "B", "matched", 1) == 2,
        rk("30s", "B", "kink", 1) == 4, rk("30s", "B", "seqtrial", 1) == 8, g("30s", "B", "kink", "admitted_ids") == "R06-",
        g("30s", "B", "seqtrial", "admitted_ids") == "R05-;R06-",
        ewr("30s", "C", "pcmciplus") == (5, 0, 0), rk("30s", "C", "pcmciplus", 1) == 1, ewr("30s", "C", "seqtrial") == (4, 0, 0),
        ewr("30s", "C", "matched") == (3, 0, 0), ewr("30s", "C", "kink") == (3, 0, 0), rk("30s", "C", "seqtrial", 1) == 2,
        rk("30s", "C", "matched", 1) == 3, rk("30s", "C", "kink", 1) == 3, ewr("30s", "C", "varlingam") == (6, 0, 2),
        rk("30s", "C", "naive", 2) == 1, rk("30s", "C", "pcmciplus", 2) == 1, rk("30s", "C", "varlingam", 2) == 3, rk("30s", "C", "seqtrial", 2) == 3,
    ]
    bad = [i for i, ok in enumerate(chk) if not ok]
    assert not bad, f"facts quoted in the verdicts/summary no longer hold (check indices {bad})"


def check_totals(run: Run, T: pd.DataFrame) -> None:
    """Tier counts must add up to the published per-estimator totals (leaderboard.csv)."""
    for m in METHODS:
        L = run.leader[run.leader.method == m].iloc[0]
        S = T[(T.tier.isin(MAIN_TIERS)) & (T.method == m)]
        tot = (int(S.expected.fillna(0).sum()), int(S.wrong.fillna(0).sum()), int(S.rt_fa.fillna(0).sum()),
               float(S.heldout_fa_mean.fillna(0).sum()))
        pub = (int(L.cal_correct), int(L.cal_wrong), int(L.null_random_fa), float(L.null_cross_fa_per_draw))
        assert tot[:3] == pub[:3] and abs(tot[3] - pub[3]) < 1e-9, f"{run.key} {m}: tiers {tot} != leaderboard {pub}"


# ── Main ───────────────────────────────────────────────────────────────────────────────────────────────
def analyse(key: str) -> dict:
    t0 = time.time()
    run = Run(key)
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    val, problems = (validate_2s if key == "2s" else validate_30s)(run)
    val.to_csv(out / "validation.csv", index=False)
    T = tier_table(run)
    check_totals(run, T)
    T.to_csv(out / "tier_table.csv", index=False)
    ranks = {t: rankings(T, t) for t in ("A", "B", "B1", "B2", "C") if run.tiers[t]}
    pd.concat(ranks.values()).to_csv(out / "rankings.csv", index=False)
    H = headline(run)
    H.to_csv(out / "tierA_headline_R01_R02.csv", index=False)
    L, W = call_matrix(run)
    L.to_csv(out / "call_matrix_long.csv", index=False)
    W.to_csv(out / "call_matrix_wide.csv", index=False)
    tex = (latex_tier_table(T, run, f"tab:tiers-{key}") + "\n\n" + latex_rank_table(ranks, run, f"tab:tiers-rank-{key}")
           + "\n\n" + latex_call_matrix(W, run, f"tab:tiers-calls-{key}"))
    (out / "tier_tables.tex").write_text(tex + "\n")
    md = [tier_md(T, t, run.n_draws) for t in ("A", "B", "B1", "B2", "C") if run.tiers[t]]
    (out / "tier_table.md").write_text("\n\n".join(md) + "\n")
    verdicts = {t: VERDICTS[(key, t)] for t in MAIN_TIERS if t in ranks}
    mech = {t: mechanical_verdict(ranks[t], t, run) for t in MAIN_TIERS if t in ranks}
    return dict(run=run, val=val, problems=problems, T=T, ranks=ranks, H=H, L=L, W=W, tex=tex, md=md,
                verdicts=verdicts, mech=mech, seconds=time.time() - t0)


def call_matrix_md(W: pd.DataFrame) -> str:
    hdr = ["Tier", "ID", "Relation", "Exp."] + [SHORT[m] for m in METHODS] + ["R", "X"]
    rows = []
    for _, x in W.iterrows():
        rows.append([x.tier, x.rid, x.label, x.expected] + [x[m] if isinstance(x[m], str) else "n/a" for m in METHODS]
                    + [x.n_correct, x.n_wrong])
    return md_table(hdr, rows)


def headline_md(H: pd.DataFrame) -> str:
    rows = [[x.rid, x.estimator, x.family, f"{x.est:.4g} [{x.lo:.4g}, {x.hi:.4g}]", f"{x.p_by:.2g}", f"{x.stable_frac:.2f}",
             x.n_cases, "yes" if x.admitted else "no", x.verdict] for _, x in H.iterrows()]
    return md_table(["Relation", "Estimator", "Family", "Calibrated est. [95% CI]", "p (BY)", "Stability", "Cases",
                     "Admitted", "Sign verdict"], rows)


def validation_md(val: pd.DataFrame, key: str) -> str:
    cols = [c for c in val.columns if c.startswith("pub_")]
    keys = [c[4:] for c in cols]
    hdr = ["Estimator"] + [f"{k} (paper)" for k in keys] + [f"{k} (recomputed)" for k in keys] + ["match"]
    rows = [[x.estimator] + [x[f"pub_{k}"] for k in keys] + [x[f"got_{k}"] for k in keys] + ["yes" if x.match else "NO"]
            for _, x in val.iterrows()]
    return md_table(hdr, rows)


def write_note(res: dict[str, dict]) -> None:
    r2, r30 = res["2s"], res["30s"]
    T2, R2, run2 = r2["T"], r2["ranks"], r2["run"]
    H2 = r2["H"]

    def cell(t, m, col):
        return T2[(T2.tier == t) & (T2.method == m)][col].iloc[0]

    # numbers used in the summary
    a = {m: (int(cell("A", m, "expected")), int(cell("A", m, "wrong")), int(cell("A", m, "rt_fa"))) for m in METHODS}
    b = {m: (int(cell("B", m, "expected")), int(cell("B", m, "wrong")), int(cell("B", m, "rt_fa"))) for m in METHODS}
    c = {m: (int(cell("C", m, "expected")), int(cell("C", m, "wrong")), int(cell("C", m, "rt_fa"))) for m in METHODS
         if cell("C", m, "n") > 0}
    nullA = {m: cell("A", m, "null_distinct_mean") for m in METHODS}
    nullB = {m: cell("B", m, "null_distinct_mean") for m in METHODS}
    nullC = {m: cell("C", m, "null_distinct_mean") for m in METHODS}
    physA = H2[(H2.rid == "R01") & H2.admitted & (H2.est < 0)].method.tolist()
    polA = H2[(H2.rid == "R01") & H2.admitted & (H2.est > 0)].method.tolist()
    noneA = H2[(H2.rid == "R01") & ~H2.admitted].method.tolist()
    r1A = R2["A"]
    top_A = r1A[r1A.rank_rule1 == 1].method.tolist()

    def ew(d, m):
        return f"{d[m][0]}/{d[m][1]}"

    def rt(d, m):
        return d[m][2]

    r01 = {m: H2[(H2.rid == "R01") & (H2.method == m)].iloc[0] for m in METHODS}
    xc = f"{r01['xcorr_pos'].est:+.3f} [{r01['xcorr_pos'].lo:.3f}, {r01['xcorr_pos'].hi:.3f}]"
    gr = f"{r01['granger'].est:+.4f} [{r01['granger'].lo:.4f}, {r01['granger'].hi:.4f}]"
    ma = f"{r01['matched'].est:.2f} [{r01['matched'].lo:.2f}, {r01['matched'].hi:.2f}]"

    summary_paragraph = f"""We report every estimator separately for three evidence tiers fixed in advance (computed from the stored benchmark outputs; nothing re-run): A, respiratory physics (R01, R02, R13-R15; 5 relations); B, pharmacology/physiology (R03-R12, R16-R19; 14); C, documented practice (P01-P10; 10). Counts are expected/wrong-sign admissions and random-time (RT) false admissions within the tier.

Tier A: the matched and interrupted-trend designs admit {a['matched'][0]}/5 with the physical sign, 0 wrong, 0 RT, with {nullA['matched']:.2f} and {nullA['kink']:.2f} of 5 null discoveries per transplant draw, and rank first under both rules (matched R01: {ma} mmHg). The sequential-trial contrast is also {ew(a, 'seqtrial')} but has {rt(a, 'seqtrial')} RT false admissions. The three cross-correlation rules and Granger admit set-rate -> EtCO2 with the policy sign (R01: {xc}; {gr}) and rank 9-12. VAR-LiNGAM is the exception: {ew(a, 'varlingam')}, but {rt(a, 'varlingam')} RT false admissions and all 3 of its distinct tier-A estimates significant on every transplant draw. PCMCI+ admits nothing.

Tier B: under the reliability-first rule the design-based three ({ew(b, 'kink')}, {ew(b, 'matched')}, {ew(b, 'seqtrial')}; 0 RT) rank 1-3 ahead of every lag-based estimator; Granger and VAR-LiNGAM admit {ew(b, 'granger')} but with {rt(b, 'granger')} and {rt(b, 'varlingam')} RT false admissions, so VAR-LiNGAM ties the interrupted-trend design under the score rule.

Tier C: PCMCI+ ranks first ({ew(c, 'pcmciplus')}, 0 RT); the design-based three follow ({ew(c, 'matched')}, {ew(c, 'kink')}, {ew(c, 'seqtrial')}); the wrong signs (P08, P10) are shared by every family.

The conclusion therefore holds on tier A for the matched and interrupted-trend designs and on tier B under the reliability-first rule, with the sequential-trial and tier-C qualifications above."""
    n_words = len(summary_paragraph.split())
    assert n_words <= 250, f"summary paragraph has {n_words} words (> 250)"

    appendix_par = f"""\\paragraph{{Results by evidence tier.}}
Table~\\ref{{tab:tiers-2s}} re-partitions the 2\\,s results by the strength of the sign evidence (App.~\\ref{{app:benchmark-relations}}): tier A, respiratory physics (R01, R02, R13--R15); tier B, pharmacology and physiology (R03--R12, R16--R19; B1 pharmacology, B2 the phenylephrine baroreflex and PEEP); tier C, documented practice (P01--P10). The tiers were fixed before the partition was computed, no estimator was re-run, and the Benjamini--Yekutieli family remains each estimator's full set of relations. On tier A the matched and interrupted-trend designs admit {a['matched'][0]} of 5 relations with the physical sign and none with the wrong sign, have no random-time false admission among the 5, and declare {nullA['matched']:.2f} and {nullA['kink']:.2f} of 5 estimates significant per transplant draw; the sequential-trial contrast admits the same {a['seqtrial'][0]} relations but has {rt(a, 'seqtrial')} random-time false admissions in the tier (R02, R15). The three cross-correlation rules and Granger admit the set-rate relations with the policy sign ({ew(a, 'xcorr_pos')}, {ew(a, 'xcorr_v1')}, {ew(a, 'xcorr_dir')} and {ew(a, 'granger')} expected/wrong; {rt(a, 'xcorr_pos')}, {rt(a, 'xcorr_v1')}, {rt(a, 'xcorr_dir')} and {rt(a, 'granger')} random-time false admissions); VAR-LiNGAM admits all 5 with the expected sign but has {rt(a, 'varlingam')} random-time false admissions (R01, R02, R15) and declares all 3 of its distinct tier-A estimates significant on every transplant draw; PCMCI+ admits none. Ranking the estimators within the tier by wrong-sign plus random-time false admissions and then by expected-sign admissions (Table~\\ref{{tab:tiers-rank-2s}}) places the matched and interrupted-trend designs first, the sequential-trial contrast sixth and VAR-LiNGAM eighth; by expected $-$ 2$\\times$wrong $-$ random-time admissions the first two are unchanged and VAR-LiNGAM, VARX, the sequential-trial and the naive event contrast tie at rank 4. On tier B the design-based estimators admit {b['matched'][0]}--{b['kink'][0]} of 14 relations with the expected sign, none with the wrong sign and none under random-time input, and rank 1--3 under the first rule; Granger and VAR-LiNGAM admit {b['granger'][0]} of 14 with the expected sign but have {rt(b, 'granger')} and {rt(b, 'varlingam')} random-time false admissions in the tier, and under the second rule VAR-LiNGAM ties the interrupted-trend design at rank 1. On tier C PCMCI+ ranks first with {ew(c, 'pcmciplus')} admissions; the design-based estimators admit {c['matched'][0]}--{c['seqtrial'][0]} of 10 policies with the expected sign and {c['seqtrial'][1]}--{c['matched'][1]} with the opposite sign (P08, P10), as do most lag-based estimators. The statement that the design-based estimators are more reliable than the lag-based estimators therefore holds on the physics tier for the matched and interrupted-trend designs and on the pharmacology tier under the reliability-first rule; it is not a statement about admission counts on the policy tier."""

    v2 = r2["val"]
    v30 = r30["val"]
    lines = [
        "# Extended analysis B: ClosedLoopBench results by evidence tier", "",
        f"Generated by `scripts/extended/B_tiers.py` on {time.strftime('%Y-%m-%d %H:%M')} "
        f"(pandas re-partition of published outputs; 1 core; {r2['seconds'] + r30['seconds']:.1f} s).", "",
        "## Summary", "",
        "- Table 1 (2 s) and App. tab:bench-30s (30 s) are reproduced exactly from the run files for all 12 estimators "
        "(Section 1), and the tier counts add up to the published per-estimator totals.",
        f"- Tier A (2 s): {r2['verdicts']['A']}",
        f"- Tier B (2 s): {r2['verdicts']['B']}",
        f"- Tier C (2 s): {r2['verdicts']['C']}",
        f"- Headline (tier A, R01/R02): admitted with the physical sign by {names(physA)}; with the policy sign by "
        f"{names(polA)}; not admitted by {names(noneA)} (Section 4).",
        "- 30 s (secondary, 18 relations): on tier A (R01, R02 only) Granger, PCMCI+, VAR-LiNGAM and VARX tie the "
        "design-based three at rank 1 and only the cross-correlation rules admit the policy sign; on tiers B and C "
        "PCMCI+ (1,200 cases at 30 s) ranks first and the design-based estimators have 0-2 wrong-sign admissions on tier B "
        "(Section 6).", "",
        "## What was run", "",
        "```", "PYTHONPATH=src .venv/bin/python scripts/extended/B_tiers.py", "```", "",
        "No estimator was re-run and no seed is involved. The script re-reads `runs/iclr/closed_loop_v4/2s_all/` "
        "(`calibrated.csv`, `pooled.csv`, `null_discovery.csv`, `leaderboard.csv`, `meta.json`) and, as the secondary "
        "resolution, `runs/iclr/closed_loop_v3/30s_all/` (the paper's App. tab:bench-30s; there is no 30 s run under "
        "`closed_loop_v4/`, and `closed_loop_v3/2s_fine` is the 500-case pilot, not used). It imports "
        "`clbench.eval.closed_loop_bench` for the relation table (`use_table`), the primary horizon of every "
        "(estimator, relation) and the transplant arm names; the metrics re-implement `null_discovery_table` "
        "(distinct estimates = identical rounded est/lo/hi), `leaderboard` (admissions, held-out and random-time false "
        "admissions) and the domain counts restricted to a tier. Outputs: `runs/extended/B_tiers/{2s,30s}/`.", "",
        "### Column definitions", "",
        "- *Expected / Wrong*: calibrated admissions (v3 rule: calibrated CI excludes 0, BY-adjusted p <= 0.05 over the "
        "estimator's family of all its relations, stability >= 0.75, >= 5 cases) on real data with the expected / wrong "
        "sign, counted over relations as in Table 1. The BY family is not re-formed per tier: the tier tables re-partition "
        "the published calls. *Distinct exp/wrong* counts the same admissions over distinct estimates (Granger and "
        "VAR-LiNGAM share one estimate for the up/down relations of a pair; the positive- and signed-peak rules share one "
        "estimate for the response and policy relation of a pair, which lie in different tiers).",
        "- *Null uncal.*: on each of the 8 transplant draws, the tier's relations whose single-arm case-bootstrap CI "
        "excludes 0, counted over distinct estimates within the tier; mean [min-max] over draws / number of distinct "
        "estimates in the tier. About 5% of the distinct estimates would be expected at the nominal level.",
        "- *Held-out FA*: each transplant draw in turn as input, calibrated against the other 7; mean admissions per draw "
        "among the tier's relations. *RT FA*: random-time arm as input, calibrated against the 8 draws; admissions among "
        "the tier's relations.",
        "- Families: lag-based = positive-peak / signed-peak / direction-aware cross-correlation, Granger, PCMCI+, "
        "VAR-LiNGAM, VARX; event designs = naive event contrast, naive level contrast (responses only, so n/a on tier C), "
        "and the design-based three: matched, interrupted trend (kink), sequential trial. The CLBench typed estimator "
        "equals VARX under the admission rule and is omitted.",
        "- PCMCI+ ran on 30 s block means of the same seeded 60 cases in every arm (3 of its 29 relations have 4-20 "
        "cases); its zero admissions are a power statement, not a reliability statement.", "",
        "## 1. Validation: Table 1 of the paper reproduced from the run files (2 s)", "",
        "Published values from `iclr2027/sections/exp_benchmark.tex` (tab:benchmark) against the same quantities "
        "recomputed from `calibrated.csv` (admissions, RT) and `pooled.csv` (null discoveries over distinct estimates at "
        "the primary horizon); the recomputed values were also checked against `leaderboard.csv` and "
        "`null_discovery.csv`. " + ("**All 12 rows match.**" if not r2["problems"] else "**DISCREPANCIES:** " + "; ".join(r2["problems"])), "",
        validation_md(v2, "2s"), "",
        "Secondary resolution, App. tab:bench-30s (v3 30 s run; 18 relations): " +
        ("**all 12 rows match.**" if not r30["problems"] else "**DISCREPANCIES:** " + "; ".join(r30["problems"])), "",
        validation_md(v30, "30s"), "",
        "## 2. Per tier x estimator (2 s, primary)", "",
        "Tier membership (fixed in advance): A = R01, R02, R13, R14, R15; B = R03-R12, R16, R17, R18, R19 "
        "(B1 = R03-R09, R11, R12, R17, R18, R19; B2 = R10, R16); C = P01-P10.", "",
        "\n\n".join(r2["md"]), "",
        "CSV: `runs/extended/B_tiers/2s/tier_table.csv`; LaTeX (booktabs, footnotesize): "
        "`runs/extended/B_tiers/2s/tier_tables.tex` (reproduced in Section 7).", "",
        "## 3. Rankings within each tier (2 s)", "",
        "Rule 1 (reliability first): sort by wrong-sign admissions + random-time false admissions ascending, then "
        "expected-sign admissions descending. Rule 2 (score): expected - 2 x wrong - RT false admissions, descending. "
        "Ties share the lower rank. Both rules were fixed before the tables were computed.", "",
    ]
    for t in ("A", "B", "B1", "B2", "C"):
        if t in R2:
            lines += [f"**Tier {t} ({TIER_NAME[t]}, {len(run2.tiers[t])} relations)**", "", ranking_md(R2[t], t), ""]
    lines += ["**Verdicts (one sentence per tier; do the design-based three outrank the lag-based estimators?):**", ""]
    for t in MAIN_TIERS:
        lines.append(f"- Tier {t} ({TIER_NAME[t]}): {r2['verdicts'][t]}")
    lines += ["", "Rule-derived statements (lag-based estimators ranked at or above the lowest-ranked design-based estimator):", ""]
    for t in MAIN_TIERS:
        lines.append(f"- {r2['mech'][t]}")
    lines += ["", "## 4. Tier-A verdict on the headline: set-rate -> EtCO2 (R01, R02)", "",
              "Calibrated estimates and 95% CIs from `calibrated.csv` (real input, 8 transplant draws). Physical sign: "
              "negative (a higher set rate lowers EtCO2); policy sign: positive (the rate is raised when EtCO2 is high).", "",
              headline_md(H2), "",
              f"R01: admitted with the physical sign by {names(physA)}; with the policy sign by {names(polA)}; "
              f"not admitted by {names(noneA)}.", "",
              "## 5. Per-relation call matrix (2 s)", "",
              "+C / -C: admitted with the expected sign (sign of the calibrated estimate); +W / -W: admitted with the "
              "wrong sign; 0: not admitted. Codes: NE/NL naive event/level; XP/XS/XD positive-/signed-peak/direction-aware "
              "cross-correlation; GR Granger; PC PCMCI+; VL VAR-LiNGAM; VX VARX; MA matched; KI kink; ST sequential trial. "
              "CSV: `runs/extended/B_tiers/2s/call_matrix_wide.csv` (and `call_matrix_long.csv` with estimates and CIs).", "",
              call_matrix_md(r2["W"]), "",
              "## 6. Secondary resolution: 30 s (v3 run, 18 relations)", "",
              "Tier membership restricted to the 18 relations of ClosedLoopBench 1.0: A = R01, R02 (2); B = R03-R12 (10; "
              "B1 = R03-R09, R11, R12; B2 = R10); C = P01-P06 (6). Settings as in App. C.8 (30 s windows, 15 s stride; "
              "PCMCI+ on 1,200 cases).", "",
              "\n\n".join(r30["md"]), ""]
    for t in MAIN_TIERS:
        if t in r30["ranks"]:
            lines += [f"**Tier {t} rankings (30 s)**", "", ranking_md(r30["ranks"][t], t), ""]
    lines += ["**Verdicts (30 s):**", ""]
    for t in MAIN_TIERS:
        lines.append(f"- Tier {t} ({TIER_NAME[t]}): {r30['verdicts'][t]}")
    lines += ["", "Rule-derived statements (30 s):", ""]
    for t in MAIN_TIERS:
        lines.append(f"- {r30['mech'][t]}")
    lines += ["", "Tier A at 30 s (R01, R02):", "", headline_md(r30["H"]), "",
              call_matrix_md(r30["W"]), "",
              "## 7. LaTeX tables (2 s), ready for the appendix", "", "```latex", r2["tex"], "```", "",
              "30 s versions: `runs/extended/B_tiers/30s/tier_tables.tex`.", "",
              "## 8. Summary paragraph (<= 250 words)", "", summary_paragraph, "",
              f"({len(summary_paragraph.split())} words.)", "",
              "## 9. Appendix paragraph (LaTeX)", "", "```latex", appendix_par, "```", "",
              "## Caveats", "",
              "- The tier tables re-partition published calls; the BY adjustment was made over each estimator's full family "
              "(29 relations; 19 for the naive level contrast), not within a tier. Re-forming the family per tier would make "
              "the adjusted p-values less conservative (fewer tests) and was not done, so the counts here are exactly those "
              "of Table 1 split by tier.",
              "- Tier A has 5 relations and tier B2 has 2, so ranks within them are coarse and many estimators tie; the "
              "Rule 1 / Rule 2 ranks are reported as pre-declared, including the ties.",
              "- Granger and VAR-LiNGAM give one estimate for R01/R02 and for R13/R14, and the positive- and signed-peak "
              "rules give one estimate for R01 and P05 (and other response/policy pairs); admissions are counted over "
              "relations as in Table 1 and also over distinct estimates (column *Distinct exp/wrong*). On tier A the "
              "VAR-LiNGAM 5/0 is 3 distinct estimates and the Granger 2/3 is 3 distinct estimates (R01/R02, R13/R14, R15).",
              "- PCMCI+ ran on 60 cases at 2 s; its 0-2 admissions per tier and zero false admissions reflect low power, "
              "so its Rule 1 ranks (4th on A and B, 1st on C) are earned by not admitting, not by discriminating. At 30 s "
              "(1,200 cases) it admits 2/0 on tier A, 6/0 on tier B and 5/0 on tier C with 0 random-time false "
              "admissions, and the paper notes that its set-rate admissions there come from the transplant offset.",
              "- Random-time false admissions are counted within the tier; an estimator's RT count over all 29 relations "
              "(Table 1) is the sum over tiers A, B and C.",
              "- The 30 s run is ClosedLoopBench 1.0 (v3 relation table, 18 relations), so tier A there has only the two "
              "set-rate relations and tier C only 6 policies; the two resolutions are not like-for-like in relation set or "
              "estimator settings (App. C.8).",
              "- Everything here is descriptive; no new hypothesis test was performed."]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT),
                    help="output folder; a 2s/ and a 30s/ subfolder are written (default: %(default)s)")
    args = ap.parse_args(argv)
    OUT = Path(args.out)   # read by analyse() at call time; the note path (NOTE) and the inputs are unchanged
    res = {k: analyse(k) for k in ("2s", "30s")}
    for k, r in res.items():
        print(f"== {k}: validation {'OK' if not r['problems'] else 'FAILED'} ({r['seconds']:.1f} s)")
        for p in r["problems"]:
            print("   ", p)
    if res["2s"]["problems"]:
        print("Table 1 not reproduced; stopping before the tier analysis is written.", file=sys.stderr)
        sys.exit(1)
    assert_facts(res)
    write_note(res)
    print(f"wrote {NOTE}")
    for t in MAIN_TIERS:
        print(res["2s"]["verdicts"][t])


if __name__ == "__main__":
    main()
