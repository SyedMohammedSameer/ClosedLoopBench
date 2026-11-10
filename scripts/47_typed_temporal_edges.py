#!/usr/bin/env python
"""
Track B: typed, pooled temporal edges between concepts (policy / response / association) on VitalDB
30 s numerics (15 s stride), compared with v1 temporal edges (precedes/follows xcorr and
granger_precedes, reimplemented through ``clbench.edges.temporal`` / ``clbench.edges.granger``),
pairwise Granger + BH and PCMCI+ + BH on the same concept series.

Stages (all in one run):
  0  cohort arrays (concept tri-state series, doses, action-change bins, vitals)       -> cache/
  P  per-patient methods on full / first half / second half / transplant / shift nulls
  A  pooled typed hazard fits: full, H1, H2, R transplant + R shift null replicates
  B  held-out future log-likelihood of concept onsets (fit H1, score H2)
  R  admissions, replication, false-edge rates, sign checks, graph sizes -> results.json, report.md

The evaluation spec is pre-registered in runs/iclr/typed_edges/protocol.md; its sha256 is recorded in
the outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clbench.edges import typed_temporal as tt  # noqa: E402

NUM_DIR = ROOT / "data/processed_vitaldb/numerics"
EV_DIR = ROOT / "data/processed_vitaldb/events"
CALIB = ROOT / "checkpoints/cbm/persistence_calibration.json"

log = logging.getLogger("typed_edges")

LABEL_COLS = {
    "tachycardia": "label_tachycardia", "bradycardia": "label_bradycardia",
    "hypotension": "label_hypotension", "hypertension": "label_hypertension",
    "tachypnea": "label_tachypnea", "bradypnea": "label_bradypnea",
}
DOSE_COLS = {
    "propofol": "tci_propofol_ct", "remifentanil": "tci_remifentanil_ct",
    "phenylephrine": "inf_phenylephrine_rate_ml_h", "norepinephrine": "inf_norepinephrine_rate_ml_h",
    "vent_rr": "vent_set_rr",
}
AGENT_ACTION = {"propofol": "propofol", "remifentanil": "remifentanil", "phenylephrine": "phenylephrine",
                "ventilator": "vent_rr"}
EXCLUDE_ZERO = {"propofol": True, "remifentanil": True, "phenylephrine": False, "vent_rr": True}
ETCO2_VALID = (15.0, 100.0)

# Pharmacological sign expectations (+1: hazard up; -1: hazard down).
RESPONSE_SIGNS = [
    ("propofol_up", "hypotension", +1),
    ("remifentanil_up", "bradycardia", +1),
    ("remifentanil_up", "hypotension", +1),
    ("vent_rr_up", "hypocapnia", +1),
    ("vent_rr_up", "hypercapnia", -1),
    ("phenylephrine_up", "hypotension", -1),
]
POLICY_SIGNS = [
    ("hypertension", "remifentanil_up", +1),
    ("tachycardia", "remifentanil_up", +1),
    ("hypotension", "phenylephrine_up", +1),
    ("hypotension", "propofol_down", +1),
    ("hypercapnia", "vent_rr_up", +1),
]
# Secondary policy checks added after the exploratory pilot (disclosed in protocol.md; not primary).
POLICY_SIGNS_POST_PILOT = [
    ("hypocapnia", "vent_rr_down", +1),
    ("hypertension", "propofol_up", +1),
]


# ---------------------------------------------------------------------------
# Stage 0: cohort
# ---------------------------------------------------------------------------


def _load_case(pid: str) -> dict | None:
    df = pd.read_parquet(NUM_DIR / f"{pid}.parquet")
    T = len(df)
    if T < 120:  # < 30 min
        return None
    ws = df["window_start_sec"].to_numpy(float)
    S = np.full((T, len(tt.CONCEPTS)), -1, dtype=np.int8)
    for j, c in enumerate(tt.CONCEPTS):
        if c in LABEL_COLS:
            v = pd.to_numeric(df[LABEL_COLS[c]], errors="coerce").to_numpy(float)
            S[:, j] = np.where(np.isfinite(v), (v > 0.5).astype(np.int8), -1)
    et = df["etco2"].to_numpy(float)
    et_ok = np.isfinite(et) & (et >= ETCO2_VALID[0]) & (et <= ETCO2_VALID[1])
    S[:, tt.CONCEPTS.index("hypercapnia")] = np.where(et_ok, (et > 45).astype(np.int8), -1)
    S[:, tt.CONCEPTS.index("hypocapnia")] = np.where(et_ok, (et < 30).astype(np.int8), -1)
    dose = np.column_stack([df[DOSE_COLS[a]].to_numpy(float) for a in tt.DOSES])
    vit = np.column_stack([df[v].to_numpy(float) for v in tt.VITAL_COLS])
    vit[:, 3] = np.where(et_ok, et, np.nan)
    chg = np.zeros((T, len(tt.ACTIONS), 2), dtype=np.int8)
    evp = EV_DIR / f"{pid}.parquet"
    n_ev = 0
    if evp.exists():
        ev = pd.read_parquet(evp)
        for agent, g in ev.groupby("agent"):
            a = AGENT_ACTION.get(agent)
            if a is None:
                continue
            ai = tt.ACTIONS.index(a)
            t = g["t_sec"].to_numpy(float)
            b = g["value_before"].to_numpy(float)
            af = g["value_after"].to_numpy(float)
            use = np.abs(af - b) > 0
            if EXCLUDE_ZERO[a]:
                use &= (b > 0) & (af > 0)
            i = np.searchsorted(ws + 15.0, t, side="right") - 1
            ok = use & (i >= 0) & (i < T)
            ok[ok] &= t[ok] < ws[i[ok]] + 30.0
            for k in np.nonzero(ok)[0]:
                chg[i[k], ai, 0 if af[k] > b[k] else 1] = 1
                n_ev += 1
    run = (np.nan_to_num(dose[:, 0]) > 0) | (np.nan_to_num(dose[:, 1]) > 0) | (np.nan_to_num(dose[:, 4]) > 0)
    a0 = int(np.argmax(run)) if run.any() else -1
    return {"pid": pid, "T": T, "S": S, "dose": dose, "vit": vit, "chg": chg, "anes_start": a0,
            "n_ev": n_ev, "ws_irregular": int(np.sum(np.abs(np.diff(ws) - 15.0) > 1e-6))}


def build_cohort(pids: list[str], workers: int, cache: Path) -> Path:
    path = cache / "cohort.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if list(z["ids"]) == list(pids) or set(z["ids"]).issubset(set(pids)):
            log.info("cohort cache hit: %s", path)
            return path
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_load_case, pids, chunksize=16))
    res = [r for r in res if r is not None]
    T = np.array([r["T"] for r in res])
    case = np.repeat(np.arange(len(res)), T)
    step = np.concatenate([np.arange(t) for t in T])
    vit = np.vstack([r["vit"] for r in res])
    mu = np.nanmean(vit, axis=0)
    sd = np.nanstd(vit, axis=0)
    np.savez(
        path, ids=np.array([r["pid"] for r in res]), T=T, case=case, step=step,
        S=np.vstack([r["S"] for r in res]), dose=np.vstack([r["dose"] for r in res]),
        chg=np.concatenate([r["chg"] for r in res]), vital_z=(vit - mu) / sd,
        anes_start=np.array([r["anes_start"] for r in res]),
        ws_irregular=np.array([r["ws_irregular"] for r in res]),
    )
    log.info("cohort: %d cases, %d rows (%.0fs)", len(res), len(case), time.time() - t0)
    return path


MIMIC_WINDOWS = ROOT / "runs/iclr/reference/per_window.parquet"
MIMIC_PROB = {  # concept -> learned-activation column (v1 fused CBM heads, cached by script 42)
    "tachycardia": "prob_tachycardia", "bradycardia": "prob_bradycardia",
    "tachypnea": "prob_tachypnea", "bradypnea": "prob_bradypnea",
    "hypotension": "probfused_hypotension", "hypertension": "probfused_hypertension",
}


def build_cohort_mimic(cache: Path, calib_thr: dict) -> Path:
    """MIMIC-IV-WDB external cohort: v1 learned concept activations (script 42 cache) on a 15 s grid of
    the record clock (cell mean over windows of any modality), thresholded at the v1 calibrated
    activation thresholds; EtCO2 concepts are unobserved; no action data."""
    path = cache / "cohort.npz"
    if path.exists():
        return path
    cols = ["patient_id", "abs_start"] + sorted(set(MIMIC_PROB.values()))
    w = pd.read_parquet(MIMIC_WINDOWS, columns=cols)
    w = w[w["abs_start"].notna()]
    ids, Ts, Ss = [], [], []
    for pid, g in w.groupby("patient_id"):
        cell = np.floor((g["abs_start"].to_numpy(float) - g["abs_start"].min()) / 15.0).astype(np.int64)
        T = int(cell.max()) + 1
        if T < 120:
            continue
        S = np.full((T, len(tt.CONCEPTS)), -1, dtype=np.int8)
        for c, col in MIMIC_PROB.items():
            v = g[col].to_numpy(float)
            ok = np.isfinite(v)
            if not ok.any():
                continue
            s_ = np.bincount(cell[ok], weights=v[ok], minlength=T)
            n_ = np.bincount(cell[ok], minlength=T)
            m = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
            S[:, tt.CONCEPTS.index(c)] = np.where(np.isfinite(m), (m >= calib_thr[c]).astype(np.int8), -1)
        ids.append(str(pid))
        Ts.append(T)
        Ss.append(S)
    T = np.array(Ts)
    n = int(T.sum())
    np.savez(path, ids=np.array(ids), T=T, case=np.repeat(np.arange(len(T)), T),
             step=np.concatenate([np.arange(t) for t in T]), S=np.vstack(Ss),
             dose=np.full((n, len(tt.DOSES)), np.nan), chg=np.zeros((n, len(tt.ACTIONS), 2), np.int8),
             vital_z=np.full((n, 4), np.nan), anes_start=np.full(len(T), -1),
             ws_irregular=np.zeros(len(T), int))
    log.info("MIMIC cohort: %d patients, %d rows", len(T), n)
    return path


_CO: tt.Cohort | None = None
_MAPS: dict = {}
_ARGS: dict = {}


def load_cohort(path: str) -> tt.Cohort:
    z = np.load(path, allow_pickle=True)
    return tt.Cohort(case=z["case"], step=z["step"], S=z["S"], dose=z["dose"], chg=z["chg"],
                     vital_z=z["vital_z"], anes_start=z["anes_start"], T=z["T"], ids=list(z["ids"]))


def case_spacing(co: tt.Cohort) -> np.ndarray:
    """Per-case shift spacing: max(41, ceil(max_k integrated time of concept k's on-series))."""
    from clbench.edges.controlled import integrated_time

    g = np.full(co.n_cases, 41, dtype=np.int64)
    for c in range(co.n_cases):
        s0, T = co.case_start[c], co.T[c]
        blk = co.S[s0:s0 + T]
        for k in range(blk.shape[1]):
            x = (blk[:, k] == 1).astype(float)
            if x.std() > 0:
                g[c] = max(g[c], int(np.ceil(integrated_time(x))))
        g[c] = min(g[c], max(1, T // 2))
    return g


def stable_seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def null_map(kind: str, rep: int) -> np.ndarray:
    key = (kind, rep)
    if key not in _MAPS:
        for k in [k for k in _MAPS if k != "spacing" and k[1] != 0]:
            del _MAPS[k]  # keep memory bounded: only rep-0 maps and the current one stay cached
        if kind == "nullT":
            _MAPS[key] = tt.transplant_map(_CO, np.random.default_rng(10_000 + rep))
        else:
            if "spacing" not in _MAPS:
                _MAPS["spacing"] = case_spacing(_CO)
            _MAPS[key] = tt.shift_map(_CO, np.random.default_rng(20_000 + rep), _MAPS["spacing"])
    return _MAPS[key]


def _init_worker(path: str, args: dict) -> None:
    global _CO, _ARGS
    _CO = load_cohort(path)
    _ARGS = args


# ---------------------------------------------------------------------------
# Stage P: per-patient methods
# ---------------------------------------------------------------------------

FAMILY = tt.cross_vital_pairs()
NULLP = tt.null_pairs()


def _lag_feats(C: np.ndarray) -> dict[str, np.ndarray]:
    """Binned-lag features for Granger+BH on a (T, 8) 0/1 matrix, rows t >= 40."""
    T = C.shape[0]
    c = np.vstack([np.zeros((1, C.shape[1])), np.cumsum(C, axis=0)])
    t = np.arange(40, T)

    def s(k0, k1):  # sum C[t-k1 .. t-k0]
        return c[t - k0 + 1] - c[t - k1]

    return {"y": C[t], "own": np.stack([C[t - 1], C[t - 2], s(3, 4), s(5, 12), s(13, 40)], axis=2),
            "src": np.stack([C[t - 2], s(3, 4), s(5, 12), s(13, 40)], axis=2)}


def granger_bh_pvals(C_tgt: np.ndarray, C_src: np.ndarray, pairs) -> dict[tuple[int, int], tuple[float, int]]:
    """F-test p-value and sign (sum of source coefficients) per ordered pair (src from C_src)."""
    from scipy import stats as st

    if C_tgt.shape[0] < 100:
        return {}
    Ft = _lag_feats(C_tgt)
    Fs = _lag_feats(C_src)
    out = {}
    n = Ft["y"].shape[0]
    for a, b in pairs:
        y = Ft["y"][:, b]
        xa = Fs["src"][:, a, :]
        if y.std() == 0 or np.all(xa.std(axis=0) == 0):
            out[(a, b)] = (1.0, 0)
            continue
        X0 = np.column_stack([np.ones(n), Ft["own"][:, b, :]])
        X1 = np.column_stack([X0, xa])
        r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
        coef = np.linalg.lstsq(X1, y, rcond=None)[0]
        r1 = y - X1 @ coef
        rss0, rss1 = float(r0 @ r0), float(r1 @ r1)
        k = int(np.linalg.matrix_rank(X1) - np.linalg.matrix_rank(X0))
        dfd = n - int(np.linalg.matrix_rank(X1))
        if k <= 0 or rss1 <= 1e-12 or dfd <= 0:
            out[(a, b)] = (1.0, 0)
            continue
        F = ((rss0 - rss1) / k) / (rss1 / dfd)
        out[(a, b)] = (float(st.f.sf(F, k, dfd)), int(np.sign(coef[-4:].sum())))
    return out


def pcmci_pvals(C: np.ndarray) -> np.ndarray:
    """PCMCI+ (ParCorr, 1-min block means, tau 1..10, pc_alpha 0.01): (8, 8) Bonferroni-over-lag
    min p-value per ordered pair (1.0 when a series is constant)."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    T = (C.shape[0] // 4) * 4
    X = C[:T].reshape(-1, 4, C.shape[1]).mean(axis=1)
    P = np.ones((C.shape[1], C.shape[1]))
    live = np.nonzero(X.std(axis=0) > 0)[0]
    if len(live) < 2 or X.shape[0] < 40:
        return P
    pc = PCMCI(dataframe=pp.DataFrame(X[:, live]), cond_ind_test=ParCorr(), verbosity=0)
    r = pc.run_pcmciplus(tau_min=1, tau_max=10, pc_alpha=0.01)
    pm = np.nan_to_num(r["p_matrix"][:, :, 1:], nan=1.0)
    pmin = np.minimum(1.0, pm.min(axis=2) * pm.shape[2])
    for i, a in enumerate(live):
        for j, b in enumerate(live):
            if a != b:
                P[a, b] = pmin[i, j]
    return P


def v1_edges(C_src: np.ndarray, C_tgt: np.ndarray, pairs_ordered, unordered: bool) -> tuple[set, set]:
    """v1 precedes (xcorr) and granger_precedes edges exactly as kg_construct emits them.
    ``unordered``: real series, test each unordered pair once (xcorr gives the direction);
    otherwise (null) test each ordered (null source, real target) pair in the source->target direction."""
    from clbench.edges.granger import granger_precedes_score, is_granger_edge
    from clbench.edges.temporal import is_directed_temporal_edge, temporal_relation

    ex, eg = set(), set()
    if unordered:
        done = set()
        for a, b in pairs_ordered:
            key = (min(a, b), max(a, b))
            if key in done:
                continue
            done.add(key)
            i, j = key
            rel = temporal_relation(C_src[:, i], C_tgt[:, j], 0.5, 0.5, window_stride_sec=15.0)
            if is_directed_temporal_edge(rel):
                ex.add((i, j) if rel.a_precedes_b else (j, i))
    else:
        for a, b in pairs_ordered:
            rel = temporal_relation(C_src[:, a], C_tgt[:, b], 0.5, 0.5, window_stride_sec=15.0)
            if is_directed_temporal_edge(rel) and rel.a_precedes_b:
                ex.add((a, b))
    for a, b in pairs_ordered:
        if is_granger_edge(granger_precedes_score(C_src[:, a], C_tgt[:, b])):
            eg.add((a, b))
    return ex, eg


def _tiers(S: np.ndarray, calib: dict) -> list[int]:
    """Number of distinct persistence tiers per concept (v1 tier rule; duration = (n-1)*15 + 30 s)."""
    out = []
    for k, c in enumerate(tt.CONCEPTS):
        th = calib.get(c, {"transient_max_sec": 60.0, "episodic_max_sec": 600.0})
        tiers = set()
        for _, ln in tt.episodes(S[:, k]):
            d = (ln - 1) * 15.0 + 30.0
            tiers.add(0 if d <= th["transient_max_sec"] else (1 if d <= th["episodic_max_sec"] else 2))
        out.append(len(tiers))
    return out


def per_patient_task(c: int) -> dict:
    co = _CO
    s0, T = int(co.case_start[c]), int(co.T[c])
    S = co.S[s0:s0 + T]
    C = (S == 1).astype(np.float64)
    mT = null_map("nullT", 0)[s0:s0 + T]
    mS = null_map("nullS", 0)[s0:s0 + T]
    CnT = (tt.apply_map_S(co.S, mT) == 1).astype(np.float64)
    CnS = (tt.apply_map_S(co.S, mS) == 1).astype(np.float64)
    h = T // 2
    obs = S >= 0
    vit_obs = {}
    for seg, sl in (("full", slice(0, T)), ("h1", slice(0, h)), ("h2", slice(h, T))):
        vit_obs[seg] = [bool(obs[sl, k].any()) for k in range(len(tt.CONCEPTS))]
    all_pairs = [(a, b) for a in range(8) for b in range(8) if a != b]
    out: dict = {"case": c, "T": T, "obs": vit_obs, "tiers": _tiers(S, _ARGS["calib"])}
    fam_idx = FAMILY
    # v1 on real segments (all 56 ordered pairs, as v1 builds graphs over every concept pair)
    for seg, sl in (("full", slice(0, T)), ("h1", slice(0, h)), ("h2", slice(h, T))):
        ex, eg = v1_edges(C[sl], C[sl], all_pairs, unordered=True)
        out[f"v1x_{seg}"] = sorted(ex)
        out[f"v1g_{seg}"] = sorted(eg)
        pv = granger_bh_pvals(C[sl], C[sl], fam_idx)
        out[f"gbh_{seg}"] = [(a, b, pv[(a, b)][0], pv[(a, b)][1]) for a, b in fam_idx if (a, b) in pv]
    for nk, Cn in (("nullT", CnT), ("nullS", CnS)):
        ex, eg = v1_edges(Cn, C, fam_idx, unordered=False)
        out[f"v1x_{nk}"] = sorted(ex)
        out[f"v1g_{nk}"] = sorted(eg)
        pv = granger_bh_pvals(C, Cn, fam_idx)
        out[f"gbh_{nk}"] = [(a, b, pv[(a, b)][0], pv[(a, b)][1]) for a, b in fam_idx if (a, b) in pv]
    if c in _ARGS["pcmci_cases"]:
        for seg, sl in (("full", slice(0, T)), ("h1", slice(0, h)), ("h2", slice(h, T))):
            out[f"pcmci_{seg}"] = pcmci_pvals(C[sl]).tolist()
        g1 = [tt.CONCEPTS.index(x) for x in tt.GROUP1]
        for nk, Cn in (("nullT", CnT), ("nullS", CnS)):
            M = C.copy()
            M[:, g1] = Cn[:, g1]
            out[f"pcmci_{nk}"] = pcmci_pvals(M).tolist()
    # presence for typed instantiation
    on_full = []
    for seg, sl in (("full", slice(0, T)), ("h1", slice(0, h)), ("h2", slice(h, T))):
        ep = [len(tt.episodes(S[sl, k])) > 0 for k in range(8)]
        ons = []
        for k in range(8):
            x = S[sl, k]
            o = (x[2:] == 1) & (x[1:-1] == 0) & (x[:-2] == 0)
            ons.append(bool(o.any()))
        ch = co.chg[s0:s0 + T][sl]
        out[f"present_{seg}"] = {"episode": ep, "onset": ons,
                                 "chg": [[bool(ch[:, ai, di].any()) for di in range(2)] for ai in range(4)]}
    del on_full
    return out


# ---------------------------------------------------------------------------
# Stage A: pooled fits
# ---------------------------------------------------------------------------

FAMILIES = ("association", "policy", "response")


def outcome_list(family: str) -> list[tuple]:
    if family == "policy":
        return [(ai, di) for ai in range(len(tt.ACTIONS)) for di in range(2)]
    return [(b,) for b in range(len(tt.CONCEPTS))]


def outcome_name(family: str, o: tuple) -> str:
    if family == "policy":
        return f"{tt.ACTIONS[o[0]]}_{tt.DIRS[o[1]]}"
    return tt.CONCEPTS[o[0]]


def family_rows(family: str, o: tuple) -> np.ndarray:
    if family == "policy":
        return tt.policy_rows(_CO, *o)
    return tt.association_rows(_CO, o[0])


def make_design(family: str, o: tuple, rows: np.ndarray, variant: str, rep: int) -> tt.Design:
    co = _CO
    S_src = None
    chg_src = None
    if variant in ("nullT", "nullS"):
        m = null_map(variant, rep)
        if family == "response":
            chg_src = tt.apply_map_chg(co.chg, m)
        else:
            S_src = tt.apply_map_S(co.S, m)
    if family == "association":
        return tt.association_design(co, o[0], rows, S_src)
    if family == "policy":
        return tt.policy_design(co, o[0], o[1], rows, S_src)
    return tt.response_design(co, o[0], rows, chg_src)


def _rows_for_variant(family: str, o: tuple, variant: str) -> np.ndarray:
    rows = family_rows(family, o)
    if variant == "h1":
        rows = rows[_CO.half_mask(1)[rows]]
    elif variant == "h2":
        rows = rows[_CO.half_mask(2)[rows]]
    return rows


def _outcome_of_rows(family: str, o: tuple, rows: np.ndarray) -> np.ndarray:
    if family == "policy":
        return _CO.chg[rows, o[0], o[1]] > 0
    on, _ = tt.onsets_and_risk(_CO.S[:, o[0]], _CO.cs)
    return on[rows]


def fit_task(job: tuple) -> dict:
    family, o, variant, rep, want_dev = job
    t0 = time.time()
    co = _CO
    rows = _rows_for_variant(family, o, variant)
    y_all = _outcome_of_rows(family, o, rows)
    if int(y_all.sum()) < 5:  # e.g. unobserved concepts in the MIMIC cohort: nothing to fit
        return {"family": family, "outcome": outcome_name(family, o), "variant": variant, "rep": rep,
                "n_rows": int(len(rows)), "n_fit_rows": 0, "n_events": int(y_all.sum()),
                "converged": False, "n_iter": 0, "edges": [], "sec": time.time() - t0}
    rng = np.random.default_rng(stable_seed(family, o, variant if variant in ("h1", "h2") else "full"))
    keep, w_all = tt.subsample_rows(y_all, _ARGS["frac"], rng)
    des = make_design(family, o, rows[keep], variant, rep)
    des.w = w_all[keep]
    fit = tt.fit_logistic_fe(des.X, des.y, des.w, des.case, co.n_cases)
    W = tt.bootstrap_weights(co.n_cases, _ARGS["n_boot"], seed=7)
    draws = tt.score_bootstrap(fit, W)
    es = tt.edge_stats(des, fit, draws)
    res = {
        "family": family, "outcome": outcome_name(family, o), "variant": variant, "rep": rep,
        "n_rows": int(len(rows)), "n_fit_rows": int(keep.sum()), "n_events": int(y_all.sum()),
        "converged": bool(fit.converged), "n_iter": fit.n_iter,
        "edges": [e.__dict__ for e in es],
    }
    if variant == "full" and rep == 0:
        res["coef"] = dict(zip(des.names, fit.beta.tolist()))
    if want_dev:
        # all at-risk rows, chunked: eta, y, exposure per source
        etas, ys, Es, cases = [], [], {s: [] for s in des.blocks}, []
        for ch in np.array_split(rows, max(1, len(rows) // 1_000_000)):
            d = make_design(family, o, ch, variant, rep)
            etas.append(d.X @ fit.beta)
            ys.append(d.y)
            cases.append(d.case)
            for s in des.blocks:
                Es[s].append(d.exposure[s])
        eta = np.concatenate(etas)
        y = np.concatenate(ys)
        cs = np.concatenate(cases)
        alpha = tt.patient_intercepts(eta, y, cs, co.n_cases)
        mu = 1 / (1 + np.exp(-(eta + alpha[cs])))
        dev = {}
        for s in des.blocks:
            Ecat = np.concatenate(Es[s])
            d = tt.deviation_scores(mu, y, Ecat, cs, co.n_cases)
            dl, se = tt.patient_delta_mle(eta, y, Ecat, cs, co.n_cases)
            dev[s] = {"U": d.U.astype(np.float32), "I": d.I.astype(np.float32),
                      "nev": d.n_exp_events.astype(np.float32), "delta": dl.astype(np.float32),
                      "se": se.astype(np.float32)}
        res["dev"] = dev
    res["sec"] = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Stage B: held-out future log-likelihood of concept onsets (association family)
# ---------------------------------------------------------------------------


def ll_task(job: tuple) -> dict:
    b, admitted_src, gate, dev_h1 = job
    co = _CO
    fam = "association"
    o = (b,)
    rows1 = _rows_for_variant(fam, o, "h1")
    rows2 = _rows_for_variant(fam, o, "h2")
    y1 = _outcome_of_rows(fam, o, rows1)
    rng = np.random.default_rng(stable_seed(fam, o, "h1"))
    keep, w1 = tt.subsample_rows(y1, _ARGS["frac"], rng)
    d1 = make_design(fam, o, rows1[keep], "real", 0)
    d1.w = w1[keep]
    srcs = list(d1.blocks)
    src_cols = {s: [d1.names.index(f"{s}:{lab}") for lab in ("unobs",) + tt.SOURCE_LAG_LABELS] for s in srcs}
    all_src_cols = sorted(c for s in srcs for c in src_cols[s])
    base_cols = [i for i in range(d1.X.shape[1]) if i not in all_src_cols]
    gate_cols = {s: [d1.names.index(f"{s}:{lab}") for lab in tt.SOURCE_LAG_LABELS] for s in srcs}

    def cols_for(model):
        if model == "base":
            return base_cols
        if model == "typed":
            return sorted(base_cols + [c for s in srcs if s in admitted_src for c in src_cols[s]])
        return list(range(d1.X.shape[1]))

    def gated(X, cases):
        X = X.copy()
        for s in srcs:
            g = gate[s][cases]
            X[:, gate_cols[s]] *= g[:, None]
        return X

    fits = {}
    for m in ("base", "typed", "pooled", "v1"):
        cols = cols_for(m)
        X = gated(d1.X, d1.case) if m == "v1" else d1.X
        f = tt.fit_logistic(X[:, cols], d1.y, d1.w, d1.case, co.n_cases)
        fits[m] = (cols, f.beta)
    dlt = {s: dev_h1[s] for s in srcs}
    out = {m: np.zeros(co.n_cases) for m in ("base", "typed", "pooled", "v1", "shrunk", "perpatient")}
    nrow = np.zeros(co.n_cases)
    nev = np.zeros(co.n_cases)
    for ch in np.array_split(rows2, max(1, len(rows2) // 400_000)):
        d2 = make_design(fam, o, ch, "real", 0)
        cs = d2.case
        nrow += np.bincount(cs, minlength=co.n_cases)
        nev += np.bincount(cs, weights=d2.y, minlength=co.n_cases)
        etas = {}
        for m, (cols, beta) in fits.items():
            X = gated(d2.X, cs) if m == "v1" else d2.X
            etas[m] = X[:, cols] @ beta
        etas["shrunk"] = etas["pooled"] + sum(dlt[s]["shrunk"][cs] * d2.exposure[s] for s in srcs)
        etas["perpatient"] = etas["pooled"] + sum(dlt[s]["raw"][cs] * d2.exposure[s] for s in srcs)
        for m, e in etas.items():
            ll = d2.y * -np.logaddexp(0, -e) + (1 - d2.y) * -np.logaddexp(0, e)
            out[m] += np.bincount(cs, weights=ll, minlength=co.n_cases)
    return {"target": tt.CONCEPTS[b], "ll": {m: v.tolist() for m, v in out.items()},
            "n_rows": nrow.tolist(), "n_events": nev.tolist()}


# ---------------------------------------------------------------------------
# Stage R: summaries
# ---------------------------------------------------------------------------


def edge_key(family: str, outcome: str, source: str) -> str:
    return f"{family}|{source}->{outcome}"


def admissions(fits: list[dict], variant: str, q: float) -> dict[str, dict]:
    """Rule 1: 1-df Wald test of omega (patient-bootstrap SE), BH at q within family, on one
    variant / rep 0 (untestable edges carry p = 1)."""
    out = {}
    for fam in FAMILIES:
        rows = [(r, e) for r in fits if r["family"] == fam and r["variant"] == variant and r["rep"] == 0
                for e in r["edges"]]
        if not rows:
            continue
        p = np.array([e["p_omega"] for _, e in rows])
        rej = tt.bh(p, q)
        for (r, e), k in zip(rows, rej):
            out[edge_key(fam, r["outcome"], e["source"])] = {**e, "bh": bool(k), "family": fam,
                                                            "outcome": r["outcome"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/iclr/typed_edges/confirmatory"))
    ap.add_argument("--n-cases", type=int, default=0, help="random subset (0 = all)")
    ap.add_argument("--n-null", type=int, default=19)
    ap.add_argument("--n-null-dev", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--pcmci-cases", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--protocol", default=str(ROOT / "runs/iclr/typed_edges/protocol.md"))
    ap.add_argument("--mimic", action="store_true", help="external MIMIC association-only run")
    args = ap.parse_args()
    global FAMILIES
    if args.mimic:
        FAMILIES = ("association",)
    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    proto = Path(args.protocol)
    proto_sha = hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else None
    log.info("protocol sha256: %s", proto_sha)
    code_sha = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (Path(__file__).resolve(), ROOT / "src/clbench/edges/typed_temporal.py")}
    log.info("code sha256: %s", code_sha)
    (out / "provenance.json").write_text(json.dumps({"protocol_sha256": proto_sha, "code_sha256": code_sha,
                                                     "argv": sys.argv}, indent=1))
    t_start = time.time()

    if args.mimic:
        cal = json.loads(CALIB.read_text())["concepts"]
        cpath = build_cohort_mimic(out / "cache", {k: v["activation_threshold"] for k, v in cal.items()})
    else:
        pids = sorted(p.stem for p in NUM_DIR.glob("*.parquet"))
        if args.n_cases:
            rng = np.random.default_rng(0)
            pids = sorted(rng.choice(pids, size=args.n_cases, replace=False).tolist())
        cpath = build_cohort(pids, args.workers, out / "cache")
    global _CO, _ARGS
    _CO = load_cohort(str(cpath))
    co = _CO
    calib = json.loads(CALIB.read_text())["concepts"] if CALIB.exists() else {}
    calib = {k: {"transient_max_sec": v["transient_max_sec"], "episodic_max_sec": v["episodic_max_sec"]}
             for k, v in calib.items()}
    rngp = np.random.default_rng(1)
    pcmci_cases = set(rngp.choice(co.n_cases, size=min(args.pcmci_cases, co.n_cases), replace=False).tolist())
    wargs = {"calib": calib, "pcmci_cases": pcmci_cases, "frac": args.frac, "n_boot": args.n_boot}
    _ARGS = wargs
    log.info("cases %d rows %d", co.n_cases, co.n)

    # ---- Stage A: pooled fits
    jobs = []
    for fam in FAMILIES:
        for o in outcome_list(fam):
            for v in ("full", "h1", "h2"):
                jobs.append((fam, o, v, 0, True))
            for kind in ("nullT", "nullS"):
                for r in range(args.n_null):
                    jobs.append((fam, o, kind, r, r < args.n_null_dev))
    fpath = out / "cache" / "fits.jsonl"
    fits = []
    if fpath.exists():
        fits = [json.loads(line) for line in fpath.read_text().splitlines()]
    done = {(r["family"], r["outcome"], r["variant"], r["rep"]) for r in fits}
    todo = [j for j in jobs if (j[0], outcome_name(j[0], j[1]), j[2], j[3]) not in done]
    log.info("stage A: %d fit jobs (%d cached)", len(todo), len(jobs) - len(todo))
    devs: dict = {}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(str(cpath), wargs)) as ex:
        with fpath.open("a") as fh:
            for i, r in enumerate(ex.map(fit_task, todo, chunksize=1)):
                dv = r.pop("dev", None)
                if dv is not None:
                    np.savez_compressed(out / "cache" / f"dev_{r['family']}_{r['outcome']}_{r['variant']}_{r['rep']}.npz",
                                        **{f"{s}__{k}": v for s, d in dv.items() for k, v in d.items()})
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                fits.append(r)
                if i % 50 == 0:
                    log.info("  fit %d/%d (%.0fs)", i + 1, len(todo), time.time() - t0)
    log.info("stage A done (%.0fs)", time.time() - t0)

    # ---- Stage P: per-patient methods
    ppath = out / "cache" / "per_patient.jsonl"
    pp_res = []
    if ppath.exists():
        pp_res = [json.loads(line) for line in ppath.read_text().splitlines()]
    have = {r["case"] for r in pp_res}
    todo_p = [c for c in range(co.n_cases) if c not in have]
    log.info("stage P: %d cases", len(todo_p))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(str(cpath), wargs)) as ex:
        with ppath.open("a") as fh:
            for i, r in enumerate(ex.map(per_patient_task, todo_p, chunksize=8)):
                fh.write(json.dumps(r) + "\n")
                pp_res.append(r)
                if i % 500 == 0:
                    log.info("  case %d/%d (%.0fs)", i + 1, len(todo_p), time.time() - t0)
    pp_res.sort(key=lambda r: r["case"])
    log.info("stage P done (%.0fs)", time.time() - t0)

    # ---- admissions on H1 (for LL) and v1 gates
    adm_h1 = admissions(fits, "h1", args.q)
    W = tt.bootstrap_weights(co.n_cases, 2000, seed=11)

    def load_dev(fam, outcome, variant, rep=0):
        p = out / "cache" / f"dev_{fam}_{outcome}_{variant}_{rep}.npz"
        z = np.load(p)
        d = {}
        for k in z.files:
            s, f = k.split("__")
            d.setdefault(s, {})[f] = z[k].astype(np.float64)
        return d

    ll_jobs = []
    for b in range(len(tt.CONCEPTS)):
        tgt = tt.CONCEPTS[b]
        if not (out / "cache" / f"dev_association_{tgt}_h1_0.npz").exists():
            log.info("no H1 fit for %s (no onsets): skipped in stage B", tgt)
            continue
        adm = {k.split("|")[1].split("->")[0] for k, e in adm_h1.items()
               if e["family"] == "association" and e["outcome"] == tgt and e["bh"]}
        gate = {}
        for a in range(len(tt.CONCEPTS)):
            if tt.CONCEPT_VITAL[tt.CONCEPTS[a]] == tt.CONCEPT_VITAL[tgt]:
                continue
            g = np.zeros(co.n_cases)
            for r in pp_res:
                e1 = {tuple(x) for x in r["v1x_h1"]} | {tuple(x) for x in r["v1g_h1"]}
                g[r["case"]] = float((a, b) in e1)
            gate[tt.CONCEPTS[a]] = g
        dv = load_dev("association", tgt, "h1")
        dev_h1 = {}
        for s, d in dv.items():
            raw = d["delta"]
            se2 = np.where(d["I"] > 1e-6, d["se"] ** 2, np.inf)  # no exposure information -> no deviation
            raw = np.where(np.isfinite(se2), raw, 0.0)
            shr, tau2 = tt.eb_shrink(raw, se2)
            dev_h1[s] = {"raw": raw, "shrunk": shr, "tau2": tau2}
        ll_jobs.append((b, adm, gate, dev_h1))
    t0 = time.time()
    with ProcessPoolExecutor(min(args.workers, 8), initializer=_init_worker, initargs=(str(cpath), wargs)) as ex:
        ll_res = list(ex.map(ll_task, ll_jobs))
    log.info("stage B done (%.0fs)", time.time() - t0)

    summarize(out, co, fits, pp_res, ll_res, ll_jobs, adm_h1, load_dev, W, args, proto_sha, time.time() - t_start)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize(out, co, fits, pp_res, ll_res, ll_jobs, adm_h1, load_dev, W, args, proto_sha, runtime):
    q = args.q
    res: dict = {"protocol_sha256": proto_sha, "n_cases": int(co.n_cases), "n_rows": int(co.n),
                 "n_null": args.n_null, "n_boot": args.n_boot, "frac": args.frac, "q": q,
                 "runtime_sec_before_summary": runtime}
    adm_full = admissions(fits, "full", q)
    adm_h2 = admissions(fits, "h2", q)
    # null distributions per edge
    null_chi = {}
    for r in fits:
        if r["variant"] in ("nullT", "nullS"):
            for e in r["edges"]:
                k = edge_key(r["family"], r["outcome"], e["source"])
                null_chi.setdefault((k, r["variant"]), {})[r["rep"]] = e["z_omega"] ** 2
    for k, e in adm_full.items():
        mT = max(null_chi.get((k, "nullT"), {0: np.inf}).values())
        mS = max(null_chi.get((k, "nullS"), {0: np.inf}).values())
        e["null_max_chi2_T"] = mT
        e["null_max_chi2_S"] = mS
        e["beats_nulls"] = bool(e["z_omega"] ** 2 > mT and e["z_omega"] ** 2 > mS)
        oms = [x["omega"] for r in fits if r["variant"] == "nullT" and r["family"] == e["family"]
               and r["outcome"] == e["outcome"] for x in r["edges"] if x["source"] == e["source"]]
        e["omega_null_corrected"] = float(e["omega"] - np.median(oms)) if oms else None
        e["admitted"] = bool(e["bh"] and e["beats_nulls"])
        e["sign"] = int(np.sign(e["omega"]))
    res["pooled_edges_full"] = adm_full

    # ---- pooled edge table
    tab = []
    for fam in FAMILIES:
        es = [e for e in adm_full.values() if e["family"] == fam]
        tab.append({"family": fam, "n_candidate": len(es), "n_bh": sum(e["bh"] for e in es),
                    "n_admitted": sum(e["admitted"] for e in es),
                    "n_pos": sum(e["admitted"] and e["omega"] > 0 for e in es),
                    "n_neg": sum(e["admitted"] and e["omega"] < 0 for e in es)})
    res["pooled_summary"] = tab

    # ---- (c) false-edge rates: pooled nulls
    null_rates = []
    for fam in FAMILIES:
        for kind in ("nullT", "nullS"):
            n_bh, n_full, n_tests = [], [], 0
            for rep in range(args.n_null):
                rows = [(r, e) for r in fits if r["family"] == fam and r["variant"] == kind and r["rep"] == rep
                        for e in r["edges"]]
                if not rows:
                    continue
                p = np.array([e["p_omega"] for _, e in rows])
                rej = tt.bh(p, q)
                n_tests = len(rows)
                n_bh.append(int(rej.sum()))
                # full rule: BH and beats the other null replicates (leave-one-out) of both kinds
                cnt = 0
                for (r, e), k in zip(rows, rej):
                    if not k:
                        continue
                    key = edge_key(fam, r["outcome"], e["source"])
                    oT = [v for rr, v in null_chi.get((key, "nullT"), {}).items() if not (kind == "nullT" and rr == rep)]
                    oS = [v for rr, v in null_chi.get((key, "nullS"), {}).items() if not (kind == "nullS" and rr == rep)]
                    if e["z_omega"] ** 2 > max(oT + [0]) and e["z_omega"] ** 2 > max(oS + [0]):
                        cnt += 1
                n_full.append(cnt)
            # nominal (unadjusted) p<0.05 rate
            pn = [e["p_omega"] for r in fits if r["family"] == fam and r["variant"] == kind for e in r["edges"]
                  if e["testable"]]
            pj = [e["p"] for r in fits if r["family"] == fam and r["variant"] == kind for e in r["edges"]]
            null_rates.append({"family": fam, "null": kind, "n_reps": len(n_bh), "n_null_edges_per_rep": n_tests,
                               "mean_bh_admitted": float(np.mean(n_bh)) if n_bh else None,
                               "reps_with_any_bh": int(sum(x > 0 for x in n_bh)),
                               "mean_full_rule_admitted": float(np.mean(n_full)) if n_full else None,
                               "frac_p_lt_0.05": float(np.mean(np.array(pn) < 0.05)) if pn else None,
                               "n_testable": len(pn),
                               "frac_joint_wald_p_lt_0.05": float(np.mean(np.array(pj) < 0.05)) if pj else None,
                               "ks_uniform_p": float(__import__("scipy").stats.kstest(pn, "uniform").pvalue) if pn else None})
    res["pooled_null_rates"] = null_rates

    # ---- patient-level deviations (BY over patient x edge) for full, h1, h2 and nulls
    def dev_edges(variant, rep=0):
        tests = []
        for fam in FAMILIES:
            for o in outcome_list(fam):
                on = outcome_name(fam, o)
                p = out / "cache" / f"dev_{fam}_{on}_{variant}_{rep}.npz"
                if not p.exists():
                    continue
                dv = load_dev(fam, on, variant, rep)
                for s, d in dv.items():
                    I = d["I"]
                    ok = I > 1e-6
                    z = np.where(ok, d["U"] / np.sqrt(np.maximum(I, 1e-300)), np.nan)
                    for c in np.nonzero(ok)[0]:
                        tests.append((fam, on, s, int(c), float(z[c])))
        if not tests:
            return {}, 0
        from scipy import stats as st
        pv = np.array([2 * st.norm.sf(abs(t[4])) for t in tests])
        rej = tt.by(pv, q)
        edges = {}
        for t, k in zip(tests, rej):
            if k:
                edges.setdefault(t[3], set()).add((t[0], t[2], t[1], int(np.sign(t[4]))))
        return edges, len(tests)

    dev_full, n_dev_tests = dev_edges("full")
    dev_h1, _ = dev_edges("h1")
    dev_h2, _ = dev_edges("h2")
    res["deviation_tests_full"] = n_dev_tests
    res["deviation_edges_full"] = int(sum(len(v) for v in dev_full.values()))
    res["deviation_patients_full"] = int(len(dev_full))
    dev_null = []
    for kind in ("nullT", "nullS"):
        for rep in range(args.n_null_dev):
            de, nt = dev_edges(kind, rep)
            dev_null.append({"null": kind, "rep": rep, "n_tests": nt, "n_edges": int(sum(len(v) for v in de.values())),
                             "n_patients": int(len(de))})
    res["deviation_null"] = dev_null

    # ---- per-patient graphs, replication, false edges
    fam_pairs = tt.cross_vital_pairs()
    fam_set = set(fam_pairs)
    g1 = {tt.CONCEPTS.index(x) for x in tt.GROUP1}
    null32 = set(tt.null_pairs())

    def cand(r, seg):
        o = r["obs"][seg]
        return {(a, b) for a, b in fam_pairs if o[a] and o[b]}

    def m_edges(r, method, seg):
        if method in ("v1x", "v1g", "v1"):
            if method == "v1":
                s = {tuple(x) for x in r[f"v1x_{seg}"]} | {tuple(x) for x in r[f"v1g_{seg}"]}
            else:
                s = {tuple(x) for x in r[f"{method}_{seg}"]}
            return {e for e in s if e in fam_set}
        if method == "gbh":
            lst = r.get(f"gbh_{seg}", [])
            if not lst:
                return set()
            p = np.array([x[2] for x in lst])
            rej = tt.bh(p, q)
            return {(x[0], x[1], x[3]) for x, k in zip(lst, rej) if k}
        if method == "pcmci":
            if f"pcmci_{seg}" not in r:
                return None
            P = np.array(r[f"pcmci_{seg}"])
            p = np.array([P[a, b] for a, b in fam_pairs])
            rej = tt.bh(p, q)
            return {e for e, k in zip(fam_pairs, rej) if k}
        raise ValueError(method)

    def typed_inst(r, seg, adm, fam="association"):
        pr = r[f"present_{seg}"]
        es = set()
        for k, e in adm.items():
            if e["family"] != fam or not e.get("bh"):
                continue
            s, o = e["source"], e["outcome"]
            if fam == "association":
                a, b = tt.CONCEPTS.index(s), tt.CONCEPTS.index(o)
                if pr["episode"][a] and pr["onset"][b]:
                    es.add((a, b, int(np.sign(e["omega"]))))
            elif fam == "policy":
                a = tt.CONCEPTS.index(s)
                an, dn = o.rsplit("_", 1)
                ai, di = tt.ACTIONS.index(an), tt.DIRS.index(dn)
                if pr["episode"][a] and pr["chg"][ai][di]:
                    es.add((s, o, int(np.sign(e["omega"]))))
            else:
                an, dn = s.rsplit("_", 1)
                ai, di = tt.ACTIONS.index(an), tt.DIRS.index(dn)
                b = tt.CONCEPTS.index(o)
                if pr["chg"][ai][di] and pr["onset"][b]:
                    es.add((s, o, int(np.sign(e["omega"]))))
        return es

    rep_rows = []

    def rep_row(name, E1, E2, M, note=""):
        d = tt.replication_with_chance(E1, E2, M)
        num = np.array([len(a & b) for a, b in zip(E1, E2)], float)
        den = np.array([len(a) for a in E1], float)
        ch = np.array([len(a) * len(b) / m if m else 0.0 for a, b, m in zip(E1, E2, M)], float)
        Wr = tt.bootstrap_weights(len(num), 1000, 5)
        est, lo, hi = tt.patient_bootstrap_ratio(num, den, Wr)
        _, clo, chi_ = tt.patient_bootstrap_ratio(ch, den, Wr)
        d.update({"method": name, "ci": [lo, hi], "chance_ci": [clo, chi_], "note": note,
                  "edges_per_patient_h1": float(den.mean())})
        rep_rows.append(d)

    for method in ("v1x", "v1g", "v1", "gbh", "pcmci"):
        E1, E2, M = [], [], []
        for r in pp_res:
            e1 = m_edges(r, method, "h1")
            e2 = m_edges(r, method, "h2")
            if e1 is None:
                continue
            E1.append(e1)
            E2.append(e2)
            M.append(len(cand(r, "h1")))
        rep_row(method, E1, E2, M)
    for fam in FAMILIES:
        E1 = [typed_inst(r, "h1", adm_h1, fam) for r in pp_res]
        E2 = [typed_inst(r, "h2", adm_h2, fam) for r in pp_res]
        M = [len(cand(r, "h1")) if fam == "association" else 64 for r in pp_res]
        rep_row(f"typed_{fam}_instantiated", E1, E2, M,
                "pooled H1 admission (BH) instantiated where both endpoints occur in the half")
    # typed deviations
    E1 = [{(x[0], x[1], x[2], x[3]) for x in dev_h1.get(r["case"], set())} for r in pp_res]
    E2 = [{(x[0], x[1], x[2], x[3]) for x in dev_h2.get(r["case"], set())} for r in pp_res]
    rep_row("typed_patient_deviation", E1, E2, [48 + 64 + 64] * len(pp_res), "BY over patient x edge")
    res["replication"] = rep_rows

    # pooled-level replication H1 -> H2
    prep = []
    for fam in FAMILIES:
        h1 = {k: e for k, e in adm_h1.items() if e["family"] == fam and e["bh"]}
        h2 = {k: e for k, e in adm_h2.items() if e["family"] == fam and e["bh"]}
        rep_ = sum(1 for k, e in h1.items() if k in h2 and np.sign(h2[k]["omega"]) == np.sign(e["omega"]))
        prep.append({"family": fam, "n_h1": len(h1), "n_h2": len(h2), "n_replicated_same_sign": rep_,
                     "replication": rep_ / len(h1) if h1 else None,
                     "chance": len(h2) / max(1, len([e for e in adm_h2.values() if e["family"] == fam]))})
    res["pooled_replication"] = prep

    # per-patient false edges under nulls
    fr = []
    for kind in ("nullT", "nullS"):
        for method in ("v1x", "v1g", "v1", "gbh", "pcmci"):
            cnts, ntests, any_ = [], 0, 0
            for r in pp_res:
                e = m_edges(r, method, kind)
                if e is None:
                    continue
                if method == "pcmci":
                    e = {x for x in e if x in null32}
                    nt = len([x for x in null32 if r["obs"]["full"][x[0]] and r["obs"]["full"][x[1]]])
                else:
                    e = {x for x in e if (x[0], x[1]) in fam_set}
                    nt = len(cand(r, "full"))
                cnts.append(len(e))
                ntests += nt
                any_ += len(e) > 0
            if cnts:
                fr.append({"null": kind, "method": method, "n_graphs": len(cnts),
                           "false_edges_per_graph": float(np.mean(cnts)),
                           "false_edges_per_graph_median": float(np.median(cnts)),
                           "frac_null_tests_positive": float(np.sum(cnts) / max(1, ntests)),
                           "frac_graphs_with_any": any_ / len(cnts)})
        # typed instantiated under null: pooled null admissions (rule 1) of rep 0 instantiated
        for fam in FAMILIES:
            rows = [(r, e) for r in fits if r["family"] == fam and r["variant"] == kind and r["rep"] == 0 for e in r["edges"]]
            p = np.array([e["p_omega"] for _, e in rows])
            rej = tt.bh(p, q) if len(p) else np.zeros(0, bool)
            admn = {}
            for (r, e), k in zip(rows, rej):
                key = edge_key(fam, r["outcome"], e["source"])
                oT = [v for rr, v in null_chi.get((key, "nullT"), {}).items() if not (kind == "nullT" and rr == 0)]
                oS = [v for rr, v in null_chi.get((key, "nullS"), {}).items() if not (kind == "nullS" and rr == 0)]
                full_rule = bool(k) and e["z_omega"] ** 2 > max(oT + [0]) and e["z_omega"] ** 2 > max(oS + [0])
                admn[key] = {**e, "bh": full_rule, "family": fam, "outcome": r["outcome"]}
            cnts = [len(typed_inst(r, "full", admn, fam)) for r in pp_res]
            fr.append({"null": kind, "method": f"typed_{fam}_instantiated", "n_graphs": len(cnts),
                       "false_edges_per_graph": float(np.mean(cnts)),
                       "false_edges_per_graph_median": float(np.median(cnts)),
                       "frac_null_tests_positive": float(rej.mean()) if len(rej) else None,
                       "frac_graphs_with_any": float(np.mean(np.array(cnts) > 0))})
    res["per_patient_null_rates"] = fr

    # ---- (b) held-out log-likelihood
    llt = []
    tot = {m: np.zeros(co.n_cases) for m in ll_res[0]["ll"]}
    tot_ev = np.zeros(co.n_cases)
    tot_rows = np.zeros(co.n_cases)
    for r in ll_res:
        nev = np.array(r["n_events"])
        base = np.array(r["ll"]["base"])
        row = {"target": r["target"], "n_events_h2": int(nev.sum()), "n_rows_h2": int(np.sum(r["n_rows"]))}
        for m, v in r["ll"].items():
            v = np.array(v)
            tot[m] += v
            if m == "base":
                row["base_ll_per_event"] = float(v.sum() / max(1, nev.sum()))
                continue
            est, lo, hi = tt.patient_bootstrap_ratio(v - base, nev, W)
            row[f"dLL_{m}"] = [est, lo, hi]
        tot_ev += nev
        tot_rows += np.array(r["n_rows"])
        llt.append(row)
    row = {"target": "ALL", "n_events_h2": int(tot_ev.sum()), "n_rows_h2": int(tot_rows.sum())}
    for m in tot:
        if m == "base":
            continue
        est, lo, hi = tt.patient_bootstrap_ratio(tot[m] - tot["base"], tot_ev, W)
        row[f"dLL_{m}"] = [est, lo, hi]
    # paired comparisons
    for a, b in (("shrunk", "pooled"), ("shrunk", "perpatient"), ("typed", "v1"), ("pooled", "v1"),
                 ("shrunk", "v1")):
        est, lo, hi = tt.patient_bootstrap_ratio(tot[a] - tot[b], tot_ev, W)
        row[f"{a}_minus_{b}"] = [est, lo, hi]
    llt.append(row)
    res["heldout_ll"] = llt
    res["eb_tau2"] = {tt.CONCEPTS[j[0]]: {s: d["tau2"] for s, d in j[3].items()} for j in ll_jobs}

    # ---- (d) sign checks
    sc = []
    for src, tgt, exp in RESPONSE_SIGNS:
        e = adm_full.get(edge_key("response", tgt, src))
        nT = [r for r in fits if r["family"] == "response" and r["outcome"] == tgt and r["variant"] == "nullT"]
        om_null = [x["omega"] for r in nT for x in r["edges"] if x["source"] == src]
        sc.append(_sign_row("response", src, tgt, exp, e, om_null))
    for lst, tag in ((POLICY_SIGNS, "primary"), (POLICY_SIGNS_POST_PILOT, "secondary_post_pilot")):
        for src, tgt, exp in lst:
            e = adm_full.get(edge_key("policy", tgt, src))
            nT = [r for r in fits if r["family"] == "policy" and r["outcome"] == tgt and r["variant"] == "nullT"]
            om_null = [x["omega"] for r in nT for x in r["edges"] if x["source"] == src]
            row = _sign_row("policy", src, tgt, exp, e, om_null)
            row["tier"] = tag
            sc.append(row)
    res["sign_checks"] = sc

    # ---- graph sizes per patient (full data)
    gs = []
    all_pairs_n = 0
    for r in pp_res:
        tiers = r["tiers"]
        nodes = int(sum(tiers))
        v1x = {tuple(x) for x in r["v1x_full"]}
        v1g = {tuple(x) for x in r["v1g_full"]}
        v1_emitted = sum(2 * tiers[a] * tiers[b] for a, b in v1x) + sum(tiers[a] * tiers[b] for a, b in v1g)
        ti = {fam: typed_inst(r, "full", {k: {**e, "bh": e["admitted"]} for k, e in adm_full.items()}, fam)
              for fam in FAMILIES}
        pr = r["present_full"]
        n_act = sum(pr["chg"][ai][di] for ai in range(4) for di in range(2))
        gs.append({"nodes_concept_tier": nodes, "action_nodes": int(n_act),
                   "v1_temporal_edges_emitted": int(v1_emitted),
                   "v1_concept_pairs_xcorr": len(v1x), "v1_concept_pairs_granger": len(v1g),
                   "v1_concept_pairs_any": len(v1x | v1g),
                   "typed_association": len(ti.get("association", ())), "typed_policy": len(ti.get("policy", ())),
                   "typed_response": len(ti.get("response", ())),
                   "typed_deviation": len(dev_full.get(r["case"], set()))})
        all_pairs_n += 1
    gdf = pd.DataFrame(gs)
    res["graph_size"] = {c: {"mean": float(gdf[c].mean()), "median": float(gdf[c].median()),
                             "p90": float(gdf[c].quantile(0.9))} for c in gdf.columns}
    gdf.to_csv(out / "graph_size_per_patient.csv", index=False)
    res["runtime_sec"] = runtime
    (out / "results.json").write_text(json.dumps(res, indent=1, default=_json_default))
    write_report(out, res)
    log.info("wrote %s", out / "results.json")


def _sign_row(fam, src, tgt, exp, e, om_null):
    if e is None:
        return {"family": fam, "source": src, "target": tgt, "expected": exp, "status": "missing"}
    lo, hi = e["omega_lo"], e["omega_hi"]
    if lo > 0:
        call = "correct" if exp > 0 else "WRONG"
    elif hi < 0:
        call = "correct" if exp < 0 else "WRONG"
    else:
        call = "inconclusive"
    return {"family": fam, "source": src, "target": tgt, "expected": exp, "omega": e["omega"],
            "ci": [lo, hi], "hr": float(np.exp(e["omega"])), "p": e["p_omega"], "p_joint": e["p"],
            "admitted": e.get("admitted"), "omega_null_corrected": e.get("omega_null_corrected"),
            "n_event_patients": e.get("n_event_patients"),
            "call": call, "coefs": e["coefs"], "n_exposed_events": e["n_exposed_events"],
            "null_omega_median": float(np.median(om_null)) if om_null else None,
            "null_omega_range": [float(np.min(om_null)), float(np.max(om_null))] if om_null else None}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _f(x, nd=3):
    if x is None:
        return "NA"
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(out: Path, res: dict) -> None:
    L = ["# Typed temporal concept edges (Track B) — results", "",
         f"Protocol sha256: `{res['protocol_sha256']}`", "",
         f"Cases: {res['n_cases']}, rows (15 s steps): {res['n_rows']}, null replicates per kind: {res['n_null']}, "
         f"bootstrap draws: {res['n_boot']}, non-event sampling fraction: {res['frac']}, q = {res['q']}.", "",
         "## Pooled typed edges (full data)", "",
         "| family | candidates | BH (joint Wald, patient bootstrap) | admitted (BH + beats transplant & shift nulls) | + | - |",
         "|---|---|---|---|---|---|"]
    for t in res["pooled_summary"]:
        L.append(f"| {t['family']} | {t['n_candidate']} | {t['n_bh']} | {t['n_admitted']} | {t['n_pos']} | {t['n_neg']} |")
    L += ["", "Admitted pooled edges (full data):", "", "| family | source -> target | omega [95% CI] | HR | null-corrected omega | exposed events / patients |", "|---|---|---|---|---|---|"]
    for k, e in sorted(res["pooled_edges_full"].items()):
        if e.get("admitted"):
            L.append(f"| {e['family']} | {e['source']} -> {e['outcome']} | {e['omega']:.2f} [{e['omega_lo']:.2f}, {e['omega_hi']:.2f}] | "
                     f"{np.exp(e['omega']):.2f} | {_f(e.get('omega_null_corrected'), 2)} | {int(e['n_exposed_events'])} / {e['n_event_patients']} |")
    L += ["", "## (a) Held-out-future replication within patient (fit first half, check second half)", "",
          "| method | H1 edges | edges/patient (H1) | replicated | replication [95% CI] | density-matched chance [95% CI] |",
          "|---|---|---|---|---|---|"]
    for r in res["replication"]:
        L.append(f"| {r['method']} | {r['n_edges_h1']} | {r['edges_per_patient_h1']:.2f} | {r['n_replicated']} | "
                 f"{_f(r['replication'])} [{_f(r['ci'][0])}, {_f(r['ci'][1])}] | {_f(r['chance'])} "
                 f"[{_f(r['chance_ci'][0])}, {_f(r['chance_ci'][1])}] |")
    L += ["", "Pooled-level replication (edge admitted on all H1 halves -> admitted on H2 halves, same sign):", "",
          "| family | H1 edges | H2 edges | replicated | rate | chance (H2 density) |", "|---|---|---|---|---|---|"]
    for r in res["pooled_replication"]:
        L.append(f"| {r['family']} | {r['n_h1']} | {r['n_h2']} | {r['n_replicated_same_sign']} | {_f(r['replication'])} | {_f(r['chance'])} |")
    L += ["", "## (b) Held-out future log-likelihood of concept onsets (second halves; nats per onset vs base)", "",
          "| target | H2 onsets | pooled | typed (admitted only) | v1-edge features | shrunk patient | per-patient |",
          "|---|---|---|---|---|---|---|"]
    for r in res["heldout_ll"]:
        L.append(f"| {r['target']} | {r['n_events_h2']} | {_f(r.get('dLL_pooled'))} | {_f(r.get('dLL_typed'))} | "
                 f"{_f(r.get('dLL_v1'))} | {_f(r.get('dLL_shrunk'))} | {_f(r.get('dLL_perpatient'))} |")
    last = res["heldout_ll"][-1]
    L += ["", "Paired differences (all targets, nats per onset): " + "; ".join(
        f"{k}: {_f(v)}" for k, v in last.items() if "_minus_" in k), ""]
    L += ["## (c) False edges under nulls", "", "Pooled typed edges (per null replicate):", "",
          "| family | null | reps | null edges/rep | mean BH-admitted | reps with any | mean full-rule admitted | frac p<0.05 (omega, testable) | KS uniform p | frac joint-Wald p<0.05 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["pooled_null_rates"]:
        L.append(f"| {r['family']} | {r['null']} | {r['n_reps']} | {r['n_null_edges_per_rep']} | {_f(r['mean_bh_admitted'], 2)} | "
                 f"{r['reps_with_any_bh']} | {_f(r['mean_full_rule_admitted'], 2)} | {_f(r['frac_p_lt_0.05'])} (n={r['n_testable']}) | {_f(r['ks_uniform_p'])} | {_f(r['frac_joint_wald_p_lt_0.05'])} |")
    L += ["", "Per-patient graphs under nulls (edges among null pairs per patient graph):", "",
          "| null | method | graphs | false edges/graph (mean) | median | frac null tests positive | frac graphs with any |",
          "|---|---|---|---|---|---|---|"]
    for r in res["per_patient_null_rates"]:
        L.append(f"| {r['null']} | {r['method']} | {r['n_graphs']} | {_f(r['false_edges_per_graph'], 2)} | "
                 f"{_f(r['false_edges_per_graph_median'], 1)} | {_f(r['frac_null_tests_positive'])} | {_f(r['frac_graphs_with_any'])} |")
    L += ["", f"Patient-level deviation edges (BY over patient x edge): full data {res['deviation_edges_full']} edges in "
          f"{res['deviation_patients_full']} patients out of {res['deviation_tests_full']} tests; under nulls: "
          + "; ".join(f"{d['null']}#{d['rep']}: {d['n_edges']} edges / {d['n_tests']} tests" for d in res["deviation_null"]), ""]
    L += ["## (d) Pharmacological sign checks (closed-loop caveat: observational, confounded by indication)", "",
          "| family | source | target | expected | HR (exp omega) | omega [95% CI] | p | call | admitted | lag-bin coefs | null omega median (transplant) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["sign_checks"]:
        if r.get("status") == "missing":
            L.append(f"| {r['family']} | {r['source']} | {r['target']} | {r['expected']:+d} | NA | NA | missing | | | |")
            continue
        L.append(f"| {r['family']} | {r['source']} | {r['target']} | {r['expected']:+d} | {r['hr']:.2f} | "
                 f"{r['omega']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['p']:.1e} | {r['call']} | {r['admitted']} | "
                 f"{', '.join(f'{c:.2f}' for c in r['coefs'])} | {_f(r['null_omega_median'])} |")
    L += ["", "## Per-patient graph size (full data)", "", "| quantity | mean | median | p90 |", "|---|---|---|---|"]
    for k, v in res["graph_size"].items():
        L.append(f"| {k} | {v['mean']:.2f} | {v['median']:.1f} | {v['p90']:.1f} |")
    (out / "report.md").write_text("\n".join(L) + "\n")


def posthoc_deviation_empirical_null(out: Path, q: float = 0.05) -> dict:
    """EXPLORATORY (not pre-registered; added after the confirmatory run showed the BY deviation layer
    admits nearly as many edges under the nulls as on real data). Per edge, the patient-level score
    z-statistics from the null replicates with deviation passes (transplant 0-1, shift 0-1) form an
    empirical null; each real test gets p = (1 + #{|z_null| >= |z|}) / (N_null + 1); BY over all
    patient x edge tests. Calibration check: each null replicate in turn is scored against the other
    three."""
    from scipy import stats as st

    cache = out / "cache"

    def zs(variant, rep):
        res = {}
        for f in sorted(cache.glob(f"dev_*_{variant}_{rep}.npz")):
            _, fam, *mid = f.stem.split("_")
            outcome = "_".join(mid[:-2])
            z = np.load(f)
            for k in z.files:
                if not k.endswith("__U"):
                    continue
                src = k[:-3]
                I = z[src + "__I"].astype(float)
                ok = I > 1e-6
                res[(fam, outcome, src)] = (np.nonzero(ok)[0], z[k].astype(float)[ok] / np.sqrt(I[ok]))
        return res

    nulls = {(v, r): zs(v, r) for v in ("nullT", "nullS") for r in (0, 1)}
    real = zs("full", 0)

    def score(target: dict, null_keys) -> tuple[int, int, int, float]:
        pv, owners = [], []
        for key, (cases, z) in target.items():
            pool = np.concatenate([np.abs(nulls[nk][key][1]) for nk in null_keys if key in nulls[nk]] or [np.zeros(0)])
            if pool.size == 0:
                continue
            pool.sort()
            ge = pool.size - np.searchsorted(pool, np.abs(z), side="left")
            pv.append((1 + ge) / (pool.size + 1))
            owners.append(cases)
        if not pv:
            return 0, 0, 0, float("nan")
        pv_all = np.concatenate(pv)
        rej = tt.by(pv_all, q)
        pats = np.concatenate(owners)[rej]
        nominal = float(np.mean(pv_all < 0.05))
        return int(len(pv_all)), int(rej.sum()), int(len(np.unique(pats))), nominal

    out_d = {"note": "exploratory, not pre-registered", "q": q}
    n, k, npat, nom = score(real, list(nulls))
    out_d["real"] = {"n_tests": n, "n_edges": k, "n_patients": npat, "frac_emp_p_lt_0.05": nom}
    cal = []
    for nk in nulls:
        others = [x for x in nulls if x != nk]
        n, k, npat, nom = score(nulls[nk], others)
        cal.append({"null": nk[0], "rep": nk[1], "n_tests": n, "n_edges": k, "n_patients": npat,
                    "frac_emp_p_lt_0.05": nom})
    out_d["null_leave_one_out"] = cal
    # overdispersion of the model-based score z under the nulls (1.0 = calibrated)
    disp = []
    for nk, d in nulls.items():
        allz = np.concatenate([z for _, z in d.values()])
        disp.append({"null": nk[0], "rep": nk[1], "var_z": float(np.var(allz)),
                     "frac_abs_z_gt_1.96": float(np.mean(np.abs(allz) > 1.96))})
    out_d["model_based_z_under_null"] = disp
    allz = np.concatenate([z for _, z in real.values()])
    out_d["model_based_z_real"] = {"var_z": float(np.var(allz)), "frac_abs_z_gt_1.96": float(np.mean(np.abs(allz) > 1.96))}
    (out / "posthoc_deviation_empirical_null.json").write_text(json.dumps(out_d, indent=1))
    return out_d


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--posthoc-deviation":
        print(json.dumps(posthoc_deviation_empirical_null(Path(sys.argv[2])), indent=1))
    else:
        main()
