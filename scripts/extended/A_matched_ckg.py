#!/usr/bin/env python
"""
Extended analysis A: matched construction ladder, CKG vs the feedback-robust graph construction (VitalDB).

Every rung shares the 176 candidate relations, the 38 null replicates of script 47, the pointer-rule
instantiation of clbench.kg_v2 and the false-edge metric FE. Stages (run in this order; each caches):

  validate    rung 6 reproduction from the cached hazard fits (null admissions + instances vs the published
              null_replicates_vitaldb.csv); per-patient / PCMCI replicate-0 checks against per_patient.jsonl
              once those stages exist
  perpatient  rungs 1-3: CKG xcorr / Granger / Jaccard statistics and Granger+BH p-values per case on
              full / h1 / h2 and all 38 null replicates
  pcmci       rung 2: PCMCI+ on the paper's 400 cases, full / h1 / h2 and the null replicates
  pooled      rungs 4, 5, 7: pooled FE cross-correlation, pooled FE Granger, per-patient G-test hazard,
              for every candidate on full / h1 / h2 / 38 nulls
  instances   pointer-rule instances of all 176 candidates per case on full / h1 / h2 / 38 nulls
  summary     admissions, FE per rung, real-data counts, replication, ladder tables -> results.json

Usage: PYTHONPATH=src .venv/bin/python scripts/extended/A_matched_ckg.py --stage <stage> [--n-cases N]
       [--workers 6] [--pcmci-cases 400] [--pcmci-reps 19]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from clbench import kg_v2 as kv  # noqa: E402
from clbench.edges import typed_temporal as tt  # noqa: E402
from clbench.persistence import CalibrationConfig  # noqa: E402
from clbench.extended import matched_ckg as mc  # noqa: E402

TE = ROOT / "runs/iclr/typed_edges/confirmatory"
V2 = ROOT / "runs/iclr/graphs_v2"
CALIB = ROOT / "checkpoints/cbm/persistence_calibration.json"
OUT = ROOT / "runs/extended/A_matched_ckg"
Q = 0.05
N_NULL = 19
NULL_VARIANTS = [(k, r) for k in ("nullT", "nullS") for r in range(N_NULL)]
REAL_VARIANTS = [("full", 0), ("h1", 0), ("h2", 0)]
ALL_VARIANTS = REAL_VARIANTS + NULL_VARIANTS
CANDS = mc.candidates()
KEYS = [mc.key(*c) for c in CANDS]
KEY_IDX = {k: i for i, k in enumerate(KEYS)}
PAIRS48 = tt.cross_vital_pairs()
PAIRS56 = [(a, b) for a in range(8) for b in range(8) if a != b]
G1 = [tt.CONCEPTS.index(x) for x in tt.GROUP1]
log = logging.getLogger("A_matched_ckg")


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S47 = load_script("s47", ROOT / "scripts/47_typed_temporal_edges.py")

_CO: tt.Cohort | None = None
_CALIB = None


def _init(path: str) -> None:
    global _CO, _CALIB
    _CO = S47.load_cohort(path)
    S47._CO = _CO
    _CALIB = CalibrationConfig.load(CALIB)


def vname(variant: tuple[str, int]) -> str:
    return f"{variant[0]}_{variant[1]}"


def mapped_sources(co: tt.Cohort, variant: tuple[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(S_src, chg_src) on the full cohort rows: real arrays, or the null-mapped copies of script 47."""
    kind, rep = variant
    if kind in ("nullT", "nullS"):
        m = S47.null_map(kind, rep)
        return tt.apply_map_S(co.S, m), tt.apply_map_chg(co.chg, m)
    return co.S, co.chg


def half_mask(co: tt.Cohort, variant: tuple[str, int]) -> np.ndarray | None:
    return {"h1": co.half_mask(1), "h2": co.half_mask(2)}.get(variant[0])


def case_subset(n_cases: int, n: int) -> np.ndarray:
    if not n or n >= n_cases:
        return np.arange(n_cases)
    return np.sort(np.random.default_rng(0).choice(n_cases, size=n, replace=False))


# ---------------------------------------------------------------------------
# Stage perpatient (rungs 1-3)
# ---------------------------------------------------------------------------


def pp_task(job: tuple) -> tuple:
    variant, cases = job
    co = _CO
    kind, rep = variant
    null = kind in ("nullT", "nullS")
    m = S47.null_map(kind, rep) if null else None
    pairs = PAIRS48 if null else PAIRS56
    out = {k: [] for k in ("xcorr_peak", "xcorr_lag", "xcorr_edge", "granger_p", "granger_ok", "jaccard")}
    gbh_p, gbh_sign = [], []
    for c in cases:
        s0, T = int(co.case_start[c]), int(co.T[c])
        S = co.S[s0:s0 + T]
        C = (S == 1).astype(np.float64)
        if null:
            Cn = (tt.apply_map_S(co.S, m[s0:s0 + T]) == 1).astype(np.float64)
            st = mc.ckg_case_stats(Cn, C, pairs, unordered=False)
            pv = S47.granger_bh_pvals(C, Cn, PAIRS48)
        else:
            sl = {"full": slice(0, T), "h1": slice(0, T // 2), "h2": slice(T // 2, T)}[kind]
            st = mc.ckg_case_stats(C[sl], C[sl], pairs, unordered=True)
            pv = S47.granger_bh_pvals(C[sl], C[sl], PAIRS48)
        for k in out:
            out[k].append(st[k])
        gbh_p.append(np.array([pv[p][0] if p in pv else 1.0 for p in PAIRS48], np.float32))
        gbh_sign.append(np.array([pv[p][1] if p in pv else 0 for p in PAIRS48], np.int8))
    res = {k: np.stack(v) for k, v in out.items()}
    res["gbh_p"] = np.stack(gbh_p)
    res["gbh_sign"] = np.stack(gbh_sign)
    return variant, np.asarray(cases), res


def stage_perpatient(args, cases: np.ndarray) -> None:
    jobs = []
    chunks = [cases[i:i + 64] for i in range(0, len(cases), 64)]
    for v in ALL_VARIANTS:
        if (OUT / "cache" / f"pp_{vname(v)}.npz").exists():
            continue
        for ch in chunks:
            jobs.append((v, ch))
    log.info("perpatient: %d jobs (%d variants x %d chunks)", len(jobs), len(set(j[0] for j in jobs)), len(chunks))
    acc: dict = {}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(TE / "cache/cohort.npz"),)) as ex:
        for i, (v, cs, res) in enumerate(ex.map(pp_task, jobs, chunksize=1)):
            acc.setdefault(v, []).append((cs, res))
            if len(acc[v]) == len(chunks):
                parts = sorted(acc.pop(v), key=lambda t: t[0][0])
                arrs = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
                arrs["cases"] = np.concatenate([p[0] for p in parts])
                arrs["pairs"] = np.array(PAIRS48 if v[0] in ("nullT", "nullS") else PAIRS56)
                np.savez_compressed(OUT / "cache" / f"pp_{vname(v)}.npz", **arrs)
            if i % 50 == 0:
                log.info("  perpatient job %d/%d (%.0fs)", i + 1, len(jobs), time.time() - t0)
    log.info("perpatient done (%.0fs)", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage pcmci (rung 2)
# ---------------------------------------------------------------------------


def pcmci_task(job: tuple) -> tuple:
    variant, cases = job
    co = _CO
    kind, rep = variant
    null = kind in ("nullT", "nullS")
    m = S47.null_map(kind, rep) if null else None
    P = []
    for c in cases:
        s0, T = int(co.case_start[c]), int(co.T[c])
        C = (co.S[s0:s0 + T] == 1).astype(np.float64)
        if null:
            Cn = (tt.apply_map_S(co.S, m[s0:s0 + T]) == 1).astype(np.float64)
            M = C.copy()
            M[:, G1] = Cn[:, G1]
            P.append(S47.pcmci_pvals(M))
        else:
            sl = {"full": slice(0, T), "h1": slice(0, T // 2), "h2": slice(T // 2, T)}[kind]
            P.append(S47.pcmci_pvals(C[sl]))
    return variant, np.asarray(cases), np.stack(P).astype(np.float32)


def stage_pcmci(args, cases: np.ndarray) -> None:
    rng = np.random.default_rng(1)  # script 47: pcmci_cases = rng(1).choice(n_cases, 400)
    co_n = len(np.load(TE / "cache/cohort.npz", allow_pickle=True)["T"])
    pc = np.array(sorted(rng.choice(co_n, size=min(args.pcmci_cases, co_n), replace=False).tolist()))
    pc = pc[np.isin(pc, cases)]
    variants = REAL_VARIANTS + [(k, r) for k in ("nullT", "nullS") for r in range(args.pcmci_reps)]
    chunks = [pc[i:i + 25] for i in range(0, len(pc), 25)]
    jobs = [(v, ch) for v in variants if not (OUT / "cache" / f"pcmci_{vname(v)}.npz").exists() for ch in chunks]
    log.info("pcmci: %d cases, %d jobs", len(pc), len(jobs))
    acc: dict = {}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(TE / "cache/cohort.npz"),)) as ex:
        for i, (v, cs, P) in enumerate(ex.map(pcmci_task, jobs, chunksize=1)):
            acc.setdefault(v, []).append((cs, P))
            if len(acc[v]) == len(chunks):
                parts = sorted(acc.pop(v), key=lambda t: t[0][0])
                np.savez_compressed(OUT / "cache" / f"pcmci_{vname(v)}.npz", cases=np.concatenate([p[0] for p in parts]),
                                    P=np.concatenate([p[1] for p in parts]))
            if i % 20 == 0:
                log.info("  pcmci job %d/%d (%.0fs)", i + 1, len(jobs), time.time() - t0)
    log.info("pcmci done (%.0fs)", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage pooled (rungs 4, 5, 7)
# ---------------------------------------------------------------------------


def pooled_task(job: tuple) -> tuple:
    variant, family, cases, adjust = job
    co_full = _CO
    S_src, chg_src = mapped_sources(co_full, variant)
    mask = half_mask(co_full, variant)
    if len(cases) < co_full.n_cases:
        sub = np.isin(co_full.case, cases)
        mask = sub if mask is None else (mask & sub)
    co = mc.sub_cohort(co_full, mask)
    if mask is not None:
        S_src, chg_src = S_src[mask], chg_src[mask]
    Z = None
    if adjust:  # the phase columns of the hazard model (tab:rel-families), on the full-cohort rows
        rows_full = np.arange(co_full.n) if mask is None else np.nonzero(mask)[0]
        pc, _ = tt._phase_cols(co_full, rows_full)
        Z = np.column_stack([np.asarray(c, dtype=np.float64) for c in pc])
    W = tt.bootstrap_weights(co.n_cases, 500, seed=7)
    lags = np.arange(mc.LAG_WINDOW[family][0], mc.LAG_WINDOW[family][1] + 1)
    rows, hz_p, hz_tab = [], [], []
    t0 = time.time()
    for fam, src, out in CANDS:
        if fam != family:
            continue
        x = mc.source_series(fam, src, S_src, chg_src)
        y = mc.target_series(fam, out, co.S, co.chg)
        expo = mc.exposure_indicator(fam, src, S_src, chg_src, co.cs)
        nev = float(np.sum(y[expo]))
        npat = int(len(np.unique(co.case[expo & (y > 0)])))
        testable = nev >= tt.MIN_EXPOSED_EVENTS and npat >= tt.MIN_EVENT_PATIENTS
        row = {"key": mc.key(fam, src, out), "family": fam, "source": src, "outcome": out,
               "n_exposed_events_series": nev, "n_event_patients_series": npat, "testable_series": bool(testable)}
        if adjust:
            row["granger"] = mc.granger_fe(y, x, co, mc.SRC_BINS[fam], W, expo, Z=Z)
            rows.append(row)
            continue
        N, Dx, Dy = mc.xcorr_suff(x, y, co, lags)
        xr = mc.xcorr_test(N, Dx, Dy, W, lags)
        if not testable:
            xr["p"] = 1.0
        gr = mc.granger_fe(y, x, co, mc.SRC_BINS[fam], W, expo)
        tab = mc.hazard_tables(fam, src, out, co, S_src, chg_src)
        hz_tab.append(tab.astype(np.int32))
        hz_p.append(mc.gtest_p(tab).astype(np.float32))
        rows.append({**row, "xcorr": xr, "granger": gr})
    if adjust:
        return variant, family, rows, None, None, time.time() - t0
    return variant, family, rows, np.stack(hz_p, axis=1), np.stack(hz_tab, axis=1), time.time() - t0


def stage_pooled_adj(args, cases: np.ndarray) -> None:
    """Rung 5+phase: the pooled FE Granger of rungs 4-5 with the hazard model's phase covariates."""
    jobs = [(v, fam, cases, True) for v in ALL_VARIANTS for fam in mc.FAMILIES
            if not (OUT / "cache" / f"pooledadj_{vname(v)}_{fam}.json").exists()]
    log.info("pooled_adj: %d jobs", len(jobs))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(TE / "cache/cohort.npz"),)) as ex:
        for i, (v, fam, rows, _, _, sec) in enumerate(ex.map(pooled_task, jobs, chunksize=1)):
            (OUT / "cache" / f"pooledadj_{vname(v)}_{fam}.json").write_text(json.dumps({"variant": v, "family": fam, "sec": sec,
                                                                                       "rows": rows}, default=float))
            log.info("  pooled_adj %s %s done in %.0fs (%d/%d, %.0fs elapsed)", vname(v), fam, sec, i + 1, len(jobs), time.time() - t0)
    log.info("pooled_adj done (%.0fs)", time.time() - t0)


def stage_pooled(args, cases: np.ndarray) -> None:
    jobs = [(v, fam, cases, False) for v in ALL_VARIANTS for fam in mc.FAMILIES
            if not (OUT / "cache" / f"pooled_{vname(v)}_{fam}.json").exists()]
    log.info("pooled: %d jobs", len(jobs))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(TE / "cache/cohort.npz"),)) as ex:
        for i, (v, fam, rows, hz_p, hz_tab, sec) in enumerate(ex.map(pooled_task, jobs, chunksize=1)):
            (OUT / "cache" / f"pooled_{vname(v)}_{fam}.json").write_text(json.dumps({"variant": v, "family": fam, "sec": sec,
                                                                                    "rows": rows}, default=float))
            np.savez_compressed(OUT / "cache" / f"hz_{vname(v)}_{fam}.npz", p=hz_p, tab=hz_tab,
                                keys=np.array([r["key"] for r in rows]), cases=cases)
            log.info("  pooled %s %s done in %.0fs (%d/%d, %.0fs elapsed)", vname(v), fam, sec, i + 1, len(jobs), time.time() - t0)
    log.info("pooled done (%.0fs)", time.time() - t0)


# ---------------------------------------------------------------------------
# Stage instances
# ---------------------------------------------------------------------------


def presence_matrix(co: tt.Cohort, cases: np.ndarray) -> np.ndarray:
    """(n_cases, 176) presence of both endpoints (script 47 ``present_*`` / v_cached ``presence``)."""
    n = co.n_cases
    ep = np.zeros((n, 8), bool)
    on = np.zeros((n, 8), bool)
    ch = np.zeros((n, 4, 2), bool)
    for c in cases:
        s0, T = int(co.case_start[c]), int(co.T[c])
        if T == 0:
            continue
        S = co.S[s0:s0 + T]
        for k in range(8):
            ep[c, k] = bool((S[:, k] == 1).any())
            on[c, k] = len(kv.valid_onsets(S[:, k])) > 0
        ch[c] = co.chg[s0:s0 + T].any(axis=0)
    P = np.zeros((n, len(CANDS)), bool)
    for j, (fam, src, out) in enumerate(CANDS):
        if fam == "association":
            P[:, j] = ep[:, tt.CONCEPTS.index(src)] & on[:, tt.CONCEPTS.index(out)]
        elif fam == "policy":
            ai, di = mc.action_index(out)
            P[:, j] = ep[:, tt.CONCEPTS.index(src)] & ch[:, ai, di]
        else:
            ai, di = mc.action_index(src)
            P[:, j] = ch[:, ai, di] & on[:, tt.CONCEPTS.index(out)]
    # CKG presence: both concepts have an episode in the segment (48 pairs)
    Pv1 = np.zeros((n, len(PAIRS48)), bool)
    for j, (a, b) in enumerate(PAIRS48):
        Pv1[:, j] = ep[:, a] & ep[:, b]
    return P, Pv1


def inst_task(job: tuple) -> tuple:
    variant, cases = job
    co_full = _CO
    S_src, chg_src = mapped_sources(co_full, variant)
    mask = half_mask(co_full, variant)
    co = mc.sub_cohort(co_full, mask)
    if mask is not None:
        S_src, chg_src = S_src[mask], chg_src[mask]
    t0 = time.time()
    inst, node = mc.instance_matrix(co, CANDS, S_src, chg_src, _CALIB, cases=cases)
    P, Pv1 = presence_matrix(co, cases)
    return variant, inst, node, P, Pv1, time.time() - t0


def stage_instances(args, cases: np.ndarray) -> None:
    jobs = [(v, cases) for v in ALL_VARIANTS if not (OUT / "cache" / f"inst_{vname(v)}.npz").exists()]
    log.info("instances: %d jobs", len(jobs))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(TE / "cache/cohort.npz"),)) as ex:
        for i, (v, inst, node, P, Pv1, sec) in enumerate(ex.map(inst_task, jobs, chunksize=1)):
            np.savez_compressed(OUT / "cache" / f"inst_{vname(v)}.npz", inst=inst, node=node, presence=P, presence_v1=Pv1,
                                cases=cases, keys=np.array(KEYS))
            log.info("  instances %s done in %.0fs (%d/%d)", vname(v), sec, i + 1, len(jobs))
    log.info("instances done (%.0fs)", time.time() - t0)


# ---------------------------------------------------------------------------
# Loading helpers for summary / validate
# ---------------------------------------------------------------------------


def load_fits() -> list[dict]:
    return [json.loads(l) for l in (TE / "cache/fits.jsonl").read_text().splitlines() if l.strip()]


def ours_tables(fits: list[dict]) -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
    """key -> (p_omega, z_omega^2) per variant from the cached hazard fits (script 47 rule-1 inputs)."""
    tab: dict = {}
    for r in fits:
        v = (r["variant"], r["rep"])
        for e in r["edges"]:
            tab.setdefault(v, {})[mc.key(r["family"], e["source"], r["outcome"])] = (e["p_omega"], e["z_omega"] ** 2)
    return tab


def pooled_tables(prefix: str = "pooled") -> dict[str, dict[tuple[str, int], dict[str, tuple[float, float]]]]:
    """Statistic tables for 4x (Bonferroni xcorr p, max z^2), 4gF (joint Wald p, chi2), 4gw (p_omega, z^2);
    ``prefix='pooledadj'`` loads the phase-adjusted Granger fits (no xcorr)."""
    tabs: dict = {"x": {}, "gF": {}, "gw": {}, "rows": {}}
    for v in ALL_VARIANTS:
        for fam in mc.FAMILIES:
            p = OUT / "cache" / f"{prefix}_{vname(v)}_{fam}.json"
            if not p.exists():
                continue
            for r in json.loads(p.read_text())["rows"]:
                k = r["key"]
                if "xcorr" in r:
                    tabs["x"].setdefault(v, {})[k] = (r["xcorr"]["p"], r["xcorr"]["stat"])
                tabs["gF"].setdefault(v, {})[k] = (r["granger"]["p_joint"], r["granger"]["chi2"])
                tabs["gw"].setdefault(v, {})[k] = (r["granger"]["p_omega"], r["granger"]["stat"])
                tabs["rows"].setdefault(v, {})[k] = r
    return tabs


def load_inst(v: tuple[str, int]) -> dict | None:
    p = OUT / "cache" / f"inst_{vname(v)}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_pp(v: tuple[str, int]) -> dict | None:
    p = OUT / "cache" / f"pp_{vname(v)}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_hz(v: tuple[str, int]) -> tuple[np.ndarray, list[str]] | None:
    ps, ks = [], []
    for fam in mc.FAMILIES:
        p = OUT / "cache" / f"hz_{vname(v)}_{fam}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        ps.append(z["p"])
        ks += list(z["keys"])
    return np.concatenate(ps, axis=1), ks


def fe_from_keys(inst: dict, keys: set[str], cases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cols = [KEY_IDX[k] for k in keys]
    if not cols:
        return np.zeros(len(cases)), np.zeros(len(cases))
    return inst["inst"][cases][:, cols].sum(axis=1).astype(float), inst["node"][cases][:, cols].sum(axis=1).astype(float)


def fam_counts(keys: set[str]) -> dict[str, int]:
    return {f: sum(1 for k in keys if k.startswith(f + "|")) for f in mc.FAMILIES}


# ---------------------------------------------------------------------------
# Stage validate
# ---------------------------------------------------------------------------


def stage_validate(args, cases: np.ndarray) -> dict:
    S50 = load_script("s50", ROOT / "scripts/50_build_kgs_v3.py")
    _init(str(TE / "cache/cohort.npz"))
    co = _CO
    fits = load_fits()
    out: dict = {}
    # ---- rung 6 through our code path vs script 50 null_admissions and the published CSV
    tabs = ours_tables(fits)
    adm50 = S50.null_admissions(fits, N_NULL, Q)
    mism = []
    for v in NULL_VARIANTS:
        mine = mc.admit(tabs, v, Q, True, NULL_VARIANTS)
        ref = {mc.key(a["family"], a["source"], a["outcome"]) for a in adm50[v]}
        if mine != ref:
            mism.append((v, sorted(mine), sorted(ref)))
    out["rung6_null_admissions_equal_script50"] = not mism
    out["rung6_null_admission_mismatches"] = mism
    res47 = json.loads((TE / "results.json").read_text())
    pub_adm = {k for k, e in res47["pooled_edges_full"].items() if e.get("admitted")}
    mine_full = mc.admit(tabs, ("full", 0), Q, True, NULL_VARIANTS)
    out["rung6_real_admitted_equal_results_json"] = mine_full == pub_adm
    out["rung6_real_admitted_n"] = {"mine": fam_counts(mine_full), "published": fam_counts(pub_adm)}
    # instantiate the null admissions with our instance code (admitted relations only) and compare per rep
    csv = pd.read_csv(V2 / "null_replicates_vitaldb.csv")
    spacing = None
    rows = []
    for v in NULL_VARIANTS:
        keys = mc.admit(tabs, v, Q, True, NULL_VARIANTS)
        ref = csv[(csv["null"] == v[0]) & (csv["rep"] == v[1])].iloc[0]
        if not keys:
            rows.append({"variant": vname(v), "n_adm": 0, "mine_rel": 0.0, "pub_rel": float(ref.false_rel_per_graph),
                         "mine_node": 0.0, "pub_node": float(ref.false_node_edges_per_graph)})
            continue
        S_src, chg_src = mapped_sources(co, v)
        cands = [mc.parse_key(k) for k in sorted(keys)]
        inst, node = mc.instance_matrix(co, cands, S_src, chg_src, _CALIB)
        rows.append({"variant": vname(v), "n_adm": len(keys), "mine_rel": float(inst.sum(1).mean()),
                     "pub_rel": float(ref.false_rel_per_graph), "mine_node": float(node.sum(1).mean()),
                     "pub_node": float(ref.false_node_edges_per_graph), "mine_any": float((inst.sum(1) > 0).mean()),
                     "pub_any": float(ref.frac_graphs_any)})
    df = pd.DataFrame(rows)
    out["rung6_per_replicate"] = rows
    out["rung6_max_abs_diff_rel"] = float((df.mine_rel - df.pub_rel).abs().max())
    out["rung6_max_abs_diff_node"] = float((df.mine_node - df.pub_node).abs().max())
    for kind in ("nullT", "nullS"):
        per = [np.array([r["mine_rel"]]) for r in rows if r["variant"].startswith(kind)]
        means = np.array([r["mine_rel"] for r in rows if r["variant"].startswith(kind)])
        rng = np.random.default_rng(0)
        bm = means[rng.integers(0, len(means), size=(2000, len(means)))].mean(axis=1)
        out[f"rung6_{kind}_mean"] = float(means.mean())
        out[f"rung6_{kind}_ci_replicate_bootstrap"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    log.info("rung 6: admissions equal script 50: %s; real admitted equal results.json: %s; max |diff| rel %.3g node %.3g; "
             "nullT %.4f %s nullS %.4f %s", out["rung6_null_admissions_equal_script50"],
             out["rung6_real_admitted_equal_results_json"], out["rung6_max_abs_diff_rel"], out["rung6_max_abs_diff_node"],
             out["rung6_nullT_mean"], out["rung6_nullT_ci_replicate_bootstrap"], out["rung6_nullS_mean"],
             out["rung6_nullS_ci_replicate_bootstrap"])
    # ---- per-patient replicate-0 and real-segment checks against per_patient.jsonl
    pp = {}
    for line in (TE / "cache/per_patient.jsonl").read_text().splitlines():
        r = json.loads(line)
        pp[r["case"]] = r
    fam_set = set(PAIRS48)
    checks = {}
    for v, seg in ((("full", 0), "full"), (("h1", 0), "h1"), (("h2", 0), "h2"), (("nullT", 0), "nullT"), (("nullS", 0), "nullS")):
        z = load_pp(v)
        if z is None:
            continue
        pairs = [tuple(p) for p in z["pairs"]]
        e = mc.ckg_edges(z)
        n_eq = {"v1x": 0, "v1g": 0, "gbh": 0, "n": 0}
        fe = {"v1x": [], "v1g": [], "v1": [], "gbh": [], "v1j": [], "v1_union_j": []}
        for i, c in enumerate(z["cases"]):
            r = pp[int(c)]
            n_eq["n"] += 1
            mx = {pairs[j] for j in np.nonzero(e["xcorr"][i])[0]}
            mg = {pairs[j] for j in np.nonzero(e["granger"][i])[0]}
            rx = {tuple(x) for x in r[f"v1x_{seg}"]}
            rg = {tuple(x) for x in r[f"v1g_{seg}"]}
            n_eq["v1x"] += mx == rx
            n_eq["v1g"] += mg == rg
            lst = r.get(f"gbh_{seg}", [])
            ref_p = {(x[0], x[1]): x[2] for x in lst}
            mine_p = {p: float(z["gbh_p"][i, j]) for j, p in enumerate(PAIRS48)}
            n_eq["gbh"] += all(abs(mine_p[p] - ref_p.get(p, 1.0)) < 1e-6 for p in PAIRS48)
            fe["v1x"].append(len(mx & fam_set))
            fe["v1g"].append(len(mg & fam_set))
            fe["v1"].append(len((mx | mg) & fam_set))
            rej = tt.bh(z["gbh_p"][i], Q)
            fe["gbh"].append(int(rej.sum()))
            mj = {pairs[j] for j in np.nonzero(e["jaccard"][i])[0]}
            fe["v1j"].append(len(mj & fam_set))
            fe["v1_union_j"].append(len((mx | mg | mj) & fam_set))
        checks[seg] = {"n_cases": n_eq["n"], "v1x_equal": n_eq["v1x"], "v1g_equal": n_eq["v1g"], "gbh_p_equal": n_eq["gbh"],
                       "mean_edges_48": {k: float(np.mean(v_)) for k, v_ in fe.items()}}
        log.info("per-patient %s: %s", seg, checks[seg])
    out["per_patient_checks"] = checks
    out["published_per_patient_null_rates"] = res47["per_patient_null_rates"]
    # ---- CKG union on replicates 0-4 of the 300-case seed-7 subset (script 50 ``v1_null_multi``: 12.63 / 10.18)
    sub = np.array(sorted(np.random.default_rng(7).choice(co.n_cases, size=300, replace=False).tolist()))
    multi = {}
    for kind in ("nullT", "nullS"):
        per_rep = []
        for rep in range(5):
            z = load_pp((kind, rep))
            if z is None or not np.isin(sub, z["cases"]).all():
                break
            rows = np.searchsorted(z["cases"], sub)
            e = mc.ckg_edges(z)
            per_rep.append(e["union"][rows].sum(1).astype(float))
        if len(per_rep) == 5:
            per_case = np.mean(per_rep, axis=0)
            multi[kind] = {"per_rep_mean": [float(x.mean()) for x in per_rep], "mean": float(per_case.mean()),
                           "ci_case_bootstrap": list(load_script("s50", ROOT / "scripts/50_build_kgs_v3.py").boot_mean(per_case, seed=4))[1:]}
    out["ckg_union_reps0-4_seed7_300cases"] = multi
    v2 = json.loads((V2 / "results.json").read_text()).get("v1_null_multi_vitaldb")
    out["published_v1_null_multi"] = {k: v2[k] for k in ("nullT", "nullS")} if v2 else None
    if multi:
        log.info("CKG union reps 0-4, 300 cases: %s (published %s)", multi, out["published_v1_null_multi"])
    # ---- PCMCI replicate 0
    pcm = {}
    for v in (("nullT", 0), ("nullS", 0)):
        p = OUT / "cache" / f"pcmci_{vname(v)}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        cnt, eq = [], 0
        null32 = set(tt.null_pairs())
        for i, c in enumerate(z["cases"]):
            P = z["P"][i]
            r = pp[int(c)]
            ref = np.array(r[f"pcmci_{v[0]}"], np.float32) if f"pcmci_{v[0]}" in r else None
            eq += ref is not None and np.allclose(P, ref, atol=1e-6)
            pv = np.array([P[a, b] for a, b in PAIRS48])
            rej = tt.bh(pv, Q)
            cnt.append(sum(1 for (a, b), k in zip(PAIRS48, rej) if k and (a, b) in null32))
        pcm[v[0]] = {"n_cases": len(cnt), "p_matrix_equal": int(eq), "false_edges_per_graph": float(np.mean(cnt))}
        log.info("pcmci %s: %s", v[0], pcm[v[0]])
    out["pcmci_checks"] = pcm
    (OUT / "validation.json").write_text(json.dumps(out, indent=1, default=float))
    return out


# ---------------------------------------------------------------------------
# Stage summary
# ---------------------------------------------------------------------------


def sets_from_matrix(M: np.ndarray, labels: list, cases: np.ndarray) -> list[set]:
    return [{labels[j] for j in np.nonzero(M[c])[0]} for c in cases]


def stage_summary(args, cases: np.ndarray) -> dict:
    _init(str(TE / "cache/cohort.npz"))
    co = _CO
    n = len(cases)
    res: dict = {"n_cases": int(n), "q": Q, "n_null": N_NULL, "candidates": fam_counts(set(KEYS))}
    inst = {v: load_inst(v) for v in ALL_VARIANTS}
    have_inst = all(inst[v] is not None for v in ALL_VARIANTS)
    W = tt.bootstrap_weights(n, 1000, 5)
    ladder: list[dict] = []
    reps: list[dict] = []

    def fe_block(name: str, per_rep: dict[str, list[np.ndarray]], node_rep: dict[str, list[np.ndarray]] | None,
                 real: dict, candidates: int, extra: dict | None = None) -> dict:
        row = {"rung": name, "candidates": candidates, **real}
        for kind in ("nullT", "nullS"):
            s = mc.fe_summary(per_rep[kind])
            row[f"FE_{kind}"] = s
            if node_rep is not None:
                row[f"FE_node_{kind}"] = mc.fe_summary(node_rep[kind])
        if extra:
            row.update(extra)
        ladder.append(row)
        return row

    # ---------------- rungs 1-3 from per-patient CKG statistics
    pp = {v: load_pp(v) for v in ALL_VARIANTS}
    pcmci = {}
    for v in ALL_VARIANTS:
        p = OUT / "cache" / f"pcmci_{vname(v)}.npz"
        if p.exists():
            z = np.load(p)
            pcmci[v] = {"cases": z["cases"], "P": z["P"]}
    have_pp = all(pp[v] is not None for v in ALL_VARIANTS)
    idx48_in56 = [PAIRS56.index(p) for p in PAIRS48]
    target_assoc = target_all = None
    P1v1 = P2v1 = None
    if have_inst:  # CKG presence: both concepts have an episode in the half (v_cached ``presence(r, seg, "v1")``)
        P1v1 = sets_from_matrix(inst[("h1", 0)]["presence_v1"][cases], PAIRS48, np.arange(n))
        P2v1 = sets_from_matrix(inst[("h2", 0)]["presence_v1"][cases], PAIRS48, np.arange(n))
    if have_pp:
        real_full = pp[("full", 0)]
        rows_full = np.searchsorted(real_full["cases"], cases)
        e_full = {k: v_[rows_full][:, idx48_in56] for k, v_ in mc.ckg_edges(real_full).items()}
        w_full = mc.ckg_weights(real_full)[rows_full][:, idx48_in56]
        real_counts = {k: float(v_.sum(1).mean()) for k, v_ in e_full.items()}
        gbh_full = np.stack([tt.bh(real_full["gbh_p"][i], Q) for i in rows_full])
        real_counts["gbh"] = float(gbh_full.sum(1).mean())
        # CKG real-data counts on all 56 ordered pairs (as v1 graphs carry them) for reference
        e56 = {k: v_[rows_full] for k, v_ in mc.ckg_edges(real_full).items()}
        real56 = {k: float(v_.sum(1).mean()) for k, v_ in e56.items()}
        res["ckg_real_edges_per_graph_56_pairs"] = real56
        res["ckg_real_edges_per_graph_48_pairs"] = real_counts
        comp = {"xcorr": "1a CKG cross-correlation", "granger": "1b CKG Granger-precedes", "jaccard": "1c CKG Jaccard co-occurrence",
                "union": "1 CKG (xcorr U Granger; the paper's CKG)", "union_jaccard": "1+ CKG incl. Jaccard"}
        per = {k: {"nullT": [], "nullS": []} for k in comp}
        per["gbh"] = {"nullT": [], "nullS": []}
        wnull = {}
        for v in NULL_VARIANTS:
            z = pp[v]
            rows_v = np.searchsorted(z["cases"], cases)
            e = mc.ckg_edges(z)
            for k in comp:
                per[k][v[0]].append(e[k][rows_v].sum(1).astype(float))
            per["gbh"][v[0]].append(np.array([tt.bh(z["gbh_p"][i], Q).sum() for i in rows_v], float))
            wnull[v] = mc.ckg_weights(z)[rows_v]
        for k, name in comp.items():
            fe_block(name, per[k], None, {"real_edges_per_graph": real_counts[k], "real_admitted": None, "pooling": "none",
                                          "multiplicity": "none (fixed thresholds)", "calibration": "none"}, 48)
        fe_block("2a per-patient Granger + BH", per["gbh"], None, {"real_edges_per_graph": real_counts["gbh"], "real_admitted": None,
                                                                   "pooling": "none", "multiplicity": "BH within patient", "calibration": "none"}, 48)
        # replication rungs 1-2 (48 pairs, presence = both concepts have an episode in the half)
        if have_inst:
            eh = {}
            for h in ("h1", "h2"):
                z = pp[(h, 0)]
                rows_h = np.searchsorted(z["cases"], cases)
                eh[h] = {k: v_[rows_h][:, idx48_in56] for k, v_ in mc.ckg_edges(z).items()}
                eh[h]["gbh"] = np.stack([tt.bh(z["gbh_p"][i], Q) for i in rows_h])
                eh[h]["w"] = mc.ckg_weights(z)[rows_h][:, idx48_in56]
            for k, name in list(comp.items()) + [("gbh", "2a per-patient Granger + BH")]:
                E1 = sets_from_matrix(eh["h1"][k], PAIRS48, np.arange(n))
                E2 = sets_from_matrix(eh["h2"][k], PAIRS48, np.arange(n))
                reps.append({"rung": name, **mc.replication(E1, E2, P1v1, P2v1, W)})
        res["ckg_weights_note"] = ("edge weight = normalised peak xcorr (precedes) or 1 - min p (Granger-precedes), max of both")
    # ---------------- PCMCI (rung 2b)
    if pcmci:
        null32 = set(tt.null_pairs())
        idx32 = [j for j, p in enumerate(PAIRS48) if p in null32]
        per = {"nullT": [], "nullS": []}
        pc_cases = None
        for v, d in pcmci.items():
            if v[0] not in per:
                continue
            keep = np.isin(d["cases"], cases)
            pc_cases = d["cases"][keep]
            cnt = []
            for P in d["P"][keep]:
                pv = np.array([P[a, b] for a, b in PAIRS48])
                rej = tt.bh(pv, Q)
                cnt.append(float(rej[idx32].sum()))
            per[v[0]].append(np.array(cnt))
        real = {}
        if ("full", 0) in pcmci:
            d = pcmci[("full", 0)]
            keep = np.isin(d["cases"], cases)
            real["real_edges_per_graph"] = float(np.mean([tt.bh(np.array([P[a, b] for a, b in PAIRS48]), Q).sum() for P in d["P"][keep]]))
        fe_block("2b PCMCI+ (block null, 32 null pairs)", per, None,
                 {**real, "real_admitted": None, "pooling": "none", "multiplicity": "BH within patient (48 pairs)", "calibration": "none"},
                 48, {"n_graphs": int(len(pc_cases)) if pc_cases is not None else 0,
                      "n_reps_per_kind": {k: len(v_) for k, v_ in per.items()}})
        if have_inst and ("h1", 0) in pcmci and ("h2", 0) in pcmci:
            d1, d2 = pcmci[("h1", 0)], pcmci[("h2", 0)]
            keep = np.isin(d1["cases"], cases)
            cc = d1["cases"][keep]
            pos = np.searchsorted(cases, cc)
            E1 = [{PAIRS48[j] for j in np.nonzero(tt.bh(np.array([P[a, b] for a, b in PAIRS48]), Q))[0]} for P in d1["P"][keep]]
            E2 = [{PAIRS48[j] for j in np.nonzero(tt.bh(np.array([P[a, b] for a, b in PAIRS48]), Q))[0]} for P in d2["P"][np.isin(d2["cases"], cases)]]
            Pv1 = {h: inst[(h, 0)]["presence_v1"][cases][pos] for h in ("h1", "h2")}
            P1 = sets_from_matrix(Pv1["h1"], PAIRS48, np.arange(len(cc)))
            P2 = sets_from_matrix(Pv1["h2"], PAIRS48, np.arange(len(cc)))
            reps.append({"rung": "2b PCMCI+", **mc.replication(E1, E2, P1, P2, tt.bootstrap_weights(len(cc), 1000, 5))})

    # ---------------- rungs 4-7 and 6 need instances
    if not have_inst:
        res["ladder"] = ladder
        res["replication"] = reps
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
        log.warning("instances stage incomplete: ladder has per-patient rungs only")
        return res
    P_fam = {h: inst[(h, 0)]["presence"][cases] for h in ("h1", "h2")}
    presence_sets = {h: sets_from_matrix(P_fam[h], KEYS, np.arange(n)) for h in ("h1", "h2")}

    def pooled_rung(name: str, tables: dict, calibrate: bool, meta: dict, variants_available: bool = True) -> dict:
        per = {"nullT": [], "nullS": []}
        node = {"nullT": [], "nullS": []}
        adm_null = {}
        for v in NULL_VARIANTS:
            keys = mc.admit(tables, v, Q, calibrate, NULL_VARIANTS)
            adm_null[vname(v)] = sorted(keys)
            fe, nd = fe_from_keys(inst[v], keys, cases)
            per[v[0]].append(fe)
            node[v[0]].append(nd)
        keys_full = mc.admit(tables, ("full", 0), Q, calibrate, NULL_VARIANTS)
        fe_r, nd_r = fe_from_keys(inst[("full", 0)], keys_full, cases)
        real = {"real_admitted": fam_counts(keys_full), "real_admitted_total": len(keys_full),
                "real_edges_per_graph": float(fe_r.mean()), "real_node_edges_per_graph": float(nd_r.mean()),
                "real_admitted_keys": sorted(keys_full), **meta}
        row = fe_block(name, per, node, real, 176, {"null_admissions": adm_null,
                                                    "mean_null_admitted_per_rep": {k: float(np.mean([len(adm_null[vname((k, r))]) for r in range(N_NULL)])) for k in ("nullT", "nullS")}})
        # replication: rule (i) on H1 / H2 (no half-specific nulls), instantiated in the half
        k1 = mc.admit(tables, ("h1", 0), Q, False, NULL_VARIANTS)
        k2 = mc.admit(tables, ("h2", 0), Q, False, NULL_VARIANTS)
        c1 = [KEY_IDX[k] for k in k1]
        c2 = [KEY_IDX[k] for k in k2]
        M1 = np.zeros((n, len(KEYS)), bool)
        M2 = np.zeros((n, len(KEYS)), bool)
        if c1:
            M1[:, c1] = inst[("h1", 0)]["inst"][cases][:, c1]
        if c2:
            M2[:, c2] = inst[("h2", 0)]["inst"][cases][:, c2]
        rep = {"rung": name, "n_h1_admitted": len(k1), "n_h2_admitted": len(k2)}
        for fam in mc.FAMILIES + ("all",):
            sel = [j for j, k in enumerate(KEYS) if fam == "all" or k.startswith(fam + "|")]
            E1 = sets_from_matrix(M1[:, sel], [KEYS[j] for j in sel], np.arange(n))
            E2 = sets_from_matrix(M2[:, sel], [KEYS[j] for j in sel], np.arange(n))
            P1 = [{k for k in s if fam == "all" or k.startswith(fam + "|")} for s in presence_sets["h1"]]
            P2 = [{k for k in s if fam == "all" or k.startswith(fam + "|")} for s in presence_sets["h2"]]
            rep[fam] = mc.replication(E1, E2, P1, P2, W)
        reps.append(rep)
        return row

    # rung 6 (ours) and 6a (ours, BH only)
    fits = load_fits()
    tab6 = ours_tables(fits)
    r6 = pooled_rung("6 ours (pooled FE hazard, BH, rule ii)", tab6, True,
                     {"pooling": "patient FE hazard", "multiplicity": "BH within family", "calibration": "rule (ii), 38 nulls"})
    r6a = pooled_rung("6a ours, BH only (no calibration)", tab6, False,
                      {"pooling": "patient FE hazard", "multiplicity": "BH within family", "calibration": "none"})
    # density targets from our real instances
    cols_assoc = [KEY_IDX[k] for k in r6["real_admitted_keys"] if k.startswith("association|")]
    target_assoc = float(inst[("full", 0)]["inst"][cases][:, cols_assoc].sum(1).mean()) if cols_assoc else 0.0
    target_all = r6["real_edges_per_graph"]
    res["density_targets"] = {"ours_association_instances_per_graph": target_assoc, "ours_concept_level_instances_per_graph": target_all}

    # rung 3: density-matched CKG
    if have_pp:
        for tgt_name, tgt in (("association", target_assoc), ("all_concept_level", target_all)):
            k, means = mc.choose_k(w_full, tgt, 48)
            per = {"nullT": [], "nullS": []}
            for v in NULL_VARIANTS:
                per[v[0]].append(mc.topk_mask(wnull[v], k).sum(1).astype(float))
            # block-null variant: real within-group pairs compete with the 32 cross-group null pairs
            null32 = set(tt.null_pairs())
            idx32 = np.array([j for j, p in enumerate(PAIRS48) if p in null32])
            idx16 = np.array([j for j, p in enumerate(PAIRS48) if p not in null32])
            per_mix = {"nullT": [], "nullS": []}
            for v in NULL_VARIANTS:
                wm = w_full.copy()
                wm[:, idx32] = wnull[v][:, idx32]
                per_mix[v[0]].append(mc.topk_mask(wm, k)[:, idx32].sum(1).astype(float))
            row = fe_block(f"3 density-matched CKG (top-k={k}, target {tgt_name} {tgt:.2f})", per, None,
                           {"real_edges_per_graph": means[k], "real_admitted": None, "pooling": "none",
                            "multiplicity": "none (top-k by CKG edge weight)", "calibration": "none"}, 48,
                           {"k": k, "target": tgt, "mean_edges_by_k": {str(kk): m for kk, m in means.items() if kk <= k + 2},
                            "block_null_mixed_graph": {kind: mc.fe_summary(per_mix[kind]) for kind in per_mix},
                            "block_null_note": "32 cross-group null pairs compete with the 16 real within-group pairs for the k slots"})
            # replication for top-k on halves
            k1m = mc.topk_mask(eh["h1"]["w"], k)
            k2m = mc.topk_mask(eh["h2"]["w"], k)
            E1 = sets_from_matrix(k1m, PAIRS48, np.arange(n))
            E2 = sets_from_matrix(k2m, PAIRS48, np.arange(n))
            reps.append({"rung": row["rung"], **mc.replication(E1, E2, P1v1, P2v1, W)})

    # rungs 4 and 5 from pooled CKG statistics
    tabs = pooled_tables()
    have_pooled = all(v in tabs["x"] and len(tabs["x"][v]) == 176 for v in ALL_VARIANTS)
    if have_pooled:
        meta = {"pooling": "patient FE (within-case centring / demeaning), case score bootstrap", "multiplicity": "BH within family"}
        for calib, rung, cal_name in ((False, "4", "none"), (True, "5", "rule (ii), 38 nulls")):
            rx = pooled_rung(f"{rung}x pooled xcorr peak (Bonferroni over lags) + BH" + (" + calibration" if calib else ""),
                             tabs["x"], calib, {**meta, "calibration": cal_name, "estimator": "pooled lagged xcorr"})
            rgF = pooled_rung(f"{rung}gF pooled Granger (joint Wald over source block) + BH" + (" + calibration" if calib else ""),
                              tabs["gF"], calib, {**meta, "calibration": cal_name, "estimator": "pooled FE OLS Granger, joint Wald"})
            rgw = pooled_rung(f"{rung}gw pooled Granger (1-df omega test) + BH" + (" + calibration" if calib else ""),
                              tabs["gw"], calib, {**meta, "calibration": cal_name, "estimator": "pooled FE OLS Granger, omega"})
            # union = CKG's union of components (xcorr U Granger joint Wald)
            per = {"nullT": [], "nullS": []}
            node = {"nullT": [], "nullS": []}
            for v in NULL_VARIANTS:
                keys = set(rx["null_admissions"][vname(v)]) | set(rgF["null_admissions"][vname(v)])
                fe, nd = fe_from_keys(inst[v], keys, cases)
                per[v[0]].append(fe)
                node[v[0]].append(nd)
            ku = set(rx["real_admitted_keys"]) | set(rgF["real_admitted_keys"])
            fe_r, nd_r = fe_from_keys(inst[("full", 0)], ku, cases)
            fe_block(f"{rung} pooled CKG union (xcorr U Granger-F) + BH" + (" + calibration" if calib else ""), per, node,
                     {"real_admitted": fam_counts(ku), "real_admitted_total": len(ku), "real_edges_per_graph": float(fe_r.mean()),
                      "real_node_edges_per_graph": float(nd_r.mean()), **meta, "calibration": cal_name, "estimator": "union"}, 176)
        # information: the pooled statistics on real data per candidate
        res["pooled_real_rows"] = {k: {"xcorr": r["xcorr"], "granger": {kk: vv for kk, vv in r["granger"].items()}}
                                   for k, r in tabs["rows"][("full", 0)].items()}
    # rungs 4/5 + phase adjustment: the pooled FE Granger with the hazard model's phase covariates
    tabs_adj = pooled_tables("pooledadj")
    have_adj = all(v in tabs_adj["gF"] and len(tabs_adj["gF"][v]) == 176 for v in ALL_VARIANTS)
    if have_adj:
        meta = {"pooling": "patient FE (within-case demeaning), case score bootstrap", "multiplicity": "BH within family"}
        for calib, rung, cal_name in ((False, "4", "none"), (True, "5", "rule (ii), 38 nulls")):
            pooled_rung(f"{rung}gF+phase pooled Granger (joint Wald) + phase adjustment + BH" + (" + calibration" if calib else ""),
                        tabs_adj["gF"], calib, {**meta, "calibration": cal_name, "estimator": "pooled FE OLS Granger + phase covariates, joint Wald"})
            pooled_rung(f"{rung}gw+phase pooled Granger (1-df omega) + phase adjustment + BH" + (" + calibration" if calib else ""),
                        tabs_adj["gw"], calib, {**meta, "calibration": cal_name, "estimator": "pooled FE OLS Granger + phase covariates, omega"})

    # rung 7: ours per patient (G-test) + BH within patient and family
    hz = {v: load_hz(v) for v in ALL_VARIANTS}
    if all(hz[v] is not None for v in ALL_VARIANTS):
        hkeys = hz[("full", 0)][1]
        assert hkeys == KEYS
        fam_cols = {f: [j for j, k in enumerate(KEYS) if k.startswith(f + "|")] for f in mc.FAMILIES}

        def bh_pp(P: np.ndarray) -> np.ndarray:
            M = np.zeros_like(P, dtype=bool)
            for f, cols in fam_cols.items():
                for i in range(P.shape[0]):
                    M[i, cols] = tt.bh(P[i, cols], Q)
            return M

        per = {"nullT": [], "nullS": []}
        for v in NULL_VARIANTS:
            per[v[0]].append(bh_pp(hz[v][0][cases]).sum(1).astype(float))
        Mf = bh_pp(hz[("full", 0)][0][cases])
        fe_block("7 ours per patient (2x2 G-test hazard) + BH within patient", per, None,
                 {"real_edges_per_graph": float(Mf.sum(1).mean()), "real_admitted": None,
                  "real_edges_per_graph_by_family": {f: float(Mf[:, cols].sum(1).mean()) for f, cols in fam_cols.items()},
                  "pooling": "none", "multiplicity": "BH within patient and family", "calibration": "none"}, 176)
        M1 = bh_pp(hz[("h1", 0)][0][cases])
        M2 = bh_pp(hz[("h2", 0)][0][cases])
        rep = {"rung": "7 ours per patient + BH"}
        for fam in mc.FAMILIES + ("all",):
            sel = [j for j, k in enumerate(KEYS) if fam == "all" or k.startswith(fam + "|")]
            E1 = sets_from_matrix(M1[:, sel], [KEYS[j] for j in sel], np.arange(n))
            E2 = sets_from_matrix(M2[:, sel], [KEYS[j] for j in sel], np.arange(n))
            P1 = [{k for k in s if fam == "all" or k.startswith(fam + "|")} for s in presence_sets["h1"]]
            P2 = [{k for k in s if fam == "all" or k.startswith(fam + "|")} for s in presence_sets["h2"]]
            rep[fam] = mc.replication(E1, E2, P1, P2, W)
        reps.append(rep)

    # ---------------- null calibration of the statistics: share of testable candidate tests with p < 0.05
    calib: dict = {}
    for kind in ("nullT", "nullS"):
        d: dict = {}
        po = [e["p_omega"] for r in fits if r["variant"] == kind for e in r["edges"] if e["testable"]]
        d["ours_hazard_p_omega"] = {"n": len(po), "frac_p_lt_0.05": float(np.mean(np.array(po) < 0.05)) if po else None}
        if have_pooled:
            rows_k = [r for v in NULL_VARIANTS if v[0] == kind for r in tabs["rows"][v].values() if r["granger"]["testable"]]
            for nm, f in (("pooled_xcorr_bonferroni", lambda r: r["xcorr"]["p"]), ("pooled_granger_joint_wald", lambda r: r["granger"]["p_joint"]),
                          ("pooled_granger_omega", lambda r: r["granger"]["p_omega"]), ("pooled_granger_F_iid", lambda r: r["granger"]["p_F"])):
                pv = np.array([f(r) for r in rows_k])
                d[nm] = {"n": int(len(pv)), "frac_p_lt_0.05": float(np.mean(pv < 0.05)) if len(pv) else None}
        if have_adj:
            rows_k = [r for v in NULL_VARIANTS if v[0] == kind for r in tabs_adj["rows"][v].values() if r["granger"]["testable"]]
            for nm, f in (("pooled_granger_phase_adjusted_joint_wald", lambda r: r["granger"]["p_joint"]),
                          ("pooled_granger_phase_adjusted_omega", lambda r: r["granger"]["p_omega"])):
                pv = np.array([f(r) for r in rows_k])
                d[nm] = {"n": int(len(pv)), "frac_p_lt_0.05": float(np.mean(pv < 0.05)) if len(pv) else None}
        if all(hz[v] is not None for v in ALL_VARIANTS):
            ps = []
            for v in NULL_VARIANTS:
                if v[0] != kind:
                    continue
                for fam in mc.FAMILIES:
                    z = np.load(OUT / "cache" / f"hz_{vname(v)}_{fam}.npz")
                    p_, tab = z["p"][cases], z["tab"][cases]
                    ok = (tab[:, :, 0] + tab[:, :, 1] > 0) & (tab[:, :, 0] + tab[:, :, 2] > 0)
                    ps.append(p_[ok])
            ps = np.concatenate(ps)
            d["per_patient_gtest"] = {"n": int(len(ps)), "frac_p_lt_0.05": float(np.mean(ps < 0.05))}
        if have_pp:
            gp = np.concatenate([pp[v]["gbh_p"][np.searchsorted(pp[v]["cases"], cases)].ravel() for v in NULL_VARIANTS if v[0] == kind])
            d["per_patient_granger_bh_raw_p"] = {"n": int(len(gp)), "frac_p_lt_0.05": float(np.mean(gp < 0.05))}
        calib[kind] = d
    res["null_calibration"] = calib
    res["ladder"] = ladder
    res["replication"] = reps
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_tables(res)
    log.info("summary written: %s", OUT / "results.json")
    return res


def _ci(s: dict, nd: int = 3) -> str:
    return f"{s['mean']:.{nd}f} [{s['ci_replicate'][0]:.{nd}f}, {s['ci_replicate'][1]:.{nd}f}]"


def write_tables(res: dict) -> None:
    L = ["| rung | cand. | pooling | multiplicity | calibration | real admitted (A/P/R) | real edges / graph | FE transplant [95% CI] | FE shift [95% CI] | graphs with >= 1 false (T / S) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    X = ["\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
         "Rung & Cand. & Pooling & Multiplicity & Calibration & Real edges/graph & $\\overline{\\mathrm{FE}}$ transplant & $\\overline{\\mathrm{FE}}$ shift \\\\", "\\midrule"]
    for r in res["ladder"]:
        adm = r.get("real_admitted")
        adm_s = "/".join(str(adm[f]) for f in mc.FAMILIES) if adm else "n/a"
        fT, fS = r["FE_nullT"], r["FE_nullS"]
        L.append(f"| {r['rung']} | {r['candidates']} | {r.get('pooling', '')} | {r.get('multiplicity', '')} | {r.get('calibration', '')} | {adm_s} | "
                 f"{r.get('real_edges_per_graph', float('nan')):.2f} | {_ci(fT)} | {_ci(fS)} | {fT['frac_graphs_any']:.3f} / {fS['frac_graphs_any']:.3f} |")
        X.append(f"{r['rung'].replace('_', ' ')} & {r['candidates']} & {r.get('pooling', '')} & {r.get('multiplicity', '')} & {r.get('calibration', '')} & "
                 f"{r.get('real_edges_per_graph', float('nan')):.2f} & {_ci(fT)} & {_ci(fS)} \\\\")
    X += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "ladder.md").write_text("\n".join(L) + "\n")
    (OUT / "ladder.tex").write_text("\n".join(X) + "\n")
    R = ["| rung | family | H1 edges (per patient) | Rep [95% CI] | presence-matched chance [95% CI] | Rep - chance [95% CI] |", "|---|---|---|---|---|---|"]
    for r in res["replication"]:
        fams = [k for k in r if isinstance(r[k], dict) and "replication" in r[k]]
        if fams:
            for f in fams:
                d = r[f]
                R.append(f"| {r['rung']} | {f} | {d['n_edges_h1']} ({d['edges_per_patient_h1']:.2f}) | {d['replication'][0]:.3f} [{d['replication'][1]:.3f}, {d['replication'][2]:.3f}] | "
                         f"{d['chance'][0]:.3f} [{d['chance'][1]:.3f}, {d['chance'][2]:.3f}] | {d['diff'][0]:.3f} [{d['diff'][1]:.3f}, {d['diff'][2]:.3f}] |")
        else:
            d = r
            R.append(f"| {r['rung']} | concept pairs | {d['n_edges_h1']} ({d['edges_per_patient_h1']:.2f}) | {d['replication'][0]:.3f} [{d['replication'][1]:.3f}, {d['replication'][2]:.3f}] | "
                     f"{d['chance'][0]:.3f} [{d['chance'][1]:.3f}, {d['chance'][2]:.3f}] | {d['diff'][0]:.3f} [{d['diff'][1]:.3f}, {d['diff'][2]:.3f}] |")
    (OUT / "replication.md").write_text("\n".join(R) + "\n")


# ---------------------------------------------------------------------------


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["validate", "perpatient", "pcmci", "pooled", "pooled_adj", "instances", "summary"])
    ap.add_argument("--n-cases", type=int, default=0, help="random subset for smoke tests (0 = all 3,442)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pcmci-cases", type=int, default=400)
    ap.add_argument("--pcmci-reps", type=int, default=N_NULL)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "run.log"), logging.StreamHandler()])
    log.info("stage %s argv %s", args.stage, sys.argv)
    n_cases = len(np.load(TE / "cache/cohort.npz", allow_pickle=True)["T"])
    cases = case_subset(n_cases, args.n_cases)
    t0 = time.time()
    {"validate": stage_validate, "perpatient": stage_perpatient, "pcmci": stage_pcmci, "pooled": stage_pooled,
     "pooled_adj": stage_pooled_adj, "instances": stage_instances, "summary": stage_summary}[args.stage](args, cases)
    log.info("stage %s finished in %.0fs", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
