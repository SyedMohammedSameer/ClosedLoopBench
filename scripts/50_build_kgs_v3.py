#!/usr/bin/env python
"""
Track A: CLBench v2 patient graphs (schema 3.0) for VitalDB (3,442 cases) and MIMIC (167 v1 graphs),
and the pre-registered graph-level evaluation (runs/iclr/graphs_v2/protocol.md).

Stages
  R  relation registry (wave-3 admitted relations, identification status)
  B  VitalDB graphs (+ v1 recount inputs, presence coverage, nc_random instantiation flags)
  V  v1 recount check against kg_construct.build_patient_kg on a sample
  N  per-patient false edges under the 38 wave-3 null replicates (VitalDB and MIMIC-51)
  M  MIMIC graphs from the v1 graph files
  A  pointer audit, case studies, FastAPI loader check -> results.json, report.md
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from clbench import kg_v2 as kv  # noqa: E402
from clbench.edges import typed_temporal as tt  # noqa: E402
from clbench.persistence import CalibrationConfig  # noqa: E402

OUT = ROOT / "runs/iclr/graphs_v2"
TE = ROOT / "runs/iclr/typed_edges/confirmatory"
TM = ROOT / "runs/iclr/typed_edges/mimic_external"
CL = ROOT / "runs/iclr/closed_loop_v2"
NUM_DIR = ROOT / "data/processed_vitaldb/numerics"
EV_DIR = ROOT / "data/processed_vitaldb/events"
CALIB = ROOT / "checkpoints/cbm/persistence_calibration.json"
V1_KGS = ROOT / "data/kgs"
log = logging.getLogger("kg_v3")

_S47 = None


def s47():
    global _S47
    if _S47 is None:
        spec = importlib.util.spec_from_file_location("s47", ROOT / "scripts/47_typed_temporal_edges.py")
        _S47 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_S47)
    return _S47


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def boot_mean(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def registry():
    rels = kv.load_concept_relations(TE / "results.json", TE / "cache/fits.jsonl", tag="vitaldb")
    vrels = kv.load_vital_relations(
        {"30s": CL / "30s_all/typed_edges.csv", "2s": CL / "2s_fine/typed_edges.csv"},
        {"30s": CL / "verify/typed_admission_by_null_30s.csv", "2s": CL / "verify/typed_admission_by_null_2s.csv"})
    mrels = kv.load_concept_relations(TM / "results.json", TM / "cache/fits.jsonl", tag="mimic")
    return rels, vrels, mrels


def null_policy_relations():
    """Generic vital-level policy relations for the nc_random secondary null (P01, P04, P05)."""
    from clbench.eval import closed_loop_bench as clb

    out = {}
    for R in clb.RELATIONS:
        if R.rid in ("P01", "P04", "P05"):
            out[R.rid] = kv.PopulationRelation(
                relation_id=f"null:{R.rid}", edge_type="policy", level="vital", source=R.vital,
                target=f"{R.action}_{'up' if R.event_dir > 0 else 'down'}", source_kind="variable",
                target_kind="action", action=R.action, direction="up" if R.event_dir > 0 else "down",
                sign=1, estimand="null", estimate=0.0, ci_lo=0.0, ci_hi=0.0, horizon_sec=R.horizon_s,
                identification_status="not identified", identification_basis="null", source_file="null")
    return out


# ---------------------------------------------------------------------------
# Stage B: VitalDB graphs (worker)
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_b(rel_json: str, calib_path: str, out_dir: str):
    _W["rels"] = [kv.PopulationRelation.model_validate(r) for r in json.loads(rel_json)]
    _W["calib"] = CalibrationConfig.load(Path(calib_path))
    _W["out"] = Path(out_dir)
    _W["nullrel"] = null_policy_relations()


def build_task(job: dict) -> dict:
    from clbench.edges.co_evidence import co_evidence_relation, is_co_evidence_edge
    from clbench.eval import closed_loop_bench as clb

    c, pid, S, chg = job["c"], job["pid"], job["S"], job["chg"]
    rels, calib = _W["rels"], _W["calib"]
    num = pd.read_parquet(NUM_DIR / f"{pid}.parquet")
    evp = EV_DIR / f"{pid}.parquet"
    ev = pd.read_parquet(evp) if evp.exists() else pd.DataFrame(
        columns=["patient_id", "t_sec", "agent", "kind", "track", "value_before", "value_after"])
    # consistency of the re-derived arrays with the wave-3 cohort
    d = s47()._load_case(pid)
    same_S = bool(d is not None and d["S"].shape == S.shape and np.array_equal(d["S"], S))
    same_chg = bool(d is not None and d["chg"].shape == chg.shape and np.array_equal(d["chg"], chg))
    ws = num["window_start_sec"].to_numpy(float)
    vit = {v: num[v].to_numpy(float) for v in ("hr", "map", "etco2")}
    vit["etco2"] = np.where((vit["etco2"] >= 15) & (vit["etco2"] <= 100), vit["etco2"], np.nan)
    case = kv.CaseInput(pid=pid, S=S, ws=ws, chg=chg, events=ev.to_dict("records"), vitals=vit)
    kg = kv.build_case_graph(case, rels, calib, cohort="vitaldb",
                             metadata={"source": "wave-3 cohort arrays", "step_sec": 15.0, "window_sec": 30.0})
    errs = kg.validate_refs()
    js = kg.model_dump_json()
    with gzip.open(_W["out"] / f"{pid}.json.gz", "wt", encoding="utf-8") as f:
        f.write(js)
    cnt = kv.graph_counts(kg)
    # presence-rule instances (wave-3 typed_inst) of concept-level relations vs pointer instances
    inst = {r.relation_id for r in kg.relations}
    pres = {r.relation_id: kv.presence_instances(r, S, chg) for r in rels if r.level == "concept"}
    # v1 recount inputs: tier node count per concept (kg_construct), co-evidence pairs
    tiers = kv.case_tiers(S, calib)
    ntier = {k: len(set(v)) for k, v in tiers.items()}
    C = (S == 1).astype(float)
    co = []
    for i in range(8):
        for j in range(i + 1, 8):
            ci, cj = tt.CONCEPTS[i], tt.CONCEPTS[j]
            if ntier[ci] and ntier[cj]:
                th_i = kv.thresholds_for(ci, calib).activation_threshold
                th_j = kv.thresholds_for(cj, calib).activation_threshold
                if is_co_evidence_edge(co_evidence_relation(C[:, i], C[:, j], th_i, th_j)):
                    co.append((i, j))
    # nc_random secondary null: instantiation of P01/P04/P05 on a seeded random-time stream
    cd = clb.case_from_frames(pid, num, ev, 30.0)
    rnd = clb.nc_random_times(cd, np.random.default_rng(5000 + c))
    ncr = {}
    tc = ws + 15.0
    for rid, R in _W["nullrel"].items():
        es = rnd.events.get(R.action)
        if es is None or len(es.t) == 0:
            ncr[rid] = False
            continue
        vname = R.source
        ncr[rid] = len(kv.vital_event_pointers(R, es.t, es.before, es.after, tc, vit[vname])) > 0
    return {"c": c, "pid": pid, "counts": cnt, "errs": len(errs), "same_S": same_S, "same_chg": same_chg,
            "inst": sorted(inst), "presence": [k for k, v in pres.items() if v], "ntier": ntier, "co": co,
            "ncr": ncr, "bytes": len(js)}


# ---------------------------------------------------------------------------
# Stage N: nulls
# ---------------------------------------------------------------------------


def null_admissions(fits: list[dict], n_null: int, q: float = 0.05) -> dict[tuple[str, int], list[dict]]:
    """Wave-3 full rule on each null replicate: BH within family + z^2 above the max over the other 37
    null replicates of both kinds (leave-one-out)."""
    null_chi: dict = {}
    for r in fits:
        if r["variant"] in ("nullT", "nullS"):
            for e in r["edges"]:
                k = f"{r['family']}|{e['source']}->{r['outcome']}"
                null_chi.setdefault((k, r["variant"]), {})[r["rep"]] = e["z_omega"] ** 2
    out = {}
    for kind in ("nullT", "nullS"):
        for rep in range(n_null):
            adm = []
            for fam in ("association", "policy", "response"):
                rows = [(r, e) for r in fits if r["family"] == fam and r["variant"] == kind and r["rep"] == rep
                        for e in r["edges"]]
                if not rows:
                    continue
                rej = tt.bh(np.array([e["p_omega"] for _, e in rows]), q)
                for (r, e), k in zip(rows, rej):
                    if not k:
                        continue
                    key = f"{fam}|{e['source']}->{r['outcome']}"
                    oT = [v for rr, v in null_chi.get((key, "nullT"), {}).items() if not (kind == "nullT" and rr == rep)]
                    oS = [v for rr, v in null_chi.get((key, "nullS"), {}).items() if not (kind == "nullS" and rr == rep)]
                    if e["z_omega"] ** 2 > max(oT + [0]) and e["z_omega"] ** 2 > max(oS + [0]):
                        adm.append({"family": fam, "source": e["source"], "outcome": r["outcome"],
                                    "omega": e["omega"], "omega_lo": e["omega_lo"], "omega_hi": e["omega_hi"]})
            out[(kind, rep)] = adm
    return out


def null_rel(a: dict, tag: str) -> kv.PopulationRelation:
    fam, src, tgt = a["family"], a["source"], a["outcome"]
    kw: dict = {}
    if fam == "association":
        kw.update(source_kind="concept", target_kind="concept")
    elif fam == "policy":
        an, dn = tgt.rsplit("_", 1)
        kw.update(source_kind="concept", target_kind="action", action=an, direction=dn)
    else:
        an, dn = src.rsplit("_", 1)
        kw.update(source_kind="action", target_kind="concept", action=an, direction=dn)
    return kv.PopulationRelation(relation_id=f"{tag}:{fam}:{src}->{tgt}", edge_type=fam, level="concept",
                                 source=src, target=tgt, sign=int(np.sign(a["omega"])), estimand="null",
                                 estimate=a["omega"], ci_lo=a["omega_lo"], ci_hi=a["omega_hi"],
                                 identification_status="not identified", identification_basis="null replicate",
                                 source_file="fits.jsonl", **kw)


def null_counts(co: tt.Cohort, adm: dict, kind: str, rep: int, spacing, calib, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-case false relation instances and node-level edges for one null replicate."""
    n = co.n_cases
    rel_cnt = np.zeros(n)
    node_cnt = np.zeros(n)
    rels = [null_rel(a, tag) for a in adm]
    if not rels:
        return rel_cnt, node_cnt
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
        for r in rels:
            if r.edge_type == "association":
                a, b = tt.CONCEPTS.index(r.source), tt.CONCEPTS.index(r.target)
                p = kv.association_pairs(Snc[:, a], S[:, b])
                if len(p):
                    tiers_null = tiers_null or kv.case_tiers(Snc, calib)
                    tiers_real = tiers_real or kv.case_tiers(S, calib)
                    ta, tb = tiers_null[r.source], tiers_real[r.target]
                    node_cnt[c] += len({(ta[s], tb[t]) for s, t in zip(p.src, p.tgt)})
            elif r.edge_type == "policy":
                ai, di = tt.ACTIONS.index(r.action), tt.DIRS.index(r.direction)
                p = kv.policy_pairs(Snc[:, tt.CONCEPTS.index(r.source)], np.nonzero(chg[:, ai, di])[0])
                if len(p):
                    tiers_null = tiers_null or kv.case_tiers(Snc, calib)
                    node_cnt[c] += len({tiers_null[r.source][s] for s in p.src})
            else:
                ai, di = tt.ACTIONS.index(r.action), tt.DIRS.index(r.direction)
                p = kv.response_pairs(np.nonzero(chgnc[:, ai, di])[0], S[:, tt.CONCEPTS.index(r.target)])
                if len(p):
                    tiers_real = tiers_real or kv.case_tiers(S, calib)
                    node_cnt[c] += len({tiers_real[r.target][t] for t in p.tgt})
            rel_cnt[c] += len(p) > 0
    return rel_cnt, node_cnt


def run_nulls(cohort_npz: Path, fits_path: Path, n_null: int, calib, tag: str) -> dict:
    co = s47().load_cohort(str(cohort_npz))
    fits = [json.loads(x) for x in fits_path.read_text().splitlines() if x.strip()]
    t0 = time.time()
    adm = null_admissions(fits, n_null)
    need_shift = any(adm[("nullS", r)] for r in range(n_null))
    spacing = s47().case_spacing(co) if need_shift else None
    per = {}
    rows = []
    for (kind, rep), a in adm.items():
        rc, nc = null_counts(co, a, kind, rep, spacing, calib, tag)
        per[(kind, rep)] = (rc, nc)
        fam_n = {f: sum(x["family"] == f for x in a) for f in ("association", "policy", "response")}
        rows.append({"cohort": tag, "null": kind, "rep": rep, "n_admitted": len(a), **{f"adm_{k}": v for k, v in fam_n.items()},
                     "admitted": ";".join(f"{x['family']}:{x['source']}->{x['outcome']}" for x in a),
                     "false_rel_per_graph": float(rc.mean()), "false_node_edges_per_graph": float(nc.mean()),
                     "frac_graphs_any": float((rc > 0).mean())})
    summ = {}
    for kind in ("nullT", "nullS"):
        R = np.mean([per[(kind, r)][0] for r in range(n_null)], axis=0)
        N = np.mean([per[(kind, r)][1] for r in range(n_null)], axis=0)
        anyf = np.mean([(per[(kind, r)][0] > 0).mean() for r in range(n_null)])
        fam_mean = {f: float(np.mean([sum(x["family"] == f for x in adm[(kind, r)]) for r in range(n_null)]))
                    for f in ("association", "policy", "response")}
        summ[kind] = {"n_reps": n_null, "n_graphs": int(co.n_cases),
                      "false_relation_instances_per_graph": boot_mean(R, seed=1),
                      "false_node_edges_per_graph": boot_mean(N, seed=2),
                      "frac_graph_replicate_pairs_with_any": float(anyf),
                      "reps_with_any_admission": int(sum(bool(adm[(kind, r)]) for r in range(n_null))),
                      "mean_full_rule_admitted_per_rep": fam_mean,
                      "max_false_relation_instances_any_graph": float(max(per[(kind, r)][0].max() for r in range(n_null)))}
    log.info("%s nulls done in %.0fs", tag, time.time() - t0)
    return {"summary": summ, "rows": rows}


def v1_null_rep0(pp_path: Path) -> dict:
    fam = set(tt.cross_vital_pairs())
    out = {}
    cnt = {k: [] for k in ("nullT", "nullS")}
    for line in pp_path.read_text().splitlines():
        r = json.loads(line)
        for k in cnt:
            s = {tuple(x) for x in r[f"v1x_{k}"]} | {tuple(x) for x in r[f"v1g_{k}"]}
            cnt[k].append(len({e for e in s if e in fam}))
    for k, v in cnt.items():
        out[k] = {"reps": [0], "n_graphs": len(v), "false_concept_pairs_per_graph": boot_mean(np.array(v), seed=3)}
    return out


def _v1_null_task(job):
    Cn_list, C, c = job
    C = C.astype(float)
    Cn_list = [x.astype(float) for x in Cn_list]
    s = s47()
    fam = tt.cross_vital_pairs()
    res = []
    for Cn in Cn_list:
        ex, eg = s.v1_edges(Cn, C, fam, unordered=False)
        res.append(len({e for e in ex | eg if e in set(fam)}))
    return c, res


def v1_null_multi(cohort_npz: Path, n_cases: int, reps: int, workers: int, seed: int = 7) -> dict:
    """Secondary: v1 false concept pairs on replicates 0..reps-1 of both kinds, random case subset."""
    co = s47().load_cohort(str(cohort_npz))
    rng = np.random.default_rng(seed)
    cases = sorted(rng.choice(co.n_cases, size=min(n_cases, co.n_cases), replace=False).tolist())
    spacing = s47().case_spacing(co)
    maps = {("nullT", r): tt.transplant_map(co, np.random.default_rng(10_000 + r)) for r in range(reps)}
    maps.update({("nullS", r): tt.shift_map(co, np.random.default_rng(20_000 + r), spacing) for r in range(reps)})
    keys = sorted(maps)
    Sn = {k: tt.apply_map_S(co.S, m) for k, m in maps.items()}
    jobs = []
    for c in cases:
        s0, T = int(co.case_start[c]), int(co.T[c])
        C = (co.S[s0:s0 + T] == 1).astype(np.int8)
        jobs.append(([(Sn[k][s0:s0 + T] == 1).astype(np.int8) for k in keys], C, c))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_v1_null_task, jobs, chunksize=4))
    out = {"cases": len(cases), "reps": reps, "seed": seed, "runtime_sec": time.time() - t0}
    for kind in ("nullT", "nullS"):
        idx = [i for i, k in enumerate(keys) if k[0] == kind]
        per_case = np.array([[res[c][i] for i in idx] for c in cases], float)
        out[kind] = {"per_rep_mean": per_case.mean(axis=0).tolist(),
                     "rep0_mean_subset": float(per_case[:, 0].mean()),
                     "false_concept_pairs_per_graph": boot_mean(per_case.mean(axis=1), seed=4)}
    return out


# ---------------------------------------------------------------------------
# Stage M: MIMIC graphs from v1 graph files
# ---------------------------------------------------------------------------

MIMIC_CONCEPT = {("3424008", "ECG"): "tachycardia", ("48867003", "ECG"): "bradycardia",
                 ("271823003", "RESP"): "tachypnea", ("426813001", "RESP"): "bradypnea",
                 ("45007003", "ABP"): "hypotension", ("38341003", "ABP"): "hypertension"}


def mimic_graph(path: Path, rels: list[kv.PopulationRelation]) -> tuple[dict, dict]:
    from clbench.schema import KGNode

    g = json.loads(path.read_text())
    pid = g["patient_id"]
    nodes = [KGNode.model_validate(n) for n in g["nodes"]]
    # episodes = the v1 node evidence pointers (one per activation run), per record
    episodes: list[kv.Episode] = []
    runs: dict[tuple[str, str], list[tuple[float, float, str, kv.PersistenceTier]]] = {}
    for n in nodes:
        mods = {e.modality for e in n.evidence}
        c = MIMIC_CONCEPT.get((n.snomed_code, next(iter(mods)) if len(mods) == 1 else ""))
        for e in n.evidence:
            key = c or f"other:{n.snomed_code}:{e.modality}"
            runs.setdefault((e.record_id, key), []).append((e.window_start_sec, e.window_end_sec, n.node_id,
                                                           n.persistence_tier))
    ep_rows = []
    by_rec: dict[str, dict[str, list]] = {}
    for (rec, c), lst in runs.items():
        by_rec.setdefault(rec, {})[c] = sorted(lst)
    k = 0
    grid_meta = {}
    series: dict[str, tuple[float, int, dict]] = {}
    for rec, cmap in by_rec.items():
        t0 = min(r[0] for lst in cmap.values() for r in lst)
        t1 = max(r[1] for lst in cmap.values() for r in lst)
        T = int(round((t1 - kv.WINDOW_SEC - t0) / kv.STEP_SEC)) + 1
        S = {}
        for c, lst in cmap.items():
            x = np.zeros(T, np.int8)
            for s_sec, e_sec, nid, tier in lst:
                i0 = int(round((s_sec - t0) / kv.STEP_SEC))
                i1 = int(round((e_sec - kv.WINDOW_SEC - t0) / kv.STEP_SEC))
                x[max(0, i0):max(i0, i1) + 1] = 1
            S[c] = x
        series[rec] = (t0, T, S)
        grid_meta[rec] = {"t0": t0, "T": T}
        for c, lst in cmap.items():
            if c.startswith("other:"):
                continue
            ons = set(kv.valid_onsets(S[c]).tolist())
            for s_sec, e_sec, nid, tier in lst:
                i0 = int(round((s_sec - t0) / kv.STEP_SEC))
                eid = f"e{k}"
                k += 1
                episodes.append(kv.Episode(episode_id=eid, concept=c, node_id=nid, tier=tier, record_id=rec,
                                           start_step=i0, n_steps=int(round((e_sec - s_sec - kv.WINDOW_SEC) / kv.STEP_SEC)) + 1,
                                           start_sec=s_sec, end_sec=e_sec, onset_valid=i0 in ons))
                ep_rows.append((rec, c, i0, eid, nid))
    ep_at = {(rec, c, i0): (eid, nid) for rec, c, i0, eid, nid in ep_rows}
    ep_obj = {e.episode_id: e for e in episodes}
    edges, used, pres_n, inst_n = [], [], 0, 0
    for rel in rels:
        groups: dict[tuple[str, str], list] = {}
        present = False
        for rec, (t0, T, S) in series.items():
            if rel.source not in S or rel.target not in S:
                continue
            sa, sb = S[rel.source], S[rel.target]
            present = present or (bool(sa.any()) and len(kv.valid_onsets(sb)) > 0)
            p = kv.association_pairs(sa, sb)
            if not len(p):
                continue
            st_a, _ = kv.runs_of(sa)
            st_b, _ = kv.runs_of(sb)
            for s, t, lag, lab in zip(p.src, p.tgt, p.lag, p.label):
                ea = ep_at.get((rec, rel.source, int(st_a[s])))
                eb = ep_at.get((rec, rel.target, int(st_b[t])))
                if ea is None or eb is None:  # merged overlapping runs of different tiers: skip, reported
                    continue
                ea_e, eb_e = ep_obj[ea[0]], ep_obj[eb[0]]
                groups.setdefault((ea[1], eb[1]), []).append(kv.EvidencePair(
                    source_ref=ea[0], target_ref=eb[0], lag_sec=float(lag) * kv.STEP_SEC, exposure=lab,
                    source_window_sec=(ea_e.start_sec, ea_e.end_sec), target_window_sec=(eb_e.start_sec, eb_e.end_sec)))
        pres_n += present
        if not groups:
            continue
        inst_n += 1
        used.append(rel)
        for (sn, tn), ptrs in sorted(groups.items()):
            edges.append(kv.TypedEdge(edge_id=f"{pid}:{rel.relation_id}:{len(edges)}", relation_id=rel.relation_id,
                                      edge_type=rel.edge_type, source_node_id=sn, target_node_id=tn, sign=rel.sign,
                                      identification_status=rel.identification_status,
                                      population_estimate=rel.estimate, population_ci=(rel.ci_lo, rel.ci_hi),
                                      estimand=rel.estimand, pointers=ptrs))
    kg = kv.PatientKGv3(patient_id=pid, cohort="mimic", nodes=nodes, episodes=episodes, relations=used, edges=edges,
                        build_metadata={"nodes": "v1 graph nodes verbatim (kg_construct, v1 learned activations)",
                                        "grid": grid_meta, "onset_rule": "approximate: gap of >= 2 steps since the "
                                        "previous run on the record grid (unobserved = off)"})
    v1 = {"v1_nodes": len(g["nodes"]), "v1_edges": len(g["edges"]),
          "v1_edges_with_evidence": int(sum(bool(e.get("evidence_window_ids")) for e in g["edges"]))}
    for e in g["edges"]:
        v1[f"v1_{e['edge_type']}"] = v1.get(f"v1_{e['edge_type']}", 0) + 1
    return kg, {"pid": pid, **kv.graph_counts(kg), **v1, "presence_instances": pres_n, "errs": len(kg.validate_refs())}


# ---------------------------------------------------------------------------
# Stage A: pointer audit (independent re-derivation from raw parquet)
# ---------------------------------------------------------------------------

LABELS = {"tachycardia": "label_tachycardia", "bradycardia": "label_bradycardia",
          "hypotension": "label_hypotension", "hypertension": "label_hypertension",
          "tachypnea": "label_tachypnea", "bradypnea": "label_bradypnea"}


def audit_case(pid: str, out_dir: Path, rels: dict) -> dict:
    from clbench.eval import closed_loop_bench as clb

    kg = json.loads(gzip.open(out_dir / f"{pid}.json.gz", "rt").read())
    num = pd.read_parquet(NUM_DIR / f"{pid}.parquet")
    evp = EV_DIR / f"{pid}.parquet"
    ev = pd.read_parquet(evp) if evp.exists() else pd.DataFrame(columns=["t_sec", "agent", "value_before", "value_after"])
    ws = num["window_start_sec"].to_numpy(float)
    raw = {}
    for c, col in LABELS.items():
        v = pd.to_numeric(num[col], errors="coerce").to_numpy(float)
        raw[c] = np.where(np.isnan(v), -1, (v > 0.5).astype(int))
    et = num["etco2"].to_numpy(float)
    ok = np.isfinite(et) & (et >= 15) & (et <= 100)
    raw["hypercapnia"] = np.where(ok, (et > 45).astype(int), -1)
    raw["hypocapnia"] = np.where(ok, (et < 30).astype(int), -1)
    eps = {e["episode_id"]: e for e in kg["episodes"]}
    inter = kg["interventions"]
    n_ptr = n_bad = 0
    bad = []

    def ep_ok(e):
        x = raw[e["concept"]]
        s, n = e["start_step"], e["n_steps"]
        if not np.all(x[s:s + n] == 1):
            return False
        if s > 0 and x[s - 1] == 1:
            return False
        if s + n < len(x) and x[s + n] == 1:
            return False
        valid = s >= 2 and x[s - 1] == 0 and x[s - 2] == 0
        return bool(valid) == bool(e["onset_valid"]) and abs(ws[s] - e["start_sec"]) < 1e-6

    def act(ref):
        i, j = (int(q) for q in ref[3:].split("."))
        a = inter[i]["actions"][j]
        hit = ev[(np.abs(ev["t_sec"].astype(float) - a["t_sec"]) < 1e-6) & (ev["agent"] == inter[i]["agent"])]
        return inter[i]["agent"], a, len(hit) > 0

    def exposure_ok(src_e, t_step, lag0, lab):
        x = raw[src_e["concept"]]
        s = src_e["start_step"]
        okk = True
        if "state" in lab:
            okk &= s <= t_step - lag0 < s + src_e["n_steps"] and x[t_step - lag0] == 1
        for part in lab.split("+"):
            if part.startswith("onset"):
                d = t_step - s
                okk &= bool(src_e["onset_valid"]) and kv.onset_label(d) == part and d >= lag0
        return bool(okk)

    for ed in kg["edges"]:
        rel = rels.get(ed["relation_id"])
        for p in ed["pointers"]:
            n_ptr += 1
            good = True
            try:
                if rel["level"] == "concept":
                    if rel["edge_type"] == "association":
                        ea, eb = eps[p["source_ref"]], eps[p["target_ref"]]
                        good = ep_ok(ea) and ep_ok(eb) and eb["onset_valid"] and ea["concept"] == rel["source"] \
                            and eb["concept"] == rel["target"] and exposure_ok(ea, eb["start_step"], 2, p["exposure"]) \
                            and p["lag_sec"] == (eb["start_step"] - ea["start_step"]) * 15.0
                    elif rel["edge_type"] == "policy":
                        ea = eps[p["source_ref"]]
                        agent, a, hit = act(p["target_ref"])
                        i = int(np.searchsorted(ws + 15.0, a["t_sec"], side="right") - 1)
                        dirn = "up" if a["value_after"] > a["value_before"] else "down"
                        good = ep_ok(ea) and hit and dirn == rel["direction"] and agent == kv.ACTION_AGENT[rel["action"]] \
                            and exposure_ok(ea, i, 1, p["exposure"]) and a["t_sec"] < ws[i] + 30
                    else:
                        eb = eps[p["target_ref"]]
                        agent, a, hit = act(p["source_ref"])
                        j = int(np.searchsorted(ws + 15.0, a["t_sec"], side="right") - 1)
                        dirn = "up" if a["value_after"] > a["value_before"] else "down"
                        d = eb["start_step"] - j
                        good = ep_ok(eb) and eb["onset_valid"] and hit and dirn == rel["direction"] \
                            and agent == kv.ACTION_AGENT[rel["action"]] and 5 <= d <= 40 and p["lag_sec"] == d * 15.0
                else:
                    ref = p["source_ref"] if rel["edge_type"] == "response" else p["target_ref"]
                    agent, a, hit = act(ref)
                    vname = rel["target"] if rel["edge_type"] == "response" else rel["source"]
                    v = num[vname].to_numpy(float)
                    if vname == "etco2":
                        v = np.where((v >= 15) & (v <= 100), v, np.nan)
                    idx = clb.SeriesIndex(ws + 15.0, v)
                    t = a["t_sec"]
                    dirn = "up" if a["value_after"] > a["value_before"] else "down"
                    wins = ([(t - 30, t), (t + rel["horizon_sec"] - 15, t + rel["horizon_sec"] + 15)]
                            if rel["edge_type"] == "response" else [(t - 150, t - 120), (t - 30, t)])
                    good = hit and dirn == rel["direction"] and all(np.isfinite(idx.mean(np.array([lo]), np.array([hi])))[0]
                                                                      for lo, hi in wins)
                    same = ev[(ev["agent"] == agent) & (ev["t_sec"].astype(float) >= t - 180) & (ev["t_sec"].astype(float) < t)]
                    good &= len(same) == 0 or not bool((np.abs(same["value_after"] - same["value_before"]) > 0).any())
            except Exception as exc:  # noqa: BLE001
                good = False
                bad.append(f"{ed['relation_id']} {p} {exc!r}")
            if not good:
                n_bad += 1
                if len(bad) < 5:
                    bad.append(f"{ed['relation_id']} {p}")
    return {"pid": pid, "pointers": n_ptr, "invalid": n_bad, "examples": bad[:5]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--v1-multi-cases", type=int, default=300)
    ap.add_argument("--v1-multi-reps", type=int, default=5)
    ap.add_argument("--skip-v1-multi", action="store_true")
    ap.add_argument("--skip-nulls", action="store_true", help="smoke tests only")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    proto = OUT / "protocol.md"
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(OUT / "run.log")])
    proto_sha = sha(proto)
    log.info("protocol sha256 %s", proto_sha)
    t_start = time.time()
    calib = CalibrationConfig.load(CALIB)
    res: dict = {"protocol_sha256": proto_sha}

    # ---- R
    rels, vrels, mrels = registry()
    allrels = rels + vrels
    reg = pd.DataFrame([{**{k: v for k, v in r.model_dump().items() if k not in ("evidence",)},
                         "evidence": json.dumps(r.evidence, default=jdefault)} for r in allrels + mrels])
    reg.insert(0, "cohort", reg.relation_id.str.split(":").str[0].replace({"vital": "vitaldb"}))
    reg.to_csv(OUT / "relation_registry.csv", index=False)
    res["registry"] = reg.groupby(["cohort", "level", "edge_type", "identification_status"]).size().reset_index(
        name="n").to_dict("records")
    res["registry_mimic_note"] = "mimic rows come from the exploratory 51-patient MIMIC fit"
    log.info("relations: %d concept + %d vital (VitalDB), %d MIMIC", len(rels), len(vrels), len(mrels))

    # ---- B
    co = s47().load_cohort(str(TE / "cache/cohort.npz"))
    kdir = OUT / "kgs_vitaldb"
    kdir.mkdir(exist_ok=True)
    cases = list(range(co.n_cases))[: args.limit or None]
    jobs = []
    for c in cases:
        s0, T = int(co.case_start[c]), int(co.T[c])
        jobs.append({"c": c, "pid": co.ids[c], "S": co.S[s0:s0 + T], "chg": co.chg[s0:s0 + T]})
    rel_json = json.dumps([r.model_dump() for r in allrels], default=jdefault)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, initializer=_init_b, initargs=(rel_json, str(CALIB), str(kdir))) as ex:
        built = list(ex.map(build_task, jobs, chunksize=8))
    log.info("built %d VitalDB graphs in %.0fs", len(built), time.time() - t0)
    pp = {}
    for line in (TE / "cache/per_patient.jsonl").read_text().splitlines():
        r = json.loads(line)
        pp[r["case"]] = r
    rows = []
    for b in built:
        r = pp[b["c"]]
        nt = b["ntier"]
        n = [nt[c] for c in tt.CONCEPTS]
        v1x = {tuple(x) for x in r["v1x_full"]}
        v1g = {tuple(x) for x in r["v1g_full"]}
        prec = sum(n[a] * n[bb] for a, bb in v1x)
        gr = sum(n[a] * n[bb] for a, bb in v1g)
        coev = sum(2 * n[a] * n[bb] for a, bb in b["co"])
        rows.append({"pid": b["pid"], **b["counts"], "v1_nodes": sum(n), "v1_precedes": prec, "v1_follows": prec,
                     "v1_granger_precedes": gr, "v1_co_occurs": coev, "v1_edges": 2 * prec + gr + coev,
                     "presence_instances": len(b["presence"]),
                     "pointer_instances_concept": sum(1 for x in b["inst"] if x.startswith("vitaldb:")),
                     "presence_with_pointer": len(set(b["presence"]) & set(b["inst"])),
                     "pointer_without_presence": len(set(x for x in b["inst"] if x.startswith("vitaldb:")) - set(b["presence"])),
                     "same_S": b["same_S"], "same_chg": b["same_chg"], "ref_errors": b["errs"], "json_bytes": b["bytes"],
                     **{f"ncr_{k}": v for k, v in b["ncr"].items()}})
    gs = pd.DataFrame(rows)
    gs.to_csv(OUT / "graph_size_vitaldb.csv", index=False)
    res["build_vitaldb"] = {"n_graphs": len(gs), "ref_errors": int(gs.ref_errors.sum()),
                            "cases_S_mismatch": int((~gs.same_S).sum()), "cases_chg_mismatch": int((~gs.same_chg).sum()),
                            "total_json_mb_uncompressed": float(gs.json_bytes.sum() / 1e6),
                            "runtime_sec": time.time() - t0}

    # ---- V: v1 recount check on 30 random cases (seed 3) against build_patient_kg
    from clbench import kg_construct as kc
    from clbench.schema import EdgeType

    rng = np.random.default_rng(3)
    vcases = sorted(rng.choice(len(cases), size=min(30, len(cases)), replace=False).tolist())
    vrows = []
    for c in vcases:
        s0, T = int(co.case_start[c]), int(co.T[c])
        C = (co.S[s0:s0 + T] == 1).astype(float)
        wsx = np.arange(T) * 15.0
        acts = [kc.ConceptActivation(catalog=kc.ConceptCatalogEntry("NUM", cn, f"9{k:05d}", cn), probs=C[:, k],
                                     window_starts_sec=wsx, window_ends_sec=wsx + 30.0, record_id="r")
                for k, cn in enumerate(tt.CONCEPTS)]
        logging.getLogger("clbench.kg_construct").setLevel(logging.ERROR)
        kg1 = kc.build_patient_kg("p", acts, calib, window_stride_sec=15.0)
        ty = {t: sum(e.edge_type == t for e in kg1.edges) for t in EdgeType}
        g = gs.iloc[c]
        vrows.append({"case": c, "nodes_eq": len(kg1.nodes) == g.v1_nodes,
                      "precedes_eq": ty[EdgeType.PRECEDES] == g.v1_precedes,
                      "follows_eq": ty[EdgeType.FOLLOWS] == g.v1_follows,
                      "granger_eq": ty[EdgeType.GRANGER_PRECEDES] == g.v1_granger_precedes,
                      "co_eq": ty[EdgeType.CO_OCCURS] == g.v1_co_occurs, "edges_kg": len(kg1.edges),
                      "edges_count": int(g.v1_edges), "nodes_kg": len(kg1.nodes), "nodes_count": int(g.v1_nodes)})
    vdf = pd.DataFrame(vrows)
    res["v1_recount_check"] = {"n_cases": len(vdf), **{k: int(vdf[k].sum()) for k in
                                                       ("nodes_eq", "precedes_eq", "follows_eq", "granger_eq", "co_eq")},
                               "all_edges_equal": int((vdf.edges_kg == vdf.edges_count).sum())}
    log.info("v1 recount check: %s", res["v1_recount_check"])

    # ---- sizes
    def dist(df, cols):
        return {c: {"mean": float(df[c].mean()), "median": float(df[c].median()), "p90": float(df[c].quantile(0.9)),
                    "total": float(df[c].sum())} for c in cols}

    size_cols = ["nodes_concept_tier", "action_nodes", "variable_nodes", "episodes", "relation_instances",
                 "relations_policy", "relations_response", "relations_association", "relations_identified",
                 "relations_sign-unresolved", "relations_not_identified", "relations_concept_level",
                 "relations_vital_level", "edges_node_level", "edges_policy", "edges_response", "edges_association",
                 "pointers", "v1_nodes", "v1_edges", "v1_precedes", "v1_follows", "v1_granger_precedes", "v1_co_occurs"]
    res["graph_size_vitaldb"] = dist(gs, size_cols)
    res["coverage_vitaldb"] = {
        "v2_edges": int(gs.edges_node_level.sum()), "v2_edges_with_pointers": int(gs.edges_with_pointers.sum()),
        "v2_coverage": float(gs.edges_with_pointers.sum() / max(1, gs.edges_node_level.sum())),
        "v1_edges": int(gs.v1_edges.sum()), "v1_edges_with_pointers": 0,
        "presence_rule_instances": int(gs.presence_instances.sum()),
        "presence_instances_with_pointer": int(gs.presence_with_pointer.sum()),
        "frac_presence_instances_with_pointer": float(gs.presence_with_pointer.sum() / max(1, gs.presence_instances.sum())),
        "pointer_instances_without_presence": int(gs.pointer_without_presence.sum())}
    per_rel = {}
    for b in built:
        for x in b["inst"]:
            per_rel[x] = per_rel.get(x, 0) + 1
    res["patients_per_relation"] = dict(sorted(per_rel.items()))

    # ---- N
    if args.skip_nulls:
        log.warning("--skip-nulls: smoke test, E1 not computed")
        res["nulls_vitaldb"] = res["nulls_mimic51"] = {"summary": {}, "rows": []}
    else:
        res["nulls_vitaldb"] = run_nulls(TE / "cache/cohort.npz", TE / "cache/fits.jsonl", 19, calib, "vitaldb_null")
    pd.DataFrame(res["nulls_vitaldb"]["rows"]).to_csv(OUT / "null_replicates_vitaldb.csv", index=False)
    if not args.skip_nulls:
        res["nulls_mimic51"] = run_nulls(TM / "cache/cohort.npz", TM / "cache/fits.jsonl", 19, calib, "mimic_null")
    pd.DataFrame(res["nulls_mimic51"]["rows"]).to_csv(OUT / "null_replicates_mimic51.csv", index=False)
    res["v1_null_rep0_vitaldb"] = v1_null_rep0(TE / "cache/per_patient.jsonl")
    res["v1_null_rep0_mimic51"] = v1_null_rep0(TM / "cache/per_patient.jsonl")
    # vital-level: cross-patient null-vs-null admissions (0 => 0 false edges), nc_random secondary
    vl = {}
    for s in ("30s", "2s"):
        nv = pd.read_csv(CL / f"verify/null_vs_null_{s}.csv")
        cross = nv[nv["input"] != "nc_random"]
        rnd = nv[nv["input"] == "nc_random"]
        vl[s] = {"cross_pairs": int(len(cross)), "cross_admissions_total": int(cross.n_admitted.sum()),
                 "cross_false_edges_per_graph": 0.0 if int(cross.n_admitted.sum()) == 0 else None,
                 "nc_random_pairs": int(len(rnd)), "nc_random_admitted": rnd.admitted.fillna("").tolist()}
        if len(rnd):
            per = []
            for adm in rnd.admitted.fillna(""):
                ids = [x for x in str(adm).split(",") if x]
                per.append(sum(gs[f"ncr_{i}"].astype(float) for i in ids) if ids else pd.Series(np.zeros(len(gs))))
            m = np.mean(np.vstack([np.asarray(p, float) for p in per]), axis=0)
            vl[s]["nc_random_false_edges_per_graph"] = boot_mean(m, seed=5)
    res["nulls_vital_level"] = vl
    if not args.skip_v1_multi:
        res["v1_null_multi_vitaldb"] = v1_null_multi(TE / "cache/cohort.npz", args.v1_multi_cases, args.v1_multi_reps,
                                                      args.workers)
        log.info("v1 multi-rep: %s", {k: v for k, v in res["v1_null_multi_vitaldb"].items() if k in ("nullT", "nullS")})

    # ---- M
    mdir = OUT / "kgs_mimic"
    mdir.mkdir(exist_ok=True)
    mrows = []
    for p in sorted(V1_KGS.glob("*.json")):
        kg, row = mimic_graph(p, mrels)
        with gzip.open(mdir / f"{kg.patient_id}.json.gz", "wt", encoding="utf-8") as f:
            f.write(kg.model_dump_json())
        mrows.append(row)
    ms = pd.DataFrame(mrows).fillna(0)
    ms.to_csv(OUT / "graph_size_mimic.csv", index=False)
    mcols = [c for c in ["nodes_concept_tier", "episodes", "relation_instances", "relations_association",
                         "relations_identified", "relations_not_identified", "relations_sign-unresolved",
                         "edges_node_level", "pointers", "v1_nodes", "v1_edges", "v1_precedes", "v1_follows",
                         "v1_granger_precedes", "v1_co_occurs"] if c in ms]
    res["graph_size_mimic"] = dist(ms, mcols)
    res["coverage_mimic"] = {"n_graphs": len(ms), "v2_edges": int(ms.edges_node_level.sum()),
                             "v2_edges_with_pointers": int(ms.edges_with_pointers.sum()),
                             "v1_edges": int(ms.v1_edges.sum()), "v1_edges_with_evidence": int(ms.v1_edges_with_evidence.sum()),
                             "presence_instances": int(ms.presence_instances.sum()),
                             "pointer_instances": int(ms.relation_instances.sum()), "ref_errors": int(ms.errs.sum())}

    # ---- A: pointer audit
    rng = np.random.default_rng(11)
    acases = sorted(rng.choice(len(cases), size=min(100, len(cases)), replace=False).tolist())
    rel_by_id = {r.relation_id: r.model_dump() for r in allrels}
    aud = [audit_case(co.ids[c], kdir, rel_by_id) for c in acases]
    adf = pd.DataFrame(aud)
    res["pointer_audit"] = {"n_cases": len(adf), "pointers": int(adf.pointers.sum()), "invalid": int(adf.invalid.sum()),
                            "frac_valid": float(1 - adf.invalid.sum() / max(1, adf.pointers.sum())),
                            "examples": [e for x in aud for e in x["examples"]][:10]}
    log.info("pointer audit: %s", {k: v for k, v in res["pointer_audit"].items() if k != "examples"})

    # ---- case studies
    cs_dir = OUT / "case_studies"
    cs_dir.mkdir(exist_ok=True)
    vid = {r.relation_id for r in vrels}
    elig = []
    for b in built:
        kgd = None
        ids = set(b["inst"])
        if not any(x in vid for x in ids):
            continue
        kgd = json.loads(gzip.open(kdir / f"{b['pid']}.json.gz", "rt").read())
        st = {r["relation_id"]: r for r in kgd["relations"]}
        pol = any(r["edge_type"] == "policy" and r["identification_status"] == "identified" for r in st.values())
        resp = any(r["edge_type"] == "response" and r["level"] == "vital" and r["identification_status"] == "identified"
                   for r in st.values())
        asc = any(r["edge_type"] == "association" for r in st.values())
        if pol and resp and asc:
            elig.append(b["pid"])
    elig = sorted(elig)
    pick = sorted(np.random.default_rng(2027).choice(len(elig), size=min(3, len(elig)), replace=False).tolist())
    chosen = [str(elig[i]) for i in pick]
    res["case_studies"] = {"n_eligible": len(elig), "chosen": chosen}
    md = ["# CLBench v2 case studies (VitalDB)", "",
          f"Selection (pre-registered): numpy default_rng(2027), 3 of {len(elig)} eligible cases.", ""]
    for pid in chosen:
        kgd = json.loads(gzip.open(kdir / f"{pid}.json.gz", "rt").read())
        (cs_dir / f"{pid}.json").write_text(json.dumps(kgd, indent=1))
        md += case_summary(kgd)
    (cs_dir / "summary.md").write_text("\n".join(md))

    # ---- API loader check (api/ untouched; KG_DIR pointed at a scratch copy in this process only)
    res["api_check"] = api_check([cs_dir / f"{p}.json" for p in chosen],
                                 [mdir / f"{x}.json.gz" for x in ms.pid.astype(str).head(2)])

    res["runtime_sec"] = time.time() - t_start
    prov = {"protocol_sha256": proto_sha,
            "code_sha256": {p: sha(ROOT / p) for p in ("src/clbench/kg_v2.py", "scripts/50_build_kgs_v3.py",
                                                       "tests/unit/test_kg_v2.py") if (ROOT / p).exists()},
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                TE / "results.json", TE / "cache/fits.jsonl", TM / "results.json", TM / "cache/fits.jsonl",
                CL / "30s_all/typed_edges.csv", CL / "2s_fine/typed_edges.csv", CALIB)},
            "args": vars(args)}
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1))
    res["provenance"] = prov
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=jdefault))
    write_report(res)
    log.info("done in %.0fs", res["runtime_sec"])


def case_summary(kgd: dict) -> list[str]:
    rels = {r["relation_id"]: r for r in kgd["relations"]}
    npt = {}
    for e in kgd["edges"]:
        npt[e["relation_id"]] = npt.get(e["relation_id"], 0) + len(e["pointers"])
    eps = {e["episode_id"]: e for e in kgd["episodes"]}
    out = [f"## {kgd['patient_id']}", "",
           f"{len(kgd['nodes'])} concept x tier nodes ({len(kgd['episodes'])} episodes), "
           f"{len(kgd['interventions'])} action nodes, {len(kgd['variables'])} variable nodes; "
           f"{len(rels)} population relations instantiated as {len(kgd['edges'])} node-level edges.", "",
           "| relation | type | status | population estimate [95% CI] | pointers | first pointer |", "|---|---|---|---|---|---|"]
    order = {"identified": 0, "sign-unresolved": 1, "not identified": 2}
    for rid, r in sorted(rels.items(), key=lambda kv_: (order[kv_[1]["identification_status"]], kv_[1]["edge_type"], kv_[0])):
        first = next(e for e in kgd["edges"] if e["relation_id"] == rid)["pointers"][0]

        def desc(ref, win):
            if ref in eps:
                e = eps[ref]
                return f"{e['concept']} {e['tier']} episode {e['start_sec']:.0f}-{e['end_sec']:.0f} s"
            if ref.startswith("act"):
                i, j = (int(q) for q in ref[3:].split("."))
                a = kgd["interventions"][i]["actions"][j]
                return f"{kgd['interventions'][i]['agent']} {a['value_before']:g}->{a['value_after']:g} at {a['t_sec']:.0f} s"
            return f"{ref} window {win[0]:.0f}-{win[1]:.0f} s" if win else ref

        fp = f"{desc(first['source_ref'], first['source_window_sec'])} -> {desc(first['target_ref'], first['target_window_sec'])} (lag {first['lag_sec']:.0f} s, {first['exposure']})"
        name = r["evidence"].get("label") or f"{r['source']} -> {r['target']}"
        out.append(f"| {name} | {r['edge_type']} ({r['level']}) | {r['identification_status']} | "
                   f"{r['estimate']:.3g} [{r['ci_lo']:.3g}, {r['ci_hi']:.3g}] | {npt[rid]} | {fp} |")
    out.append("")
    return out


def api_check(json_paths: list[Path], gz_paths: list[Path]) -> dict:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="kgv3_api_"))
    pids = []
    for p in json_paths:
        shutil.copy(p, tmp / p.name)
        pids.append(p.stem)
    for p in gz_paths:
        (tmp / p.name.replace(".json.gz", ".json")).write_text(gzip.open(p, "rt").read())
        pids.append(p.name.replace(".json.gz", ""))
    out = {"patients": pids, "summary_nodes_match": 0, "summary_edges_match": 0, "http_200": 0, "errors": []}
    try:
        from fastapi.testclient import TestClient

        import api.data as adata
        from api.main import app

        adata.KG_DIR = tmp
        client = TestClient(app)
        for pid in pids:
            kg = json.loads((tmp / f"{pid}.json").read_text())
            s = adata._summarize_kg(kg)
            out["summary_nodes_match"] += int(s["n_nodes"] == len(kg["nodes"]))
            out["summary_edges_match"] += int(s["n_edges"] == len(kg["edges"]))
            r = client.get(f"/patients/{pid}/kg")
            out["http_200"] += int(r.status_code == 200)
            if r.status_code == 200:
                body = r.json()
                out.setdefault("returned_n_nodes", []).append(body["summary"]["n_nodes"])
                out.setdefault("returned_interventions", []).append(body["summary"]["n_interventions"])
            else:
                out["errors"].append(f"{pid}: {r.status_code} {r.text[:200]}")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(repr(exc))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def _ci(x, nd=2):
    return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]"


def write_report(res: dict) -> None:
    g, gm = res["graph_size_vitaldb"], res["graph_size_mimic"]
    cv, cm = res["coverage_vitaldb"], res["coverage_mimic"]
    nv, nm = res["nulls_vitaldb"]["summary"], res["nulls_mimic51"]["summary"]
    L = ["# CLBench v2 graphs (schema 3.0): build and graph-level evaluation", "",
         f"Protocol sha256: `{res['protocol_sha256']}`. Runtime {res['runtime_sec']:.0f} s.", "",
         "## Relation registry (population-admitted relations only)", "",
         "| cohort | level | type | identification status | n |", "|---|---|---|---|---|"]
    for r in res["registry"]:
        L.append(f"| {r['cohort']} | {r['level']} | {r['edge_type']} | {r['identification_status']} | {r['n']} |")
    L += ["", "MIMIC rows come from the exploratory 51-patient MIMIC fit (association only; no action data).", "",
          "## E1 False edges per patient graph under the nulls", "",
          "| cohort | null | v2 false relation instances / graph [95% CI] | v2 false node-level edges / graph | "
          "graph x replicate pairs with any | replicates with any null admission | v1 false concept pairs / graph (rep 0) |",
          "|---|---|---|---|---|---|---|"]
    for name, S, v1 in (("VitalDB (3,442)", nv, res["v1_null_rep0_vitaldb"]), ("MIMIC (51)", nm, res["v1_null_rep0_mimic51"])):
        if not S:
            continue
        for k in ("nullT", "nullS"):
            L.append(f"| {name} | {'transplant' if k == 'nullT' else 'shift'} (19 reps) | {_ci(S[k]['false_relation_instances_per_graph'], 3)} | "
                     f"{_ci(S[k]['false_node_edges_per_graph'], 3)} | {S[k]['frac_graph_replicate_pairs_with_any']:.3f} | "
                     f"{S[k]['reps_with_any_admission']}/19 | {_ci(v1[k]['false_concept_pairs_per_graph'])} |")
    if "v1_null_multi_vitaldb" in res:
        m = res["v1_null_multi_vitaldb"]
        L += ["", f"v1 on replicates 0-{m['reps'] - 1} ({m['cases']} random cases, seed {m['seed']}): transplant "
              f"{_ci(m['nullT']['false_concept_pairs_per_graph'])}, per-rep means {[round(x, 2) for x in m['nullT']['per_rep_mean']]}; "
              f"shift {_ci(m['nullS']['false_concept_pairs_per_graph'])}, per-rep means {[round(x, 2) for x in m['nullS']['per_rep_mean']]}."]
    vl = res["nulls_vital_level"]
    L += ["", "Vital-level relations: " + "; ".join(
        f"{s}: cross-patient null-vs-null {v['cross_admissions_total']} admissions in {v['cross_pairs']} pairs"
        + (f", nc_random input {v['nc_random_admitted']} -> {_ci(v['nc_random_false_edges_per_graph'], 3)} false edges/graph"
           if v.get("nc_random_false_edges_per_graph") else "") for s, v in vl.items()), "",
          "## E2 Edge-evidence coverage", "",
          f"* VitalDB v2: {cv['v2_edges_with_pointers']}/{cv['v2_edges']} node-level edges carry pointers (by construction). "
          f"v1 (recounted): 0/{cv['v1_edges']} (kg_construct never fills edge evidence).",
          f"* Presence rule (wave 3): {cv['presence_rule_instances']} concept-level relation instances; "
          f"{cv['presence_instances_with_pointer']} ({cv['frac_presence_instances_with_pointer']:.3f}) have >= 1 in-window evidence pair; "
          f"pointer instances without presence: {cv['pointer_instances_without_presence']}.",
          f"* MIMIC v2: {cm['v2_edges_with_pointers']}/{cm['v2_edges']}; v1 {cm['v1_edges_with_evidence']}/{cm['v1_edges']}.",
          f"* Pointer audit (100 VitalDB cases, raw parquet re-derivation): {res['pointer_audit']['pointers']} pointers, "
          f"{res['pointer_audit']['invalid']} invalid ({res['pointer_audit']['frac_valid']:.4f} valid).", "",
          "## E3 Graph size per patient (mean / median / p90)", "",
          "| quantity | VitalDB v2 | MIMIC v2 |", "|---|---|---|"]
    for c in g:
        a = g[c]
        b = gm.get(c)
        L.append(f"| {c} | {a['mean']:.2f} / {a['median']:.0f} / {a['p90']:.0f} | "
                 + (f"{b['mean']:.2f} / {b['median']:.0f} / {b['p90']:.0f}" if b else "-") + " |")
    L += ["", f"v1 recount check vs build_patient_kg ({res['v1_recount_check']['n_cases']} cases): "
          f"{json.dumps(res['v1_recount_check'])}",
          f"Build: {json.dumps(res['build_vitaldb'])}", "",
          f"## E4 Case studies: {res['case_studies']['chosen']} (of {res['case_studies']['n_eligible']} eligible); "
          "see case_studies/summary.md", "",
          f"## API loader check: {json.dumps(res['api_check'])}"]
    (OUT / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
