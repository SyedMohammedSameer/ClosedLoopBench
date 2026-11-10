#!/usr/bin/env python
"""
Track D (final wave): beat-level concept nodes and mechanistic fast edges in the CLBench v2 graphs
(schema 3.1). Pre-registered protocol: runs/iclr/graphs_v2_fast/protocol.md.

Stages
  build   copy each wave-4 v2 graph, add beat concepts (irregular_rr, low_hrv, high_ppv) and the fast layer
          (F1 respiration->SBP, F2 respiration->RRI admitted; F3 baroreflex and F4 PAT<->SBP abstained), run
          the N1 (within-epoch circular shift) and N2 (per-breath phase randomisation) nulls
  report  graph sizes vs v2 and v1, evidence coverage, prevalence, instantiation and null rates, pointer audit,
          FastAPI check, case studies -> results.json, report.md

Addendum B1 (runs/iclr/graphs_v2_fast/protocol_pat.md): F4 PAT<->SBP admitted from the A1 reading
(runs/iclr/pat_vitaldb) and tested per PAT-QC patient (circular-shift surrogates, one-sided), nulls N1 / N2 / N-b;
`pat` variable node and the within-case `prolonged_pat` beat concept; comparison with the frozen pre_pat/ state.

USAGE  PYTHONPATH=src:. python scripts/55_graphs_fast.py --stage all --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import time
import warnings
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
from clbench.edges.fast_coupling import pat_coupling  # noqa: E402
from clbench.persistence import CalibrationConfig  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("55_fast_graphs")

OUT = ROOT / "runs/iclr/graphs_v2_fast"
KGS = OUT / "kgs"
V2 = ROOT / "runs/iclr/graphs_v2"
FB = ROOT / "runs/iclr/fast_vitaldb"
NAT = ROOT / "data/processed_vitaldb/native"
NUM = ROOT / "data/processed_vitaldb/numerics"
CALIB = ROOT / "checkpoints/cbm/persistence_calibration.json"
PROTOCOL = OUT / "protocol.md"
PROTOCOL_SHA = OUT / "protocol.sha256"
PROTOCOL_PAT = OUT / "protocol_pat.md"
PROTOCOL_PAT_SHA = OUT / "protocol_pat.sha256"
PATD = ROOT / "runs/iclr/pat_vitaldb"
PRE = OUT / "pre_pat"

K_SURR = 499
B_EPOCH = 200
B_PATIENT = 1000
ALPHA = 0.05
MIN_EPOCHS = 3
SEED = 20260926
FAST_TESTED = (("fast:F1:resp->sbp_beat", "sbp", "sbp_beat"), ("fast:F2:resp->rri", "rri", "rri"))
F4 = "fast:F4:pat<->sbp_beat"
F4_RI = 3  # relation index in the seeds (addendum B1)
REL_IDS = tuple(r for r, _, _ in FAST_TESTED) + (F4,)
CONDS = ("real", "N1", "N2", "Nb")
NOTE_F4 = ("Association between beat PAT and beat SBP (bidirectional, not causal), expected negative; one-sided "
           "circular-shift surrogate test. The patient estimate is descriptive: population split-half persistence of "
           "the slope is REAL and survives the twin adjustment (addendum A1) but is below that of mean HR, so no "
           "per-patient specificity claim is attached. Absolute PAT includes a per-case PLETH device delay and is "
           "not interpretable; only within-case variation is used.")
NOTE = ("Population-admitted mechanism with this patient's phase-locking evidence. The patient estimate is "
        "descriptive: persistence of fast couplings is not coupling-specific (ledger C3; track B final wave), "
        "so no patient-specific coupling claim is attached.")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_protocol() -> str:
    h = sha(PROTOCOL)
    reg = PROTOCOL_SHA.read_text().split()[0]
    if h != reg:
        raise SystemExit(f"protocol.md sha256 {h} != registered {reg}; refusing to run")
    return h


def check_protocol_pat() -> str:
    h = sha(PROTOCOL_PAT)
    reg = PROTOCOL_PAT_SHA.read_text().split()[0]
    if h != reg:
        raise SystemExit(f"protocol_pat.md sha256 {h} != registered {reg}; refusing to run")
    return h


def jdefault(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ---------------------------------------------------------------------------
# Fast relation registry (population level, from track B)
# ---------------------------------------------------------------------------


def registry() -> list[kv.FastRelation]:
    dist = pd.read_csv(FB / "population_distributions.csv").set_index("quantity")
    sig = pd.read_csv(FB / "population_signs.csv").set_index("quantity")
    cv = pd.read_csv(FB / "construct_validity.csv").set_index("id")
    dec = pd.read_csv(FB / "specificity_decision.csv").set_index("edge")

    def row(df, key):
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in df.loc[key].items()}

    k_ppv = next(k for k in dist.index if k.startswith("PPV"))
    k_spv = next(k for k in dist.index if k.startswith("SPV"))
    k_rsa = next(k for k in dist.index if k.startswith("RSA harmonic"))
    k_brs = next(k for k in sig.index if k.startswith("baroreflex"))
    k_pat = next(k for k in sig.index if k.startswith("PAT-SBP slope"))
    d_ppv = next(k for k in dec.index if "PPV" in k)
    d_rsa = next(k for k in dec.index if "RSA" in k)
    d_brs = next(k for k in dec.index if "BRS" in k)
    src = str(FB.relative_to(ROOT))
    return [
        kv.FastRelation(
            relation_id="fast:F1:resp->sbp_beat", source="resp", target="sbp_beat", status="admitted",
            estimand="harmonic amplitude (peak-to-trough, mmHg) of 15-beat-detrended beat SBP on the respiratory "
                     "phase; phase locking tested against per-breath phase randomisation", unit="mmHg",
            basis="PPV construct validity CV5 and CV6 hold; PPV patient specificity REAL and survives its twin "
                  "adjustment (track B final wave)",
            population_reference={"ppv": row(dist, k_ppv), "spv": row(dist, k_spv), "CV5": row(cv, "CV5"),
                                  "CV6": row(cv, "CV6"), "specificity": row(dec, d_ppv), "source": src}),
        kv.FastRelation(
            relation_id="fast:F2:resp->rri", source="resp", target="rri", status="admitted",
            estimand="harmonic amplitude (peak-to-trough, ms) of 15-beat-detrended RR intervals on the respiratory "
                     "phase (RSA, script 52 rsa_h); phase locking tested against per-breath phase randomisation",
            unit="ms",
            basis="CV2 (age vs RSA) holds (weak); RSA patient specificity REAL and survives its twin adjustment "
                  "(track B final wave)",
            population_reference={"rsa_h": row(dist, k_rsa), "CV2": row(cv, "CV2"), "specificity": row(dec, d_rsa),
                                  "source": src}),
        kv.FastRelation(
            relation_id="fast:F3:sbp_beat->rri", source="sbp_beat", target="rri", status="abstained",
            estimand="beat ARX SBP(n..n-2) -> RRI(n) (baroreflex)", unit="ms/mmHg",
            basis="Not identified: positive in 63.3% [61.5, 64.9] of 2,944 cases, below the registered 90% "
                  "positive-control bar; sequence-BRS persistence is a signal property (twin-adjusted r 0.16); "
                  "CV1 fails (reversed)",
            population_reference={"arx": row(sig, k_brs), "CV1": row(cv, "CV1"), "specificity": row(dec, d_brs),
                                  "source": src}),
        f4_relation(row(sig, k_pat)),
    ]


def f4_relation(native_censored: dict) -> kv.FastRelation:
    """F4 from addendum A1 (runs/iclr/pat_vitaldb/summary.json) under the reading fixed in A1 section 6."""
    S = json.loads((PATD / "summary.json").read_text())
    pop = next(r for r in S["population"] if r["case set"].startswith("PAT-QC") and "slope" in r["quantity"])
    nul = {r["null"]: r for r in S["nulls"]}
    cvs = {r["id"]: r for r in S["construct_validity"]}
    rd = S["f4_admission_reading"]
    ok = bool(rd["population sign meets the F4 admission rule"])
    sp = S["specificity_decision"]
    nums = (f"delay-aware PAT (script 56), PAT-QC {S['qc']['coverage_valid >= 0.80 (PAT-QC set)']} cases: slope negative "
            f"in {pop['cases negative']} of {pop['n_cases']} cases (A1 bar: CI lower bound >= 90%) and "
            f"{pop['epochs negative (cluster CI)']} of epochs, median {pop['median of case medians']} ms/mmHg; "
            f"within-case nulls N-a {nul['N-a']['null cases negative (mean over draws) [95% CI]']} and N-b "
            f"{nul['N-b']['null cases negative (mean over draws) [95% CI]']} negative (observed - null "
            f"{nul['N-a']['observed - null (pp) [95% CI]']} / {nul['N-b']['observed - null (pp) [95% CI]']} pp); "
            f"CV7w within-case PAT vs MAP {cvs['CV7w']['estimate [95% CI]']} ms/mmHg (consistency check); slope "
            f"patient specificity REAL={sp['REAL']} (twin-adjusted {sp['twin-adj (H3)']}); CV7 (age vs between-case "
            f"PAT) not physiologically interpretable (device delay), not counted")
    basis = (("Admitted (A1 section 6: F3 positive-control rule and within-case artefact reading both met): " if ok else
              "Abstained (A1 section 6 reading not met): ") + nums)
    return kv.FastRelation(
        relation_id=F4, source="pat", target="sbp_beat", status="admitted" if ok else "abstained",
        estimand="beat PAT-SBP slope (ms/mmHg; 31-beat detrended, pat_coupling), expected negative; tested one-sided "
                 "against circular-shift surrogates of SBP (U{30..n-30} beats)", unit="ms/mmHg", basis=basis,
        population_reference={"a1_population": pop, "a1_reading": S["reading"], "a1_nulls": S["nulls"],
                              "a1_f4_admission_reading": rd, "a1_construct_validity": S["construct_validity"],
                              "a1_specificity": sp, "a1_exploratory_posthoc": S.get("exploratory_posthoc"),
                              "a1_protocol_sha256": S["protocol_sha256"], "native_censored": native_censored,
                              "caveat": "absolute and between-case PAT include a per-case PLETH device delay "
                                        "(median case PAT ~654 ms); only within-case variation is interpretable",
                              "source": str(PATD.relative_to(ROOT))})


# ---------------------------------------------------------------------------
# Build (worker)
# ---------------------------------------------------------------------------

_W: dict = {}


def _init(reg_json: str, calib_path: str, t_low: float, psha: str, delta_pat: float = float("nan"),
          psha_pat: str = ""):
    _W["reg"] = [kv.FastRelation.model_validate(r) for r in json.loads(reg_json)]
    _W["calib"] = CalibrationConfig.load(Path(calib_path))
    _W["t_low"] = t_low
    _W["psha"] = psha
    _W["delta_pat"] = delta_pat
    _W["psha_pat"] = psha_pat


def _f4_tests(b: pd.DataFrame, pat: np.ndarray, epochs: list[dict], ci: int) -> dict:
    """F4 per-epoch results per condition (addendum B1 sections 4-5) and integrity counts vs A1."""
    tr = b["t_r"].to_numpy(float)
    sbp_all = b["sbp"].to_numpy(float)
    v_all = b["valid"].to_numpy(bool)
    seg = {}
    for ep in epochs:
        seg[int(ep["epoch_index"])] = (int(np.searchsorted(tr, ep["t0"], "left")), int(np.searchsorted(tr, ep["t1"], "left")))
    ei_list = [int(ep["epoch_index"]) for ep in epochs]
    res = {c: [] for c in ("real", "N1", "N2", "Nb")}
    chk = {"compared": 0, "mismatch": 0, "max_abs_diff": 0.0, "a1_finite_not_tested": 0, "undefined_surrogates": 0,
           "surrogates": 0}
    for ep in epochs:
        ei = int(ep["epoch_index"])
        a, z = seg[ei]
        n = z - a
        pt, sb, vv = pat[a:z], sbp_all[a:z], v_all[a:z]
        base = {"epoch_index": ei, "t0": float(ep["t0"]), "t1": float(ep["t1"]), "beat_lo": a, "beat_hi": z - 1,
                "n_beats": n}
        r = kv.f4_epoch_test(pt, sb, vv, np.random.default_rng([SEED, ci, ei, F4_RI, 0]), K_SURR, B_EPOCH)
        a1 = ep.get("a1_slope")
        if r is None:
            if a1 is not None and np.isfinite(a1):
                chk["a1_finite_not_tested"] += 1
            continue
        if a1 is not None and np.isfinite(a1):
            d = abs(r["A"] - float(a1))
            chk["compared"] += 1
            chk["mismatch"] += int(d > 1e-9)
            chk["max_abs_diff"] = max(chk["max_abs_diff"], d)
        else:
            chk["mismatch"] += 1  # tested here but no finite A1 slope
        chk["undefined_surrogates"] += r["n_undefined_surrogates"]
        chk["surrogates"] += K_SURR
        clin = {"pat_sbp_corr": r["corr"], **{k: ep.get(k) for k in ("sbp_mean", "map_mean", "hr_mean")}}
        res["real"].append({**base, "A": r["A"], "Ak": r["Ak"], "p": r["p"], "lo": r["lo"], "hi": r["hi"],
                            "n_pairs": r["n_pairs"], "n_blocks": r["n_blocks"],
                            "clinical": {k: float(x) for k, x in clin.items() if x is not None and np.isfinite(x)}})
        # N1: SBP (with validity) rolled against PAT by u ~ U[ceil(n/4), floor(3n/4)]
        g1 = np.random.default_rng([SEED, ci, ei, F4_RI, 1])
        u = int(g1.integers(math.ceil(n / 4), math.floor(3 * n / 4) + 1))
        r1 = kv.f4_epoch_test(pt, np.roll(sb, u), vv & np.roll(vv, u), g1, K_SURR, 0)
        if r1 is not None:
            res["N1"].append({**base, "A": r1["A"], "Ak": r1["Ak"], "p": r1["p"]})
        # N2: one draw of the test's own surrogate distribution, independent stream
        g2 = np.random.default_rng([SEED, ci, ei, F4_RI, 2])
        u0 = int(g2.integers(kv.PAT_SHIFT_MIN, n - kv.PAT_SHIFT_MIN + 1))
        r2 = kv.f4_epoch_test(pt, np.roll(sb, u0), vv & np.roll(vv, u0), g2, K_SURR, 0)
        if r2 is not None:
            res["N2"].append({**base, "A": r2["A"], "Ak": r2["Ak"], "p": r2["p"]})
        # N-b: PAT of this epoch with SBP of another usable epoch of the same case
        others = [e for e in ei_list if e != ei]
        if others:
            g5 = np.random.default_rng([SEED, ci, ei, F4_RI, 5])
            e2 = others[int(g5.integers(0, len(others)))]
            a2, z2 = seg[e2]
            m = min(n, z2 - a2)
            r5 = kv.f4_epoch_test(pt[:m], sbp_all[a2:a2 + m], vv[:m] & v_all[a2:a2 + m], g5, K_SURR, 0)
            if r5 is not None:
                res["Nb"].append({**base, "A": r5["A"], "Ak": r5["Ak"], "p": r5["p"]})
    return {"res": res, "check": chk}


def _epoch_tests(b: pd.DataFrame, epochs: list[dict], ci: int) -> dict:
    """Per relation and condition: list of per-epoch results (dicts)."""
    tr = b["t_r"].to_numpy(float)
    res = {rid: {"real": [], "N1": [], "N2": []} for rid, _, _ in FAST_TESTED}
    rsa_diff = []
    for ep in epochs:
        ei = int(ep["epoch_index"])
        a = int(np.searchsorted(tr, ep["t0"], "left"))
        z = int(np.searchsorted(tr, ep["t1"], "left"))
        if z - a < kv.HARM_MIN_BEATS:
            continue
        sub = b.iloc[a:z]
        v = sub["valid"].to_numpy(bool)
        ph = sub["resp_phase"].to_numpy(float)
        if not np.isfinite(ph).mean() > 0.8:  # track B / wave-2 gate for respiration-dependent edges
            continue
        bid = kv.breath_ids(ph)
        nbid = int(bid.max()) + 1
        n = z - a
        for ri, (rid, col, _) in enumerate(FAST_TESTED):
            tgt = sub[col].to_numpy(float)
            base = {"epoch_index": ei, "t0": float(ep["t0"]), "t1": float(ep["t1"]), "beat_lo": a, "beat_hi": z - 1,
                    "n_beats": n}
            # real
            prep = kv.fast_epoch_prepare(tgt, v, ph, bid)
            if prep is None:
                continue
            y, pho, bo, _ = prep
            r = kv.fast_epoch_test(y, pho, bo, nbid, np.random.default_rng([SEED, ci, ei, ri, 0]), K_SURR, B_EPOCH)
            clin = ({"ppv": ep.get("ppv"), "spv": ep.get("spv")} if col == "sbp" else {"rsa_h": ep.get("rsa_h")})
            res[rid]["real"].append({**base, "A": r["A"], "Ak": r["Ak"], "p": r["p"], "lo": r["lo"], "hi": r["hi"],
                                     "n_breaths": r["n_breaths"],
                                     "clinical": {k: float(x) for k, x in clin.items() if x is not None and np.isfinite(x)}})
            if col == "rri" and ep.get("rsa_h") is not None and np.isfinite(ep["rsa_h"]):
                rsa_diff.append(abs(r["A"] - float(ep["rsa_h"])))
            # N1: target rolled against the phase by u beats
            g1 = np.random.default_rng([SEED, ci, ei, ri, 1])
            u = int(g1.integers(math.ceil(n / 4), math.floor(3 * n / 4) + 1))
            p1 = kv.fast_epoch_prepare(np.roll(tgt, u), np.roll(v, u), ph, bid)
            if p1 is not None:
                r1 = kv.fast_epoch_test(p1[0], p1[1], p1[2], nbid, g1, K_SURR, 0)
                res[rid]["N1"].append({**base, "A": r1["A"], "Ak": r1["Ak"], "p": r1["p"]})
            # N2: per-breath phase randomisation of the data
            g2 = np.random.default_rng([SEED, ci, ei, ri, 2])
            ph2 = ph + g2.uniform(0.0, 2 * np.pi, nbid)[bid]
            p2 = kv.fast_epoch_prepare(tgt, v, ph2, bid)
            if p2 is not None:
                r2 = kv.fast_epoch_test(p2[0], p2[1], p2[2], nbid, g2, K_SURR, 0)
                res[rid]["N2"].append({**base, "A": r2["A"], "Ak": r2["Ak"], "p": r2["p"]})
    return {"res": res, "rsa_max_abs_diff": float(max(rsa_diff)) if rsa_diff else None,
            "rsa_n_compared": len(rsa_diff)}


def _patient_decisions(res: dict) -> dict:
    """Patient p, Holm over the case's tested relations, BH pointers, instantiation; per condition.

    ``res`` maps relation id -> condition -> per-epoch results. F4 is one-sided (expected negative). A condition
    that a relation lacks (N-b for F1 / F2) takes that relation's N2 null in the Holm family (addendum B1 s. 5)."""
    out = {}
    conds = [c for c in CONDS if any(c in res[r] for r in res)]
    for cond in conds:
        ps = {}
        for rid in res:
            src = cond if cond in res[rid] else ("N2" if cond == "Nb" else None)
            eps = res[rid].get(src, []) if src else []
            if len(eps) >= MIN_EPOCHS:
                ps[rid] = kv.patient_fast_p(np.array([e["A"] for e in eps]), np.stack([e["Ak"] for e in eps]),
                                            "less" if rid == F4 else "greater")
        hol = dict(zip(ps, kv.holm(list(ps.values())))) if ps else {}
        for rid in res:
            if cond not in res[rid]:
                continue
            eps = res[rid][cond]
            d = {"n_epochs": len(eps), "eligible": rid in ps, "p": ps.get(rid), "p_holm": hol.get(rid),
                 "n_pointers": 0, "instantiated": False}
            if rid in ps:
                pbh = kv.bh_adjust([e["p"] for e in eps])
                ptr = [i for i, q in enumerate(pbh) if q <= ALPHA]
                d["p_bh"] = pbh.tolist()
                d["pointer_idx"] = ptr
                d["n_pointers"] = len(ptr)
                d["instantiated"] = bool(hol[rid] <= ALPHA and len(ptr) >= 1)
                d["mean_A"] = float(np.mean([e["A"] for e in eps]))
            out[(rid, cond)] = d
    return out


def _pre_compare(pid: str, new: dict) -> dict:
    """Changes vs the frozen clock-fixed pre-PAT graph (pre_pat/kgs)."""
    f = PRE / "kgs" / f"{pid}.json.gz"
    if not f.exists():
        return {"pre_found": False}
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        pre = json.loads(fh.read())
    out = {"pre_found": True,
           "nodes_prefix_same": new["nodes"][:len(pre["nodes"])] == pre["nodes"],
           "variables_prefix_same": new["variables"][:len(pre["variables"])] == pre["variables"],
           "beat_episodes_prefix_same": new["beat_episodes"][:len(pre["beat_episodes"])] == pre["beat_episodes"],
           "v2_layer_same_as_pre": all(new[k] == pre[k] for k in ("episodes", "interventions", "relations", "edges")),
           "d_nodes": len(new["nodes"]) - len(pre["nodes"]), "d_variables": len(new["variables"]) - len(pre["variables"]),
           "d_beat_episodes": len(new["beat_episodes"]) - len(pre["beat_episodes"])}
    for rid in REL_IDS:
        k = rid.split(":")[1]
        e0 = [e for e in pre["fast_edges"] if e["relation_id"] == rid]
        e1 = [e for e in new["fast_edges"] if e["relation_id"] == rid]
        out[f"{k}_pre"] = len(e0)
        out[f"{k}_post"] = len(e1)
        if e0 and e1:
            x0, x1 = dict(e0[0]), dict(e1[0])
            drop = ("patient_p_holm", "expected_sign", "pointers")
            out[f"{k}_edge_same_except_holm"] = ({kk: v for kk, v in x0.items() if kk not in drop}
                                                 == {kk: v for kk, v in x1.items() if kk not in drop})
            out[f"{k}_holm_pre"] = x0["patient_p_holm"]
            out[f"{k}_holm_post"] = x1["patient_p_holm"]
            out[f"{k}_ptr_same"] = x0["pointers"] == [{kk: v for kk, v in q.items() if kk != "n_blocks"} for q in x1["pointers"]]
    return out


def build_task(job: dict) -> dict:
    pid, qc = job["pid"], job["qc"]
    t_start = time.time()
    with gzip.open(V2 / "kgs_vitaldb" / f"{pid}.json.gz", "rt", encoding="utf-8") as f:
        orig = json.loads(f.read())
    kg = kv.PatientKGv3.model_validate(orig)
    kg.schema_version = kv.FAST_SCHEMA_VERSION
    kg.fast_relations = [r.model_copy() for r in _W["reg"]]
    f4_admitted = any(r.relation_id == F4 and r.status == "admitted" for r in kg.fast_relations)
    n_nodes0, n_vars0 = len(kg.nodes), len(kg.variables)
    meta = {"protocol_sha256": _W["psha"], "protocol_pat_sha256": _W["psha_pat"],
            "source_v2_graph": f"runs/iclr/graphs_v2/kgs_vitaldb/{pid}.json.gz",
            "clock": "all times in seconds from clinical casestart, the clock shared by the v2 30 s grid and "
                     "the native tables; beat pointers index rows of the beats file (and of the aligned PAT file)",
            "beat_concepts_abstained": {}}
    rec: dict = {"pid": pid, "qc": qc, "pat_qc": bool(job.get("pat_qc", False))}
    if qc:
        off = float(job["offset"])
        beats_rel = f"data/processed_vitaldb/native/beats/{pid}.parquet"
        b = pd.read_parquet(ROOT / beats_rel)
        ws = pd.read_parquet(NUM / f"{pid}.parquet", columns=["window_start_sec"])["window_start_sec"].to_numpy(float)
        tr = b["t_r"].to_numpy(float)
        # PAT (addendum B1 section 2): aligned script-56 PAT for PAT-QC cases only
        pat = None
        pat_rel = f"data/processed_vitaldb/native/pat/{pid}.parquet"
        if rec["pat_qc"]:
            pf = pd.read_parquet(ROOT / pat_rel)
            if len(pf) == len(b) and np.array_equal(pf["t_r"].to_numpy(float), tr):
                pat = pf["pat_ms"].to_numpy(float)
            else:
                rec["pat_qc"] = False
                rec["pat_misaligned"] = True
        cvep = [(e["t0"] - off, e["t1"] - off) for e in job["epochs"]]
        bw = kv.beat_window_concepts(tr - off, b["rri"].to_numpy(float), ws, _W["t_low"], b["sbp"].to_numpy(float),
                                     b["pp"].to_numpy(float), b["resp_phase"].to_numpy(float),
                                     b["valid"].to_numpy(bool), np.array(cvep).reshape(-1, 2))
        if pat is not None:
            s_pat, st_pat, base_pat = kv.pat_window_concept(tr - off, pat, b["valid"].to_numpy(bool), ws, _W["delta_pat"])
        else:
            s_pat, st_pat, base_pat = np.full(len(ws), -1, np.int8), np.full(len(ws), np.nan), float("nan")
            meta["beat_concepts_abstained"][kv.PAT_CONCEPT] = (
                "case fails the A1 PAT-QC rule (" + job.get("pat_gate", "misaligned PAT file") + "); absolute PAT is "
                "not interpretable (device delay) and a within-case baseline needs PAT-QC")
        bw = kv.with_pat_concept(bw, s_pat, st_pat)
        nodes, beps = kv.beat_concept_nodes(pid, bw, ws, _W["calib"], beats_file=beats_rel)
        kg.nodes.extend(nodes)
        kg.beat_episodes = beps
        # beat variable nodes (observed = rhythm-observed windows; SBP / phase: >= 20 finite values in window;
        # PAT: prolonged_pat-observed windows)
        v_ok = b["valid"].to_numpy(bool)
        fin_s = np.concatenate([[0], np.cumsum(v_ok & np.isfinite(b["sbp"].to_numpy(float)))])
        fin_p = np.concatenate([[0], np.cumsum(np.isfinite(b["resp_phase"].to_numpy(float)))])
        lo = np.searchsorted(tr - off, ws, "left")
        hi = np.searchsorted(tr - off, ws + kv.WINDOW_SEC, "left")
        nobs = {"rri": int((bw.S[:, 0] >= 0).sum()), "sbp_beat": int(((fin_s[hi] - fin_s[lo]) >= 20).sum()),
                "resp": int(((fin_p[hi] - fin_p[lo]) >= 20).sum()), "pat": int((bw.S[:, 3] >= 0).sum())}
        for vname, unit in kv.BEAT_VARIABLES.items():
            if nobs[vname] > 0:
                kg.variables.append(kv.VariableNode(node_id=f"{pid}:var:{vname}", variable=vname, unit=unit,
                                                    n_observed_windows=nobs[vname]))
        var_ids = {v.variable: v.node_id for v in kg.variables}
        # fast edges
        et = _epoch_tests(b, job["epochs"], int(job["case_index"]))
        res = {rid: et["res"][rid] for rid, _, _ in FAST_TESTED}
        f4 = None
        if pat is not None and f4_admitted:
            f4 = _f4_tests(b, pat, job["epochs_all"], int(job["case_index"]))
            res[F4] = f4["res"]
        dec = _patient_decisions(res)
        for ri, (rid, col, tvar) in enumerate(list(FAST_TESTED) + [(F4, "sbp", "sbp_beat")]):
            if rid not in res:
                continue
            ri_seed = F4_RI if rid == F4 else ri
            svar = "pat" if rid == F4 else "resp"
            d = dec[(rid, "real")]
            if not d["instantiated"] or svar not in var_ids or tvar not in var_ids:
                if d["instantiated"]:
                    d["instantiated"] = False
                    d["dropped_missing_variable"] = True
                continue
            eps = res[rid]["real"]
            A = np.array([e["A"] for e in eps])
            g = np.random.default_rng([SEED, int(job["case_index"]), 999, ri_seed, 3])
            bm = A[g.integers(0, len(A), size=(B_PATIENT, len(A)))].mean(axis=1)
            ptrs = []
            for i in d["pointer_idx"]:
                e = eps[i]
                if rid == F4:
                    ptrs.append(kv.FastPointer(
                        epoch_start_sec=e["t0"] - off, epoch_end_sec=e["t1"] - off, beat_start_idx=e["beat_lo"],
                        beat_end_idx=e["beat_hi"], estimate=e["A"], ci_lo=e["lo"], ci_hi=e["hi"], p_value=e["p"],
                        p_bh=float(d["p_bh"][i]), n_beats=e["n_pairs"], n_blocks=e["n_blocks"], clinical=e["clinical"]))
                else:
                    ptrs.append(kv.FastPointer(
                        epoch_start_sec=e["t0"] - off, epoch_end_sec=e["t1"] - off, beat_start_idx=e["beat_lo"],
                        beat_end_idx=e["beat_hi"], estimate=e["A"], ci_lo=e["lo"], ci_hi=e["hi"], p_value=e["p"],
                        p_bh=float(d["p_bh"][i]), n_beats=e["n_beats"], n_breaths=e["n_breaths"],
                        clinical=e["clinical"]))
            rel = next(r for r in kg.fast_relations if r.relation_id == rid)
            kg.fast_edges.append(kv.FastEdge(
                edge_id=f"{pid}:{rid}", relation_id=rid, source_node_id=var_ids[svar], target_node_id=var_ids[tvar],
                estimand=rel.estimand, unit=rel.unit, patient_estimate=float(A.mean()),
                patient_ci=(float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))), patient_p=d["p"],
                patient_p_holm=float(d["p_holm"]), n_epochs_tested=len(eps),
                expected_sign=-1 if rid == F4 else None, note=NOTE_F4 if rid == F4 else NOTE, pointers=ptrs))
        meta.update({"offset_loader_minus_casestart_s": off, "t_low_rmssd_ms": _W["t_low"],
                     "tiers": "fallback 60 s / 600 s (beat concepts are not in the persistence calibration)",
                     "n_controlled_vent_epochs": len(job["epochs"]),
                     "pat": ({"pat_file": pat_rel, "pat_qc": True, "delta_pat_ms": _W["delta_pat"],
                              "prolonged_pat_baseline_ms_incl_device_delay": base_pat,
                              "n_usable_epochs_f4": len(job["epochs_all"])} if pat is not None else
                             {"pat_qc": False, "gate": job.get("pat_gate", "misaligned PAT file")})})
        # window-level prevalence
        rec["win_obs"] = [int((bw.S[:, k] >= 0).sum()) for k in range(bw.S.shape[1])]
        rec["win_on"] = [int((bw.S[:, k] == 1).sum()) for k in range(bw.S.shape[1])]
        rec["ep_steps"] = [int(sum(e.n_steps for e in beps if e.concept == c)) for c in kv.BEAT_CONCEPTS_ALL]
        rec["fast"] = {f"{rid}|{cond}": {k: v for k, v in d.items() if k not in ("p_bh", "pointer_idx")}
                       for (rid, cond), d in dec.items()}
        rec["rsa_max_abs_diff"] = et["rsa_max_abs_diff"]
        rec["rsa_n_compared"] = et["rsa_n_compared"]
        rec["n_windows"] = len(ws)
        if f4 is not None:
            rec["f4_check"] = f4["check"]
    else:
        meta["fast_layer"] = "no beat content: case not in track B QC (beats file / valid fraction / HR bias / 20 min)"
    kg.build_metadata = {**kg.build_metadata, "fast_layer_v3_1": meta}
    errs = kg.validate_refs()
    new = kg.model_dump(mode="json")
    same = (new["nodes"][:n_nodes0] == orig["nodes"] and new["variables"][:n_vars0] == orig["variables"]
            and all(new[k] == orig[k] for k in ("episodes", "interventions", "relations", "edges")))
    js = json.dumps(new, default=jdefault)
    with gzip.open(KGS / f"{pid}.json.gz", "wt", encoding="utf-8") as f:
        f.write(js)
    rec.update({"counts": kv.graph_counts(new), "errs": errs[:5], "n_errs": len(errs), "v2_same": bool(same),
                "vs_pre": _pre_compare(pid, new), "bytes": len(js), "sec": time.time() - t_start})
    return rec


def pat_gates() -> tuple[set, dict, float]:
    """A1 PAT-QC set (stage-1 cases.parquet q3_cover), the gate each other QC case fails, and Delta_pat
    (addendum B1 section 3: 90th percentile of within-case epoch PAT deviations over A1's PAT-QC usable epochs)."""
    Q = pd.read_parquet(PATD / "cases.parquet")
    ok = set(Q.loc[Q["q3_cover"] == True, "patient_id"])  # noqa: E712
    gate = {}
    for _, r in Q.iterrows():
        if r["q3_cover"] == True:  # noqa: E712
            continue
        gate[r["patient_id"]] = ("no aligned PAT file" if r["q0_file"] != True else  # noqa: E712
                                 "clock_warning" if r["q1_clock"] != True else  # noqa: E712
                                 "delay_mode_s outside [0.45, 0.90] s" if r["q2_delay"] != True else  # noqa: E712
                                 "coverage_valid < 0.80")
    E = pd.read_parquet(PATD / "epochs_pat.parquet", columns=["patient_id", "pat_median"])
    E = E[E["patient_id"].isin(ok)]
    dev = E["pat_median"] - E.groupby("patient_id")["pat_median"].transform("median")
    return ok, gate, float(np.nanpercentile(dev.to_numpy(float), 90))


def stage_build(workers: int, limit: int | None = None, pids: list[str] | None = None) -> None:
    psha = check_protocol()
    psha_pat = check_protocol_pat()
    KGS.mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(FB / "cases.parquet")
    ep = pd.read_parquet(FB / "epochs.parquet")
    usable = ep[ep["usable"].astype(bool)]
    t_low = float(np.nanpercentile(usable.loc[usable["patient_id"].isin(cases.loc[cases["g4_20min"] == True,  # noqa: E712
                                                                                     "patient_id"]), "rmssd"], 10))
    cv = usable[usable["controlled_vent"].astype(bool)]
    cols = ["epoch_index", "t0", "t1", "ppv", "spv", "rsa_h"]
    by = {pid: g[cols].to_dict("records") for pid, g in cv.groupby("patient_id")}
    pat_ok, pat_gate, delta_pat = pat_gates()
    a1 = pd.read_parquet(PATD / "epochs_pat.parquet", columns=["patient_id", "epoch_index", "pat_sbp_slope"])
    ua = usable[usable["patient_id"].isin(pat_ok)][["patient_id", "epoch_index", "t0", "t1", "sbp_mean", "map_mean",
                                                     "hr_mean"]].merge(a1.rename(columns={"pat_sbp_slope": "a1_slope"}),
                                                                       on=["patient_id", "epoch_index"], how="left")
    by_all = {pid: g.drop(columns="patient_id").to_dict("records") for pid, g in ua.groupby("patient_id")}
    all_pids = sorted(p.name.replace(".json.gz", "") for p in (V2 / "kgs_vitaldb").glob("*.json.gz"))
    qc = cases.set_index("patient_id")
    jobs = []
    for pid in all_pids:
        is_qc = bool(pid in qc.index and qc.loc[pid, "g4_20min"] == True)  # noqa: E712
        j = {"pid": pid, "qc": is_qc}
        if is_qc:
            s = json.loads((NAT / "summaries" / f"{pid}.json").read_text())
            j.update({"offset": grid_offset(s), "case_index": int(qc.loc[pid, "case_index"]),
                      "epochs": by.get(pid, []), "pat_qc": pid in pat_ok, "epochs_all": by_all.get(pid, []),
                      "pat_gate": pat_gate.get(pid, "not in A1 cases table")})
        jobs.append(j)
    if pids:
        jobs = [j for j in jobs if j["pid"] in pids]
    if limit:
        jobs = jobs[:limit]
    reg = registry()
    log.info("build: %d graphs (%d QC, %d PAT-QC), t_low RMSSD %.3f ms, Delta_pat %.3f ms, F4 %s, workers %d",
             len(jobs), sum(j["qc"] for j in jobs), sum(bool(j.get("pat_qc")) for j in jobs), t_low, delta_pat,
             next(r.status for r in reg if r.relation_id == F4), workers)
    t0 = time.time()
    out = OUT / ("build_records.jsonl" if not (limit or pids) else "build_records_smoke.jsonl")
    n = 0
    with out.open("w") as fo, ProcessPoolExecutor(workers, initializer=_init,
                                                  initargs=(json.dumps([r.model_dump() for r in reg], default=jdefault),
                                                            str(CALIB), t_low, psha, delta_pat, psha_pat)) as ex:
        for rec in ex.map(build_task, jobs, chunksize=4):
            fo.write(json.dumps(rec, default=jdefault) + "\n")
            n += 1
            if n % 250 == 0:
                log.info("  %d / %d graphs (%.0f s)", n, len(jobs), time.time() - t0)
    meta = {"protocol_sha256": psha, "protocol_pat_sha256": psha_pat, "script_sha256": sha(Path(__file__)),
            "kg_v2_sha256": sha(ROOT / "src/clbench/kg_v2.py"),
            "a1_summary_sha256": sha(PATD / "summary.json"), "a1_protocol_sha256": sha(PATD / "protocol.md"),
            "n_graphs": n, "t_low_rmssd_ms": t_low, "delta_pat_ms": delta_pat, "n_pat_qc": len(pat_ok),
            "seconds": time.time() - t0, "workers": workers,
            "K_surrogates": K_SURR, "B_epoch": B_EPOCH, "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "registry": [r.model_dump() for r in reg]}
    (OUT / ("build_meta.json" if not (limit or pids) else "build_meta_smoke.json")).write_text(
        json.dumps(meta, indent=1, default=jdefault))
    log.info("build done: %d graphs in %.0f s", n, time.time() - t0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def boot_ci(x, n=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = stat(x[idx], axis=1)
    return (float(stat(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def fci(t, nd=2, pct=False):
    if pct:
        return f"{100 * t[0]:.1f}% [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def pointer_audit(pids: list[str]) -> dict:
    """Beat ranges inside their epoch / episode, and the fast estimate recomputed from the beats file."""
    out = {"graphs": 0, "fast_pointers": 0, "fast_range_ok": 0, "fast_estimate_ok": 0, "beat_episodes": 0,
           "episode_range_ok": 0, "max_abs_estimate_diff": 0.0}
    for pid in pids:
        g = json.loads(gzip.open(KGS / f"{pid}.json.gz", "rt").read())
        off = g["build_metadata"]["fast_layer_v3_1"]["offset_loader_minus_casestart_s"]
        b = pd.read_parquet(NAT / "beats" / f"{pid}.parquet")
        t = b["t_r"].to_numpy(float) - off
        out["graphs"] += 1
        patv = None
        for e in g["fast_edges"]:
            col = "sbp" if e["relation_id"].startswith("fast:F1") else "rri"
            is4 = e["relation_id"] == F4
            if is4 and patv is None:
                pf = pd.read_parquet(NAT / "pat" / f"{pid}.parquet")
                patv = pf["pat_ms"].to_numpy(float)
                out["f4_pat_rows_aligned"] = out.get("f4_pat_rows_aligned", 0) + int(
                    np.array_equal(pf["t_r"].to_numpy(float), b["t_r"].to_numpy(float)))
            for p in e["pointers"]:
                out["fast_pointers"] += 1
                a, z = p["beat_start_idx"], p["beat_end_idx"]
                inside = (t[a] >= p["epoch_start_sec"] - 1e-6 and t[z] < p["epoch_end_sec"]
                          and (a == 0 or t[a - 1] < p["epoch_start_sec"]) and (z + 1 >= len(t) or t[z + 1] >= p["epoch_end_sec"]))
                out["fast_range_ok"] += int(inside)
                sub = b.iloc[a:z + 1]
                if is4:
                    out["f4_pointers"] = out.get("f4_pointers", 0) + 1
                    out["f4_range_ok"] = out.get("f4_range_ok", 0) + int(inside)
                    A = pat_coupling(patv[a:z + 1], sub["sbp"].to_numpy(float), sub["valid"].to_numpy(bool))["pat_sbp_slope"]
                    dlt = abs(A - p["estimate"])
                    out["max_abs_estimate_diff"] = max(out["max_abs_estimate_diff"], dlt)
                    out["fast_estimate_ok"] += int(dlt < 1e-9)
                    out["f4_estimate_ok"] = out.get("f4_estimate_ok", 0) + int(dlt < 1e-9)
                    continue
                prep = kv.fast_epoch_prepare(sub[col].to_numpy(float), sub["valid"].to_numpy(bool),
                                             sub["resp_phase"].to_numpy(float))
                if prep is not None:
                    A = kv.harmonic_amplitude(prep[0], prep[1])
                    dlt = abs(A - p["estimate"])
                    out["max_abs_estimate_diff"] = max(out["max_abs_estimate_diff"], dlt)
                    out["fast_estimate_ok"] += int(dlt < 1e-9)
        for e in g["beat_episodes"]:
            out["beat_episodes"] += 1
            a, z = e["beat_start_idx"], e["beat_end_idx"]
            out["episode_range_ok"] += int(t[a] >= e["start_sec"] - 1e-6 and t[z] < e["end_sec"] and a <= z)
    return out


def api_check(paths: list[Path]) -> dict:
    spec = importlib.util.spec_from_file_location("s50", ROOT / "scripts/50_build_kgs_v3.py")
    s50 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s50)
    return s50.api_check([], paths)


def case_summary(pid: str) -> str:
    g = json.loads(gzip.open(KGS / f"{pid}.json.gz", "rt").read())
    c = kv.graph_counts(g)
    L = [f"## {pid}", "",
         f"- v2 layer: {c['nodes_v1_concept_tier']} concept x tier nodes, {c['episodes']} episodes, "
         f"{c['action_nodes']} action nodes, {c['relation_instances']} relation instances, "
         f"{c['edges_node_level']} typed edges ({c['pointers']} pointers).",
         f"- Fast layer: {c['nodes_beat_concept_tier']} beat concept x tier nodes, {c['beat_episodes']} beat "
         f"episodes, {c['beat_variable_nodes']} beat variable nodes, {c['fast_edges']} fast edges "
         f"({c['fast_pointers']} epoch pointers)."]
    for conc in kv.BEAT_CONCEPTS_ALL:
        eps = [e for e in g["beat_episodes"] if e["concept"] == conc]
        if eps:
            tot = sum(e["n_steps"] for e in eps) * kv.STEP_SEC / 60.0
            longest = max(eps, key=lambda e: e["n_steps"])
            L.append(f"- `{conc}`: {len(eps)} episodes, {tot:.1f} min of on-steps; longest {longest['start_sec']:.0f}-"
                     f"{longest['end_sec']:.0f} s ({longest['tier']}), beats rows {longest['beat_start_idx']}-"
                     f"{longest['beat_end_idx']}, median {longest['statistic']} {longest['statistic_median']:.3g}.")
    for e in g["fast_edges"]:
        ptr = e["pointers"]
        top = max(ptr, key=lambda p: p["estimate"])
        top = min(ptr, key=lambda p: p["estimate"]) if e["relation_id"] == F4 else top
        L.append(f"- `{e['relation_id']}`: patient mean {'slope' if e['relation_id'] == F4 else 'amplitude'} "
                 f"{e['patient_estimate']:.2f} {e['unit']} "
                 f"[{e['patient_ci'][0]:.2f}, {e['patient_ci'][1]:.2f}] over {e['n_epochs_tested']} epochs, patient p "
                 f"{e['patient_p']:.3f} (Holm {e['patient_p_holm']:.3f}); {len(ptr)} pointer epochs. "
                 + ("Most negative: " if e["relation_id"] == F4 else "Largest: ") +
                 f"{top['epoch_start_sec']:.0f}-{top['epoch_end_sec']:.0f} s, beats {top['beat_start_idx']}-"
                 f"{top['beat_end_idx']}, {top['estimate']:.2f} [{top['ci_lo']:.2f}, {top['ci_hi']:.2f}], p "
                 f"{top['p_value']:.3f}, clinical {json.dumps({k: round(v, 2) for k, v in top['clinical'].items()})}.")
    for r in g["fast_relations"]:
        if r["status"] == "abstained":
            L.append(f"- Abstained `{r['relation_id']}`: {r['basis']}.")
    return "\n".join(L) + "\n"


def stage_report() -> None:
    psha = check_protocol()
    psha_pat = check_protocol_pat()
    recs = [json.loads(x) for x in (OUT / "build_records.jsonl").read_text().splitlines() if x.strip()]
    meta = json.loads((OUT / "build_meta.json").read_text())
    R = pd.DataFrame([{"pid": r["pid"], "qc": r["qc"], "pat_qc": bool(r.get("pat_qc", False)), **r["counts"],
                       "n_errs": r["n_errs"], "v2_same": r["v2_same"], "bytes": r["bytes"]} for r in recs])
    v2s = pd.read_csv(V2 / "graph_size_vitaldb.csv").set_index("pid")
    R = R.join(v2s[["v1_nodes", "v1_edges", "nodes_concept_tier", "edges_node_level", "episodes"]].add_prefix("w4_"),
               on="pid")
    R.to_csv(OUT / "graph_size_fast.csv", index=False)
    res: dict = {"protocol_sha256": psha, "protocol_pat_sha256": psha_pat,
                 "build_meta": {k: v for k, v in meta.items() if k != "registry"}, "registry": meta["registry"]}
    Q = [r for r in recs if r["qc"]]
    PQ = [r for r in Q if r.get("pat_qc")]
    # integrity
    f4c = pd.DataFrame([r["f4_check"] for r in PQ if r.get("f4_check")])
    res["integrity"] = {
        "graphs": len(R), "qc_graphs": int(R.qc.sum()), "pat_qc_graphs": int(R.pat_qc.sum()),
        "pat_files_misaligned_at_build": int(sum(bool(r.get("pat_misaligned")) for r in Q)),
        "validate_refs_errors": int(R.n_errs.sum()),
        "v2_layer_unchanged": int(R.v2_same.sum()),
        "v2_counts_match_wave4_csv": int(((R.nodes_v1_concept_tier == R.w4_nodes_concept_tier)
                                          & (R.edges_node_level == R.w4_edges_node_level)
                                          & (R.episodes == R.w4_episodes)).sum()),
        "rsa_h_max_abs_diff": float(np.nanmax([r["rsa_max_abs_diff"] for r in recs if r.get("rsa_max_abs_diff") is not None])),
        "rsa_h_epochs_compared": int(sum(r.get("rsa_n_compared", 0) for r in recs)),
        "f4_slope_vs_A1_compared": int(f4c["compared"].sum()) if len(f4c) else 0,
        "f4_slope_vs_A1_mismatch": int(f4c["mismatch"].sum()) if len(f4c) else 0,
        "f4_slope_vs_A1_max_abs_diff": float(f4c["max_abs_diff"].max()) if len(f4c) else None,
        "f4_A1_finite_but_untested (< 60 beats)": int(f4c["a1_finite_not_tested"].sum()) if len(f4c) else 0,
        "f4_undefined_surrogate_share": (float(f4c["undefined_surrogates"].sum() / max(f4c["surrogates"].sum(), 1))
                                         if len(f4c) else None)}
    # changes vs pre_pat (frozen clock-fixed state)
    V = pd.DataFrame([{"pid": r["pid"], "qc": r["qc"], "pat_qc": bool(r.get("pat_qc", False)), **r["vs_pre"]}
                      for r in recs])
    ch = {"graphs_compared": int(V.pre_found.sum()),
          "v2_layer_same_as_pre": int(V.v2_layer_same_as_pre.sum()),
          "old_nodes_unchanged (prefix)": int(V.nodes_prefix_same.sum()),
          "old_variables_unchanged (prefix)": int(V.variables_prefix_same.sum()),
          "old_beat_episodes_unchanged (prefix)": int(V.beat_episodes_prefix_same.sum()),
          "graphs with added nodes": int((V.d_nodes > 0).sum()), "added nodes (total)": int(V.d_nodes.sum()),
          "added variable nodes (total)": int(V.d_variables.sum()),
          "added beat episodes (total)": int(V.d_beat_episodes.sum())}
    for k in ("F1", "F2", "F4"):
        pre, post = V[f"{k}_pre"].fillna(0), V[f"{k}_post"].fillna(0)
        ch[f"{k} edges pre -> post"] = f"{int(pre.sum())} -> {int(post.sum())}"
        ch[f"{k} lost (pre 1, post 0)"] = int(((pre == 1) & (post == 0)).sum())
        ch[f"{k} gained (pre 0, post 1)"] = int(((pre == 0) & (post == 1)).sum())
        if f"{k}_edge_same_except_holm" in V:
            both = V[f"{k}_edge_same_except_holm"].notna()
            ch[f"{k} kept edges identical except Holm p"] = f"{int(V.loc[both, f'{k}_edge_same_except_holm'].astype(bool).sum())} / {int(both.sum())}"
            ch[f"{k} kept edges with identical pointers"] = f"{int(V.loc[both, f'{k}_ptr_same'].astype(bool).sum())} / {int(both.sum())}"
            ch[f"{k} kept edges whose Holm p changed"] = int((V.loc[both, f"{k}_holm_pre"] != V.loc[both, f"{k}_holm_post"]).sum())
    # F1 / F2 per-case real p and mean amplitude identical to pre_pat records
    pre_recs = {json.loads(x)["pid"]: json.loads(x) for x in (PRE / "build_records.jsonl").read_text().splitlines() if x.strip()}
    same_p = tot_p = 0
    for r in Q:
        pr = pre_recs.get(r["pid"])
        if not pr or "fast" not in pr:
            continue
        for rid, _, _ in FAST_TESTED:
            for cond in ("real", "N1", "N2"):
                a, b0 = r["fast"][f"{rid}|{cond}"], pr["fast"][f"{rid}|{cond}"]
                tot_p += 1
                same_p += int(a["p"] == b0["p"] and a.get("mean_A") == b0.get("mean_A") and a["n_epochs"] == b0["n_epochs"]
                              and a["n_pointers"] == b0["n_pointers"])
    ch["F1/F2 per-case p, mean amplitude, n epochs, n pointers identical to pre_pat (real, N1, N2)"] = f"{same_p} / {tot_p}"
    res["changes_vs_pre_pat"] = ch
    # graph size
    rows = []
    fams = [("v1 nodes (kg_construct)", "w4_v1_nodes"), ("v1 edges (emitted)", "w4_v1_edges"),
            ("v2 concept x tier nodes", "nodes_v1_concept_tier"), ("v2 episodes", "episodes"),
            ("action nodes", "action_nodes"), ("variable nodes (all)", "variable_nodes"),
            ("  of which beat variables (incl. pat)", "beat_variable_nodes"),
            ("beat concept x tier nodes", "nodes_beat_concept_tier")]
    fams += [(f"  {c} nodes", f"beat_nodes_{c}") for c in kv.BEAT_CONCEPTS_ALL]
    fams += [("beat episodes", "beat_episodes")] + [(f"  {c} episodes", f"beat_episodes_{c}") for c in kv.BEAT_CONCEPTS_ALL]
    fams += [("relation instances (v2, unchanged)", "relation_instances"), ("policy edges", "edges_policy"),
             ("response edges", "edges_response"), ("association edges", "edges_association"),
             ("typed edges (v2, unchanged)", "edges_node_level"), ("fast edges F1 resp->SBP", "fast_edges_F1"),
             ("fast edges F2 resp->RRI", "fast_edges_F2"), ("fast edges F4 PAT<->SBP", "fast_edges_F4"),
             ("fast edges (all)", "fast_edges"), ("typed edge pointers", "pointers"),
             ("fast edge epoch pointers", "fast_pointers")]
    for lab, col in fams:
        rows.append({"per patient (mean [95% CI])": lab, "all 3,442": fci(boot_ci(R[col], seed=1)),
                     "QC 2,954": fci(boot_ci(R.loc[R.qc, col], seed=2)),
                     "PAT-QC 2,893": fci(boot_ci(R.loc[R.pat_qc, col], seed=9)),
                     "QC median": f"{R.loc[R.qc, col].median():.0f}"})
    G = pd.DataFrame(rows)
    G.to_csv(OUT / "graph_size_table.csv", index=False)
    res["graph_size"] = rows
    tot_nodes_v2 = R.nodes_v1_concept_tier + R.action_nodes + (R.variable_nodes - R.beat_variable_nodes)
    tot_nodes_fast = tot_nodes_v2 + R.nodes_beat_concept_tier + R.beat_variable_nodes
    tot_edges_fast = R.edges_node_level + R.fast_edges
    res["totals"] = {"QC nodes v2": fci(boot_ci(tot_nodes_v2[R.qc], seed=3)),
                     "QC nodes v2-fast": fci(boot_ci(tot_nodes_fast[R.qc], seed=4)),
                     "QC edges v2": fci(boot_ci(R.loc[R.qc, "edges_node_level"], seed=5)),
                     "QC edges v2-fast": fci(boot_ci(tot_edges_fast[R.qc], seed=6)),
                     "QC v1 nodes": fci(boot_ci(R.loc[R.qc, "w4_v1_nodes"], seed=7)),
                     "QC v1 edges": fci(boot_ci(R.loc[R.qc, "w4_v1_edges"], seed=8))}
    try:
        res["totals_pre_pat"] = json.loads((PRE / "results.json").read_text())["totals"]
    except Exception:  # noqa: BLE001
        res["totals_pre_pat"] = None
    # prevalence
    prev = []
    for k, c in enumerate(kv.BEAT_CONCEPTS_ALL):
        QQ = [r for r in Q if len(r["win_obs"]) > k]
        obs = np.array([r["win_obs"][k] for r in QQ], float)
        on = np.array([r["win_on"][k] for r in QQ], float)
        has = np.array([r["counts"][f"beat_episodes_{c}"] > 0 for r in QQ], float)
        share_on = on[obs > 0] / obs[obs > 0]
        base = PQ if c == kv.PAT_CONCEPT else QQ
        nwin = sum(r["n_windows"] for r in base)
        prev.append({"concept": c, "cases observed": int((obs > 0).sum()),
                     "windows observed (share of grid)": f"{100 * obs.sum() / nwin:.1f}%" + (" (PAT-QC grid)" if c == kv.PAT_CONCEPT else ""),
                     "on / observed windows (pooled)": f"{100 * on.sum() / max(obs.sum(), 1):.2f}%",
                     "case mean share on": fci(boot_ci(share_on, seed=10 + k), pct=True),
                     "cases with >= 1 episode": fci(boot_ci(has[obs > 0] if c == kv.PAT_CONCEPT else has, seed=20 + k), pct=True)
                     + (" (of observed cases)" if c == kv.PAT_CONCEPT else ""),
                     "min on-steps per case (mean)": f"{np.mean([r['ep_steps'][k] for r in QQ]) * kv.STEP_SEC / 60:.1f}"})
    P = pd.DataFrame(prev)
    P.to_csv(OUT / "beat_concept_prevalence.csv", index=False)
    res["prevalence"] = prev
    # fast instantiation and nulls
    frows = []
    per_graph = {}
    for cond in CONDS:
        anyg = np.zeros(len(Q))
        cnt = np.zeros(len(Q))
        elig_any = np.zeros(len(Q), bool)
        for rid in REL_IDS:
            key = f"{rid}|{cond}"
            if not any(key in r["fast"] for r in Q):
                continue
            d = [r["fast"].get(key, {"eligible": False, "instantiated": False, "p": None, "n_pointers": 0,
                                     "n_epochs": 0}) for r in Q]
            el = np.array([x["eligible"] for x in d])
            inst = np.array([x["instantiated"] for x in d], float)
            pv = np.array([x["p"] if x["p"] is not None else np.nan for x in d], float)
            npt = np.array([x["n_pointers"] for x in d], float)
            nep = np.array([x["n_epochs"] for x in d], float)
            anyg += inst
            cnt += inst
            elig_any |= el
            row = {"relation": rid, "condition": cond, "eligible cases": int(el.sum()),
                   "instantiated (share of eligible)": fci(boot_ci(inst[el], seed=30), pct=True),
                   "patient p <= 0.05 (unadjusted)": f"{100 * np.mean(pv[el] <= ALPHA):.1f}%",
                   "pointers / tested epochs (eligible, pooled)": f"{100 * npt[el].sum() / max(nep[el].sum(), 1):.1f}%",
                   "median pointers (instantiated)": f"{np.median(npt[inst > 0]):.0f}" if (inst > 0).any() else "-"}
            if rid == F4:
                row["instantiated (share of all 2,954 QC)"] = fci(boot_ci(inst, seed=31), pct=True)
                row["instantiated (share of PAT-QC)"] = fci(boot_ci(np.array([x for x, r in zip(inst, Q) if r.get("pat_qc")]), seed=32), pct=True)
            frows.append(row)
        if cond != "Nb":
            per_graph[cond] = {"eligible_graphs": int(elig_any.sum()),
                               "fast_edges_per_graph (F1+F2+F4)": fci(boot_ci(cnt[elig_any], seed=40)),
                               "graphs_with_any": fci(boot_ci((anyg[elig_any] > 0).astype(float), seed=41), pct=True)}
    F = pd.DataFrame(frows)
    F.to_csv(OUT / "fast_edges_and_nulls.csv", index=False)
    res["fast"] = frows
    res["fast_per_graph"] = per_graph
    # F4 edge summaries (instantiated cases)
    f4e = [(r["pid"], r["fast"][f"{F4}|real"]) for r in PQ if f"{F4}|real" in r["fast"]]
    inst4 = [d for _, d in f4e if d["instantiated"]]
    res["f4_instantiated_summary"] = {
        "n_instantiated": len(inst4),
        "patient mean slope, median over instantiated [IQR]":
            (f"{np.median([d['mean_A'] for d in inst4]):.3f} [{np.percentile([d['mean_A'] for d in inst4], 25):.3f}, "
             f"{np.percentile([d['mean_A'] for d in inst4], 75):.3f}]") if inst4 else None,
        "pointers per instantiated edge, median [IQR]":
            (f"{np.median([d['n_pointers'] for d in inst4]):.0f} [{np.percentile([d['n_pointers'] for d in inst4], 25):.0f}, "
             f"{np.percentile([d['n_pointers'] for d in inst4], 75):.0f}]") if inst4 else None,
        "tested epochs per eligible case, median": float(np.median([d["n_epochs"] for _, d in f4e if d["eligible"]])) if f4e else None,
        "eligible but not instantiated: Holm fails / Holm passes with 0 pointers":
            f"{sum(1 for _, d in f4e if d['eligible'] and not d['instantiated'] and (d['p_holm'] or 1) > ALPHA)} / "
            f"{sum(1 for _, d in f4e if d['eligible'] and not d['instantiated'] and (d['p_holm'] or 1) <= ALPHA)}",
        "PAT-QC cases not eligible (< 3 tested epochs)": sum(1 for _, d in f4e if not d["eligible"])}
    # evidence coverage
    res["evidence"] = {"fast_edges": int(R.fast_edges.sum()),
                       "fast_edges_with_pointer": int(R.fast_edges.sum()) if not R.n_errs.any() else None,
                       "f4_edges": int(R.fast_edges_F4.sum()),
                       "f4_edges_with_pointer": int(R.fast_edges_F4.sum()) if not R.n_errs.any() else None,
                       "fast_pointers": int(R.fast_pointers.sum()), "beat_episodes": int(R.beat_episodes.sum()),
                       "prolonged_pat_episodes": int(R.beat_episodes_prolonged_pat.sum())}
    rng = np.random.default_rng(2027)
    qc_pids = R.loc[R.qc, "pid"].tolist()
    aud = sorted(rng.choice(qc_pids, size=min(100, len(qc_pids)), replace=False).tolist())
    res["pointer_audit"] = pointer_audit(aud)
    res["api_check"] = api_check([KGS / f"{p}.json.gz" for p in aud[:5]])
    # case studies by rule
    both = {r["pid"] for r in Q if r["fast"]["fast:F1:resp->sbp_beat|real"]["instantiated"]
            and r["fast"]["fast:F2:resp->rri|real"]["instantiated"]}
    k_ppv = kv.BEAT_CONCEPTS_ALL.index("high_ppv")
    k_irr = kv.BEAT_CONCEPTS_ALL.index("irregular_rr")
    k_pat = kv.BEAT_CONCEPTS_ALL.index(kv.PAT_CONCEPT)
    c1 = sorted([r for r in Q if r["pid"] in both], key=lambda r: (-r["ep_steps"][k_ppv], r["pid"]))
    c2 = sorted(Q, key=lambda r: (-r["ep_steps"][k_irr], r["pid"]))
    c3 = sorted([r for r in PQ if r["fast"].get(f"{F4}|real", {}).get("instantiated")],
                key=lambda r: (-r["ep_steps"][k_pat], r["pid"]))
    cs = []
    if c1:
        cs.append(("(i) most high_ppv episode time with F1 and F2 instantiated", c1[0]["pid"]))
    if c2:
        cs.append(("(ii) most irregular_rr episode time", c2[0]["pid"]))
    if c3:
        cs.append(("(iii) most prolonged_pat episode time with F4 instantiated (addendum B1)", c3[0]["pid"]))
    csd = OUT / "case_studies"
    csd.mkdir(exist_ok=True)
    text = ["# Track D case studies (selection rules pre-registered: protocol section 5, addendum B1 section 6)", ""]
    for lab, pid in cs:
        text += [f"Rule {lab}: **{pid}**", "", case_summary(pid)]
        (csd / f"{pid}.json").write_text(gzip.open(KGS / f"{pid}.json.gz", "rt").read())
    (csd / "summary.md").write_text("\n".join(text))
    res["case_studies"] = [{"rule": lab, "pid": pid} for lab, pid in cs]
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=jdefault))
    # report
    f4reg = next(r for r in meta["registry"] if r["relation_id"] == F4)
    L = ["# Track D: beat-level concepts and mechanistic fast edges in CLBench v2 (schema 3.1), with the PAT edge", "",
         f"Protocol sha256 `{psha[:12]}...` (eab2d583, registered before any computation) and addendum B1 "
         f"`{psha_pat[:12]}...` (protocol_pat.md, registered before any per-patient F4 or prolonged_pat quantity). "
         f"Build: {meta['n_graphs']} graphs in {meta['seconds'] / 60:.1f} min, {meta['workers']} workers. "
         f"T_low (10th percentile of track B epoch RMSSD) = {meta['t_low_rmssd_ms']:.3f} ms. Delta_pat (90th percentile "
         f"of within-case epoch PAT deviations, A1 table) = {meta.get('delta_pat_ms', float('nan')):.3f} ms. "
         f"Pre-PAT (clock-fixed) state frozen in `pre_pat/`.", "",
         "## F4 admission (addendum B1 section 1)", "", f"Status: **{f4reg['status']}**. {f4reg['basis']}.", "",
         "## Integrity", "", "```", json.dumps(res["integrity"], indent=1), "```", "",
         "## Changes vs pre_pat (clock-fixed, pre-PAT graphs)", "", "```", json.dumps(ch, indent=1), "```", "",
         "## Graph size per patient", "", md(G), "",
         "Totals (QC): " + "; ".join(f"{k} {v}" for k, v in res["totals"].items()), "",
         "Totals (QC) in pre_pat: " + ("; ".join(f"{k} {v}" for k, v in res["totals_pre_pat"].items())
                                        if res["totals_pre_pat"] else "-"), "",
         "## Beat concept prevalence (QC cases; prolonged_pat on PAT-QC cases)", "", md(P), "",
         "## Fast edges: instantiation and within-patient nulls", "", md(F.fillna("")), "",
         "Per graph (F1 + F2 + F4): " + json.dumps(per_graph), "",
         "F4 instantiated edges: " + json.dumps(res["f4_instantiated_summary"]), "",
         "Reading (pre-stated in protocol section 4 and addendum B1 section 5): N1 keeps the shifted F1 / F2 target "
         "ventilator-locked (controlled ventilation is near-periodic), so its F1 / F2 rate measures the limitation of "
         "shift nulls for periodic drivers, not the error rate of the rule. N2 is (approximately) exchangeable with each "
         "test's surrogates (implementation check: rate above 8 % flags a bug). N-b (F4 only) pairs PAT with SBP from "
         "another epoch of the same case (not exchangeable with the surrogates; Holm family uses F1 / F2 under N2).", ""]
    n3p = OUT / "n3_crosscase_exploratory.csv"
    if n3p.exists():
        n3 = pd.read_csv(n3p)
        res["n3_exploratory"] = n3.to_dict("records")
        L += ["### Exploratory (post hoc): N3 cross-case pairing null (F1 / F2; from the pre-PAT build, Holm over F1, F2)",
              "", md(n3), ""]
    afp = OUT / "irregular_rr_vs_preop_ecg_exploratory.md"
    if afp.exists():
        L += [afp.read_text().replace("# ", "### ", 1), ""]
    dgp = OUT / "irregular_rr_diagnostic_exploratory.csv"
    if dgp.exists():
        dg = pd.read_csv(dgp)
        res["irregular_rr_diagnostic_exploratory"] = dg.to_dict("records")
        L += ["### Exploratory (post hoc): what drives irregular_rr windows", "", md(dg), ""]
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=jdefault))
    L += ["## Deviations, clarifications and corrections", "",
          "- Correction: the protocol says VitalDB has no rhythm labels. clinical_data.csv has `preop_ecg`, a "
          "preoperative 12-lead interpretation (not intraoperative); it is used only in the exploratory check above.",
          "- Clarification: the interval-end rule applies to RR intervals; `high_ppv` uses the beats whose R-peak lies "
          "in the window (SBP / PP of each beat).",
          "- Clarification: fast tests also apply the wave-2 / track B phase-coverage gate (finite respiratory phase in "
          "> 80 % of the epoch's beats), which `rsa_h` used; F2 reproduces `rsa_h` exactly on all compared epochs.",
          "- Clarification: `low_hrv` needs >= 15 successive differences left after dropping the 2 largest.",
          "- Post hoc, exploratory (not in the protocol): the irregular_rr diagnostic, the preop-ECG check, and the "
          "N3 cross-case null with its frequency-match breakdown. They were added after the primary outputs were seen.",
          "- B1 clarification: an F4 surrogate whose shifted slope is undefined (< 30 valid pairs after the shift) is "
          "counted as <= the observed slope (conservative); its share is reported under Integrity.",
          "- B1: the Holm family is {F1, F2, F4} (eab2d583 section 3, 'admitted fast relations'); F1 / F2 per-epoch "
          "values are unchanged, only their Holm p (and possibly instantiation) can change; see 'Changes vs pre_pat'.",
          "- B1: build run with 8 workers (eab2d583 said <= 4); seeds are per case / epoch, so results do not depend on it.",
          ""]
    L += ["## Evidence coverage and pointer audit", "",
          "```", json.dumps({**res["evidence"], "audit": res["pointer_audit"]}, indent=1), "```", "",
          "## FastAPI loader", "", "```", json.dumps(res["api_check"], default=jdefault)[:1500], "```", "",
          "## Case studies", ""] + [f"- {x['rule']}: {x['pid']} (case_studies/summary.md)" for x in res["case_studies"]]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    log.info("report written")


# ---------------------------------------------------------------------------
# Exploratory (post hoc, not in the protocol): what drives irregular_rr?
# ---------------------------------------------------------------------------


def grid_offset(summary: dict) -> float:
    """Seconds to subtract from native (casestart-clock) beat times to reach the v2 30 s grid clock.

    Zero. The first build assumed the data/processed_vitaldb numerics grid used the VitalFile loader
    origin and subtracted ``loader_origin_minus_casestart_s`` (median 0.61 s, max 5.3 s). Both the
    track-D verifier (shift 0 reproduces 100% of grid windows, the shift 88%) and the ClosedLoopBench
    v4 run (events already on the native clock in 3,388 cases) showed the grid is on casestart, so the
    shift misplaced beat windows and pointers. The summary field is kept for provenance only.
    """
    return 0.0


def _diag_case(args) -> list[dict]:
    pid, t_low = args
    s = json.loads((NAT / "summaries" / f"{pid}.json").read_text())
    off = grid_offset(s)
    b = pd.read_parquet(NAT / "beats" / f"{pid}.parquet")
    num = pd.read_parquet(NUM / f"{pid}.parquet", columns=["window_start_sec", "hr"])
    ws = num["window_start_sec"].to_numpy(float)
    hr_mon = num["hr"].to_numpy(float)
    t = b["t_r"].to_numpy(float) - off
    rri = b["rri"].to_numpy(float)
    v = b["valid"].to_numpy(bool)
    bw = kv.beat_window_concepts(t, rri, ws, t_low)
    T = len(ws)
    rows = []
    for i in np.nonzero(bw.S[:, 0] >= 0)[0]:
        a, z = bw.beat_lo[i], bw.beat_hi[i] + 1
        rr = rri[a:z]
        ok = np.isfinite(rr) & (t[a:z] + rr / 1000.0 < ws[i] + kv.WINDOW_SEC) & (rr >= kv.RR_MIN_MS) & (rr <= kv.RR_MAX_MS)
        hr_b = 60000.0 / rr[ok].mean()
        rows.append({"pid": pid, "irr": int(bw.S[i, 0]), "dhr": abs(hr_b - hr_mon[i]) if np.isfinite(hr_mon[i]) else np.nan,
                     "valid_frac": float(v[a:z].mean()), "edge10": bool(ws[i] < ws[0] + 600 or ws[i] > ws[-1] - 600),
                     "pos": i / max(T - 1, 1)})
    return rows


def stage_diag(workers: int, n_cases: int = 300) -> None:
    meta = json.loads((OUT / "build_meta.json").read_text())
    cases = pd.read_parquet(FB / "cases.parquet")
    qc = sorted(cases.loc[cases["g4_20min"] == True, "patient_id"])  # noqa: E712
    pids = sorted(np.random.default_rng(55).choice(qc, size=min(n_cases, len(qc)), replace=False).tolist())
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for lst in ex.map(_diag_case, [(p, meta["t_low_rmssd_ms"]) for p in pids], chunksize=4) for r in lst]
    D = pd.DataFrame(rows)
    out = []
    for lab, g in (("irregular_rr on", D[D.irr == 1]), ("observed, off", D[D.irr == 0])):
        per = g.groupby("pid").agg(dhr=("dhr", "median"), vf=("valid_frac", "median"))
        out.append({"windows": lab, "n windows": len(g), "n cases": g.pid.nunique(),
                    "median |beat HR - monitor HR| (bpm)": f"{g.dhr.median():.1f}",
                    "share |dHR| > 10 bpm": f"{100 * (g.dhr > 10).mean():.1f}%",
                    "median valid-beat fraction": f"{g.valid_frac.median():.2f}",
                    "share in first/last 10 min": f"{100 * g.edge10.mean():.1f}%",
                    "case-median |dHR| (median over cases)": f"{per.dhr.median():.1f}"})
    X = pd.DataFrame(out)
    X.to_csv(OUT / "irregular_rr_diagnostic_exploratory.csv", index=False)
    (OUT / "irregular_rr_diagnostic_exploratory.md").write_text(
        "# Exploratory (post hoc): irregular_rr windows vs regular windows, 300 random QC cases (seed 55)\n\n" + md(X) + "\n")
    log.info("diag done\n%s", md(X))


# ---------------------------------------------------------------------------
# Exploratory (post hoc, not in the protocol): N3 cross-case pairing null for F1 / F2.
# The target series of case i is paired beat by beat with the respiratory phase of another case, epoch by
# epoch in order (truncated to the shorter epoch). Unlike N1 it does not keep the target locked to the paired
# respiration, while both series keep their own periodicity.
# ---------------------------------------------------------------------------


def _n3_case(args) -> dict:
    pid_t, pid_r, ci, ep_t, ep_r = args
    bt = pd.read_parquet(NAT / "beats" / f"{pid_t}.parquet")
    br = pd.read_parquet(NAT / "beats" / f"{pid_r}.parquet")
    tt_, trr = bt["t_r"].to_numpy(float), br["t_r"].to_numpy(float)
    res = {rid: [] for rid, _, _ in FAST_TESTED}
    for j in range(min(len(ep_t), len(ep_r))):
        a = int(np.searchsorted(tt_, ep_t[j]["t0"], "left"))
        z = int(np.searchsorted(tt_, ep_t[j]["t1"], "left"))
        a2 = int(np.searchsorted(trr, ep_r[j]["t0"], "left"))
        z2 = int(np.searchsorted(trr, ep_r[j]["t1"], "left"))
        n = min(z - a, z2 - a2)
        if n < kv.HARM_MIN_BEATS:
            continue
        sub = bt.iloc[a:a + n]
        ph = br["resp_phase"].to_numpy(float)[a2:a2 + n]
        if not np.isfinite(ph).mean() > 0.8:
            continue
        v = sub["valid"].to_numpy(bool)
        bid = kv.breath_ids(ph)
        for ri, (rid, col, _) in enumerate(FAST_TESTED):
            prep = kv.fast_epoch_prepare(sub[col].to_numpy(float), v, ph, bid)
            if prep is None:
                continue
            r = kv.fast_epoch_test(prep[0], prep[1], prep[2], int(bid.max()) + 1,
                                   np.random.default_rng([SEED, ci, j, ri, 4]), K_SURR, 0)
            res[rid].append({"A": r["A"], "Ak": r["Ak"], "p": r["p"]})
    dec = _patient_decisions({rid: {"real": v, "N1": [], "N2": []} for rid, v in res.items()})
    return {"pid": pid_t, "resp_from": pid_r,
            **{rid: {k: v for k, v in dec[(rid, "real")].items() if k not in ("p_bh", "pointer_idx")}
               for rid, _, _ in FAST_TESTED}}


def stage_n3(workers: int, n_cases: int = 600) -> None:
    cases = pd.read_parquet(FB / "cases.parquet")
    ep = pd.read_parquet(FB / "epochs.parquet")
    cv = ep[ep["usable"].astype(bool) & ep["controlled_vent"].astype(bool)]
    by = {pid: g.sort_values("t0")[["t0", "t1"]].to_dict("records") for pid, g in cv.groupby("patient_id")}
    qc = cases.loc[cases["g4_20min"] == True].set_index("patient_id")  # noqa: E712
    pool = sorted(p for p in qc.index if len(by.get(p, [])) >= MIN_EPOCHS)
    rng = np.random.default_rng(56)
    pids = sorted(rng.choice(pool, size=min(n_cases, len(pool)), replace=False).tolist())
    perm = np.roll(np.arange(len(pids)), 1 + int(rng.integers(0, len(pids) - 1)))  # derangement by rotation
    jobs = [(p, pids[perm[i]], int(qc.loc[p, "case_index"]), by[p], by[pids[perm[i]]]) for i, p in enumerate(pids)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        recs = list(ex.map(_n3_case, jobs, chunksize=4))
    rows = []
    for rid, _, _ in FAST_TESTED:
        el = [r[rid] for r in recs if r[rid]["eligible"]]
        inst = np.array([x["instantiated"] for x in el], float)
        rows.append({"relation": rid, "condition": "N3 cross-case pairing (exploratory)", "eligible cases": len(el),
                     "instantiated (share of eligible)": fci(boot_ci(inst, seed=60), pct=True),
                     "patient p <= 0.05 (unadjusted)": f"{100 * np.mean([x['p'] <= ALPHA for x in el]):.1f}%"})
    # frequency matching of the pair: breaths per beat (median breath rate / median HR over controlled epochs)
    g = cv.groupby("patient_id").agg(rr=("br_rate", "median"), hr=("hr_mean", "median"))
    fm = pd.DataFrame([{"pid": r["pid"], "resp_from": r["resp_from"],
                        "F1": r[FAST_TESTED[0][0]]["instantiated"], "F2": r[FAST_TESTED[1][0]]["instantiated"],
                        "rel_ratio_diff": abs(g.loc[r["pid"], "rr"] / g.loc[r["pid"], "hr"]
                                              - g.loc[r["resp_from"], "rr"] / g.loc[r["resp_from"], "hr"])
                        / (g.loc[r["pid"], "rr"] / g.loc[r["pid"], "hr"])} for r in recs])
    fm["bin"] = pd.cut(fm["rel_ratio_diff"], [0, 0.02, 0.05, 0.1, 0.2, 1, np.inf], include_lowest=True).astype(str)
    fm.to_csv(OUT / "n3_frequency_match_exploratory.csv", index=False)
    for rid, col in ((FAST_TESTED[0][0], "F1"), (FAST_TESTED[1][0], "F2")):
        for lab, m in (("breaths-per-beat within 5%", fm.rel_ratio_diff < 0.05),
                       ("breaths-per-beat differs >= 20%", fm.rel_ratio_diff >= 0.2)):
            rows.append({"relation": rid, "condition": f"N3, pairs with {lab}", "eligible cases": int(m.sum()),
                         "instantiated (share of eligible)": fci(boot_ci(fm.loc[m, col].astype(float), seed=61), pct=True),
                         "patient p <= 0.05 (unadjusted)": "-"})
    X = pd.DataFrame(rows)
    X.to_csv(OUT / "n3_crosscase_exploratory.csv", index=False)
    (OUT / "n3_crosscase_exploratory.jsonl").write_text("\n".join(json.dumps(r, default=jdefault) for r in recs))
    log.info("N3 done in %.0f s\n%s", time.time() - t0, md(X))


# ---------------------------------------------------------------------------
# Exploratory (post hoc): irregular_rr against the preoperative 12-lead ECG interpretation (clinical_data.csv
# `preop_ecg`), the only rhythm reference in VitalDB. The protocol's "no rhythm labels" was wrong for this field.
# ---------------------------------------------------------------------------


def _auc_ci(y: np.ndarray, x: np.ndarray, n: int = 2000, seed: int = 58) -> str:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return "-"
    rng = np.random.default_rng(seed)
    i1, i0 = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    bs = []
    for _ in range(n):
        j = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        bs.append(roc_auc_score(y[j], x[j]))
    return f"{roc_auc_score(y, x):.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"


def stage_afcheck(workers: int, n_nsr: int = 300) -> None:
    meta = json.loads((OUT / "build_meta.json").read_text())
    cases = pd.read_parquet(FB / "cases.parquet")
    qc = set(cases.loc[cases["g4_20min"] == True, "patient_id"])  # noqa: E712
    cl = pd.read_csv(ROOT / "data/raw/physionet.org/files/vitaldb/1.0.0/clinical_data.csv", usecols=["caseid", "preop_ecg"])
    cl["pid"] = cl["caseid"].map(lambda c: f"vdb{int(c):04d}")
    cl = cl[cl["pid"].isin(qc)]
    af = sorted(cl.loc[cl["preop_ecg"].astype(str).str.contains("Atrial fibrillation"), "pid"])
    nsr_all = sorted(cl.loc[cl["preop_ecg"] == "Normal Sinus Rhythm", "pid"])
    nsr = sorted(np.random.default_rng(57).choice(nsr_all, size=min(n_nsr, len(nsr_all)), replace=False).tolist())
    pids = af + nsr
    with ProcessPoolExecutor(workers) as ex:
        per = list(ex.map(_diag_case, [(p, meta["t_low_rmssd_ms"]) for p in pids], chunksize=2))
    rows = []
    for pid, lst in zip(pids, per):
        D = pd.DataFrame(lst)
        g = D[(D.valid_frac >= 0.8) & (~D.edge10)] if len(D) else D
        h = D[(D.dhr <= 5) & (~D.edge10)] if len(D) else D
        rows.append({"pid": pid, "group": "preop AF" if pid in af else "preop NSR",
                     "share_raw": float(D.irr.mean()) if len(D) else np.nan,
                     "share_validgate": float(g.irr.mean()) if len(g) else np.nan,
                     "share_monitorgate": float(h.irr.mean()) if len(h) else np.nan, "n_validgate": len(g),
                     "n_monitorgate": len(h)})
    X = pd.DataFrame(rows)
    X.to_csv(OUT / "irregular_rr_vs_preop_ecg_exploratory.csv", index=False)
    out = []
    for col in ("share_raw", "share_validgate", "share_monitorgate"):
        d = X.dropna(subset=[col])
        y = (d.group == "preop AF").astype(int)
        out.append({"irregular_rr window share": col.replace("share_", ""),
                    "preop AF median (n)": f"{d.loc[y == 1, col].median():.3f} ({int(y.sum())})",
                    "preop NSR median (n)": f"{d.loc[y == 0, col].median():.3f} ({int((1 - y).sum())})",
                    "AUROC AF vs NSR [stratified case bootstrap 95% CI]": _auc_ci(y.to_numpy(), d[col].to_numpy())})
    T = pd.DataFrame(out)
    (OUT / "irregular_rr_vs_preop_ecg_exploratory.md").write_text(
        "# Exploratory (post hoc): irregular_rr vs preoperative ECG (clinical_data.csv preop_ecg)\n\n"
        "validgate = windows with valid-beat fraction >= 0.8 outside the first and last 10 min of the grid "
        "(circular for rhythm: gate_beats marks beats with > 20 % RR change invalid); monitorgate = windows with "
        "|beat HR - Solar8000 HR| <= 5 bpm outside the first and last 10 min\n\n" + md(T) + "\n")
    log.info("afcheck\n%s", md(T))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "report", "all", "diag", "n3", "afcheck"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pids", nargs="*", default=None)
    a = ap.parse_args()
    if a.stage in ("build", "all"):
        stage_build(min(a.workers, 8), a.limit, a.pids)
    if a.stage == "diag":
        stage_diag(min(a.workers, 4))
    if a.stage == "n3":
        stage_n3(min(a.workers, 4))
    if a.stage == "afcheck":
        stage_afcheck(min(a.workers, 4))
    if a.stage in ("report", "all") and not (a.limit or a.pids):
        stage_report()


if __name__ == "__main__":
    main()
