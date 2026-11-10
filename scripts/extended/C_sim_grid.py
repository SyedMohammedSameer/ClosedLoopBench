"""
Extended analysis C: simulated closed-loop systems with known ground truth, every ClosedLoopBench estimator.

Runs the clinician-in-the-loop simulator (``clbench.sim.clinician_loop``) through the actual
ClosedLoopBench pipeline (``clbench.eval.closed_loop_bench``: 12 estimators + the typed composite, the
random-time and K = 8 cross-patient transplant nulls, the v3 calibrated admission rule) over a design of
feedback strength x unrecorded confounding x response nonlinearity x sampling rate x policy type, with an
active and an inert drug, and scores true-positive recovery, wrong-sign admissions, false positives on the
inert drug and under held-out transplant draws, and bias / RMSE against the exact ITT truth of the paired
twin simulation.

    PYTHONPATH=src .venv/bin/python scripts/extended/C_sim_grid.py smoke   --n 100          # one cell + profile
    PYTHONPATH=src .venv/bin/python scripts/extended/C_sim_grid.py run     --n 600 --workers 4
    PYTHONPATH=src .venv/bin/python scripts/extended/C_sim_grid.py summarize

Outputs: runs/extended/C_sim_grid/cells/<cell>/{pooled,calibrated,null_discovery,truth,eligibility}.csv + meta.json,
then cells_long.csv, cells_summary.csv, figures/*.png and tables.md.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one core per cell process (worker cap applies to cells)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from clbench.bench import sim as bsim  # noqa: E402
from clbench.eval import closed_loop_bench as clb  # noqa: E402
from clbench.sim import clinician_loop as cl  # noqa: E402

OUT = ROOT / "runs" / "extended" / "C_sim_grid"
log = logging.getLogger("C_sim_grid")

# ── design ───────────────────────────────────────────────────────────────────────────────────────────
# One configuration = simulator settings; every configuration is run with an active and an inert drug (same
# seed: identical drivers, noise and clinician draws, only the drug effect switched off). The inert cohort
# of an Emax configuration is bit-identical to the linear inert cohort (zero effect saturates to zero), so it
# is not re-run (``inert_alias``).
DEFAULT = dict(gain=1.0, w_z=0.0, emax_c50=0.0, emax_gamma=1.0, scale="2s", ant=0.0, exog_remi=1.0, exog_vent=0.3)
EMAX_C50 = 3.0  # ng/mL = the baseline target c_0: the operating point sits at the half-maximal effect
BASE_SEED = 741000
CONFIGS: dict[str, dict] = {}


def _cfg(name: str, **kw) -> None:
    CONFIGS[name] = {**DEFAULT, **kw}


# feedback x confounding grid (gain 1, w_z 0 = the paper's "recorded" clinician = the default cell)
for _g in (0.5, 1.0, 2.0):
    for _wz in (0.0, 0.5, 1.0):
        _cfg(f"g{_g:g}_wz{_wz:g}", gain=_g, w_z=_wz)
_cfg("openloop", gain=0.0)  # exogenous changes only (1/h remifentanil, 0.3/h ventilator)
_cfg("openloop_matched", gain=0.0, exog_remi=2.6, exog_vent=1.0)  # open loop with the gain-1 event rate
_cfg("emax_g1", emax_c50=EMAX_C50, emax_gamma=1.0)
_cfg("emax_g3", emax_c50=EMAX_C50, emax_gamma=3.0)
_cfg("scale30", scale="30s")
_cfg("anticip", ant=60.0)  # anticipatory policy, plan only (paper: w_ant = 1.5, H_ant = 60 s)
_cfg("anticip_field", ant=60.0, w_z=1.0)  # the paper's recorded+field+plan clinician
# second-wave cells (run if the budget allows): 30 s sampling of the grid corners, Emax under confounding
_cfg("scale30_g2_wz1", scale="30s", gain=2.0, w_z=1.0)
_cfg("scale30_g2_wz0", scale="30s", gain=2.0, w_z=0.0)
_cfg("scale30_g1_wz1", scale="30s", gain=1.0, w_z=1.0)
_cfg("emax_g3_wz1", emax_c50=EMAX_C50, emax_gamma=3.0, w_z=1.0)
_cfg("emax_g1_wz1", emax_c50=EMAX_C50, emax_gamma=1.0, w_z=1.0)

# Strong saturation: C50 = 1 ng/mL with c_0 = 3 puts the operating point well above half-maximal, so the effect per
# unit dose varies about five-fold across the realized effect-site range (2.19x at 2 ng/mL to 0.40x at 5 ng/mL)
# while the slope at c_0 is still exactly the linear coefficient. Appended after the first launch (indices, and so
# every earlier cell's seed, are unchanged).
_cfg("emax_strong", emax_c50=1.0, emax_gamma=3.0)
_cfg("emax_strong_wz1", emax_c50=1.0, emax_gamma=3.0, w_z=1.0)

WAVE1 = [f"g{g:g}_wz{wz:g}" for g in (0.5, 1.0, 2.0) for wz in (0.0, 0.5, 1.0)] + [
    "openloop", "openloop_matched", "emax_g1", "emax_g3", "scale30", "anticip", "anticip_field"]
WAVE2 = ["scale30_g2_wz1", "scale30_g2_wz0", "scale30_g1_wz1", "emax_g3_wz1", "emax_g1_wz1"]
WAVE3 = ["emax_strong", "emax_strong_wz1"]
CONFIG_INDEX = {name: i for i, name in enumerate(CONFIGS)}


def inert_alias(name: str) -> str:
    """Configuration whose inert cohort is identical to this one's (Emax does not act on an inert drug)."""
    c = CONFIGS[name]
    if c["emax_c50"] > 0:
        base = {**c, "emax_c50": 0.0, "emax_gamma": 1.0}
        for k, v in CONFIGS.items():
            if v == base:
                return k
    return name


def cell_seed(name: str, rep: int) -> int:
    return BASE_SEED + 1000 * rep + 10 * CONFIG_INDEX[name]


def cell_id(name: str, drug: str, rep: int) -> str:
    return f"{name}__{drug}__r{rep}"


def make_sim_config(c: dict, drug: str) -> cl.SimConfig:
    cfg = bsim.sim_config("recorded", drug, w_z=c["w_z"], anticipation_s=c["ant"], gain=c["gain"],
                          remi_emax_c50=c["emax_c50"], remi_emax_gamma=c["emax_gamma"])
    if (c["exog_remi"], c["exog_vent"]) != (DEFAULT["exog_remi"], DEFAULT["exog_vent"]):
        cfg = cl.with_controller(cfg, exog_remi_per_h=c["exog_remi"], exog_vent_per_h=c["exog_vent"])
    return cfg


def scale_config(scale: str) -> clb.ScaleConfig:
    """ClosedLoopBench scale configuration. At 2 s, PCMCI+ runs on 30 s block means with lags to 240 s (tau_max
    8), the paper's primary-resolution setting (ClosedLoopBench 1.1); everything else is the v3 2 s / 30 s config."""
    cfg = clb.SCALES[scale]
    if scale == "2s":
        cfg = dataclasses.replace(cfg, pcmci_step_sec=30.0)
    return cfg


def jobs_for(names: list[str], reps: int, n: int, k_cross: int, boot: int, lingam_boot: int, pcmci_cases: int,
             out: Path) -> tuple[list[dict], dict[str, int]]:
    """(jobs still to run, seed of every requested cell including the ones already finished)."""
    jobs, seeds = [], {}
    for rep in range(reps):
        for name in names:
            for drug in ("active", "inert"):
                if drug == "inert" and inert_alias(name) != name:
                    continue
                cid = cell_id(name, drug, rep)
                seeds[cid] = cell_seed(name, rep)
                if (out / "cells" / cid / "meta.json").exists():
                    continue
                jobs.append(dict(cell=cid, config=name, drug=drug, rep=rep, seed=cell_seed(name, rep), n=n,
                                 k_cross=k_cross, boot=boot, lingam_boot=lingam_boot, pcmci_cases=pcmci_cases,
                                 out=str(out), profile=False))
    return jobs, seeds


# ── one cell ─────────────────────────────────────────────────────────────────────────────────────────

def _limit_threads() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    clb.NN_TORCH_THREADS = 1


def run_cell(job: dict) -> dict:
    _limit_threads()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s")
    t0 = time.time()
    tm: dict[str, float] = {}
    c = CONFIGS[job["config"]]
    scale, n, seed, drug = c["scale"], job["n"], job["seed"], job["drug"]
    out = Path(job["out"]) / "cells" / job["cell"]
    out.mkdir(parents=True, exist_ok=True)
    scfg = make_sim_config(c, drug)
    cfg = scale_config(scale)

    cohort = cl.simulate_cohort(scfg, n, seed=seed)
    tm["simulate"] = time.time() - t0
    cases = bsim.cohort_to_cases(cohort, scale)
    pids = [x.pid for x in cases]
    rng = np.random.default_rng(seed + 1)
    crosses = clb.CROSS_ARMS[: job["k_cross"]]
    perms = {arm: {act: bsim.cross_derangement(n, rng) for act in bsim.SIM_ACTIONS.values()} for arm in crosses}
    pc_set = set(np.random.default_rng(seed + 5).permutation(n)[: job["pcmci_cases"]].tolist())
    lingam_rows = 60 if scale == "30s" else 300
    methods = clb.METHODS_V3_COMPUTE
    agg = clb.Aggregator(pids, cfg, n_boot=job["boot"], seed=0, opts=clb.V3)
    agg.nn_workers = 1
    prof: dict[str, list[float]] = {}
    t1 = time.time()
    n_pc_err = 0
    for i, case in enumerate(cases):
        cross = {arm: {act: cases[perms[arm][act][i]].events[act] for act in bsim.SIM_ACTIONS.values()} for arm in crosses}
        arms = clb.build_arms(case, None, seed=seed * 7 + i, arms=("real", "nc_random") + crosses, cross_events=cross)
        for arm, cc in arms.items():
            if job["profile"] and i < 5 and arm == "real":  # per-method cost on the first 5 real cases
                for grp in ("naive", "xcorr_v1", "granger", "varx", "pcmciplus", "varlingam", "seqtrial"):
                    ta = time.time()
                    clb.analyse_case(cc, cfg, do_pcmci=True, lingam_resid_rows=lingam_rows, seed=seed * 7 + i,
                                     methods=(grp,), opts=clb.V3)
                    prof.setdefault(grp, []).append(time.time() - ta)
            res = clb.analyse_case(cc, cfg, do_pcmci=(i in pc_set), lingam_resid_rows=lingam_rows, seed=seed * 7 + i,
                                   methods=methods, opts=clb.V3)
            n_pc_err += "pcmci_error" in res
            agg.add(case.pid, arm, res)
        if (i + 1) % 100 == 0:
            log.info("%s: %d/%d cases (%.0f s)", job["cell"], i + 1, n, time.time() - t1)
    tm["cases"] = time.time() - t1
    t2 = time.time()
    P, pp = agg.finalize(lingam_boot=job["lingam_boot"])
    tm["finalize"] = time.time() - t2
    C, N = agg.calib_table, agg.null_table
    t3 = time.time()
    truth = (bsim.itt_truth_bench(scfg, n, seed, scale, k_max=3, base=cohort) if drug == "active"
             else pd.DataFrame(columns=["rid", "patient", "k", "tau", "h", "itt"]))
    tm["truth"] = time.time() - t3
    # per-relation truth summary at every horizon (cluster bootstrap over patients)
    trows = []
    for r in bsim.SIM_RELATIONS:
        if r.kind != "response":
            continue
        for h in cfg.horizons:
            if drug == "inert":
                trows.append(dict(rid=r.rid, h=h, truth=0.0, truth_lo=0.0, truth_hi=0.0, truth_n=0, truth_sign=0))
                continue
            tt = truth[(truth.rid == r.rid) & (truth.h == h)]
            m, lo, hi = cl.cluster_bootstrap_mean(tt.itt.to_numpy(), tt.patient.to_numpy(), seed=0)
            sgn = int(np.sign(m)) if (lo > 0 or hi < 0) else 0
            trows.append(dict(rid=r.rid, h=h, truth=m, truth_lo=lo, truth_hi=hi, truth_n=len(tt), truth_sign=sgn))
    T = pd.DataFrame(trows)
    P.to_csv(out / "pooled.csv", index=False)
    if C is not None:
        C.to_csv(out / "calibrated.csv", index=False)
    if N is not None:
        N.to_csv(out / "null_discovery.csv", index=False)
    T.to_csv(out / "truth.csv", index=False)
    truth.to_csv(out / "truth_events.csv", index=False)
    E = agg.eligibility()
    E.groupby(["arm", "rid"]).agg(n_cases_with_events=("n_events", lambda x: int((x > 0).sum())),
                                  n_events=("n_events", "sum"), n_eligible=("eligible", "sum")).reset_index().to_csv(
        out / "eligibility.csv", index=False)
    ev_counts = {k: dict(n_events=int(len(cohort.events[k])), per_patient=float(len(cohort.events[k]) / n),
                         exog_share=float(cohort.events[k]["exog"].mean()) if len(cohort.events[k]) else np.nan,
                         alarm_share=float(cohort.events[k]["alarm"].mean()) if len(cohort.events[k]) else np.nan)
                 for k in cl.INPUTS}
    tm["total"] = time.time() - t0
    meta = dict(cell=job["cell"], config=job["config"], settings=c, drug=drug, rep=job["rep"], seed=seed, n=n,
                scale=scale, k_cross=job["k_cross"], boot=job["boot"], lingam_boot=job["lingam_boot"],
                pcmci_cases=len(pc_set), pcmci_errors=n_pc_err, lingam_resid_rows=lingam_rows,
                methods=list(methods), scale_config=str(cfg), sim_config=str(scfg), events=ev_counts,
                errors=agg.errors[:20], seconds=tm,
                profile={k: float(np.mean(v)) for k, v in prof.items()} if prof else None)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("%s done: %s", job["cell"], {k: round(v) for k, v in tm.items()})
    return meta


# ── commands ─────────────────────────────────────────────────────────────────────────────────────────

def cmd_smoke(args) -> None:
    out = Path(args.out) / "smoke"
    job = dict(cell=f"{args.config}__{args.drug}__smoke", config=args.config, drug=args.drug, rep=0,
               seed=cell_seed(args.config, 0), n=args.n, k_cross=args.k_cross, boot=args.boot,
               lingam_boot=args.lingam_boot, pcmci_cases=args.pcmci_cases, out=str(out), profile=True)
    meta = run_cell(job)
    print(json.dumps(dict(seconds=meta["seconds"], profile=meta["profile"], events=meta["events"],
                          pcmci_errors=meta["pcmci_errors"]), indent=2))


def cmd_run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [x for x in args.configs.split(",") if x] if args.configs else (WAVE1 + (WAVE2 if args.wave2 else []))
    for nm in names:
        if nm not in CONFIGS:
            raise SystemExit(f"unknown configuration {nm}")
    jobs, seeds = jobs_for(names, args.reps, args.n, args.k_cross, args.boot, args.lingam_boot, args.pcmci_cases, out)
    log.info("%d cells to run (%d workers): %s", len(jobs), args.workers, [j["cell"] for j in jobs])
    dpath = out / "design.json"
    prev = json.loads(dpath.read_text()) if dpath.exists() else {}
    dpath.write_text(json.dumps(dict(configs=CONFIGS, default=DEFAULT, wave1=WAVE1, wave2=WAVE2, wave3=WAVE3,
                                     n=args.n, reps=args.reps, k_cross=args.k_cross, boot=args.boot,
                                     lingam_boot=args.lingam_boot, pcmci_cases=args.pcmci_cases,
                                     seeds={**prev.get("seeds", {}), **seeds}, base_seed=BASE_SEED), indent=2))
    t0 = time.time()
    # replicate 0 first (a complete replicate early), heavy cells first within it (gain-2 cells have the most events)
    jobs.sort(key=lambda j: (j["rep"], CONFIGS[j["config"]]["scale"] == "30s", -CONFIGS[j["config"]]["gain"]))
    with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for k, meta in enumerate(pool.imap_unordered(run_cell, jobs, chunksize=1), 1):
            log.info("[%d/%d] %s finished in %.0f s (elapsed %.0f s)", k, len(jobs), meta["cell"], meta["seconds"]["total"],
                     time.time() - t0)
    log.info("all cells done in %.0f s", time.time() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "run"):
        q = sub.add_parser(name)
        q.add_argument("--out", default=str(OUT))
        q.add_argument("--n", type=int, default=600)
        q.add_argument("--k-cross", type=int, default=8)
        q.add_argument("--boot", type=int, default=200)
        q.add_argument("--lingam-boot", type=int, default=50)
        q.add_argument("--pcmci-cases", type=int, default=60)
    sub.choices["smoke"].add_argument("--config", default="g1_wz0")
    sub.choices["smoke"].add_argument("--drug", default="active")
    sub.choices["run"].add_argument("--workers", type=int, default=4)
    sub.choices["run"].add_argument("--reps", type=int, default=1)
    sub.choices["run"].add_argument("--configs", default="", help="comma-separated configuration names (default: wave 1)")
    sub.choices["run"].add_argument("--wave2", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        from clbench.extended.c_sim_grid_summary import summarize

        summarize(Path(args.out))


if __name__ == "__main__":
    main()
