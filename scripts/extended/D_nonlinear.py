"""
Extended analysis D: nonlinear robustness of the ClosedLoopBench conclusions.

Stages (run from the repo root; every stage writes to runs/extended/D_nonlinear/<stage>/):

  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage pcmci --test parcorr --workers 4   # validation
  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage pcmci --test gpdc    --workers 4   # D1a
  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage pcmci --test cmiknn  --workers 4   # D1b
  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage xcorr --workers 4                  # D1d (+ Pearson check)
  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage gbm   --workers 4 --n-cases 1000   # D1c
  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage varx  --workers 4 --dose-bins 3    # D2 (1 = paper check)
  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage pat   --workers 4                  # D3
  PYTHONPATH=src .venv/bin/python scripts/extended/D_nonlinear.py --stage report                             # tables

Library: src/clbench/extended/nonlinear.py (cohort, arms, seeds and bootstrap weights of the paper's v4 run;
the new estimators; scoring through closed_loop_bench.calibrated_admission_table).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clbench.eval import closed_loop_bench as clb  # noqa: E402
from clbench.extended import nonlinear as nl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("D_nonlinear")

OUT = ROOT / "runs" / "extended" / "D_nonlinear"
PAPER = nl.PAPER_RUN
PAT_RUN = ROOT / "runs" / "iclr" / "pat_vitaldb"
NATIVE = ROOT / "data" / "processed_vitaldb" / "native"
TIERS = (("gas_exchange", "Gas (5)"), ("drug", "Drug (11)"), ("drug_derived", "Vol. (2)"), ("policy", "Policy (10)"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", required=True, choices=["pcmci", "xcorr", "gbm", "varx", "pat", "report"])
    p.add_argument("--test", default="gpdc", choices=list(nl.PCMCI_TESTS))
    p.add_argument("--arms", default="all", help="'all' or a comma list, e.g. real,nc_random,nc_cross,nc_cross2")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="first N cases (smoke tests)")
    p.add_argument("--n-cases", type=int, default=500, help="gbm: fixed seeded case subsample (0 = all)")
    p.add_argument("--stride-sec", type=float, default=40.0, help="gbm: anchor spacing (s)")
    p.add_argument("--model", default="gbm", choices=["gbm", "linear"], help="gbm: boosted trees or the ridge linear twin")
    p.add_argument("--folds", type=int, default=3, help="gbm: case-grouped folds")
    p.add_argument("--max-iter", type=int, default=200, help="gbm: boosting iterations (early stopping)")
    p.add_argument("--sig-samples", type=int, default=500, help="pcmci: GPDC null draws / CMIknn permutations")
    p.add_argument("--knn", type=float, default=0.2, help="pcmci: CMIknn neighbour fraction")
    p.add_argument("--gp-opt", default="default", choices=["default", "maxiter5", "none"],
                   help="pcmci gpdc: GP hyperparameter optimiser (default: sklearn L-BFGS-B to convergence)")
    p.add_argument("--dose-bins", type=int, default=3, help="varx: |step| quantile bins per scored channel (1 = paper)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--tag", default="", help="suffix of the stage output directory")
    p.add_argument("--score-only", action="store_true",
                   help="pcmci: do not run PCMCI+; score the cases already complete in <out>/per_case.parquet")
    return p.parse_args()


def df_to_md(df: pd.DataFrame, nd: int = 3) -> str:
    def f(x):
        if isinstance(x, (float, np.floating)):
            return "nan" if not np.isfinite(x) else (f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{nd}f}")
        return str(x)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def arms_of(args, cohort: nl.Cohort) -> tuple[str, ...]:
    return cohort.arms if args.arms == "all" else tuple(a for a in args.arms.split(",") if a)


def save_scored(out_dir: Path, store: dict, cfg, meta: dict, methods=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "store.pkl", "wb") as f:
        pickle.dump(store, f)
    pooled, calib, nulls, lb = nl.score_store(store, cfg, methods)
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    if calib is not None:
        calib.to_csv(out_dir / "calibrated.csv", index=False)
    nulls.to_csv(out_dir / "null_discovery.csv", index=False)
    lb.to_csv(out_dir / "leaderboard.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s", out_dir)
    print(df_to_md(lb[[c for c in ("method", "score", "cal_correct", "cal_wrong", "n_relations", "null_cross_fa_per_draw",
                                  "null_random_fa", "uncal_null_cross_distinct_sig_mean", "uncal_null_cross_distinct_mean",
                                  "uncal_null_random_sig") if c in lb]]))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# pcmci
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

METHOD_OF_TEST = {"parcorr": "pcmciplus_parcorr_check", "gpdc": "pcmciplus_gpdc", "gpdc_exact": "pcmciplus_gpdc_exact",
                  "cmiknn": "pcmciplus_cmiknn", "cmiknn_shuffle": "pcmciplus_cmiknn_shuffle"}


def _gp_opt(name: str):
    if name == "default":
        return None
    if name == "none":
        return {"optimizer": None}
    from scipy import optimize

    def opt5(obj_func, initial_theta, bounds):
        res = optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                                options={"maxiter": 5})
        return res.x, res.fun

    return {"optimizer": opt5}


def _pcmci_job(args):
    job, test_name, arms, sig_samples, knn, gp_opt = args
    t0 = time.time()
    cfg = clb.SCALES[job["scale"]]
    all_arms = nl.case_arms(job)
    out = {}
    for arm in arms:
        case = all_arms[arm]
        rec = {"elig": nl.eligibility(case), "pcmci": {}, "error": None, "seconds": 0.0}
        t1 = time.time()
        try:
            test = nl.make_ci_test(test_name, sig_samples=sig_samples, knn=knn, seed=job["seed"])
            if test_name == "gpdc_exact" and gp_opt != "default":
                test.gauss_pr.gp_params = _gp_opt(gp_opt)
            rec["pcmci"] = clb.pcmci_case(case, cfg, opts=clb.V4, cond_ind_test=test, sign_fn=nl.SIGN_FN[test_name])
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
        rec["seconds"] = time.time() - t1
        out[arm] = rec
    return job["pid"], out, time.time() - t0


def stage_pcmci(args) -> None:
    t0 = time.time()
    C = nl.load_cohort(seed=args.seed)
    arms = arms_of(args, C)
    pcs = sorted(C.pc_set)
    if args.limit:
        pcs = pcs[: args.limit]
    method = METHOD_OF_TEST[args.test]
    out_dir = args.out / f"pcmci_{args.test}{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("pcmci %s: %d cases x %d arms, sig_samples %d, workers %d", args.test, len(pcs), len(arms), args.sig_samples,
             args.workers)
    rows, elig, errors, secs = [], [], [], []
    done_pids: set = set()
    if (out_dir / "per_case.parquet").exists():  # resume: keep the cases that have every requested arm
        P0 = pd.read_parquet(out_dir / "per_case.parquet")
        have = P0.groupby("pid").arm.apply(lambda a: set(a))
        done_pids = {p for p, a in have.items() if set(arms) <= a and p in set(pcs)}
        P0 = P0[P0.pid.isin(done_pids)]
        rows = [tuple(r) for r in P0[["arm", "pid", "case", "cause", "effect", "val", "p", "det", "tau"]].itertuples(index=False)]
        for pid in sorted(done_pids):  # eligibility is a property of the case and arm (cheap to rebuild)
            for arm, case in nl.case_arms(C.job(pid)).items():
                if arm in arms:
                    for rid, (n_, cov) in nl.eligibility(case).items():
                        elig.append((arm, C.pos[pid], rid, n_, cov))
        if (out_dir / "seconds.csv").exists():
            secs = [tuple(r) for r in pd.read_csv(out_dir / "seconds.csv").itertuples(index=False) if r.pid in done_pids]
        log.info("resuming: %d of %d cases already complete in %s", len(done_pids), len(pcs), out_dir / "per_case.parquet")
    jobs = [(C.job(pid), args.test, arms, args.sig_samples, args.knn, args.gp_opt) for pid in pcs if pid not in done_pids]
    if args.score_only:
        log.info("score-only: %d complete cases, %d not run", len(done_pids), len(jobs))
        pcs = sorted(done_pids)
        jobs = []
    ctx = get_context("spawn")
    with ctx.Pool(max(1, min(args.workers, len(jobs) or 1)), initializer=nl.worker_init) as pool:
        for k, (pid, res, dt) in enumerate(pool.imap(_pcmci_job, jobs, chunksize=1), 1):
            c = C.pos[pid]
            for arm, rec in res.items():
                secs.append((arm, pid, rec["seconds"]))
                for rid, (n, cov) in rec["elig"].items():
                    elig.append((arm, c, rid, n, cov))
                if rec["error"]:
                    errors.append((arm, pid, rec["error"]))
                for (cause, effect), (val, p, det, tau) in rec["pcmci"].items():
                    rows.append((arm, pid, c, cause, effect, val, p, bool(det), tau))
            log.info("%d/%d cases done (%.0f s elapsed, last case %.0f s)", k, len(jobs), time.time() - t0, dt)
            if k % 2 == 0 or k == len(jobs):
                pd.DataFrame(rows, columns=["arm", "pid", "case", "cause", "effect", "val", "p", "det", "tau"]).to_parquet(
                    out_dir / "per_case.parquet", index=False)
                pd.DataFrame(secs, columns=["arm", "pid", "seconds"]).to_csv(out_dir / "seconds.csv", index=False)
    P = pd.DataFrame(rows, columns=["arm", "pid", "case", "cause", "effect", "val", "p", "det", "tau"])
    P.to_parquet(out_dir / "per_case.parquet", index=False)
    E = pd.DataFrame(elig, columns=["arm", "case", "rid", "n_events", "coverage"])
    el = {(a, r): set(g.case[[nl.is_eligible(n_, c_) for n_, c_ in zip(g.n_events, g.coverage)]])
          for (a, r), g in E.groupby(["arm", "rid"])}
    n = len(C.pids)
    store, aux = {}, []
    for r in clb.RELATIONS:
        for arm, g in P[(P.cause == r.cause) & (P.effect == r.effect)].groupby("arm"):
            g = g[g.case.isin(el.get((arm, r.rid), set()))]
            if len(g) == 0:
                continue
            vals = np.zeros(n)
            cnt = np.zeros(n)
            vals[g.case.values] = g.val.values
            cnt[g.case.values] = 1
            est, reps = clb.ratio_boot(vals, cnt, C.W)
            store[(arm, method, r.rid, 0.0)] = (est, reps, len(g))
            aux.append(dict(arm=arm, method=method, rid=r.rid, n_cases=len(g), frac_detected=float(g.det.mean()),
                            frac_det_correct=float((g.det & (np.sign(g.val) == r.expected)).mean()),
                            frac_det_wrong=float((g.det & (np.sign(g.val) == -r.expected)).mean()),
                            frac_sign_expected=float((np.sign(g.val) == r.expected).mean())))
    pd.DataFrame(aux).to_csv(out_dir / "detection.csv", index=False)
    S = pd.DataFrame(secs, columns=["arm", "pid", "seconds"])
    S.to_csv(out_dir / "seconds.csv", index=False)
    meta = dict(stage="pcmci", test=args.test, method=method, n_cases=len(pcs), n_resumed=len(done_pids), score_only=args.score_only,
                arms=list(arms),
                sig_samples=args.sig_samples,
                knn=args.knn, gp_opt=args.gp_opt, workers=args.workers, seed=args.seed, shims=nl.install_shims(),
                seconds_total=time.time() - t0, seconds_per_case_arm_median=float(S.seconds.median()),
                seconds_per_case_arm_mean=float(S.seconds.mean()), errors=errors[:50], n_errors=len(errors),
                command=" ".join(sys.argv), config=str(C.cfg))
    save_scored(out_dir, store, C.cfg, meta, (method,))
    if args.test == "parcorr" and args.arms == "all" and not args.limit:
        _validate_pcmci(out_dir, store, method)


def _validate_pcmci(out_dir: Path, store: dict, method: str) -> None:
    """The ParCorr path must reproduce the paper's pooled PCMCI+ rows (every arm) and its calibrated table."""
    PP = pd.read_csv(PAPER / "pooled.csv")
    PP = PP[(PP.method == "pcmciplus") & (PP.arm.isin(clb.ARMS_V3))]
    rows = []
    for (arm, m, rid, h), (est, reps, n) in store.items():
        q = PP[(PP.arm == arm) & (PP.rid == rid)]
        if len(q):
            lo, hi = clb.ci(reps)
            rows.append(dict(arm=arm, rid=rid, est=est, est_paper=float(q.est.iloc[0]), lo=lo, lo_paper=float(q.lo.iloc[0]),
                             hi=hi, hi_paper=float(q.hi.iloc[0]), n=n, n_paper=int(q.n_cases.iloc[0])))
    V = pd.DataFrame(rows)
    V["max_abs_diff"] = np.max(np.abs(V[["est", "lo", "hi"]].to_numpy() - V[["est_paper", "lo_paper", "hi_paper"]].to_numpy()), 1)
    V.to_csv(out_dir / "validation_vs_paper.csv", index=False)
    C1 = pd.read_csv(out_dir / "calibrated.csv")
    C0 = pd.read_csv(PAPER / "calibrated.csv")
    C0 = C0[C0.method == "pcmciplus"]
    M = C1.merge(C0, on=["input", "rid"], suffixes=("", "_paper"))
    same_calls = int((M.call == M.call_paper).sum())
    summary = dict(n_rows=len(V), n_rows_paper=len(PP), max_abs_diff=float(V.max_abs_diff.max()),
                   n_cases_equal=bool((V.n == V.n_paper).all()), calibrated_rows=len(M), calibrated_same_call=same_calls,
                   calibrated_max_abs_diff_est=float(np.abs(M.est - M.est_paper).max()))
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("validation vs paper: %s", summary)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# xcorr
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def stage_xcorr(args) -> None:
    t0 = time.time()
    C = nl.load_cohort(seed=args.seed, limit=args.limit)
    out_dir = args.out / f"xcorr{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = nl.XcorrAccumulator(C)
    jobs = [(C.job(pid), True) for pid in C.pids]
    ctx = get_context("spawn")
    times = []
    with ctx.Pool(args.workers, initializer=nl.worker_init) as pool:
        for k, (pid, res, dt) in enumerate(pool.imap(nl.xcorr_job, jobs, chunksize=4), 1):
            acc.add(pid, res)
            times.append(dt)
            if k % 200 == 0 or k == len(jobs):
                log.info("%d/%d cases (%.0f s, median %.2f s/case)", k, len(jobs), time.time() - t0, np.median(times))
    store = acc.estimates("rank", "xcorr_rank")
    store.update(acc.estimates("pearson", "xcorr_check"))
    methods = tuple(sorted({k[1] for k in store}))
    meta = dict(stage="xcorr", n_cases=len(C.pids), arms=list(C.arms), workers=args.workers, seed=args.seed,
                seconds_total=time.time() - t0, median_case_seconds=float(np.median(times)), command=" ".join(sys.argv))
    save_scored(out_dir, store, C.cfg, meta, methods)
    if not args.limit:
        PP = pd.read_csv(PAPER / "pooled.csv")
        rows = []
        for (arm, m, rid, h), (est, reps, n) in store.items():
            if not m.startswith("xcorr_check"):
                continue
            pm = "xcorr_" + m.split("_")[-1]
            q = PP[(PP.arm == arm) & (PP.method == pm) & (PP.rid == rid)]
            if len(q):
                lo, hi = clb.ci(reps)
                rows.append(dict(arm=arm, method=pm, rid=rid, est=est, est_paper=float(q.est.iloc[0]), lo=lo,
                                 lo_paper=float(q.lo.iloc[0]), hi=hi, hi_paper=float(q.hi.iloc[0]), n=n, n_paper=int(q.n_cases.iloc[0])))
        V = pd.DataFrame(rows)
        V["max_abs_diff"] = np.max(np.abs(V[["est", "lo", "hi"]].to_numpy() - V[["est_paper", "lo_paper", "hi_paper"]].to_numpy()), 1)
        V.to_csv(out_dir / "validation_vs_paper.csv", index=False)
        log.info("Pearson path vs paper (real arm): %d rows, max |diff| %.3g, n equal %s", len(V), V.max_abs_diff.max(),
                 bool((V.n == V.n_paper).all()))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# varx (dose bins)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def stage_varx(args) -> None:
    t0 = time.time()
    C = nl.load_cohort(seed=args.seed, limit=args.limit)
    arms = arms_of(args, C)
    nb = args.dose_bins
    out_dir = args.out / ((f"varx_dosebins{nb}" if nb > 1 else "varx_check") + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = nl.dose_bin_edges(C.ev, nb)
    cols, specs = nl.dosebin_columns(C.cfg, edges, nb)
    prefix = "varx_dosebins" if nb > 1 else "varx_check"
    pd.DataFrame([dict(action=a, dir=c, key=k, lo=lo, hi=hi) for a, c, k, lo, hi in specs]).to_csv(out_dir / "channels.csv", index=False)
    log.info("varx: %d dose bins, %d columns, %d arms, %d cases", nb, cols["_total"].stop, len(arms), len(C.pids))
    store: dict = {}
    if (out_dir / "store_partial.pkl").exists():
        with open(out_dir / "store_partial.pkl", "rb") as f:
            store = pickle.load(f)
        log.info("resuming: %d entries", len(store))
    ctx = get_context("spawn")
    arm_secs = {}
    for arm in arms:
        if any(k[0] == arm for k in store):
            continue
        t1 = time.time()
        acc = nl.VarxAccumulator(C)
        jobs = [(C.job(pid), (arm,), specs) for pid in C.pids]
        with ctx.Pool(args.workers, initializer=nl.worker_init) as pool:
            for k, (pid, res, dt) in enumerate(pool.imap(nl.varx_job, jobs, chunksize=4), 1):
                acc.add(pid, res)
                if k % 500 == 0 or k == len(jobs):
                    log.info("arm %s: %d/%d cases (%.0f s)", arm, k, len(jobs), time.time() - t1)
        store.update(acc.step_responses(arm, C.cfg, cols, specs, nb, prefix))
        arm_secs[arm] = time.time() - t1
        with open(out_dir / "store_partial.pkl", "wb") as f:
            pickle.dump(store, f)
        del acc
    methods = tuple(sorted({k[1] for k in store}))
    meta = dict(stage="varx", dose_bins=nb, edges={f"{a}:{c}": list(map(float, e)) for (a, c), e in edges.items()},
                n_columns=cols["_total"].stop, n_cases=len(C.pids), arms=list(arms), workers=args.workers, seed=args.seed,
                seconds_total=time.time() - t0, seconds_per_arm=arm_secs, command=" ".join(sys.argv))
    save_scored(out_dir, store, C.cfg, meta, methods)
    if nb == 1 and not args.limit:
        PP = pd.read_csv(PAPER / "pooled.csv")
        rows = []
        for (arm, m, rid, h), (est, reps, n) in store.items():
            q = PP[(PP.arm == arm) & (PP.method == "varx") & (PP.rid == rid) & (PP.h == h)]
            if len(q):
                lo, hi = clb.ci(reps)
                rows.append(dict(arm=arm, rid=rid, h=h, est=est, est_paper=float(q.est.iloc[0]), lo=lo, lo_paper=float(q.lo.iloc[0]),
                                 hi=hi, hi_paper=float(q.hi.iloc[0]), n=n, n_paper=int(q.n_cases.iloc[0])))
        V = pd.DataFrame(rows)
        V["max_abs_diff"] = np.max(np.abs(V[["est", "lo", "hi"]].to_numpy() - V[["est_paper", "lo_paper", "hi_paper"]].to_numpy()), 1)
        V["rel_diff"] = V.max_abs_diff / np.maximum(np.abs(V.est_paper), 1e-9)
        V.to_csv(out_dir / "validation_vs_paper.csv", index=False)
        log.info("VARX path vs paper: %d rows, max |diff| %.3g (max rel %.3g), n equal %s", len(V), V.max_abs_diff.max(),
                 V.rel_diff.max(), bool((V.n == V.n_paper).all()))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# gbm
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def stage_gbm(args) -> None:
    t0 = time.time()
    C = nl.load_cohort(seed=args.seed, limit=args.limit)
    arms = arms_of(args, C)
    nl.GBM_MODEL = args.model
    method = "granger_gbm" if args.model == "gbm" else "granger_gain_linear"
    out_dir = args.out / (f"gbm{args.tag}" if args.model == "gbm" else f"gbm_linear{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = list(C.pids)
    if args.n_cases and args.n_cases < len(pids):
        rng = np.random.default_rng(args.seed + 11)
        pids = sorted(rng.permutation(pids)[: args.n_cases].tolist())
    nl.GBM_STRIDE_SEC = float(args.stride_sec)
    feat = nl.gbm_feature_names(C.cfg)
    targets = nl.gbm_targets()
    log.info("gbm: %d cases, %d arms, %d features, %d targets, %d folds", len(pids), len(arms), len(feat), len(targets), args.folds)
    ctx = get_context("spawn")
    store: dict = {}
    aux: list[dict] = []
    if (out_dir / "store_partial.pkl").exists():
        with open(out_dir / "store_partial.pkl", "rb") as f:
            store, aux = pickle.load(f)
        log.info("resuming: %d entries", len(store))
    n = len(C.pids)
    arm_secs = {}
    for arm in arms:
        if any(k[0] == arm for k in store):
            continue
        t1 = time.time()
        jobs = [(C.job(pid), (arm,)) for pid in pids]
        Xs, Ys, cs, pres, el = [], {t: [] for t in targets}, [], [], []
        os.environ["OMP_NUM_THREADS"] = "1"
        with ctx.Pool(args.workers, initializer=nl.worker_init) as pool:
            for k, (pid, res, dt) in enumerate(pool.imap(nl.gbm_job, jobs, chunksize=4), 1):
                d = res[arm]
                if d is None:
                    continue
                Xs.append(d["X"])
                cs.append(np.full(d["n"], C.pos[pid], dtype=np.int64))
                for t in targets:
                    Ys[t].append(d["Y"][t])
                pres.append((C.pos[pid], d["present"]))
                el.append((C.pos[pid], d["elig"]))
                if k % 200 == 0 or k == len(jobs):
                    log.info("arm %s: %d/%d designs (%.0f s)", arm, k, len(jobs), time.time() - t1)
        X = np.vstack(Xs)
        cidx = np.concatenate(cs)
        del Xs
        log.info("arm %s: design %s (%.0f MB)", arm, X.shape, X.nbytes / 1e6)
        os.environ["OMP_NUM_THREADS"] = str(args.workers)
        present = {c: p for c, p in pres}
        elig = {c: e for c, e in el}
        bins = nl.gbm_bins(C.cfg)
        nf = 1 + len(bins)
        series = nl.gbm_series_names()
        col_of = {s: np.arange(i * nf, (i + 1) * nf) for i, s in enumerate(series)}
        for kind, s, h in targets:
            y = np.concatenate(Ys[(kind, s, h)])
            rels = [r for r in clb.RELATIONS if r.kind == kind and (r.vital if kind == "response" else r.action) == s]
            # rows of cases where the target series is present
            rows_ok = np.array([present[c][s] for c in cidx]) & np.isfinite(y)
            t2 = time.time()
            full = None
            by_src: dict[str, tuple[dict, dict]] = {}
            for r in rels:
                src = r.action if kind == "response" else r.vital
                sc = col_of[src]
                if src in by_src:  # same source, same target: identical statistic (direction-agnostic, as Granger)
                    st, row = by_src[src]
                    hh = clb.primary_h(r, method, C.cfg)
                    store[(arm, method, r.rid, hh)] = (st["est"], st["reps"], st["n_cases"])
                    aux.append({**row, "rid": r.rid, "seconds": 0.0})
                    continue
                # eligible rows: the relation's eligibility (>= 3 events of the direction, vital coverage) is a
                # per-case property; the GBM pools every row of cases where both series are present
                m = rows_ok & np.array([present[c][src] for c in cidx]) & np.isfinite(X[:, sc[0]])
                if m.sum() < 1000 or len(np.unique(cidx[m])) < 10:
                    continue
                delta = 0.5 * float(np.nanstd(X[m][:, sc[0]]))
                if not np.isfinite(delta) or delta <= 0:
                    continue
                Xm, ym, cm = X[m], y[m].astype(float), cidx[m]
                key = (tuple(np.flatnonzero(m)[:5].tolist()), int(m.sum()))
                if full is None or full[0] != key:  # one full fit per (target, row set)
                    oof, models = nl.gbm_fit_predict(Xm, ym, cm, args.folds, args.seed, max_iter=args.max_iter)
                    full = (key, oof, models)
                st = nl.gbm_relation_stats(Xm, ym, cm, n, C.W, sc, delta, args.folds, args.seed, oof_full=full[1],
                                           models_full=full[2])
                hh = clb.primary_h(r, method, C.cfg)
                store[(arm, method, r.rid, hh)] = (st["est"], st["reps"], st["n_cases"])
                lo_g, hi_g = clb.ci(st["gain_reps"])
                lo_a, hi_a = clb.ci(st["reps"])
                row = dict(arm=arm, rid=r.rid, target=s, horizon=h, source=src, n_rows=st["n_rows"], n_cases=st["n_cases"],
                           ame=st["est"], ame_lo=lo_a, ame_hi=hi_a, gain=st["gain"], gain_lo=lo_g, gain_hi=hi_g,
                           r2_full=st["r2_full"], delta=delta, seconds=time.time() - t2)
                by_src[src] = (st, row)
                aux.append(row)
                log.info("arm %s %s: AME %.4g gain %.4g [%.3g, %.3g] R2 %.3f (%d rows, %d cases, %.0f s)", arm, r.rid, st["est"],
                         st["gain"], lo_g, hi_g, st["r2_full"], st["n_rows"], st["n_cases"], time.time() - t2)
        arm_secs[arm] = time.time() - t1
        with open(out_dir / "store_partial.pkl", "wb") as f:
            pickle.dump((store, aux), f)
        pd.DataFrame(aux).to_csv(out_dir / "gain_ame.csv", index=False)
        del X
    meta = dict(stage="gbm", model=args.model, method=method, n_cases=len(pids), n_cases_cohort=len(C.pids), case_ids=pids,
                arms=list(arms), folds=args.folds,
                max_iter=args.max_iter, stride_sec=nl.GBM_STRIDE_SEC,
                policy_h=nl.GBM_POLICY_H, features=feat, workers=args.workers, seed=args.seed, seconds_total=time.time() - t0,
                seconds_per_arm=arm_secs, command=" ".join(sys.argv))
    pd.DataFrame(aux).to_csv(out_dir / "gain_ame.csv", index=False)
    save_scored(out_dir, store, C.cfg, meta, (method,))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# pat (D3)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def _load_script(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_pat(args) -> None:
    t0 = time.time()
    out_dir = args.out / f"pat{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    Q = pd.read_parquet(PAT_RUN / "cases.parquet")
    keep = set(Q.loc[Q["q3_cover"], "patient_id"])
    E = pd.read_parquet(PAT_RUN / "epochs_pat.parquet", columns=["patient_id", "epoch_index", "t0", "t1", "pat_sbp_slope", "pat_sbp_corr"])
    E = E[E.patient_id.isin(keep) & np.isfinite(E.pat_sbp_slope)]
    by = {p: list(zip(g.epoch_index.astype(int), g.t0.astype(float), g.t1.astype(float))) for p, g in E.groupby("patient_id")}
    pids = sorted(by)
    if args.limit:
        pids = pids[: args.limit]
    log.info("pat: %d cases, %d epochs", len(pids), sum(len(by[p]) for p in pids))
    jobs = [(p, by[p], str(NATIVE)) for p in pids]
    rows, errors = [], []
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap(nl.pat_case_job, jobs, chunksize=4), 1):
            rows.extend(res["rows"])
            if res["error"]:
                errors.append((res["patient_id"], res["error"]))
            if k % 250 == 0 or k == len(jobs):
                log.info("%d/%d cases (%.0f s)", k, len(jobs), time.time() - t0)
    R = pd.DataFrame(rows)
    R.to_parquet(out_dir / "epochs.parquet", index=False)
    # validation: the OLS slope recomputed here equals the paper's per-epoch slope
    M = R.merge(E, on=["patient_id", "epoch_index"], suffixes=("", "_paper"))
    val = dict(n_epochs=len(M), max_abs_diff_slope=float(np.nanmax(np.abs(M.pat_sbp_slope - M.pat_sbp_slope_paper))),
               max_abs_diff_corr=float(np.nanmax(np.abs(M.pat_sbp_corr - M.pat_sbp_corr_paper))))
    s54 = _load_script("s54", "54_fast_edges_vitaldb.py")
    s52 = _load_script("s52", "52_fast_edge_specificity.py")
    med = R.groupby("patient_id")[["pat_sbp_slope", "pat_sbp_corr", "spearman", "iso_slope", "iso_dir"]].median()
    signs = []
    for col, label in (("pat_sbp_slope", "OLS slope (ms/mmHg) [paper statistic]"), ("pat_sbp_corr", "Pearson r [paper]"),
                       ("spearman", "Spearman rho (rank)"), ("iso_slope", "isotonic average slope (ms/mmHg)"),
                       ("iso_dir", "isotonic direction (sign of the better monotone fit)")):
        m = med[col].to_numpy(float)
        m = m[np.isfinite(m)]
        cs = s54.boot_cases(m, lambda a: np.mean(a < 0))
        mm = s54.boot_cases(m, np.median)
        ep = R[col].to_numpy(float)
        signs.append(dict(statistic=label, column=col, n_cases=len(m), n_epochs=int(np.isfinite(ep).sum()),
                          cases_negative=cs[0], cases_negative_lo=cs[1], cases_negative_hi=cs[2],
                          epochs_negative=float(np.mean(ep[np.isfinite(ep)] < 0)), median_case_median=mm[0],
                          median_lo=mm[1], median_hi=mm[2]))
    S = pd.DataFrame(signs)
    S.to_csv(out_dir / "sign_share.csv", index=False)
    print(df_to_md(S))
    # specificity: the registered split-half rule (S1, adjusted for epoch mean HR and SBP) on the paper's frame
    # with the new per-epoch statistics merged in and winsorised like the others
    spec_rows = []
    if (PAT_RUN / "spec_frame.parquet").exists() and not args.limit:
        F = pd.read_parquet(PAT_RUN / "spec_frame.parquet")
        F = F.merge(R[["patient_id", "epoch_index", "spearman", "iso_slope"]], on=["patient_id", "epoch_index"], how="left")
        F = s52.winsorise(F, ["spearman", "iso_slope"])
        for col in ("pat_sbp_slope", "pat_sbp_corr", "spearman", "iso_slope"):
            for adj, covs in (("raw", []), ("adj_hs", list(s52.COV_HS))):
                t1 = time.time()
                groups, _ = s52.build_groups(F, col, covs, "S1")
                res = s52.split_stats(groups, seed=0)
                spec_rows.append(dict(column=col, adjust=adj, split="S1", seconds=time.time() - t1, **res))
                log.info("spec %s %s: r_EL %.3f [%.3f, %.3f], n %d (%.0f s)", col, adj, res.get("r_early_late", np.nan),
                         res.get("r_early_late_lo", np.nan), res.get("r_early_late_hi", np.nan), res.get("n_patients", 0),
                         time.time() - t1)
        SP = pd.DataFrame(spec_rows)
        SP.to_csv(out_dir / "specificity.csv", index=False)
    meta = dict(stage="pat", n_cases=len(pids), n_epochs=len(R), errors=errors[:20], n_errors=len(errors), validation=val,
                workers=args.workers, seconds_total=time.time() - t0, command=" ".join(sys.argv))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("validation: %s", val)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# report
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

COUNTERPART = {"pcmciplus_gpdc": "pcmciplus", "pcmciplus_cmiknn": "pcmciplus", "pcmciplus_parcorr_check": "pcmciplus",
               "granger_gbm": "granger", "granger_gain_linear": "granger", "xcorr_rank_v1": "xcorr_v1", "xcorr_rank_dir": "xcorr_dir", "xcorr_rank_pos": "xcorr_pos",
               "xcorr_check_v1": "xcorr_v1", "xcorr_check_dir": "xcorr_dir", "xcorr_check_pos": "xcorr_pos",
               "varx_dosebins": "varx", "varx_dosebins_b1": "varx", "varx_dosebins_b2": "varx", "varx_dosebins_b3": "varx",
               "varx_check": "varx"}


def table1_row(method: str, calib: pd.DataFrame, lb: pd.DataFrame) -> dict:
    """Table 1 metrics of one method: uncalibrated null discoveries per transplant draw over distinct estimates,
    calibrated expected/wrong admissions by tier, held-out transplant and random-time false admissions."""
    L = lb[lb.method == method]
    d: dict = dict(method=method)
    if len(L) == 0:
        return d
    L = L.iloc[0].to_dict()
    nd = L.get("uncal_null_cross_distinct_mean", np.nan)
    d["null_uncal"] = f"{L.get('uncal_null_cross_distinct_sig_mean', np.nan):.2f}/{nd:.0f}" if np.isfinite(nd) else "na"
    R = calib[(calib.method == method) & (calib.input == "real")].copy()
    R["domain"] = [clb.relation_domain(clb.REL_BY_ID[r]) for r in R.rid]
    for dom, lab in TIERS:
        q = R[R.domain == dom]
        d[lab] = f"{int((q.call == 'correct').sum())}/{int((q.call == 'wrong').sum())}" if len(q) else "—"
    d["held-out transplant FA / draw"] = f"{L.get('null_cross_fa_per_draw', np.nan):.2f}"
    d["RT"] = f"{int(L.get('null_random_fa', 0))}" if np.isfinite(L.get("null_random_fa", np.nan)) else "na"
    d["score"] = f"{int(L['score'])}" if np.isfinite(L.get("score", np.nan)) else "na"
    d["n_rel"] = f"{int(L['n_relations'])}" if np.isfinite(L.get("n_relations", np.nan)) else "na"
    return d


def rel_rows(method: str, calib: pd.DataFrame, rids=("R01", "R02")) -> list[dict]:
    out = []
    for rid in rids:
        q = calib[(calib.method == method) & (calib.input == "real") & (calib.rid == rid)]
        if len(q) == 0:
            out.append(dict(method=method, rid=rid, estimate="—"))
            continue
        x = q.iloc[0]
        out.append(dict(method=method, rid=rid, estimate=f"{x.est:+.4g} [{x.lo:.4g}, {x.hi:.4g}]",
                        sign="physical (−)" if np.sign(x.est) == x.expected else "policy (+)" if np.isfinite(x.est) else "na",
                        admitted="yes" if bool(x.admitted) else "no", call=x.call, p_by=f"{x.p_by:.3g}", stable=f"{x.stable_frac:.2f}",
                        n_cases=int(x.n_cases), est_real=f"{x.est_input:+.4g}", null_mean=f"{x.est_null_mean:+.4g}"))
    return out


def stage_report(args) -> None:
    out = args.out
    C0 = pd.read_csv(PAPER / "calibrated.csv")
    LB0 = pd.read_csv(PAPER / "leaderboard.csv")
    parts, metas = [], {}
    for d in sorted(out.iterdir()):
        if d.is_dir() and (d / "calibrated.csv").exists():
            c = pd.read_csv(d / "calibrated.csv")
            lb = pd.read_csv(d / "leaderboard.csv")
            parts.append((d.name, c, lb))
            if (d / "meta.json").exists():
                metas[d.name] = json.loads((d / "meta.json").read_text())
    lines = ["# D_nonlinear: generated tables", ""]
    t1_rows, rel_tab = [], []
    for name, c, lb in parts:
        for m in sorted(set(c.method)):
            if m == "typed_pooled" or "_check" in m:  # validation-only methods are compared in validation_vs_paper.csv
                continue
            row = table1_row(m, c, lb)
            row["run"] = name
            t1_rows.append(row)
            cp = COUNTERPART.get(m)
            if cp and cp in set(LB0.method):
                r0 = table1_row(cp, C0, LB0)
                r0["run"] = "paper"
                t1_rows.append(r0)
            rel_tab += rel_rows(m, c)
            if cp:
                rel_tab += rel_rows(cp, C0)
    T1 = pd.DataFrame(t1_rows).drop_duplicates()
    lines += ["## Table 1 metrics, new estimators next to their linear counterparts", "", df_to_md(T1), ""]
    lines += ["## R01 / R02 (set rate -> EtCO2) calibrated estimates", "", df_to_md(pd.DataFrame(rel_tab).drop_duplicates()), ""]
    for name, c, lb in parts:
        lines += [f"## Per-relation calibrated calls: {name}", ""]
        R = c[c.input == "real"]
        ms = [m for m in sorted(set(R.method)) if m != "typed_pooled" and "_check" not in m]
        hdr = "| relation | domain | " + " | ".join(ms) + " |"
        lines += [hdr, "|---|---|" + "---|" * len(ms)]
        for r in clb.RELATIONS:
            cells = []
            for m in ms:
                q = R[(R.method == m) & (R.rid == r.rid)]
                if len(q) == 0 or not np.isfinite(q.iloc[0].est):
                    cells.append("—")
                    continue
                x = q.iloc[0]
                mark = {"correct": "**R**", "wrong": "**X**", "ns_right": "r", "ns_wrong": "x"}.get(x.call, "?")
                cells.append(f"{mark} {x.est:.3g}")
            lines.append(f"| {r.rid} {r.label} | {clb.relation_domain(r)} | " + " | ".join(cells) + " |")
        lines.append("")
    for name, m in metas.items():
        lines.append(f"- {name}: {m.get('seconds_total', float('nan')) / 3600:.2f} h, workers {m.get('workers')}, command `{m.get('command')}`")
    (out / "report_tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    clb.use_table("v4")
    {"pcmci": stage_pcmci, "xcorr": stage_xcorr, "gbm": stage_gbm, "varx": stage_varx, "pat": stage_pat,
     "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()
