"""
Closed-loop measurement benchmark on real VitalDB data (see ``clbench.eval.closed_loop_bench``).

Scores eight estimators (naive event pre/post, v1 lagged cross-correlation, pairwise Granger/VAR,
pooled VARX with action inputs, PCMCI+, pooled VAR-LiNGAM, matched-placebo event design,
placebo-calibrated local pre-trend design) against a table of response relations (pharmacology /
physiology) and policy relations (clinical practice) with known sign, on real data and under two
negative controls (random action times; another patient's actions time-aligned).

USAGE
─────
  PY=.venv/bin/python
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --scale 30s --cases all  --workers 6
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --scale 30s --cases fine --workers 6
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --scale 2s  --cases fine --workers 6
  # quick smoke test
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --scale 2s --cases fine --limit 12 --out-dir /tmp/x

Outputs (runs/iclr/closed_loop/<scale>_<cases>/): pooled.csv, per_patient.parquet,
patient_summary.csv, eligibility.csv, main_table.md (real data), nc_tables.md, nc_summary.csv, meta.json.

v2 (default, ``--bench-version v2``; outputs under runs/iclr/closed_loop_v2/<scale>_<cases>/): the ten
verifier fixes (see the module docstring of ``clbench.eval.closed_loop_bench``), the CLBench typed
estimator (``typed_pooled``), EB patient-level edges and the patient-specificity split-half test; adds
typed_edges.csv, typed_patient_eb.csv, patient_specificity.csv, sensitivity.md. ``--bench-version v1``
reproduces the Wave-2 benchmark (writes to runs/iclr/closed_loop/ unless --out-dir is given).
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --scale 30s --cases all --workers 10 \
      --protocol runs/iclr/closed_loop_v2/protocol.md
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --combine-v2

v3 = ClosedLoopBench 1.0 (default, ``--bench-version v3``; outputs under runs/iclr/closed_loop_v3/): K = 8
cross-patient null draws with derangement-aware calibration and a stability requirement, the pooled
sequential-trial contrast (``seqtrial``), pre-t co-intervention windows, ordered (reproducible) result
consumption, code hashes in meta.json, the calibrated admission rule applied to every method and the
leaderboard (see the module docstrings of ``clbench.eval.closed_loop_bench`` and ``clbench.bench``).
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --sim-validate --workers 8      # simulator gate
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --scale 2s  --cases fine --workers 8 \
      --protocol runs/iclr/closed_loop_v3/protocol.md
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --scale 30s --cases all  --workers 8 \
      --protocol runs/iclr/closed_loop_v3/protocol.md
  PYTHONPATH=src $PY scripts/44_closed_loop_bench.py --combine-v3
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from clbench.bench import loader as bl  # noqa: E402
from clbench.bench import nulls as bn  # noqa: E402
from clbench.eval import closed_loop_bench as clb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("44_closed_loop_bench")

DATA = ROOT / "data" / "processed_vitaldb"
OUT = ROOT / "runs" / "iclr" / "closed_loop"
OUT_V2 = ROOT / "runs" / "iclr" / "closed_loop_v2"
OUT_V3 = ROOT / "runs" / "iclr" / "closed_loop_v3"
OUT_V4 = ROOT / "runs" / "iclr" / "closed_loop_v4"
GRID_V4 = DATA / "native_2s_v4"  # clbench.bench.native_grid output (grid/, events/, stops.parquet, qc.csv)
CODE_FILES: tuple[str, ...] = (
    "src/clbench/eval/closed_loop_bench.py", "src/clbench/edges/typed_temporal_pooled.py",
    "scripts/44_closed_loop_bench.py", "tests/unit/test_closed_loop_bench.py", "src/clbench/bench/__init__.py",
    "src/clbench/bench/ground_truth.py", "src/clbench/bench/loader.py", "src/clbench/bench/nulls.py",
    "src/clbench/bench/scoring.py", "src/clbench/bench/sim.py", "src/clbench/sim/clinician_loop.py",
    "src/clbench/bench/native_grid.py", "src/clbench/data/vitaldb.py")


def code_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES if (ROOT / f).exists()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scale", choices=list(clb.SCALES), default="30s")
    p.add_argument("--cases", choices=["all", "fine"], default="all",
                   help="all: every case with numerics at this scale; fine: the 500 cases with 2 s numerics")
    p.add_argument("--limit", type=int, default=0, help="Use only the first N cases (smoke tests)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--boot", type=int, default=200, help="Case-bootstrap replicates")
    p.add_argument("--lingam-boot", type=int, default=50)
    p.add_argument("--lingam-max-rows", type=int, default=50_000)
    p.add_argument("--pcmci-cases", type=int, default=-1, help="Cases run through PCMCI+ on real data (-1: scale default)")
    p.add_argument("--pcmci-nc-cases", type=int, default=-1, help="Cases run through PCMCI+ per negative control")
    p.add_argument("--methods", default=",".join(clb.METHODS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--combine", action="store_true",
                   help="Only combine finished runs under runs/iclr/closed_loop into combined_tables.md")
    p.add_argument("--combine-v2", action="store_true",
                   help="Only combine finished v2 runs under runs/iclr/closed_loop_v2 into tables.md")
    p.add_argument("--bench-version", choices=["v1", "v2", "v3", "v4"], default="v3")
    p.add_argument("--build-grid", action="store_true", help="v4: build the 2 s native grid for every case and exit")
    p.add_argument("--grid-dir", type=Path, default=GRID_V4, help="v4: native 2 s grid directory")
    p.add_argument("--combine-v3", action="store_true",
                   help="Only combine finished v3 runs under runs/iclr/closed_loop_v3 into tables.md / leaderboard.md")
    p.add_argument("--cross-draws", type=int, default=len(clb.CROSS_SALTS), help="v3: cross-patient null draws (K)")
    p.add_argument("--sim-validate", action="store_true", help="v3: simulator validation of the event designs")
    p.add_argument("--sim-n", type=int, default=600, help="Simulated patients per cohort (--sim-validate)")
    p.add_argument("--sim-scales", default="2s,30s")
    p.add_argument("--sim-cross", type=int, default=4, help="Cross null draws in the simulator validation")
    p.add_argument("--sim-summary-only", action="store_true", help="Rewrite sim_validation/summary.md from the CSV")
    p.add_argument("--recompute-eb", action="store_true",
                   help="Recompute typed_patient_eb.csv of a finished v2 run (--out-dir) from its saved "
                        "pooled.csv and per_patient.parquet (no re-run)")
    p.add_argument("--protocol", type=Path, default=None, help="Pre-registered protocol file (sha256 recorded)")
    p.add_argument("--spec-boot", type=int, default=1000, help="Patient-bootstrap reps, patient-specificity test")
    p.add_argument("--typed-q", type=float, default=0.05, help="BY level for typed-edge admission")
    return p.parse_args()


# (moved to clbench.bench.loader / clbench.bench.nulls in v3; the logic is unchanged)
def load_events(pids: set[str]) -> dict[str, pd.DataFrame]:
    return bl.load_events(pids, DATA)


_ev_records = bl.ev_records
_to_events = bl.to_events


def _stop_job(path: str) -> tuple[str, dict[str, float]]:
    return bl.track_stops(path)


cross_partners = bn.cross_partners


def df_to_md(df: pd.DataFrame) -> str:
    def f(x):
        return f"{x:.3f}" if isinstance(x, (float, np.floating)) else str(x)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def run_job(job: dict) -> tuple[str, dict, float]:
    t0 = time.time()
    cfg = clb.SCALES[job["scale"]]
    opts = {"v1": clb.BenchOptions.v1(), "v2": clb.BenchOptions.v2()}.get(job["version"], clb.V3)
    cols = ["window_start_sec", *clb.VITALS]
    d = pd.read_parquet(job["path"], columns=cols)
    events = _to_events(job["events"], job.get("stops"))
    case = clb.CaseData(job["pid"], d.window_start_sec.to_numpy(float) + cfg.window_sec / 2, cfg.window_sec,
                        {v: d[v].to_numpy(float) for v in clb.VITALS}, events)
    if job["version"] in ("v3", "v4"):
        if job["version"] == "v4":
            opts = clb.V4
        cross = {arm: _to_events(rec, st) for arm, (rec, st) in job["cross"].items()}
        arms = clb.build_arms(case, None, seed=job["seed"], arms=("real", "nc_random") + tuple(job["cross"]),
                              cross_events=cross)
        out = {}
        for arm, c in arms.items():
            kw = dict(keep_pp=False) if job["version"] == "v4" else {}  # v4: no per-patient Granger fits
            out[arm] = clb.analyse_case(c, cfg, do_pcmci=job["pcmci"], lingam_resid_rows=job["lingam_rows"],
                                        seed=job["seed"], methods=tuple(job["methods"]), opts=opts, **kw)
        return job["pid"], out, time.time() - t0
    partner = _to_events(job["partner_events"], job.get("partner_stops"))
    partner2 = _to_events(job["partner2_events"], job.get("partner2_stops")) if job.get("partner2_events") is not None else None
    arms_list = clb.ARMS_V1 if opts.is_v1 else clb.ARMS
    arms = clb.build_arms(case, partner, seed=job["seed"], arms=arms_list, partner2_events=partner2)
    out = {}
    for arm, c in arms.items():
        methods = tuple(job["methods"]) if arm != "nc_cross2" else ("varx",)
        out[arm] = clb.analyse_case(c, cfg, do_pcmci=job["pcmci"].get(arm, False),
                                    lingam_resid_rows=job["lingam_rows"], seed=job["seed"],
                                    methods=methods, opts=opts)
    return job["pid"], out, time.time() - t0


def _cell(q: pd.DataFrame) -> str:
    if len(q) == 0 or not np.isfinite(q.iloc[0].est):
        return "—"
    r = q.iloc[0]
    mark = {"correct": "R", "wrong": "X", "ns_right": "r", "ns_wrong": "x"}.get(r.call, "?")
    return f"{mark} {r.est:.3g} [{r.lo:.3g}, {r.hi:.3g}]"


def combine(root: Path = OUT) -> None:
    """Timescale comparison, horizon profiles and negative-control summaries across finished runs."""
    runs = {d.name: d for d in sorted(root.iterdir()) if (d / "pooled.csv").exists()}
    if not runs:
        log.warning("no finished runs under %s", root)
        return
    P = {k: pd.read_csv(d / "pooled.csv") for k, d in runs.items()}
    out = ["# Closed-loop benchmark — combined tables", "",
           "R/X = expected/wrong sign with 95% case-bootstrap CI excluding 0; r/x = sign only.", ""]
    # 1. call counts per method and run (real data)
    out += ["## 1. Pooled sign calls (count of relations: R / X / r / x); real data and null-calibrated", ""]
    hdr = "| run | kind | " + " | ".join(clb.METHODS) + " |"
    out += [hdr, "|---|---|" + "---|" * len(clb.METHODS)]
    for k, df in P.items():
        cfg = clb.SCALES[k.split("_")[0]]
        for arm, kind in (("real", "response"), ("real", "policy"), ("calib_cross", "response"), ("calib_cross", "policy")):
            cells = []
            for m in clb.METHODS:
                calls = []
                for r in clb.RELATIONS:
                    if r.kind != kind:
                        continue
                    q = df[(df.arm == arm) & (df.method == m) & (df.rid == r.rid) & (df.h == clb.primary_h(r, m, cfg))]
                    if len(q):
                        calls.append(q.iloc[0].call)
                c = pd.Series(calls).value_counts() if calls else pd.Series(dtype=int)
                cells.append(f"{c.get('correct', 0)}/{c.get('wrong', 0)}/{c.get('ns_right', 0)}/{c.get('ns_wrong', 0)}")
            out.append(f"| {k} {arm} | {kind} | " + " | ".join(cells) + " |")
    # 2. horizon profiles of the event designs and VARX (response relations)
    for k, df in P.items():
        cfg = clb.SCALES[k.split("_")[0]]
        out += ["", f"## 2. Horizon profile, {k} (response relations, real data)", ""]
        for m in ("naive", "matched", "kink", "varx"):
            out += [f"### {m}", "", "| relation | " + " | ".join(f"{int(h)} s" for h in cfg.horizons) + " |",
                    "|---|" + "---|" * len(cfg.horizons)]
            for r in clb.RELATIONS:
                if r.kind != "response":
                    continue
                cells = [_cell(df[(df.arm == "real") & (df.method == m) & (df.rid == r.rid) & (df.h == h)]) for h in cfg.horizons]
                out.append(f"| {r.rid} {r.label} | " + " | ".join(cells) + " |")
            out.append("")
    # 3. negative controls
    out += ["## 3. Negative controls (share of relations with pooled CI excluding 0; per-patient rates)", ""]
    for k, d in runs.items():
        f = d / "nc_summary.csv"
        if f.exists():
            out += [f"### {k}", "", df_to_md(pd.read_csv(f)), ""]
    (root / "combined_tables.md").write_text("\n".join(out) + "\n")
    log.info("wrote %s", root / "combined_tables.md")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    a = abs(x)
    return f"{x:.{nd}g}" if a >= 1e-3 or a == 0 else f"{x:.1e}"


def typed_edges_md(tt: pd.DataFrame) -> str:
    if tt is None or len(tt) == 0:
        return "(no typed-edge rows)"
    lines = ["| input arm | relation | type | expected | calibrated est [95% CI] | p (cal.) | p (BY) | admitted | call |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in tt.itertuples():
        lines.append(f"| {r.input_arm} | {r.rid} {r.label} | {r.kind} | {'+' if r.expected > 0 else '-'} | "
                     f"{_fmt(r.est)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_fmt(r.p_cal, 2)} | {_fmt(r.p_by, 2)} | "
                     f"{'yes' if r.admitted else 'no'} | {r.call} |")
    return "\n".join(lines)


def spec_md(sp: pd.DataFrame) -> str:
    if sp is None or len(sp) == 0:
        return "(no patient-specificity rows)"
    lines = ["| arm | relation | outcome | patients | MSE pooled | MSE per-patient | MSE shrunk | "
             "shrunk - pooled [95% CI] | per-patient - pooled [95% CI] | r(early, late) [95% CI] | sign agree pooled / pp / shrunk |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sp.itertuples():
        if not hasattr(r, "mse_pooled") or not np.isfinite(getattr(r, "mse_pooled", np.nan)):
            continue
        lines.append(
            f"| {r.arm} | {r.rid} {r.label} | {r.outcome} | {r.n_patients} | {_fmt(r.mse_pooled)} | {_fmt(r.mse_pp)} | "
            f"{_fmt(r.mse_shrunk)} | {_fmt(r.d_shrunk_minus_pooled)} [{_fmt(r.d_shrunk_minus_pooled_lo)}, "
            f"{_fmt(r.d_shrunk_minus_pooled_hi)}] | {_fmt(r.d_pp_minus_pooled)} [{_fmt(r.d_pp_minus_pooled_lo)}, "
            f"{_fmt(r.d_pp_minus_pooled_hi)}] | {_fmt(r.r_early_late, 2)} [{_fmt(r.r_early_late_lo, 2)}, "
            f"{_fmt(r.r_early_late_hi, 2)}] | {100 * r.sign_pooled:.0f}% / {100 * r.sign_pp:.0f}% / {100 * r.sign_shrunk:.0f}% |")
    return "\n".join(lines)


def run_v1(args, cfg, methods, out_dir, files, pids, ev, partners) -> None:
    """Wave-2 benchmark, unchanged (kept for v1-vs-v2 reproducibility)."""
    n_pc = args.pcmci_cases if args.pcmci_cases >= 0 else (600 if cfg.name == "30s" else 60)
    n_pc_nc = args.pcmci_nc_cases if args.pcmci_nc_cases >= 0 else (300 if cfg.name == "30s" else 30)
    press_agents = {clb.ACTIONS[a].agent for a in clb.PRESSORS}
    rng = np.random.default_rng(args.seed + 5)
    press = [p for p in pids if p in ev and ev[p].agent.isin(press_agents).any()]
    rest = [p for p in pids if p not in set(press)]
    order = list(rng.permutation(press)) + list(rng.permutation(rest))
    pc_real = set(order[:n_pc])
    pc_nc = set(order[:n_pc_nc])
    lingam_rows = 60 if cfg.name == "30s" else 300
    jobs = []
    for i, (f, pid) in enumerate(zip(files, pids)):
        prt = {agent: _ev_records(ev.get(pp_))[agent] for agent, pp_ in partners[pid].items()
               if agent in _ev_records(ev.get(pp_))}
        jobs.append(dict(path=str(f), pid=pid, scale=cfg.name, seed=args.seed * 100_003 + i, version="v1",
                         events=_ev_records(ev.get(pid)), partner_events=prt, partner2_events=None,
                         lingam_rows=lingam_rows, methods=list(methods),
                         pcmci={"real": pid in pc_real, "nc_random": pid in pc_nc, "nc_cross": pid in pc_nc}))
    jobs.sort(key=lambda j: -sum(j["pcmci"].values()))
    agg = clb.Aggregator(pids, cfg, n_boot=args.boot, seed=args.seed, opts=clb.BenchOptions.v1())
    t0, times = time.time(), []
    with get_context("spawn").Pool(args.workers) as pool:
        for k, (pid, res, dt) in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            for arm, r in res.items():
                agg.add(pid, arm, r)
            times.append(dt)
            if k % 100 == 0 or k == len(jobs):
                log.info("%d/%d cases  (%.0f s elapsed, median %.2f s/case)", k, len(jobs), time.time() - t0, np.median(times))
    t_cases = time.time() - t0
    pooled, pp = agg.finalize(lingam_boot=args.lingam_boot, lingam_max_rows=args.lingam_max_rows, seed=args.seed)
    psum = clb.patient_summary(pp)
    elig = agg.eligibility()
    el = (elig.groupby(["arm", "rid"]).agg(n_cases_with_events=("n_events", lambda x: int((x > 0).sum())),
                                          n_events=("n_events", "sum"), n_eligible=("eligible", "sum")).reset_index())
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    pp.to_parquet(out_dir / "per_patient.parquet", index=False)
    psum.to_csv(out_dir / "patient_summary.csv", index=False)
    el.to_csv(out_dir / "eligibility.csv", index=False)
    ncs = clb.negative_control_summary(pooled, psum, cfg)
    ncs.to_csv(out_dir / "nc_summary.csv", index=False)
    head = f"# Closed-loop benchmark v1 — scale {cfg.name}, cases {args.cases} (n={len(pids)})\n\n"
    (out_dir / "main_table.md").write_text(head + clb.main_table(pooled, psum, cfg, "real") + "\n\n## calib_cross\n\n"
                                           + clb.main_table(pooled, pd.DataFrame(), cfg, "calib_cross") + "\n")
    meta = dict(args={k: str(v) for k, v in vars(args).items()}, n_cases=len(pids), seconds_cases=t_cases,
                seconds_total=time.time() - t0, pcmci_cases_real=len(pc_real), pcmci_cases_nc=len(pc_nc),
                n_errors=len(agg.errors), errors=agg.errors[:50], config=str(cfg))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s", out_dir)


def main() -> None:
    args = parse_args()
    if args.combine:
        combine(args.out_dir or OUT)
        return
    if args.combine_v2:
        combine_v2(args.out_dir or OUT_V2)
        return
    if args.combine_v3:
        combine_v3(args.out_dir or OUT_V3)
        return
    if args.sim_validate:
        sim_validate(args)
        return
    if args.recompute_eb:
        d = args.out_dir or (OUT_V2 / f"{args.scale}_{args.cases}")
        E = clb.eb_patient_table(pd.read_csv(d / "pooled.csv", low_memory=False),
                                 pd.read_parquet(d / "per_patient.parquet"), clb.SCALES[args.scale])
        E.to_csv(d / "typed_patient_eb.csv", index=False)
        log.info("recomputed %s", d / "typed_patient_eb.csv")
        return
    if args.build_grid:
        from clbench.bench import native_grid as ng

        log.info("building the v4 2 s grid -> %s", args.grid_dir)
        log.info("%s", ng.build_all(args.grid_dir, DATA, workers=args.workers, limit=args.limit))
        return
    if args.bench_version == "v4":
        run_v4(args)
        return
    cfg = clb.SCALES[args.scale]
    methods = tuple(m for m in args.methods.split(",") if m)
    root = {"v1": OUT, "v2": OUT_V2}.get(args.bench_version, OUT_V3)
    out_dir = args.out_dir or (root / f"{args.scale}_{args.cases}")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = DATA / ("numerics" if args.scale == "30s" else "fine_2s")
    files = sorted(src.glob("*.parquet"))
    if args.cases == "fine":
        fine = {f.stem for f in (DATA / "fine_2s").glob("*.parquet")}
        files = [f for f in files if f.stem in fine]
    if args.limit:
        files = files[: args.limit]
    pids = [f.stem for f in files]
    log.info("scale %s, %d cases from %s, bench %s", cfg.name, len(pids), src, args.bench_version)
    ev = load_events(set(pids))
    partners = cross_partners(pids, ev, args.seed)
    if args.bench_version == "v1":
        run_v1(args, cfg, methods, out_dir, files, pids, ev, partners)
        return
    if args.bench_version == "v3":
        if args.methods == ",".join(clb.METHODS):
            methods = clb.METHODS_V3_COMPUTE
        run_v3(args, cfg, methods, out_dir, files, pids, ev)
        return

    # ── v2 ──
    proto_sha = sha256(args.protocol) if args.protocol and args.protocol.exists() else None
    if args.protocol and proto_sha is None:
        raise SystemExit(f"protocol file not found: {args.protocol}")
    opts = clb.BenchOptions.v2()
    partners2 = cross_partners(pids, ev, args.seed, salt=7919)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        stops = dict(pool.imap_unordered(_stop_job, [str(f) for f in files], chunksize=16))
    log.info("track stop times read (%.0f s)", time.time() - t0)
    # PCMCI+: one seeded subsample (pressor cases first), the SAME cases in every arm
    n_pc = args.pcmci_cases if args.pcmci_cases >= 0 else (1200 if cfg.name == "30s" else 500)
    press_agents = {clb.ACTIONS[a].agent for a in clb.PRESSORS}
    rng = np.random.default_rng(args.seed + 5)
    press = [p for p in pids if p in ev and ev[p].agent.isin(press_agents).any()]
    rest = [p for p in pids if p not in set(press)]
    order = list(rng.permutation(press)) + list(rng.permutation(rest))
    pc_set = set(order[:n_pc])
    lingam_rows = 60 if cfg.name == "30s" else 300

    def prt_for(pid, parts):
        out, st = {}, {}
        for agent, pp_ in parts[pid].items():
            rec = _ev_records(ev.get(pp_))
            if agent in rec:
                out[agent] = rec[agent]
                st[agent] = stops.get(pp_, {}).get(agent, np.inf)
        return out, st

    jobs = []
    for i, (f, pid) in enumerate(zip(files, pids)):
        p1, s1 = prt_for(pid, partners)
        p2, s2 = prt_for(pid, partners2)
        inpc = pid in pc_set
        jobs.append(dict(path=str(f), pid=pid, scale=cfg.name, seed=args.seed * 100_003 + i, version="v2",
                         events=_ev_records(ev.get(pid)), stops=stops.get(pid, {}),
                         partner_events=p1, partner_stops=s1, partner2_events=p2, partner2_stops=s2,
                         lingam_rows=lingam_rows, methods=list(methods),
                         pcmci={"real": inpc, "nc_random": inpc, "nc_cross": inpc}))
    jobs.sort(key=lambda j: -sum(j["pcmci"].values()))
    agg = clb.Aggregator(pids, cfg, n_boot=args.boot, seed=args.seed, opts=opts, typed_q=args.typed_q,
                         spec_boot=args.spec_boot)
    times = []
    t1 = time.time()
    with ctx.Pool(args.workers) as pool:
        for k, (pid, res, dt) in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            for arm, r in res.items():
                agg.add(pid, arm, r)
            times.append(dt)
            if k % 200 == 0 or k == len(jobs):
                log.info("%d/%d cases  (%.0f s elapsed, median %.2f s/case)", k, len(jobs), time.time() - t1, np.median(times))
    t_cases = time.time() - t1
    log.info("finalising (bootstrap, VAR-LiNGAM, typed edges, EB, patient-specificity) ...")
    pooled, pp = agg.finalize(lingam_boot=args.lingam_boot, lingam_max_rows=args.lingam_max_rows, seed=args.seed)
    psum = clb.patient_summary(pp)
    elig = agg.eligibility()
    el = (elig.groupby(["arm", "rid"]).agg(n_cases_with_events=("n_events", lambda x: int((x > 0).sum())),
                                          n_events=("n_events", "sum"), n_eligible=("eligible", "sum")).reset_index())
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    pp.to_parquet(out_dir / "per_patient.parquet", index=False)
    psum.to_csv(out_dir / "patient_summary.csv", index=False)
    el.to_csv(out_dir / "eligibility.csv", index=False)
    allm = clb.METHODS_V2 + clb.SENSITIVITY_METHODS
    ncs = clb.negative_control_summary(pooled, psum, cfg, methods=allm)
    ncs.to_csv(out_dir / "nc_summary.csv", index=False)
    if agg.typed_table is not None:
        agg.typed_table.to_csv(out_dir / "typed_edges.csv", index=False)
    if agg.eb_table is not None:
        agg.eb_table.to_csv(out_dir / "typed_patient_eb.csv", index=False)
    if agg.spec_table is not None:
        agg.spec_table.to_csv(out_dir / "patient_specificity.csv", index=False)
    head = (f"# Closed-loop benchmark v2 — scale {cfg.name}, cases {args.cases} (n={len(pids)})\n\n"
            f"Protocol sha256: `{proto_sha}`\n\n"
            "Cell: call estimate [95% case-bootstrap CI]; % of eligible patients with the expected sign.\n"
            "**R** / **X** = expected / wrong sign with CI excluding 0 (typed_pooled: admitted edge); r / x = sign only.\n"
            f"PCMCI+ on the same {len(pc_set)} cases in every arm.\n\n")
    main_txt = (head + "## Calibrated (real minus nc_cross, paired case bootstrap); % patients from the real arm\n\n"
                + clb.main_table(pooled[pooled.arm == "calib_cross"].assign(arm="calib_cross"),
                                 psum.assign(arm=psum.arm.replace({"real": "calib_cross"})) if len(psum) else psum,
                                 cfg, "calib_cross", clb.METHODS_V2)
                + "\n\n## Real data, uncalibrated\n\n" + clb.main_table(pooled, psum, cfg, "real", clb.METHODS_V2) + "\n")
    (out_dir / "main_table.md").write_text(main_txt)
    sens_txt = head + "## Sensitivity variants, calibrated (real minus nc_cross)\n\n" + clb.main_table(
        pooled, pd.DataFrame(), cfg, "calib_cross", ("matched", "kink") + clb.SENSITIVITY_METHODS) + \
        "\n\n## Sensitivity variants, real data\n\n" + clb.main_table(
        pooled, psum, cfg, "real", ("matched", "kink") + clb.SENSITIVITY_METHODS) + "\n"
    (out_dir / "sensitivity.md").write_text(sens_txt)
    nc_txt = head
    for arm in ("nc_cross", "nc_random"):
        nc_txt += f"\n\n## Negative control: {arm} (any R/X is a false discovery)\n\n"
        nc_txt += clb.main_table(pooled, psum, cfg, arm, clb.METHODS_V2) + "\n"
    nc_txt += "\n\n## Summary\n\n" + df_to_md(ncs) + "\n"
    (out_dir / "nc_tables.md").write_text(nc_txt)
    (out_dir / "typed_edges.md").write_text(head + typed_edges_md(agg.typed_table) + "\n\n## Patient-specificity\n\n"
                                            + spec_md(agg.spec_table) + "\n")
    meta = dict(args={k: str(v) for k, v in vars(args).items()}, bench_version="v2", protocol=str(args.protocol),
                protocol_sha256=proto_sha, n_cases=len(pids), seconds_cases=t_cases, seconds_total=time.time() - t0,
                median_case_seconds=float(np.median(times)) if times else None, pcmci_cases_all_arms=len(pc_set),
                pcmci_done={a: len(s_) for a, s_ in agg.pcmci_done.items()}, errors=agg.errors[:50],
                n_errors=len(agg.errors), config=str(cfg), options=str(opts),
                n_tracks_masked=int(sum(np.isfinite(v) for st in stops.values() for v in st.values())))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s (%.0f s)", out_dir, time.time() - t0)
    print(typed_edges_md(agg.typed_table))
    print(ncs[ncs.arm.isin(["nc_cross", "nc_random"])][["arm", "method", "n_relations", "n_sig", "n_distinct", "n_distinct_sig"]].to_string())


def _counts(df: pd.DataFrame, arm: str, methods, cfg, kind: str | None = None) -> dict[str, str]:
    out = {}
    for m in methods:
        calls = []
        for r in clb.RELATIONS:
            if kind and r.kind != kind:
                continue
            q = df[(df.arm == arm) & (df.method == m) & (df.rid == r.rid) & (df.h == clb.primary_h(r, m, cfg))]
            if len(q):
                calls.append(q.iloc[0].call)
        c = pd.Series(calls).value_counts() if calls else pd.Series(dtype=int)
        out[m] = f"{c.get('correct', 0)}/{c.get('wrong', 0)}" if calls else "—"
    return out


DEVIATIONS = [
    "1. EB patient-level edges (protocol §5.5), found after the confirmatory runs: the posterior SD omitted the "
    "uncertainty of the pooled mean. With tau^2 = 0 it was 0, so every patient was 'admitted', e.g. "
    "30s_all R07 real 100%. Fixed in `eb_shrink` by adding (1 - B_i)^2 var(mu), with var(mu) from the pooled "
    "bootstrap CI. The table was recomputed from the saved `per_patient.parquet` and `pooled.csv` with "
    "`--recompute-eb`; nothing was re-run. The buggy table is kept as `typed_patient_eb_v0_bug_postsd.csv`. "
    "No other output is affected. Because of this fix, the current sha256 of `typed_temporal_pooled.py`, "
    "`closed_loop_bench.py` (EB moved into `eb_patient_table`), this script (`--recompute-eb`, this list) and the "
    "test file (one regression test added) differs from the hashes in protocol.md. The confirmatory runs used "
    "the registered code.",
    "2. PCMCI+ used the same 1,200 (30 s) / 500 (2 s) cases in every arm, as registered. The number of cases "
    "with a usable PCMCI+ system still differs by arm (a transplanted stream can leave a case with no usable "
    "action series): 30s_all real 1,177, nc_random 1,178, nc_cross 1,156; 2s_fine 498, 498, 496.",
]


def combine_v2(root: Path = OUT_V2) -> None:
    """tables.md: call counts, null discovery (distinct estimates), typed edges, patient-specificity."""
    runs = {d.name: d for d in sorted(root.iterdir()) if (d / "pooled.csv").exists()}
    if not runs:
        log.warning("no finished runs under %s", root)
        return
    proto = root / "protocol.md"
    out = ["# Closed-loop benchmark v2 — combined tables", "",
           f"Protocol: `protocol.md`, sha256 `{sha256(proto) if proto.exists() else 'missing'}`.", ""]
    for k, d in runs.items():
        meta = json.loads((d / "meta.json").read_text())
        out.append(f"- {k}: n={meta['n_cases']}, protocol sha256 at run time `{meta.get('protocol_sha256')}`, "
                   f"{meta['seconds_total']:.0f} s, PCMCI+ cases (all arms) {meta.get('pcmci_cases_all_arms')}, errors {meta['n_errors']}")
    out.append("")
    allm = clb.METHODS_V2 + clb.SENSITIVITY_METHODS
    out += ["## 1. Relation-level calls, R/X (count of 18 relations: expected sign / wrong sign, CI excluding 0)", "",
            "| run | arm | kind | " + " | ".join(allm) + " |", "|---|---|---|" + "---|" * len(allm)]
    for k, d in runs.items():
        cfg = clb.SCALES[k.split("_")[0]]
        P = pd.read_csv(d / "pooled.csv")
        for arm in ("real", "calib_cross"):
            for kind in ("response", "policy"):
                c = _counts(P, arm, allm, cfg, kind)
                out.append(f"| {k} | {arm} | {kind} | " + " | ".join(c[m] for m in allm) + " |")
    out += ["", "## 2. Null discovery: relations (distinct estimates) with CI excluding 0 / admitted", "",
            "| run | method | nc_cross: sig of n | nc_cross: distinct sig of distinct | nc_random: sig of n | nc_random: distinct sig of distinct |",
            "|---|---|---|---|---|---|"]
    for k, d in runs.items():
        N = pd.read_csv(d / "nc_summary.csv")
        for m in allm:
            cells = []
            for arm in ("nc_cross", "nc_random"):
                q = N[(N.arm == arm) & (N.method == m)]
                if len(q):
                    r = q.iloc[0]
                    cells += [f"{int(r.n_sig)}/{int(r.n_relations)}", f"{int(r.n_distinct_sig)}/{int(r.n_distinct)}"]
                else:
                    cells += ["—", "—"]
            out.append(f"| {k} | {m} | " + " | ".join(cells) + " |")
    for k, d in runs.items():
        f = d / "typed_edges.csv"
        if f.exists():
            out += ["", f"## 3. Typed estimator (typed_pooled), {k}", "", typed_edges_md(pd.read_csv(f))]
    for k, d in runs.items():
        f = d / "typed_patient_eb.csv"
        if f.exists():
            E = pd.read_csv(f)
            out += ["", f"## 4. Patient-level response edges (EB shrinkage toward the pooled estimate), {k}", "", df_to_md(E)]
    for k, d in runs.items():
        f = d / "patient_specificity.csv"
        if f.exists():
            out += ["", f"## 5. Patient-specificity split-half test, {k}", "", spec_md(pd.read_csv(f))]
    out += ["", "## 6. Deviations from protocol.md", "", *DEVIATIONS]
    (root / "tables.md").write_text("\n".join(out) + "\n")
    log.info("wrote %s", root / "tables.md")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# v3 = ClosedLoopBench 1.0
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def run_v3(args, cfg, methods, out_dir, files, pids, ev) -> None:
    proto_sha = sha256(args.protocol) if args.protocol and args.protocol.exists() else None
    if args.protocol and proto_sha is None:
        raise SystemExit(f"protocol file not found: {args.protocol}")
    from concurrent.futures import ProcessPoolExecutor

    opts = clb.V3
    K = args.cross_draws
    part_sets = bn.cross_partner_sets(pids, ev, args.seed, K)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        stops = dict(pool.imap(_stop_job, [str(f) for f in files], chunksize=16))
    log.info("track stop times read (%.0f s)", time.time() - t0)
    # PCMCI+: one seeded subsample (pressor cases first), the SAME cases in every arm (as v2)
    n_pc = args.pcmci_cases if args.pcmci_cases >= 0 else (1200 if cfg.name == "30s" else 500)
    press_agents = {clb.ACTIONS[a].agent for a in clb.PRESSORS}
    rng = np.random.default_rng(args.seed + 5)
    press = [p for p in pids if p in ev and ev[p].agent.isin(press_agents).any()]
    rest = [p for p in pids if p not in set(press)]
    order = list(rng.permutation(press)) + list(rng.permutation(rest))
    pc_set = set(order[:n_pc])
    lingam_rows = 60 if cfg.name == "30s" else 300

    def prt_for(pid, parts):
        out, st = {}, {}
        for agent, pp_ in parts[pid].items():
            rec = _ev_records(ev.get(pp_))
            if agent in rec:
                out[agent] = rec[agent]
                st[agent] = stops.get(pp_, {}).get(agent, np.inf)
        return out, st

    jobs = []
    for i, (f, pid) in enumerate(zip(files, pids)):
        jobs.append(dict(path=str(f), pid=pid, scale=cfg.name, seed=args.seed * 100_003 + i, version="v3",
                         events=_ev_records(ev.get(pid)), stops=stops.get(pid, {}),
                         cross={arm: prt_for(pid, parts) for arm, parts in part_sets.items()},
                         lingam_rows=lingam_rows, methods=list(methods), pcmci=pid in pc_set))
    jobs.sort(key=lambda j: -int(j["pcmci"]))  # stable: PCMCI+ cases first, then file order
    agg = clb.Aggregator(pids, cfg, n_boot=args.boot, seed=args.seed, opts=opts, typed_q=args.typed_q,
                         spec_boot=args.spec_boot)
    agg.nn_workers = args.workers  # threads for the seqtrial neighbour searches in finalize
    times = []
    t1 = time.time()
    with ctx.Pool(args.workers) as pool:
        # imap (ordered): results are consumed in job order, so every floating-point accumulation is
        # reproducible run to run (v2 used imap_unordered)
        for k, (pid, res, dt) in enumerate(pool.imap(run_job, jobs, chunksize=1), 1):
            for arm, r in res.items():
                agg.add(pid, arm, r)
            times.append(dt)
            if k % 200 == 0 or k == len(jobs):
                log.info("%d/%d cases  (%.0f s elapsed, median %.2f s/case)", k, len(jobs), time.time() - t1, np.median(times))
    t_cases = time.time() - t1
    log.info("finalising (VAR-LiNGAM in parallel, seqtrial, calibration, EB, patient-specificity) ...")
    with ProcessPoolExecutor(max_workers=min(args.workers, 10), mp_context=ctx) as ex:
        pooled, pp = agg.finalize(lingam_boot=args.lingam_boot, lingam_max_rows=args.lingam_max_rows, seed=args.seed,
                                  executor=ex)
    t_fin = time.time() - t1 - t_cases
    psum = clb.patient_summary(pp)
    elig = agg.eligibility()
    el = (elig.groupby(["arm", "rid"]).agg(n_cases_with_events=("n_events", lambda x: int((x > 0).sum())),
                                          n_events=("n_events", "sum"), n_eligible=("eligible", "sum")).reset_index())
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    pp.to_parquet(out_dir / "per_patient.parquet", index=False)
    psum.to_csv(out_dir / "patient_summary.csv", index=False)
    el.to_csv(out_dir / "eligibility.csv", index=False)
    C, N = agg.calib_table, agg.null_table
    C.to_csv(out_dir / "calibrated.csv", index=False)
    N.to_csv(out_dir / "null_discovery.csv", index=False)
    lb_methods = clb.METHODS_V3
    LB = clb.leaderboard(C, N, lb_methods)
    LB.to_csv(out_dir / "leaderboard.csv", index=False)
    LBs = clb.leaderboard(C, N, clb.SENSITIVITY_METHODS_V3)
    LBs.to_csv(out_dir / "leaderboard_sensitivity.csv", index=False)
    if agg.typed_table is not None:
        agg.typed_table.to_csv(out_dir / "typed_edges.csv", index=False)
    if agg.eb_table is not None:
        agg.eb_table.to_csv(out_dir / "typed_patient_eb.csv", index=False)
    if agg.spec_table is not None:
        agg.spec_table.to_csv(out_dir / "patient_specificity.csv", index=False)
    head = (f"# ClosedLoopBench 1.0 (closed-loop benchmark v3) — scale {cfg.name}, cases {args.cases} (n={len(pids)})\n\n"
            f"Protocol sha256: `{proto_sha}`\n\n"
            "Cell: call estimate [95% CI]; % of eligible patients with the expected sign (real arm).\n"
            "**R** / **X** = expected / wrong sign, admitted (calibrated table: v3 rule) or CI excluding 0 "
            "(single-arm tables); r / x = sign only.\n"
            f"Null draws: nc_random + {K} cross-patient derangements. PCMCI+ on the same {len(pc_set)} cases in every arm.\n\n")
    psum_c = psum.assign(arm=psum.arm.replace({"real": "calib_multi"})) if len(psum) else psum
    main_txt = (head + f"## Calibrated against {K} cross draws (v3 admission rule)\n\n"
                + clb.main_table(pooled, psum_c, cfg, "calib_multi", clb.METHODS_V3)
                + "\n\n## Real data, uncalibrated (single-arm case-bootstrap CI)\n\n"
                + clb.main_table(pooled, psum, cfg, "real", clb.METHODS_V3) + "\n")
    (out_dir / "main_table.md").write_text(main_txt)
    sens_txt = (head + "## Sensitivity variants, calibrated (v3 rule)\n\n"
                + clb.main_table(pooled, pd.DataFrame(), cfg, "calib_multi", ("matched", "kink", "seqtrial") + clb.SENSITIVITY_METHODS_V3)
                + "\n\n## Sensitivity variants, real data\n\n"
                + clb.main_table(pooled, psum, cfg, "real", ("matched", "kink", "seqtrial") + clb.SENSITIVITY_METHODS_V3) + "\n")
    (out_dir / "sensitivity.md").write_text(sens_txt)
    nc_txt = head
    for arm in ("nc_cross", "nc_random"):
        nc_txt += f"\n\n## Negative control: {arm} (single arm; any R/X is a false discovery)\n\n"
        nc_txt += clb.main_table(pooled, psum, cfg, arm, tuple(m for m in clb.METHODS_V3 if m != "typed_pooled")) + "\n"
    nc_txt += "\n\n## Null discovery per arm (uncalibrated)\n\n" + df_to_md(N) + "\n"
    (out_dir / "nc_tables.md").write_text(nc_txt)
    (out_dir / "leaderboard.md").write_text(head + leaderboard_md(LB) + "\n\n## Sensitivity variants\n\n"
                                            + leaderboard_md(LBs) + "\n")
    (out_dir / "typed_edges.md").write_text(head + calibrated_md(C[C.method == "typed_pooled"]) + "\n\n## Patient-specificity\n\n"
                                            + spec_md(agg.spec_table) + "\n")
    meta = dict(args={k: str(v) for k, v in vars(args).items()}, bench_version="v3", bench_name="ClosedLoopBench 1.0",
                protocol=str(args.protocol), protocol_sha256=proto_sha, code_sha256=code_hashes(),
                n_cases=len(pids), cross_draws=K, cross_salts=list(clb.CROSS_SALTS[:K]),
                seconds_cases=t_cases, seconds_finalize=t_fin, seconds_total=time.time() - t0,
                median_case_seconds=float(np.median(times)) if times else None, pcmci_cases_all_arms=len(pc_set),
                pcmci_done={a: len(s_) for a, s_ in agg.pcmci_done.items()}, errors=agg.errors[:50],
                n_errors=len(agg.errors), config=str(cfg), options=str(opts),
                n_tracks_masked=int(sum(np.isfinite(v) for st in stops.values() for v in st.values())))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s (%.0f s)", out_dir, time.time() - t0)
    print(leaderboard_md(LB))


def _init_v4() -> None:
    clb.use_table("v4")


def domain_summary(C: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    """v4 interpretive axis: per method and relation domain, calibrated admissions with the expected / wrong sign
    among the relations of that domain (real input)."""
    rows = []
    if C is None or len(C) == 0:
        return pd.DataFrame()
    R = C[C.input == "real"].copy()
    R["domain"] = [clb.relation_domain(clb.REL_BY_ID[r]) for r in R.rid]
    for m in methods:
        Q = R[R.method == m]
        if len(Q) == 0:
            continue
        for dom in ("gas_exchange", "cardiopulmonary", "drug", "drug_derived", "policy"):
            q = Q[Q.domain == dom]
            if len(q) == 0:
                continue
            rows.append(dict(method=m, domain=dom, n_relations=len(q), correct=int((q.call == "correct").sum()),
                             wrong=int((q.call == "wrong").sum()), ns_right=int((q.call == "ns_right").sum()),
                             ns_wrong=int((q.call == "ns_wrong").sum()),
                             admitted=";".join(f"{r.rid}{'+' if r.call == 'correct' else '-'}" for r in q.itertuples() if r.admitted)))
    return pd.DataFrame(rows)


def per_relation_calls(C: pd.DataFrame, methods: tuple[str, ...]) -> str:
    """Relation x method grid of calibrated calls (real input): R / X admitted, r / x sign only."""
    if C is None or len(C) == 0:
        return "(no rows)"
    R = C[C.input == "real"]
    lines = ["| relation | domain | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for r in clb.RELATIONS:
        cells = []
        for m in methods:
            q = R[(R.method == m) & (R.rid == r.rid)]
            if len(q) == 0 or not np.isfinite(q.iloc[0].est):
                cells.append("—")
                continue
            x = q.iloc[0]
            mark = {"correct": "**R**", "wrong": "**X**", "ns_right": "r", "ns_wrong": "x"}.get(x.call, "?")
            cells.append(f"{mark} {_fmt(x.est, 2)}")
        lines.append(f"| {r.rid} {r.label} | {clb.relation_domain(r)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_v4(args) -> None:
    """ClosedLoopBench 1.1 (v4): v3 methods and admission rule, v4 relation table, 2 s native grid, all cases."""
    clb.use_table("v4")
    clb.NN_BACKEND = "torch"  # exact brute-force kNN for the pooled seqtrial matching (identical neighbours)
    cfg = clb.SCALES["2s_v4"]
    proto_sha = sha256(args.protocol) if args.protocol and args.protocol.exists() else None
    if args.protocol and proto_sha is None:
        raise SystemExit(f"protocol file not found: {args.protocol}")
    from concurrent.futures import ProcessPoolExecutor

    grid = args.grid_dir
    files = sorted((grid / "grid").glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]
    pids = [f.stem for f in files]
    out_dir = args.out_dir or (OUT_V4 / "2s_all")
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = clb.METHODS_V3_COMPUTE if args.methods == ",".join(clb.METHODS) else tuple(m for m in args.methods.split(",") if m)
    log.info("v4: scale 2s, %d cases from %s", len(pids), grid)
    ev = bl.load_events(set(pids), grid)
    st = pd.read_parquet(grid / "stops.parquet")
    stops = {pid: dict(zip(g.agent, g.t_stop)) for pid, g in st.groupby("pid")}
    opts = clb.V4
    K = args.cross_draws
    part_sets = bn.cross_partner_sets(pids, ev, args.seed, K)
    t0 = time.time()
    ctx = get_context("spawn")
    # PCMCI+: one seeded subsample, the SAME cases in every arm; pressor cases first (as v3), then the rest
    n_pc = args.pcmci_cases if args.pcmci_cases >= 0 else 60
    press_agents = {clb.ACTIONS[a].agent for a in clb.PRESSORS}
    rng = np.random.default_rng(args.seed + 5)
    press = [p for p in pids if p in ev and ev[p].agent.isin(press_agents).any()]
    rest = [p for p in pids if p not in set(press)]
    order = list(rng.permutation(press)) + list(rng.permutation(rest))
    pc_set = set(order[:n_pc])
    lingam_rows = 300

    def prt_for(pid, parts):
        out, st_ = {}, {}
        for agent, pp_ in parts[pid].items():
            rec = _ev_records(ev.get(pp_))
            if agent in rec:
                out[agent] = rec[agent]
                st_[agent] = stops.get(pp_, {}).get(agent, np.inf)
        return out, st_

    jobs = []
    for i, (f, pid) in enumerate(zip(files, pids)):
        jobs.append(dict(path=str(f), pid=pid, scale="2s_v4", seed=args.seed * 100_003 + i, version="v4",
                         events=_ev_records(ev.get(pid)), stops=stops.get(pid, {}),
                         cross={arm: prt_for(pid, parts) for arm, parts in part_sets.items()},
                         lingam_rows=lingam_rows, methods=list(methods), pcmci=pid in pc_set))
    jobs.sort(key=lambda j: -int(j["pcmci"]))
    agg = clb.Aggregator(pids, cfg, n_boot=args.boot, seed=args.seed, opts=opts, typed_q=args.typed_q,
                         spec_boot=args.spec_boot)
    agg.nn_workers = args.workers
    times = []
    t1 = time.time()
    with ctx.Pool(args.workers, initializer=_init_v4) as pool:
        for k, (pid, res, dt) in enumerate(pool.imap(run_job, jobs, chunksize=1), 1):
            for arm, r in res.items():
                agg.add(pid, arm, r)
            times.append(dt)
            if k % 100 == 0 or k == len(jobs):
                log.info("%d/%d cases  (%.0f s elapsed, median %.2f s/case)", k, len(jobs), time.time() - t1, np.median(times))
    t_cases = time.time() - t1
    log.info("finalising (VAR-LiNGAM in parallel, seqtrial, calibration) ...")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_v4) as ex:
        pooled, pp = agg.finalize(lingam_boot=args.lingam_boot, lingam_max_rows=args.lingam_max_rows, seed=args.seed,
                                  executor=ex)
    t_fin = time.time() - t1 - t_cases
    psum = clb.patient_summary(pp)
    elig = agg.eligibility()
    el = (elig.groupby(["arm", "rid"]).agg(n_cases_with_events=("n_events", lambda x: int((x > 0).sum())),
                                          n_events=("n_events", "sum"), n_eligible=("eligible", "sum")).reset_index())
    pooled.to_csv(out_dir / "pooled.csv", index=False)
    pp.to_parquet(out_dir / "per_patient.parquet", index=False)
    psum.to_csv(out_dir / "patient_summary.csv", index=False)
    el.to_csv(out_dir / "eligibility.csv", index=False)
    C, N = agg.calib_table, agg.null_table
    C.to_csv(out_dir / "calibrated.csv", index=False)
    N.to_csv(out_dir / "null_discovery.csv", index=False)
    lb_methods = clb.METHODS_V3
    LB = clb.leaderboard(C, N, lb_methods)
    LB.to_csv(out_dir / "leaderboard.csv", index=False)
    DS = domain_summary(C, lb_methods)
    DS.to_csv(out_dir / "domain_summary.csv", index=False)
    if agg.typed_table is not None:
        agg.typed_table.to_csv(out_dir / "typed_edges.csv", index=False)
    if agg.eb_table is not None:
        agg.eb_table.to_csv(out_dir / "typed_patient_eb.csv", index=False)
    head = (f"# ClosedLoopBench 1.1 (closed-loop benchmark v4) — scale 2 s (native grid), all cases (n={len(pids)})\n\n"
            f"Protocol sha256: `{proto_sha}`\n\n"
            "Cell: call estimate [95% CI]; % of eligible patients with the expected sign (real arm).\n"
            "**R** / **X** = expected / wrong sign, admitted (calibrated table: v3 rule) or CI excluding 0 "
            "(single-arm tables); r / x = sign only.\n"
            f"Null draws: nc_random + {K} cross-patient derangements. PCMCI+ on the same {len(pc_set)} cases in every arm "
            "(seeded subsample, pressor cases first).\n\n")
    psum_c = psum.assign(arm=psum.arm.replace({"real": "calib_multi"})) if len(psum) else psum
    main_txt = (head + f"## Calibrated against {K} cross draws (v3 admission rule)\n\n"
                + clb.main_table(pooled, psum_c, cfg, "calib_multi", clb.METHODS_V3)
                + "\n\n## Real data, uncalibrated (single-arm case-bootstrap CI)\n\n"
                + clb.main_table(pooled, psum, cfg, "real", clb.METHODS_V3) + "\n")
    (out_dir / "main_table.md").write_text(main_txt)
    nc_txt = head
    for arm in ("nc_cross", "nc_random"):
        nc_txt += f"\n\n## Negative control: {arm} (single arm; any R/X is a false discovery)\n\n"
        nc_txt += clb.main_table(pooled, psum, cfg, arm, tuple(m for m in clb.METHODS_V3 if m != "typed_pooled")) + "\n"
    nc_txt += "\n\n## Null discovery per arm (uncalibrated)\n\n" + df_to_md(N) + "\n"
    (out_dir / "nc_tables.md").write_text(nc_txt)
    (out_dir / "leaderboard.md").write_text(head + leaderboard_md(LB) + "\n\n## Domain summary (calibrated, real input)\n\n"
                                            + (df_to_md(DS) if len(DS) else "(none)") + "\n\n## Per-relation calibrated calls\n\n"
                                            + per_relation_calls(C, lb_methods) + "\n")
    (out_dir / "typed_edges.md").write_text(head + calibrated_md(C[C.method == "typed_pooled"]) + "\n")
    meta = dict(args={k: str(v) for k, v in vars(args).items()}, bench_version="v4", bench_name="ClosedLoopBench 1.1",
                table_version=clb.TABLE_VERSION, n_relations=len(clb.RELATIONS),
                protocol=str(args.protocol), protocol_sha256=proto_sha, code_sha256=code_hashes(),
                n_cases=len(pids), cross_draws=K, cross_salts=list(clb.CROSS_SALTS[:K]),
                seconds_cases=t_cases, seconds_finalize=t_fin, seconds_total=time.time() - t0,
                median_case_seconds=float(np.median(times)) if times else None, pcmci_cases_all_arms=len(pc_set),
                pcmci_done={a: len(s_) for a, s_ in agg.pcmci_done.items()}, errors=agg.errors[:50],
                n_errors=len(agg.errors), config=str(cfg), options=str(opts),
                n_tracks_masked=int(sum(np.isfinite(v) for st_ in stops.values() for v in st_.values())))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s (%.0f s)", out_dir, time.time() - t0)
    print(leaderboard_md(LB))
    print(df_to_md(DS) if len(DS) else "")


def _i(x) -> str:
    try:
        return "na" if x is None or not np.isfinite(float(x)) else f"{int(x)}"
    except (TypeError, ValueError):
        return str(x)


def leaderboard_md(LB: pd.DataFrame) -> str:
    lines = ["| method | score (R - X) | calibrated R / X | null FA per cross draw (LOO) [max; draws with any] | "
             "null FA random-time | uncalibrated real R / X | uncalibrated null sig per cross draw [min-max] | "
             "uncalibrated null sig random-time |", "|---|---|---|---|---|---|---|---|"]
    for r in LB.itertuples():
        g = r._asdict()
        lines.append(
            f"| {r.method} | {_i(g.get('score'))} | {_i(g.get('cal_correct'))} / {_i(g.get('cal_wrong'))} of {_i(g.get('n_relations'))} | "
            f"{_fmt(g.get('null_cross_fa_per_draw', np.nan), 2)} [{_i(g.get('null_cross_fa_max'))}; {_i(g.get('null_cross_draws_any'))}/{_i(g.get('null_cross_draws'))}] | "
            f"{_i(g.get('null_random_fa'))} | {_i(g.get('uncal_correct'))} / {_i(g.get('uncal_wrong'))} | "
            f"{_fmt(g.get('uncal_null_cross_sig_mean', np.nan), 2)} [{_i(g.get('uncal_null_cross_sig_min'))}-{_i(g.get('uncal_null_cross_sig_max'))}] | "
            f"{_i(g.get('uncal_null_random_sig'))} |")
    return "\n".join(lines)


def calibrated_md(C: pd.DataFrame) -> str:
    if C is None or len(C) == 0:
        return "(no rows)"
    lines = ["| input | relation | type | base | expected | calibrated est [95% CI] | SE boot / draw | p | p (BY) | "
             "stable | admitted | call |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in C.itertuples():
        lines.append(f"| {r.input} | {r.rid} {r.label} | {r.kind} | {r.base} | {'+' if r.expected > 0 else '-'} | "
                     f"{_fmt(r.est)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_fmt(r.se_boot, 2)} / {_fmt(r.se_draw, 2)} | "
                     f"{_fmt(r.p, 2)} | {_fmt(r.p_by, 2)} | {_fmt(r.stable_frac, 2)} | {'yes' if r.admitted else 'no'} | {r.call} |")
    return "\n".join(lines)


def combine_v3(root: Path = OUT_V3) -> None:
    runs = {d.name: d for d in sorted(root.iterdir()) if (d / "pooled.csv").exists() and (d / "leaderboard.csv").exists()}
    if not runs:
        log.warning("no finished v3 runs under %s", root)
        return
    proto = root / "protocol.md"
    out = ["# ClosedLoopBench 1.0 (closed-loop benchmark v3) — combined tables", "",
           f"Protocol: `protocol.md`, sha256 `{sha256(proto) if proto.exists() else 'missing'}`.", ""]
    for k, d in runs.items():
        meta = json.loads((d / "meta.json").read_text())
        out.append(f"- {k}: n={meta['n_cases']}, protocol sha256 at run time `{meta.get('protocol_sha256')}`, "
                   f"{meta['seconds_total']:.0f} s, cross draws {meta.get('cross_draws')}, PCMCI+ cases (all arms) "
                   f"{meta.get('pcmci_cases_all_arms')}, errors {meta['n_errors']}")
    out += ["", "## 1. Leaderboard (v3 admission rule applied to every method)", ""]
    for k, d in runs.items():
        out += [f"### {k}", "", leaderboard_md(pd.read_csv(d / "leaderboard.csv")), ""]
    out += ["## 2. Calibrated method x relation table", ""]
    for k, d in runs.items():
        cfg = clb.SCALES[k.split("_")[0]]
        P = pd.read_csv(d / "pooled.csv", low_memory=False)
        out += [f"### {k}", "", clb.main_table(P, pd.DataFrame(), cfg, "calib_multi", clb.METHODS_V3), ""]
    out += ["## 3. Uncalibrated real-data table", ""]
    for k, d in runs.items():
        cfg = clb.SCALES[k.split("_")[0]]
        P = pd.read_csv(d / "pooled.csv", low_memory=False)
        out += [f"### {k}", "", clb.main_table(P, pd.DataFrame(), cfg, "real", clb.METHODS_V3), ""]
    out += ["## 4. Typed edges (typed_pooled: seqtrial response + VARX policy, v3 rule)", ""]
    for k, d in runs.items():
        C = pd.read_csv(d / "calibrated.csv")
        out += [f"### {k}", "", calibrated_md(C[(C.method == "typed_pooled") & C.input.isin(["real", "nc_random"])]), ""]
    out += ["## 5. Sensitivity leaderboard", ""]
    for k, d in runs.items():
        f = d / "leaderboard_sensitivity.csv"
        if f.exists():
            out += [f"### {k}", "", leaderboard_md(pd.read_csv(f)), ""]
    for k, d in runs.items():
        f = d / "patient_specificity.csv"
        if f.exists():
            out += ["", f"## 6. Patient-specificity split-half test, {k}", "", spec_md(pd.read_csv(f))]
    dev = root / "deviations.md"
    if dev.exists():
        out += ["", "## 7. Deviations from protocol.md", "", dev.read_text()]
    (root / "tables.md").write_text("\n".join(out) + "\n")
    log.info("wrote %s", root / "tables.md")


# ── simulator validation of the event designs (development gate for seqtrial) ─────────────────────

SIM_CONDITIONS: tuple[tuple[str, str, float], ...] = (
    ("recorded", "inert", 0.0), ("recorded", "active", 0.0), ("reversed", "inert", 0.0), ("reversed", "active", 0.0),
    ("recorded", "inert", 1.0), ("recorded", "active", 1.0))  # last two: unrecorded confounder (stress test)
SIM_SEEDS: tuple[int, ...] = (741000, 742000)
SIM_METHODS: tuple[str, ...] = ("naive", "matched", "kink", "seqtrial", "varx")
SIM_REPORT: tuple[str, ...] = ("naive", "naive_lvl", "matched", "kink", "seqtrial", "varx", "matched_v1", "kink_v1",
                               "matched_nobc", "kink_nobc", "seqtrial_nobc", "seqtrial_noco")


def _sim_job(job: dict) -> list[dict]:
    from clbench.bench import sim as bsim
    from clbench.sim import clinician_loop as cl

    t0 = time.time()
    scale, n, seed = job["scale"], job["n"], job["seed"]
    cfg = clb.SCALES[scale]
    scfg = bsim.sim_config(job["clinician"], job["drug"], w_z=job["w_z"])
    cohort = cl.simulate_cohort(scfg, n, seed=seed)
    cases = bsim.cohort_to_cases(cohort, scale)
    pids = [c.pid for c in cases]
    rng = np.random.default_rng(seed + 1)
    crosses = clb.CROSS_ARMS[: job["k_cross"]]
    perms = {arm: {act: bsim.cross_derangement(n, rng) for act in bsim.SIM_ACTIONS.values()} for arm in crosses}
    agg = clb.Aggregator(pids, cfg, n_boot=job["boot"], seed=0, opts=clb.V3)
    for i, c in enumerate(cases):
        cross = {arm: {act: cases[perms[arm][act][i]].events[act] for act in bsim.SIM_ACTIONS.values()} for arm in crosses}
        arms = clb.build_arms(c, None, seed=seed * 7 + i, arms=("real", "nc_random") + crosses, cross_events=cross)
        for arm, cc in arms.items():
            agg.add(c.pid, arm, clb.analyse_case(cc, cfg, methods=SIM_METHODS, opts=clb.V3))
    P, _ = agg.finalize(lingam_boot=0)
    C = agg.calib_table
    truth = bsim.itt_truth_bench(scfg, n, seed, scale, k_max=3, base=cohort) if job["drug"] == "active" else None
    from clbench.sim.clinician_loop import cluster_bootstrap_mean

    rows = []
    sgn = 1 if job["clinician"] == "recorded" else -1
    for r in bsim.SIM_RELATIONS:
        hs = cfg.horizons if r.kind == "response" else (0.0,)
        for h in hs:
            if r.kind == "policy":
                tm, tlo, thi, tn, texp = np.nan, np.nan, np.nan, 0, r.expected * sgn
            elif truth is None:
                tm, tlo, thi, tn, texp = 0.0, 0.0, 0.0, 0, 0
            else:
                tt = truth[(truth.rid == r.rid) & (truth.h == h)]
                tm, tlo, thi = cluster_bootstrap_mean(tt.itt.to_numpy(), tt.patient.to_numpy(), seed=0)
                tn, texp = len(tt), int(np.sign(tm)) if (tlo > 0 or thi < 0) else 0
            for m in SIM_REPORT:
                q = P[(P.arm == "real") & (P.method == m) & (P.rid == r.rid) & (P.h == h)]
                if not len(q):
                    continue
                q = q.iloc[0]
                d = dict(scale=scale, clinician=job["clinician"], drug=job["drug"], w_z=job["w_z"], seed=seed, n=n,
                         rid=r.rid, kind=r.kind, label=r.label, h=h, primary=bool(h == clb.primary_h(r, m, cfg)),
                         method=m, est=float(q.est), lo=float(q.lo), hi=float(q.hi), n_cases=int(q.n_cases),
                         n_units=q.n_units, truth=tm, truth_lo=tlo, truth_hi=thi, truth_n=tn, truth_sign=texp)
                c = C[(C.input == "real") & (C.method == m) & (C.rid == r.rid) & (C.h == h)] if C is not None else []
                if len(c):
                    c = c.iloc[0]
                    d.update(cal_est=float(c.est), cal_lo=float(c.lo), cal_hi=float(c.hi), cal_admitted=bool(c.admitted),
                             cal_stable=float(c.stable_frac))
                rows.append(d)
    # false admissions of the calibrated rule on the simulator's null inputs (every method)
    if C is not None:
        for m in SIM_REPORT + ("typed_pooled",):
            cm = C[(C.method == m) & (C.rid.isin([r.rid for r in bsim.SIM_RELATIONS]))]
            loo = cm[cm.input.str.startswith("loo:")]
            rows.append(dict(scale=scale, clinician=job["clinician"], drug=job["drug"], w_z=job["w_z"], seed=seed, n=n,
                             rid="*", kind="null_fa", method=m, est=float(loo.admitted.sum()), lo=float(len(loo)),
                             hi=float((cm[cm.input == "nc_random"].admitted).sum())))
    log.info("sim %s %s %s w_z=%s seed %d: %.0f s", scale, job["clinician"], job["drug"], job["w_z"], seed, time.time() - t0)
    return rows


def sim_validate(args) -> None:
    out = OUT_V3 / "sim_validation"
    if getattr(args, "sim_summary_only", False):
        R = pd.read_csv(out / "sim_validation.csv")
        (out / "summary.md").write_text(sim_summary_md(R))
        print(sim_summary_md(R))
        return
    out.mkdir(parents=True, exist_ok=True)
    scales = [x for x in args.sim_scales.split(",") if x]
    jobs = []
    for sc in scales:
        for ci, (clin, drug, wz) in enumerate(SIM_CONDITIONS):
            for s0 in SIM_SEEDS:
                jobs.append(dict(scale=sc, clinician=clin, drug=drug, w_z=wz, seed=s0 + 10 * ci, n=args.sim_n,
                                 k_cross=args.sim_cross, boot=args.boot))
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        rows = [r for res in pool.imap(_sim_job, jobs, chunksize=1) for r in res]
    R = pd.DataFrame(rows)
    R.to_csv(out / "sim_validation.csv", index=False)
    (out / "summary.md").write_text(sim_summary_md(R))
    meta = dict(args={k: str(v) for k, v in vars(args).items()}, code_sha256=code_hashes(), conditions=SIM_CONDITIONS,
                seeds=[j["seed"] for j in jobs], seconds=time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("wrote %s (%.0f s)", out, time.time() - t0)
    print(sim_summary_md(R))


def sim_summary_md(R: pd.DataFrame) -> str:
    """Coverage of the ITT truth by the 95% CI, bias, and sign errors, per method; primary horizons,
    response relations; policy sign recovery; calibrated false admissions on the simulator's nulls."""
    out = ["# Simulator validation of the event designs (ClosedLoopBench v3)", "",
           "Truth: exact ITT of the benchmark outcome by paired counterfactual simulation (inert drug: 0).",
           "Cells: relation x condition x seed at the primary horizon. cover = 95% CI contains the truth; "
           "wrong = CI excludes 0 with the sign opposite to the truth (inert: CI excludes 0 at all).",
           "varx reports a step response per unit dose (a different unit from the event contrasts): its "
           "cover / bias columns are not meaningful, only its signs are.",
           "Relations: R01, R02 (ventilator -> EtCO2), R05-R08 (remifentanil -> HR, MAP); policy P01, P02, P05.",
           "Conditions: recorded / reversed clinician x inert / active drug, w_z = 0 (no unrecorded "
           "confounder); w_z = 1 = the clinician also reacts to the unrecorded surgical field (stress test).", ""]
    Rr = R[(R.kind == "response") & R.primary.astype(bool)]
    for (sc, wz), G in Rr.groupby(["scale", "w_z"]):
        out += [f"## scale {sc}, unrecorded confounder w_z = {wz}", "",
                "| method | drug | cells | cover truth | CI excl. 0, wrong / spurious | correct sign (CI excl. 0) | mean bias | mean |bias| |",
                "|---|---|---|---|---|---|---|---|"]
        for (m, drug), g in G.groupby(["method", "drug"]):
            cover = ((g.lo <= g.truth) & (g.hi >= g.truth)).mean()
            sig = (g.lo > 0) | (g.hi < 0)
            if drug == "inert":
                wrong, right = sig.sum(), 0
            else:
                wrong = (sig & (np.sign(g.est) != np.sign(g.truth))).sum()
                right = (sig & (np.sign(g.est) == np.sign(g.truth))).sum()
            bias = g.est - g.truth
            out.append(f"| {m} | {drug} | {len(g)} | {100 * cover:.0f}% | {wrong} | {right} | {bias.mean():+.3f} | {bias.abs().mean():.3f} |")
        out.append("")
    Rp = R[R.kind == "policy"]
    if len(Rp):
        out += ["## Policy relations (expected sign flips with the reversed clinician)", "",
                "| scale | w_z | method | cells | correct (CI excl. 0) | wrong (CI excl. 0) |", "|---|---|---|---|---|---|"]
        for (sc, wz, m), g in Rp.groupby(["scale", "w_z", "method"]):
            sig = (g.lo > 0) | (g.hi < 0)
            out.append(f"| {sc} | {wz} | {m} | {len(g)} | {(sig & (np.sign(g.est) == g.truth_sign)).sum()} | "
                       f"{(sig & (np.sign(g.est) == -g.truth_sign)).sum()} |")
        out.append("")
    Rn = R[R.kind == "null_fa"]
    if len(Rn):
        out += ["## Calibrated rule on the simulator's null inputs (sim relations only)", "",
                "| scale | method | cross LOO admissions / tests | random-time admissions |", "|---|---|---|---|"]
        for (sc, m), g in Rn.groupby(["scale", "method"]):
            out.append(f"| {sc} | {m} | {int(g.est.sum())} / {int(g.lo.sum())} | {int(g.hi.sum())} |")
        out.append("")
    cal = Rr[Rr.cal_est.notna()] if "cal_est" in Rr else Rr.iloc[:0]
    if len(cal):
        out += ["## Calibrated admissions on real (simulated) data, primary horizons", "",
                "| scale | w_z | method | drug | cells | admitted correct | admitted wrong / spurious |", "|---|---|---|---|---|---|---|"]
        for (sc, wz, m, drug), g in cal.groupby(["scale", "w_z", "method", "drug"]):
            adm = g.cal_admitted.astype(bool)
            if drug == "inert":
                right, wrong = 0, adm.sum()
            else:
                right = (adm & (np.sign(g.cal_est) == np.sign(g.truth))).sum()
                wrong = (adm & (np.sign(g.cal_est) != np.sign(g.truth))).sum()
            out.append(f"| {sc} | {wz} | {m} | {drug} | {len(g)} | {right} | {wrong} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
