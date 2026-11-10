"""
Closed-loop bias and timescale-separated identifiability — simulation study (theory note:
paper/closed_loop_theory.md; simulator: src/clbench/sim/clinician_loop.py).

Experiments (``--exp``):

    attribution  y→y (baroreflex MAP→HR, β < 0) under {no driver, no controller}, {controller
                 only}, {driver only}, {both}: which confounder breaks which estimator
    headline     realistic cohort (gain 1, anticipation 60 s, driver timescale 200 s): every
                 estimator at every scale, patient-bootstrap CIs, status vs the planted truth
    phase        grid over controller gain × anticipation horizon × driver timescale; status of
                 each estimator per cell (the phase diagram)
    sandwich     residualisation width W × driver timescale for the fine-scale ARX: tests the
                 O(τ_loop / W) + leakage(W / τ_s) bias bound (Proposition 4)
    monitor      monitor averaging width for HR / MAP at 2 s: when fine sampling stops helping
    horizon      event estimators vs horizon h for several anticipation horizons H (Props 1–3)
    info         clinician information set (recorded vitals only / + surgical field / + plan):
                 when does placebo calibration work (Proposition 2)
    real         prediction check on VitalDB 2 s numerics: does residualisation move the
                 MAP→HR sign the way the theory predicts?

v2 experiments (theory-note revision after verification; pre-registered in
runs/iclr/sim_closed_loop/v2/protocol.md, whose sha256 is written to v2/manifest.json):

    lemma        Lemma FI: future-inaction vs eligible control pools, sufficient-statistic matching
                 at 1 s, inert and active drug, four clinicians, ITT truth by paired twin simulation
    headline2    the headline event designs (30 s / 2 s / 1 s) with the eligible (ITT) pool vs the v1
                 future-inaction pool, active and inert drug, three clinicians, estimator-specific
                 truths, plus the 30 s direct-method ARX
    beta_grid    residualised / raw ARX for MAP→HR over β × driver timescale × width W
    monitor_delay  HR display latency and averaging at 2 s: sign of residualised MAP→HR
    theory       numerical checks of Cor. 1(b) crossover and the Prop. 4 leakage exponent
    v2report     compact results tables (v2/results.md)

Everything is written under ``--out-dir`` (default runs/iclr/sim_closed_loop/) as CSV / JSON,
with figures and report.md produced by ``--exp report``.

USAGE
─────
  PYTHONPATH=src python scripts/46_sim_closed_loop.py --exp all
  PYTHONPATH=src python scripts/46_sim_closed_loop.py --exp phase --n-patients 40
  PYTHONPATH=src python scripts/46_sim_closed_loop.py --exp report
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from clbench.sim.clinician_loop import (
    EventSpec,
    SimConfig,
    arx_fit,
    bootstrap_median,
    cluster_bootstrap_mean,
    contemporaneous_slope,
    event_effects,
    exog_labels,
    granger_step_response,
    observe,
    onset_precedence,
    sign_status,
    simulate_cohort,
    step_truth,
    with_controller,
    with_drivers,
    with_physiology,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("46_sim_closed_loop")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "iclr" / "sim_closed_loop"
FINE_DIR = ROOT / "data" / "processed_vitaldb" / "fine_2s"

H_EVENTS = (30, 60, 120)
UY_SPECS = {
    "remi_up_hr": EventSpec("remi", "hr", +1),
    "remi_up_map": EventSpec("remi", "map", +1),
    "vent_up_co2": EventSpec("vent", "co2", +1),
    "vent_down_co2": EventSpec("vent", "co2", -1),
}
# ARX settings per scale: (lags, residualisation width in samples)
ARX = {"beat": (6, 61), "2s": (4, 31), "30s": (2, 11)}
N_BOOT = 1000


def base_config() -> SimConfig:
    return with_controller(SimConfig(), gain=1.0, anticipation_s=60.0)


# ---------------------------------------------------------------------------
# Estimator batteries
# ---------------------------------------------------------------------------


def yy_rows(cohort, *, scales=("beat", "2s", "30s"), monitor=(2, 2), widths: dict | None = None,
            reverse: bool = False) -> list[dict]:
    """Baroreflex (MAP → HR) estimators; the planted sign is negative. ``reverse`` adds HR → MAP."""
    truth = float(np.median(cohort.gains["beta"]))
    kappa = float(np.median(cohort.gains["kappa"]))
    rows: list[dict] = []

    def add(name: str, vals, truth_sign: float, truth_val: float, kind: str) -> None:
        vals = np.asarray(vals, dtype=float)
        est, lo, hi = bootstrap_median(vals, n_boot=N_BOOT)
        fin = vals[np.isfinite(vals)]
        rows.append({
            "family": "y->y", "estimator": name, "kind": kind, "est": est, "lo": lo, "hi": hi,
            "truth": truth_val, "truth_sign": truth_sign, "status": sign_status(lo, hi, truth_sign),
            "sign_agreement": float(np.mean(np.sign(fin) == truth_sign)) if len(fin) else np.nan,
            "n": int(len(fin)),
        })

    for scale in scales:
        o = observe(cohort, scale, monitor_hr_s=monitor[0], monitor_map_s=monitor[1])
        p, w = ARX[scale]
        if widths and scale in widths:
            w = widths[scale]
        h, m = o.hr, o.map
        if scale == "30s":
            add("30s_contemp", [contemporaneous_slope(h[i], m[i]) for i in range(cohort.n)], -1.0, truth, "slope")
            add("30s_onset", [onset_precedence(h[i], m[i]) for i in range(cohort.n)], +1.0, np.nan, "score")
            h, m = h[:, ::2], m[:, ::2]  # non-overlapping 30 s windows
            add("30s_granger", [arx_fit(h[i], m[i], p)["coupling"] for i in range(cohort.n)], -1.0, np.nan, "coupling")
            add("30s_granger_resid", [arx_fit(h[i], m[i], p, width=w)["coupling"] for i in range(cohort.n)],
                -1.0, np.nan, "coupling")
            continue
        add(f"{scale}_arx_raw", [arx_fit(h[i], m[i], p)["static"] for i in range(cohort.n)], -1.0, truth, "static")
        add(f"{scale}_arx_resid", [arx_fit(h[i], m[i], p, width=w)["static"] for i in range(cohort.n)],
            -1.0, truth, "static")
        if reverse:  # HR → MAP (cardiac output, κ > 0)
            add(f"{scale}_arx_raw_rev", [arx_fit(m[i], h[i], p)["static"] for i in range(cohort.n)], +1.0, kappa, "static")
            add(f"{scale}_arx_resid_rev", [arx_fit(m[i], h[i], p, width=w)["static"] for i in range(cohort.n)],
                +1.0, kappa, "static")
    return rows


_TRUTH_CACHE: dict = {}


def _truth(cfg: SimConfig, key: str, scale: str, horizons, functional: str = "pretrend") -> dict[int, float]:
    spec = UY_SPECS[key]
    ck = (key, scale, tuple(horizons), cfg.physiology, cfg.controller.remi_step, cfg.controller.vent_step, functional)
    if ck not in _TRUTH_CACHE:
        _TRUTH_CACHE[ck] = step_truth(cfg, spec, scale, tuple(horizons), functional=functional)
    return _TRUTH_CACHE[ck]


# estimand functional of each event estimator (verifier v2: the 30 s naive functional differs from the
# pre-trend functional by up to 1.5x; v1 used the pre-trend functional for every estimator)
FUNCTIONAL = {"naive": "naive", "matched": "naive", "naive_exog": "naive", "pretrend": "pretrend",
              "kink": "pretrend", "kink_rich": "pretrend"}


def uy_rows(cohort, *, specs=("remi_up_hr", "vent_up_co2"), horizons=H_EVENTS, granger: bool = True,
            scales=("30s", "2s", "beat"), v1_pool: bool = False) -> list[dict]:
    """Event-study, matched, pre-trend, placebo-kink, oracle and Granger estimators of u → y.

    v2: controls are eligible non-decision times (pre-t information only) and events are kept
    whatever happens after them (ITT); truths are estimator-specific functionals. ``v1_pool`` adds
    the v1 future-inaction pool with clean-event selection for matched / kink / kink_rich, reported
    as ``<scale>_<method>_v1pool``."""
    cfg = cohort.config
    rows: list[dict] = []
    obs = {sc: observe(cohort, sc) for sc in scales}
    methods = {"30s": ["naive", "matched"], "2s": ["naive", "pretrend", "kink", "kink_rich"], "beat": ["kink", "kink_rich"]}
    for key in specs:
        spec = UY_SPECS[key]
        for scale in scales:
            o = obs[scale]
            lab = exog_labels(cohort, o, spec.input)
            variants = [("", pd.DataFrame(event_effects(o, spec, horizons, exog_labels=lab)))]
            if v1_pool:
                variants.append(("_v1pool", pd.DataFrame(event_effects(o, spec, horizons, exog_labels=lab,
                                                                        control_pool="future_inaction", itt=False))))
            for suffix, R in variants:
                meths = methods[scale] + (["naive_exog"] if scale == "2s" else [])
                if suffix:
                    meths = [m for m in meths if m in ("matched", "kink", "kink_rich")]
                for h in horizons:
                    g = R[R.h == h] if len(R) else R
                    for meth in meths:
                        tr = _truth(cfg, key, scale, horizons, FUNCTIONAL[meth])
                        if len(g) == 0:
                            est = lo = hi = np.nan
                            n_ev = n_pt = 0
                        else:
                            gg = g[g.exog] if meth == "naive_exog" else g
                            col = "naive" if meth == "naive_exog" else meth
                            est, lo, hi = cluster_bootstrap_mean(gg[col].values, gg.patient.values, n_boot=N_BOOT)
                            n_ev, n_pt = len(gg), gg.patient.nunique()
                        rows.append({
                            "family": "u->y", "spec": key, "estimator": f"{scale}_{meth}{suffix}", "h": h, "est": est,
                            "lo": lo, "hi": hi, "truth": tr[h], "truth_sign": float(np.sign(tr[h])),
                            "status": sign_status(lo, hi, np.sign(tr[h])), "n_events": n_ev, "n_patients": n_pt,
                        })
            tr = _truth(cfg, key, scale, horizons, "naive")
            if scale == "30s" and granger:
                o30 = obs["30s"]
                y = getattr(o30, spec.target)[:, ::2]
                u = getattr(o30, spec.input)[:, ::2]
                other = o30.map[:, ::2] if spec.target == "hr" else o30.hr[:, ::2]
                for h in horizons:
                    K = max(1, int(round(h / 30)))
                    vals = [granger_step_response(y[i], u[i], [other[i]], 2, K) * abs(
                        cfg.controller.remi_step if spec.input == "remi" else cfg.controller.vent_step) * spec.direction
                        for i in range(cohort.n)]
                    est, lo, hi = bootstrap_median(np.array(vals), n_boot=N_BOOT)
                    rows.append({
                        "family": "u->y", "spec": key, "estimator": "30s_granger", "h": h, "est": est, "lo": lo,
                        "hi": hi, "truth": tr[h], "truth_sign": float(np.sign(tr[h])),
                        "status": sign_status(lo, hi, np.sign(tr[h])), "n_events": np.nan,
                        "n_patients": int(np.isfinite(vals).sum()),
                    })
    return rows


def cohort_meta(cohort) -> dict:
    ev = cohort.events
    hours = cohort.n * cohort.T / 3600.0
    return {
        "remi_changes_per_h": len(ev["remi"]) / hours,
        "vent_changes_per_h": len(ev["vent"]) / hours,
        "remi_exog_share": float(ev["remi"]["exog"].mean()) if len(ev["remi"]) else np.nan,
        "vent_exog_share": float(ev["vent"]["exog"].mean()) if len(ev["vent"]) else np.nan,
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def exp_attribution(args) -> None:
    # The driver always exists and the clinician can see it; what changes is whether it moves the
    # physiology (stim gains) and whether the clinician reacts (gain). Exogenous dose changes
    # (1 / h remifentanil) are present in every configuration.
    base = base_config()
    inert = with_physiology(base, stim_hr=0.0, stim_map=0.0)
    configs = {
        "neither": with_controller(inert, gain=0.0),
        "controller_only": inert,
        "driver_only": with_controller(base, gain=0.0),
        "both": base,
    }
    rows = []
    for name, cfg in configs.items():
        c = simulate_cohort(cfg, args.n_patients, seed=args.seed)
        for r in yy_rows(c):
            rows.append({"config": name, **cohort_meta(c), **r})
        log.info("attribution %s done", name)
    pd.DataFrame(rows).to_csv(args.out_dir / "attribution.csv", index=False)


def exp_headline(args) -> None:
    rows = []
    for name, cfg in {"realistic": base_config(), "open_loop": with_controller(base_config(), gain=0.0)}.items():
        c = simulate_cohort(cfg, args.n_headline, seed=args.seed + 1)
        meta = cohort_meta(c)
        for r in yy_rows(c) + uy_rows(c, specs=tuple(UY_SPECS)):
            rows.append({"config": name, **meta, **r})
        log.info("headline %s done (%s)", name, {k: round(v, 2) for k, v in meta.items()})
    pd.DataFrame(rows).to_csv(args.out_dir / "headline.csv", index=False)


def exp_phase(args) -> None:
    gains = [0.0, 0.5, 1.0, 2.0, 4.0]
    antic = [0.0, 60.0, 180.0]
    taus = [20.0, 60.0, 200.0, 600.0, 2000.0]
    out = args.out_dir / "phase.csv"
    done = set()
    rows: list[dict] = []
    if out.exists() and not args.fresh:
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = set(zip(prev.gain, prev.anticipation_s, prev.tau_d))
    k = 0
    t0 = time.time()
    for tau in taus:
        for H in antic:
            for g in gains:
                k += 1
                if (g, H, tau) in done:
                    continue
                if g == 0.0 and H > 0:
                    continue  # anticipation is irrelevant without feedback; filled from H = 0 below
                cfg = with_drivers(with_controller(base_config(), gain=g, anticipation_s=H), timescale_s=tau)
                c = simulate_cohort(cfg, args.n_patients, seed=args.seed + 100 + k)
                meta = cohort_meta(c)
                cell = yy_rows(c) + uy_rows(c, v1_pool=getattr(args, "v1_pool", False))
                for r in cell:
                    rows.append({"gain": g, "anticipation_s": H, "tau_d": tau, **meta, **r})
                    if g == 0.0:
                        for H2 in antic[1:]:
                            rows.append({"gain": g, "anticipation_s": H2, "tau_d": tau, **meta, **r})
                pd.DataFrame(rows).to_csv(out, index=False)
                log.info("phase cell gain=%s H=%s tau=%s  (%.0f s elapsed)", g, H, tau, time.time() - t0)


def exp_sandwich(args) -> None:
    taus = [20.0, 60.0, 200.0, 600.0, 2000.0]
    widths = [5, 9, 15, 21, 31, 61, 121, 241, 481, None]
    rows = []
    for tau in taus:
        for meas in ("default", "no_meas_noise"):
            cfg = with_drivers(base_config(), timescale_s=tau)
            if meas == "no_meas_noise":
                cfg = with_physiology(cfg, meas_hr=0.0, meas_map=0.0)
            c = simulate_cohort(cfg, args.n_patients, seed=args.seed + 7)
            truth = float(np.median(c.gains["beta"]))
            for scale in ("beat", "2s"):
                o = observe(c, scale)
                p = ARX[scale][0]
                for w in widths:
                    ws = None if w is None else max(3, int(round(w / o.step_s)) | 1)
                    vals = np.array([arx_fit(o.hr[i], o.map[i], p, width=ws)["static"] for i in range(c.n)])
                    est, lo, hi = bootstrap_median(vals, n_boot=N_BOOT)
                    rows.append({"tau_d": tau, "tau_symp": cfg.drivers.tau_symp, "meas": meas, "scale": scale,
                                 "width_s": np.nan if w is None else ws * o.step_s, "est": est, "lo": lo,
                                 "hi": hi, "truth": truth, "rel_bias": (est - truth) / abs(truth),
                                 "sign_agreement": float(np.mean(vals[np.isfinite(vals)] < 0))})
            log.info("sandwich tau=%s %s done", tau, meas)
    pd.DataFrame(rows).to_csv(args.out_dir / "sandwich.csv", index=False)


def exp_monitor(args) -> None:
    """Monitor averaging widths × an optional ventilation-frequency common driver (0.2 Hz)."""
    rows = []
    for vent_name, osc in (("no_vent_osc", (0.0, 0.0)), ("vent_osc", (1.0, 4.0))):
        cfg = with_physiology(base_config(), vent_osc_hr=osc[0], vent_osc_map=osc[1])
        c = simulate_cohort(cfg, args.n_patients, seed=args.seed + 11)
        for mh in (1, 2, 4, 8, 12, 16):
            for mm in (1, 2, 4, 8):
                for r in yy_rows(c, scales=("2s",), monitor=(mh, mm), reverse=True):
                    rows.append({"vent_osc": vent_name, "monitor_hr_s": mh, "monitor_map_s": mm, **r})
        for r in yy_rows(c, scales=("beat",), reverse=True):
            rows.append({"vent_osc": vent_name, "monitor_hr_s": 0, "monitor_map_s": 0, **r})
        log.info("monitor %s done", vent_name)
    pd.DataFrame(rows).to_csv(args.out_dir / "monitor.csv", index=False)


def exp_horizon(args) -> None:
    horizons = (10, 20, 30, 45, 60, 90, 120, 180, 240)
    rows = []
    for H in (0.0, 30.0, 60.0, 120.0, 240.0):
        cfg = with_controller(base_config(), anticipation_s=H)
        c = simulate_cohort(cfg, args.n_headline, seed=args.seed + 21)
        for r in uy_rows(c, specs=("remi_up_hr", "vent_up_co2"), horizons=horizons, granger=False, scales=("2s",)):
            rows.append({"anticipation_s": H, **r})
        log.info("horizon H=%s done", H)
    pd.DataFrame(rows).to_csv(args.out_dir / "horizon.csv", index=False)


def exp_info(args) -> None:
    sets = {
        "vitals_only": dict(w_z=0.0, anticipation_s=0.0),
        "vitals+field": dict(w_z=1.0, anticipation_s=0.0),
        "vitals+field+plan120": dict(w_z=1.0, anticipation_s=120.0),
        "plan120_only": dict(w_y=0.0, w_z=0.0, anticipation_s=120.0),
    }
    rows = []
    for name, kw in sets.items():
        c = simulate_cohort(with_controller(base_config(), **kw), args.n_headline, seed=args.seed + 31)
        meta = cohort_meta(c)
        for r in uy_rows(c, specs=("remi_up_hr", "vent_up_co2"), granger=False):
            rows.append({"info_set": name, **meta, **r})
        log.info("info %s done (%s)", name, {k: round(v, 2) for k, v in meta.items()})
    pd.DataFrame(rows).to_csv(args.out_dir / "info.csv", index=False)


def exp_pathway(args) -> None:
    """Pathway-speed criterion (Proposition 3): drug onset τ_pk against driver-footprint τ_s."""
    rows = []
    for tau_pk in (5.0, 20.0, 100.0, 300.0):
        for tau_d in (60.0, 200.0, 2000.0):
            cfg = with_drivers(with_physiology(base_config(), tau_pk=tau_pk), timescale_s=tau_d)
            c = simulate_cohort(cfg, args.n_headline, seed=args.seed + 41)
            for r in uy_rows(c, specs=("remi_up_hr",), horizons=(30, 60, 120), granger=False, scales=("2s",)):
                rows.append({"tau_pk": tau_pk, "tau_d": tau_d, "tau_symp": cfg.drivers.tau_symp, **r})
            log.info("pathway tau_pk=%s tau_d=%s done", tau_pk, tau_d)
    pd.DataFrame(rows).to_csv(args.out_dir / "pathway.csv", index=False)


def exp_real(args) -> None:
    """MAP→HR and HR→MAP lagged gains on VitalDB 2 s numerics, raw vs residualised."""
    files = sorted(glob.glob(str(FINE_DIR / "*.parquet")))
    widths = [None, 301, 151, 61, 31, 21, 11]  # samples of 2 s
    per: list[dict] = []
    for f in files:
        d = pd.read_parquet(f, columns=["hr", "map"])
        h = d["hr"].to_numpy(dtype=float, copy=True)
        m = d["map"].to_numpy(dtype=float, copy=True)
        h[(h < 30) | (h > 180)] = np.nan
        m[(m < 30) | (m > 200)] = np.nan
        if (np.isfinite(h) & np.isfinite(m)).sum() < 1500:
            continue
        pid = Path(f).stem
        for w in widths:
            fwd = arx_fit(h, m, 4, width=w)
            rev = arx_fit(m, h, 4, width=w)
            per.append({"patient": pid, "width_samples": -1 if w is None else w, "scale": "2s",
                        "map_to_hr_static": fwd["static"], "map_to_hr_coupling": fwd["coupling"],
                        "hr_to_map_static": rev["static"], "hr_to_map_coupling": rev["coupling"], "n": fwd["n"]})
        # 30 s aggregation of the same case (median of 15 samples, non-overlapping)
        L = (len(h) // 15) * 15
        h30 = np.nanmedian(h[:L].reshape(-1, 15), axis=1)
        m30 = np.nanmedian(m[:L].reshape(-1, 15), axis=1)
        for w in (None, 11):
            fwd = arx_fit(h30, m30, 2, width=w)
            rev = arx_fit(m30, h30, 2, width=w)
            per.append({"patient": pid, "width_samples": -1 if w is None else w, "scale": "30s",
                        "map_to_hr_static": fwd["static"], "map_to_hr_coupling": fwd["coupling"],
                        "hr_to_map_static": rev["static"], "hr_to_map_coupling": rev["coupling"], "n": fwd["n"]})
    P = pd.DataFrame(per)
    P.to_parquet(args.out_dir / "real_per_patient.parquet", index=False)
    rows = []
    for (scale, w), g in P.groupby(["scale", "width_samples"]):
        for col, sgn in (("map_to_hr_coupling", -1), ("map_to_hr_static", -1), ("hr_to_map_coupling", +1),
                         ("hr_to_map_static", +1)):
            v = g[col].to_numpy(dtype=float)
            est, lo, hi = bootstrap_median(v, n_boot=N_BOOT)
            fin = v[np.isfinite(v)]
            rng = np.random.default_rng(0)
            agree = np.sign(fin) == sgn
            bs = [agree[rng.integers(0, len(agree), len(agree))].mean() for _ in range(N_BOOT)]
            rows.append({"scale": scale, "width_samples": w, "width_s": np.nan if w < 0 else w * (2 if scale == "2s" else 30),
                         "quantity": col, "expected_sign": sgn, "median": est, "lo": lo, "hi": hi,
                         "sign_agreement": float(agree.mean()), "agree_lo": float(np.percentile(bs, 2.5)),
                         "agree_hi": float(np.percentile(bs, 97.5)), "n_patients": int(len(fin))})
    pd.DataFrame(rows).to_csv(args.out_dir / "real_check.csv", index=False)
    log.info("real check: %d patients", P.patient.nunique())


RAW_DIR = ROOT / "data" / "raw" / "physionet.org" / "files" / "vitaldb" / "1.0.0" / "vital_files"


def arterial_beats(art: np.ndarray, fs: float) -> pd.DataFrame:
    """
    Beat-to-beat series from an arterial pressure waveform: systolic peaks (low-passed at 8 Hz),
    inter-beat interval (IBI, s) to the next peak, SBP at the peak, DBP (minimum) and MAP (mean)
    over the beat. Implausible or ectopic-looking beats are NaN; a time gap > 2 s between
    accepted peaks inserts a NaN row so lags never bridge a gap.
    """
    from scipy.signal import butter, filtfilt, find_peaks

    x = np.asarray(art, dtype=float).ravel()
    fin = np.isfinite(x)
    xf = np.full_like(x, np.nan)
    b, a = butter(3, 8.0 / (fs / 2.0))
    # filter finite runs separately
    edges = np.flatnonzero(np.diff(np.concatenate([[0], fin.astype(int), [0]])))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        if s1 - s0 > 3 * fs:
            xf[s0:s1] = filtfilt(b, a, x[s0:s1])
    peaks, _ = find_peaks(np.nan_to_num(xf, nan=-1e3), distance=int(0.33 * fs), prominence=15.0)
    if len(peaks) < 100:
        return pd.DataFrame()
    t = peaks / fs
    ibi = np.diff(t)
    pk = peaks[:-1]
    sbp = xf[pk]
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(xf))])
    cn = np.concatenate([[0], np.cumsum(np.isfinite(xf))])
    nxt = peaks[1:]
    mp = (cs[nxt] - cs[pk]) / np.maximum(nxt - pk, 1)
    dbp = np.array([np.nanmin(xf[i:j]) for i, j in zip(pk, nxt)])
    complete = (cn[nxt] - cn[pk]) == (nxt - pk)
    med = pd.Series(ibi).rolling(11, center=True, min_periods=5).median().to_numpy()
    ok = (complete & (ibi > 0.33) & (ibi < 2.0) & (sbp > 60) & (sbp < 250) & (dbp > 25) & (dbp < 150)
          & (sbp - dbp > 15) & (np.abs(ibi / med - 1.0) < 0.2))
    df = pd.DataFrame({"t": t[:-1], "ibi": ibi, "hr": 60.0 / ibi, "sbp": sbp, "map": mp, "dbp": dbp})
    df.loc[~ok, ["ibi", "hr", "sbp", "map", "dbp"]] = np.nan
    gap = np.flatnonzero(np.diff(df.t.to_numpy()) > 2.0)
    if len(gap):
        filler = pd.DataFrame({"t": df.t.to_numpy()[gap] + 1e-3})
        df = pd.concat([df, filler], ignore_index=True).sort_values("t", ignore_index=True)
    return df


def _lead(x: np.ndarray) -> np.ndarray:
    """x shifted so that its lag-1 value at k is x[k] (lets arx_fit include a lag-0 pressure term)."""
    return np.append(x[1:], np.nan)


def exp_beat_real(args) -> None:
    """Pilot: MAP/SBP → HR at beat level (from the arterial waveform) vs the 2 s monitor numerics of
    the same cases, raw and residualised."""
    import vitaldb

    files = sorted(RAW_DIR.glob("*.vital"))
    rows = []
    for f in files:
        pid = f"vdb{f.stem}"
        try:
            vf = vitaldb.VitalFile(str(f), ["SNUADC/ART"])
            art = vf.to_numpy(["SNUADC/ART"], 1.0 / 500.0)
        except Exception as e:  # pragma: no cover - data dependent
            log.warning("%s: could not read ART (%s)", pid, e)
            continue
        if art is None or np.isfinite(art).sum() < 500 * 600:
            log.warning("%s: too little ART", pid)
            continue
        B = arterial_beats(art[:, 0], 500.0)
        if len(B) < 2000:
            continue
        n_ok = int(B["hr"].notna().sum())
        hr_b, sbp_b, map_b = B["hr"].to_numpy(), B["sbp"].to_numpy(), B["map"].to_numpy()
        for w in (None, 121, 61, 31, 15):
            for pname, pv in (("sbp", sbp_b), ("map", map_b)):
                for lag0 in (True, False):
                    x = _lead(pv) if lag0 else pv
                    r = arx_fit(hr_b, x, 4, width=w)
                    rev = arx_fit(pv, hr_b, 4, width=w)
                    rows.append({"patient": pid, "source": "beat", "pressure": pname, "lag0": lag0,
                                 "width": -1 if w is None else w, "map_to_hr_static": r["static"],
                                 "map_to_hr_coupling": r["coupling"], "hr_to_map_static": rev["static"],
                                 "n": r["n"], "n_beats_ok": n_ok})
        # beat series block-averaged onto a 2 s grid (same physiology, monitor-like resolution)
        tb = B["t"].to_numpy()
        grid = np.floor(tb / 2.0).astype(int)
        G = pd.DataFrame({"g": grid, "hr": hr_b, "map": map_b}).groupby("g").mean()
        full = G.reindex(np.arange(G.index.min(), G.index.max() + 1))
        for w in (None, 31):
            r = arx_fit(full.hr.to_numpy(), full["map"].to_numpy(), 4, width=w)
            rev = arx_fit(full["map"].to_numpy(), full.hr.to_numpy(), 4, width=w)
            rows.append({"patient": pid, "source": "beat_to_2s", "pressure": "map", "lag0": False,
                         "width": -1 if w is None else w, "map_to_hr_static": r["static"],
                         "map_to_hr_coupling": r["coupling"], "hr_to_map_static": rev["static"], "n": r["n"],
                         "n_beats_ok": n_ok})
        fp = FINE_DIR / f"{pid}.parquet"
        if fp.exists():
            d = pd.read_parquet(fp, columns=["hr", "map"])
            h = d["hr"].to_numpy(dtype=float, copy=True)
            m = d["map"].to_numpy(dtype=float, copy=True)
            h[(h < 30) | (h > 180)] = np.nan
            m[(m < 30) | (m > 200)] = np.nan
            for w in (None, 31):
                r = arx_fit(h, m, 4, width=w)
                rev = arx_fit(m, h, 4, width=w)
                rows.append({"patient": pid, "source": "monitor_2s", "pressure": "map", "lag0": False,
                             "width": -1 if w is None else w, "map_to_hr_static": r["static"],
                             "map_to_hr_coupling": r["coupling"], "hr_to_map_static": rev["static"], "n": r["n"],
                             "n_beats_ok": n_ok})
        log.info("%s: %d beats (%d accepted)", pid, len(B), n_ok)
    pd.DataFrame(rows).to_csv(args.out_dir / "beat_real.csv", index=False)


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------

GOOD, BAD, NEUTRAL, INK, INK2 = "#0ca30c", "#d03b3b", "#e4e3df", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _status_grid(ax, sub: pd.DataFrame, title: str, taus, gains, value_fmt) -> None:
    import matplotlib.patches as mpatches  # noqa: F401

    ax.set_xlim(-0.5, len(taus) - 0.5)
    ax.set_ylim(-0.5, len(gains) - 0.5)
    for xi, tau in enumerate(taus):
        for yi, g in enumerate(gains):
            r = sub[(sub.tau_d == tau) & (sub.gain == g)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            col = {"correct": GOOD, "wrong": BAD}.get(r.status, NEUTRAL)
            ax.add_patch(__import__("matplotlib").patches.Rectangle((xi - 0.47, yi - 0.47), 0.94, 0.94, color=col, lw=0))
            mark = {"correct": "+", "wrong": "×"}.get(r.status, "?")
            ax.text(xi, yi, f"{mark}\n{value_fmt(r)}", ha="center", va="center", fontsize=6.5,
                    color="white" if r.status in ("correct", "wrong") else INK)
    ax.set_xticks(range(len(taus)), [f"{int(t)}" for t in taus], fontsize=7)
    ax.set_yticks(range(len(gains)), [f"{g:g}" for g in gains], fontsize=7)
    ax.set_title(title, fontsize=8, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK2)


def fig_phase(out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = pd.read_csv(out_dir / "phase.csv")
    taus = sorted(P.tau_d.unique())
    gains = sorted(P.gain.unique())
    antic = sorted(P.anticipation_s.unique())
    made = []
    yy = ["30s_granger", "30s_contemp", "2s_arx_raw", "2s_arx_resid", "beat_arx_raw", "beat_arx_resid"]
    fig, axes = plt.subplots(len(yy), len(antic), figsize=(2.3 * len(antic) + 0.6, 1.9 * len(yy)), squeeze=False)
    for i, est in enumerate(yy):
        for j, H in enumerate(antic):
            sub = P[(P.family == "y->y") & (P.estimator == est) & (P.anticipation_s == H)]
            fmt = (lambda r: f"{r.est / r.truth:.2f}") if est.endswith(("raw", "resid")) else (lambda r: f"{r.est:.2f}")
            _status_grid(axes[i, j], sub, f"{est}  |  H = {int(H)} s", taus, gains, fmt)
            if j == 0:
                axes[i, j].set_ylabel("controller gain", fontsize=7, color=INK2)
            if i == len(yy) - 1:
                axes[i, j].set_xlabel("driver timescale τ_d (s)", fontsize=7, color=INK2)
    fig.suptitle("Baroreflex MAP→HR (β < 0): + correct, × wrong sign (95% CI excludes 0), ? inconclusive.\n"
                 "Cell text: median estimate / truth (fine scales) or median coupling (30 s).", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_yy.png", dpi=170)
    plt.close(fig)
    made.append("phase_yy.png")

    for spec in ("remi_up_hr", "vent_up_co2"):
        ests = ["30s_naive", "30s_matched", "30s_granger", "2s_pretrend", "2s_kink", "2s_kink_rich", "2s_naive_exog"]
        for h in (60, 120):
            fig, axes = plt.subplots(len(ests), len(antic), figsize=(2.3 * len(antic) + 0.6, 1.9 * len(ests)),
                                     squeeze=False)
            for i, est in enumerate(ests):
                for j, H in enumerate(antic):
                    sub = P[(P.family == "u->y") & (P.spec == spec) & (P.estimator == est) & (P.h == h)
                            & (P.anticipation_s == H)]
                    _status_grid(axes[i, j], sub, f"{est}  |  H = {int(H)} s", taus, gains,
                                 lambda r: f"{r.est / r.truth:.1f}" if np.isfinite(r.est) else "n/a")
                    if j == 0:
                        axes[i, j].set_ylabel("controller gain", fontsize=7, color=INK2)
                    if i == len(ests) - 1:
                        axes[i, j].set_xlabel("driver timescale τ_d (s)", fontsize=7, color=INK2)
            fig.suptitle(f"{spec} at h = {h} s: + correct, × wrong sign, ? inconclusive; cell text: estimate / truth",
                         fontsize=8, color=INK)
            fig.tight_layout()
            name = f"phase_{spec}_h{h}.png"
            fig.savefig(out_dir / name, dpi=170)
            plt.close(fig)
            made.append(name)
    return made


def fig_sandwich(out_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = pd.read_csv(out_dir / "sandwich.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, scale in zip(axes, ("beat", "2s")):
        sub = S[(S.scale == scale) & (S.meas == "default")]
        for k, tau in enumerate(sorted(sub.tau_d.unique())):
            g = sub[(sub.tau_d == tau) & sub.width_s.notna()].sort_values("width_s")
            ax.plot(g.width_s, g.est / g.truth, color=SERIES[k], lw=2, marker="o", ms=4,
                    label=f"τ_s = {g.tau_symp.iloc[0]:g} s")
            raw = sub[(sub.tau_d == tau) & sub.width_s.isna()]
            if len(raw):
                ax.scatter([900], raw.est / raw.truth, color=SERIES[k], marker="s", s=22, zorder=3)
        ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
        ax.axhline(0.0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("residualisation width W (s); square = no residualisation", fontsize=8, color=INK2)
        ax.set_title(f"{scale}: median static gain / planted β", fontsize=9, color=INK)
        ax.tick_params(labelsize=7, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("estimate / truth (1 = unbiased, < 0 = wrong sign)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left", title="driver footprint", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "sandwich.png", dpi=170)
    plt.close(fig)
    return "sandwich.png"


def fig_horizon(out_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Hd = pd.read_csv(out_dir / "horizon.csv")
    specs = ["remi_up_hr", "vent_up_co2"]
    meths = ["2s_naive", "2s_pretrend", "2s_kink", "2s_naive_exog"]
    antic = sorted(Hd.anticipation_s.unique())
    fig, axes = plt.subplots(len(specs), len(meths), figsize=(12, 5.4), squeeze=False)
    for i, spec in enumerate(specs):
        for j, meth in enumerate(meths):
            ax = axes[i, j]
            sub = Hd[(Hd.spec == spec) & (Hd.estimator == meth)]
            tr = sub.groupby("h").truth.first()
            ax.plot(tr.index, tr.values, color=INK, lw=2, ls="--", label="truth")
            for k, H in enumerate(antic):
                g = sub[sub.anticipation_s == H].sort_values("h")
                ax.plot(g.h, g.est, color=SERIES[k], lw=1.6, marker="o", ms=3, label=f"H={int(H)} s")
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_title(f"{spec}: {meth}", fontsize=8, color=INK)
            ax.tick_params(labelsize=7, colors=INK2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == len(specs) - 1:
                ax.set_xlabel("horizon h (s)", fontsize=7, color=INK2)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "horizon.png", dpi=170)
    plt.close(fig)
    return "horizon.png"


def exp_report(args) -> None:
    out = args.out_dir
    made = []
    for fn in (fig_phase, fig_sandwich, fig_horizon):
        try:
            r = fn(out)
            made += r if isinstance(r, list) else [r]
        except FileNotFoundError as e:
            log.warning("figure skipped: %s", e)
    summary = {}
    if (out / "phase.csv").exists():
        P = pd.read_csv(out / "phase.csv")
        P = P.drop_duplicates(subset=[c for c in ("gain", "anticipation_s", "tau_d", "family", "estimator", "spec", "h")
                                      if c in P.columns])
        ratio = P.est / P.truth
        P = P.assign(correct=P.status == "correct", wrong=P.status == "wrong",
                     within_2x=(P.status == "correct") & (ratio >= 0.5) & (ratio <= 2.0),
                     regime=np.where(P.gain > 0, "closed_loop", "open_loop"))
        tab = (P.groupby(["family", "spec", "estimator", "h", "regime"], dropna=False)
               [["correct", "wrong", "within_2x"]].mean().reset_index())
        tab.to_csv(out / "phase_summary.csv", index=False)
        summary["phase_cells"] = int(P[["gain", "anticipation_s", "tau_d"]].drop_duplicates().shape[0])
    summary["figures"] = made
    (out / "report_meta.json").write_text(json.dumps(summary, indent=2))
    log.info("report assets: %s", made)


# ---------------------------------------------------------------------------
# v2 experiments (pre-registered: v2/protocol.md)
# ---------------------------------------------------------------------------

V2_DIR = OUT_DIR / "v2"
SEED_V2 = 270000  # confirmatory seeds are SEED_V2 + offsets fixed in the protocol; --dev adds 9_000_000

# clinicians for the lemma (Lemma FI) and headline2 experiments
CLINICIANS = {
    "recorded": dict(gain=1.0, anticipation_s=0.0, w_z=0.0, w_y=1.0),
    "recorded_reversed": dict(gain=1.0, anticipation_s=0.0, w_z=0.0, w_y=-1.0),
    "recorded+field": dict(gain=1.0, anticipation_s=0.0, w_z=1.0, w_y=1.0),
    "recorded+field+plan60": dict(gain=1.0, anticipation_s=60.0, w_z=1.0, w_y=1.0),
}
LEMMA_H = (10, 30, 60, 110)  # < refractory (120 s): the treated arm has no further change up to h


def _seed(args, offset: int) -> int:
    return SEED_V2 + offset + (9_000_000 if args.dev else 0)


def _clin_cfg(name: str, drug: str) -> SimConfig:
    cfg = with_controller(SimConfig(), **CLINICIANS[name])
    if drug == "inert":
        cfg = with_physiology(cfg, remi_hr=0.0, remi_map=0.0)
    return cfg


def exp_lemma(args) -> None:
    """Lemma FI in the simulator: 1 s sufficient-statistic matching, eligible vs future-inaction
    pools, inert and active remifentanil, remi up and down, ITT truth from paired twin runs."""
    from clbench.sim.clinician_loop import decision_contrasts, itt_truth

    n = 60 if args.dev else args.n_lemma
    rows, ev_rows = [], []
    for ci, clin in enumerate(CLINICIANS):
        for di, drug in enumerate(("inert", "active")):
            cfg = _clin_cfg(clin, drug)
            seed = _seed(args, 1000 + 10 * ci + di)
            c = simulate_cohort(cfg, n, seed=seed)
            meta = cohort_meta(c)
            for key, spec in (("remi_up_hr", EventSpec("remi", "hr", +1)), ("remi_down_hr", EventSpec("remi", "hr", -1))):
                R = pd.DataFrame(decision_contrasts(c, spec, LEMMA_H))
                if drug == "active":
                    Tt = pd.DataFrame(itt_truth(cfg, n, spec, LEMMA_H, seed=seed, k_max=3))
                else:  # the drug is the only channel from the input to y: the ITT effect is exactly 0
                    Tt = pd.DataFrame(columns=["patient", "i", "k", "h", "itt_effect"])
                sus = step_truth(cfg, spec, "beat", LEMMA_H, functional="naive")
                for (pool, path, h), g in R.groupby(["pool", "path", "h"]):
                    est, lo, hi = cluster_bootstrap_mean(g.effect.values, g.patient.values, n_boot=N_BOOT)
                    if drug == "inert":
                        itt, itt_lo, itt_hi, n_pair, est_p, lo_p, hi_p = 0.0, 0.0, 0.0, len(g), est, lo, hi
                    else:
                        m = g.merge(Tt, on=["patient", "i", "h"])
                        n_pair = len(m)
                        itt, itt_lo, itt_hi = cluster_bootstrap_mean(m.itt_effect.values, m.patient.values, n_boot=N_BOOT)
                        # estimator minus truth on the paired events: bias with a patient-bootstrap CI
                        est_p, lo_p, hi_p = cluster_bootstrap_mean((m.effect - m.itt_effect).values, m.patient.values,
                                                                   n_boot=N_BOOT)
                    rows.append({
                        "clinician": clin, "drug": drug, "spec": key, "direction": spec.direction, "pool": pool,
                        "path": path, "h": h, "est": est, "lo": lo, "hi": hi, "n_events": len(g),
                        "n_patients": g.patient.nunique(), "median_match_dist": float(g.match_dist.median()),
                        "itt_truth": itt, "itt_lo": itt_lo, "itt_hi": itt_hi, "sustained_truth": sus[h],
                        "n_paired": n_pair, "bias_vs_itt": est_p if drug == "active" else est,
                        "bias_lo": lo_p if drug == "active" else lo, "bias_hi": hi_p if drug == "active" else hi,
                        "status": sign_status(lo, hi, float(np.sign(sus[h]))) if drug == "active" else
                        ("nonzero" if (lo > 0 or hi < 0) else "null"),
                        "seed": seed, **meta,
                    })
                R = R.assign(clinician=clin, drug=drug, spec=key)
                ev_rows.append(R)
            log.info("lemma %s / %s done", clin, drug)
    out = args.v2_dir
    pd.DataFrame(rows).to_csv(out / "lemma.csv", index=False)
    pd.concat(ev_rows, ignore_index=True).to_parquet(out / "lemma_events.parquet", index=False)
    # paired pool difference (future_inaction minus eligible) on the same events
    E = pd.concat(ev_rows, ignore_index=True)
    W = E.pivot_table(index=["clinician", "drug", "spec", "path", "h", "patient", "i"], columns="pool",
                      values="effect").dropna().reset_index()
    diff = []
    for (clin, drug, key, path, h), g in W.groupby(["clinician", "drug", "spec", "path", "h"]):
        d = (g["future_inaction"] - g["eligible"]).values
        est, lo, hi = cluster_bootstrap_mean(d, g.patient.values, n_boot=N_BOOT)
        sgn = -1.0 if key.endswith("down_hr") else 1.0  # coefficient space: multiply by the event direction
        diff.append({"clinician": clin, "drug": drug, "spec": key, "path": path, "h": h,
                     "fi_minus_eligible": est, "lo": lo, "hi": hi,
                     "coef_space": sgn * est, "coef_lo": min(sgn * lo, sgn * hi), "coef_hi": max(sgn * lo, sgn * hi),
                     "n_events": len(g)})
    pd.DataFrame(diff).to_csv(out / "lemma_pool_difference.csv", index=False)


def exp_headline2(args) -> None:
    """Headline event designs with the eligible (ITT) pool vs the v1 future-inaction pool."""
    n = 40 if args.dev else args.n_headline2
    hz = {"30s": (30, 60, 120), "2s": (10, 30, 60, 120), "beat": (10, 30, 60, 120)}
    methods = {"30s": ["naive", "matched"], "2s": ["naive", "pretrend", "kink", "kink_rich"],
               "beat": ["kink", "kink_rich"]}
    rows, diffs = [], []
    for ci, clin in enumerate(("recorded", "recorded+field", "recorded+field+plan60")):
        for di, drug in enumerate(("active", "inert")):
            cfg = _clin_cfg(clin, drug)
            seed = _seed(args, 2000 + 10 * ci + di)
            c = simulate_cohort(cfg, n, seed=seed)
            meta = cohort_meta(c)
            for key in ("remi_up_hr", "remi_up_map", "vent_up_co2", "vent_down_co2"):
                spec = UY_SPECS[key]
                if drug == "inert" and spec.input != "remi":
                    continue
                for scale in ("30s", "2s", "beat"):
                    o = observe(c, scale)
                    lab = exog_labels(c, o, spec.input)
                    by_pool = {}
                    for pool, itt in (("eligible", True), ("future_inaction", True), ("v1_future_inaction_clean", False)):
                        cp = "eligible" if pool == "eligible" else "future_inaction"
                        by_pool[pool] = pd.DataFrame(event_effects(o, spec, hz[scale], exog_labels=lab,
                                                                   control_pool=cp, itt=itt))
                    for h in hz[scale]:
                        for meth in methods[scale] + (["naive_exog"] if scale == "2s" else []):
                            fun = FUNCTIONAL[meth]
                            tr = 0.0 if drug == "inert" else _truth(cfg, key, scale, hz[scale], fun)[h]
                            tr_sign = float(np.sign(_truth(_clin_cfg(clin, "active"), key, scale, hz[scale], fun)[h]))
                            pools = ["eligible"] if meth in ("naive", "pretrend", "naive_exog") else list(by_pool)
                            for pool in pools:
                                R = by_pool[pool]
                                g = R[R.h == h] if len(R) else R
                                gg = g[g.exog] if meth == "naive_exog" else g
                                col = "naive" if meth == "naive_exog" else meth
                                if len(gg) == 0:
                                    est = lo = hi = np.nan
                                else:
                                    est, lo, hi = cluster_bootstrap_mean(gg[col].values, gg.patient.values, n_boot=N_BOOT)
                                rows.append({
                                    "clinician": clin, "drug": drug, "spec": key, "scale": scale,
                                    "estimator": f"{scale}_{meth}", "pool": pool if meth in ("matched", "kink", "kink_rich") else "none",
                                    "h": h, "est": est, "lo": lo, "hi": hi, "truth": tr, "functional": fun,
                                    "truth_sign_active": tr_sign,
                                    "status": sign_status(lo, hi, tr_sign),
                                    "bias": est - tr, "n_events": len(gg), "seed": seed, **meta,
                                })
                            if meth in ("matched", "kink", "kink_rich"):
                                a = by_pool["eligible"]
                                b = by_pool["future_inaction"]
                                if len(a) and len(b):
                                    m = a[a.h == h].merge(b[b.h == h], on=["patient", "i", "h"], suffixes=("_e", "_f"))
                                    d = (m[f"{meth}_f"] - m[f"{meth}_e"]).values
                                    est, lo, hi = cluster_bootstrap_mean(d, m.patient.values, n_boot=N_BOOT)
                                    diffs.append({"clinician": clin, "drug": drug, "spec": key,
                                                  "estimator": f"{scale}_{meth}", "h": h,
                                                  "fi_minus_eligible": est, "lo": lo, "hi": hi,
                                                  "coef_space": spec.direction * est, "n_events": len(m)})
                # 30 s direct-method ARX (Ljung's direct method), non-overlapping windows
                o30 = observe(c, "30s")
                y = getattr(o30, spec.target)[:, ::2]
                u = getattr(o30, spec.input)[:, ::2]
                other = o30.map[:, ::2] if spec.target == "hr" else o30.hr[:, ::2]
                size = abs(cfg.controller.remi_step if spec.input == "remi" else cfg.controller.vent_step)
                for h in hz["30s"]:
                    K = max(1, int(round(h / 30)))
                    vals = np.array([granger_step_response(y[i], u[i], [other[i]], 2, K) * size * spec.direction
                                     for i in range(c.n)])
                    est, lo, hi = bootstrap_median(vals, n_boot=N_BOOT)
                    tr_a = _truth(_clin_cfg(clin, "active"), key, "30s", hz["30s"], "naive")[h]
                    rows.append({"clinician": clin, "drug": drug, "spec": key, "scale": "30s",
                                 "estimator": "30s_granger", "pool": "none", "h": h, "est": est, "lo": lo, "hi": hi,
                                 "truth": 0.0 if drug == "inert" else np.nan, "functional": "arx_step",
                                 "truth_sign_active": float(np.sign(tr_a)), "status": sign_status(lo, hi, np.sign(tr_a)),
                                 "bias": est if drug == "inert" else np.nan, "n_events": np.nan, "seed": seed, **meta})
            log.info("headline2 %s / %s done", clin, drug)
    pd.DataFrame(rows).to_csv(args.v2_dir / "headline2.csv", index=False)
    pd.DataFrame(diffs).to_csv(args.v2_dir / "headline2_pool_difference.csv", index=False)


def exp_beta_grid(args) -> None:
    """Residualised and raw ARX for MAP→HR over β × τ_d × W (headline controller)."""
    n = 20 if args.dev else args.n_patients_v2
    betas = (-0.4, -0.2, -0.1, -0.05)
    taus = (20.0, 60.0, 200.0, 600.0, 2000.0)
    widths_s = (15, 31, 61, 121, 241, None)
    rows = []
    for bi, beta in enumerate(betas):
        for ti, tau in enumerate(taus):
            cfg = with_physiology(with_drivers(base_config(), timescale_s=tau), baro_gain=beta)
            seed = _seed(args, 3000 + 10 * bi + ti)
            c = simulate_cohort(cfg, n, seed=seed)
            truth = float(np.median(c.gains["beta"]))
            for scale in ("beat", "2s"):
                o = observe(c, scale)
                p = ARX[scale][0]
                for w in widths_s:
                    ws = None if w is None else max(3, int(round(w / o.step_s)) | 1)
                    v = np.array([arx_fit(o.hr[i], o.map[i], p, width=ws)["static"] for i in range(c.n)])
                    est, lo, hi = bootstrap_median(v, n_boot=N_BOOT)
                    fin = v[np.isfinite(v)]
                    rows.append({"beta": beta, "tau_d": tau, "tau_symp": cfg.drivers.tau_symp, "scale": scale,
                                 "width_s": np.nan if ws is None else ws * o.step_s, "est": est, "lo": lo, "hi": hi,
                                 "truth": truth, "bias": est - truth, "status": sign_status(lo, hi, -1.0),
                                 "sign_agreement": float(np.mean(fin < 0)), "n": len(fin), "seed": seed})
            log.info("beta_grid beta=%s tau=%s done", beta, tau)
    pd.DataFrame(rows).to_csv(args.v2_dir / "beta_grid.csv", index=False)


def exp_monitor_delay(args) -> None:
    """HR display latency / averaging at 2 s: residualised and raw MAP→HR (and HR→MAP)."""
    n = 20 if args.dev else args.n_patients_v2
    rows = []
    for bi, beta in enumerate((-0.4, -0.2)):
        cfg = with_physiology(base_config(), baro_gain=beta)
        seed = _seed(args, 4000 + bi)
        c = simulate_cohort(cfg, n, seed=seed)
        truth = float(np.median(c.gains["beta"]))
        kappa = float(np.median(c.gains["kappa"]))
        for avg in (2, 8):
            for d in (0, 2, 4, 6, 8, 12):
                o = observe(c, "2s", monitor_hr_s=avg, monitor_map_s=2, monitor_hr_delay_s=d)
                for wname, ws in (("raw", None), ("resid62s", 31)):
                    f = np.array([arx_fit(o.hr[i], o.map[i], 4, width=ws)["static"] for i in range(c.n)])
                    r = np.array([arx_fit(o.map[i], o.hr[i], 4, width=ws)["static"] for i in range(c.n)])
                    est, lo, hi = bootstrap_median(f, n_boot=N_BOOT)
                    rest, rlo, rhi = bootstrap_median(r, n_boot=N_BOOT)
                    rows.append({"beta": beta, "hr_avg_s": avg, "hr_delay_s": d, "fit": wname,
                                 "map_to_hr": est, "lo": lo, "hi": hi, "truth": truth,
                                 "status": sign_status(lo, hi, -1.0),
                                 "agree_neg": float(np.mean(f[np.isfinite(f)] < 0)),
                                 "hr_to_map": rest, "rev_lo": rlo, "rev_hi": rhi, "kappa": kappa,
                                 "rev_agree_pos": float(np.mean(r[np.isfinite(r)] > 0)), "seed": seed})
        log.info("monitor_delay beta=%s done", beta)
    pd.DataFrame(rows).to_csv(args.v2_dir / "monitor_delay.csv", index=False)


def exp_theory(args) -> None:
    """Deterministic numerics for Cor. 1(b) and Prop. 4 (no randomness)."""
    out: dict = {}
    # Cor 1(b): Cov(C^N_h y, D_t) for a stationary AR(1) with correlation time T on a 1 s grid,
    # C^N_h = mean(y_{t+h-1}, y_{t+h}) - mean(y_{t-P..t-1}), D_t = mean(y_{t-Wd+1..t}); P = 120, Wd = 10.
    P, Wd = 120, 10
    disp = np.arange(-(Wd - 1), 1)
    pre = np.arange(-P, 0)

    def cov(T: float, h: int) -> float:
        rho = lambda k: np.exp(-np.abs(k) / T)  # noqa: E731
        post = np.array([h - 1, h])
        return float(rho(post[:, None] - disp[None, :]).mean() - rho(pre[:, None] - disp[None, :]).mean())

    cross = {}
    for T in (5, 10, 20, 30, 60, 120, 300, 1000, 10000):
        hs = np.arange(1, 400)
        v = np.array([cov(T, h) for h in hs])
        neg = hs[v < 0]
        cross[str(T)] = {"h_star": int(neg[0]) if len(neg) else None}
    # T -> infinity: mean post lag (h - 1/2 - E[q]) equals mean |pre - display| lag
    pre_lag = float(np.abs(pre[:, None] - disp[None, :]).mean())
    cross["inf"] = {"h_star": float(pre_lag + disp.mean() + 0.5), "mean_abs_pre_minus_display_lag": pre_lag}
    out["cor1b_crossover_P120_Wd10"] = cross
    # Prop 4: leakage exponent for (1 + w^2 tau^2)^-m
    tau = 4000.0
    om = np.concatenate([np.geomspace(1e-8, 1e-2, 200001), np.linspace(1e-2, np.pi, 200001)[1:]])
    Ws = np.array([7, 11, 21, 41, 81, 161])
    rate = {}
    for m in (1, 2, 3, 4):
        spec = (1 + (om * tau) ** 2) ** (-m)
        tot = np.trapezoid(spec, om)
        vals = []
        for W in Ws:
            w = W // 2
            M = np.ones_like(om) / W + sum(2 * np.cos(j * om) / W for j in range(1, w + 1))
            vals.append(np.trapezoid((1 - M) ** 2 * spec, om) / tot)
        slope = float(np.polyfit(np.log(Ws), np.log(vals), 1)[0])
        r = min(2 * m - 1, 4)
        rate[str(m)] = {"fitted_exponent": round(slope, 3), "min(2m-1,4)": r,
                        "W*_exponents(tau_loop,tau_s)": [round(1 / (r + 1), 3), round(r / (r + 1), 3)]}
    out["prop4_leakage_rate"] = rate
    (args.v2_dir / "theory_numerics.json").write_text(json.dumps(out, indent=1))
    log.info("theory: %s", json.dumps(out)[:400])


def _fmt(est, lo, hi, nd=2) -> str:
    if not np.isfinite(est):
        return "n/a"
    return f"{est:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _prereg_lines(out: Path) -> list[str]:
    """Evaluate the pre-registered hypotheses of v2/protocol.md from the CSVs (computed, not hand-typed)."""
    from scipy.stats import spearmanr

    L: list[str] = ["\n## Pre-registered hypotheses (protocol.md): outcomes\n",
                    "| id | criterion | result | verdict |", "|---|---|---|---|"]

    def add(i, crit, res, ok):
        L.append(f"| {i} | {crit} | {res} | {'PASS' if ok else 'FAIL'} |")

    def excl_pos(r):
        return r.lo > 0

    def covers0(r):
        return r.lo <= 0 <= r.hi

    if (out / "lemma.csv").exists():
        D = pd.read_csv(out / "lemma.csv")
        Dd = pd.read_csv(out / "lemma_pool_difference.csv")
        rt = D[D.path == "routine"]
        s = rt[(rt.drug == "inert") & (rt.clinician == "recorded") & (rt.spec == "remi_up_hr")]
        e = s[s.pool == "eligible"].sort_values("h")
        add("L1", "inert, recorded, up: eligible CI covers 0 at all h",
            "; ".join(_fmt(r.est, r.lo, r.hi) for r in e.itertuples()), all(covers0(r) for r in e.itertuples()))
        f = s[(s.pool == "future_inaction") & s.h.isin([30, 60, 110])].sort_values("h")
        add("L2", "inert, recorded, up: future-inaction CI > 0 at 30/60/110 s",
            "; ".join(_fmt(r.est, r.lo, r.hi) for r in f.itertuples()), all(excl_pos(r) for r in f.itertuples()))
        d = Dd[(Dd.drug == "inert") & (Dd.path == "routine") & Dd.h.isin([30, 60, 110])]
        a = d[(d.clinician == "recorded") & (d.spec == "remi_up_hr")].sort_values("h")
        b = d[(d.clinician == "recorded_reversed") & (d.spec == "remi_up_hr")].sort_values("h")
        ok = all(r.coef_lo > 0 for r in a.itertuples()) and all(r.coef_hi < 0 for r in b.itertuples())
        add("L3", "pool difference (coef. space): recorded up CI > 0 and reversed up CI < 0 at 30/60/110 s",
            "recorded: " + "; ".join(_fmt(r.coef_space, r.coef_lo, r.coef_hi) for r in a.itertuples())
            + " / reversed: " + "; ".join(_fmt(r.coef_space, r.coef_lo, r.coef_hi) for r in b.itertuples()), ok)
        a = d[(d.clinician == "recorded") & (d.spec == "remi_down_hr")].sort_values("h")
        b = d[(d.clinician == "recorded_reversed") & (d.spec == "remi_down_hr")].sort_values("h")
        ok = all(r.coef_lo > 0 for r in a.itertuples()) and all(r.coef_hi < 0 for r in b.itertuples())
        add("L3-secondary", "same for decreases",
            "recorded: " + "; ".join(_fmt(r.coef_space, r.coef_lo, r.coef_hi) for r in a.itertuples())
            + " / reversed: " + "; ".join(_fmt(r.coef_space, r.coef_lo, r.coef_hi) for r in b.itertuples()), ok)
        s = rt[(rt.drug == "active") & (rt.clinician == "recorded") & (rt.spec == "remi_up_hr")]
        e = s[s.pool == "eligible"].sort_values("h")
        f = s[(s.pool == "future_inaction") & s.h.isin([30, 60, 110])].sort_values("h")
        ok = all(r.bias_lo <= 0 <= r.bias_hi for r in e.itertuples()) and sum(r.bias_lo > 0 for r in f.itertuples()) >= 2
        add("L4", "active, recorded, up: eligible bias-vs-ITT covers 0 at all h; FI bias > 0 at >= 2 of 30/60/110",
            "eligible: " + "; ".join(_fmt(r.bias_vs_itt, r.bias_lo, r.bias_hi) for r in e.itertuples())
            + " / FI: " + "; ".join(_fmt(r.bias_vs_itt, r.bias_lo, r.bias_hi) for r in f.itertuples()), ok)
        res, ok = [], True
        for clin in ("recorded+field", "recorded+field+plan60"):
            e = rt[(rt.drug == "inert") & (rt.clinician == clin) & (rt.spec == "remi_up_hr") & (rt.pool == "eligible")]
            dd = Dd[(Dd.drug == "inert") & (Dd.path == "routine") & (Dd.clinician == clin) & (Dd.spec == "remi_up_hr")]
            n1, n2 = int((e.lo > 0).sum()), int((dd.coef_lo > 0).sum())
            ok &= n1 >= 2 and n2 >= 2
            res.append(f"{clin}: eligible CI>0 at {n1}/4, FI−eligible CI>0 at {n2}/4")
        add("L5", "unrecorded info, inert, up: eligible CI > 0 at >= 2/4 and FI − eligible > 0 at >= 2/4", "; ".join(res), ok)
        a = D[(D.path == "alarm") & (D.drug == "inert") & (D.clinician == "recorded") & (D.spec == "remi_up_hr")
              & (D.pool == "eligible")].sort_values("h")
        add("L6 (secondary)", "alarm path, inert, recorded, up: eligible CI excludes 0 at all h",
            "; ".join(_fmt(r.est, r.lo, r.hi) for r in a.itertuples()), all(not covers0(r) for r in a.itertuples()))
    if (out / "headline2.csv").exists():
        H = pd.read_csv(out / "headline2.csv")
        g = H[(H.drug == "inert") & (H.clinician == "recorded") & (H.spec == "remi_up_hr") & (H.pool == "eligible")
              & H.estimator.isin(["2s_kink", "beat_kink"])].sort_values(["estimator", "h"])
        add("H1", "inert, recorded: 2 s and 1 s kink (eligible) CI covers 0 at every h",
            "; ".join(f"{r.estimator} h{r.h}: {_fmt(r.est, r.lo, r.hi)}" for r in g.itertuples()),
            all(covers0(r) for r in g.itertuples()))
        P = pd.read_csv(out / "headline2_pool_difference.csv")
        g = P[(P.drug == "inert") & (P.spec == "remi_up_hr") & P.estimator.isin(["2s_kink", "beat_kink"]) & P.h.isin([10, 30, 60])]
        add("H2", "inert: FI − eligible CI > 0 for 2 s and 1 s kink at 10/30/60 s, all three clinicians",
            f"{int((g.lo > 0).sum())}/{len(g)} cells with CI > 0 (min {g.fi_minus_eligible.min():+.2f})", bool((g.lo > 0).all()))
        g = H[(H.drug == "inert") & (H.spec == "remi_up_hr") & (H.estimator == "30s_granger")]
        res = {c: int(((x.lo > 0) | (x.hi < 0)).sum()) for c, x in g.groupby("clinician")}
        add("H3", "inert: 30 s direct ARX CI excludes 0 at >= 1 horizon, each clinician",
            "; ".join(f"{c}: {k}/3" for c, k in res.items()), all(k >= 1 for k in res.values()))
    if (out / "beta_grid.csv").exists():
        B = pd.read_csv(out / "beta_grid.csv")
        res = B[B.width_s.notna()]
        main = res[((res.scale == "beat") & (res.width_s == 61)) | ((res.scale == "2s") & (res.width_s == 62))]
        rng = main.groupby(["scale", "tau_d"]).bias.agg(lambda x: x.max() - x.min())
        add("B1", "bias range across β <= 0.06 in every (τ_d, scale), W = 61/62 s",
            f"max range {rng.max():.3f} ({(rng <= 0.06).sum()}/{len(rng)} cells <= 0.06)", bool((rng <= 0.06).all()))
        ok_n = tot = 0
        for _, g in res.groupby(["scale", "tau_d", "width_s"]):
            ref = g[g.beta == -0.4]
            if not len(ref):
                continue
            bh = float(ref.bias.iloc[0])
            for r in g[g.beta != -0.4].itertuples():
                pred = r.beta + bh
                if abs(pred) < 0.02:
                    continue
                tot += 1
                ok_n += int(np.sign(r.est) == np.sign(pred))
        add("B2", "sign(est) = sign(β + B̂(−0.4)) in >= 90% of non-ambiguous cells", f"{ok_n}/{tot} = {ok_n / max(tot, 1):.3f}",
            ok_n / max(tot, 1) >= 0.9)
    if (out / "monitor_delay.csv").exists():
        M = pd.read_csv(out / "monitor_delay.csv")
        m = M[(M.fit == "resid62s") & (M.hr_avg_s == 2)]
        ok = all(m[(m.beta == b) & (m.hr_delay_s == 0)].hi.iloc[0] < 0 and m[(m.beta == b) & (m.hr_delay_s == 12)].lo.iloc[0] > 0
                 for b in m.beta.unique())
        add("M1", "HR avg 2 s, residualised: CI < 0 at d = 0 and CI > 0 at d = 12 s, both β",
            "; ".join(f"β={b:g}: d0 {_fmt(g[g.hr_delay_s == 0].map_to_hr.iloc[0], g[g.hr_delay_s == 0].lo.iloc[0], g[g.hr_delay_s == 0].hi.iloc[0], 3)}, "
                      f"d12 {_fmt(g[g.hr_delay_s == 12].map_to_hr.iloc[0], g[g.hr_delay_s == 12].lo.iloc[0], g[g.hr_delay_s == 12].hi.iloc[0], 3)}"
                      for b, g in m.groupby("beta")), ok)
        rhos = {f"β={b:g}, avg {a}": spearmanr(g.hr_delay_s, g.map_to_hr)[0]
                for (b, a), g in M[M.fit == "resid62s"].groupby(["beta", "hr_avg_s"])}
        add("M2", "Spearman(d, estimate) >= 0.9 in all four sequences", "; ".join(f"{k}: {v:.2f}" for k, v in rhos.items()),
            all(v >= 0.9 for v in rhos.values()))
    return L


def fig_lemma(out: Path) -> str:
    """Lemma FI figure: inert and active drug (recorded clinician), and the policy-signed pool difference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = pd.read_csv(out / "lemma.csv")
    Dd = pd.read_csv(out / "lemma_pool_difference.csv")
    rt = D[(D.path == "routine") & (D.clinician == "recorded") & (D.spec == "remi_up_hr")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    cols = {"eligible": SERIES[0], "future_inaction": SERIES[1]}
    names = {"eligible": "eligible controls (pre-t info)", "future_inaction": "future-inaction controls"}
    for ax, drug, title in ((axes[0], "inert", "(a) inert drug (truth 0)"), (axes[1], "active", "(b) active drug")):
        sub = rt[rt.drug == drug]
        for k, (pool, g) in enumerate(sub.groupby("pool")):
            g = g.sort_values("h")
            x = g.h.values + (k - 0.5) * 4
            ax.errorbar(x, g.est, yerr=[g.est - g.lo, g.hi - g.est], color=cols[pool], marker="os"[k], ms=6, lw=2,
                        capsize=0, label=names[pool])
        if drug == "active":
            g = sub[sub.pool == "eligible"].sort_values("h")
            ax.plot(g.h, g.itt_truth, color=INK, ls="--", lw=1.5, label="ITT truth (twin runs)")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(sorted(sub.h.unique()))
        ax.set_xlabel("horizon h (s)", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=INK2)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("HR contrast after remifentanil ↑ (bpm)", fontsize=8, color=INK2)
    hdl, lab = axes[1].get_legend_handles_labels()
    axes[0].legend(hdl, lab, fontsize=7, frameon=False, loc="center", bbox_to_anchor=(0.55, 0.52))
    ax = axes[2]
    rows = [("recorded", "remi_up_hr", "standard policy, ↑"), ("recorded", "remi_down_hr", "standard policy, ↓"),
            ("recorded_reversed", "remi_up_hr", "reversed policy, ↑"),
            ("recorded_reversed", "remi_down_hr", "reversed policy, ↓"),
            ("recorded+field", "remi_up_hr", "+ field, ↑"), ("recorded+field+plan60", "remi_up_hr", "+ field + plan, ↑")]
    for j, (clin, spec, lab) in enumerate(rows):
        r = Dd[(Dd.clinician == clin) & (Dd.spec == spec) & (Dd.drug == "inert") & (Dd.path == "routine") & (Dd.h == 60)]
        if not len(r):
            continue
        r = r.iloc[0]
        y = len(rows) - 1 - j
        ax.errorbar([r.coef_space], [y], xerr=[[r.coef_space - r.coef_lo], [r.coef_hi - r.coef_space]], color=INK,
                    marker="o", ms=6, lw=2, capsize=0)
    ax.set_yticks(range(len(rows) - 1, -1, -1), [r[2] for r in rows], fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-1.0, 1.3)
    ax.set_xlabel("future-inaction − eligible at h = 60 s, × event direction", fontsize=8, color=INK2)
    ax.set_title("(c) bias sign follows the policy (inert drug)", fontsize=9, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=INK2)
    ax.tick_params(axis="y", length=0)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "lemma_fi.png", dpi=170)
    plt.close(fig)
    return "lemma_fi.png"


def exp_v2report(args) -> None:
    """Compact markdown tables from the v2 CSVs (numbers are copied, never recomputed by hand)."""
    out = args.v2_dir
    L = []
    man = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    L.append("# Closed-loop simulator v2 results\n")
    L.append(f"Protocol: `v2/protocol.md`, sha256 `{man.get('protocol_sha256', 'n/a')}`. "
             f"Code commit at run: `{man.get('git_head', 'n/a')}`. CIs: 95% patient-cluster bootstrap "
             f"({N_BOOT} draws).\n")
    if (out / "lemma.csv").exists():
        D = pd.read_csv(out / "lemma.csv")
        L.append("## Lemma FI (1 s, matching on the recorded decision statistic)\n")
        L.append("Event minus matched-control contrast of HR (bpm) per 0.5 ng/mL remifentanil step. "
                 "Inert drug: every value is bias (truth 0). Active drug: `ITT truth` from paired twin runs; "
                 "`bias` = estimate minus ITT truth on the paired events. Routine path only (overlap region); "
                 "the alarm path is in `lemma.csv`.\n")
        for drug in ("inert", "active"):
            L.append(f"\n### {drug} drug\n")
            hdr = "| clinician | event | pool | " + " | ".join(f"h={h}s" for h in LEMMA_H) + " |"
            L.append(hdr)
            L.append("|" + "---|" * (3 + len(LEMMA_H)))
            sub = D[(D.drug == drug) & (D.path == "routine")]
            for (clin, spec, pool), g in sub.groupby(["clinician", "spec", "pool"], sort=False):
                g = g.set_index("h")
                cells = []
                for h in LEMMA_H:
                    if h not in g.index:
                        cells.append("n/a")
                        continue
                    r = g.loc[h]
                    if drug == "inert":
                        cells.append(_fmt(r.est, r.lo, r.hi))
                    else:
                        cells.append(f"{_fmt(r.est, r.lo, r.hi)}; ITT {r.itt_truth:+.2f}; bias {_fmt(r.bias_vs_itt, r.bias_lo, r.bias_hi)}")
                L.append(f"| {clin} | {spec} | {pool} | " + " | ".join(cells) + " |")
    if (out / "lemma_pool_difference.csv").exists():
        Dd = pd.read_csv(out / "lemma_pool_difference.csv")
        L.append("\n### Pool difference (future-inaction minus eligible, same events), coefficient space\n")
        L.append("Coefficient space = contrast multiplied by the event direction (+1 up, −1 down). Lemma FI predicts "
                 "sign = sign of the policy's reaction to HR: + for the standard policy, − for the reversed one.\n")
        L.append("| clinician | drug | event | path | " + " | ".join(f"h={h}s" for h in LEMMA_H) + " |")
        L.append("|" + "---|" * (4 + len(LEMMA_H)))
        for (clin, drug, spec, path), g in Dd.groupby(["clinician", "drug", "spec", "path"], sort=False):
            g = g.set_index("h")
            L.append(f"| {clin} | {drug} | {spec} | {path} | " + " | ".join(
                _fmt(g.loc[h].coef_space, g.loc[h].coef_lo, g.loc[h].coef_hi) if h in g.index else "n/a"
                for h in LEMMA_H) + " |")
    if (out / "headline2.csv").exists():
        Hh = pd.read_csv(out / "headline2.csv")
        L.append("\n## Headline event designs: eligible (ITT) vs future-inaction pools\n")
        L.append("remi↑→HR, estimate [CI] (bpm); truth = sustained-step estimand of the estimator's functional "
                 "(0 for the inert drug). Pools: `eligible` (v2 default), `future_inaction` (same events), "
                 "`v1_future_inaction_clean` (v1: also drops events with a later change before t + h).\n")
        sub = Hh[(Hh.spec == "remi_up_hr")]
        hs = sorted(sub.h.unique())
        L.append("| clinician | drug | estimator | pool | " + " | ".join(f"h={h}s" for h in hs) + " |")
        L.append("|" + "---|" * (4 + len(hs)))
        for (clin, drug, est_name, pool), g in sub.groupby(["clinician", "drug", "estimator", "pool"], sort=False):
            g = g.set_index("h")
            cells = []
            for h in hs:
                if h not in g.index:
                    cells.append("")
                    continue
                r = g.loc[h]
                mark = {"correct": "R", "wrong": "X"}.get(r.status, "·") if drug == "active" else (
                    "≠0" if (r.lo > 0 or r.hi < 0) else "0")
                t = f" (truth {r.truth:+.2f})" if drug == "active" and np.isfinite(r.truth) else ""
                cells.append(f"{mark} {_fmt(r.est, r.lo, r.hi)}{t}")
            L.append(f"| {clin} | {drug} | {est_name} | {pool} | " + " | ".join(cells) + " |")
        L.append("\nR / X: CI excludes 0 on the true / opposite side; · inconclusive. For the inert drug: ≠0 = CI "
                 "excludes 0 (a false effect).\n")
        v = Hh[(Hh.spec.str.startswith("vent"))]
        if len(v):
            L.append("\n### Ventilator rate → EtCO2 (active only)\n")
            hs = sorted(v.h.unique())
            L.append("| clinician | event | estimator | pool | " + " | ".join(f"h={h}s" for h in hs) + " |")
            L.append("|" + "---|" * (4 + len(hs)))
            for (clin, spec, est_name, pool), g in v.groupby(["clinician", "spec", "estimator", "pool"], sort=False):
                g = g.set_index("h")
                L.append(f"| {clin} | {spec} | {est_name} | {pool} | " + " | ".join(
                    (f"{ {'correct': 'R', 'wrong': 'X'}.get(g.loc[h].status, '·')} {_fmt(g.loc[h].est, g.loc[h].lo, g.loc[h].hi)}"
                     if h in g.index else "") for h in hs) + " |")
    if (out / "beta_grid.csv").exists():
        B = pd.read_csv(out / "beta_grid.csv")
        L.append("\n## β grid: residualised ARX MAP→HR, bias = median estimate − median planted β\n")
        for scale, w in (("beat", 61.0), ("2s", 62.0)):
            sub = B[(B.scale == scale) & (B.width_s == w)]
            if not len(sub):
                continue
            L.append(f"\n{scale}, W = {w:g} s. Cell: bias [status].\n")
            taus = sorted(sub.tau_d.unique())
            L.append("| β | " + " | ".join(f"τ_d={t:g} (τ_s={max(1, t / 10):g})" for t in taus) + " |")
            L.append("|" + "---|" * (1 + len(taus)))
            for beta, g in sub.groupby("beta", sort=False):
                g = g.set_index("tau_d")
                L.append(f"| {beta:g} | " + " | ".join(
                    f"{g.loc[t].bias:+.3f} [{ {'correct': 'R', 'wrong': 'X'}.get(g.loc[t].status, '·')}]" for t in taus) + " |")
    if (out / "monitor_delay.csv").exists():
        M = pd.read_csv(out / "monitor_delay.csv")
        L.append("\n## HR display latency at 2 s (residualised W = 62 s)\n")
        L.append("| β | HR avg (s) | HR delay (s) | MAP→HR median [CI] | agree(<0) | HR→MAP agree(>0) |")
        L.append("|---|---|---|---|---|---|")
        for r in M[M.fit == "resid62s"].itertuples():
            L.append(f"| {r.beta:g} | {r.hr_avg_s} | {r.hr_delay_s} | {_fmt(r.map_to_hr, r.lo, r.hi, 3)} | "
                     f"{r.agree_neg:.2f} | {r.rev_agree_pos:.2f} |")
    if (out / "phase_summary.csv").exists():
        S = pd.read_csv(out / "phase_summary.csv")
        S = S[(S.regime == "closed_loop") & (S.family == "u->y")]
        L.append("\n## Phase grid re-run (protocol_amendment1.md): closed-loop cells, share wrong-signed / within 2x\n")
        L.append("`*_v1pool` = v1 future-inaction pool with clean events on the same cohorts.\n")
        for spec in ("remi_up_hr", "vent_up_co2"):
            t = S[S.spec == spec]
            hs = sorted(t.h.unique())
            L.append(f"\n{spec}\n")
            L.append("| estimator | " + " | ".join(f"wrong h={h:g}" for h in hs) + " | " + " | ".join(f"within 2x h={h:g}" for h in hs) + " |")
            L.append("|" + "---|" * (1 + 2 * len(hs)))
            for est, g in t.groupby("estimator"):
                g = g.set_index("h")
                L.append(f"| {est} | " + " | ".join(f"{g.loc[h].wrong:.2f}" for h in hs) + " | "
                         + " | ".join(f"{g.loc[h].within_2x:.2f}" for h in hs) + " |")
    if (out / "lemma.csv").exists():
        L.append(f"\nFigure: `{fig_lemma(out)}`\n")
    L += _prereg_lines(out)
    L.append("\n## Deviations and post-registration changes\n")
    L.append("- Code changed after the confirmatory run only in reporting (this report, the hypothesis table, "
             "`lemma_fi.png`), an unused-variable lint fix in `exp_theory`, and an optional exploratory "
             "`split_future` argument of `decision_contrasts` (default off; the confirmatory numbers do not use it).")
    L.append("- Exploratory, not pre-registered: `exploratory/explore_h10.py` (two extra seeds for the L4 miss at "
             "10 s) and `exploratory/explore_dir.py` → `future_direction_split.csv` (same- vs opposite-direction "
             "later actions, to explain the L3 miss).")
    L.append("- Phase-grid re-run: registered separately in `protocol_amendment1.md` before it was run.")
    L.append("- L3-secondary was evaluated at h = 30/60/110 s, the horizons of L3 (the protocol did not list them).")
    (out / "results.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s", out / "results.md")


def write_manifest(args, done: list[str], timings: dict) -> None:
    import hashlib
    import subprocess

    out = args.v2_dir
    proto = out / "protocol.md"
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["protocol_sha256"] = hashlib.sha256(proto.read_bytes()).hexdigest() if proto.exists() else None
    for am in sorted(out.glob("protocol_amendment*.md")):
        man[f"{am.stem}_sha256"] = hashlib.sha256(am.read_bytes()).hexdigest()
    try:
        man["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        man["git_dirty_files"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).split("\n")[:40]
    except Exception:  # pragma: no cover
        man["git_head"] = None
    runs = man.get("runs", {})
    for k in done:
        runs[k] = {"seconds": round(timings[k], 1), "dev": bool(args.dev),
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "n": {"lemma": args.n_lemma, "headline2": args.n_headline2}.get(k, args.n_patients_v2)}
    man["runs"] = runs
    man_path.write_text(json.dumps(man, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="all",
                   choices=["all", "attribution", "headline", "phase", "sandwich", "monitor", "horizon", "info",
                            "real", "beat_real", "pathway", "report",
                            "v2", "lemma", "headline2", "beta_grid", "monitor_delay", "theory", "v2report"])
    p.add_argument("--n-patients", type=int, default=40)
    p.add_argument("--n-headline", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="recompute the phase grid from scratch")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--v2-dir", type=Path, default=V2_DIR)
    p.add_argument("--n-lemma", type=int, default=1000)
    p.add_argument("--n-headline2", type=int, default=200)
    p.add_argument("--n-patients-v2", type=int, default=60)
    p.add_argument("--dev", action="store_true", help="small n, dev seeds (+9e6), outputs under v2/dev/")
    p.add_argument("--v1-pool", action="store_true", help="phase: also report the v1 future-inaction pool")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exps = {"attribution": exp_attribution, "headline": exp_headline, "sandwich": exp_sandwich,
            "monitor": exp_monitor, "horizon": exp_horizon, "info": exp_info, "real": exp_real,
            "beat_real": exp_beat_real, "pathway": exp_pathway,
            "phase": exp_phase, "report": exp_report}
    v2 = {"theory": exp_theory, "lemma": exp_lemma, "headline2": exp_headline2, "beta_grid": exp_beta_grid,
          "monitor_delay": exp_monitor_delay, "v2report": exp_v2report}
    if args.exp in v2 or args.exp == "v2":
        if args.dev:
            args.v2_dir = args.v2_dir / "dev"
        args.v2_dir.mkdir(parents=True, exist_ok=True)
        if not args.dev and not (V2_DIR / "protocol.md").exists():
            raise SystemExit("confirmatory v2 runs need the pre-registered v2/protocol.md")
        if not args.dev and args.v2_dir != V2_DIR:
            (args.v2_dir / "protocol.md").write_bytes((V2_DIR / "protocol.md").read_bytes())
        order = [k for k in v2 if k != "v2report"] + ["v2report"] if args.exp == "v2" else [args.exp]
        timings = {}
        for name in order:
            t0 = time.time()
            if name == "v2report":
                write_manifest(args, [k for k in order if k != "v2report"], timings)
            v2[name](args)
            timings[name] = time.time() - t0
            log.info("%s finished in %.0f s", name, timings[name])
        write_manifest(args, [k for k in order if k != "v2report"], timings)
        return
    order = list(exps) if args.exp == "all" else [args.exp]
    for name in order:
        t0 = time.time()
        exps[name](args)
        log.info("%s finished in %.0f s", name, time.time() - t0)


if __name__ == "__main__":
    main()
