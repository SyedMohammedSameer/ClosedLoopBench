"""
Simulator adapter for ClosedLoopBench: clinician-in-the-loop cohorts (``clbench.sim.clinician_loop``)
as benchmark cases, and the exact intention-to-treat (ITT) value of the benchmark's event functional by
paired counterfactual ("twin") simulation.

Mapping: simulator input ``remi`` -> action ``remifentanil`` (TCI target), ``vent`` -> ``vent_rr``; vitals
``hr``/``map``/``co2`` (deviations) + baselines 70 / 80 / 38 -> ``hr``/``map``/``etco2``, observed with the
simulator's monitor operator at 2 s or 30 s. A change the clinician logs at second t is recorded at
t + 0.5 s; the vitals respond from t + 2 s on, so the pre-t state never contains the effect.

ITT estimand of a response relation (action a, direction s, vital v, horizon h): for an eligible event
(the benchmark's definition: usable titration, no change of a in the pre window) at decision second t,
s * [Y_factual - Y_twin], Y = mean vital in the post window at t + h minus the last pre-level (the
``matched`` / ``seqtrial`` outcome), twin = the same run with that one decision vetoed and usual care
afterwards. With an inert drug the twin's vitals are identical, so the ITT is exactly 0.
"""

from __future__ import annotations

import numpy as np

from clbench.eval import closed_loop_bench as clb
from clbench.sim import clinician_loop as cl

BASELINE: dict[str, float] = {"hr": 70.0, "map": 80.0, "etco2": 38.0}
SIM_ACTIONS: dict[str, str] = {"remi": "remifentanil", "vent": "vent_rr"}
ACTION_TO_INPUT: dict[str, str] = {v: k for k, v in SIM_ACTIONS.items()}
SIM_RELATIONS: tuple[clb.Relation, ...] = tuple(r for r in clb.RELATIONS if r.action in ACTION_TO_INPUT
                                                and r.vital in ("hr", "map", "etco2"))


def sim_config(clinician: str = "recorded", drug: str = "active", w_z: float = 0.0,
               anticipation_s: float = 0.0, gain: float = 1.0, remi_emax_c50: float = 0.0,
               remi_emax_gamma: float = 1.0) -> cl.SimConfig:
    """``clinician``: "recorded" (reacts to the display, w_y = +1) or "reversed" (w_y = -1: the opposite
    policy, same physiology). ``drug``: "active" (remifentanil and ventilator act) or "inert" (both inert).
    ``w_z`` > 0 adds an unrecorded confounder (the surgical field) to the remifentanil policy.
    Extended analysis C (defaults reproduce the paper's configurations exactly): ``gain`` scales every feedback
    term of the controller (0 = open loop, exogenous changes only); ``remi_emax_c50`` > 0 replaces the linear
    remifentanil PD by a sigmoid Emax model with Hill coefficient ``remi_emax_gamma`` whose slope at the
    baseline target equals the linear coefficient (``clinician_loop.emax_deviation``)."""
    w_y = {"recorded": 1.0, "reversed": -1.0}[clinician]
    cfg = cl.with_controller(cl.SimConfig(), gain=gain, anticipation_s=anticipation_s, w_z=w_z, w_y=w_y)
    if drug == "inert":
        cfg = cl.with_physiology(cfg, remi_hr=0.0, remi_map=0.0, vent_co2=0.0)
    if remi_emax_c50 > 0:
        cfg = cl.with_physiology(cfg, remi_emax_c50=remi_emax_c50, remi_emax_gamma=remi_emax_gamma)
    return cfg


def cohort_to_cases(cohort: cl.Cohort, scale: str = "2s", prefix: str = "sim") -> list[clb.CaseData]:
    obs = cl.observe(cohort, scale)
    bcfg = clb.SCALES[scale]
    centre = (obs.t_end - 0.5) if scale == "2s" else (obs.t_end - 14.0)
    T = cohort.T
    cases = []
    for p in range(cohort.n):
        vit = {"hr": obs.hr[p] + BASELINE["hr"], "map": obs.map[p] + BASELINE["map"],
               "etco2": obs.co2[p] + BASELINE["etco2"]}
        evs = {}
        for inp, act in SIM_ACTIONS.items():
            e = cohort.events[inp]
            m = (e["patient"] == p) & (e["t"] + 1 < T)
            t = e["t"][m]
            U = getattr(cohort, inp)[p]
            evs[act] = clb.EventSet.build(t.astype(float) + 0.5, U[t], U[t + 1], clb.ACTIONS[act])
        cases.append(clb.CaseData(f"{prefix}{p:04d}", centre.astype(float), bcfg.window_sec, vit, evs))
    return cases


def eligible_events(case: clb.CaseData, action: str, direction: int, cfg: clb.ScaleConfig) -> np.ndarray:
    """Event times the benchmark uses for (action, direction): usable, clean pre window."""
    ev = case.events.get(action)
    if ev is None or len(ev.t) == 0:
        return np.zeros(0)
    clean = ev.count_in(ev.t - cfg.excl_pre_sec, ev.t, hi_open=True) == 0
    return ev.t[ev.usable & clean & (ev.direction == direction)]


def _outcome(case: clb.CaseData, cfg: clb.ScaleConfig, action: str, tau: float) -> np.ndarray:
    """(len(VITALS) * len(horizons),) benchmark outcome at event time tau."""
    t = np.array([tau])
    F = clb.st_state_features(case, cfg, action, t, np.zeros(1))
    return clb.st_outcomes(case, cfg, t, F)[0]


def itt_truth_bench(sim_cfg: cl.SimConfig, n: int, seed: int, scale: str = "2s", k_max: int = 2,
                    base: cl.Cohort | None = None):
    """Per-event ITT of the benchmark outcome for every simulator response relation (first ``k_max``
    eligible events per patient and (action, direction)). Rows: rid, patient, k, tau, h, itt (coefficient
    space: multiplied by the event direction)."""
    import pandas as pd

    cfg = clb.SCALES[scale]
    base = base if base is not None else cl.simulate_cohort(sim_cfg, n, seed=seed)
    cases_f = cohort_to_cases(base, scale)
    nH = len(cfg.horizons)
    rows = []
    for act, inp in ACTION_TO_INPUT.items():
        for s in (+1, -1):
            rels = [r for r in SIM_RELATIONS if r.action == act and r.event_dir == s and r.kind == "response"]
            if not rels:
                continue
            taus = [eligible_events(c, act, s, cfg)[:k_max] for c in cases_f]
            for k in range(k_max):
                v = np.array([int(round(tt[k] - 0.5)) if len(tt) > k else -1 for tt in taus], dtype=np.int64)
                if (v < 0).all():
                    break
                twin = cl.simulate_cohort(sim_cfg, n, seed=seed, veto={inp: v})
                cases_c = cohort_to_cases(twin, scale)
                for p in np.flatnonzero(v >= 0):
                    tau = float(v[p]) + 0.5
                    yf = _outcome(cases_f[p], cfg, act, tau)
                    yc = _outcome(cases_c[p], cfg, act, tau)
                    for r in rels:
                        vi = clb.VITALS.index(r.vital)
                        for j, h in enumerate(cfg.horizons):
                            d = s * (yf[vi * nH + j] - yc[vi * nH + j])
                            rows.append(dict(rid=r.rid, patient=int(p), k=k, tau=tau, h=float(h), itt=float(d)))
    return pd.DataFrame(rows)


def cross_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    perm = rng.permutation(n)
    fix = np.flatnonzero(perm == np.arange(n))
    for i in fix:  # rotate fixed points away
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


__all__ = ["ACTION_TO_INPUT", "BASELINE", "SIM_ACTIONS", "SIM_RELATIONS", "cohort_to_cases", "cross_derangement",
           "eligible_events", "itt_truth_bench", "sim_config"]
