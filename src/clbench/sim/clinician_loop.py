"""
Clinician-in-the-loop physiology at a 1 s step, for studying closed-loop bias in temporal edges.

Companion to ``paper/closed_loop_theory.md``. :mod:`clbench.sim.closed_loop` plants a loop and
drives it with scheduled (open-loop) interventions. Here the interventions are chosen by a simulated
clinician who watches the monitor, sees the surgical field and knows the surgical plan, so the
treatment is confounded with the physiology it acts on — the regime of the VitalDB data.

Model (1 s step, deviations from baseline; every coefficient has a known sign):

    HR_{t+1}   = a_h HR_t  + (1 − a_h)[β MAP_{t+1−L_b} + γ_h s_t + D_h (c_t − c_0)] + σ_h ε
    MAP_{t+1}  = a_m MAP_t + (1 − a_m)[κ HR_{t+1−L_k}  + γ_m s_t + D_m (c_t − c_0)] + σ_m ε
    CO2_{t+1}  = a_c CO2_t + (1 − a_c)[G_v (v_t − v_0) + G_q q_t]                   + σ_c ε
    c_{t+1}    = c_t + (1 − a_pk)(u_t − c_t)                       (effect site, 1/ke0 = τ_pk)
    s          = first-order low-pass (τ_s) of the surgical stimulation z = z_OU + episodes

β < 0 is the baroreflex (MAP → HR), κ > 0 cardiac output (HR → MAP), D_h, D_m < 0 remifentanil,
G_v < 0 the ventilator rate on EtCO2. ``s`` and the insufflation driver ``q`` are the slow latent
drivers; the clinician sees ``z`` and ``q`` (surgical field) and knows upcoming stimulation
episodes and insufflation up to ``anticipation_s`` ahead (the plan).

Controller: routine checks (Poisson), alarm-triggered checks after a reaction delay, a refractory
period per input, fixed dose steps, and a small rate of exogenous (signal-independent) changes that
play the role of the external reference signal in closed-loop identification. ``gain = 0`` gives
an open-loop cohort with exogenous changes only.

Observation operators mirror the data: 1 s ("beat"), 2 s monitor numerics (trailing averages of
configurable width, an optional HR display latency relative to MAP, sampled every 2 s) and 30 s
windows at a 15 s stride (vital signs: window median of the 2 s numerics; inputs: last value in the
window, as in :func:`clbench.data.vitaldb.window_numerics`).

Estimators (all return per-patient or per-event quantities; cohort pooling is by patient):
lagged ARX / Granger fits with optional residualisation of slow components, a within-window
contemporaneous slope, concept-onset precedence, and event studies (naive, matched, local
pre-trend, placebo-calibrated pre-trend). :func:`step_truth` gives the exact estimand of an event
estimator by applying the same functional to a noise-free, driver-free, open-loop step response.

Control pools (v2). Event-study contrasts compare decision times with *eligible non-decision times*
defined from pre-t information only (sequential-trial-emulation logic; ``control_pool="eligible"``,
the default). The v1 pool additionally excluded control times at which the clinician acted within
the outcome horizon (``control_pool="future_inaction"``); that conditions on the clinician's future
behaviour, which responds to the untreated outcome path, and is biased under closed-loop control
(Lemma FI in the theory note). :func:`decision_contrasts` is the sufficient-statistic version at
1 s (matching on the clinician's recorded decision statistic) and :func:`itt_truth` computes the
per-decision intention-to-treat estimand by paired counterfactual ("twin") simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import lfilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Physiology:
    """Fast physiology and drug / ventilator effects (units: bpm, mmHg, ng/mL, breaths/min)."""

    tau_hr: float = 2.0
    tau_map: float = 3.0
    tau_co2: float = 60.0
    baro_gain: float = -0.4  # β, bpm per mmHg (static partial effect of MAP on HR)
    baro_lag: int = 2  # HR_{t+1} responds to MAP_{t+1-baro_lag}
    output_gain: float = 0.3  # κ, mmHg per bpm
    output_lag: int = 1
    stim_hr: float = 8.0  # γ_h, bpm per unit sympathetic tone
    stim_map: float = 10.0  # γ_m
    remi_hr: float = -4.0  # D_h, bpm per ng/mL effect-site remifentanil
    remi_map: float = -5.0  # D_m
    tau_pk: float = 100.0  # 1 / ke0 in seconds (Minto ke0 ≈ 0.6 / min)
    vent_co2: float = -2.5  # G_v, mmHg per breath/min
    insuff_co2: float = 8.0  # G_q, mmHg per unit insufflation
    noise_hr: float = 1.2  # per-step innovation sd
    noise_map: float = 2.0
    noise_co2: float = 0.15
    meas_hr: float = 0.3  # measurement noise sd at 1 s
    meas_map: float = 0.5
    meas_co2: float = 0.2
    vent_osc_hr: float = 0.0  # bpm amplitude of the ventilation-frequency drive on HR (fast common driver)
    vent_osc_map: float = 0.0  # mmHg amplitude on MAP
    vent_osc_phase: float = 0.6  # radians by which the HR drive lags the MAP drive
    jitter: float = 0.2  # log-normal sd of patient-specific gain multipliers (signs kept)
    # Optional saturating (sigmoid Emax) remifentanil pharmacodynamics (extended analysis C). 0 = linear (the
    # default; every existing configuration is unchanged). With ``remi_emax_c50`` > 0 the drug term D (c − c_0)
    # becomes D [f(c) − f(c_0)] with f(c) = Emax c^γ / (C50^γ + c^γ) scaled so that f'(c_0) = 1: the linear
    # model is the first-order expansion of the Emax model at the baseline target c_0 (see ``emax_deviation``).
    remi_emax_c50: float = 0.0  # C50 in ng/mL (0: linear)
    remi_emax_gamma: float = 1.0  # Hill coefficient γ


@dataclass(frozen=True)
class Drivers:
    """Slow latent drivers. ``timescale_s`` is the single bandwidth knob swept in the phase diagram."""

    timescale_s: float = 200.0  # τ_d: OU correlation time and episode decay
    symp_ratio: float = 0.1  # τ_s = max(1, symp_ratio · τ_d)
    ou_sd: float = 0.8
    episode_rate_per_h: float = 4.0
    episode_amp_mean: float = 1.0
    insuff_rate_per_h: float = 0.75
    insuff_min_s: float = 900.0
    insuff_max_s: float = 2400.0
    co2_ou_sd: float = 0.15

    @property
    def tau_symp(self) -> float:
        return max(1.0, self.symp_ratio * self.timescale_s)

    @property
    def tau_insuff(self) -> float:
        return max(30.0, 0.5 * self.timescale_s)


@dataclass(frozen=True)
class Controller:
    """Clinician policy. ``gain`` scales every feedback term; ``anticipation_s`` is the plan horizon."""

    gain: float = 1.0
    anticipation_s: float = 0.0
    w_y: float = 1.0  # reacts to the displayed vital signs
    w_z: float = 1.0  # reacts to what she sees in the surgical field (unrecorded)
    w_a: float = 1.5  # reacts to stimulation she knows is coming within anticipation_s
    threshold: float = 1.5
    check_rate_per_s: float = 1.0 / 120.0
    alarm_level: float = 2.0  # |display signal| that triggers an alarm check
    reaction_delay_s: int = 10
    display_s: int = 10  # the display is a trailing mean over this many seconds
    refractory_s: int = 120
    exog_remi_per_h: float = 1.0  # signal-independent changes per hour
    exog_vent_per_h: float = 0.3
    remi_start: float = 3.0
    remi_step: float = 0.5
    remi_bounds: tuple[float, float] = (1.0, 8.0)
    vent_start: float = 12.0
    vent_step: float = 2.0
    vent_bounds: tuple[float, float] = (8.0, 24.0)


@dataclass(frozen=True)
class SimConfig:
    physiology: Physiology = field(default_factory=Physiology)
    drivers: Drivers = field(default_factory=Drivers)
    controller: Controller = field(default_factory=Controller)
    duration_s: int = 7200
    burn_in_s: int = 600


INPUTS = ("remi", "vent")


@dataclass
class Cohort:
    """1 s trajectories for ``n`` patients (arrays are ``(n, T)``) plus the change log."""

    config: SimConfig
    hr: np.ndarray  # latent physiology
    map: np.ndarray
    co2: np.ndarray
    hr_obs: np.ndarray  # with measurement noise
    map_obs: np.ndarray
    co2_obs: np.ndarray
    remi: np.ndarray  # TCI target
    remi_ce: np.ndarray
    vent: np.ndarray
    z: np.ndarray
    s: np.ndarray
    q: np.ndarray
    events: dict[str, np.ndarray]  # input -> structured rows (patient, t, delta, exog, alarm)
    gains: dict[str, np.ndarray]  # patient-specific β, κ, ...
    alarm_check: np.ndarray | None = None  # (n, T) bool: an alarm-triggered check happens at t

    @property
    def n(self) -> int:
        return self.hr.shape[0]

    @property
    def T(self) -> int:
        return self.hr.shape[1]


EVENT_DTYPE = np.dtype([("patient", np.int64), ("t", np.int64), ("delta", np.float64), ("exog", np.bool_),
                        ("alarm", np.bool_)])  # alarm: the decision was made at an alarm-triggered check


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _ar1(x: np.ndarray, tau: float, *, gain_one: bool = True) -> np.ndarray:
    """First-order low-pass along axis 1: y_t = a y_{t-1} + (1 − a) x_t (unit DC gain)."""
    a = float(np.exp(-1.0 / tau))
    b = [1.0 - a] if gain_one else [1.0]
    return lfilter(b, [1.0, -a], x, axis=1)


def _ou(rng: np.random.Generator, n: int, T: int, tau: float, sd: float) -> np.ndarray:
    a = float(np.exp(-1.0 / tau))
    eps = rng.normal(0.0, sd * np.sqrt(1.0 - a * a), (n, T))
    eps[:, 0] = rng.normal(0.0, sd, n)
    return lfilter([1.0], [1.0, -a], eps, axis=1)


def _drivers(rng: np.random.Generator, n: int, T: int, d: Drivers) -> dict[str, np.ndarray]:
    """Exogenous drivers: stimulation z (OU + decaying episodes), its sympathetic filter s,
    insufflation q, and the onset impulses the clinician can anticipate."""
    tau = d.timescale_s
    onsets = np.zeros((n, T))
    for i in range(n):
        k = rng.poisson(d.episode_rate_per_h * T / 3600.0)
        times = rng.integers(0, T, k)
        amps = rng.gamma(2.0, d.episode_amp_mean / 2.0, k)
        np.add.at(onsets[i], times, amps)
    episodes = lfilter([1.0], [1.0, -np.exp(-1.0 / tau)], onsets, axis=1)
    z = _ou(rng, n, T, tau, d.ou_sd) + episodes
    s = _ar1(z, d.tau_symp)

    ins_on = np.zeros((n, T), dtype=bool)
    ins_onsets = np.zeros((n, T))
    for i in range(n):
        k = rng.poisson(d.insuff_rate_per_h * T / 3600.0)
        for t0 in np.sort(rng.integers(0, T, k)):
            dur = int(rng.uniform(d.insuff_min_s, d.insuff_max_s))
            if ins_on[i, t0]:
                continue
            ins_on[i, t0 : t0 + dur] = True
            ins_onsets[i, t0] = 1.0
    q = _ar1(ins_on.astype(float), d.tau_insuff) + _ou(rng, n, T, tau, d.co2_ou_sd)
    return {"z": z, "s": s, "onsets": onsets, "q": q, "ins_on": ins_on, "ins_onsets": ins_onsets}


def emax_deviation(c: np.ndarray, c0: float, c50: float, gamma: float) -> np.ndarray:
    """Saturating drug effect in units of the linear coefficient: f(c) − f(c_0), where
    f(c) = Emax c^γ / (C50^γ + c^γ) and Emax is chosen so that f'(c_0) = 1. For small |c − c_0| this equals
    c − c_0 (the linear model); for large doses it saturates. ``c50 <= 0`` returns c − c_0 exactly."""
    c = np.asarray(c, dtype=np.float64)
    if c50 <= 0:
        return c - c0
    g = float(gamma)
    cg, c0g, kg = np.power(np.maximum(c, 0.0), g), c0**g, c50**g
    emax = (kg + c0g) ** 2 / (g * c0 ** (g - 1.0) * kg)  # f'(c_0) = 1
    return emax * (cg / (kg + cg) - c0g / (kg + c0g))


def _upcoming(impulses: np.ndarray, horizon: int) -> np.ndarray:
    """Sum of impulses in (t, t + horizon] for every t."""
    if horizon <= 0:
        return np.zeros_like(impulses)
    c = np.cumsum(impulses, axis=1)
    T = impulses.shape[1]
    ahead = np.minimum(np.arange(T) + horizon, T - 1)
    return c[:, ahead] - c


def simulate_cohort(cfg: SimConfig, n_patients: int, *, seed: int = 0,
                    veto: dict[str, np.ndarray] | None = None) -> Cohort:
    """Simulate ``n_patients`` closed-loop anaesthetics at 1 s. Burn-in is discarded.

    ``veto`` maps an input name to an int array of shape ``(n_patients,)`` of post-burn-in times
    (``-1``: none) at which any change of that input is suppressed for that patient; the clinician
    is otherwise unchanged and may act at later checks. With the same ``seed`` every random draw is
    shared, so a vetoed run is the paired counterfactual "no action at t, usual care afterwards"
    (used by :func:`itt_truth`).
    """
    rng = np.random.default_rng(seed)
    P, D, C = cfg.physiology, cfg.drivers, cfg.controller
    n, T = n_patients, cfg.burn_in_s + cfg.duration_s

    def jit(v: float) -> np.ndarray:
        return v * np.exp(rng.normal(0.0, P.jitter, n))

    beta, kappa = jit(P.baro_gain), jit(P.output_gain)
    g_h, g_m = jit(P.stim_hr), jit(P.stim_map)
    d_h, d_m = jit(P.remi_hr), jit(P.remi_map)
    g_v, g_q = jit(P.vent_co2), jit(P.insuff_co2)

    drv = _drivers(rng, n, T, D)
    z, s, q = drv["z"], drv["s"], drv["q"]
    H = int(round(C.anticipation_s))
    ant_stim = _upcoming(drv["onsets"], H)
    ant_ins = _upcoming(drv["ins_onsets"], H)
    ins_known = drv["ins_on"].astype(float)

    a_h, a_m, a_c = (np.exp(-1.0 / x) for x in (P.tau_hr, P.tau_map, P.tau_co2))
    a_pk = np.exp(-1.0 / P.tau_pk)
    eps = rng.normal(0.0, 1.0, (3, n, T))
    meas = rng.normal(0.0, 1.0, (3, n, T))
    routine = rng.random((n, T)) < C.check_rate_per_s
    exog_p = {"remi": C.exog_remi_per_h / 3600.0, "vent": C.exog_vent_per_h / 3600.0}
    exog_draw = {k: rng.random((n, T)) < exog_p[k] for k in INPUTS}
    exog_sign = {k: np.where(rng.random((n, T)) < 0.5, -1.0, 1.0) for k in INPUTS}

    veto_abs = {k: np.full(n, -1, dtype=np.int64) for k in INPUTS}
    for k, v in (veto or {}).items():
        v = np.asarray(v, dtype=np.int64)
        veto_abs[k] = np.where(v >= 0, v + cfg.burn_in_s, -1)
    alarm_rec = np.zeros((n, T), dtype=bool)

    hr = np.zeros((n, T))
    mp = np.zeros((n, T))
    co2 = np.zeros((n, T))
    hr_o = np.zeros((n, T))
    mp_o = np.zeros((n, T))
    co2_o = np.zeros((n, T))
    remi = np.zeros((n, T))
    ce = np.zeros((n, T))
    vent = np.zeros((n, T))

    u_r = np.full(n, C.remi_start)
    c_r = np.full(n, C.remi_start)
    u_v = np.full(n, C.vent_start)
    last = {k: np.full(n, -(10**9)) for k in INPUTS}
    pending = np.full(n, -1)
    logs: dict[str, list[tuple]] = {k: [] for k in INPUTS}
    Lb, Lk, Wd = P.baro_lag, P.output_lag, C.display_s
    osc_on = P.vent_osc_hr != 0.0 or P.vent_osc_map != 0.0
    vent_phase = rng.uniform(0.0, 2.0 * np.pi, n)

    for t in range(T):
        remi[:, t], ce[:, t], vent[:, t] = u_r, c_r, u_v
        hr_o[:, t] = hr[:, t] + P.meas_hr * meas[0, :, t]
        mp_o[:, t] = mp[:, t] + P.meas_map * meas[1, :, t]
        co2_o[:, t] = co2[:, t] + P.meas_co2 * meas[2, :, t]

        # ---- clinician -------------------------------------------------------------------
        lo = max(0, t - Wd + 1)
        disp_h = hr_o[:, lo : t + 1].mean(axis=1)
        disp_m = mp_o[:, lo : t + 1].mean(axis=1)
        disp_c = co2_o[:, lo : t + 1].mean(axis=1)
        y_r = 0.5 * (disp_h / 8.0 + disp_m / 10.0)
        y_v = disp_c / 4.0
        alarm = ((np.abs(y_r) > C.alarm_level) | (np.abs(y_v) > C.alarm_level)) & (pending < t)
        pending = np.where(alarm, t + C.reaction_delay_s, pending)
        is_alarm = pending == t
        alarm_rec[:, t] = is_alarm
        check = routine[:, t] | is_alarm

        sig = {
            "remi": C.gain * (C.w_y * y_r + C.w_z * z[:, t] + C.w_a * ant_stim[:, t]),
            "vent": C.gain * (C.w_y * y_v + C.w_z * ins_known[:, t] + C.w_a * ant_ins[:, t]),
        }
        cur = {"remi": u_r, "vent": u_v}
        step = {"remi": C.remi_step, "vent": C.vent_step}
        bounds = {"remi": C.remi_bounds, "vent": C.vent_bounds}
        for k in INPUTS:
            free = (t - last[k]) >= C.refractory_s
            delta = np.zeros(n)
            if C.gain > 0:
                can = check & free
                delta = np.where(can & (sig[k] > C.threshold), step[k], delta)
                delta = np.where(can & (sig[k] < -C.threshold), -step[k], delta)
            ex = exog_draw[k][:, t] & free & (delta == 0)
            delta = np.where(ex, step[k] * exog_sign[k][:, t], delta)
            vetoed = veto_abs[k] == t
            if vetoed.any():
                delta = np.where(vetoed, 0.0, delta)
                ex = ex & ~vetoed
            lo_b, hi_b = bounds[k]
            new = np.clip(cur[k] + delta, lo_b, hi_b)
            changed = new != cur[k]
            if changed.any():
                idx = np.flatnonzero(changed)
                logs[k].append((idx, t, (new - cur[k])[idx], ex[idx], (is_alarm & ~routine[:, t])[idx]))
                last[k] = np.where(changed, t, last[k])
                cur[k][:] = new
        # (cur dict aliases u_r / u_v, so they are updated in place)

        if t == T - 1:
            break
        # ---- physiology -------------------------------------------------------------------
        m_b = mp[:, t + 1 - Lb] if t + 1 - Lb >= 0 else 0.0
        h_k = hr[:, t + 1 - Lk] if t + 1 - Lk >= 0 else 0.0
        if P.remi_emax_c50 > 0:  # saturating PD (extended analysis C); the default branch below is unchanged
            dc = emax_deviation(c_r, C.remi_start, P.remi_emax_c50, P.remi_emax_gamma)
        else:
            dc = c_r - C.remi_start
        hr[:, t + 1] = a_h * hr[:, t] + (1 - a_h) * (beta * m_b + g_h * s[:, t] + d_h * dc) + P.noise_hr * eps[0, :, t]
        mp[:, t + 1] = a_m * mp[:, t] + (1 - a_m) * (kappa * h_k + g_m * s[:, t] + d_m * dc) + P.noise_map * eps[1, :, t]
        co2[:, t + 1] = (
            a_c * co2[:, t]
            + (1 - a_c) * (g_v * (u_v - C.vent_start) + g_q * q[:, t])
            + P.noise_co2 * eps[2, :, t]
        )
        c_r = c_r + (1 - a_pk) * (u_r - c_r)
        if osc_on:
            vent_phase = vent_phase + 2.0 * np.pi * u_v / 60.0  # one cycle per breath at the set rate
            hr[:, t + 1] += (1 - a_h) * P.vent_osc_hr * np.sin(vent_phase - P.vent_osc_phase) / (1 - a_h)
            mp[:, t + 1] += (1 - a_m) * P.vent_osc_map * np.sin(vent_phase) / (1 - a_m)

    b = cfg.burn_in_s
    events: dict[str, np.ndarray] = {}
    for k in INPUTS:
        rows = [
            (int(i), t - b, float(dl), bool(e), bool(al) and not bool(e))
            for idx, t, dls, exs, als in logs[k]
            if t >= b
            for i, dl, e, al in zip(idx, dls, exs, als)
        ]
        events[k] = np.array(rows, dtype=EVENT_DTYPE)

    cut = slice(b, None)
    return Cohort(
        config=cfg,
        hr=hr[:, cut], map=mp[:, cut], co2=co2[:, cut],
        hr_obs=hr_o[:, cut], map_obs=mp_o[:, cut], co2_obs=co2_o[:, cut],
        remi=remi[:, cut], remi_ce=ce[:, cut], vent=vent[:, cut],
        z=z[:, cut], s=s[:, cut], q=q[:, cut],
        events=events,
        gains={"beta": beta, "kappa": kappa, "remi_hr": d_h, "remi_map": d_m, "vent_co2": g_v,
               "stim_hr": g_h, "stim_map": g_m},
        alarm_check=alarm_rec[:, cut],
    )


# ---------------------------------------------------------------------------
# Observation operators
# ---------------------------------------------------------------------------


def trailing_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Trailing moving average along axis 1 (shorter at the start)."""
    if width <= 1:
        return x.copy()
    c = np.cumsum(x, axis=1)
    out = c.copy()
    out[:, width:] = c[:, width:] - c[:, :-width]
    counts = np.minimum(np.arange(1, x.shape[1] + 1), width)
    return out / counts


@dataclass
class Observed:
    """Series at one observation scale; ``step_s`` seconds between samples; ``t_end`` the time (s,
    1 s grid) each sample becomes available."""

    step_s: float
    t_end: np.ndarray
    hr: np.ndarray
    map: np.ndarray
    co2: np.ndarray
    remi: np.ndarray
    vent: np.ndarray


def delay(x: np.ndarray, d: int) -> np.ndarray:
    """Pure delay of ``d`` samples along axis 1 (the first value is held at the start)."""
    if d <= 0:
        return x
    return np.concatenate([np.repeat(x[:, :1], d, axis=1), x[:, :-d]], axis=1)


def observe(cohort: Cohort, scale: str, *, monitor_hr_s: int = 2, monitor_map_s: int = 2,
            monitor_hr_delay_s: int = 0) -> Observed:
    """
    ``scale`` is ``"beat"`` (1 s), ``"2s"`` (monitor numerics: trailing means of ``monitor_*_s``
    seconds sampled every 2 s, inputs as set at the sample time) or ``"30s"`` (medians of the 2 s
    numerics over 30 s windows at a 15 s stride; inputs: value at the window end).

    ``monitor_hr_delay_s`` delays the displayed HR relative to MAP (monitor HR is computed from a
    multi-beat ECG/pleth average and is published later than the arterial-pressure numerics). The
    delay moves part of the HR → MAP (κ > 0) path into the lagged MAP → HR coefficient and is the
    candidate explanation for the positive real-data 2 s MAP→HR bias (theory note §5, §9.1).
    """
    if scale == "beat":
        t = np.arange(cohort.T)
        return Observed(1.0, t, cohort.hr_obs, cohort.map_obs, cohort.co2_obs, cohort.remi, cohort.vent)
    mon = {
        "hr": delay(trailing_mean(cohort.hr_obs, monitor_hr_s), int(monitor_hr_delay_s)),
        "map": trailing_mean(cohort.map_obs, monitor_map_s),
        "co2": trailing_mean(cohort.co2_obs, 2),
    }
    t2 = np.arange(1, cohort.T, 2)
    fine = {k: v[:, t2] for k, v in mon.items()}
    if scale == "2s":
        return Observed(2.0, t2, fine["hr"], fine["map"], fine["co2"], cohort.remi[:, t2], cohort.vent[:, t2])
    if scale != "30s":
        raise ValueError(f"unknown scale {scale!r}")
    starts = np.arange(0, cohort.T - 30 + 1, 15)
    idx_2s = [np.flatnonzero((t2 >= a) & (t2 < a + 30)) for a in starts]
    med = {k: np.stack([np.median(v[:, ix], axis=1) for ix in idx_2s], axis=1) for k, v in fine.items()}
    t_end = starts + 29
    return Observed(15.0, t_end, med["hr"], med["map"], med["co2"], cohort.remi[:, t_end], cohort.vent[:, t_end])


# ---------------------------------------------------------------------------
# y → y estimators
# ---------------------------------------------------------------------------


def residualize(x: np.ndarray, width: int | None, *, min_frac: float = 0.8) -> np.ndarray:
    """
    Subtract a centred moving average of ``width`` samples (forced odd) along the last axis.

    NaN-aware: the average uses the finite samples in the window and is NaN when fewer than
    ``min_frac`` of them are finite; edges are NaN. ``None`` / ``0`` subtracts the series mean only.
    This is the prefilter L_W = 1 − M_W of the theory note.
    """
    x = np.asarray(x, dtype=np.float64)
    if not width:
        return x - np.nanmean(x, axis=-1, keepdims=True)
    width = int(width) | 1
    half = width // 2
    fin = np.isfinite(x)
    zero = np.zeros(x.shape[:-1] + (1,))
    cs = np.cumsum(np.concatenate([zero, np.where(fin, x, 0.0)], axis=-1), axis=-1)
    cn = np.cumsum(np.concatenate([zero, fin.astype(np.float64)], axis=-1), axis=-1)
    tot = cs[..., width:] - cs[..., :-width]
    cnt = cn[..., width:] - cn[..., :-width]
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(cnt >= min_frac * width, tot / np.maximum(cnt, 1.0), np.nan)
    out = np.full_like(x, np.nan)
    L = x.shape[-1]
    if L > 2 * half:
        out[..., half : L - half] = x[..., half : L - half] - ma
    return out


def arx_fit(y: np.ndarray, x: np.ndarray, p: int, *, width: int | None = None) -> dict[str, float]:
    """
    OLS of y_t on (y_{t-1..t-p}, x_{t-1..t-p}) for one series, after residualising both series by a
    centred moving average of ``width`` samples (``None``: demean only).

    Returns the summed cross coefficient ``coupling`` = Σ b_j, the own persistence Σ a_j and the
    implied static partial gain Σ b_j / (1 − Σ a_j).
    """
    yr, xr = residualize(y, width), residualize(x, width)
    n = len(yr)
    cols = [yr[p - j : n - j] for j in range(1, p + 1)] + [xr[p - j : n - j] for j in range(1, p + 1)]
    Z = np.column_stack(cols)
    target = yr[p:]
    ok = np.isfinite(target) & np.isfinite(Z).all(axis=1)
    if ok.sum() < 5 * Z.shape[1]:
        return {"coupling": np.nan, "persistence": np.nan, "static": np.nan, "n": int(ok.sum())}
    coef, *_ = np.linalg.lstsq(Z[ok], target[ok], rcond=None)
    a, b = coef[:p].sum(), coef[p:].sum()
    return {"coupling": float(b), "persistence": float(a), "static": float(b / (1.0 - a)) if a < 0.999 else np.nan,
            "n": int(ok.sum())}


def contemporaneous_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Within-patient OLS slope of y on x at the same time (the window-level association)."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 10 or np.var(x[ok]) == 0:
        return np.nan
    return float(np.cov(y[ok], x[ok])[0, 1] / np.var(x[ok], ddof=1))


def onset_precedence(y: np.ndarray, x: np.ndarray, *, lag_steps: int = 8, z_thr: float = 1.0) -> float:
    """
    Concept-transition statistic in the style of v1 temporal edges: after onsets of "x high"
    (x crossing +z_thr sd), the excess rate of "y low" onsets minus the excess rate of "y high"
    onsets within ``lag_steps`` samples, each relative to its base rate. Negative-coupling
    physiology (β < 0) predicts a positive value.
    """
    def zs(v: np.ndarray) -> np.ndarray:
        return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)

    zx, zy = zs(x), zs(y)
    def onsets(flag: np.ndarray) -> np.ndarray:
        return np.flatnonzero(flag[1:] & ~flag[:-1]) + 1

    ox = onsets(zx > z_thr)
    oy_low, oy_high = onsets(zy < -z_thr), onsets(zy > z_thr)
    n = len(y)
    if len(ox) == 0:
        return np.nan

    def excess(oy: np.ndarray) -> float:
        if len(oy) == 0:
            return 0.0
        mark = np.zeros(n, dtype=bool)
        mark[oy] = True
        c = np.concatenate([[0], np.cumsum(mark)])
        hits = np.array([c[min(n, i + lag_steps + 1)] - c[i + 1] for i in ox]) > 0
        base = 1.0 - (1.0 - len(oy) / n) ** lag_steps
        return float(hits.mean() - base)

    return excess(oy_low) - excess(oy_high)


# ---------------------------------------------------------------------------
# u → y event estimators
# ---------------------------------------------------------------------------


@dataclass
class EventSpec:
    input: str  # "remi" | "vent"
    target: str  # "hr" | "map" | "co2"
    direction: int  # +1 increases, -1 decreases


def _series(obs: Observed, name: str) -> np.ndarray:
    return getattr(obs, name)


def _pretrend(y: np.ndarray, npre: int) -> tuple[np.ndarray, np.ndarray]:
    """Intercept at t = 0 and slope (per sample) of an OLS line through y[i-npre:i], for every i
    (NaN where the window is incomplete). Time runs −npre..−1."""
    n = len(y)
    tt = np.arange(-npre, 0, dtype=np.float64)
    tbar, stt = tt.mean(), ((tt - tt.mean()) ** 2).sum()
    a = np.full(n, np.nan)
    b = np.full(n, np.nan)
    if n <= npre:
        return a, b
    from numpy.lib.stride_tricks import sliding_window_view

    W = sliding_window_view(y, npre)[:-1]  # W[k] = y[k:k+npre] -> window ending before i = k + npre
    ybar = W.mean(axis=1)
    slope = ((W - ybar[:, None]) * (tt - tbar)).sum(axis=1) / stt
    a[npre:] = ybar - slope * tbar
    b[npre:] = slope
    return a, b


def event_effects(
    obs: Observed,
    spec: EventSpec,
    horizons_s: tuple[int, ...],
    *,
    exog_labels: np.ndarray | None = None,
    pre_s: int = 60,
    naive_pre_s: int = 120,
    n_placebo: int = 5,
    placebo_every: int = 5,
    recent_s: int = 10,
    control_pool: str = "eligible",
    itt: bool = True,
) -> list[dict]:
    """
    Per-event effects of changes of ``spec.input`` on ``spec.target`` at the scale of ``obs``.

    For each event i (first sample whose input value differs from the previous sample, in the
    requested direction, with no change of the same input in the pre-window or up to the horizon)
    and horizon h (K = h / step samples):

    * ``naive``    mean(y[i+K-1 : i+K+1]) − mean(y[i−P_naive : i])
    * ``matched``  (event change) − (change at the nearest placebo time), both measured as
                   mean(y[i+K-1 : i+K+1]) − mean of the last 3 pre samples, matching on pre-level,
                   pre-trend and current dose (the matched_quick design)
    * ``pretrend`` mean(y[i+K-2 : i+K+1]) − (a + b·K), (a, b) the OLS line over the pre-window
    * ``kink``     pretrend deviation minus the mean pretrend deviation of the ``n_placebo`` placebo
                   times nearest in (a, b) (the kink_placebo design)
    * ``kink_rich`` the same, matching on the recorded information set: (a, b) plus the mean of
                   every recorded vital sign over the last ``recent_s`` seconds (what the monitor
                   displays) and the current dose

    Control ("placebo") times. ``control_pool="eligible"`` (default): times with no change of the
    input in the pre-window and none at the time itself, i.e. eligible non-decision times defined
    from pre-t information only (sequential trial emulation). ``"future_inaction"`` (v1, kept as a
    sensitivity analysis): additionally no change up to the largest horizon after the control time.
    That pool conditions on the clinician's future inaction and is biased under closed-loop control
    (Lemma FI of the theory note), with a sign set by the policy.

    ``itt=True`` (default) keeps every event whatever happens after it (intention to treat: the
    decision at t, followed by usual care); ``itt=False`` is the v1 "clean event" selection that
    drops events with another change of the input before t + h (a selection on post-treatment
    decisions). In the simulator the refractory period makes the two identical for h < 120 s.
    ``exog_labels`` (bool, shape of the input series) marks exogenous changes in the simulator.
    """
    if control_pool not in ("eligible", "future_inaction"):
        raise ValueError(f"unknown control_pool {control_pool!r}")
    step = obs.step_s
    npre = max(3, int(round(pre_s / step)))
    nnaive = max(3, int(round(naive_pre_s / step)))
    Ks = [max(1, int(round(h / step))) for h in horizons_s]
    kmax = max(Ks) + 1
    U, Y = _series(obs, spec.input), _series(obs, spec.target)
    lo = max(npre, nnaive)
    nrec = max(1, int(round(recent_s / step)))
    vitals = [obs.hr, obs.map, obs.co2]
    rows: list[dict] = []
    for p in range(U.shape[0]):
        u, y = U[p], Y[p]
        n = len(y)
        du = np.diff(u, prepend=u[0])
        changes = np.flatnonzero(du != 0)
        ev = changes[np.sign(du[changes]) == spec.direction]
        ev = ev[(ev >= lo) & (ev < n - kmax - 1)]
        if len(ev) == 0:
            continue
        pos = np.searchsorted(changes, ev)
        prev = np.where(pos > 0, changes[np.maximum(pos - 1, 0)], -(10**9))
        nxt = np.where(pos + 1 < len(changes), changes[np.minimum(pos + 1, len(changes) - 1)], 10**9)
        ev, nxt = ev[prev < ev - npre], nxt[prev < ev - npre]
        if len(ev) == 0:
            continue

        a_all, b_all = _pretrend(y, npre)
        cs = np.concatenate([[0.0], np.cumsum(y)])

        def wmean(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # mean of y[a:b]
            return (cs[b] - cs[a]) / (b - a)

        bad = np.zeros(n + 1, dtype=np.int64)
        # future_inaction: no change within [i - npre - 1, i + kmax]  <=>  i not in [j - kmax, j + npre + 1]
        # eligible:        no change within [i - npre - 1, i]         <=>  i not in [j,        j + npre + 1]
        post_excl = kmax if control_pool == "future_inaction" else 0
        for j in changes:
            bad[max(0, j - post_excl)] += 1
            bad[min(n, j + npre + 2)] -= 1
        cand = np.flatnonzero(np.cumsum(bad)[:n] == 0)
        cand = cand[(cand >= lo) & (cand < n - kmax - 1)][::placebo_every]
        have_pl = len(cand) >= max(n_placebo, 1)

        if have_pl:
            fc = np.column_stack([a_all[cand], b_all[cand]])
            fe = np.column_stack([a_all[ev], b_all[ev]])
            sd = fc.std(axis=0) + 1e-9
            d2 = ((((fe[:, None, :] - fc[None, :, :]) / sd) ** 2).sum(axis=2))
            nn = np.argpartition(d2, min(n_placebo, len(cand)) - 1, axis=1)[:, :n_placebo]

            def mfeat(ix: np.ndarray) -> np.ndarray:
                return np.column_stack([
                    wmean(ix - nnaive, ix),
                    wmean(ix - 3, ix) - wmean(ix - nnaive, ix - nnaive + 3),
                    u[ix - 1],
                ])

            mc, me = mfeat(cand), mfeat(ev)
            msd = mc.std(axis=0) + 1e-9
            mj = cand[np.argmin(((((me[:, None, :] - mc[None, :, :]) / msd) ** 2).sum(axis=2)), axis=1)]

            recs = []
            for v in vitals:
                cv = np.concatenate([[0.0], np.cumsum(v[p])])
                recs.append(lambda ix, cv=cv: (cv[ix] - cv[ix - nrec]) / nrec)

            def rfeat(ix: np.ndarray) -> np.ndarray:
                return np.column_stack([a_all[ix], b_all[ix]] + [r(ix) for r in recs] + [u[ix - 1]])

            rc, re_ = rfeat(cand), rfeat(ev)
            rsd = rc.std(axis=0) + 1e-9
            d2r = (((re_[:, None, :] - rc[None, :, :]) / rsd) ** 2).sum(axis=2)
            nn_r = np.argpartition(d2r, min(n_placebo, len(cand)) - 1, axis=1)[:, :n_placebo]

        for h, K in zip(horizons_s, Ks):
            ok = np.ones(len(ev), dtype=bool) if itt else nxt > ev + K + 1
            if not ok.any():
                continue
            e = ev[ok]

            def post2(ix: np.ndarray) -> np.ndarray:
                return wmean(ix + K - 1, ix + K + 1)

            def dev(ix: np.ndarray) -> np.ndarray:
                p3 = wmean(ix + K - 2, ix + K + 1) if K >= 2 else post2(ix)
                return p3 - (a_all[ix] + b_all[ix] * K)

            naive = post2(e) - wmean(e - nnaive, e)
            pre = dev(e)
            if have_pl:
                dev_c = dev(cand)
                kink = pre - dev_c[nn[ok]].mean(axis=1)
                kink_rich = pre - dev_c[nn_r[ok]].mean(axis=1)
                mjk = mj[ok]
                matched = (post2(e) - wmean(e - 3, e)) - (post2(mjk) - wmean(mjk - 3, mjk))
            else:
                kink = np.full(len(e), np.nan)
                kink_rich = np.full(len(e), np.nan)
                matched = np.full(len(e), np.nan)
            ex = exog_labels[p, e] if exog_labels is not None else np.zeros(len(e), dtype=bool)
            for k in range(len(e)):
                rows.append({
                    "patient": p, "i": int(e[k]), "h": int(h), "delta": float(du[e[k]]), "exog": bool(ex[k]),
                    "naive": float(naive[k]), "matched": float(matched[k]),
                    "pretrend": float(pre[k]), "kink": float(kink[k]), "kink_rich": float(kink_rich[k]),
                })
    return rows


def exog_labels(cohort: Cohort, obs: Observed, input_name: str) -> np.ndarray:
    """Bool array on the grid of ``obs``: True at samples where an exogenous change of the input
    first becomes visible."""
    lab = np.zeros(getattr(obs, input_name).shape, dtype=bool)
    ev = cohort.events[input_name]
    if len(ev) == 0:
        return lab
    # a decision at second t is first recorded at t + 1: visible at the first sample with t_end >= t + 1
    pos = np.searchsorted(obs.t_end, ev["t"] + 1, side="left")
    ok = pos < lab.shape[1]
    lab[ev["patient"][ok & ev["exog"]], pos[ok & ev["exog"]]] = True
    return lab


def granger_step_response(y: np.ndarray, u: np.ndarray, others: list[np.ndarray], p: int, K: int) -> float:
    """
    Fit y_t on p lags of y, u and each series in ``others`` (with intercept) and return the
    response of y after K samples to a unit step in u held from time 0, all else fixed.
    """
    n = len(y)
    cols = [y[p - j : n - j] for j in range(1, p + 1)] + [u[p - j : n - j] for j in range(1, p + 1)]
    for o in others:
        cols += [o[p - j : n - j] for j in range(1, p + 1)]
    Z = np.column_stack(cols + [np.ones(n - p)])
    target = y[p:]
    ok = np.isfinite(target) & np.isfinite(Z).all(axis=1)
    if ok.sum() < 5 * Z.shape[1] or np.var(u[p:][ok]) == 0:
        return np.nan
    coef, *_ = np.linalg.lstsq(Z[ok], target[ok], rcond=None)
    a, b = coef[:p], coef[p : 2 * p]
    resp = np.zeros(K + p + 1)
    ustep = np.zeros(K + p + 1)
    ustep[p:] = 1.0
    for t in range(p, K + p + 1):
        resp[t] = sum(a[j - 1] * resp[t - j] for j in range(1, p + 1)) + sum(
            b[j - 1] * ustep[t - j] for j in range(1, p + 1)
        )
    return float(resp[K + p])


# ---------------------------------------------------------------------------
# Ground truth for event estimators
# ---------------------------------------------------------------------------


def step_truth(
    cfg: SimConfig,
    spec: EventSpec,
    scale: str,
    horizons_s: tuple[int, ...],
    *,
    monitor_hr_s: int = 2,
    monitor_map_s: int = 2,
    n_phase: int = 15,
    functional: str = "pretrend",
) -> dict[int, float]:
    """
    Estimand of the event estimators at ``scale``: the functional applied to the noise-free,
    driver-free, open-loop response to one sustained step of the size the controller uses, averaged
    over the step's phase relative to the sampling grid.

    ``functional="pretrend"`` (3-sample post mean minus the extrapolated pre-trend) is the estimand
    of ``pretrend``, ``kink`` and ``kink_rich``; ``functional="naive"`` (2-sample post mean minus the
    pre-mean) is the estimand of ``naive``, ``matched`` and ``naive_exog``. They agree to within 3% at
    1 s and 2 s but differ by up to 1.5x at 30 s (15 s stride). Uses the cohort-level (unjittered)
    gains. This is the *sustained-step* estimand Γ_C(h); the ITT estimand of a decision under usual
    care afterwards is :func:`itt_truth`.
    """
    if functional not in ("pretrend", "naive"):
        raise ValueError(f"unknown functional {functional!r}")
    P, C = cfg.physiology, cfg.controller
    size = C.remi_step if spec.input == "remi" else C.vent_step
    size *= spec.direction
    T = 1200
    out = {h: [] for h in horizons_s}
    for phase in range(n_phase):
        t0 = 500 + phase
        u = np.zeros(T)
        u[t0:] = size
        hr, mp, co2, ce = (np.zeros(T) for _ in range(4))
        a_h, a_m, a_c = (np.exp(-1.0 / x) for x in (P.tau_hr, P.tau_map, P.tau_co2))
        a_pk = np.exp(-1.0 / P.tau_pk)
        for t in range(T - 1):
            m_b = mp[t + 1 - P.baro_lag] if t + 1 - P.baro_lag >= 0 else 0.0
            h_k = hr[t + 1 - P.output_lag] if t + 1 - P.output_lag >= 0 else 0.0
            # the simulator records a decision at t + 1 and acts on it from that step on
            dr = ce[t] if spec.input == "remi" else 0.0
            dv = u[t + 1] if spec.input == "vent" else 0.0
            hr[t + 1] = a_h * hr[t] + (1 - a_h) * (P.baro_gain * m_b + P.remi_hr * dr)
            mp[t + 1] = a_m * mp[t] + (1 - a_m) * (P.output_gain * h_k + P.remi_map * dr)
            co2[t + 1] = a_c * co2[t] + (1 - a_c) * P.vent_co2 * dv
            ce[t + 1] = ce[t] + (1 - a_pk) * (u[t + 1] - ce[t])
        fake = Cohort(
            config=cfg, hr=hr[None], map=mp[None], co2=co2[None], hr_obs=hr[None], map_obs=mp[None],
            co2_obs=co2[None], remi=(u if spec.input == "remi" else np.zeros(T))[None], remi_ce=ce[None],
            vent=(u if spec.input == "vent" else np.zeros(T))[None], z=np.zeros((1, T)), s=np.zeros((1, T)),
            q=np.zeros((1, T)), events={k: np.zeros(0, EVENT_DTYPE) for k in INPUTS}, gains={},
        )
        obs = observe(fake, scale, monitor_hr_s=monitor_hr_s, monitor_map_s=monitor_map_s)
        rows = event_effects(obs, spec, horizons_s, n_placebo=10**9)  # no placebo needed
        for r in rows:
            out[r["h"]].append(r[functional])
    return {h: float(np.mean(v)) if v else np.nan for h, v in out.items()}


# ---------------------------------------------------------------------------
# Sufficient-statistic decision design and the ITT estimand (Lemma FI)
# ---------------------------------------------------------------------------


def decision_statistic(cohort: Cohort, input_name: str) -> np.ndarray:
    """
    The recorded part of the clinician's decision signal at 1 s, as in :func:`simulate_cohort`:
    the 10 s display means of the *recorded* vitals, combined as the controller combines them
    (remifentanil: 0.5 (HR/8 + MAP/10); ventilator: EtCO2/4). A function of pre-t records only.
    """
    C = cohort.config.controller
    dh, dm, dc = (trailing_mean(x, C.display_s) for x in (cohort.hr_obs, cohort.map_obs, cohort.co2_obs))
    return 0.5 * (dh / 8.0 + dm / 10.0) if input_name == "remi" else dc / 4.0


def _contrast_1s(y: np.ndarray, i: np.ndarray, h: int, pre_s: int) -> np.ndarray:
    """Post mean over [i+h-1, i+h] minus pre mean over [i-pre_s, i-1] on the 1 s grid."""
    cs = np.concatenate([[0.0], np.cumsum(y)])
    return (cs[i + h + 1] - cs[i + h - 1]) / 2.0 - (cs[i] - cs[i - pre_s]) / pre_s


def decision_contrasts(
    cohort: Cohort,
    spec: EventSpec,
    horizons_s: tuple[int, ...],
    *,
    pre_s: int = 60,
    n_match: int = 5,
    every: int = 3,
    target_obs: str = "obs",
    split_future: bool = False,
) -> list[dict]:
    """
    Event-minus-matched-control contrasts at 1 s, matching on the clinician's recorded decision
    statistic (:func:`decision_statistic`), for two control pools:

    * ``eligible``: eligible non-decision times, from pre-t information only. Eligible means the
      input can be changed at t: no change in the previous ``refractory_s`` seconds, the dose is
      not at the bound in the event direction, and t is not an alarm-triggered check (the alarm
      path is a deterministic function of the display history, so it has no overlap; events on
      that path are reported separately as ``path="alarm"`` and matched to the same pool).
    * ``future_inaction``: the same, and additionally no change of the input in (t, t + h_max].

    Each event gets the mean contrast of its ``n_match`` nearest same-patient controls in the
    decision statistic. The contrast is the 2 s post mean at t + h minus the ``pre_s`` pre mean, on
    the recorded (``target_obs="obs"``) or latent (``"latent"``) target series. Rows carry the event
    identity so they can be paired with :func:`itt_truth`.

    ``split_future=True`` (exploratory) adds the pools ``fi_same`` / ``fi_opposite`` that exclude
    only controls followed by a change in the same / opposite direction as the event within h_max
    (the one-sided cases of Lemma FI(c)).
    """
    C = cohort.config.controller
    D = decision_statistic(cohort, spec.input)
    U = getattr(cohort, spec.input)
    Y = getattr(cohort, f"{spec.target}_obs" if target_obs == "obs" else spec.target)
    lo_b, hi_b = C.remi_bounds if spec.input == "remi" else C.vent_bounds
    hmax = max(horizons_s)
    lo = max(pre_s, C.refractory_s) + 1
    alarm = cohort.alarm_check if cohort.alarm_check is not None else np.zeros_like(U, dtype=bool)
    rows: list[dict] = []
    for p in range(cohort.n):
        u, y, d = U[p], Y[p], D[p]
        T = len(u)
        du = np.diff(u, prepend=u[0])
        ch = np.flatnonzero(du != 0)
        last = np.full(T, -(10**9))
        last[ch] = ch
        last = np.maximum.accumulate(last)
        prev = np.concatenate([[-(10**9)], last[:-1]])  # last change strictly before i
        nxt = np.full(T + 1, 10**9)
        nxt[ch] = ch
        nxt = np.minimum.accumulate(nxt[::-1])[::-1][1:]  # next change strictly after i
        idx = np.arange(lo, T - hmax - 2)
        t = idx - 1  # decision second; the change is recorded at i = t + 1
        free = (t - prev[idx]) >= C.refractory_s
        room = (u[idx - 1] < hi_b) if spec.direction > 0 else (u[idx - 1] > lo_b)
        is_ev = (np.sign(du[idx]) == spec.direction) & (du[idx] != 0)
        base = free & room & ~alarm[p, t]
        ctrl = base & (du[idx] == 0)
        ctrl_idx = idx[ctrl][::every]
        fut_ok = nxt[ctrl_idx] > ctrl_idx + hmax + 1
        pools = [("eligible", ctrl_idx), ("future_inaction", ctrl_idx[fut_ok])]
        if split_future:
            chd = np.sign(du[ch])
            same = np.zeros(len(ctrl_idx), dtype=bool)
            opp = np.zeros(len(ctrl_idx), dtype=bool)
            for j, c0 in enumerate(ctrl_idx):
                k0, k1 = np.searchsorted(ch, [c0 + 1, c0 + hmax + 2])
                dd = chd[k0:k1]
                same[j] = np.any(dd == spec.direction)
                opp[j] = np.any(dd == -spec.direction)
            pools += [("fi_same", ctrl_idx[~same]), ("fi_opposite", ctrl_idx[~opp])]
        ev_idx = idx[is_ev & free & room]
        ev_alarm = alarm[p, ev_idx - 1]
        if len(ev_idx) == 0 or len(ctrl_idx) < n_match:
            continue
        for pool, cand in pools:
            if len(cand) < n_match:
                continue
            xs = d[cand - 1]
            order = np.argsort(xs)
            xs_s, cand_s = xs[order], cand[order]
            k = np.searchsorted(xs_s, d[ev_idx - 1])
            nn = np.empty((len(ev_idx), n_match), dtype=np.int64)
            dist = np.empty(len(ev_idx))
            for j, kk in enumerate(k):
                a, b = max(0, kk - n_match), min(len(xs_s), kk + n_match)
                win = np.arange(a, b)
                sel = win[np.argsort(np.abs(xs_s[win] - d[ev_idx[j] - 1]))[:n_match]]
                nn[j] = cand_s[sel]
                dist[j] = np.abs(xs_s[sel] - d[ev_idx[j] - 1]).max()
            for h in horizons_s:
                ce = _contrast_1s(y, ev_idx, h, pre_s)
                cc = _contrast_1s(y, nn.ravel(), h, pre_s).reshape(nn.shape).mean(axis=1)
                for j in range(len(ev_idx)):
                    rows.append({"patient": p, "i": int(ev_idx[j]), "h": int(h), "pool": pool,
                                 "path": "alarm" if ev_alarm[j] else "routine",
                                 "event": float(ce[j]), "control": float(cc[j]),
                                 "effect": float(ce[j] - cc[j]), "match_dist": float(dist[j])})
    return rows


def itt_truth(
    cfg: SimConfig,
    n_patients: int,
    spec: EventSpec,
    horizons_s: tuple[int, ...],
    *,
    seed: int,
    k_max: int = 3,
    pre_s: int = 60,
    target_obs: str = "obs",
) -> list[dict]:
    """
    Per-decision intention-to-treat effect by paired counterfactual simulation. For each patient
    and k = 0..k_max-1, take the k-th change of the input in the event direction (as recorded,
    at i = t + 1), re-simulate with the same seed but that single decision vetoed, and report the
    difference of the event contrast (same functional as :func:`decision_contrasts`) between the
    factual and the vetoed run. Everything before t is identical in the pair, so the difference is
    the effect of "act at t, usual care afterwards" versus "do not act at t, usual care afterwards"
    at exactly that state. Rows carry (patient, i) to pair with :func:`decision_contrasts`.
    """
    base = simulate_cohort(cfg, n_patients, seed=seed)
    U = getattr(base, spec.input)
    ser = f"{spec.target}_obs" if target_obs == "obs" else spec.target
    hmax = max(horizons_s)
    rows: list[dict] = []
    ev_lists = []
    for p in range(n_patients):
        du = np.diff(U[p], prepend=U[p][0])
        e = np.flatnonzero((np.sign(du) == spec.direction) & (du != 0))
        e = e[(e > pre_s + 1) & (e < base.T - hmax - 2)]
        ev_lists.append(e)
    for k in range(k_max):
        v = np.array([ev[k] - 1 if len(ev) > k else -1 for ev in ev_lists], dtype=np.int64)
        if (v < 0).all():
            break
        twin = simulate_cohort(cfg, n_patients, seed=seed, veto={spec.input: v})
        Yf, Yc = getattr(base, ser), getattr(twin, ser)
        for p in np.flatnonzero(v >= 0):
            i = np.array([v[p] + 1])
            assert getattr(twin, spec.input)[p, i[0]] == getattr(twin, spec.input)[p, i[0] - 1]
            for h in horizons_s:
                eff = _contrast_1s(Yf[p], i, h, pre_s)[0] - _contrast_1s(Yc[p], i, h, pre_s)[0]
                rows.append({"patient": int(p), "i": int(i[0]), "k": k, "h": int(h), "itt_effect": float(eff)})
    return rows



# ---------------------------------------------------------------------------
# Cohort statistics
# ---------------------------------------------------------------------------


def cluster_bootstrap_mean(values: np.ndarray, groups: np.ndarray, *, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile 95% CI from resampling ``groups`` (patients)."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    values, groups = values[ok], np.asarray(groups)[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(ug))
    cnts = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(ug), (n_boot, len(ug)))
    bs = sums[draw].sum(axis=1) / np.maximum(cnts[draw].sum(axis=1), 1.0)
    return float(values.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def bootstrap_median(values: np.ndarray, *, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Median over patients with a patient-bootstrap percentile 95% CI."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.median(v[rng.integers(0, len(v), (n_boot, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def sign_status(lo: float, hi: float, truth_sign: float) -> str:
    """'correct' / 'wrong' when the CI excludes zero on the true / opposite side, else 'inconclusive'."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if lo > 0 or hi < 0:
        return "correct" if np.sign(lo if lo > 0 else hi) == np.sign(truth_sign) else "wrong"
    return "inconclusive"


def with_controller(cfg: SimConfig, **kw) -> SimConfig:
    return replace(cfg, controller=replace(cfg.controller, **kw))


def with_drivers(cfg: SimConfig, **kw) -> SimConfig:
    return replace(cfg, drivers=replace(cfg.drivers, **kw))


def with_physiology(cfg: SimConfig, **kw) -> SimConfig:
    return replace(cfg, physiology=replace(cfg.physiology, **kw))
