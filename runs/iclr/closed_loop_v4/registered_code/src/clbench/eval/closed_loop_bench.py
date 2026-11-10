"""
Closed-loop measurement benchmark on real perioperative data (VitalDB).

Question
--------
Anaesthesia is a closed loop: the clinician watches the vitals and moves the drugs and the ventilator,
and the drugs and the ventilator move the vitals. A temporal / causal edge learned from such data can
therefore reflect the *response* (action -> vital, pharmacology/physiology) or the *policy*
(vital -> action, the clinician's control law), and a method that cannot tell them apart can report a
textbook relation with the wrong sign. This module scores eight estimators against a table of relations
whose sign is known from physiology, pharmacology or clinical practice:

    (a) naive        event-triggered pre/post difference
    (b) xcorr_v1     v1-style lagged cross-correlation (``clbench.edges.temporal``), signed
    (c) granger      pairwise VAR(p) in levels: sum of cause-lag coefficients (per patient; pooled FE)
    (d) varx         pooled VAR in differences of (MAP, HR, EtCO2) with exogenous action inputs
                     (up / down increments, distributed lags) -> step response; policy equations for
                     the action increments given vital trends
    (e) pcmciplus    PCMCI+ (tigramite, ParCorr) per patient on [actions, MAP, HR, EtCO2]
    (f) varlingam    VAR-LiNGAM, pooled: FE VAR + DirectLiNGAM on pooled VAR residuals
    (g) matched      matched-placebo event design (match on level, 2-min trend, current dose)
    (h) kink         placebo-calibrated local pre-trend extrapolation ("kink") design

and runs every estimator under two negative controls: action times drawn at random within the case
(``nc_random``) and another patient's action stream time-aligned to this patient (``nc_cross``).

Sign conventions
----------------
Every relation has a *cause* and an *effect*. Response: cause = action dose, effect = vital.
Policy: cause = vital, effect = action dose. ``Relation.expected`` is the sign of d(effect)/d(cause).
All estimators report in this "coefficient space": event designs multiply the observed vital change
by the event direction (+1 increase, -1 decrease), so a correct estimate always has sign ``expected``.
For policy relations event designs use the vital's change over the ~2 min *before* the action.

Units differ by method (mmHg, correlation, regression coefficient); only signs and CIs are compared.
All CIs are case-level bootstrap (resampling patients with replacement; one shared weight matrix).

v2 (``BenchOptions()``; ``BenchOptions.v1()`` reproduces the Wave-2 benchmark exactly)
-------------------------------------------------------------------------------------
 1. Control (placebo) pools use only pre-t information: eligible times (no change of the action in the
    pre window) that are untreated at t (no change within the decision interval +-grid/2). v1 also
    excluded anchors followed by a change up to t + max(h) -- conditioning on the clinician's *future*
    inaction, which selects anchors whose vital did not go on to trigger an action (sequential-trial
    emulation forbids this). v1 pool kept as ``*_v1`` rows.
 2. Event designs are intention-to-treat (every titration with a clean pre window); v1's selection of
    events without a same-action change after t (post-treatment) is the ``*_clean`` sensitivity.
 2b. Matched contrasts use Abadie-Imbens regression bias correction (``*_nobc``: without).
 3. Co-interventions: ``*_coint`` excludes events and controls with any other action change within
    +-60 s; ``*_coadj`` adds other-action up/down counts in +-60 s to the matching features.
 4. Null arms: nc_cross (schedule-preserving) is the primary null for level-based methods; nc_random
    (timing-only) is reported alongside; nc_cross2 (a second derangement, VARX only) is the null input
    for the typed estimator's false-admission rate.
 5. xcorr: ``xcorr_dir`` takes the peak only over lags where the relation's cause leads;
    ``xcorr_pos`` is v1's positive-peak rule; ``xcorr_v1`` is the Wave-2 signed |peak| (both directions).
 6. PCMCI+: the same case subsample for every arm; lags cover 240 s (30 s grid, tau_max 8 at 30 s;
    10 s block means, tau_max 24 at 2 s).
 7. Dose series are NaN after an action's recorded track stops (v1 forward-filled to case end).
 8. Null discovery rates are also counted over *distinct* estimates (direction-agnostic methods give one
    estimate for up and down rows).
 9. ``naive_lvl``: post minus the last 30 s pre-level (v1 ``naive`` uses the 2-min pre-mean).
10. ``kink_prealt``: kink with the alternative pre-trend window (60 s at 30 s; 180 s at 2 s).
Plus the CLBench typed estimator ``typed_pooled`` (``clbench.edges.typed_temporal_pooled``) and the
patient-specificity split-half test.

v3 = ClosedLoopBench 1.0 (``BenchOptions()``; ``BenchOptions.v2()`` reproduces v2, ``.v1()`` Wave 2)
-----------------------------------------------------------------------------------------------------
 1. Null calibration against K = 8 cross-patient derangements (``CROSS_ARMS``, salts ``CROSS_SALTS``;
    draws 1-2 are v2's nc_cross / nc_cross2). The calibrated estimate is real minus the mean of the K
    null estimates; its SE adds the between-draw variance of that mean to the paired case-bootstrap
    variance, and an edge is admitted only if the single-draw calibrated CI excludes 0 with the same
    sign in >= 75% of the draws (``typed_temporal_pooled.calibrated_test_multi`` / ``admit_multi``).
    The same admission rule is applied to every method (the leaderboard).
 2. ``seqtrial``: pooled sequential-trial contrast (scored as a method; it did NOT pass the pre-stated
    simulator gate, runs/iclr/closed_loop_v3/sim_validation/, so it is not the typed edge's response base).
    Events = intention-to-treat titrations with a clean
    pre window. Controls = eligible, untreated-at-t decision times (pre-t information only) of *other*
    patients, matched (k = 10 NN) on the pre-t state: last 30 s level, last 10 s level (2 s scale only)
    and 2-min trend of MAP, HR and EtCO2, the current dose, elapsed case time, and other actions' up /
    down changes in [t - 60 s, t); regression bias correction fitted on the pooled control bank. The within-patient matched pools keep a
    policy-dependent bias under feedback (simulator, runs/iclr/sim_closed_loop/verify/v3/v3_pooled.csv).
    Simulator validation (v3): at 2 s the pooled contrast has the fewest spurious inert admissions of the
    event designs and no wrong-signed active admission; at 30 s, where the clinician's recent (10 s) state
    is not observable, it is worse than the within-patient pre-trend design.
 3. Co-intervention sensitivities (``*_coint``, ``*_coadj``) count other actions' changes in
    [t - 60 s, t) only (v2: [t - 60, t + 60], which conditions on post-treatment co-interventions).
 4. Reproducible bootstraps: per-case outputs are consumed in submission order (``imap``) and every
    per-case list is sorted by case before it is stacked or resampled.
 5. Code hashes in meta.json (runner).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 1. Actions, vitals and the ground-truth relation table
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

VITALS: tuple[str, ...] = ("map", "hr", "etco2")


@dataclass(frozen=True)
class ActionSpec:
    key: str
    agent: str  # ``agent`` column of data/processed_vitaldb/events
    unit: str
    # TCI targets and the ventilator: a change from 0 (induction / start of IPPV) or to 0 (stop) is not a
    # titration step and is excluded from the event set (kept as a nuisance input in the VARX).
    exclude_zero_endpoints: bool


ACTIONS: dict[str, ActionSpec] = {
    "vent_rr": ActionSpec("vent_rr", "ventilator", "breaths/min", True),
    "propofol": ActionSpec("propofol", "propofol", "ug/mL (TCI Ct)", True),
    "remifentanil": ActionSpec("remifentanil", "remifentanil", "ng/mL (TCI Ct)", True),
    "phenylephrine": ActionSpec("phenylephrine", "phenylephrine", "mL/h", False),
    "norepinephrine": ActionSpec("norepinephrine", "norepinephrine", "mL/h", False),
    "nitroglycerin": ActionSpec("nitroglycerin", "nitroglycerin", "mL/h", False),
}
AGENT_TO_ACTION: dict[str, str] = {a.agent: k for k, a in ACTIONS.items()}
PRESSORS: tuple[str, ...] = ("phenylephrine", "norepinephrine", "nitroglycerin")


@dataclass(frozen=True)
class Relation:
    rid: str
    kind: str  # "response" (action -> vital) | "policy" (vital -> action)
    action: str
    vital: str
    event_dir: int  # +1: the relation is about increases of the action; -1: decreases
    expected: int  # sign of d(effect)/d(cause)
    horizon_s: float  # response: primary horizon after the action; policy: trend window before it
    label: str
    latency: str = ""
    # v4: event designs and seqtrial use only events (and control anchors) whose last 30 s pre-level of the
    # relation's vital is < pre_max (FiO2 -> SpO2: the oximeter ceiling). Regression methods are unrestricted.
    pre_max: float = math.inf
    domain: str = ""  # v4 interpretive axis: gas_exchange | cardiopulmonary | drug | drug_derived | policy

    @property
    def cause(self) -> str:
        return self.action if self.kind == "response" else self.vital

    @property
    def effect(self) -> str:
        return self.vital if self.kind == "response" else self.action

    @property
    def expected_delta(self) -> int:
        """Expected sign of the raw vital change after (response) / before (policy) the event."""
        return self.expected * self.event_dir


# Physiology / pharmacology / practice basis (citations kept here, next to the table they justify).
#
# Ventilator rate -> EtCO2.  Alveolar ventilation equation, PaCO2 ~ VCO2 / VA (West, Respiratory
#   Physiology: The Essentials; Nunn's Applied Respiratory Physiology). At fixed tidal volume a higher
#   set rate raises VA and lowers alveolar / end-tidal CO2. EtCO2 moves within a few breaths and then
#   approaches the new steady state over several minutes because of the body CO2 stores
#   (Taskar et al., Chest 1995, "Dynamics of carbon dioxide elimination following ventilator
#   resetting"). Primary horizon 60 s.
# Propofol -> MAP.  Dose-dependent fall in arterial pressure from reduced systemic vascular resistance
#   and sympathetic tone. Effect-site equilibration: Schnider et al., Anesthesiology 1998/1999
#   (ke0 ~0.46/min, t1/2ke0 ~1.5 min); the blood-pressure effect equilibrates more slowly than the EEG
#   effect (Kazama et al., Anesthesiology 1999). A TCI target increase gives a plasma bolus, so MAP
#   starts to fall within ~1-2 min. Primary horizon 120 s.
# Remifentanil -> HR, MAP.  mu-opioid: dose-dependent vagally mediated bradycardia and hypotension
#   (Glass, Gan & Howell, Anesth Analg 1999, review of remifentanil PK/PD). Effect-site
#   t1/2ke0 ~1-1.5 min (Minto et al., Anesthesiology 1997). Primary horizon 120 s.
# Phenylephrine -> MAP up, HR down.  Pure alpha-1 agonist raises SVR and MAP; the baroreflex lowers HR
#   (the basis of the modified Oxford baroreflex method; Smyth, Sleight & Pickering, Circ Res 1969).
#   Onset < 1 min i.v.; an infusion-rate change approaches steady state over minutes. Primary 120 s.
# Norepinephrine -> MAP up.  alpha-1 (+ beta-1) agonist (Goodman & Gilman, adrenergic agonists). HR is
#   ambiguous (beta-1 vs reflex), so only MAP is scored. Primary 120 s.
# Nitroglycerin -> MAP down.  Venodilator (preload reduction), onset 1-2 min i.v. (Goodman & Gilman).
#   Primary 120 s.
# Policy (clinical practice, not physiology):
#   * rising MAP / HR during surgery is read as nociception and treated by deepening analgesia, i.e.
#     a remifentanil target increase (Guignard, Best Pract Res Clin Anaesthesiol 2006, on haemodynamic
#     signs as the de-facto nociception monitor);
#   * falling MAP / hypotension is treated with a vasopressor (phenylephrine, norepinephrine) and by
#     lightening the hypnotic (propofol target decrease) (Sessler et al., BJA 2019, POQI consensus on
#     intraoperative blood pressure; Walsh et al., Anesthesiology 2013, harm below MAP 55-65 mmHg);
#   * rising EtCO2 under controlled ventilation is corrected by raising the set rate (normocapnia
#     target; Miller's Anesthesia, intraoperative ventilation).
RELATIONS: tuple[Relation, ...] = (
    Relation("R01", "response", "vent_rr", "etco2", +1, -1, 60, "vent RR up -> EtCO2 down", "breaths-min"),
    Relation("R02", "response", "vent_rr", "etco2", -1, -1, 60, "vent RR down -> EtCO2 up", "breaths-min"),
    Relation("R03", "response", "propofol", "map", +1, -1, 120, "propofol Ct up -> MAP down", "1-5 min"),
    Relation("R04", "response", "propofol", "map", -1, -1, 120, "propofol Ct down -> MAP up", "1-5 min"),
    Relation("R05", "response", "remifentanil", "hr", +1, -1, 120, "remi Ct up -> HR down", "1-3 min"),
    Relation("R06", "response", "remifentanil", "map", +1, -1, 120, "remi Ct up -> MAP down", "1-3 min"),
    Relation("R07", "response", "remifentanil", "hr", -1, -1, 120, "remi Ct down -> HR up", "1-3 min"),
    Relation("R08", "response", "remifentanil", "map", -1, -1, 120, "remi Ct down -> MAP up", "1-3 min"),
    Relation("R09", "response", "phenylephrine", "map", +1, +1, 120, "phenylephrine up -> MAP up", "<1-3 min"),
    Relation("R10", "response", "phenylephrine", "hr", +1, -1, 120, "phenylephrine up -> HR down (reflex)", "<1-3 min"),
    Relation("R11", "response", "norepinephrine", "map", +1, +1, 120, "norepinephrine up -> MAP up", "<1-3 min"),
    Relation("R12", "response", "nitroglycerin", "map", +1, -1, 120, "nitroglycerin up -> MAP down", "1-2 min"),
    Relation("P01", "policy", "remifentanil", "map", +1, +1, 120, "MAP rising -> remi Ct up"),
    Relation("P02", "policy", "remifentanil", "hr", +1, +1, 120, "HR rising -> remi Ct up"),
    Relation("P03", "policy", "phenylephrine", "map", +1, -1, 120, "MAP falling -> phenylephrine up"),
    Relation("P04", "policy", "propofol", "map", -1, +1, 120, "MAP falling -> propofol Ct down"),
    Relation("P05", "policy", "vent_rr", "etco2", +1, +1, 120, "EtCO2 rising -> vent RR up"),
    Relation("P06", "policy", "norepinephrine", "map", +1, -1, 120, "MAP falling -> norepinephrine up"),
)
REL_BY_ID: dict[str, Relation] = {r.rid: r for r in RELATIONS}
# Bivariate (action, vital) systems shared by the direction-agnostic methods (xcorr, granger, pcmci,
# varlingam): response and policy relations on the same pair read the two directions of one fit.
PAIRS: tuple[tuple[str, str], ...] = tuple(dict.fromkeys((r.action, r.vital) for r in RELATIONS))

# ── v4 (ClosedLoopBench 1.1): physics / physiology relations, full-cohort 2 s grid ──────────────────
# The v4 table is the v3 table plus the relations below; ``use_table("v4")`` switches the module-level
# table (VITALS, ACTIONS, RELATIONS and everything derived from them); ``use_table("v3")`` restores v3
# exactly (the default at import). New actions are appended AFTER the v3 actions, so the per-action
# cross-patient derangements of the v3 actions (seeded, consumed in ACTIONS order) are unchanged.
#
# Sources for the new signs (full references in ``clbench.bench.ground_truth.SOURCES``):
# Tidal volume -> EtCO2.  Alveolar ventilation VA = RR x (VT - VD); at fixed rate a larger set VT raises VA
#   and lowers alveolar / end-tidal CO2 (West; Lumb, Nunn's Applied Respiratory Physiology). Same kinetics as
#   the rate change (Taskar et al., Chest 1995). Primary horizon 60 s, both directions.
# FiO2 -> SpO2.  Raising the inspired O2 fraction raises alveolar PO2 (alveolar gas equation) and, below the
#   flat part of the oxyhaemoglobin dissociation curve, SaO2 / SpO2 (West; Lumb). Near 100 % there is a
#   ceiling, so the event designs use only events with a pre-level SpO2 < 97 % (``pre_max``). Nitrogen
#   wash-out of the FRC (time constant FRC / VA, ~0.5-1 min) plus circulation and oximeter averaging:
#   primary horizon 120 s.
# PEEP -> MAP.  Raised intrathoracic pressure lowers venous return and cardiac output (Cournand et al., Am J
#   Physiol 1948; Luecke & Pelosi, Crit Care 2005). Modest at the PEEP steps used in theatre; within breaths:
#   primary horizon 60 s.
# Volatile agent (MAC) -> MAP, BIS.  Sevoflurane / desflurane lower SVR and MAP dose-dependently (Ebert et
#   al., Anesthesiology 1995) and deepen the hypnotic EEG effect (Katoh et al., Anesthesiology 1998, BIS vs
#   sevoflurane concentration). MAC events are DERIVED steps of the measured end-tidal MAC (the vaporiser
#   dial is not recorded; ``clbench.bench.native_grid``), so these relations are ``drug_derived``.
#   Primary horizon 120 s.
# Propofol -> BIS.  Effect-site model fitted to the EEG / BIS (Schnider et al., Anesthesiology 1999, ke0 ~0.46
#   /min). Primary horizon 120 s.
# Policy: falling SpO2 is treated by raising FiO2 (Miller's Anesthesia, intraoperative hypoxaemia); rising
#   EtCO2 by raising minute ventilation (set TV as well as set rate; Miller); rising BIS (lightening) by
#   deepening the hypnotic, propofol target or volatile (Punjasawadwong et al., Cochrane 2014, BIS-guided
#   titration).
RELATIONS_V4_EXTRA: tuple[Relation, ...] = (
    Relation("R13", "response", "vent_tv", "etco2", +1, -1, 60, "set TV up -> EtCO2 down", "breaths-min",
             domain="gas_exchange"),
    Relation("R14", "response", "vent_tv", "etco2", -1, -1, 60, "set TV down -> EtCO2 up", "breaths-min",
             domain="gas_exchange"),
    Relation("R15", "response", "fio2", "spo2", +1, +1, 120, "set FiO2 up -> SpO2 up (pre SpO2 < 97)", "0.5-2 min",
             pre_max=97.0, domain="gas_exchange"),
    Relation("R16", "response", "peep", "map", +1, -1, 60, "PEEP up -> MAP down", "breaths", domain="cardiopulmonary"),
    Relation("R17", "response", "mac", "map", +1, -1, 120, "MAC up -> MAP down (derived MAC steps)", "1-5 min",
             domain="drug_derived"),
    Relation("R18", "response", "mac", "bis", +1, -1, 120, "MAC up -> BIS down (derived MAC steps)", "1-5 min",
             domain="drug_derived"),
    Relation("R19", "response", "propofol", "bis", +1, -1, 120, "propofol Ct up -> BIS down", "1-3 min", domain="drug"),
    Relation("P07", "policy", "fio2", "spo2", +1, -1, 120, "SpO2 falling -> FiO2 up", domain="policy"),
    Relation("P08", "policy", "vent_tv", "etco2", +1, +1, 120, "EtCO2 rising -> set TV up", domain="policy"),
    Relation("P09", "policy", "propofol", "bis", +1, +1, 120, "BIS rising -> propofol Ct up", domain="policy"),
    Relation("P10", "policy", "mac", "bis", +1, +1, 120, "BIS rising -> MAC up (derived MAC steps)", domain="policy"),
)
# domain of the v3 relations on the v4 interpretive axis (the v3 Relation objects are left untouched)
DOMAIN_V3: dict[str, str] = {"R01": "gas_exchange", "R02": "gas_exchange", **{f"R{i:02d}": "drug" for i in range(3, 13)},
                             **{f"P0{i}": "policy" for i in range(1, 7)}}
VITALS_V4: tuple[str, ...] = ("map", "hr", "etco2", "spo2", "bis")
ACTIONS_V4_EXTRA: dict[str, ActionSpec] = {
    "vent_tv": ActionSpec("vent_tv", "vent_tv", "L (set TV)", True),
    "peep": ActionSpec("peep", "peep", "cmH2O (set PEEP)", False),
    "fio2": ActionSpec("fio2", "fio2", "% (set FiO2)", False),
    "mac": ActionSpec("mac", "mac", "MAC (derived steps of measured end-tidal MAC)", True),
}
LINGAM_SETS_V4_EXTRA: dict[str, tuple[str, ...]] = {
    "vent": ("vent_tv", "fio2", "peep", "etco2", "spo2"),
    "peepmap": ("peep", "map", "hr"),
    "depth": ("mac", "propofol", "bis", "map"),
}
TABLE_VERSION: str = "v3"
XCORR_STORE_DTYPE = np.float32  # v4: float16 (memory: 23 pairs x 10 arms x 3,442 cases of 1,801-lag curves)


def relation_domain(r: Relation) -> str:
    return r.domain or DOMAIN_V3.get(r.rid, "")


def use_table(version: str = "v3") -> None:
    """Switch the module-level relation table. "v3": the 18-relation ClosedLoopBench 1.0 table (default);
    "v4": + 11 relations, vitals + SpO2 and BIS, actions + set TV, PEEP, FiO2 and derived MAC steps, VARX
    per-equation designs (missing vitals do not drop rows of the other equations)."""
    g = globals()
    if version not in ("v3", "v4"):
        raise ValueError(version)
    if "_V3_TABLE" not in g:
        g["_V3_TABLE"] = dict(VITALS=VITALS, ACTIONS=dict(ACTIONS), RELATIONS=RELATIONS, LINGAM_SETS=dict(LINGAM_SETS))
    base = g["_V3_TABLE"]
    if version == "v3":
        vit, act, rel, ls = base["VITALS"], dict(base["ACTIONS"]), base["RELATIONS"], dict(base["LINGAM_SETS"])
    else:
        vit = VITALS_V4
        act = {**base["ACTIONS"], **ACTIONS_V4_EXTRA}
        rel = base["RELATIONS"] + RELATIONS_V4_EXTRA
        ls = {**base["LINGAM_SETS"], **LINGAM_SETS_V4_EXTRA}
    g["VITALS"], g["ACTIONS"], g["RELATIONS"], g["LINGAM_SETS"] = vit, act, rel, ls
    g["AGENT_TO_ACTION"] = {a.agent: k for k, a in act.items()}
    g["REL_BY_ID"] = {r.rid: r for r in rel}
    g["PAIRS"] = tuple(dict.fromkeys((r.action, r.vital) for r in rel))
    g["VARX_CHANNELS"] = tuple([(a, "up") for a in act] + [(a, "down") for a in act]
                               + [(a, "nuis") for a, s_ in act.items() if s_.exclude_zero_endpoints])
    g["POLICY_KEYS"] = tuple(dict.fromkeys((r.action, r.event_dir) for r in rel if r.kind == "policy"))
    g["ST_FEATURES"] = tuple(f"{p}_{v}" for v in vit for p in ("L", "R", "T")) + ("dose", "elapsed_h", "co_up", "co_dn")
    g["VARX_PER_EQUATION"] = version == "v4"
    g["XCORR_STORE_DTYPE"] = np.float16 if version == "v4" else np.float32
    g["TABLE_VERSION"] = version


def lingam_owner(r: Relation) -> str | None:
    """The one VAR-LiNGAM variable set that scores relation ``r``: the first set (dict order) holding its cause
    and effect (v3: identical to scoring every set that holds both, since no v3 relation is in two sets)."""
    for sname, names in LINGAM_SETS.items():
        if r.cause in names and r.effect in names:
            return sname
    return None


ARMS: tuple[str, ...] = ("real", "nc_random", "nc_cross", "nc_cross2")
ARMS_V1: tuple[str, ...] = ("real", "nc_random", "nc_cross")
# v3: K = 8 cross-patient derangements. Draw 1 (salt 991) and 2 (7919) are v2's nc_cross / nc_cross2;
# 101-606 are the six fresh draws of the wave-3 verifier (runs/iclr/closed_loop_v2/verify/).
CROSS_ARMS: tuple[str, ...] = ("nc_cross",) + tuple(f"nc_cross{k}" for k in range(2, 9))
CROSS_SALTS: tuple[int, ...] = (991, 7919, 101, 202, 303, 404, 505, 606)
ARMS_V3: tuple[str, ...] = ("real", "nc_random") + CROSS_ARMS
PP_ARMS: tuple[str, ...] = ("real", "nc_random", "nc_cross")  # arms that keep per-patient rows
# computation groups (what analyse_case runs)
METHODS: tuple[str, ...] = ("naive", "xcorr_v1", "granger", "varx", "pcmciplus", "varlingam", "matched", "kink")
EVENT_METHODS: tuple[str, ...] = ("naive", "matched", "kink")
# v2 table columns (primary methods) and pre-specified sensitivity variants
METHODS_V2: tuple[str, ...] = ("naive", "naive_lvl", "xcorr_v1", "xcorr_dir", "xcorr_pos", "granger", "varx",
                               "pcmciplus", "varlingam", "matched", "kink", "typed_pooled")
SENSITIVITY_METHODS: tuple[str, ...] = ("matched_clean", "kink_clean", "matched_v1", "kink_v1", "matched_nobc",
                                        "kink_nobc", "matched_coint", "kink_coint", "matched_coadj", "kink_coadj",
                                        "kink_prealt")
HORIZON_FAMILIES: tuple[str, ...] = ("naive", "matched", "kink", "varx", "typed", "seqtrial")
# v3: computation groups add the pooled sequential-trial bank; table columns add seqtrial
METHODS_V3_COMPUTE: tuple[str, ...] = METHODS + ("seqtrial",)
METHODS_V3: tuple[str, ...] = ("naive", "naive_lvl", "xcorr_v1", "xcorr_dir", "xcorr_pos", "granger", "varx",
                               "pcmciplus", "varlingam", "matched", "kink", "seqtrial", "typed_pooled")
SENSITIVITY_METHODS_V3: tuple[str, ...] = SENSITIVITY_METHODS + ("seqtrial_nobc", "seqtrial_noco")
# pre-t state used by the pooled sequential-trial matching (response relations); policy relations match
# on the non-vital part only (the vital trend is the outcome)
# L: last level [t-30 s, t); R: recent level [t-10 s, t) (what the clinician sees; at 30 s no sub-window
# information exists, R is missing and set to the bank mean, i.e. neutral); T: 2-min trend.
ST_FEATURES: tuple[str, ...] = ("L_map", "R_map", "T_map", "L_hr", "R_hr", "T_hr", "L_etco2", "R_etco2", "T_etco2",
                                "dose", "elapsed_h", "co_up", "co_dn")
ST_RECENT_SEC: float = 10.0
ST_POLICY_FEATURES: tuple[str, ...] = ("dose", "elapsed_h", "co_up", "co_dn")
ST_K: int = 10
SPEC_OUTCOMES: tuple[str, ...] = ("obs", "kinkraw", "kink")
SPEC_MIN_PER_HALF: int = 3
MIN_EVENTS_PER_PATIENT: int = 3
MIN_VITAL_COVERAGE: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 2. Timescale configuration
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScaleConfig:
    name: str
    window_sec: float  # length of one numerics window
    row_stride_sec: float  # spacing between rows of the table
    reg_every: int  # rows kept for the regression / graph methods (2 at 30 s -> non-overlapping)
    horizons: tuple[float, ...]  # response horizons (s)
    kink_pre_sec: float  # local linear pre-trend fit length
    placebo_every: int  # spacing (rows) of placebo anchor candidates
    granger_p: int  # lags (regression-grid steps)
    varx_p: int  # autoregressive lags of the vitals in the VARX
    varx_bins: tuple[tuple[float, float], ...]  # distributed-lag bins for the action inputs, (lo, hi] s
    pcmci_tau_max: int
    lingam_p: int
    pre_level_sec: float = 120.0  # naive baseline
    trend_late: tuple[float, float] = (30.0, 0.0)  # "current level": [t-30, t)
    trend_early: tuple[float, float] = (150.0, 120.0)  # level 2 min earlier: [t-150, t-120]
    post_halfwidth_sec: float = 3.0
    policy_trend_sec: float = 120.0
    n_match: int = 5
    min_candidates: int = 20
    xcorr_max_lag_sec: float = 1800.0  # v1 default (edges/temporal.py DEFAULT_MAX_LAG_SEC)
    kink_pre_alt_sec: float = 60.0  # v2 kink pre-window sensitivity
    pcmci_step_sec: float = 30.0  # v2 PCMCI+ grid (block means of the native rows)
    pcmci_max_lag_sec: float = 240.0  # v2 PCMCI+ lag coverage (>= the 120 s primary drug horizon)
    st_max_controls: int = 40  # v3 seqtrial: control anchors kept per case and action (evenly spaced)

    @property
    def decision_half_sec(self) -> float:
        """Half-width of the decision interval that defines 'untreated at t' for a control anchor."""
        return self.placebo_every * self.row_stride_sec / 2.0

    @property
    def reg_step_sec(self) -> float:
        return self.row_stride_sec * self.reg_every

    @property
    def hw(self) -> float:
        return max(self.window_sec / 2.0, self.post_halfwidth_sec)

    @property
    def excl_pre_sec(self) -> float:
        return max(self.trend_early[0], self.kink_pre_sec)

    def bin_lags(self) -> list[tuple[int, int]]:
        """Regression-grid lag ranges [k0, k1] for the VARX distributed-lag bins."""
        s = self.reg_step_sec
        out = []
        for lo, hi in self.varx_bins:
            k0 = int(math.floor(lo / s + 1e-9)) + 1
            k1 = int(math.floor(hi / s + 1e-9))
            if k1 >= k0:
                out.append((k0, k1))
        return out


SCALES: dict[str, ScaleConfig] = {
    "30s": ScaleConfig(
        name="30s", window_sec=30.0, row_stride_sec=15.0, reg_every=2,
        horizons=(30.0, 60.0, 120.0), kink_pre_sec=180.0, placebo_every=1,
        granger_p=4, varx_p=4,
        varx_bins=((0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 240)),
        pcmci_tau_max=4, lingam_p=4,
    ),
    "2s": ScaleConfig(
        name="2s", window_sec=2.0, row_stride_sec=2.0, reg_every=1,
        horizons=(10.0, 30.0, 60.0, 120.0), kink_pre_sec=60.0, placebo_every=5,
        granger_p=30, varx_p=10,
        varx_bins=((0, 10), (10, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 240)),
        pcmci_tau_max=15, lingam_p=15, kink_pre_alt_sec=180.0, pcmci_step_sec=10.0, st_max_controls=120,
    ),
}


# v4 (ClosedLoopBench 1.1, 2 s on all 3,442 cases): the v3 2 s configuration except, for runtime, PCMCI+ on 30 s
# block means with lags to 240 s (tau_max 8; v3 2 s: 10 s blocks, tau_max 24) and a smaller pooled seqtrial control
# bank per case and action (the pooled bank of 3,442 cases stays larger than v3's 500-case bank).
import dataclasses as _dc  # noqa: E402

SCALES["2s_v4"] = _dc.replace(SCALES["2s"], name="2s_v4", pcmci_step_sec=30.0, st_max_controls=10)


@dataclass(frozen=True)
class BenchOptions:
    """Version switches (see module docstring). ``BenchOptions()`` is v3; ``.v2()`` is v2; ``.v1()`` is v1."""

    control_pool: str = "pre_only"  # "future_inaction" = v1
    event_itt: bool = True  # False = v1 (post-treatment 'clean' selection is primary)
    sensitivities: bool = True  # emit *_clean, *_v1, *_coint, *_coadj, kink_prealt, naive_lvl rows
    mask_stopped_tracks: bool = True  # False = v1 (dose forward-filled after the track stops)
    pcmci_v2_grid: bool = True  # False = v1 (reg grid, cfg.pcmci_tau_max)
    xcorr_variants: bool = True
    spec_test: bool = True  # per-event split-half sums for the patient-specificity test
    coint_window_sec: float = 60.0
    bias_correct: bool = True  # Abadie-Imbens regression bias correction of the matched contrasts
    coint_pre_only: bool = True  # v3: co-intervention counts use [t - w, t) only (v2: [t - w, t + w])
    seqtrial: bool = True  # v3: pooled sequential-trial control bank

    @staticmethod
    def v1() -> BenchOptions:
        return BenchOptions(control_pool="future_inaction", event_itt=False, sensitivities=False,
                            mask_stopped_tracks=False, pcmci_v2_grid=False, xcorr_variants=False, spec_test=False,
                            coint_window_sec=60.0, bias_correct=False, coint_pre_only=False, seqtrial=False)

    @staticmethod
    def v2() -> BenchOptions:
        return BenchOptions(coint_pre_only=False, seqtrial=False)

    @property
    def is_v1(self) -> bool:
        return self.control_pool == "future_inaction"


V2 = BenchOptions.v2()
V3 = BenchOptions()
# v4 (ClosedLoopBench 1.1): the v3 primary methods and admission rule; the pre-specified sensitivity variants and
# the patient-specificity split-half test are not recomputed (runtime on 3,442 cases at 2 s; both were v3 results)
V4 = BenchOptions(sensitivities=False, spec_test=False)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 3. Case data
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


@dataclass
class EventSet:
    t: np.ndarray
    before: np.ndarray
    after: np.ndarray
    usable: np.ndarray  # titration steps (starts / stops of TCI and IPPV excluded)
    t_stop: float = np.inf  # recorded track stops here; a positive dose after it is unknown (NaN)

    @property
    def direction(self) -> np.ndarray:
        return np.sign(self.after - self.before).astype(int)

    @staticmethod
    def empty() -> EventSet:
        z = np.zeros(0)
        return EventSet(z, z, z, np.zeros(0, bool))

    @staticmethod
    def build(t, before, after, spec: ActionSpec, t_stop: float = np.inf) -> EventSet:
        t = np.asarray(t, float)
        before = np.asarray(before, float)
        after = np.asarray(after, float)
        o = np.argsort(t, kind="stable")
        t, before, after = t[o], before[o], after[o]
        usable = np.abs(after - before) > 0
        if spec.exclude_zero_endpoints:
            usable &= (before > 0) & (after > 0)
        ts = float(t_stop) if np.isfinite(t_stop) else np.inf
        if len(t) and np.isfinite(ts):
            ts = max(ts, float(t[-1]))
        return EventSet(t, before, after, usable, ts)

    def dose_at(self, tau: np.ndarray, strictly_before: bool = False) -> np.ndarray:
        """Dose in force at time(s) ``tau`` (0 before the first event when it starts from 0); NaN where
        a positive dose would be forward-filled past the end of the recorded track (``t_stop``)."""
        tau = np.asarray(tau, float)
        if len(self.t) == 0:
            return np.zeros_like(tau)
        idx = np.searchsorted(self.t, tau, side="left" if strictly_before else "right") - 1
        out = np.where(idx >= 0, self.after[np.clip(idx, 0, None)], self.before[0]).astype(float)
        if np.isfinite(self.t_stop):
            out = np.where((tau > self.t_stop) & (out > 0), np.nan, out)
        return out

    def count_in(self, lo: np.ndarray, hi: np.ndarray, lo_open=False, hi_open=False) -> np.ndarray:
        i0 = np.searchsorted(self.t, lo, side="right" if lo_open else "left")
        i1 = np.searchsorted(self.t, hi, side="left" if hi_open else "right")
        return i1 - i0


class SeriesIndex:
    """O(1) NaN-aware range statistics (mean, local linear fit) of one series on a sorted time grid."""

    def __init__(self, t: np.ndarray, y: np.ndarray):
        self.t = np.asarray(t, float)
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        yv = np.where(ok, y, 0.0)
        tv = np.where(ok, self.t, 0.0)
        z = np.zeros(1)
        self.cn = np.concatenate([z, np.cumsum(ok)])
        self.cy = np.concatenate([z, np.cumsum(yv)])
        self.ct = np.concatenate([z, np.cumsum(tv)])
        self.ctt = np.concatenate([z, np.cumsum(tv * self.t)])
        self.cty = np.concatenate([z, np.cumsum(tv * yv)])

    def _idx(self, lo, hi):
        return np.searchsorted(self.t, lo, "left"), np.searchsorted(self.t, hi, "right")

    def mean(self, lo, hi) -> np.ndarray:
        i0, i1 = self._idx(np.asarray(lo, float), np.asarray(hi, float))
        n = self.cn[i1] - self.cn[i0]
        s = self.cy[i1] - self.cy[i0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    def linfit(self, anchor, lo, hi, min_n: int) -> tuple[np.ndarray, np.ndarray]:
        """Least-squares y = a + b (t - anchor) on rows with t in [lo, hi]; NaN when < min_n rows."""
        anchor = np.asarray(anchor, float)
        i0, i1 = self._idx(np.asarray(lo, float), np.asarray(hi, float))
        n = self.cn[i1] - self.cn[i0]
        st = self.ct[i1] - self.ct[i0]
        stt = self.ctt[i1] - self.ctt[i0]
        sy = self.cy[i1] - self.cy[i0]
        sty = self.cty[i1] - self.cty[i0]
        sx = st - n * anchor
        sxx = stt - 2 * anchor * st + n * anchor**2
        sxy = sty - anchor * sy
        den = n * sxx - sx**2
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.where((n >= min_n) & (den > 1e-9), (n * sxy - sx * sy) / den, np.nan)
            a = np.where(np.isfinite(b), (sy - b * sx) / np.maximum(n, 1), np.nan)
        return a, b


@dataclass
class CaseData:
    pid: str
    t: np.ndarray  # window centres (s), native row stride
    window_sec: float
    vitals: dict[str, np.ndarray]
    events: dict[str, EventSet]
    _idx: dict[str, SeriesIndex] = field(default_factory=dict, repr=False)

    @property
    def t_start(self) -> float:
        return float(self.t[0] - self.window_sec / 2)

    @property
    def t_end(self) -> float:
        return float(self.t[-1] + self.window_sec / 2)

    def index(self, vital: str) -> SeriesIndex:
        if vital not in self._idx:
            self._idx[vital] = SeriesIndex(self.t, self.vitals[vital])
        return self._idx[vital]

    def coverage(self, vital: str) -> float:
        return float(np.isfinite(self.vitals[vital]).mean()) if len(self.t) else 0.0

    def dose_series(self, action: str, t: np.ndarray | None = None) -> np.ndarray:
        return self.events.get(action, EventSet.empty()).dose_at(self.t if t is None else t)

    def with_events(self, events: dict[str, EventSet]) -> CaseData:
        return CaseData(self.pid, self.t, self.window_sec, self.vitals, events, self._idx)

    def n_usable(self, action: str, direction: int) -> int:
        ev = self.events.get(action)
        if ev is None:
            return 0
        return int((ev.usable & (ev.direction == direction)).sum())


def case_from_frames(pid: str, numerics, events_df, window_sec: float) -> CaseData:
    """Build a CaseData from a numerics DataFrame and that case's rows of the events table."""
    t = numerics["window_start_sec"].to_numpy(float) + window_sec / 2.0
    vitals = {v: numerics[v].to_numpy(float) for v in VITALS}
    return CaseData(pid, t, window_sec, vitals, events_from_frame(events_df))


def events_from_frame(events_df, t_stop: dict[str, float] | None = None) -> dict[str, EventSet]:
    out: dict[str, EventSet] = {}
    if events_df is None or len(events_df) == 0:
        return out
    for agent, g in events_df.groupby("agent"):
        a = AGENT_TO_ACTION.get(agent)
        if a is None:
            continue
        out[a] = EventSet.build(g["t_sec"], g["value_before"], g["value_after"], ACTIONS[a],
                                (t_stop or {}).get(a, np.inf))
    return out


# numerics columns holding each action's recorded set-point / rate (last value at window end, 60 s hold)
TRACK_COLUMNS: dict[str, str] = {"propofol": "tci_propofol_ct", "remifentanil": "tci_remifentanil_ct",
                                 "vent_rr": "vent_set_rr", "phenylephrine": "inf_phenylephrine_rate_ml_h",
                                 "norepinephrine": "inf_norepinephrine_rate_ml_h",
                                 "nitroglycerin": "inf_nitroglycerin_rate_ml_h"}


def track_stop_times(numerics) -> dict[str, float]:
    """Per action: window end of the last row with a recorded track value (inf when never recorded, so a
    stream known only from the events table is not masked)."""
    out = {}
    te = numerics["window_end_sec"].to_numpy(float)
    for a, col in TRACK_COLUMNS.items():
        if col not in numerics:
            continue
        ok = np.isfinite(numerics[col].to_numpy(float))
        out[a] = float(te[ok].max()) if ok.any() else np.inf
    return out


# ── negative controls ─────────────────────────────────────────────────────────────────────────────


def nc_random_times(case: CaseData, rng: np.random.Generator) -> CaseData:
    """Each action's steps (same sizes, same order) moved to uniformly random times within the case."""
    new = {}
    for a, ev in case.events.items():
        n = len(ev.t)
        tt = np.sort(rng.uniform(case.t_start, case.t_end, n)) if n else ev.t
        ts = max(ev.t_stop, float(tt[-1])) if (n and np.isfinite(ev.t_stop)) else ev.t_stop
        new[a] = EventSet(tt, ev.before.copy(), ev.after.copy(), ev.usable.copy(), ts)
    return case.with_events(new)


def nc_cross_patient(case: CaseData, partner_events: dict[str, EventSet]) -> CaseData:
    """Another patient's action stream on this patient's clock (elapsed time from case start)."""
    new = {}
    for a, ev in partner_events.items():
        keep = (ev.t >= case.t_start) & (ev.t <= case.t_end)
        new[a] = EventSet(ev.t[keep], ev.before[keep], ev.after[keep], ev.usable[keep], ev.t_stop)
    return case.with_events(new)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 4. Event designs: (a) naive, (g) matched placebo, (h) placebo-calibrated local pre-trend
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def _nn_mean(feat_e: np.ndarray, feat_c: np.ndarray, out_c: np.ndarray, k: int,
             bias_correct: bool = False) -> np.ndarray:
    """Mean outcome of the k nearest candidates (z-scored features, candidate scale).

    ``bias_correct`` (v2): Abadie-Imbens (2011) regression bias correction -- add
    (z_event - mean z_matched) @ beta, beta from the within-case OLS of the control outcome on the
    z-scored features, so imperfect matches in the tail (e.g. the steep pre-trends that trigger the
    clinician) do not leave a residual trend difference in the contrast."""
    mu = feat_c.mean(0)
    sd = feat_c.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    ze = (feat_e - mu) / sd
    zc = (feat_c - mu) / sd
    d = ((ze[:, None, :] - zc[None, :, :]) ** 2).sum(-1)
    k = min(k, zc.shape[0])
    nn = np.argpartition(d, k - 1, axis=1)[:, :k]
    out = out_c[nn].mean(1)
    if bias_correct:
        X = np.column_stack([np.ones(len(zc)), zc])
        beta = np.linalg.lstsq(X, out_c, rcond=None)[0]
        out = out + (ze - zc[nn].mean(1)) @ beta[1:]
    return out


def _match_rows(rows, name, h, s, fe, ye, fc, yc, emask, cmask, cfg, contrast_out=None, bc: bool = False) -> None:
    """Append s * (y_event - mean y of the k nearest eligible controls) for the masked events."""
    okc = cmask & np.isfinite(fc).all(1) & np.isfinite(yc)
    oke = emask & np.isfinite(fe).all(1) & np.isfinite(ye)
    if okc.sum() >= cfg.min_candidates and oke.any():
        pl = _nn_mean(fe[oke], fc[okc], yc[okc], cfg.n_match, bias_correct=bc)
        v = s * (ye[oke] - pl)
        for x in v:
            rows.append((name, h, float(x)))
        if contrast_out is not None:
            contrast_out[oke] = v


def _coint_counts(case: CaseData, action: str, t: np.ndarray, w: float, pre_only: bool = False) -> np.ndarray:
    """(n, 2) counts of other actions' up / down changes within [t - w, t + w] (v2), or within
    [t - w, t) when ``pre_only`` (v3: pre-t information only; a change after t may respond to the
    treatment, and for a control it is the clinician's future behaviour)."""
    t = np.asarray(t, float)
    out = np.zeros((len(t), 2))
    for a, e in case.events.items():
        if a == action or len(e.t) == 0:
            continue
        d = e.direction
        for j, m in enumerate((d > 0, d < 0)):
            if m.any():
                tt = e.t[m]
                hi = np.searchsorted(tt, t, "left") if pre_only else np.searchsorted(tt, t + w, "right")
                out[:, j] += hi - np.searchsorted(tt, t - w, "left")
    return out


def event_design_rows(case: CaseData, rel: Relation, cfg: ScaleConfig, opts: BenchOptions = V3,
                      spec_out: dict | None = None) -> list[tuple[str, float, float]]:
    """Per-event estimates (method, horizon, estimate in coefficient space) for one relation.

    Primary (v2): intention-to-treat events (clean pre window only) against controls that are eligible and
    untreated at t using only pre-t information. ``opts.is_v1``: v1 behaviour as primary. With
    ``opts.sensitivities`` the v1 pool / clean / co-intervention / pre-window variants are emitted too.
    ``spec_out`` (response relations) receives per-event arrays at the primary horizon for the
    patient-specificity test: tau, obs (post - last pre-level), kinkraw (post - pre-trend line), kink.
    """
    ev = case.events.get(rel.action)
    if ev is None or len(ev.t) == 0 or case.coverage(rel.vital) < 0.05:
        return []
    sel = ev.usable & (ev.direction == rel.event_dir)
    if not sel.any():
        return []
    S = case.index(rel.vital)
    W = cfg.window_sec
    restrict = rel.kind == "response" and math.isfinite(rel.pre_max)
    if restrict:  # v4: only events whose last 30 s pre-level of the vital is below pre_max
        with np.errstate(invalid="ignore"):
            sel = sel & (S.mean(ev.t - cfg.trend_late[0], ev.t - W / 2) < rel.pre_max)
        if not sel.any():
            return []
    tau = ev.t[sel]
    D_e = ev.before[sel]
    s = rel.event_dir
    rows: list[tuple[str, float, float]] = []
    Pex = cfg.excl_pre_sec
    kmin = max(3, int(0.7 * cfg.kink_pre_sec / cfg.row_stride_sec))
    late = lambda x: S.mean(x - cfg.trend_late[0], x - W / 2)  # noqa: E731
    early = lambda x: S.mean(x - cfg.trend_early[0], x - cfg.trend_early[1])  # noqa: E731
    # eligible events: no other change of this action in the pre window (pre-t information only)
    clean_pre = ev.count_in(tau - Pex, tau, hi_open=True) == 0
    grid = case.t[:: cfg.placebo_every]
    dur = max(case.t_end - case.t_start, 1.0)
    v1 = opts.is_v1
    sens = opts.sensitivities and not v1
    # v2 control pool: eligible (no change in the pre window) and untreated at g (decision interval)
    pool_v2 = ev.count_in(grid - Pex, grid + cfg.decision_half_sec) == 0
    if restrict:  # the same pre-level restriction for the control anchors
        with np.errstate(invalid="ignore"):
            pool_v2 = pool_v2 & (late(grid) < rel.pre_max)
    if sens:
        cw = opts.coint_window_sec
        co_e = _coint_counts(case, rel.action, tau, cw, opts.coint_pre_only)
        co_c = _coint_counts(case, rel.action, grid, cw, opts.coint_pre_only)
        noco_e = co_e.sum(1) == 0
        noco_c = co_c.sum(1) == 0

    if rel.kind == "response":
        H = cfg.horizons
        post = {h: S.mean(np.maximum(tau + h - cfg.hw, tau + W / 2), tau + h + cfg.hw) for h in H}
        base = S.mean(tau - cfg.pre_level_sec, tau - W / 2)
        L_e = late(tau)
        for h in H:
            est = s * (post[h] - base)
            for v in est[np.isfinite(est)]:
                rows.append(("naive", h, float(v)))
            if not v1:
                est = s * (post[h] - L_e)
                for v in est[np.isfinite(est)]:
                    rows.append(("naive_lvl", h, float(v)))
        T_e = L_e - early(tau)
        a_e, b_e = S.linfit(tau, tau - cfg.kink_pre_sec, tau - W / 2, kmin)
        g = grid
        # v1 placebo pool: no change of this action anywhere near, *including after g* (future inaction)
        pool_v1 = ev.count_in(g - Pex, g + max(H) + cfg.hw) == 0
        if (pool_v1 if v1 else pool_v2).sum() < cfg.min_candidates and not sens and spec_out is None:
            return rows
        L_c = late(g)
        T_c = L_c - early(g)
        D_c = ev.dose_at(g)
        a_c, b_c = S.linfit(g, g - cfg.kink_pre_sec, g - W / 2, kmin)
        post_c = {h: S.mean(np.maximum(g + h - cfg.hw, g + W / 2), g + h + cfg.hw) for h in H}
        if sens:
            alt = cfg.kink_pre_alt_sec
            Pex2 = max(Pex, alt)
            kmin2 = max(3, int(0.7 * alt / cfg.row_stride_sec))
            clean_pre2 = ev.count_in(tau - Pex2, tau, hi_open=True) == 0
            pool2 = ev.count_in(g - Pex2, g + cfg.decision_half_sec) == 0
            a_e2, b_e2 = S.linfit(tau, tau - alt, tau - W / 2, kmin2)
            a_c2, b_c2 = S.linfit(g, g - alt, g - W / 2, kmin2)
        h_spec = rel.horizon_s if rel.horizon_s in H else max(H)
        for h in H:
            post_clean = clean_pre & (ev.count_in(tau, tau + h + cfg.hw, lo_open=True) == 0)
            y_e = post[h] - L_e
            y_c = post_c[h] - L_c
            dv_e = post[h] - (a_e + b_e * h)
            dv_c = post_c[h] - (a_c + b_c * h)
            fem, fcm = np.column_stack([L_e, T_e, D_e]), np.column_stack([L_c, T_c, D_c])
            fek, fck = np.column_stack([a_e, b_e, D_e]), np.column_stack([a_c, b_c, D_c])
            kink_c = np.full(len(tau), np.nan) if (spec_out is not None and h == h_spec) else None
            bc = opts.bias_correct and not v1
            if v1:
                variants = [("", post_clean, pool_v1, None, False)]
            else:
                variants = [("", clean_pre, pool_v2, None, bc)]
                if sens:
                    variants += [("_clean", post_clean, pool_v2, None, bc), ("_v1", post_clean, pool_v1, None, False),
                                 ("_nobc", clean_pre, pool_v2, None, False),
                                 ("_coint", clean_pre & noco_e, pool_v2 & noco_c, None, bc),
                                 ("_coadj", clean_pre, pool_v2, "coadj", bc)]
            for suf, emask, cmask, extra, bcv in variants:
                if extra == "coadj":
                    fe1, fc1 = np.column_stack([fem, co_e]), np.column_stack([fcm, co_c])
                    fe2, fc2 = np.column_stack([fek, co_e]), np.column_stack([fck, co_c])
                else:
                    fe1, fc1, fe2, fc2 = fem, fcm, fek, fck
                _match_rows(rows, "matched" + suf, h, s, fe1, y_e, fc1, y_c, emask, cmask, cfg, bc=bcv)
                _match_rows(rows, "kink" + suf, h, s, fe2, dv_e, fc2, dv_c, emask, cmask, cfg,
                            kink_c if suf == "" else None, bc=bcv)
            if sens:
                dv_e2 = post[h] - (a_e2 + b_e2 * h)
                dv_c2 = post_c[h] - (a_c2 + b_c2 * h)
                _match_rows(rows, "kink_prealt", h, s, np.column_stack([a_e2, b_e2, D_e]), dv_e2,
                            np.column_stack([a_c2, b_c2, D_c]), dv_c2, clean_pre2, pool2, cfg, bc=bc)
            if kink_c is not None:
                spec_out.update(tau=tau[clean_pre], obs=(s * y_e)[clean_pre], kinkraw=(s * dv_e)[clean_pre],
                                kink=kink_c[clean_pre])
        return rows

    # ── policy: the vital's trend *before* the action ──
    h = 0.0
    T_e = late(tau) - early(tau)
    for v in (s * T_e)[np.isfinite(T_e)]:
        rows.append(("naive", h, float(v)))
    _, b_e = S.linfit(tau, tau - cfg.kink_pre_sec, tau - W / 2, kmin)
    g = grid
    pool_v1 = ev.count_in(g - Pex, g + 30.0) == 0  # v1: no change in the next 30 s either
    T_c = late(g) - early(g)
    _, b_c = S.linfit(g, g - cfg.kink_pre_sec, g - W / 2, kmin)
    D_c = ev.dose_at(g)
    # policy placebo: same dose and same phase of the case, vital trend left free
    fe = np.column_stack([D_e, (tau - case.t_start) / dur])
    fc = np.column_stack([D_c, (g - case.t_start) / dur])
    # (policy matching features are dose and phase; the outcome is the trend itself, so no bias correction)
    variants = [("", pool_v1 if v1 else pool_v2)] + ([("_v1", pool_v1)] if sens else [])
    for suf, cmask in variants:
        for name, ye, yc, scale in (("matched", T_e, T_c, 1.0), ("kink", b_e, b_c, cfg.policy_trend_sec)):
            _match_rows(rows, name + suf, h, s, fe, scale * ye, fc, scale * yc, clean_pre, cmask, cfg)
    return rows


# ── (i) v3: pooled sequential-trial contrast ─────────────────────────────────────────────────────────


def st_state_features(case: CaseData, cfg: ScaleConfig, action: str, t: np.ndarray, dose: np.ndarray,
                      coint_window_sec: float = 60.0) -> np.ndarray:
    """(n, len(ST_FEATURES)) pre-t state at decision times ``t``: per vital the last level [t-30 s, t), the
    recent level [t-10 s, t) (NaN at 30 s: no window ends inside it) and the 2-min trend (last level minus
    the level in [t-150, t-120] s); the dose in force before t; elapsed case time (h); other actions' up /
    down changes in [t - 60 s, t). Nothing at or after t is used."""
    t = np.asarray(t, float)
    if len(t) == 0:
        return np.zeros((0, len(ST_FEATURES)))
    W = cfg.window_sec
    cols = []
    for v in VITALS:
        S = case.index(v)
        L = S.mean(t - cfg.trend_late[0], t - W / 2)
        R = S.mean(t - ST_RECENT_SEC, t - W / 2)
        E = S.mean(t - cfg.trend_early[0], t - cfg.trend_early[1])
        cols += [L, R, L - E]
    co = _coint_counts(case, action, t, coint_window_sec, pre_only=True)
    cols += [np.asarray(dose, float), (t - case.t_start) / 3600.0, co[:, 0], co[:, 1]]
    return np.column_stack(cols)


def st_outcomes(case: CaseData, cfg: ScaleConfig, t: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(n, len(VITALS) * len(horizons)), vital-major: post-window mean at t + h minus the last pre-level
    (the ``matched`` outcome)."""
    t = np.asarray(t, float)
    nh = len(cfg.horizons)
    if len(t) == 0:
        return np.zeros((0, len(VITALS) * nh))
    W = cfg.window_sec
    out = []
    for v in VITALS:
        S = case.index(v)
        L = F[:, ST_FEATURES.index(f"L_{v}")]
        for h in cfg.horizons:
            out.append(S.mean(np.maximum(t + h - cfg.hw, t + W / 2), t + h + cfg.hw) - L)
    return np.column_stack(out)


def seqtrial_bank(case: CaseData, cfg: ScaleConfig, opts: BenchOptions = V3) -> dict[str, dict]:
    """Per action: this case's eligible events (usable titrations, both directions, no change of the
    action in the pre window) and a deterministic evenly spaced subsample (<= ``cfg.st_max_controls``) of
    its control anchors (eligible and untreated at g: the v2 pool, pre-t information only), each with the
    pre-t state (``ST_FEATURES``) and the outcomes (``st_outcomes``). The aggregator pools the banks across
    patients and matches every event to other patients' controls (``pooled_nn_contrast``)."""
    out: dict[str, dict] = {}
    Pex = cfg.excl_pre_sec
    grid = case.t[:: cfg.placebo_every]
    for a in sorted(case.events):
        ev = case.events[a]
        if a not in ACTIONS or len(ev.t) == 0 or not ev.usable.any():
            continue
        clean_pre = ev.count_in(ev.t - Pex, ev.t, hi_open=True) == 0
        sel = ev.usable & clean_pre
        pool = np.flatnonzero(ev.count_in(grid - Pex, grid + cfg.decision_half_sec) == 0)
        if len(pool) > cfg.st_max_controls:
            pool = pool[np.unique(np.round(np.linspace(0, len(pool) - 1, cfg.st_max_controls)).astype(int))]
        g = grid[pool]
        tau = ev.t[sel]
        Fe = st_state_features(case, cfg, a, tau, ev.before[sel], opts.coint_window_sec)
        Fc = st_state_features(case, cfg, a, g, ev.dose_at(g), opts.coint_window_sec)
        out[a] = dict(dir=ev.direction[sel].astype(np.int8), tau=tau, Fe=Fe.astype(np.float32),
                      Ye=st_outcomes(case, cfg, tau, Fe).astype(np.float32), tc=g, Fc=Fc.astype(np.float32),
                      Yc=st_outcomes(case, cfg, g, Fc).astype(np.float32))
    return out


NN_BACKEND: str = "kdtree"  # v3; v4 runner sets "torch" (exact chunked brute force, float32, CPU threads)
NN_TORCH_THREADS: int = 4
NN_TORCH_CHUNK: int = 2048


def _nn_torch(ze: np.ndarray, zc: np.ndarray, ge: np.ndarray, gc: np.ndarray, k: int,
              exclude_same: bool) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest controls (squared Euclidean, float32) of every event row, excluding controls of the
    event's own group; chunked brute force with torch on the CPU. Returns (nn index (n_e, k), valid mask)."""
    import torch

    torch.set_num_threads(NN_TORCH_THREADS)
    C = torch.from_numpy(np.ascontiguousarray(zc, dtype=np.float32))
    cn = (C * C).sum(1)
    gC = torch.from_numpy(np.asarray(gc, dtype=np.int64))
    kk = int(min(k, len(zc)))
    nn = np.zeros((len(ze), kk), np.int64)
    with torch.no_grad():
        for s0 in range(0, len(ze), NN_TORCH_CHUNK):
            Q = torch.from_numpy(np.ascontiguousarray(ze[s0 : s0 + NN_TORCH_CHUNK], dtype=np.float32))
            d = (Q * Q).sum(1, keepdim=True) + cn[None, :] - 2.0 * (Q @ C.T)
            if exclude_same:
                gQ = torch.from_numpy(np.asarray(ge[s0 : s0 + NN_TORCH_CHUNK], dtype=np.int64))
                d.masked_fill_(gQ[:, None] == gC[None, :], float("inf"))
            nn[s0 : s0 + len(Q)] = torch.topk(d, kk, dim=1, largest=False, sorted=True).indices.numpy()
    valid = (np.asarray(gc)[nn] != np.asarray(ge)[:, None]) if exclude_same else np.ones(nn.shape, bool)
    return nn, valid


def pooled_nn_contrast(fe: np.ndarray, ye: np.ndarray, ge: np.ndarray, fc: np.ndarray, yc: np.ndarray,
                       gc: np.ndarray, k: int = ST_K, bias_correct: bool | str = True, exclude_same: bool = True,
                       min_match: int | None = None, workers: int = 1):
    """Event outcome minus the mean outcome of its ``k`` nearest controls from *other* groups (patients).

    Features are z-scored on the control bank; a missing feature is set to the bank mean. With
    ``bias_correct`` the Abadie-Imbens (2011) regression correction (z_event - mean z_matched) @ beta is
    added, beta from the OLS of the control outcome on the z-features over the whole pooled bank.
    Returns (n_events, n_outcomes); NaN where fewer than ``min_match`` (default max(3, k // 2)) matched
    controls have a finite outcome. ``bias_correct="both"`` returns (corrected, uncorrected) from one
    neighbour search. Deterministic (ties broken by bank order).

    Search: the 3k nearest controls first; rows with fewer than k controls from other groups among them are
    searched again with k + (largest group size) neighbours, which always suffices."""
    from scipy.spatial import cKDTree

    fe = np.asarray(fe, float)
    fc = np.asarray(fc, float)
    ye = np.asarray(ye, float).reshape(len(fe), -1)
    yc = np.asarray(yc, float).reshape(len(fc), -1)
    ge = np.asarray(ge)
    gc = np.asarray(gc)
    n_e, n_out = ye.shape
    both = bias_correct == "both"
    out_bc = np.full((n_e, n_out), np.nan)
    out_raw = np.full((n_e, n_out), np.nan)
    if n_e == 0 or len(fc) < k:
        return (out_bc, out_raw) if both else (out_bc if bias_correct else out_raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.nanmean(fc, 0)
        sd = np.nanstd(fc, 0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    zc = np.nan_to_num((fc - mu) / sd)
    ze = np.nan_to_num((fe - mu) / sd)
    tree = cKDTree(zc)

    def search(q_idx, kq):
        _, ix = tree.query(ze[q_idx], k=kq, workers=workers)
        ix = np.asarray(ix).reshape(len(q_idx), kq)
        ok_ = (gc[ix] != ge[q_idx, None]) if exclude_same else np.ones(ix.shape, bool)
        o = np.argsort(~ok_, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(ix, o, 1), np.take_along_axis(ok_, o, 1)

    all_rows = np.arange(n_e)
    if NN_BACKEND == "torch":  # v4: exact brute force (same neighbours as the KD-tree up to float32 near-ties)
        nn, valid = _nn_torch(ze, zc, ge, gc, k, exclude_same)
    else:
        nn, valid = search(all_rows, int(min(len(zc), 3 * k if exclude_same else k)))
    if exclude_same and NN_BACKEND != "torch":
        short = np.flatnonzero(valid.sum(1) < k)
        kfull = int(min(len(zc), k + np.unique(gc, return_counts=True)[1].max()))
        if len(short) and kfull > 3 * k:
            nn2, valid2 = search(short, kfull)
            nn[short], valid[short] = nn2, valid2
    mm = min_match if min_match is not None else max(3, k // 2)
    X = np.column_stack([np.ones(len(zc)), zc])
    zn = zc[nn]  # (n_e, k, F)
    for j in range(n_out):
        y = yc[:, j]
        fin = np.isfinite(y)
        m = valid & fin[nn]
        n = m.sum(1)
        raw = np.where(m, np.nan_to_num(y)[nn], 0.0).sum(1) / np.maximum(n, 1)
        bad = (n < mm) | ~np.isfinite(ye[:, j])
        v = ye[:, j] - raw
        v[bad] = np.nan
        out_raw[:, j] = v
        if bias_correct and fin.sum() > X.shape[1] + 5:
            beta = np.linalg.lstsq(X[fin], y[fin], rcond=None)[0]
            zbar = (zn * m[..., None]).sum(1) / np.maximum(n, 1)[:, None]
            v = ye[:, j] - (raw + (ze - zbar) @ beta[1:])
            v[bad] = np.nan
            out_bc[:, j] = v
        elif bias_correct:
            out_bc[:, j] = out_raw[:, j]
    if both:
        return out_bc, out_raw
    return out_bc if bias_correct else out_raw


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 5. (b) v1-style lagged cross-correlation
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def xcorr_curve(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray | None:
    """Normalised cross-correlation c[k] = sum_t xc[t] yc[t+k], k in [-max_lag, max_lag].

    Mirrors ``clbench.edges.temporal.temporal_relation`` (centre, normalise by the product of
    norms, positive lag = x leads y) on continuous series; missing y samples are mean-imputed.
    """
    from scipy.signal import correlate

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    okx = np.isfinite(x)  # doses are NaN after a track stops (v2 masking)
    if ok.mean() < MIN_VITAL_COVERAGE or okx.mean() < MIN_VITAL_COVERAGE or len(x) < 10:
        return None
    xc = np.where(okx, x - x[okx].mean(), 0.0)
    yc = np.where(ok, y - y[ok].mean(), 0.0)
    den = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if den <= 1e-12:
        return None
    n = len(x)
    L = min(max_lag, n - 1)
    full = correlate(yc, xc, mode="full", method="fft") / den  # index n-1+k  <->  lag k
    out = np.full(2 * max_lag + 1, np.nan)
    out[max_lag - L : max_lag + L + 1] = full[n - 1 - L : n + L]
    return out


def xcorr_peak(curve: np.ndarray, max_lag: int, mode: str = "abs", cause_sign: int = 0) -> tuple[float, int]:
    """Value and lag (grid steps, + = x leads) of the cross-correlation peak.

    mode "abs": signed value at argmax |c| (Wave-2 ``xcorr_v1``); "pos": v1's positive-peak rule
    (argmax c); "dir": argmax |c| restricted to lags where the cause leads (cause_sign=+1: lags >= 1,
    x = action leads; -1: lags <= -1, the vital leads)."""
    c = np.where(np.isfinite(curve), curve, 0.0)
    lags = np.arange(len(c)) - max_lag
    if mode == "pos":
        i = int(np.argmax(c))
    elif mode == "dir":
        m = lags * cause_sign >= 1
        idx = np.flatnonzero(m)
        i = int(idx[np.argmax(np.abs(c[idx]))])
    else:
        i = int(np.argmax(np.abs(c)))
    return float(c[i]), i - max_lag


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 6. Regression utilities: lag designs, per-case OLS, pooled sufficient statistics
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def lag_matrix(Z: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """X[t] = [Z[t-1, 0..k-1], ..., Z[t-p, 0..k-1]] (lag-major), Y[t] = Z[t]; rows t = p..n-1."""
    n, k = Z.shape
    X = np.empty((n - p, k * p))
    for lag in range(1, p + 1):
        X[:, (lag - 1) * k : lag * k] = Z[p - lag : n - lag]
    return X, Z[p:]


def ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def suff_stats(X: np.ndarray, Y: np.ndarray, demean: bool = True) -> np.ndarray:
    """Flattened [X'X, X'Y, n] of the (optionally within-case demeaned) design."""
    if demean:
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
    return np.concatenate([(X.T @ X).ravel(), (X.T @ Y).ravel(), [len(X)]])


def solve_suff(S: np.ndarray, k: int, m: int, ridge: float = 1e-8) -> np.ndarray:
    """beta (k x m) from flattened sufficient statistics."""
    XtX = S[: k * k].reshape(k, k)
    XtY = S[k * k : k * k + k * m].reshape(k, m)
    lam = ridge * max(np.trace(XtX) / max(k, 1), 1e-12)
    try:
        return np.linalg.solve(XtX + lam * np.eye(k), XtY)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(XtX + lam * np.eye(k), XtY, rcond=None)[0]


def reg_grid(case: CaseData, cfg: ScaleConfig) -> np.ndarray:
    return np.arange(0, len(case.t), cfg.reg_every)


# ── (c) pairwise Granger / VAR ─────────────────────────────────────────────────────────────────────


@dataclass
class GrangerFit:
    n: int
    est_xy: float  # sum of x-lag coefficients in the y equation (x -> y)
    p_xy: float  # Granger F-test p-value, x -> y
    est_yx: float
    p_yx: float


def granger_design(x: np.ndarray, y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    Z = np.column_stack([x, y])
    X, Y = lag_matrix(Z, p)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return X[ok], Y[ok]


def granger_fit(x: np.ndarray, y: np.ndarray, p: int, min_rows_per_param: int = 5) -> GrangerFit | None:
    from scipy import stats

    X, Y = granger_design(x, y, p)
    k = 2 * p + 1
    if len(X) < min_rows_per_param * k or X[:, 0::2].std() < 1e-12 or X[:, 1::2].std() < 1e-12:
        return None
    Xi = np.column_stack([np.ones(len(X)), X])
    B = ols(Xi, Y)
    res = Y - Xi @ B
    rss_u = (res**2).sum(0)
    xcols = 1 + np.arange(0, 2 * p, 2)  # lag-major: x at even offsets
    ycols = 1 + np.arange(1, 2 * p, 2)
    dfd = len(X) - k
    out = []
    for eq, drop in ((1, xcols), (0, ycols)):
        keep = np.setdiff1d(np.arange(k), drop)
        Br = ols(Xi[:, keep], Y[:, eq])
        rss_r = ((Y[:, eq] - Xi[:, keep] @ Br) ** 2).sum()
        F = ((rss_r - rss_u[eq]) / p) / max(rss_u[eq] / dfd, 1e-300)
        pv = float(stats.f.sf(F, p, dfd)) if np.isfinite(F) else np.nan
        est = float(B[drop, eq].sum())
        out.append((est, pv))
    return GrangerFit(len(X), out[0][0], out[0][1], out[1][0], out[1][1])


def granger_suff(x: np.ndarray, y: np.ndarray, p: int) -> np.ndarray | None:
    X, Y = granger_design(x, y, p)
    if len(X) < 5 * (2 * p + 1):
        return None
    return suff_stats(X, Y)


def granger_pooled_est(S: np.ndarray, p: int) -> tuple[float, float]:
    """(x -> y, y -> x) sums of lag coefficients from pooled FE sufficient statistics."""
    B = solve_suff(S, 2 * p, 2)
    return float(B[0::2, 1].sum()), float(B[1::2, 0].sum())


# ── (d) VARX with exogenous action inputs ──────────────────────────────────────────────────────────


def increment_series(ev: EventSet, t_grid: np.ndarray, window_sec: float, mask: np.ndarray) -> np.ndarray:
    """Signed dose increments of the selected events, summed into the grid window containing them."""
    u = np.zeros(len(t_grid))
    if len(ev.t) == 0 or not mask.any():
        return u
    starts = t_grid - window_sec / 2
    idx = np.searchsorted(starts, ev.t[mask], side="right") - 1
    ok = (idx >= 0) & (ev.t[mask] < t_grid[np.clip(idx, 0, None)] + window_sec / 2)
    np.add.at(u, idx[ok], (ev.after - ev.before)[mask][ok])
    return u


def binned_lags(u: np.ndarray, bins: list[tuple[int, int]]) -> np.ndarray:
    """Column j = sum_{k=k0_j}^{k1_j} u[t-k] (zeros before the start)."""
    c = np.concatenate([[0.0], np.cumsum(u)])  # c[i] = sum u[:i]
    n = len(u)
    t = np.arange(n)
    out = np.empty((n, len(bins)))
    for j, (k0, k1) in enumerate(bins):
        hi = np.clip(t - k0 + 1, 0, n)  # sum u[t-k1 .. t-k0] = c[t-k0+1] - c[t-k1]
        lo = np.clip(t - k1, 0, n)
        out[:, j] = c[hi] - c[lo]
    return out


VARX_PER_EQUATION: bool = False  # v4 (``use_table("v4")``): per-equation row masks, see varx_response_design_v4
VARX_CHANNELS: tuple[tuple[str, str], ...] = tuple(
    [(a, "up") for a in ACTIONS] + [(a, "down") for a in ACTIONS]
    + [(a, "nuis") for a, s in ACTIONS.items() if s.exclude_zero_endpoints]
)


def varx_columns(cfg: ScaleConfig) -> dict[str, slice]:
    nb = len(cfg.bin_lags())
    k = len(VITALS)
    cols = {"ar": slice(0, k * cfg.varx_p)}
    off = k * cfg.varx_p
    for a, c in VARX_CHANNELS:
        cols[f"{a}:{c}"] = slice(off, off + nb)
        off += nb
    cols["_total"] = slice(0, off)
    return cols


def _channel_masks(ev: EventSet) -> dict[str, np.ndarray]:
    d = ev.direction
    return {"up": ev.usable & (d > 0), "down": ev.usable & (d < 0), "nuis": ~ev.usable & (d != 0)}


def varx_response_design(case: CaseData, cfg: ScaleConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Rows: dV_t (MAP, HR, EtCO2) on dV lags 1..p and binned action increments (up/down/nuisance)."""
    gi = reg_grid(case, cfg)
    tg = case.t[gi]
    Wg = cfg.reg_step_sec
    V = np.column_stack([case.vitals[v][gi] for v in VITALS])
    dV = np.vstack([np.full((1, V.shape[1]), np.nan), np.diff(V, axis=0)])
    p = cfg.varx_p
    if len(tg) <= p + 10:
        return None
    Xar, Y = lag_matrix(dV, p)
    bins = cfg.bin_lags()
    blocks = [Xar]
    for a, c in VARX_CHANNELS:
        ev = case.events.get(a, EventSet.empty())
        m = _channel_masks(ev)[c] if len(ev.t) else np.zeros(0, bool)
        u = increment_series(ev, tg, Wg, m)
        blocks.append(binned_lags(u, bins)[p:])
    X = np.hstack(blocks)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    if ok.sum() < 50:
        return None
    return X[ok], Y[ok]


def varx_response_design_v4(case: CaseData, cfg: ScaleConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """v4: as ``varx_response_design`` but a missing vital does not drop the row: missing dV lags enter the
    regressors as 0 (no change information) and each equation later uses only its own rows with a finite
    outcome (``suff_stats_per_equation``). Returns (X, Y) with NaN in Y."""
    gi = reg_grid(case, cfg)
    tg = case.t[gi]
    Wg = cfg.reg_step_sec
    V = np.column_stack([case.vitals[v][gi] for v in VITALS])
    dV = np.vstack([np.full((1, V.shape[1]), np.nan), np.diff(V, axis=0)])
    p = cfg.varx_p
    if len(tg) <= p + 10:
        return None
    Xar, Y = lag_matrix(np.nan_to_num(dV, nan=0.0), p)
    Y = dV[p:]
    bins = cfg.bin_lags()
    blocks = [Xar]
    for a, c in VARX_CHANNELS:
        ev = case.events.get(a, EventSet.empty())
        m = _channel_masks(ev)[c] if len(ev.t) else np.zeros(0, bool)
        u = increment_series(ev, tg, Wg, m)
        blocks.append(binned_lags(u, bins)[p:])
    X = np.hstack(blocks)
    if np.isfinite(Y).any(1).sum() < 50:
        return None
    return X, Y


def suff_stats_per_equation(X: np.ndarray, Y: np.ndarray, min_rows: int = 50) -> dict[int, np.ndarray]:
    """Equation j -> within-case demeaned [X'X, X'y_j, n_j] over the rows with a finite y_j (X is finite)."""
    out: dict[int, np.ndarray] = {}
    full = None
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        nj = int(ok.sum())
        if nj < min_rows:
            continue
        Xj = X[ok]
        y = Y[ok, j]
        if (~ok).sum() * 2 > len(ok):
            Xc = Xj - Xj.mean(0)
            XtX = Xc.T @ Xc
        else:  # the full cross-products (once per case) minus the missing rows (same numbers, cheaper)
            if full is None:
                full = (X.T @ X, X.sum(0))
            XtX_full, sx_full = full
            Xm = X[~ok]
            sx = sx_full - Xm.sum(0)
            XtX = XtX_full - Xm.T @ Xm - np.outer(sx, sx) / nj
        yc = y - y.mean()
        Xty = Xj.T @ yc  # sum (x - mu)(y - ybar) = sum x (y - ybar)
        out[j] = np.concatenate([XtX.ravel(), Xty.ravel(), [nj]])
    return out


def varx_step_response(beta: np.ndarray, cfg: ScaleConfig, channel: str, horizon_s: float) -> np.ndarray:
    """Cumulative response of (MAP, HR, EtCO2) at ``horizon_s`` to a unit increment on ``channel``."""
    k = len(VITALS)
    p = cfg.varx_p
    cols = varx_columns(cfg)
    Phi = [beta[(lag - 1) * k : lag * k, :].T for lag in range(1, p + 1)]  # Phi[lag-1][eq, var]
    bcoef = beta[cols[channel], :]  # (nb, k)
    bins = cfg.bin_lags()
    T = int(round(horizon_s / cfg.reg_step_sec))
    dv = np.zeros((T + 1, k))
    for t in range(1, T + 1):
        x = np.zeros(k)
        for lag in range(1, min(p, t) + 1):
            x += Phi[lag - 1] @ dv[t - lag]
        for j, (k0, k1) in enumerate(bins):
            if k0 <= t <= k1:
                x += bcoef[j]
        dv[t] = x
    return dv[1:].sum(0)


POLICY_KEYS: tuple[tuple[str, int], ...] = tuple(dict.fromkeys((r.action, r.event_dir) for r in RELATIONS if r.kind == "policy"))


def varx_policy_columns(cfg: ScaleConfig) -> list[str]:
    own = [b for b in cfg.bin_lags() if b[1] * cfg.reg_step_sec <= cfg.policy_trend_sec]
    return [f"trend:{v}" for v in VITALS] + [f"level:{v}" for v in VITALS] + [f"own:{i}" for i in range(len(own))]


def varx_policy_design(case: CaseData, cfg: ScaleConfig, action: str, direction: int):
    """u^dir_{a,t} on vital 2-min trends and levels at t-1 and the action's own recent increments."""
    ev = case.events.get(action)
    if ev is None or len(ev.t) == 0:
        return None
    gi = reg_grid(case, cfg)
    tg = case.t[gi]
    Wg = cfg.reg_step_sec
    K = int(round(cfg.policy_trend_sec / Wg))
    V = np.column_stack([case.vitals[v][gi] for v in VITALS])
    n = len(tg)
    if n <= K + 10:
        return None
    lvl = np.full_like(V, np.nan)
    lvl[1:] = V[:-1]
    trd = np.full_like(V, np.nan)
    trd[K + 1 :] = V[K:-1] - V[: n - K - 1]
    own_bins = [b for b in cfg.bin_lags() if b[1] * Wg <= cfg.policy_trend_sec]
    u_all = increment_series(ev, tg, Wg, ev.usable)
    own = binned_lags(u_all, own_bins)
    m = ev.usable & (ev.direction == direction)
    y = increment_series(ev, tg, Wg, m)
    if VARX_PER_EQUATION:  # v4: a missing vital does not drop the row (trend -> 0, level -> case mean)
        trd = np.nan_to_num(trd, nan=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu = np.nanmean(lvl, 0)
        lvl = np.where(np.isfinite(lvl), lvl, np.nan_to_num(mu, nan=0.0)[None, :])
        lvl[: 1] = np.nan  # the first row has no previous level
    X = np.hstack([trd, lvl, own])
    ok = np.isfinite(X).all(1)
    if ok.sum() < 50:
        return None
    return X[ok], y[ok, None]


# ── (e) PCMCI+ ─────────────────────────────────────────────────────────────────────────────────────


def pcmci_grid(case: CaseData, cfg: ScaleConfig, opts: BenchOptions = V3) -> tuple[np.ndarray, dict, int]:
    """(grid times, vitals on the grid, tau_max). v1: regression grid and cfg.pcmci_tau_max. v2: block means
    of ``pcmci_step_sec`` (the non-overlapping 30 s windows at 30 s) and lags up to ``pcmci_max_lag_sec``."""
    if not opts.pcmci_v2_grid:
        gi = reg_grid(case, cfg)
        return case.t[gi], {v: case.vitals[v][gi] for v in VITALS}, cfg.pcmci_tau_max
    tau_max = int(round(cfg.pcmci_max_lag_sec / cfg.pcmci_step_sec))
    if abs(cfg.pcmci_step_sec - cfg.reg_step_sec) < 1e-9:
        gi = reg_grid(case, cfg)
        return case.t[gi], {v: case.vitals[v][gi] for v in VITALS}, tau_max
    k = max(1, int(round(cfg.pcmci_step_sec / cfg.row_stride_sec)))
    n = len(case.t) // k
    tg = case.t[: n * k].reshape(n, k).mean(1)
    vit = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for v in VITALS:
            vit[v] = np.nanmean(case.vitals[v][: n * k].reshape(n, k), 1)
    return tg, vit, tau_max


def pcmci_case(case: CaseData, cfg: ScaleConfig, pc_alpha: float = 0.01,
               opts: BenchOptions = V3) -> dict[tuple[str, str], tuple[float, float, bool, int]]:
    """PCMCI+ (ParCorr) on [action doses with >=1 event, MAP, HR, EtCO2]; per (cause, effect) the
    lagged link (tau >= 1) with the smallest p-value: (val, p, in final graph, tau)."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    tg, vit, tau_max = pcmci_grid(case, cfg, opts)
    names, cols = [], []
    for a in ACTIONS:
        ev = case.events.get(a)
        if ev is None or not ev.usable.any():
            continue
        x = ev.dose_at(tg)
        if np.isfinite(x).mean() >= MIN_VITAL_COVERAGE and np.nanstd(x) > 1e-9:
            names.append(a)
            cols.append(x)
    for v in VITALS:
        y = vit[v]
        if np.isfinite(y).mean() >= MIN_VITAL_COVERAGE and np.nanstd(y) > 1e-9:
            names.append(v)
            cols.append(y)
    if not any(n in ACTIONS for n in names) or not any(n in VITALS for n in names):
        return {}
    X = np.column_stack(cols)
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12)
    flag = 999.0
    X = np.where(np.isfinite(X), X, flag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pp.DataFrame(X, var_names=names, missing_flag=flag)
        pc = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
        r = pc.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)
    val, pm, graph = r["val_matrix"], r["p_matrix"], r["graph"]
    out = {}
    for a, v in PAIRS:
        if a not in names or v not in names:
            continue
        for c, e in ((a, v), (v, a)):
            i, j = names.index(c), names.index(e)
            pv = pm[i, j, 1:]
            if not np.isfinite(pv).any():
                continue
            tau = int(np.nanargmin(pv)) + 1
            out[(c, e)] = (float(val[i, j, tau]), float(pm[i, j, tau]), graph[i, j, tau] == "-->", tau)
    return out


# ── (f) VAR-LiNGAM (pooled) ─────────────────────────────────────────────────────────────────────────

LINGAM_SETS: dict[str, tuple[str, ...]] = {
    "core": ("propofol", "remifentanil", "vent_rr", "map", "hr", "etco2"),
    "pressor": ("phenylephrine", "norepinephrine", "nitroglycerin", "map", "hr"),
}


def lingam_matrix(case: CaseData, cfg: ScaleConfig, set_name: str) -> np.ndarray | None:
    gi = reg_grid(case, cfg)
    tg = case.t[gi]
    names = LINGAM_SETS[set_name]
    if set_name == "pressor" and not any(case.events.get(a) is not None and case.events[a].usable.any() for a in PRESSORS):
        return None
    cols = []
    for nme in names:
        if nme in ACTIONS:
            cols.append(case.dose_series(nme, tg))
        else:
            y = case.vitals[nme][gi]
            if np.isfinite(y).mean() < MIN_VITAL_COVERAGE:
                return None
            cols.append(y)
    return np.column_stack(cols)


def lingam_case_stats(Z: np.ndarray, p: int, n_resid: int, rng: np.random.Generator):
    """Pooled-VAR sufficient statistics and a subsample of this case's own VAR(p) residuals."""
    X, Y = lag_matrix(Z, p)
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[ok], Y[ok]
    k = X.shape[1]
    if len(X) < 5 * (k + 1):
        return None, None
    S = suff_stats(X, Y)
    Xd = X - X.mean(0)
    Yd = Y - Y.mean(0)
    B = ols(Xd, Yd)
    R = Yd - Xd @ B
    if len(R) > n_resid:
        R = R[rng.choice(len(R), n_resid, replace=False)]
    return S, R


def varlingam_fit(S: np.ndarray, R: np.ndarray, n_var: int, p: int, max_rows: int, seed: int) -> np.ndarray:
    """VAR-LiNGAM from pooled FE-VAR sufficient statistics and pooled VAR residuals.

    x_t = sum_tau M_tau x_{t-tau} + e_t,  e_t = B0 e_t + eps_t (DirectLiNGAM on the residuals)
    =>  B_tau = (I - B0) M_tau.  Returns E[effect, cause] = sum_{tau=0..p} B_tau[effect, cause].
    """
    import lingam

    M = solve_suff(S, n_var * p, n_var)  # rows lag-major (lag1 vars, lag2 vars, ...), columns = equation
    rng = np.random.default_rng(seed)
    if len(R) > max_rows:
        R = R[rng.choice(len(R), max_rows, replace=False)]
    sd = R.std(0) + 1e-12
    model = lingam.DirectLiNGAM(random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(R / sd)
    B0 = model.adjacency_matrix_ * sd[:, None] / sd[None, :]  # back to residual units
    eye = np.eye(n_var)
    E = B0.copy()
    for lag in range(1, p + 1):
        Mt = M[(lag - 1) * n_var : lag * n_var, :].T  # [equation, variable]
        E += (eye - B0) @ Mt
    return E


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 7. Bootstrap helpers
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def case_bootstrap_weights(n_cases: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_cases, n_boot) multinomial counts: each column resamples the cases with replacement."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_cases, n_boot), dtype=np.float64)
    for b in range(n_boot):
        W[:, b] = np.bincount(rng.integers(0, n_cases, n_cases), minlength=n_cases)
    return W


def ratio_boot(sums: np.ndarray, counts: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray]:
    """Event-weighted mean sum/count and its case-bootstrap replicates (rows of W align with sums)."""
    est = float(sums.sum() / max(counts.sum(), 1e-12))
    num = W.T @ sums
    den = W.T @ counts
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = np.where(den > 0, num / den, np.nan)
    return est, reps


def ci(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    r = reps[np.isfinite(reps)]
    if len(r) < 10:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))


MIN_CASES_FOR_CI: int = 5


def sign_call(est: float, lo: float, hi: float, expected: int, n_cases: float = np.inf) -> str:
    """'correct' / 'wrong' (CI excludes 0), 'ns_right' / 'ns_wrong' (sign only), 'na'.

    With fewer than MIN_CASES_FOR_CI contributing cases the case bootstrap is degenerate, so the call
    is sign-only whatever the interval says."""
    if not np.isfinite(est):
        return "na"
    enough = not (np.isfinite(n_cases) and n_cases < MIN_CASES_FOR_CI)
    sig = enough and np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
    right = np.sign(est) == expected
    if sig:
        return "correct" if right else "wrong"
    return "ns_right" if right else "ns_wrong"


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 8. Per-case worker
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def build_arms(case: CaseData, partner_events: dict[str, EventSet] | None, seed: int,
               arms: tuple[str, ...] = ARMS_V1, partner2_events: dict[str, EventSet] | None = None,
               cross_events: dict[str, dict[str, EventSet]] | None = None) -> dict[str, CaseData]:
    """``cross_events`` (v3): cross arm name -> that derangement's partner action streams (overrides
    ``partner_events`` / ``partner2_events`` for the arms it names)."""
    out = {}
    cross_events = dict(cross_events or {})
    if partner_events is not None:
        cross_events.setdefault("nc_cross", partner_events)
    if partner2_events is not None:
        cross_events.setdefault("nc_cross2", partner2_events)
    for arm in arms:
        if arm == "real":
            out[arm] = case
        elif arm == "nc_random":
            out[arm] = nc_random_times(case, np.random.default_rng(seed))
        elif arm in cross_events:
            out[arm] = nc_cross_patient(case, cross_events[arm])
    return out


def analyse_case(case: CaseData, cfg: ScaleConfig, *, do_pcmci: bool = False, lingam_resid_rows: int = 60,
                 seed: int = 0, methods: tuple[str, ...] = METHODS, opts: BenchOptions = V3,
                 keep_pp: bool = True) -> dict:
    """All per-case outputs of one arm of one case (compact, picklable). ``keep_pp=False`` (v4 runner, arms
    outside ``PP_ARMS``) skips the per-patient Granger fits, whose rows the aggregator drops for those arms
    anyway (identical pooled outputs, less time)."""
    out: dict = {"elig": {}, "events": {}, "xcorr": {}, "granger": {}, "granger_S": {}, "varx_S": {},
                 "pcmci": {}, "lingam": {}, "spec": {}, "st": {}}
    for r in RELATIONS:
        out["elig"][r.rid] = (case.n_usable(r.action, r.event_dir), case.coverage(r.vital))
    if "seqtrial" in methods and opts.seqtrial:
        out["st"] = seqtrial_bank(case, cfg, opts)
    # event designs
    if any(m in methods for m in EVENT_METHODS):
        for r in RELATIONS:
            acc: dict[tuple[str, float], list[float]] = {}
            want_spec = opts.spec_test and r.kind == "response" and case.coverage(r.vital) >= MIN_VITAL_COVERAGE
            spec: dict | None = {} if want_spec else None
            for m, h, v in event_design_rows(case, r, cfg, opts, spec):
                acc.setdefault((m, h), []).append(v)
            for (m, h), vals in acc.items():
                a = np.asarray(vals)
                out["events"][(m, r.rid, h)] = (float(a.sum()), float((a**2).sum()), int(len(a)))
            if spec:
                from clbench.edges.typed_temporal_pooled import split_half_sums

                for o in SPEC_OUTCOMES:
                    sh = split_half_sums(spec["tau"], spec[o], SPEC_MIN_PER_HALF)
                    if sh is not None:
                        out["spec"][(r.rid, o)] = sh
    # xcorr (native rows, v1 lag window)
    if "xcorr_v1" in methods:
        L = int(round(cfg.xcorr_max_lag_sec / cfg.row_stride_sec))
        for a, v in PAIRS:
            ev = case.events.get(a)
            if ev is None or not ev.usable.any():
                continue
            c = xcorr_curve(case.dose_series(a), case.vitals[v], L)
            if c is not None:
                out["xcorr"][(a, v)] = c.astype(XCORR_STORE_DTYPE)
    gi = reg_grid(case, cfg)
    tg = case.t[gi]
    if "granger" in methods:
        for a, v in PAIRS:
            ev = case.events.get(a)
            if ev is None or not ev.usable.any():
                continue
            x = ev.dose_at(tg)
            y = case.vitals[v][gi]
            f = granger_fit(x, y, cfg.granger_p) if keep_pp else None
            if f is not None:
                out["granger"][(a, v)] = (f.n, f.est_xy, f.p_xy, f.est_yx, f.p_yx)
            S = granger_suff(x, y, cfg.granger_p)
            if S is not None:
                out["granger_S"][(a, v)] = S
    if "varx" in methods:
        if VARX_PER_EQUATION:
            d = varx_response_design_v4(case, cfg)
            if d is not None:
                for j, S_ in suff_stats_per_equation(*d).items():
                    out["varx_S"][("resp", VITALS[j])] = S_
        else:
            d = varx_response_design(case, cfg)
            if d is not None:
                out["varx_S"]["resp"] = suff_stats(*d)
        for a, sgn in POLICY_KEYS:
            d = varx_policy_design(case, cfg, a, sgn)
            if d is not None and np.abs(d[1]).sum() > 0:
                out["varx_S"][("pol", a, sgn)] = suff_stats(*d)
    if do_pcmci and "pcmciplus" in methods:
        try:
            out["pcmci"] = pcmci_case(case, cfg, opts=opts)
        except Exception as e:  # noqa: BLE001 - one bad case must not stop the benchmark
            out["pcmci_error"] = repr(e)
    if "varlingam" in methods:
        rng = np.random.default_rng(seed + 17)
        for sname in LINGAM_SETS:
            Z = lingam_matrix(case, cfg, sname)
            if Z is None:
                continue
            S, R = lingam_case_stats(Z, cfg.lingam_p, lingam_resid_rows, rng)
            if S is not None:
                out["lingam"][sname] = (S, R.astype(np.float32))
    return out


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 9. Aggregation across cases
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


class Aggregator:
    """Streams per-case outputs; pooled regressions keep full + case-bootstrap-weighted sums."""

    def __init__(self, pids: list[str], cfg: ScaleConfig, n_boot: int = 200, seed: int = 0,
                 opts: BenchOptions = V3, typed_q: float = 0.05, spec_boot: int = 1000):
        self.opts = opts
        self.typed_q = typed_q
        self.spec_boot = spec_boot
        self.spec_rows: list[tuple] = []
        self.spec_table = None
        self.eb_table = None
        self.typed_table = None
        self.pids = list(pids)
        self.pos = {p: i for i, p in enumerate(self.pids)}
        self.cfg = cfg
        self.W = case_bootstrap_weights(len(self.pids), n_boot, seed)
        self.B = n_boot
        self.elig: list[tuple] = []
        self.events: list[tuple] = []
        self.xcorr: dict[tuple, list] = {}
        self.granger_rows: list[tuple] = []
        self.pcmci_rows: list[tuple] = []
        self.pcmci_done: dict[str, set] = {}
        self.suff_full: dict[tuple, np.ndarray] = {}
        self.suff_boot: dict[tuple, np.ndarray] = {}
        self.suff_count: dict[tuple, int] = {}
        self.lingam_R: dict[tuple, list] = {}
        self.st: dict[tuple, list] = {}  # (arm, action) -> [(case, seqtrial bank)]
        self.errors: list[tuple] = []
        self.nn_workers = 1  # threads for the seqtrial neighbour searches (results do not depend on it)
        self.calib_table = None  # v3: multi-draw calibrated admission, every method (calibrated.csv)
        self.null_table = None  # v3: uncalibrated null discovery per method and null draw

    def _acc(self, key, c: int, S: np.ndarray):
        if key not in self.suff_full:
            self.suff_full[key] = np.zeros_like(S)
            self.suff_boot[key] = np.zeros((self.B, len(S)))
        if len(S) != len(self.suff_full[key]):
            return
        self.suff_full[key] += S
        self.suff_count[key] = self.suff_count.get(key, 0) + 1
        w = self.W[c]
        nz = np.nonzero(w)[0]
        self.suff_boot[key][nz] += w[nz, None] * S[None, :]

    def add(self, pid: str, arm: str, res: dict):
        c = self.pos[pid]
        for rid, (n, cov) in res["elig"].items():
            self.elig.append((arm, rid, c, n, cov))
        for (m, rid, h), (s, ss, n) in res["events"].items():
            self.events.append((arm, m, rid, h, c, s, ss, n))
        for pair, curve in res["xcorr"].items():
            self.xcorr.setdefault((arm, pair), []).append((c, curve))
        for pair, g in res["granger"].items():
            self.granger_rows.append((arm, pair[0], pair[1], c) + tuple(g))
        for pair, S in res["granger_S"].items():
            self._acc((arm, "granger", pair), c, S)
        for key, S in res["varx_S"].items():
            self._acc((arm, "varx", key), c, S)
        if "pcmci_error" in res:
            self.errors.append((arm, pid, "pcmci", res["pcmci_error"]))
        if res["pcmci"] or "pcmci_error" in res:
            self.pcmci_done.setdefault(arm, set()).add(c)
        for (cause, effect), (val, p, det, tau) in res["pcmci"].items():
            self.pcmci_rows.append((arm, cause, effect, c, val, p, det, tau))
        for sname, (S, R) in res["lingam"].items():
            self._acc((arm, "lingam", sname), c, S)
            self.lingam_R.setdefault((arm, sname), []).append((c, R))
        for (rid, o), sh in res.get("spec", {}).items():
            self.spec_rows.append((arm, rid, o, c) + tuple(sh))
        for a, b in res.get("st", {}).items():
            self.st.setdefault((arm, a), []).append((c, b))

    # ── finalisation ──────────────────────────────────────────────────────────────────────────────

    def eligibility(self):
        import pandas as pd

        E = pd.DataFrame(self.elig, columns=["arm", "rid", "case", "n_events", "coverage"])
        E["eligible"] = (E.n_events >= MIN_EVENTS_PER_PATIENT) & (E.coverage >= MIN_VITAL_COVERAGE)
        return E

    def finalize(self, lingam_boot: int = 50, lingam_max_rows: int = 50_000, seed: int = 0, executor=None):
        """Returns (pooled DataFrame, per-patient DataFrame). ``executor`` (optional
        concurrent.futures executor) runs the VAR-LiNGAM bootstrap of each (arm, variable set) in
        parallel; the results are identical to the serial run."""
        import pandas as pd
        from scipy import stats

        cfg = self.cfg
        lingam_futures = {}
        if executor is not None:
            nb = min(lingam_boot, self.B)
            for (arm, sname), lst in sorted(self.lingam_R.items(), key=lambda kv: (_arm_order(kv[0][0]), kv[0][1])):
                lst = sorted(lst, key=lambda x: x[0])
                key = (arm, "lingam", sname)
                idx = np.array([c for c, _ in lst])
                job = (self.suff_full[key], self.suff_boot[key][:nb], [R.astype(float) for _, R in lst],
                       self.W[idx, :nb], len(LINGAM_SETS[sname]), cfg.lingam_p, lingam_max_rows, seed)
                lingam_futures[(arm, sname)] = executor.submit(_lingam_job, job)
        n = len(self.pids)
        E = self.eligibility()
        elig = {(a, r): set(g.case[g.eligible]) for (a, r), g in E.groupby(["arm", "rid"])}
        pooled: list[dict] = []
        pp: list[dict] = []

        rep_store: dict[tuple, tuple[float, np.ndarray, float]] = {}

        def add_pooled(arm, method, r, h, est, reps, n_cases, n_units, extra=None):
            if reps is not None and len(reps) == self.B and not arm.startswith("calib"):
                rep_store[(arm, method, r.rid, h)] = (est, np.asarray(reps, float), n_cases)
            lo, hi = ci(reps)
            d = dict(arm=arm, method=method, rid=r.rid, kind=r.kind, label=r.label, h=h, expected=r.expected,
                     est=est, lo=lo, hi=hi, call=sign_call(est, lo, hi, r.expected, n_cases), n_cases=n_cases,
                     n_units=n_units, n_boot_ok=int(np.isfinite(reps).sum()) if reps is not None else 0)
            if extra:
                d.update(extra)
            pooled.append(d)

        def add_pp(arm, method, r, h, c, est, p, nunits, extra=None):
            if arm not in PP_ARMS:
                return
            d = dict(arm=arm, method=method, rid=r.rid, h=h, case=c, pid=self.pids[c], est=est, p=p, n=nunits,
                     correct=bool(np.sign(est) == r.expected) if np.isfinite(est) else np.nan)
            if extra:
                d.update(extra)
            pp.append(d)

        # (a)(g)(h) event designs
        if self.events:
            ev = pd.DataFrame(self.events, columns=["arm", "method", "rid", "h", "case", "s", "ss", "n"])
            for (arm, m, rid, h), g in ev.groupby(["arm", "method", "rid", "h"]):
                r = REL_BY_ID[rid]
                sums = np.zeros(n)
                cnts = np.zeros(n)
                sums[g.case.values] = g.s.values
                cnts[g.case.values] = g.n.values
                est, reps = ratio_boot(sums, cnts, self.W)
                add_pooled(arm, m, r, h, est, reps, int((cnts > 0).sum()), int(cnts.sum()))
                for row in g.itertuples():
                    if row.n >= MIN_EVENTS_PER_PATIENT:
                        mu = row.s / row.n
                        var = max(row.ss / row.n - mu**2, 0.0) * row.n / (row.n - 1)
                        se = math.sqrt(var / row.n)
                        p = float(2 * stats.t.sf(abs(mu) / se, row.n - 1)) if se > 0 else np.nan
                        add_pp(arm, m, r, h, row.case, mu, p, row.n, dict(se=se))

        # (b) xcorr, v1 style
        L = int(round(cfg.xcorr_max_lag_sec / cfg.row_stride_sec))
        for (arm, pair), lst in sorted(self.xcorr.items()):
            lst = sorted(lst, key=lambda x: x[0])  # reproducible: stack in case order
            idx = np.array([c for c, _ in lst])
            curves = np.vstack([cv for _, cv in lst]).astype(float)
            for r in RELATIONS:
                if (r.action, r.vital) != pair:
                    continue
                keep = np.isin(idx, list(elig.get((arm, r.rid), set())))
                if keep.sum() == 0:
                    continue
                ci_, cv = idx[keep], np.nan_to_num(curves[keep])
                orient = 1 if r.kind == "response" else -1  # policy: cause (vital) leads = negative lag
                modes = [("xcorr_v1", "abs")]
                if self.opts.xcorr_variants:
                    modes += [("xcorr_dir", "dir"), ("xcorr_pos", "pos")]
                Wk = self.W[ci_]
                den = Wk.sum(0)
                ok = den > 0
                bc = (Wk[:, ok].T @ cv) / den[ok, None]
                for mname, mode in modes:
                    for c, curve in zip(ci_, cv):
                        val, lag = xcorr_peak(curve, L, mode, orient)
                        lag_s = orient * lag * cfg.row_stride_sec
                        v1_edge = abs(val) >= 0.15 and abs(lag_s) >= 15.0
                        if mode == "pos":  # v1: positive peak >= 0.15 at |lag| >= 15 s with the cause leading
                            v1_edge = val >= 0.15 and lag_s >= 15.0
                        add_pp(arm, mname, r, 0.0, c, val, np.nan, 1,
                               dict(lag_s=lag_s, cause_leads=lag_s > 0, v1_edge=v1_edge))
                    val, lag = xcorr_peak(cv.mean(0), L, mode, orient)
                    reps = np.full(self.B, np.nan)
                    reps[ok] = [xcorr_peak(b, L, mode, orient)[0] for b in bc]
                    add_pooled(arm, mname, r, 0.0, val, reps, len(ci_), len(ci_),
                               dict(lag_s=orient * lag * cfg.row_stride_sec))

        # (c) Granger per patient + pooled FE
        if self.granger_rows:
            G = pd.DataFrame(self.granger_rows, columns=["arm", "a", "v", "case", "n", "est_xy", "p_xy", "est_yx", "p_yx"])
            for r in RELATIONS:
                col_e, col_p = ("est_xy", "p_xy") if r.kind == "response" else ("est_yx", "p_yx")
                for arm, g in G[(G.a == r.action) & (G.v == r.vital)].groupby("arm"):
                    el = elig.get((arm, r.rid), set())
                    for row in g[g.case.isin(el)].itertuples():
                        add_pp(arm, "granger", r, 0.0, row.case, getattr(row, col_e), getattr(row, col_p), row.n)
        p = cfg.granger_p
        for (arm, meth, key), Sf in self.suff_full.items():
            if meth != "granger":
                continue
            for r in RELATIONS:
                if (r.action, r.vital) != key:
                    continue
                j = 0 if r.kind == "response" else 1
                est = granger_pooled_est(Sf, p)[j]
                reps = np.array([granger_pooled_est(Sb, p)[j] if Sb[-1] > 0 else np.nan for Sb in self.suff_boot[(arm, meth, key)]])
                add_pooled(arm, "granger", r, 0.0, est, reps, self.suff_count[(arm, meth, key)], int(Sf[-1]))

        # (d) VARX pooled
        cols = varx_columns(cfg)
        kx = cols["_total"].stop
        pcols = varx_policy_columns(cfg)
        varx_arms = sorted({k[0] for k in self.suff_full if k[1] == "varx"}, key=_arm_order)
        for arm in varx_arms:  # every arm that ran the VARX (v2: incl. nc_cross2; v3: all K cross draws)
            key = (arm, "varx", "resp")
            if VARX_PER_EQUATION:
                self._varx_per_equation(arm, cols, kx, add_pooled)
            elif key in self.suff_full:
                Sf = self.suff_full[key]
                bf = solve_suff(Sf, kx, len(VITALS))
                bb = [solve_suff(Sb, kx, len(VITALS)) if Sb[-1] > 0 else None for Sb in self.suff_boot[key]]
                for r in RELATIONS:
                    if r.kind != "response":
                        continue
                    ch = f"{r.action}:{'up' if r.event_dir > 0 else 'down'}"
                    vi = VITALS.index(r.vital)
                    for h in cfg.horizons:
                        est = varx_step_response(bf, cfg, ch, h)[vi]
                        reps = np.array([varx_step_response(b, cfg, ch, h)[vi] if b is not None else np.nan for b in bb])
                        add_pooled(arm, "varx", r, h, est, reps, self.suff_count[key], int(Sf[-1]))
            for r in RELATIONS:
                if r.kind != "policy":
                    continue
                key = (arm, "varx", ("pol", r.action, r.event_dir))
                if key not in self.suff_full:
                    continue
                Sf = self.suff_full[key]
                j = pcols.index(f"trend:{r.vital}")
                jl = pcols.index(f"level:{r.vital}")
                bf = solve_suff(Sf, len(pcols), 1)
                bb = [solve_suff(Sb, len(pcols), 1) if Sb[-1] > 0 else None for Sb in self.suff_boot[key]]
                reps = np.array([b[j, 0] if b is not None else np.nan for b in bb])
                reps_l = np.array([b[jl, 0] if b is not None else np.nan for b in bb])
                lo_l, hi_l = ci(reps_l)
                add_pooled(arm, "varx", r, 0.0, float(bf[j, 0]), reps, self.suff_count[key], int(Sf[-1]),
                           dict(level_coef=float(bf[jl, 0]), level_lo=lo_l, level_hi=hi_l))

        # (e) PCMCI+
        if self.pcmci_rows:
            P = pd.DataFrame(self.pcmci_rows, columns=["arm", "cause", "effect", "case", "val", "p", "det", "tau"])
            for r in RELATIONS:
                for arm, g in P[(P.cause == r.cause) & (P.effect == r.effect)].groupby("arm"):
                    g = g[g.case.isin(elig.get((arm, r.rid), set()))]
                    if len(g) == 0:
                        continue
                    for row in g.itertuples():
                        add_pp(arm, "pcmciplus", r, 0.0, row.case, row.val, row.p, 1, dict(detected=bool(row.det), tau=row.tau))
                    vals = np.zeros(n)
                    cnt = np.zeros(n)
                    vals[g.case.values] = g.val.values
                    cnt[g.case.values] = 1
                    est, reps = ratio_boot(vals, cnt, self.W)
                    add_pooled(arm, "pcmciplus", r, 0.0, est, reps, len(g), len(g),
                               dict(frac_detected=float(g.det.mean()),
                                    frac_det_correct=float((g.det & (np.sign(g.val) == r.expected)).mean()),
                                    frac_det_wrong=float((g.det & (np.sign(g.val) == -r.expected)).mean())))

        # (f) VAR-LiNGAM pooled
        for (arm, sname), lst in sorted(self.lingam_R.items(), key=lambda kv: (_arm_order(kv[0][0]), kv[0][1])):
            lst = sorted(lst, key=lambda x: x[0])  # reproducible: residual rows stacked in case order
            key = (arm, "lingam", sname)
            names = LINGAM_SETS[sname]
            kv = len(names)
            pl = cfg.lingam_p
            idx = np.array([c for c, _ in lst])
            Rs = [R.astype(float) for _, R in lst]
            nb = min(lingam_boot, self.B)
            job = (self.suff_full[key], self.suff_boot[key][:nb], Rs, self.W[idx, :nb], kv, pl, lingam_max_rows, seed)
            if executor is not None and (arm, sname) in lingam_futures:
                Ef, reps_E, err = lingam_futures[(arm, sname)].result()
            else:
                Ef, reps_E, err = _lingam_job(job)
            if Ef is None:
                self.errors.append((arm, "*", "varlingam", err))
                continue
            for r in RELATIONS:
                if r.cause not in names or r.effect not in names:
                    continue
                if sname == "core" and r.action in PRESSORS:
                    continue
                if lingam_owner(r) != sname:  # v4: a relation in two sets is scored once (v3: never happens)
                    continue
                i, j = names.index(r.effect), names.index(r.cause)
                reps = np.full(self.B, np.nan)
                for b, E_ in reps_E.items():
                    reps[b] = E_[i, j]
                add_pooled(arm, "varlingam", r, 0.0, float(Ef[i, j]), reps, len(idx), int(self.suff_full[key][-1]))

        # (i) v3 pooled sequential-trial contrast
        if self.st:
            self._seqtrial(add_pooled, add_pp)

        # Null-calibrated estimates: real minus negative-control estimate of the same method and
        # relation. Both arms share the case-bootstrap weights, so the difference is a paired bootstrap.
        for nc, tag in (("nc_cross", "calib_cross"), ("nc_random", "calib_random")):
            for (arm, m, rid, h), (est, reps, n_c) in list(rep_store.items()):
                if arm != "real" or (nc, m, rid, h) not in rep_store:
                    continue
                est0, reps0, _ = rep_store[(nc, m, rid, h)]
                add_pooled(tag, m, REL_BY_ID[rid], h, est - est0, reps - reps0, n_c, np.nan,
                           dict(nc_est=est0))

        pp_df = pd.DataFrame(pp)
        if not self.opts.is_v1:
            if self.opts.seqtrial:
                self._calibrated_v3(pooled, rep_store)
            else:
                self._typed(pooled, rep_store)
            self._eb(pooled, pp_df)
            self._spec()
        return pd.DataFrame(pooled), pp_df

    # ── v4: VARX response step responses from per-equation sufficient statistics ─────────────────────
    def _varx_per_equation(self, arm, cols, kx, add_pooled) -> None:
        cfg = self.cfg
        keys = [(arm, "varx", ("resp", v)) for v in VITALS]
        if not any(k in self.suff_full for k in keys):
            return
        nv = len(VITALS)

        def beta_of(Ss):
            B = np.zeros((kx, nv))
            for j, S_ in enumerate(Ss):
                if S_ is None or S_[-1] <= 0:
                    B[:, j] = np.nan
                else:
                    B[:, j] = solve_suff(S_, kx, 1)[:, 0]
            return B

        bf = beta_of([self.suff_full.get(k) for k in keys])
        bb = [beta_of([self.suff_boot[k][b] if k in self.suff_boot else None for k in keys]) for b in range(self.B)]
        for r in RELATIONS:
            if r.kind != "response":
                continue
            vi = VITALS.index(r.vital)
            k_ = keys[vi]
            if k_ not in self.suff_full:
                continue
            ch = f"{r.action}:{'up' if r.event_dir > 0 else 'down'}"
            for h in cfg.horizons:
                # the step response of vital vi uses the AR rows of every equation; NaN equations (no rows)
                # are set to 0 in the propagation (their dV do not feed back), as documented in the protocol
                est = varx_step_response(np.nan_to_num(bf), cfg, ch, h)[vi]
                reps = np.array([varx_step_response(np.nan_to_num(b), cfg, ch, h)[vi] if np.isfinite(b[:, vi]).all()
                                 else np.nan for b in bb])
                add_pooled(arm, "varx", r, h, est, reps, self.suff_count[k_], int(self.suff_full[k_][-1]))

    # ── v3: pooled sequential-trial contrast ────────────────────────────────────────────────────────
    def _seqtrial(self, add_pooled, add_pp) -> None:
        from scipy import stats

        cfg = self.cfg
        H = cfg.horizons
        nH = len(H)
        n = len(self.pids)
        fi = {f: i for i, f in enumerate(ST_FEATURES)}
        all_idx = list(range(len(ST_FEATURES)))
        noco = [i for f, i in fi.items() if f not in ("co_up", "co_dn")]  # seqtrial_noco: no co-interventions
        pol = [fi[f] for f in ST_POLICY_FEATURES]
        pol_noco = [fi[f] for f in ST_POLICY_FEATURES if f not in ("co_up", "co_dn")]
        sens = self.opts.sensitivities
        for arm, a in sorted(self.st, key=lambda k: (_arm_order(k[0]), k[1])):
            lst = sorted(self.st[(arm, a)], key=lambda x: x[0])
            ce = np.concatenate([np.full(len(b["tau"]), c, dtype=np.int64) for c, b in lst])
            de = np.concatenate([b["dir"] for _, b in lst]).astype(int)
            Fe = np.vstack([b["Fe"] for _, b in lst]).astype(float)
            Ye = np.vstack([b["Ye"] for _, b in lst]).astype(float)
            cc = np.concatenate([np.full(len(b["tc"]), c, dtype=np.int64) for c, b in lst])
            Fc = np.vstack([b["Fc"] for _, b in lst]).astype(float)
            Yc = np.vstack([b["Yc"] for _, b in lst]).astype(float)
            for r in RELATIONS:
                if r.action != a:
                    continue
                s = r.event_dir
                vi = VITALS.index(r.vital)
                tcol = fi[f"T_{r.vital}"]
                if r.kind == "response":
                    req = [fi[f"L_{r.vital}"], tcol, fi["dose"]]
                    em = (de == s) & np.isfinite(Fe[:, req]).all(1)
                    cm = np.isfinite(Fc[:, req]).all(1)
                    if math.isfinite(r.pre_max):  # v4: pre-level restriction (events and controls)
                        with np.errstate(invalid="ignore"):
                            em &= Fe[:, fi[f"L_{r.vital}"]] < r.pre_max
                            cm &= Fc[:, fi[f"L_{r.vital}"]] < r.pre_max
                    ye, yc = Ye[em][:, vi * nH:(vi + 1) * nH], Yc[cm][:, vi * nH:(vi + 1) * nH]
                    hs = H
                    variants = [("seqtrial", all_idx, True)]
                    if sens:
                        variants += [("seqtrial_nobc", all_idx, False), ("seqtrial_noco", noco, True)]
                else:
                    req = [tcol, fi["dose"]]
                    em = (de == s) & np.isfinite(Fe[:, req]).all(1)
                    cm = np.isfinite(Fc[:, req]).all(1)
                    ye, yc = Fe[em][:, [tcol]], Fc[cm][:, [tcol]]
                    hs = (0.0,)
                    variants = [("seqtrial", pol, True)]
                    if sens:
                        variants += [("seqtrial_nobc", pol, False), ("seqtrial_noco", pol_noco, True)]
                if em.sum() == 0 or cm.sum() < cfg.min_candidates:
                    continue
                ge = ce[em]
                cache: dict[tuple, np.ndarray] = {}
                for name, fidx, bc in variants:
                    key = tuple(fidx)
                    if key not in cache:  # one neighbour search per feature set (corrected + uncorrected)
                        cache[key] = pooled_nn_contrast(Fe[em][:, fidx], ye, ge, Fc[cm][:, fidx], yc, cc[cm], k=ST_K,
                                                        bias_correct="both", workers=self.nn_workers)
                    V = s * cache[key][0 if bc else 1]
                    for j, h in enumerate(hs):
                        v = V[:, j]
                        ok = np.isfinite(v)
                        sums = np.bincount(ge[ok], weights=v[ok], minlength=n)
                        cnts = np.bincount(ge[ok], minlength=n).astype(float)
                        est, reps = ratio_boot(sums, cnts, self.W)
                        add_pooled(arm, name, r, h, est, reps, int((cnts > 0).sum()), int(cnts.sum()))
                        if name != "seqtrial" or arm not in PP_ARMS:
                            continue
                        ss = np.bincount(ge[ok], weights=v[ok] ** 2, minlength=n)
                        for c in np.flatnonzero(cnts >= MIN_EVENTS_PER_PATIENT):
                            k_ = cnts[c]
                            mu = sums[c] / k_
                            var = max(ss[c] / k_ - mu**2, 0.0) * k_ / (k_ - 1)
                            se = math.sqrt(var / k_)
                            p = float(2 * stats.t.sf(abs(mu) / se, k_ - 1)) if se > 0 else np.nan
                            add_pp(arm, "seqtrial", r, h, int(c), float(mu), p, int(k_), dict(se=se))

    # ── v3: multi-draw calibrated admission of every method + the CLBench typed composite ─────────
    def _calibrated_v3(self, pooled: list[dict], rep_store: dict) -> None:
        import pandas as pd

        self.calib_table = calibrated_admission_table(rep_store, self.cfg, q=self.typed_q)
        self.null_table = null_discovery_table(pd.DataFrame(pooled), self.cfg)
        C = self.calib_table
        if C is None or len(C) == 0:
            return
        # calibrated (K-draw) rows into pooled, arm "calib_multi" (real input only)
        for row in C[C.input_arm == "real"].itertuples(index=False):
            d = row._asdict()
            pooled.append(dict(arm="calib_multi", method=d["method"], rid=d["rid"], kind=d["kind"], label=d["label"],
                               h=d["h"], expected=d["expected"], est=d["est"], lo=d["lo"], hi=d["hi"], call=d["call"],
                               n_cases=d["n_cases"], n_units=np.nan, n_boot_ok=d["n_boot_ok"], p_cal=d["p"],
                               p_by=d["p_by"], admitted=d["admitted"], stable_frac=d["stable_frac"]))
        # typed_pooled rows on the real arm too (so the uncalibrated table shows the typed call)
        T = C[(C.method == "typed_pooled") & (C.input_arm == "real")]
        for row in T.itertuples(index=False):
            d = row._asdict()
            pooled.append(dict(arm="real", method="typed_pooled", rid=d["rid"], kind=d["kind"], label=d["label"], h=d["h"],
                               expected=d["expected"], est=d["est"], lo=d["lo"], hi=d["hi"], call=d["call"],
                               n_cases=d["n_cases"], n_units=np.nan, n_boot_ok=d["n_boot_ok"], p_cal=d["p"],
                               p_by=d["p_by"], admitted=d["admitted"], stable_frac=d["stable_frac"]))
        self.typed_table = T.reset_index(drop=True)

    # ── CLBench typed estimator: calibrated admission with BY across relations ───────────────────
    def _typed(self, pooled: list[dict], rep_store: dict) -> None:
        import pandas as pd

        from clbench.edges import typed_temporal_pooled as tp

        rows = []
        # (input arm, arm label in the tables): real data, and two null inputs for the false-admission rate
        for in_arm, out_arm in (("real", "real"), ("nc_random", "nc_random"), ("nc_cross2", "nc_cross")):
            tests, rels = [], []
            for r in RELATIONS:
                h = primary_h(r, "varx", self.cfg)
                k1, k0 = (in_arm, "varx", r.rid, h), ("nc_cross", "varx", r.rid, h)
                if k1 in rep_store and k0 in rep_store:
                    e1, r1, n1 = rep_store[k1]
                    e0, r0, _ = rep_store[k0]
                    tests.append(tp.calibrated_test(e1, r1, e0, r0))
                    rels.append((r, h, n1))
            if not tests:
                continue
            adm, p_by = tp.admit(tests, self.typed_q)
            for (r, h, n1), t, a, pb in zip(rels, tests, adm, p_by):
                right = bool(np.sign(t.est) == r.expected)
                call = ("correct" if right else "wrong") if a else ("ns_right" if right else "ns_wrong")
                if not np.isfinite(t.est):
                    call = "na"
                d = dict(arm=out_arm, method="typed_pooled", rid=r.rid, kind=r.kind, label=r.label, h=h,
                         expected=r.expected, est=t.est, lo=t.lo, hi=t.hi, call=call, n_cases=n1, n_units=np.nan,
                         n_boot_ok=t.n_boot_ok, p_cal=t.p, p_by=float(pb), admitted=bool(a), est_real=t.est_real,
                         est_null=t.est_null, input_arm=in_arm, null_arm="nc_cross")
                pooled.append(d)
                if out_arm == "real":
                    pooled.append({**d, "arm": "calib_cross"})
                rows.append(d)
        self.typed_table = pd.DataFrame(rows)

    # ── patient-level response edges: EB shrinkage of per-patient kink estimates toward the pool ────
    def _eb(self, pooled: list[dict], pp) -> None:
        import pandas as pd

        self.eb_table = eb_patient_table(pd.DataFrame(pooled), pp, self.cfg)

    # ── pre-registered patient-specificity split-half test ──────────────────────────────────────────
    def _spec(self) -> None:
        import pandas as pd

        from clbench.edges import typed_temporal_pooled as tp

        if not self.spec_rows:
            return
        D = pd.DataFrame(self.spec_rows, columns=["arm", "rid", "outcome", "case", "ne", "se", "sse", "nl", "sl", "ssl"])
        # reproducible: the patient bootstrap resamples positions, so fix the order (v2 used arrival order)
        D = D.sort_values(["arm", "rid", "outcome", "case"], kind="stable").reset_index(drop=True)
        rows = []
        for (arm, rid, o), g in D.groupby(["arm", "rid", "outcome"]):
            if arm not in ("real", "nc_cross"):
                continue
            res = tp.patient_specificity(g["ne"], g["se"], g["sse"], g["nl"], g["sl"], g["ssl"], n_boot=self.spec_boot,
                                         seed=int(self.W.shape[1]) + len(rows))
            r = REL_BY_ID[rid]
            rows.append(dict(arm=arm, rid=rid, label=r.label, outcome=o, h=r.horizon_s, **res))
        self.spec_table = pd.DataFrame(rows)


def eb_patient_table(P, pp, cfg: ScaleConfig):
    """Patient-level response edges: per-patient event estimates (kink, matched; primary horizon) shrunk
    toward the pooled estimate of the same method and arm (normal-normal EB, DerSimonian-Laird tau^2;
    posterior SD includes the pooled estimate's bootstrap variance). Admitted: |shrunk| > 1.96 post. SD."""
    import pandas as pd

    from clbench.edges import typed_temporal_pooled as tp

    if pp is None or len(pp) == 0 or "se" not in pp:
        return None
    rows = []
    for arm in ("real", "nc_cross"):
        for m in ("kink", "matched"):
            for r in RELATIONS:
                if r.kind != "response":
                    continue
                h = primary_h(r, m, cfg)
                g = pp[(pp.arm == arm) & (pp.method == m) & (pp.rid == r.rid) & (pp.h == h)]
                q = P[(P.arm == arm) & (P.method == m) & (P.rid == r.rid) & (P.h == h)]
                if len(g) < 5 or len(q) == 0:
                    continue
                q = q.iloc[0]
                mu = float(q.est)
                mu_var = ((q.hi - q.lo) / (2 * 1.959964)) ** 2 if np.isfinite(q.lo) and np.isfinite(q.hi) else None
                e = g.est.to_numpy(float)
                se = g.se.to_numpy(float)
                res = tp.eb_shrink(e, se**2, mu=mu, mu_var=mu_var)
                ok = np.isfinite(res.shrunk)
                shr, psd = res.shrunk[ok], res.post_sd[ok]
                sig = np.abs(shr) > 1.959964 * psd
                right = np.sign(shr) == r.expected
                raw_sig = g.p.to_numpy(float)[ok] < 0.05
                raw_right = np.sign(e[ok]) == r.expected
                rows.append(dict(arm=arm, method=m, rid=r.rid, label=r.label, h=h, n_patients=int(ok.sum()),
                                 pooled_mu=mu, pooled_lo=float(q.lo), pooled_hi=float(q.hi), tau2=res.tau2,
                                 mean_weight_own=float(np.nanmean(res.weight[ok])),
                                 frac_raw_expected=float(raw_right.mean()),
                                 frac_raw_sig_expected=float((raw_sig & raw_right).mean()),
                                 frac_raw_sig_wrong=float((raw_sig & ~raw_right).mean()),
                                 frac_eb_expected=float(right.mean()),
                                 frac_eb_admitted_expected=float((sig & right).mean()),
                                 frac_eb_admitted_wrong=float((sig & ~right).mean())))
    return pd.DataFrame(rows)


def patient_summary(pp, pooled=None):
    """% of eligible patients whose per-patient estimate has the expected sign (+ per-patient tests)."""
    import pandas as pd

    if pp is None or len(pp) == 0:
        return pd.DataFrame()
    rows = []
    for (arm, m, rid, h), g in pp.groupby(["arm", "method", "rid", "h"]):
        r = REL_BY_ID[rid]
        e = g.est.values
        ok = np.isfinite(e)
        corr = np.sign(e[ok]) == r.expected
        pv = g.p.values[ok]
        sig = np.isfinite(pv) & (pv < 0.05)
        d = dict(arm=arm, method=m, rid=rid, h=h, n_patients=int(ok.sum()),
                 frac_correct=float(corr.mean()) if ok.any() else np.nan,
                 frac_sig=float(sig.mean()) if np.isfinite(pv).any() else np.nan,
                 frac_sig_correct=float((sig & corr).mean()) if np.isfinite(pv).any() else np.nan,
                 frac_sig_wrong=float((sig & ~corr).mean()) if np.isfinite(pv).any() else np.nan)
        if m.startswith("xcorr"):
            d["frac_cause_leads"] = float(g.cause_leads.mean())
            d["frac_v1_edge"] = float(g.v1_edge.mean())
            d["frac_v1_edge_correct"] = float((g.v1_edge & g.cause_leads & (np.sign(g.est) == r.expected)).mean())
        if m == "pcmciplus":
            det = g.detected.values.astype(bool)[ok]
            d["frac_sig"] = float(det.mean())
            d["frac_sig_correct"] = float((det & corr).mean())
            d["frac_sig_wrong"] = float((det & ~corr).mean())
        rows.append(d)
    return pd.DataFrame(rows)


def primary_h(r: Relation, method: str, cfg: ScaleConfig) -> float:
    if r.kind == "policy" or method.split("_")[0] not in HORIZON_FAMILIES:
        return 0.0
    return r.horizon_s if r.horizon_s in cfg.horizons else max(cfg.horizons)


def fmt_cell(row, ps_row) -> str:
    if row is None or not np.isfinite(row["est"]):
        return "—"
    mark = {"correct": "**R**", "wrong": "**X**", "ns_right": "r", "ns_wrong": "x"}.get(row["call"], "?")
    def f(x):
        if not np.isfinite(x):
            return "nan"
        a = abs(x)
        return f"{x:.3g}" if a >= 1e-3 or a == 0 else f"{x:.1e}"
    s = f"{mark} {f(row['est'])} [{f(row['lo'])}, {f(row['hi'])}]"
    if ps_row is not None and np.isfinite(ps_row["frac_correct"]):
        s += f"; {100 * ps_row['frac_correct']:.0f}% of {int(ps_row['n_patients'])}"
    return s


def main_table(pooled, psum, cfg: ScaleConfig, arm: str = "real", methods: tuple[str, ...] = METHODS) -> str:
    """Markdown table: relation x method -> call est [CI]; % patients with the expected sign."""
    lines = ["| relation | expected | " + " | ".join(methods) + " |",
             "|---|---|" + "---|" * len(methods)]
    P = pooled[pooled.arm == arm] if len(pooled) else pooled
    S = psum[psum.arm == arm] if len(psum) else psum
    for r in RELATIONS:
        cells = []
        for m in methods:
            h = primary_h(r, m, cfg)
            q = P[(P.method == m) & (P.rid == r.rid) & (P.h == h)] if len(P) else []
            row = q.iloc[0] if len(q) else None
            s = S[(S.method == m) & (S.rid == r.rid) & (S.h == h)] if len(S) else []
            srow = s.iloc[0] if len(s) else None
            cells.append(fmt_cell(row, srow))
        exp = f"{'+' if r.expected > 0 else '-'} (h={int(r.horizon_s)}s)" if r.kind == "response" else ("+" if r.expected > 0 else "-")
        lines.append(f"| {r.rid} {r.label} | {exp} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def negative_control_summary(pooled, psum, cfg: ScaleConfig, methods: tuple[str, ...] = METHODS):
    """Per method and control arm: share of relations whose pooled CI excludes 0 (a false discovery),
    share with a textbook-sign discovery, and per-patient rates (sign agreement should be ~50%).
    ``n_distinct`` / ``n_distinct_sig`` count distinct estimates (direction-agnostic methods report the
    same estimate for the up and down rows of a pair). For ``typed_pooled`` a discovery is an admission."""
    import pandas as pd

    rows = []
    arms = [a for a in ("real", "nc_random", "nc_cross", "calib_cross", "calib_random") if (pooled.arm == a).any()]
    for arm in arms:
        for m in methods:
            sel = []
            for r in RELATIONS:
                h = primary_h(r, m, cfg)
                q = pooled[(pooled.arm == arm) & (pooled.method == m) & (pooled.rid == r.rid) & (pooled.h == h)]
                if len(q):
                    sel.append(q.iloc[0])
            if not sel:
                continue
            Q = pd.DataFrame(sel)
            fin = Q[np.isfinite(Q.lo) & np.isfinite(Q.hi)]
            if m == "typed_pooled":
                sig = fin.admitted.astype(bool)
            else:
                sig = (fin.lo > 0) | (fin.hi < 0)
            key = fin.est.round(12).astype(str) + "|" + fin.lo.round(12).astype(str) + "|" + fin.hi.round(12).astype(str)
            d = dict(arm=arm, method=m, n_relations=len(fin), frac_ci_excludes_0=float(sig.mean()) if len(fin) else np.nan,
                     n_sig=int(sig.sum()), n_distinct=int(key.nunique()),
                     n_distinct_sig=int(key[sig.values].nunique()) if len(fin) else 0,
                     frac_sig_textbook=float((fin.call == "correct").mean()) if len(fin) else np.nan,
                     frac_sig_wrong=float((fin.call == "wrong").mean()) if len(fin) else np.nan)
            if len(psum):
                S = psum[(psum.arm == arm) & (psum.method == m)]
                S = S[[primary_h(REL_BY_ID[x], m, cfg) == hh for x, hh in zip(S.rid, S.h)]]
                if len(S):
                    wts = S.n_patients.values
                    d["pp_frac_textbook_sign"] = float(np.nansum(S.frac_correct.values * wts) / max(wts.sum(), 1))
                    fs = S.frac_sig.values
                    d["pp_frac_sig"] = float(np.nansum(np.nan_to_num(fs) * wts) / max(wts[np.isfinite(fs)].sum(), 1)) if np.isfinite(fs).any() else np.nan
            rows.append(d)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 10. v3: arm ordering, parallel VAR-LiNGAM job, multi-draw calibrated admission, null rates, leaderboard
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

_ARM_RANK: dict[str, int] = {a: i for i, a in enumerate(ARMS_V3)}


def _arm_order(arm: str) -> tuple[int, str]:
    return (_ARM_RANK.get(arm, 99), arm)


def _lingam_job(job) -> tuple[np.ndarray | None, dict, str]:
    """VAR-LiNGAM on the pooled statistics of one (arm, variable set) and its first ``nb`` case-bootstrap
    replicates (identical to the serial loop; picklable for a process pool)."""
    S_full, S_boot, Rs, Widx, kv, pl, max_rows, seed = job
    try:
        Ef = varlingam_fit(S_full, np.vstack(Rs), kv, pl, max_rows, seed)
    except Exception as e:  # noqa: BLE001
        return None, {}, repr(e)
    reps_E: dict[int, np.ndarray] = {}  # keyed by bootstrap column so arms stay paired
    for b in range(S_boot.shape[0]):
        w = Widx[:, b].astype(int)
        if w.sum() == 0:
            continue
        Rb = np.vstack([Rs[i] for i in range(len(Rs)) for _ in range(w[i])])
        try:
            reps_E[b] = varlingam_fit(S_boot[b], Rb, kv, pl, max_rows, seed + b + 1)
        except Exception:  # noqa: BLE001
            continue
    return Ef, reps_E, ""


# Response base of the typed edge. The simulator gate (runs/iclr/closed_loop_v3/sim_validation/GATE.md,
# written before the validation results) required seqtrial to have <= 10% inert cells with a CI excluding
# 0 and no wrong-signed active cell; it failed (2 s: 6/24 inert cells, 0 wrong; 30 s: 19/24, 2 wrong),
# so by the pre-stated rule the typed edge keeps the VARX base (v2 continuity) and seqtrial is scored as a
# method only.
TYPED_RESPONSE_BASE: str = "varx"


def typed_base(r: Relation) -> str:
    """Base estimator of the v3 CLBench typed edge (``TYPED_RESPONSE_BASE`` for response relations, the
    VARX policy equations for policy relations)."""
    return TYPED_RESPONSE_BASE if r.kind == "response" else "varx"


def _admit_family(label: str, in_arm: str, method: str, fam: list, q: float, min_stable: float) -> list[dict]:
    from clbench.edges import typed_temporal_pooled as tp

    if not fam:
        return []
    tests = [t for *_, t in fam]
    enough = [n1 >= MIN_CASES_FOR_CI for (_, _, n1, _) in fam]
    adm, p_by = tp.admit_multi(tests, q, min_stable, eligible=enough)
    rows = []
    for (r, h, n1, t), a, pb in zip(fam, adm, p_by):
        right = bool(np.sign(t.est) == r.expected)
        if not np.isfinite(t.est):
            call = "na"
        elif a:
            call = "correct" if right else "wrong"
        else:
            call = "ns_right" if right else "ns_wrong"
        rows.append(dict(input=label, input_arm=in_arm, method=method, rid=r.rid, kind=r.kind, label=r.label, h=h,
                         expected=r.expected, est=t.est, lo=t.lo, hi=t.hi, se=t.se, se_boot=t.se_boot,
                         se_draw=t.se_draw, p=t.p, p_by=float(pb), stable_frac=t.stable_frac, n_draws=t.n_draws,
                         admitted=bool(a), call=call, est_input=t.est_real, est_null_mean=t.est_null_mean,
                         null_sd=t.null_sd, n_cases=n1, n_boot_ok=t.n_boot_ok,
                         base=typed_base(r) if method == "typed_pooled" else method))
    return rows


def calibrated_admission_table(rep_store: dict, cfg: ScaleConfig, q: float = 0.05, min_stable: float | None = None,
                               cross_arms: tuple[str, ...] = CROSS_ARMS):
    """The v3 admission rule applied to every method (and the typed composite) for three kinds of input:

    * ``real``: real data calibrated against all K cross draws (the benchmark result);
    * ``nc_random``: the random-time null as input, calibrated against the K cross draws;
    * ``loo:<arm>``: each cross draw in turn as input, calibrated against the other K - 1 draws
      (null vs null; admissions are false admissions).

    ``rep_store`` maps (arm, method, rid, h) -> (est, case-bootstrap replicates, n_cases); every arm uses
    the same bootstrap weights. Family for BY = one method's relations at their primary horizons."""
    import pandas as pd

    from clbench.edges import typed_temporal_pooled as tp

    ms = tp.MIN_STABLE if min_stable is None else min_stable
    crosses = [a for a in cross_arms if any(k[0] == a for k in rep_store)]
    if len(crosses) < 2:
        return None
    methods = sorted({k[1] for k in rep_store if k[0] == "real"})
    inputs = [("real", "real", crosses), ("nc_random", "nc_random", crosses)]
    inputs += [(f"loo:{a}", a, [c for c in crosses if c != a]) for a in crosses]
    rows: list[dict] = []
    for label, in_arm, nulls in inputs:
        by_m: dict[tuple[str, str], tuple] = {}
        for m in methods:
            fam = []
            for r in RELATIONS:
                h = primary_h(r, m, cfg)
                k1 = (in_arm, m, r.rid, h)
                if k1 not in rep_store:
                    continue
                e1, r1, n1 = rep_store[k1]
                ns = [rep_store[(a, m, r.rid, h)] for a in nulls if (a, m, r.rid, h) in rep_store]
                t = tp.calibrated_test_multi(e1, r1, [x[0] for x in ns], [x[1] for x in ns])
                fam.append((r, h, n1, t))
                by_m[(m, r.rid)] = (r, h, n1, t)
            rows += _admit_family(label, in_arm, m, fam, q, ms)
        fam = [by_m[(typed_base(r), r.rid)] for r in RELATIONS if (typed_base(r), r.rid) in by_m]
        rows += _admit_family(label, in_arm, "typed_pooled", fam, q, ms)
    return pd.DataFrame(rows)


def null_discovery_table(P, cfg: ScaleConfig, methods: tuple[str, ...] | None = None):
    """Uncalibrated discoveries per method and arm: relations whose single-arm case-bootstrap CI excludes 0
    (``sign_call`` in correct / wrong, i.e. with >= MIN_CASES_FOR_CI contributing cases), also counted over
    distinct estimates (direction-agnostic methods give one estimate for the up and down rows). On a null
    arm every discovery is false."""
    import pandas as pd

    rows = []
    arms = [a for a in ARMS_V3 if (P.arm == a).any()]
    methods = methods or tuple(sorted(set(P.method) - {"typed_pooled"}))
    for arm in arms:
        Q = P[P.arm == arm]
        for m in methods:
            if m == "typed_pooled":
                continue
            sel = []
            for r in RELATIONS:
                q_ = Q[(Q.method == m) & (Q.rid == r.rid) & (Q.h == primary_h(r, m, cfg))]
                if len(q_):
                    sel.append(q_.iloc[0])
            if not sel:
                continue
            F = pd.DataFrame(sel)
            F = F[np.isfinite(F.est.astype(float))]
            sig = F.call.isin(["correct", "wrong"]).to_numpy()
            key = (F.est.astype(float).round(12).astype(str) + "|" + F.lo.astype(float).round(12).astype(str)
                   + "|" + F.hi.astype(float).round(12).astype(str)).to_numpy()
            rows.append(dict(arm=arm, method=m, n_relations=len(F), n_sig=int(sig.sum()),
                             n_sig_textbook=int((F.call == "correct").sum()), n_sig_wrong=int((F.call == "wrong").sum()),
                             n_distinct=int(len(set(key))), n_distinct_sig=int(len(set(key[sig])))))
    return pd.DataFrame(rows)


def leaderboard(calib, nulls, methods: tuple[str, ...]):
    """One line per method. ``score`` = calibrated correct-sign admissions minus wrong-sign admissions
    (real data, v3 rule); null false-admission rates of the same rule: cross-draw leave-one-out
    (mean admissions per held-out draw, and admissions / tests) and random-time input; plus the
    uncalibrated counts (single-arm CI excluding 0) on real data and on the null arms."""
    import pandas as pd

    rows = []
    for m in methods:
        C = calib[calib.method == m] if calib is not None and len(calib) else pd.DataFrame()
        d: dict = dict(method=m)
        if len(C):
            real = C[C.input == "real"]
            d.update(n_relations=len(real), cal_correct=int((real.call == "correct").sum()),
                     cal_wrong=int((real.call == "wrong").sum()))
            d["score"] = d["cal_correct"] - d["cal_wrong"]
            loo = C[C.input.str.startswith("loo:")]
            per = loo.groupby("input").admitted.sum()
            d.update(null_cross_fa_per_draw=float(per.mean()) if len(per) else np.nan,
                     null_cross_fa_max=int(per.max()) if len(per) else 0,
                     null_cross_fa_rate=float(loo.admitted.mean()) if len(loo) else np.nan,
                     null_cross_draws_any=int((per > 0).sum()), null_cross_draws=int(len(per)))
            rnd = C[C.input == "nc_random"]
            d.update(null_random_fa=int(rnd.admitted.sum()), null_random_fa_rate=float(rnd.admitted.mean()) if len(rnd) else np.nan)
        N = nulls[nulls.method == m] if nulls is not None and len(nulls) else pd.DataFrame()
        if len(N):
            re = N[N.arm == "real"]
            cr = N[N.arm.isin(CROSS_ARMS)]
            rn = N[N.arm == "nc_random"]
            if len(re):
                d.update(uncal_correct=int(re.n_sig_textbook.iloc[0]), uncal_wrong=int(re.n_sig_wrong.iloc[0]))
            if len(cr):
                d.update(uncal_null_cross_sig_mean=float(cr.n_sig.mean()), uncal_null_cross_sig_min=int(cr.n_sig.min()),
                         uncal_null_cross_sig_max=int(cr.n_sig.max()),
                         uncal_null_cross_distinct_sig_mean=float(cr.n_distinct_sig.mean()),
                         uncal_null_cross_distinct_mean=float(cr.n_distinct.mean()))
            if len(rn):
                d.update(uncal_null_random_sig=int(rn.n_sig.iloc[0]), uncal_null_random_distinct_sig=int(rn.n_distinct_sig.iloc[0]),
                         uncal_null_random_distinct=int(rn.n_distinct.iloc[0]))
        rows.append(d)
    return pd.DataFrame(rows)
