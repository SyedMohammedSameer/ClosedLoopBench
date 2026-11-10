"""
CLBench v2 patient graphs (serialized as schema 3.0).

A v2 patient graph carries two kinds of content, kept apart on purpose:

* **Patient-specific evidence.** Concept x persistence-tier nodes built exactly as
  :mod:`clbench.kg_construct` builds them, every activation run as an *episode* with its exact steps
  and times, action (intervention) nodes with every timestamped set-point change, and variable nodes.
* **Population-validated typed relations.** An edge exists only for a relation admitted at the
  population level by the wave-3 typed estimators (policy / response / association). It carries the
  population sign, estimate and CI and an identification status, never a patient-specific weight (the
  pre-registered patient-specificity test found none). What is patient-specific about an edge is its
  **evidence pointers**: the exposed outcome events in this patient (episode pairs, or action event and
  outcome window) with their lags, in the population estimator's exposure / lag structure. A relation is
  instantiated in a patient iff it has at least one pointer.

Step grid: VitalDB 30 s windows at a 15 s stride (step ``i`` covers ``[ws_i, ws_i + 30)``). Concepts are
tri-state series (1 on, 0 off, -1 unobserved). An onset at step ``i`` is ``S[i] = 1`` with
``S[i-1] = S[i-2] = 0`` (both observed), as in :mod:`clbench.edges.typed_temporal`.

The pure array functions (``*_pairs``) are shared by the graph builder and the null evaluation, so a null
graph is built by exactly the same rule as a real one.

``schema.py`` is not changed: v3 models live here and reuse ``KGNode`` / ``InterventionNode``. The
top-level ``nodes`` / ``interventions`` / ``edges`` names are the ones the FastAPI loader reads.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from clbench import kg_construct as kc
from clbench.edges import typed_temporal as tt
from clbench.persistence import CalibrationConfig, TierThresholds, find_activation_runs
from clbench.schema import InterventionNode, KGNode, PersistenceTier

SCHEMA_VERSION = "3.0"
STEP_SEC = 15.0
WINDOW_SEC = 30.0

EdgeKind = Literal["policy", "response", "association"]
IdStatus = Literal["identified", "sign-unresolved", "not identified"]

# v1 registry codes (clbench.cbm.CONCEPT_REGISTRY); the EtCO2 concepts have no SNOMED code in the
# repository, so they get explicit local codes instead of an invented SNOMED id.
SNOMED: dict[str, str] = {
    "tachycardia": "3424008", "bradycardia": "48867003", "tachypnea": "271823003",
    "bradypnea": "426813001", "hypotension": "45007003", "hypertension": "38341003",
    "hypercapnia": "local:hypercapnia", "hypocapnia": "local:hypocapnia",
}
CONCEPT_MODALITY: dict[str, str] = {
    "tachycardia": "NUM:hr", "bradycardia": "NUM:hr", "hypotension": "NUM:map", "hypertension": "NUM:map",
    "tachypnea": "NUM:rr", "bradypnea": "NUM:rr", "hypercapnia": "NUM:etco2", "hypocapnia": "NUM:etco2",
}
# concept-level action name (typed estimator) -> event-table agent
ACTION_AGENT: dict[str, str] = {"propofol": "propofol", "remifentanil": "remifentanil",
                                "phenylephrine": "phenylephrine", "vent_rr": "ventilator"}
EXCLUDE_ZERO: dict[str, bool] = {"propofol": True, "remifentanil": True, "phenylephrine": False,
                                 "vent_rr": True}
VITAL_UNITS: dict[str, str] = {"hr": "bpm", "map": "mmHg", "etco2": "mmHg", "rr": "/min"}

# Lag windows in steps, exactly the wave-3 exposure structure.
ASSOC_STATE_LAG = 2
ASSOC_ONSET = (2, 40)
POLICY_STATE_LAG = 1
POLICY_ONSET = (1, 40)
RESPONSE_LAG = (5, 40)
ONSET_BIN_LABELS: tuple[tuple[int, int, str], ...] = (
    (1, 4, "onset_0-1min"), (5, 12, "onset_1-3min"), (13, 40, "onset_3-10min"))
# Vital-level pointer windows (30 s numerics, seconds relative to the event time t).
VITAL_PRE_EXCL_SEC = 180.0  # ITT: no change of the action in [t - 180, t)  (30 s scale excl_pre_sec)
VITAL_HALF_SEC = 15.0


def onset_label(d: int) -> str:
    for lo, hi, lab in ONSET_BIN_LABELS:
        if lo <= d <= hi:
            return lab
    return ""


# ---------------------------------------------------------------------------
# Models (schema 3.0)
# ---------------------------------------------------------------------------


class Episode(BaseModel):
    """One activation run of a concept: the unit of patient-specific evidence."""

    episode_id: str
    concept: str
    node_id: str
    tier: PersistenceTier
    record_id: str
    start_step: int
    n_steps: int
    start_sec: float
    end_sec: float
    onset_valid: bool = Field(description="Run starts after two observed-off steps (wave-3 onset rule)")


class VariableNode(BaseModel):
    node_id: str
    variable: str
    unit: str
    n_observed_windows: int


class PopulationRelation(BaseModel):
    """A population-admitted typed relation, as estimated by the wave-3 typed estimators."""

    relation_id: str
    edge_type: EdgeKind
    level: Literal["concept", "vital"]
    source: str
    target: str
    source_kind: Literal["concept", "action", "variable"]
    target_kind: Literal["concept", "action", "variable"]
    action: str | None = None
    direction: Literal["up", "down"] | None = None
    sign: int
    estimand: str
    estimate: float
    ci_lo: float
    ci_hi: float
    lag_steps: tuple[int, int] | None = None
    horizon_sec: float | None = None
    identification_status: IdStatus
    identification_basis: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    source_file: str


class EvidencePair(BaseModel):
    """One exposed outcome event of a relation in this patient."""

    source_ref: str  # episode id | "act<i>.<j>" (interventions[i].actions[j]) | "var:<name>"
    target_ref: str
    lag_sec: float
    exposure: str
    source_window_sec: tuple[float, float] | None = None
    target_window_sec: tuple[float, float] | None = None


class TypedEdge(BaseModel):
    edge_id: str
    relation_id: str
    edge_type: EdgeKind
    source_node_id: str
    target_node_id: str
    sign: int
    identification_status: IdStatus
    population_estimate: float
    population_ci: tuple[float, float]
    estimand: str
    pointers: list[EvidencePair]


class BeatEpisode(BaseModel):
    """One activation run of a beat-level concept (schema 3.1 fast layer), with pointers into the beat table."""

    episode_id: str  # "b<k>.<r>"
    concept: str
    node_id: str
    tier: PersistenceTier
    record_id: str
    start_step: int
    n_steps: int
    start_sec: float
    end_sec: float
    beat_start_idx: int = Field(description="First row of the beats table in the episode's windows")
    beat_end_idx: int = Field(description="Last row (inclusive) of the beats table in the episode's windows")
    n_beats: int
    statistic: str
    statistic_median: float | None = None
    beats_file: str = ""


class FastRelation(BaseModel):
    """A population-level mechanistic fast relation and whether it is admitted for per-patient testing."""

    relation_id: str
    source: str
    target: str
    status: Literal["admitted", "abstained"]
    estimand: str
    unit: str = ""
    basis: str
    population_reference: dict[str, Any] = Field(default_factory=dict)


class FastPointer(BaseModel):
    """One 5-min epoch supporting a fast edge: time range, beat rows, per-epoch estimate, CI and p-value."""

    epoch_start_sec: float
    epoch_end_sec: float
    beat_start_idx: int
    beat_end_idx: int
    estimate: float
    ci_lo: float
    ci_hi: float
    p_value: float
    p_bh: float
    n_beats: int
    n_breaths: int | None = None  # F1 / F2: breaths in the epoch
    n_blocks: int | None = None  # F4: 31-beat bootstrap blocks in the epoch
    clinical: dict[str, float] = Field(default_factory=dict)


class FastEdge(BaseModel):
    edge_id: str
    relation_id: str
    edge_type: Literal["mechanistic_fast"] = "mechanistic_fast"
    source_node_id: str
    target_node_id: str
    estimand: str
    unit: str
    patient_estimate: float
    patient_ci: tuple[float, float]
    patient_p: float
    patient_p_holm: float
    n_epochs_tested: int
    expected_sign: int | None = None  # F4: -1 (one-sided test); None for magnitude couplings
    note: str = ""
    pointers: list[FastPointer]


class PatientKGv3(BaseModel):
    schema_version: Literal["3.0", "3.1"] = SCHEMA_VERSION
    patient_id: str
    cohort: str
    nodes: list[KGNode] = Field(default_factory=list)
    episodes: list[Episode] = Field(default_factory=list)
    interventions: list[InterventionNode] = Field(default_factory=list)
    variables: list[VariableNode] = Field(default_factory=list)
    relations: list[PopulationRelation] = Field(default_factory=list,
                                                description="Population relations instantiated here")
    edges: list[TypedEdge] = Field(default_factory=list)
    beat_episodes: list[BeatEpisode] = Field(default_factory=list, description="Schema 3.1 fast layer")
    fast_relations: list[FastRelation] = Field(default_factory=list, description="Schema 3.1 fast registry")
    fast_edges: list[FastEdge] = Field(default_factory=list, description="Schema 3.1 mechanistic fast edges")
    build_metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_refs(self) -> list[str]:
        """Dangling node / episode / action references (empty list = consistent)."""
        errs: list[str] = []
        node_ids = {n.node_id for n in self.nodes} | {n.node_id for n in self.interventions} | {
            v.node_id for v in self.variables}
        eps = {e.episode_id: e for e in self.episodes}
        rels = {r.relation_id for r in self.relations}
        for e in self.beat_episodes:
            if e.node_id not in node_ids:
                errs.append(f"beat episode {e.episode_id}: node {e.node_id} missing")
            if not (0 <= e.beat_start_idx <= e.beat_end_idx):
                errs.append(f"beat episode {e.episode_id}: bad beat range")
        frels = {r.relation_id: r for r in self.fast_relations}
        for fe in self.fast_edges:
            if fe.source_node_id not in node_ids or fe.target_node_id not in node_ids:
                errs.append(f"fast edge {fe.edge_id}: dangling endpoint")
            r = frels.get(fe.relation_id)
            if r is None or r.status != "admitted":
                errs.append(f"fast edge {fe.edge_id}: relation {fe.relation_id} not admitted")
            if not fe.pointers:
                errs.append(f"fast edge {fe.edge_id}: no evidence pointer")
            for p in fe.pointers:
                if not (0 <= p.beat_start_idx <= p.beat_end_idx) or not p.epoch_start_sec < p.epoch_end_sec:
                    errs.append(f"fast edge {fe.edge_id}: bad pointer range")
        for e in self.episodes:
            if e.node_id not in node_ids:
                errs.append(f"episode {e.episode_id}: node {e.node_id} missing")
        for ed in self.edges:
            if ed.source_node_id not in node_ids or ed.target_node_id not in node_ids:
                errs.append(f"edge {ed.edge_id}: dangling endpoint")
            if ed.relation_id not in rels:
                errs.append(f"edge {ed.edge_id}: unknown relation {ed.relation_id}")
            if not ed.pointers:
                errs.append(f"edge {ed.edge_id}: no evidence pointer")
            for p in ed.pointers:
                for ref in (p.source_ref, p.target_ref):
                    if ref.startswith("e") and ref not in eps:
                        errs.append(f"edge {ed.edge_id}: unknown episode {ref}")
                    elif ref.startswith("act"):
                        i, j = (int(x) for x in ref[3:].split("."))
                        if i >= len(self.interventions) or j >= len(self.interventions[i].actions):
                            errs.append(f"edge {ed.edge_id}: unknown action {ref}")
        return errs


# ---------------------------------------------------------------------------
# Tiers and episodes (exactly kg_construct / persistence)
# ---------------------------------------------------------------------------


def thresholds_for(concept: str, calibration: CalibrationConfig) -> TierThresholds:
    """kg_construct's thresholds, including its fallback for uncalibrated concepts (no warning spam)."""
    th = calibration.concepts.get(concept)
    if th is not None:
        return th
    return TierThresholds(concept=concept, transient_max_sec=60.0, episodic_max_sec=600.0,
                          activation_threshold=0.5, n_runs_observed=0, source="fallback")


def run_tiers(lengths: np.ndarray, th: TierThresholds, stride: float = STEP_SEC) -> list[PersistenceTier]:
    """Tier of runs of ``lengths`` windows: find_activation_runs' duration n * stride, then classify."""
    return [th.classify(float(n) * stride) for n in np.asarray(lengths)]


def valid_onsets(S: np.ndarray) -> np.ndarray:
    """Steps i with S[i] = 1, S[i-1] = S[i-2] = 0 (1-D tri-state series of one case)."""
    S = np.asarray(S)
    if len(S) < 3:
        return np.zeros(0, dtype=np.int64)
    i = np.arange(2, len(S))
    ok = (S[2:] == 1) & (S[1:-1] == 0) & (S[:-2] == 0)
    return i[ok]


def runs_of(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, lengths) of runs of S == 1."""
    ep = tt.episodes(np.asarray(S))
    if not ep:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = np.array(ep, dtype=np.int64)
    return a[:, 0], a[:, 1]


def run_index_at(starts: np.ndarray, lengths: np.ndarray, T: int) -> np.ndarray:
    """(T,) index of the run covering each step, -1 where off."""
    out = np.full(T, -1, dtype=np.int64)
    for k, (s, n) in enumerate(zip(starts, lengths)):
        out[s:s + n] = k
    return out


# ---------------------------------------------------------------------------
# Evidence pairs on arrays (shared by builder and null evaluation)
# ---------------------------------------------------------------------------


@dataclass
class Pairs:
    """Exposed outcome events: source item index, target item index, lag (steps), exposure label.

    For a concept endpoint the item index is the run index of that concept; for an action endpoint it
    is the step (bin) of the change."""

    src: np.ndarray
    tgt: np.ndarray
    lag: np.ndarray
    label: list[str]

    def __len__(self) -> int:
        return len(self.src)

    @staticmethod
    def empty() -> Pairs:
        z = np.zeros(0, np.int64)
        return Pairs(z, z, z, [])


def _source_exposures(src_S: np.ndarray, targets: np.ndarray, state_lag: int, onset_rng: tuple[int, int]
                      ) -> Pairs:
    """For each target step t: the source run covering t - state_lag ('state'), and source runs starting
    at a valid onset s with onset_rng[0] <= t - s <= onset_rng[1] (onset bin label). One pair per
    (source run, target); labels are joined with '+' when both apply. ``tgt`` holds the target step."""
    T = len(src_S)
    st, ln = runs_of(src_S)
    if len(st) == 0 or len(targets) == 0:
        return Pairs.empty()
    cover = run_index_at(st, ln, T)
    ons = valid_onsets(src_S)
    run_of_onset = cover[ons] if len(ons) else np.zeros(0, np.int64)
    out: dict[tuple[int, int], list] = {}
    for t in np.asarray(targets, dtype=np.int64):
        j = t - state_lag
        if 0 <= j < T and cover[j] >= 0:
            k = int(cover[j])
            out.setdefault((k, int(t)), [int(t - st[k]), []])[1].append("state")
        if len(ons):
            lo = np.searchsorted(ons, t - onset_rng[1], "left")
            hi = np.searchsorted(ons, t - onset_rng[0], "right")
            for q in range(lo, hi):
                s = int(ons[q])
                k = int(run_of_onset[q])
                rec = out.setdefault((k, int(t)), [int(t - st[k]), []])
                rec[1].append(onset_label(int(t - s)))
    if not out:
        return Pairs.empty()
    keys = sorted(out)
    return Pairs(np.array([k[0] for k in keys], np.int64), np.array([k[1] for k in keys], np.int64),
                 np.array([out[k][0] for k in keys], np.int64), ["+".join(out[k][1]) for k in keys])


def association_pairs(S_a: np.ndarray, S_b: np.ndarray) -> Pairs:
    """A -> B: B valid onsets, A 'state' at lag 2 or A valid onset 2-40 steps earlier.
    ``src`` = A run index, ``tgt`` = B run index."""
    ons_b = valid_onsets(S_b)
    p = _source_exposures(S_a, ons_b, ASSOC_STATE_LAG, ASSOC_ONSET)
    if not len(p):
        return p
    stb, lnb = runs_of(S_b)
    cover_b = run_index_at(stb, lnb, len(S_b))
    return Pairs(p.src, cover_b[p.tgt], p.lag, p.label)


def policy_pairs(S_a: np.ndarray, chg_bins: np.ndarray) -> Pairs:
    """A -> action change: change bins i, A 'state' at lag 1 or A valid onset 1-40 steps earlier.
    ``src`` = A run index, ``tgt`` = change bin."""
    return _source_exposures(S_a, np.asarray(chg_bins, np.int64), POLICY_STATE_LAG, POLICY_ONSET)


def response_pairs(chg_bins: np.ndarray, S_b: np.ndarray) -> Pairs:
    """Action change -> B: B valid onset i with 5 <= i - j <= 40 for a change bin j.
    ``src`` = change bin, ``tgt`` = B run index, lag = i - j (bin to onset)."""
    ons_b = valid_onsets(S_b)
    bins = np.sort(np.asarray(chg_bins, np.int64))
    if len(ons_b) == 0 or len(bins) == 0:
        return Pairs.empty()
    stb, lnb = runs_of(S_b)
    cover_b = run_index_at(stb, lnb, len(S_b))
    src, tgt, lag, lab = [], [], [], []
    for i in ons_b:
        lo = np.searchsorted(bins, i - RESPONSE_LAG[1], "left")
        hi = np.searchsorted(bins, i - RESPONSE_LAG[0], "right")
        for q in range(lo, hi):
            j = int(bins[q])
            src.append(j)
            tgt.append(int(cover_b[i]))
            lag.append(int(i - j))
            lab.append("change_" + ("1-3min" if i - j <= 12 else "3-10min"))
    if not src:
        return Pairs.empty()
    return Pairs(np.array(src, np.int64), np.array(tgt, np.int64), np.array(lag, np.int64), lab)


def concept_relation_pairs(rel: PopulationRelation, S: np.ndarray, chg: np.ndarray) -> Pairs:
    """Pairs of a concept-level relation on one case's (T, 8) concept and (T, 4, 2) change arrays."""
    if rel.edge_type == "association":
        a, b = tt.CONCEPTS.index(rel.source), tt.CONCEPTS.index(rel.target)
        return association_pairs(S[:, a], S[:, b])
    ai, di = tt.ACTIONS.index(rel.action), tt.DIRS.index(rel.direction)
    bins = np.nonzero(chg[:, ai, di])[0]
    if rel.edge_type == "policy":
        return policy_pairs(S[:, tt.CONCEPTS.index(rel.source)], bins)
    return response_pairs(bins, S[:, tt.CONCEPTS.index(rel.target)])


def node_level_count(rel: PopulationRelation, pairs: Pairs, tiers: Mapping[str, list[PersistenceTier]]
                     ) -> int:
    """Number of node-level edges a relation instance produces: distinct endpoint-node pairs (tier node of
    each concept run; the single action node of an action endpoint)."""
    if not len(pairs):
        return 0
    if rel.edge_type == "association":
        ta, tb = tiers[rel.source], tiers[rel.target]
        return len({(ta[s], tb[t]) for s, t in zip(pairs.src, pairs.tgt)})
    if rel.edge_type == "policy":
        ta = tiers[rel.source]
        return len({ta[s] for s in pairs.src})
    tb = tiers[rel.target]
    return len({tb[t] for t in pairs.tgt})


def case_tiers(S: np.ndarray, calibration: CalibrationConfig) -> dict[str, list[PersistenceTier]]:
    out = {}
    for k, c in enumerate(tt.CONCEPTS):
        _, ln = runs_of(S[:, k])
        out[c] = run_tiers(ln, thresholds_for(c, calibration))
    return out


# ---------------------------------------------------------------------------
# Relation registry
# ---------------------------------------------------------------------------


def _bh_admitted(fits: Sequence[Mapping], variant: str, q: float) -> dict[str, dict]:
    """Rule 1 (BH at q within family) on one fitted variant, rep 0 — wave-3 ``admissions``."""
    out = {}
    for fam in ("association", "policy", "response"):
        rows = [(r, e) for r in fits if r["family"] == fam and r["variant"] == variant and r["rep"] == 0
                for e in r["edges"]]
        if not rows:
            continue
        rej = tt.bh(np.array([e["p_omega"] for _, e in rows]), q)
        for (r, e), k in zip(rows, rej):
            out[f"{fam}|{e['source']}->{r['outcome']}"] = {**e, "bh": bool(k)}
    return out


def concept_identification(key: str, e: Mapping, h1: Mapping, h2: Mapping, wrong_checks: set[str]
                           ) -> tuple[IdStatus, str]:
    """Pre-registered status rule for a concept-level edge (protocol section 3)."""
    fam = key.split("|")[0]
    sgn = int(np.sign(e["omega"]))
    halves = []
    for name, d in (("H1", h1), ("H2", h2)):
        x = d.get(key)
        halves.append((name, bool(x and x["bh"]), int(np.sign(x["omega"])) if x else 0))
    opposite = [n for n, adm, s in halves if adm and s != sgn]
    both_same = all(adm and s == sgn for _, adm, s in halves)
    desc = ", ".join(f"{n} {'BH-admitted' if a else 'not admitted'} sign {s:+d}" for n, a, s in halves)
    if key in wrong_checks:
        return "sign-unresolved", f"sign contradicts a pre-specified wave-3 sign check; {desc}"
    if opposite:
        return "sign-unresolved", f"half fit(s) {','.join(opposite)} admitted with the opposite sign; {desc}"
    if fam == "response":
        return "not identified", ("concept-level response family failed its pre-specified pharmacological "
                                  f"checks (0/6 correct, 2 wrong); {desc}")
    if both_same:
        return "identified", f"sign replicated in both half fits; {desc}"
    return "not identified", f"sign not replicated in both half fits; {desc}"


def load_concept_relations(results_json: Path, fits_jsonl: Path, q: float = 0.05,
                           tag: str = "vitaldb") -> list[PopulationRelation]:
    res = json.loads(Path(results_json).read_text())
    fits = [json.loads(line) for line in Path(fits_jsonl).read_text().splitlines() if line.strip()]
    h1 = _bh_admitted(fits, "h1", q)
    h2 = _bh_admitted(fits, "h2", q)
    wrong = {f"{c['family']}|{c['source']}->{c['target']}" for c in res.get("sign_checks", [])
             if c.get("call") == "WRONG"}
    rels = []
    for key, e in sorted(res["pooled_edges_full"].items()):
        if not e.get("admitted"):
            continue
        fam = e["family"]
        src, tgt = e["source"], e["outcome"]
        status, basis = concept_identification(key, e, h1, h2, wrong)
        kw: dict[str, Any] = {}
        if fam == "association":
            kw.update(source_kind="concept", target_kind="concept", lag_steps=(2, 40))
        elif fam == "policy":
            an, dn = tgt.rsplit("_", 1)
            kw.update(source_kind="concept", target_kind="action", action=an, direction=dn, lag_steps=(1, 40))
        else:
            an, dn = src.rsplit("_", 1)
            kw.update(source_kind="action", target_kind="concept", action=an, direction=dn, lag_steps=(5, 40))
        rels.append(PopulationRelation(
            relation_id=f"{tag}:{fam}:{src}->{tgt}", edge_type=fam, level="concept", source=src, target=tgt,
            sign=int(np.sign(e["omega"])), estimand="omega: average log hazard ratio over exposed rows "
            "(pooled patient-FE discrete-time hazard)", estimate=float(e["omega"]), ci_lo=float(e["omega_lo"]),
            ci_hi=float(e["omega_hi"]), identification_status=status, identification_basis=basis,
            evidence={"hazard_ratio": float(np.exp(e["omega"])), "p_omega": e["p_omega"],
                      "n_exposed_events": e["n_exposed_events"], "n_event_patients": e["n_event_patients"],
                      "omega_null_corrected": e.get("omega_null_corrected"),
                      "null_max_chi2_T": e.get("null_max_chi2_T"), "null_max_chi2_S": e.get("null_max_chi2_S")},
            source_file=str(results_json), **kw))
    return rels


def vital_identification(calls: Mapping[str, str | None]) -> tuple[IdStatus, str]:
    """calls: scale -> call of the admitted rows ('correct' / 'wrong'), None when not admitted."""
    adm = {s: c for s, c in calls.items() if c is not None}
    desc = "; ".join(f"{s}: {c or 'not admitted'}" for s, c in calls.items())
    if any(c == "wrong" for c in adm.values()):
        return "sign-unresolved", f"admitted with the opposite sign at >= 1 scale ({desc})"
    if any(c == "correct" for c in adm.values()):
        return "identified", f"admitted with the expected sign, never with the opposite ({desc})"
    return "not identified", desc


def load_vital_relations(typed_csvs: Mapping[str, Path], by_null_csvs: Mapping[str, Path] | None = None
                         ) -> list[PopulationRelation]:
    """Vital-level relations admitted (arm real) at >= 1 scale of closed_loop_v2."""
    import pandas as pd

    from clbench.eval import closed_loop_bench as clb

    rel_by_id = {r.rid: r for r in clb.RELATIONS}
    tabs = {s: pd.read_csv(p) for s, p in typed_csvs.items()}
    rob = {s: pd.read_csv(p) for s, p in (by_null_csvs or {}).items()}
    rels = []
    for rid in sorted(rel_by_id):
        rows = {s: df[(df.arm == "real") & (df.rid == rid)] for s, df in tabs.items()}
        rows = {s: r.iloc[0] for s, r in rows.items() if len(r)}
        calls = {s: (str(r["call"]) if bool(r["admitted"]) else None) for s, r in rows.items()}
        if not any(c is not None for c in calls.values()):
            continue
        status, basis = vital_identification(calls)
        R = rel_by_id[rid]
        # headline estimate: the admitted scale with the most cases (30 s over 2 s)
        s0 = max((s for s in rows if calls[s] is not None), key=lambda s: rows[s]["n_cases"])
        r0 = rows[s0]
        ev: dict[str, Any] = {"label": R.label, "expected_sign": R.expected, "headline_scale": s0,
                              "per_scale": {s: {"est": float(r["est"]), "lo": float(r["lo"]), "hi": float(r["hi"]),
                                                "call": str(r["call"]), "admitted": bool(r["admitted"]),
                                                "p_by": float(r["p_by"]), "n_cases": int(r["n_cases"])}
                                            for s, r in rows.items()}}
        for s, df in rob.items():
            d = df[df.rid == rid]
            ev.setdefault("calibrator_robustness", {})[s] = {
                "n_calibrators": int(d["null"].nunique()), "n_admitted": int(d["admitted"].sum()),
                "calls_when_admitted": sorted(d.loc[d["admitted"], "call"].astype(str).unique().tolist())}
        dirn = "up" if R.event_dir > 0 else "down"
        common = dict(relation_id=f"vital:{rid}", edge_type=R.kind, level="vital", action=R.action,
                      direction=dirn, sign=int(R.expected) if calls[s0] == "correct" else -int(R.expected),
                      estimand=("step response of the vital at the horizon per unit set-point change "
                                "(pooled direction-split VARX, cross-patient calibrated)" if R.kind == "response"
                                else "set-point change per unit 2-min vital trend (pooled VARX, calibrated)"),
                      estimate=float(r0["est"]), ci_lo=float(r0["lo"]), ci_hi=float(r0["hi"]),
                      horizon_sec=float(R.horizon_s), identification_status=status, identification_basis=basis,
                      evidence=ev, source_file=",".join(str(p) for p in typed_csvs.values()))
        if R.kind == "response":
            rels.append(PopulationRelation(source=f"{R.action}_{dirn}", target=R.vital, source_kind="action",
                                           target_kind="variable", **common))
        else:
            rels.append(PopulationRelation(source=R.vital, target=f"{R.action}_{dirn}", source_kind="variable",
                                           target_kind="action", **common))
    return rels


# ---------------------------------------------------------------------------
# Vital-level pointers
# ---------------------------------------------------------------------------


def vital_event_pointers(rel: PopulationRelation, t_ev: np.ndarray, before: np.ndarray, after: np.ndarray,
                         t_centres: np.ndarray, vital: np.ndarray) -> list[tuple[int, tuple, tuple]]:
    """ITT events of a vital-level relation with the vital observed in its design windows.

    Returns (event index into the given arrays, source window, target window) per pointer."""
    from clbench.eval import closed_loop_bench as clb

    if len(t_ev) == 0:
        return []
    spec = clb.ACTIONS[rel.action]
    ev = clb.EventSet.build(t_ev, before, after, spec)
    order = np.argsort(np.asarray(t_ev, float), kind="stable")  # EventSet sorts stably by time
    d = 1 if rel.direction == "up" else -1
    sel = ev.usable & (ev.direction == d)
    if not sel.any():
        return []
    tau = ev.t
    clean = ev.count_in(tau - VITAL_PRE_EXCL_SEC, tau, hi_open=True) == 0
    S = clb.SeriesIndex(t_centres, vital)
    out = []
    if rel.edge_type == "response":
        h = float(rel.horizon_sec)
        base = S.mean(tau - 30.0, tau)
        post = S.mean(tau + h - VITAL_HALF_SEC, tau + h + VITAL_HALF_SEC)
        ok = sel & clean & np.isfinite(base) & np.isfinite(post)
        for k in np.nonzero(ok)[0]:
            out.append((int(order[k]), (float(tau[k]), float(tau[k])),
                        (float(tau[k] + h - VITAL_HALF_SEC), float(tau[k] + h + VITAL_HALF_SEC))))
    else:
        early = S.mean(tau - 150.0, tau - 120.0)
        late = S.mean(tau - 30.0, tau)
        ok = sel & clean & np.isfinite(early) & np.isfinite(late)
        for k in np.nonzero(ok)[0]:
            out.append((int(order[k]), (float(tau[k] - 150.0), float(tau[k])), (float(tau[k]), float(tau[k]))))
    return out


# ---------------------------------------------------------------------------
# Event table -> action bins and intervention nodes
# ---------------------------------------------------------------------------


def event_bins(t: np.ndarray, before: np.ndarray, after: np.ndarray, action: str, ws: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Wave-3 assignment of changes to 15 s bins [ws_i + 15, ws_i + 30): (bin, direction index) per event,
    bin = -1 when the event is excluded (no change, start/stop from/to 0 for TCI and ventilator, or
    outside the grid)."""
    t, b, a = (np.asarray(x, float) for x in (t, before, after))
    use = np.abs(a - b) > 0
    if EXCLUDE_ZERO[action]:
        use &= (b > 0) & (a > 0)
    T = len(ws)
    i = np.searchsorted(ws + 15.0, t, side="right") - 1
    ok = use & (i >= 0) & (i < T)
    ok[ok] &= t[ok] < ws[i[ok]] + 30.0
    return np.where(ok, i, -1), np.where(a > b, 0, 1)


def intervention_nodes(events: Sequence[Mapping[str, Any]], record_id: str, pid: str
                       ) -> tuple[list[InterventionNode], list[tuple[int, int]]]:
    """kg_construct.intervention_nodes_from_events plus, per input row, its (node index, action index).
    Rows of unknown kinds are skipped (index (-1, -1))."""
    known = [r for r in events if str(r["kind"]) in kc._EVENT_KINDS]
    nodes = kc.intervention_nodes_from_events(known, record_id)
    for n in nodes:
        n.node_id = f"{pid}:act:{n.agent}:{n.kind.value}:{n.parameter or ''}"
    lookup: dict[tuple, list[tuple[int, int]]] = {}
    for i, n in enumerate(nodes):
        for j, act in enumerate(n.actions):
            lookup.setdefault((n.agent, round(act.t_sec, 6), act.value_before, act.value_after), []).append((i, j))
    idx = []
    for r in events:
        key = (str(r["agent"]), round(float(r["t_sec"]), 6), float(r["value_before"]), float(r["value_after"]))
        lst = lookup.get(key)
        idx.append(lst.pop(0) if lst else (-1, -1))
    return nodes, idx


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


@dataclass
class CaseInput:
    """One case on the 15 s step grid."""

    pid: str
    S: np.ndarray  # (T, 8) tri-state, tt.CONCEPTS order
    ws: np.ndarray  # (T,) window start seconds
    chg: np.ndarray  # (T, 4, 2) wave-3 action changes (tt.ACTIONS x tt.DIRS)
    events: list[dict]  # event-table rows (agent, kind, t_sec, value_before, value_after)
    vitals: dict[str, np.ndarray]  # 30 s numerics per vital, aligned with ws (centres ws + 15)
    record_id: str = ""


def concept_nodes(case: CaseInput, calibration: CalibrationConfig, concepts: Sequence[str] = tt.CONCEPTS,
                  modality: Mapping[str, str] | None = None
                  ) -> tuple[list[KGNode], list[Episode], dict[str, list[str]]]:
    """Concept x tier nodes via kg_construct._nodes_for_concept, plus one Episode per run.
    Returns nodes, episodes and, per concept, the episode id of each run (run order)."""
    modality = modality or CONCEPT_MODALITY
    nodes: list[KGNode] = []
    episodes: list[Episode] = []
    run_ids: dict[str, list[str]] = {}
    rec = case.record_id or case.pid
    for k, c in enumerate(concepts):
        s_col = case.S[:, k]
        probs = (s_col == 1).astype(np.float64)
        th = thresholds_for(c, calibration)
        cat = kc.ConceptCatalogEntry(modality=modality[c], concept_key=c, snomed_code=SNOMED[c],
                                     preferred_term=c.replace("_", " ").title())
        act = kc.ConceptActivation(catalog=cat, probs=probs, window_starts_sec=case.ws,
                                   window_ends_sec=case.ws + WINDOW_SEC, record_id=rec)
        cn = kc._nodes_for_concept(act, th, STEP_SEC)
        by_tier = {}
        for n in cn:
            n.node_id = f"{case.pid}:c:{c}:{n.persistence_tier.value}"
            by_tier[n.persistence_tier] = n
        nodes.extend(cn)
        runs = find_activation_runs(probs, threshold=th.activation_threshold, window_stride_sec=STEP_SEC)
        ons = set(valid_onsets(s_col).tolist())
        ids = []
        for r_i, run in enumerate(runs):
            tier = th.classify(run.duration_sec)
            eid = f"e{k}.{r_i}"
            ids.append(eid)
            episodes.append(Episode(
                episode_id=eid, concept=c, node_id=by_tier[tier].node_id, tier=tier, record_id=rec,
                start_step=int(run.start_window_idx), n_steps=int(run.n_windows),
                start_sec=float(case.ws[run.start_window_idx]),
                end_sec=float(case.ws[run.end_window_idx] + WINDOW_SEC),
                onset_valid=int(run.start_window_idx) in ons))
        run_ids[c] = ids
    return nodes, episodes, run_ids


def build_case_graph(case: CaseInput, relations: Sequence[PopulationRelation], calibration: CalibrationConfig,
                     cohort: str = "vitaldb", metadata: Mapping[str, Any] | None = None) -> PatientKGv3:
    """Build one patient's v2 graph (schema 3.0) from its arrays and the admitted relations."""
    nodes, episodes, run_ids = concept_nodes(case, calibration)
    ep_by_id = {e.episode_id: e for e in episodes}
    inter, row_idx = intervention_nodes(case.events, case.record_id or case.pid, case.pid)
    node_of_agent = {n.agent: n for n in inter}
    # concept-level action bins -> action refs (all events of the bin, in time order)
    bin_refs: dict[tuple[int, int, int], list[str]] = {}
    ev_arr: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}
    for a, agent in ACTION_AGENT.items():
        rows = [(r, ix) for r, ix in zip(case.events, row_idx) if r["agent"] == agent]
        if not rows:
            continue
        t = np.array([float(r["t_sec"]) for r, _ in rows])
        b = np.array([float(r["value_before"]) for r, _ in rows])
        af = np.array([float(r["value_after"]) for r, _ in rows])
        ev_arr[a] = (t, b, af, [ix for _, ix in rows])
        bins, dirs = event_bins(t, b, af, a, case.ws)
        for k in np.argsort(t, kind="stable"):
            if bins[k] >= 0 and rows[k][1][0] >= 0:
                bin_refs.setdefault((tt.ACTIONS.index(a), int(dirs[k]), int(bins[k])), []).append(
                    f"act{rows[k][1][0]}.{rows[k][1][1]}")
    # variable nodes
    variables = []
    for v in ("hr", "map", "etco2"):
        x = case.vitals.get(v)
        if x is not None and np.isfinite(x).any():
            variables.append(VariableNode(node_id=f"{case.pid}:var:{v}", variable=v, unit=VITAL_UNITS[v],
                                          n_observed_windows=int(np.isfinite(x).sum())))
    var_by = {v.variable: v for v in variables}
    t_centres = case.ws + WINDOW_SEC / 2.0

    edges: list[TypedEdge] = []
    used: list[PopulationRelation] = []
    for rel in relations:
        groups: dict[tuple[str, str], list[EvidencePair]] = {}
        if rel.level == "concept":
            pairs = concept_relation_pairs(rel, case.S, case.chg)
            if not len(pairs):
                continue
            for s, t, lag, lab in zip(pairs.src, pairs.tgt, pairs.lag, pairs.label):
                if rel.edge_type == "association":
                    ea, eb = ep_by_id[run_ids[rel.source][s]], ep_by_id[run_ids[rel.target][t]]
                    key = (ea.node_id, eb.node_id)
                    ptr = EvidencePair(source_ref=ea.episode_id, target_ref=eb.episode_id, lag_sec=float(lag) * STEP_SEC,
                                       exposure=lab, source_window_sec=(ea.start_sec, ea.end_sec),
                                       target_window_sec=(eb.start_sec, eb.end_sec))
                    groups.setdefault(key, []).append(ptr)
                elif rel.edge_type == "policy":
                    ea = ep_by_id[run_ids[rel.source][s]]
                    node = node_of_agent.get(ACTION_AGENT[rel.action])
                    refs = bin_refs.get((tt.ACTIONS.index(rel.action), tt.DIRS.index(rel.direction), int(t)), [])
                    if node is None or not refs:
                        continue
                    for ref in refs:
                        groups.setdefault((ea.node_id, node.node_id), []).append(EvidencePair(
                            source_ref=ea.episode_id, target_ref=ref, lag_sec=float(lag) * STEP_SEC, exposure=lab,
                            source_window_sec=(ea.start_sec, ea.end_sec),
                            target_window_sec=(float(case.ws[t] + 15.0), float(case.ws[t] + 30.0))))
                else:
                    eb = ep_by_id[run_ids[rel.target][t]]
                    node = node_of_agent.get(ACTION_AGENT[rel.action])
                    refs = bin_refs.get((tt.ACTIONS.index(rel.action), tt.DIRS.index(rel.direction), int(s)), [])
                    if node is None or not refs:
                        continue
                    for ref in refs:
                        groups.setdefault((node.node_id, eb.node_id), []).append(EvidencePair(
                            source_ref=ref, target_ref=eb.episode_id, lag_sec=float(lag) * STEP_SEC, exposure=lab,
                            source_window_sec=(float(case.ws[s] + 15.0), float(case.ws[s] + 30.0)),
                            target_window_sec=(eb.start_sec, eb.end_sec)))
        else:
            from clbench.eval import closed_loop_bench as clb

            vname = rel.target if rel.edge_type == "response" else rel.source
            agent = clb.ACTIONS[rel.action].agent
            node = node_of_agent.get(agent)
            var = var_by.get(vname)
            rows = [(r, ix) for r, ix in zip(case.events, row_idx) if r["agent"] == agent]
            if node is None or var is None or not rows:
                continue
            t = np.array([float(r["t_sec"]) for r, _ in rows])
            b = np.array([float(r["value_before"]) for r, _ in rows])
            af = np.array([float(r["value_after"]) for r, _ in rows])
            ixs = [ix for _, ix in rows]
            for k, sw, tw in vital_event_pointers(rel, t, b, af, t_centres, case.vitals[vname]):
                i, j = ixs[k]
                if i < 0:
                    continue
                ref = f"act{i}.{j}"
                if rel.edge_type == "response":
                    groups.setdefault((node.node_id, var.node_id), []).append(EvidencePair(
                        source_ref=ref, target_ref=f"var:{vname}", lag_sec=float(rel.horizon_sec),
                        exposure="itt_event", source_window_sec=sw, target_window_sec=tw))
                else:
                    groups.setdefault((var.node_id, node.node_id), []).append(EvidencePair(
                        source_ref=f"var:{vname}", target_ref=ref, lag_sec=0.0, exposure="itt_event_trend_window",
                        source_window_sec=sw, target_window_sec=tw))
        if not groups:
            continue
        used.append(rel)
        for (sn, tn), ptrs in sorted(groups.items()):
            edges.append(TypedEdge(
                edge_id=f"{case.pid}:{rel.relation_id}:{len(edges)}",
                relation_id=rel.relation_id, edge_type=rel.edge_type, source_node_id=sn, target_node_id=tn,
                sign=rel.sign, identification_status=rel.identification_status,
                population_estimate=rel.estimate, population_ci=(rel.ci_lo, rel.ci_hi), estimand=rel.estimand,
                pointers=ptrs))
    kg = PatientKGv3(patient_id=case.pid, cohort=cohort, nodes=nodes, episodes=episodes, interventions=inter,
                     variables=variables, relations=used, edges=edges, build_metadata=dict(metadata or {}))
    return kg


def graph_counts(kg: PatientKGv3 | Mapping) -> dict[str, int]:
    """Size summary of a v3 graph (object or dict)."""
    d = kg.model_dump() if isinstance(kg, PatientKGv3) else kg
    out = {"nodes_concept_tier": len(d["nodes"]), "action_nodes": len(d["interventions"]),
           "variable_nodes": len(d.get("variables", [])), "episodes": len(d.get("episodes", [])),
           "edges_node_level": len(d["edges"]), "relation_instances": len(d.get("relations", [])),
           "pointers": int(sum(len(e["pointers"]) for e in d["edges"])),
           "edges_with_pointers": int(sum(bool(e["pointers"]) for e in d["edges"]))}
    for kind in ("policy", "response", "association"):
        out[f"relations_{kind}"] = sum(r["edge_type"] == kind for r in d.get("relations", []))
        out[f"edges_{kind}"] = sum(e["edge_type"] == kind for e in d["edges"])
    for st in ("identified", "sign-unresolved", "not identified"):
        out[f"relations_{st.replace(' ', '_')}"] = sum(r["identification_status"] == st
                                                       for r in d.get("relations", []))
    for lvl in ("concept", "vital"):
        out[f"relations_{lvl}_level"] = sum(r["level"] == lvl for r in d.get("relations", []))
    # schema 3.1 fast layer (zero for 3.0 graphs)
    beat_nodes = [n for n in d["nodes"] if ":b:" in n["node_id"]]
    out["nodes_beat_concept_tier"] = len(beat_nodes)
    out["nodes_v1_concept_tier"] = len(d["nodes"]) - len(beat_nodes)
    out["beat_episodes"] = len(d.get("beat_episodes", []))
    out["beat_variable_nodes"] = sum(v["variable"] in BEAT_VARIABLES for v in d.get("variables", []))
    out["fast_edges"] = len(d.get("fast_edges", []))
    out["fast_pointers"] = int(sum(len(e["pointers"]) for e in d.get("fast_edges", [])))
    for c in BEAT_CONCEPTS_ALL:
        out[f"beat_nodes_{c}"] = sum(f":b:{c}:" in n["node_id"] for n in beat_nodes)
        out[f"beat_episodes_{c}"] = sum(e["concept"] == c for e in d.get("beat_episodes", []))
    for rid in FAST_RELATION_IDS:
        out[f"fast_edges_{rid.split(':')[1]}"] = sum(e["relation_id"] == rid for e in d.get("fast_edges", []))
    return out


def presence_instances(rel: PopulationRelation, S: np.ndarray, chg: np.ndarray) -> bool:
    """Wave-3 presence rule (script 47 ``typed_inst``): source episode / change anywhere and target onset
    / change anywhere in the case."""
    on_ep = lambda k: bool((S[:, k] == 1).any())  # noqa: E731
    on_ons = lambda k: len(valid_onsets(S[:, k])) > 0  # noqa: E731
    if rel.edge_type == "association":
        return on_ep(tt.CONCEPTS.index(rel.source)) and on_ons(tt.CONCEPTS.index(rel.target))
    ai, di = tt.ACTIONS.index(rel.action), tt.DIRS.index(rel.direction)
    has_chg = bool(chg[:, ai, di].any())
    if rel.edge_type == "policy":
        return on_ep(tt.CONCEPTS.index(rel.source)) and has_chg
    return has_chg and on_ons(tt.CONCEPTS.index(rel.target))


def iter_relations_by(relations: Iterable[PopulationRelation], **kw) -> list[PopulationRelation]:
    return [r for r in relations if all(getattr(r, k) == v for k, v in kw.items())]


# ---------------------------------------------------------------------------
# Schema 3.1 fast layer: beat-level concepts and mechanistic fast edges
# (runs/iclr/graphs_v2_fast/protocol.md). Graphs without it stay valid 3.0 graphs.
# ---------------------------------------------------------------------------

FAST_SCHEMA_VERSION = "3.1"
BEAT_CONCEPTS: tuple[str, ...] = ("irregular_rr", "low_hrv", "high_ppv")  # columns of beat_window_concepts
PAT_CONCEPT = "prolonged_pat"  # addendum B1 (protocol_pat.md): within-case, appended as column 3
BEAT_CONCEPTS_ALL: tuple[str, ...] = BEAT_CONCEPTS + (PAT_CONCEPT,)
BEAT_CODES: dict[str, str] = {"irregular_rr": "local:irregular_rr", "low_hrv": "local:low_hrv",
                              "high_ppv": "local:high_ppv", "prolonged_pat": "local:prolonged_pat"}
BEAT_TERMS: dict[str, str] = {"irregular_rr": "Irregular RR intervals (signal concept, not a rhythm label)",
                              "low_hrv": "Low heart-rate variability (cohort-relative)",
                              "high_ppv": "High pulse-pressure variation (>13 %, controlled ventilation)",
                              "prolonged_pat": "Pulse arrival time prolonged relative to the patient's own baseline "
                                               "(within-case signal concept; device delay cancels)"}
BEAT_MODALITY: dict[str, str] = {"irregular_rr": "BEAT:ECG", "low_hrv": "BEAT:ECG", "high_ppv": "BEAT:ABP+RESP",
                                 "prolonged_pat": "BEAT:ECG+PPG"}
BEAT_STATISTIC: dict[str, str] = {"irregular_rr": "nRMSSD (trimmed)", "low_hrv": "RMSSD ms (trimmed)",
                                  "high_ppv": "PPV %", "prolonged_pat": "PAT - case baseline (ms)"}
BEAT_VARIABLES: dict[str, str] = {"resp": "rad (respiratory phase)", "sbp_beat": "mmHg", "rri": "ms",
                                  "pat": "ms (within-case only; includes a constant per-case PLETH device delay)"}
FAST_RELATION_IDS: tuple[str, ...] = ("fast:F1:resp->sbp_beat", "fast:F2:resp->rri",
                                      "fast:F3:sbp_beat->rri", "fast:F4:pat<->sbp_beat")
RR_MIN_MS, RR_MAX_MS = 300.0, 2000.0
RR_MIN_N, RR_MIN_COVER_SEC = 20, 24.0
IRR_TRIM, IRR_NRMSSD, IRR_SE, IRR_BINS = 2, 0.10, 0.70, 16
LOWHRV_DROP, LOWHRV_MIN_DIFFS = 2, 15
PPV_THRESHOLD = 13.0
PPV_MIN_BREATHS = 3
HARM_DETREND, HARM_MIN_BEATS = 15, 60


def rr_irregularity(rri: np.ndarray) -> tuple[float, float, bool]:
    """(nRMSSD, normalised Shannon entropy, turning-point count in the random-series range) of an RR sequence
    after removing its ``IRR_TRIM`` largest and smallest values (order of the rest kept; Dash et al. 2009)."""
    x = np.asarray(rri, float)
    if len(x) < 2 * IRR_TRIM + 3:
        return np.nan, np.nan, False
    order = np.argsort(x, kind="stable")
    keep = np.ones(len(x), bool)
    keep[order[:IRR_TRIM]] = False
    keep[order[-IRR_TRIM:]] = False
    y = x[keep]
    n = len(y)
    d = np.diff(y)
    nrmssd = float(np.sqrt(np.mean(d ** 2)) / np.mean(y))
    lo, hi = y.min(), y.max()
    if hi > lo:
        cnt, _ = np.histogram(y, bins=IRR_BINS, range=(lo, hi))
        p = cnt[cnt > 0] / n
        se = float(-(p * np.log(p)).sum() / np.log(IRR_BINS))
    else:
        se = 0.0
    tp = int(np.sum(d[:-1] * d[1:] < 0))
    mu, sd = (2 * n - 4) / 3.0, np.sqrt((16 * n - 29) / 90.0)
    return nrmssd, se, bool(abs(tp - mu) <= 2 * sd)


def trimmed_rmssd(rri: np.ndarray, adjacent: np.ndarray) -> float:
    """RMSSD over successive differences of adjacent passing intervals, dropping the ``LOWHRV_DROP`` largest
    absolute differences; NaN with fewer than ``LOWHRV_MIN_DIFFS`` differences left."""
    x = np.asarray(rri, float)
    d = np.diff(x)[np.asarray(adjacent, bool)]
    if len(d) - LOWHRV_DROP < LOWHRV_MIN_DIFFS:
        return np.nan
    d = np.sort(np.abs(d))[:len(d) - LOWHRV_DROP]
    return float(np.sqrt(np.mean(d ** 2)))


@dataclass
class BeatWindows:
    """Beat-level concept series on the v2 step grid."""

    S: np.ndarray  # (T, 3) tri-state, BEAT_CONCEPTS order; (T, 4) with prolonged_pat appended (BEAT_CONCEPTS_ALL)
    stat: np.ndarray  # (T, 3) defining statistic per window (NaN when unobserved)
    beat_lo: np.ndarray  # (T,) first beats row in the window (-1 if none)
    beat_hi: np.ndarray  # (T,) last beats row in the window (-1 if none)
    n_beats: np.ndarray  # (T,)


def beat_window_concepts(t_beat: np.ndarray, rri: np.ndarray, ws: np.ndarray, t_low_rmssd: float,
                         sbp: np.ndarray | None = None, pp: np.ndarray | None = None,
                         phase: np.ndarray | None = None, valid: np.ndarray | None = None,
                         cv_epochs: np.ndarray | None = None) -> BeatWindows:
    """Tri-state beat concepts per 30-s window [ws_i, ws_i + 30).

    ``t_beat`` are R-peak times on the grid's clock (sorted), ``rri`` the interval to the next R-peak (ms).
    ``cv_epochs`` (n, 2) are usable controlled-ventilation epochs on the same clock; ``high_ppv`` is observed only
    inside one of them, in a regular (observed, not irregular) window, with >= 3 qualifying breaths."""
    from clbench.edges.fast_coupling import respiratory_pressure_variation

    t = np.asarray(t_beat, float)
    r = np.asarray(rri, float)
    ws = np.asarray(ws, float)
    T = len(ws)
    S = np.full((T, 3), -1, np.int8)
    stat = np.full((T, 3), np.nan)
    lo_i = np.searchsorted(t, ws, "left")
    hi_i = np.searchsorted(t, ws + WINDOW_SEC, "left")
    beat_lo = np.where(hi_i > lo_i, lo_i, -1)
    beat_hi = np.where(hi_i > lo_i, hi_i - 1, -1)
    nb = hi_i - lo_i
    cv = np.zeros((0, 2)) if cv_epochs is None else np.asarray(cv_epochs, float).reshape(-1, 2)
    for i in range(T):
        a, b = lo_i[i], hi_i[i]
        if b - a < RR_MIN_N:
            continue
        idx = np.arange(a, b)
        rr = r[idx]
        inside = np.isfinite(rr) & (t[idx] + rr / 1000.0 < ws[i] + WINDOW_SEC)
        ok = inside & (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)
        if ok.sum() < RR_MIN_N or rr[ok].sum() / 1000.0 < RR_MIN_COVER_SEC:
            continue
        seq = rr[ok]
        nr, se, tp_ok = rr_irregularity(seq)
        if not np.isfinite(nr):
            continue
        irr = bool(nr >= IRR_NRMSSD and se >= IRR_SE and tp_ok)
        S[i, 0] = int(irr)
        stat[i, 0] = nr
        oi = idx[ok]
        rm = trimmed_rmssd(r[oi], np.diff(oi) == 1)
        if np.isfinite(rm):
            S[i, 1] = int(rm < t_low_rmssd)
            stat[i, 1] = rm
        if irr or sbp is None or not len(cv):
            continue
        inside_cv = np.any((cv[:, 0] <= ws[i]) & (ws[i] + WINDOW_SEC <= cv[:, 1]))
        if not inside_cv:
            continue
        rv = respiratory_pressure_variation(np.asarray(sbp)[a:b], np.asarray(pp)[a:b], np.asarray(phase)[a:b],
                                            np.asarray(valid)[a:b])
        if rv["n_breaths"] >= PPV_MIN_BREATHS and np.isfinite(rv["ppv"]):
            S[i, 2] = int(rv["ppv"] > PPV_THRESHOLD)
            stat[i, 2] = rv["ppv"]
    return BeatWindows(S=S, stat=stat, beat_lo=beat_lo, beat_hi=beat_hi, n_beats=nb)


def beat_concept_nodes(pid: str, bw: BeatWindows, ws: np.ndarray, calibration: CalibrationConfig,
                       record_id: str = "", beats_file: str = "") -> tuple[list[KGNode], list[BeatEpisode]]:
    """Beat concept x tier nodes (kg_construct's node rule, fallback tiers) and one BeatEpisode per run."""
    nodes: list[KGNode] = []
    episodes: list[BeatEpisode] = []
    rec = record_id or pid
    for k, c in enumerate(BEAT_CONCEPTS_ALL[:bw.S.shape[1]]):
        probs = (bw.S[:, k] == 1).astype(np.float64)
        if not probs.any():
            continue
        th = thresholds_for(c, calibration)
        cat = kc.ConceptCatalogEntry(modality=BEAT_MODALITY[c], concept_key=c, snomed_code=BEAT_CODES[c],
                                     preferred_term=BEAT_TERMS[c])
        act = kc.ConceptActivation(catalog=cat, probs=probs, window_starts_sec=ws, window_ends_sec=ws + WINDOW_SEC,
                                   record_id=rec)
        cn = kc._nodes_for_concept(act, th, STEP_SEC)
        by_tier = {}
        for n in cn:
            n.node_id = f"{pid}:b:{c}:{n.persistence_tier.value}"
            by_tier[n.persistence_tier] = n
        nodes.extend(cn)
        for r_i, run in enumerate(find_activation_runs(probs, threshold=th.activation_threshold,
                                                       window_stride_sec=STEP_SEC)):
            s, e = int(run.start_window_idx), int(run.end_window_idx)
            tier = th.classify(run.duration_sec)
            los = bw.beat_lo[s:e + 1]
            his = bw.beat_hi[s:e + 1]
            vals = bw.stat[s:e + 1, k]
            episodes.append(BeatEpisode(
                episode_id=f"b{k}.{r_i}", concept=c, node_id=by_tier[tier].node_id, tier=tier, record_id=rec,
                start_step=s, n_steps=int(run.n_windows), start_sec=float(ws[s]), end_sec=float(ws[e] + WINDOW_SEC),
                beat_start_idx=int(los[los >= 0].min()), beat_end_idx=int(his[his >= 0].max()),
                n_beats=int(his[his >= 0].max() - los[los >= 0].min() + 1), statistic=BEAT_STATISTIC[c],
                statistic_median=float(np.nanmedian(vals)) if np.isfinite(vals).any() else None,
                beats_file=beats_file))
    return nodes, episodes


# --- fast edges: harmonic phase locking with per-breath phase-randomisation surrogates ---------------------


def breath_ids(ph: np.ndarray) -> np.ndarray:
    """Breath index per beat from respiratory-phase wraps (as script 52 / ``respiratory_pressure_variation``)."""
    wraps = np.where(np.diff(np.unwrap(np.where(np.isfinite(ph), ph, 0.0)) // (2 * np.pi)) > 0)[0] + 1
    ind = np.zeros(len(ph), dtype=int)
    ind[wraps] = 1
    return np.cumsum(ind)


def harmonic_amplitude(y: np.ndarray, ph: np.ndarray) -> float:
    """2 sqrt(b^2 + c^2) of the OLS fit of y on [1, cos ph, sin ph] (script 52 ``harmonic_rsa`` after detrend)."""
    X = np.column_stack([np.ones(len(y)), np.cos(ph), np.sin(ph)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(2.0 * np.hypot(c[1], c[2]))


def harmonic_amplitude_batch(y: np.ndarray, PH: np.ndarray | None = None, W: np.ndarray | None = None,
                             C: np.ndarray | None = None, Sn: np.ndarray | None = None) -> np.ndarray:
    """Vectorised harmonic amplitude for K phase rows ``PH`` (K, m) (or their cosines ``C`` and sines ``Sn``)
    and optional weights ``W`` (K, m) (weighted normal equations of y on [1, cos, sin])."""
    if C is None:
        PH = np.atleast_2d(PH)
        C, Sn = np.cos(PH), np.sin(PH)
    K, m = C.shape
    y = np.asarray(y, float)
    if W is None:
        wC, wS = C, Sn
        s0 = np.full(K, float(m))
        sy = np.full(K, float(y.sum()))
    else:
        wC, wS = W * C, W * Sn
        s0 = W.sum(axis=1)
        sy = W @ y
    G = np.empty((K, 3, 3))
    G[:, 0, 0] = s0
    G[:, 0, 1] = G[:, 1, 0] = wC.sum(axis=1)
    G[:, 0, 2] = G[:, 2, 0] = wS.sum(axis=1)
    G[:, 1, 1] = (wC * C).sum(axis=1)
    G[:, 1, 2] = G[:, 2, 1] = (wC * Sn).sum(axis=1)
    G[:, 2, 2] = (wS * Sn).sum(axis=1)
    r = np.stack([sy, wC @ y, wS @ y], axis=1)
    try:
        coef = np.linalg.solve(G, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coef = np.array([np.linalg.lstsq(G[k], r[k], rcond=None)[0] for k in range(K)])
    return 2.0 * np.hypot(coef[:, 1], coef[:, 2])


def fast_epoch_prepare(target: np.ndarray, valid: np.ndarray, ph: np.ndarray, bid: np.ndarray | None = None):
    """Detrended target, phase and breath ids on usable beats; None with fewer than 60 usable beats."""
    from clbench.edges.fast_coupling import running_mean_detrend

    ph = np.asarray(ph, float)
    y = running_mean_detrend(np.asarray(target, float), np.asarray(valid, bool), HARM_DETREND)
    ok = np.isfinite(y) & np.isfinite(ph)
    if ok.sum() < HARM_MIN_BEATS:
        return None
    b = breath_ids(ph) if bid is None else np.asarray(bid)
    return y[ok], ph[ok], b[ok], b


def fast_epoch_test(y: np.ndarray, ph: np.ndarray, bid_ok: np.ndarray, n_breath_ids: int, rng: np.random.Generator,
                    K: int = 499, B: int = 200) -> dict[str, Any]:
    """Observed amplitude, K per-breath phase-randomisation surrogate amplitudes, Monte Carlo p and a
    breath-cluster bootstrap percentile CI (B = 0 skips the CI)."""
    A = harmonic_amplitude(y, ph)
    delta = rng.uniform(0.0, 2 * np.pi, size=(K, n_breath_ids))
    cd, sd = np.cos(delta)[:, bid_ok], np.sin(delta)[:, bid_ok]
    cp, sp = np.cos(ph), np.sin(ph)
    Ak = harmonic_amplitude_batch(y, C=cp * cd - sp * sd, Sn=sp * cd + cp * sd)  # cos / sin of ph + delta
    p = float((1 + np.sum(Ak >= A)) / (K + 1))
    lo = hi = np.nan
    if B:
        ub, inv = np.unique(bid_ok, return_inverse=True)
        cnt = np.stack([np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)) for _ in range(B)])
        Ab = harmonic_amplitude_batch(y, W=cnt[:, inv].astype(float), C=np.broadcast_to(np.cos(ph), (B, len(ph))),
                                      Sn=np.broadcast_to(np.sin(ph), (B, len(ph))))
        lo, hi = (float(x) for x in np.nanpercentile(Ab, [2.5, 97.5]))
    return {"A": A, "Ak": Ak, "p": p, "lo": lo, "hi": hi, "n_breaths": int(len(np.unique(bid_ok)))}


def patient_fast_p(A: np.ndarray, Ak: np.ndarray, alternative: str = "greater") -> float:
    """Monte Carlo p of the mean statistic over epochs against the mean of the k-th surrogates (E, K).
    ``alternative="less"`` for a sign-bearing statistic expected negative (F4)."""
    T = float(np.mean(A))
    Tk = np.asarray(Ak).mean(axis=0)
    hits = np.sum(Tk <= T) if alternative == "less" else np.sum(Tk >= T)
    return float((1 + hits) / (len(Tk) + 1))


def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def bh_adjust(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


# --- addendum B1 (runs/iclr/graphs_v2_fast/protocol_pat.md): PAT within case --------------------------------

PAT_DETREND, PAT_MIN_PAIRS = 31, 30  # clbench.edges.fast_coupling.pat_coupling
PAT_SHIFT_MIN = 30  # surrogate shifts U{30, ..., n - 30} beats (registered T1-twin range)
PAT_BLOCK = 31  # bootstrap block = detrend window
PAT_MIN_BEATS = 60
PAT_WIN_MIN = 20


def pat_window_concept(t_beat: np.ndarray, pat: np.ndarray, valid: np.ndarray, ws: np.ndarray,
                       delta_ms: float) -> tuple[np.ndarray, np.ndarray, float]:
    """``prolonged_pat`` per 30-s window: observed iff >= 20 valid beats with a finite PAT; statistic = window
    median PAT - the case baseline (median over observed windows of the window median); on iff >= ``delta_ms``.
    Returns (S column, statistic column, baseline)."""
    t = np.asarray(t_beat, float)
    x = np.where(np.asarray(valid, bool) & np.isfinite(pat), np.asarray(pat, float), np.nan)
    ws = np.asarray(ws, float)
    lo = np.searchsorted(t, ws, "left")
    hi = np.searchsorted(t, ws + WINDOW_SEC, "left")
    med = np.full(len(ws), np.nan)
    for i in range(len(ws)):
        v = x[lo[i]:hi[i]]
        v = v[np.isfinite(v)]
        if len(v) >= PAT_WIN_MIN:
            med[i] = np.median(v)
    S = np.full(len(ws), -1, np.int8)
    if not np.isfinite(med).any():
        return S, np.full(len(ws), np.nan), float("nan")
    base = float(np.nanmedian(med))
    d = med - base
    ok = np.isfinite(d)
    S[ok] = (d[ok] >= delta_ms).astype(np.int8)
    return S, d, base


def with_pat_concept(bw: BeatWindows, S_pat: np.ndarray, stat_pat: np.ndarray) -> BeatWindows:
    """Append the ``prolonged_pat`` column (BEAT_CONCEPTS_ALL order)."""
    return BeatWindows(S=np.column_stack([bw.S, np.asarray(S_pat, np.int8)]),
                       stat=np.column_stack([bw.stat, np.asarray(stat_pat, float)]), beat_lo=bw.beat_lo,
                       beat_hi=bw.beat_hi, n_beats=bw.n_beats)


def _detrend_rows(X: np.ndarray, V: np.ndarray, k: int = PAT_DETREND) -> np.ndarray:
    """Row-wise ``running_mean_detrend`` (centred k-beat running mean over valid entries; NaN elsewhere)."""
    n = X.shape[1]
    h = k // 2
    Xv = np.where(V, X, 0.0)
    P = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(Xv, axis=1)], axis=1)
    Q = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(V.astype(float), axis=1)], axis=1)
    ar = np.arange(n)
    hi_ = np.minimum(ar + h + 1, n)
    lo_ = np.maximum(ar - h, 0)
    c = P[:, hi_] - P[:, lo_]
    cnt = Q[:, hi_] - Q[:, lo_]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = c / cnt
    return np.where(V & (cnt >= h), X - m, np.nan)


def pat_slope_rows(PAT: np.ndarray, SBP: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Row-wise ``pat_coupling`` slope (ms/mmHg) for (K, n) PAT, SBP and validity; NaN where undefined."""
    ok = V & np.isfinite(PAT) & np.isfinite(SBP)
    a = _detrend_rows(PAT, ok)
    b = _detrend_rows(SBP, ok)
    m = np.isfinite(a) & np.isfinite(b)
    cnt = m.sum(axis=1)
    a0, b0 = np.where(m, a, 0.0), np.where(m, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        am = a0.sum(axis=1) / cnt
        bm = b0.sum(axis=1) / cnt
        da = np.where(m, a - am[:, None], 0.0)
        db = np.where(m, b - bm[:, None], 0.0)
        sbb = (db * db).sum(axis=1)
        saa = (da * da).sum(axis=1)
        sl = (da * db).sum(axis=1) / sbb
    bad = (ok.sum(axis=1) < PAT_MIN_PAIRS) | (cnt < PAT_MIN_PAIRS) | ~(sbb > 0) | ~(saa > 0)
    return np.where(bad, np.nan, sl)


def f4_epoch_test(pat: np.ndarray, sbp: np.ndarray, valid: np.ndarray, rng: np.random.Generator, K: int = 499,
                  B: int = 200) -> dict[str, Any] | None:
    """F4 per-epoch test (addendum B1 section 4): observed ``pat_coupling`` slope, K circular-shift surrogate
    slopes (SBP and its validity rolled by U{30..n-30} beats), one-sided p (expected negative) and a 31-beat
    block-cluster bootstrap percentile CI (B = 0 skips it). None if the epoch is not tested."""
    from clbench.edges.fast_coupling import pat_coupling, running_mean_detrend

    pat = np.asarray(pat, float)
    sbp = np.asarray(sbp, float)
    v = np.asarray(valid, bool)
    n = len(pat)
    if n < PAT_MIN_BEATS:
        return None
    pc = pat_coupling(pat, sbp, v)
    s = pc["pat_sbp_slope"]
    if not np.isfinite(s):
        return None
    u = rng.integers(PAT_SHIFT_MIN, n - PAT_SHIFT_MIN + 1, size=K)
    idx = (np.arange(n)[None, :] - u[:, None]) % n  # np.roll(x, u)[i] = x[(i - u) % n]
    sk = pat_slope_rows(np.broadcast_to(pat, (K, n)), sbp[idx], v[None, :] & v[idx])
    n_undef = int((~np.isfinite(sk)).sum())
    sk = np.where(np.isfinite(sk), sk, -np.inf)  # conservative: an undefined surrogate counts as <= s
    p = float((1 + np.sum(sk <= s)) / (K + 1))
    lo = hi = np.nan
    ok = v & np.isfinite(pat) & np.isfinite(sbp)
    a = running_mean_detrend(pat, ok, PAT_DETREND)
    b = running_mean_detrend(sbp, ok, PAT_DETREND)
    m = np.isfinite(a) & np.isfinite(b)
    blk = (np.arange(n) // PAT_BLOCK)[m]
    ub, inv = np.unique(blk, return_inverse=True)
    if B:
        cnt = np.stack([np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)) for _ in range(B)])
        W = cnt[:, inv].astype(float)
        am, bm = a[m], b[m]
        sw = W.sum(axis=1)
        ma = (W @ am) / sw
        mb = (W @ bm) / sw
        da = am[None, :] - ma[:, None]
        db = bm[None, :] - mb[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            sb = (W * da * db).sum(axis=1) / (W * db * db).sum(axis=1)
        lo, hi = (float(x) for x in np.nanpercentile(sb, [2.5, 97.5]))
    return {"A": float(s), "Ak": sk, "p": p, "lo": lo, "hi": hi, "n_pairs": int(m.sum()), "n_blocks": int(len(ub)),
            "corr": float(pc["pat_sbp_corr"]), "n_undefined_surrogates": n_undef}
