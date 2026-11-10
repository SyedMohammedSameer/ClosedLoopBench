"""
Phase 3.1 — Patient KG construction.

Turns one patient's per-(modality, concept) activation sequences into a
:class:`clbench.schema.PatientKG`:

* Nodes are ``(SNOMED concept, persistence tier)`` pairs, with persistence
  tiers derived from a :class:`clbench.persistence.CalibrationConfig`.
* Edges:

    - :mod:`clbench.edges.co_evidence`  → ``CO_OCCURS`` (always)
    - learned two-timescale effects        → ``FEEDBACK`` / ``LAGGED`` (passed in as
      :class:`ConceptEffect`; schema 2.0)
    - treatment effects                    → ``ACTS_ON`` from :class:`InterventionNode`
      (passed in as :class:`InterventionEffect`; schema 2.0)
    - V1 pairwise tests, now baselines     → ``PRECEDES`` / ``FOLLOWS``
      (:mod:`clbench.edges.temporal`) and ``GRANGER_PRECEDES``
      (:mod:`clbench.edges.granger`); on unless ``legacy_edges=False``

* Every node carries :class:`EvidencePointer` back-references to the
  source signal windows that produced its activation runs.  This is the
  evidence-traceability invariant from the locked design.

This module is intentionally I/O-free.  ``build_patient_kg`` takes the
materialised activation arrays + calibration config + a small concept
catalog (modality + snomed code + preferred term per concept) and returns
a fully-populated :class:`PatientKG`.  The orchestration script
:mod:`scripts.05_build_kgs` handles loading the inputs and writing the
output.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from clbench.edges import (
    co_evidence_relation,
    granger_edge_weight,
    granger_precedes_score,
    is_co_evidence_edge,
    is_directed_temporal_edge,
    is_granger_edge,
    temporal_relation,
)
from clbench.persistence import (
    DEFAULT_ACTIVATION_THRESHOLD,
    WINDOW_STRIDE_SEC_DEFAULT,
    ActivationRun,
    CalibrationConfig,
    TierThresholds,
    find_activation_runs,
)
from clbench.schema import (
    EdgeProvenance,
    EdgeType,
    EvidencePointer,
    EvidenceRecord,
    InterventionAction,
    InterventionKind,
    InterventionNode,
    KGEdge,
    KGNode,
    PatientKG,
    PersistenceTier,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class ConceptCatalogEntry:
    """Minimal per-concept info needed to mint a KG node."""

    modality: str  # ECG | PPG | RESP | ABP
    concept_key: str  # e.g. "tachycardia", matches CBM CONCEPT_REGISTRY
    snomed_code: str
    preferred_term: str


@dataclass
class ConceptActivation:
    """One concept's per-window probabilities and aligned source-window times.

    All three arrays share the same length N (one entry per window):

    * ``probs[i]``           — CBM sigmoid activation at window i
    * ``window_starts_sec[i]`` — absolute time of the window start, seconds
    * ``window_ends_sec[i]`` — absolute time of the window end, seconds
    """

    catalog: ConceptCatalogEntry
    probs: np.ndarray
    window_starts_sec: np.ndarray
    window_ends_sec: np.ndarray
    record_id: str  # MIMIC-IV-WDB record id this came from

    def __post_init__(self) -> None:
        n = len(self.probs)
        if len(self.window_starts_sec) != n or len(self.window_ends_sec) != n:
            raise ValueError(
                f"ConceptActivation arrays must share length; "
                f"got probs={n}, starts={len(self.window_starts_sec)}, "
                f"ends={len(self.window_ends_sec)}"
            )


@dataclass
class ConceptEffect:
    """A learned concept → concept effect, stated at concept level.

    Emitted onto every (source tier node, target tier node) pair, like the V1 edges.
    """

    source_concept: str
    target_concept: str
    edge_type: EdgeType  # FEEDBACK or LAGGED
    effect: float  # signed structural coefficient
    lag_sec: float | None = None
    reliability: float | None = None
    provenance: EdgeProvenance | None = None


@dataclass
class InterventionEffect:
    """A learned treatment → concept effect for one intervention (agent, kind, parameter)."""

    agent: str
    kind: InterventionKind
    target_concept: str
    effect: float
    lag_sec: float | None = None
    reliability: float | None = None
    parameter: str | None = None
    provenance: EdgeProvenance | None = None


# ---------------------------------------------------------------------------
# Per-concept node construction
# ---------------------------------------------------------------------------


def _get_thresholds(
    concept_key: str,
    calibration: CalibrationConfig,
) -> TierThresholds:
    """Return calibrated tier thresholds for ``concept_key``, or a fallback."""
    th = calibration.concepts.get(concept_key)
    if th is not None:
        return th
    log.warning(
        "No calibration entry for concept '%s' — using fallback thresholds",
        concept_key,
    )
    return TierThresholds(
        concept=concept_key,
        transient_max_sec=60.0,
        episodic_max_sec=600.0,
        activation_threshold=DEFAULT_ACTIVATION_THRESHOLD,
        n_runs_observed=0,
        source="fallback",
    )


def _nodes_for_concept(
    activation: ConceptActivation,
    thresholds: TierThresholds,
    window_stride_sec: float,
) -> list[KGNode]:
    """
    Extract activation runs, classify by tier, and mint one node per run.

    Two runs of the same (concept, tier) collapse into a single node carrying
    both evidence pointers — this keeps the (snomed_code, tier) identity
    invariant from :class:`KGNode.identity_key` honoured at the patient level.
    """
    runs = find_activation_runs(
        activation.probs,
        threshold=thresholds.activation_threshold,
        window_stride_sec=window_stride_sec,
    )
    if not runs:
        return []

    # Group runs by their tier classification.
    grouped: dict[PersistenceTier, list[ActivationRun]] = {}
    for run in runs:
        tier = thresholds.classify(run.duration_sec)
        grouped.setdefault(tier, []).append(run)

    catalog = activation.catalog
    nodes: list[KGNode] = []
    for tier, run_list in grouped.items():
        evidence: list[EvidencePointer] = []
        for run in run_list:
            start_idx = run.start_window_idx
            end_idx = run.end_window_idx
            evidence.append(
                EvidencePointer(
                    record_id=activation.record_id,
                    modality=catalog.modality,
                    window_start_sec=float(activation.window_starts_sec[start_idx]),
                    window_end_sec=float(activation.window_ends_sec[end_idx]),
                    activation_score=float(run.mean_activation),
                )
            )
        nodes.append(
            KGNode(
                snomed_code=catalog.snomed_code,
                preferred_term=catalog.preferred_term,
                persistence_tier=tier,
                evidence=evidence,
                external_mappings={"SNOMEDCT": catalog.snomed_code},
            )
        )
    return nodes


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------


def _emit_temporal_edges(
    a_nodes: list[KGNode],
    b_nodes: list[KGNode],
    a_probs: np.ndarray,
    b_probs: np.ndarray,
    a_threshold: float,
    b_threshold: float,
    window_stride_sec: float,
) -> list[KGEdge]:
    """
    Emit PRECEDES / FOLLOWS edges between every (a_node, b_node) pair when
    the underlying activation series show a consistent ordered lag.

    The relation is computed once at the concept level; the same lag is
    attributed to every (tier-A, tier-B) cross-product node pair because
    persistence tiers are derived from durations of the same underlying runs.
    """
    if not a_nodes or not b_nodes:
        return []
    rel = temporal_relation(
        a_probs,
        b_probs,
        a_threshold,
        b_threshold,
        window_stride_sec=window_stride_sec,
    )
    if not is_directed_temporal_edge(rel):
        return []
    weight = float(max(0.0, min(1.0, abs(rel.peak_xcorr))))
    if rel.a_precedes_b:
        src_nodes, tgt_nodes = a_nodes, b_nodes
    else:
        src_nodes, tgt_nodes = b_nodes, a_nodes
    edges: list[KGEdge] = []
    for s in src_nodes:
        for t in tgt_nodes:
            edges.append(
                KGEdge(
                    source_node_id=s.node_id,
                    target_node_id=t.node_id,
                    edge_type=EdgeType.PRECEDES,
                    weight=weight,
                )
            )
            edges.append(
                KGEdge(
                    source_node_id=t.node_id,
                    target_node_id=s.node_id,
                    edge_type=EdgeType.FOLLOWS,
                    weight=weight,
                )
            )
    return edges


def _emit_co_evidence_edges(
    a_nodes: list[KGNode],
    b_nodes: list[KGNode],
    a_probs: np.ndarray,
    b_probs: np.ndarray,
    a_threshold: float,
    b_threshold: float,
    primary_only: bool = False,
) -> list[KGEdge]:
    if not a_nodes or not b_nodes:
        return []
    if primary_only:  # one pair of edges between the concepts, not one per tier pair
        a_nodes, b_nodes = _primary_node(a_nodes), _primary_node(b_nodes)
    rel = co_evidence_relation(a_probs, b_probs, a_threshold, b_threshold)
    if not is_co_evidence_edge(rel):
        return []
    weight = float(max(0.0, min(1.0, rel.jaccard)))
    edges: list[KGEdge] = []
    for s in a_nodes:
        for t in b_nodes:
            edges.append(
                KGEdge(
                    source_node_id=s.node_id,
                    target_node_id=t.node_id,
                    edge_type=EdgeType.CO_OCCURS,
                    weight=weight,
                )
            )
            edges.append(
                KGEdge(
                    source_node_id=t.node_id,
                    target_node_id=s.node_id,
                    edge_type=EdgeType.CO_OCCURS,
                    weight=weight,
                )
            )
    return edges


def _emit_granger_edges(
    a_nodes: list[KGNode],
    b_nodes: list[KGNode],
    a_probs: np.ndarray,
    b_probs: np.ndarray,
) -> list[KGEdge]:
    if not a_nodes or not b_nodes:
        return []
    rel = granger_precedes_score(a_probs, b_probs)
    if not is_granger_edge(rel):
        return []
    weight = granger_edge_weight(rel)
    edges: list[KGEdge] = []
    for s in a_nodes:
        for t in b_nodes:
            edges.append(
                KGEdge(
                    source_node_id=s.node_id,
                    target_node_id=t.node_id,
                    edge_type=EdgeType.GRANGER_PRECEDES,
                    weight=weight,
                )
            )
    return edges


_LEARNED_EDGE_TYPES = (EdgeType.FEEDBACK, EdgeType.LAGGED)

# Event-table kind (``clbench.data.vitaldb.EVENT_KINDS``) → (InterventionKind, parameter).
_EVENT_KINDS: dict[str, tuple[InterventionKind, str | None]] = {
    "tci_target": (InterventionKind.TCI_TARGET, None),
    "infusion_rate": (InterventionKind.INFUSION_RATE, None),
    "vent_set_rr": (InterventionKind.VENTILATOR_SETTING, "set_rr"),
}
_TCI_UNITS: dict[str, str] = {"propofol": "µg/mL", "remifentanil": "ng/mL"}


def _effect_weight(effect: float) -> float:
    return float(min(1.0, abs(effect)))


def _intervention_unit(kind: InterventionKind, agent: str, parameter: str | None) -> str | None:
    if kind == InterventionKind.TCI_TARGET:
        return _TCI_UNITS.get(agent)
    if kind == InterventionKind.INFUSION_RATE:
        return "mL/h"
    if parameter == "set_rr":
        return "/min"
    return None


def intervention_nodes_from_events(
    events: Iterable[Mapping[str, Any]],
    record_id: str,
) -> list[InterventionNode]:
    """
    Group intervention event rows into one node per (agent, kind, parameter).

    Rows need ``agent``, ``kind`` (``tci_target`` | ``infusion_rate`` | ``vent_set_rr``),
    ``t_sec``, ``value_before`` and ``value_after`` — the columns of VitalDB
    ``events/{patient_id}.parquet`` and of the synthetic benchmark's events.
    """
    grouped: dict[tuple[str, InterventionKind, str | None], list[InterventionAction]] = {}
    for row in events:
        kind_key = str(row["kind"])
        if kind_key not in _EVENT_KINDS:
            raise ValueError(
                f"Unknown intervention kind {kind_key!r}; expected one of {sorted(_EVENT_KINDS)}"
            )
        kind, parameter = _EVENT_KINDS[kind_key]
        grouped.setdefault((str(row["agent"]), kind, parameter), []).append(
            InterventionAction(
                record_id=record_id,
                t_sec=float(row["t_sec"]),
                value_before=float(row["value_before"]),
                value_after=float(row["value_after"]),
            )
        )

    nodes: list[InterventionNode] = []
    for (agent, kind, parameter), actions in sorted(
        grouped.items(), key=lambda item: (item[0][1].value, item[0][0], item[0][2] or "")
    ):
        nodes.append(
            InterventionNode(
                agent=agent,
                kind=kind,
                parameter=parameter,
                unit=_intervention_unit(kind, agent, parameter),
                actions=sorted(actions, key=lambda a: a.t_sec),
            )
        )
    return nodes


def _primary_node(nodes: list[KGNode]) -> list[KGNode]:
    """The tier node with the most evidence time — where a concept-level effect belongs."""
    if not nodes:
        return []
    duration = lambda n: sum(e.window_end_sec - e.window_start_sec for e in n.evidence)  # noqa: E731
    return [max(nodes, key=duration)]


def _emit_learned_edges(
    nodes_per_concept: dict[str, list[KGNode]],
    effects: Iterable[ConceptEffect],
    primary_only: bool = False,
) -> list[KGEdge]:
    """FEEDBACK / LAGGED edges; a feedback loop arrives as two opposite effects.

    A learned effect is stated at concept level. ``primary_only`` attaches it once, to each
    concept's primary tier node, instead of copying it onto every tier pair.
    """
    pick = _primary_node if primary_only else (lambda nodes: nodes)
    edges: list[KGEdge] = []
    for eff in effects:
        if eff.edge_type not in _LEARNED_EDGE_TYPES:
            raise ValueError(
                f"ConceptEffect edge_type must be FEEDBACK or LAGGED, got {eff.edge_type}"
            )
        if eff.source_concept == eff.target_concept:
            continue  # a concept's own dynamics are what persistence tiers describe
        for s in pick(nodes_per_concept.get(eff.source_concept, [])):
            for t in pick(nodes_per_concept.get(eff.target_concept, [])):
                edges.append(
                    KGEdge(
                        source_node_id=s.node_id,
                        target_node_id=t.node_id,
                        edge_type=eff.edge_type,
                        weight=_effect_weight(eff.effect),
                        effect=float(eff.effect),
                        lag_sec=eff.lag_sec,
                        reliability=eff.reliability,
                        provenance=eff.provenance,
                    )
                )
    return edges


def _emit_acts_on_edges(
    interventions: list[InterventionNode],
    nodes_per_concept: dict[str, list[KGNode]],
    effects: Iterable[InterventionEffect],
    primary_only: bool = False,
) -> list[KGEdge]:
    """ACTS_ON edges, only for treatments this patient actually received."""
    pick = _primary_node if primary_only else (lambda nodes: nodes)
    by_key = {n.identity_key: n for n in interventions}
    edges: list[KGEdge] = []
    for eff in effects:
        node = by_key.get((eff.agent, eff.kind, eff.parameter))
        if node is None:
            continue
        for t in pick(nodes_per_concept.get(eff.target_concept, [])):
            edges.append(
                KGEdge(
                    source_node_id=node.node_id,
                    target_node_id=t.node_id,
                    edge_type=EdgeType.ACTS_ON,
                    weight=_effect_weight(eff.effect),
                    effect=float(eff.effect),
                    lag_sec=eff.lag_sec,
                    reliability=eff.reliability,
                    provenance=eff.provenance,
                )
            )
    return edges


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_patient_kg(
    patient_id: str,
    activations: list[ConceptActivation],
    calibration: CalibrationConfig,
    hadm_id: str | None = None,
    window_stride_sec: float = WINDOW_STRIDE_SEC_DEFAULT,
    enable_granger: bool = True,
    concept_vocabulary_version: str = "1.0",
    concept_effects: Iterable[ConceptEffect] | None = None,
    interventions: list[InterventionNode] | None = None,
    intervention_effects: Iterable[InterventionEffect] | None = None,
    legacy_edges: bool = True,
    learned_edges_on: str = "all",
    enable_co_evidence: bool = True,
    evidence_records: dict[str, EvidenceRecord] | None = None,
) -> PatientKG:
    """
    Build one patient's concept-grounded KG from CBM activation streams.

    Parameters
    ----------
    patient_id : str
    activations : list[ConceptActivation]
        One per (modality, concept) pair the patient has data for.  All
        ``activations`` are assumed to share a common window axis when
        their ``record_id`` matches; cross-record pairs are skipped from
        edge construction (we don't try to align across records here —
        that's a deliberate scope limit for V1).
    calibration : CalibrationConfig
        Per-concept tier thresholds + activation thresholds.
    hadm_id : optional admission identifier.
    window_stride_sec : float
        Cadence between consecutive windows in ``activations``.
    enable_granger : bool
        Skip Granger edges when False — useful for fast smoke-tests since
        statsmodels can be slow on long series.
    concept_effects : iterable of ConceptEffect, optional
        Learned FEEDBACK / LAGGED effects (schema 2.0).
    interventions : list of InterventionNode, optional
        Treatments the patient received (see :func:`intervention_nodes_from_events`).
    intervention_effects : iterable of InterventionEffect, optional
        Learned treatment effects, emitted as ACTS_ON edges for received treatments only.
    learned_edges_on : "all" | "primary"
        Attach learned effects and co-evidence to every (tier, tier) node pair, like V1, or once to
        each concept's primary node (the tier with the most evidence time).
    legacy_edges : bool
        Emit the V1 PRECEDES / FOLLOWS and GRANGER_PRECEDES edges.  Keep True to
        reproduce V1 graphs; set False for schema 2.0 graphs, where they are baselines.
    enable_co_evidence : bool
        Disable the legacy coactivation emitter when supplying audited coactivation edges.
    evidence_records : dict, optional
        Stored fitting records referenced by learned-edge provenance.

    The graph is stamped ``schema_version="2.0"`` when any schema 2.0 input is given
    or ``legacy_edges`` is False; otherwise it is identical to a V1 graph.
    """
    if learned_edges_on not in ("all", "primary"):
        raise ValueError(f"learned_edges_on must be 'all' or 'primary', got {learned_edges_on!r}")
    primary = learned_edges_on == "primary"

    # 1. Materialise nodes per concept.
    nodes_per_concept: dict[str, list[KGNode]] = {}
    thresholds_per_concept: dict[str, TierThresholds] = {}
    for act in activations:
        th = _get_thresholds(act.catalog.concept_key, calibration)
        thresholds_per_concept[act.catalog.concept_key] = th
        nodes = _nodes_for_concept(act, th, window_stride_sec)
        nodes_per_concept[act.catalog.concept_key] = nodes

    all_nodes: list[KGNode] = [node for nodes in nodes_per_concept.values() for node in nodes]

    # 2. Pairwise edges between every (concept A, concept B) pair, A != B.
    #    Skip pairs whose activation arrays come from different records —
    #    we don't realign across records in V1.
    edges: list[KGEdge] = []
    keys = [a.catalog.concept_key for a in activations]
    activations_by_key = {a.catalog.concept_key: a for a in activations}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            ai, aj = activations_by_key[ki], activations_by_key[kj]
            if ai.record_id != aj.record_id:
                continue
            if len(ai.probs) != len(aj.probs):
                # Different N within the same record — likely different
                # quality_ok masks at preprocess time.  Skip pair safely.
                continue
            ni, nj = nodes_per_concept[ki], nodes_per_concept[kj]
            ti = thresholds_per_concept[ki].activation_threshold
            tj = thresholds_per_concept[kj].activation_threshold

            if legacy_edges:
                edges.extend(
                    _emit_temporal_edges(
                        ni,
                        nj,
                        ai.probs,
                        aj.probs,
                        ti,
                        tj,
                        window_stride_sec,
                    )
                )
            if enable_co_evidence:
                edges.extend(_emit_co_evidence_edges(
                    ni,
                    nj,
                    ai.probs,
                    aj.probs,
                    ti,
                    tj,
                    primary,
                ))
            if legacy_edges and enable_granger:
                edges.extend(_emit_granger_edges(ni, nj, ai.probs, aj.probs))
                # And the reverse Granger direction (asymmetric).
                edges.extend(_emit_granger_edges(nj, ni, aj.probs, ai.probs))

    # 3. Schema 2.0: learned concept effects, intervention nodes, treatment effects.
    intervention_nodes = list(interventions or [])
    if concept_effects is not None:
        edges.extend(_emit_learned_edges(nodes_per_concept, concept_effects, primary))
    if intervention_effects is not None:
        edges.extend(
            _emit_acts_on_edges(intervention_nodes, nodes_per_concept, intervention_effects, primary)
        )
    schema_v2 = (
        not legacy_edges
        or concept_effects is not None
        or interventions is not None
        or intervention_effects is not None
    )

    kg = PatientKG(
        patient_id=patient_id,
        hadm_id=hadm_id,
        nodes=all_nodes,
        edges=edges,
        interventions=intervention_nodes,
        evidence_records=dict(evidence_records or {}),
        concept_vocabulary_version=concept_vocabulary_version,
        schema_version="2.0" if schema_v2 else "1.0",
    )

    # 4. Schema sanity.
    errors = kg.validate_edge_refs()
    if errors:
        for e in errors[:5]:
            log.warning("KG validation: %s", e)
        log.warning(
            "Patient %s: %d dangling edge refs (first 5 logged)",
            patient_id,
            len(errors),
        )

    log.info(
        "Patient %s KG: %d nodes (%d concepts), %d intervention nodes, %d edges",
        patient_id,
        len(all_nodes),
        len(nodes_per_concept),
        len(intervention_nodes),
        len(edges),
    )
    return kg


# ---------------------------------------------------------------------------
# Convenience helper for tests / orchestration
# ---------------------------------------------------------------------------


def concept_catalog_from_registry(
    registry: Iterable[dict[str, str]],
    preferred_terms: dict[str, str] | None = None,
) -> dict[str, ConceptCatalogEntry]:
    """
    Convert ``CONCEPT_REGISTRY``-shaped dicts into a {concept_key: entry} map.

    ``preferred_terms`` overrides the default preferred-term construction
    (title-case of the concept key).  Pass the SNOMED preferred term where
    available — e.g. {"tachycardia": "Tachycardia"}.
    """
    out: dict[str, ConceptCatalogEntry] = {}
    pt = preferred_terms or {}
    for entry in registry:
        concept_key = entry["concept"]
        out[concept_key] = ConceptCatalogEntry(
            modality=entry["modality"],
            concept_key=concept_key,
            snomed_code=entry.get("snomed", "0"),
            preferred_term=pt.get(concept_key, concept_key.replace("_", " ").title()),
        )
    return out
