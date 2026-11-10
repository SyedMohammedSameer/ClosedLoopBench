"""
Core Pydantic schema for CLBench.

Defines the node/edge/graph data model:
  - PersistenceTier     — transient / episodic / persistent
  - EvidencePointer     — back-pointer to source signal window
  - KGNode              — (SNOMED concept, persistence tier) pair + evidence
  - InterventionKind    — tci_target / infusion_rate / ventilator_setting
  - InterventionAction  — one timestamped clinician action on a set-point
  - InterventionNode    — a treatment the patient received, with its actions
  - EdgeType            — feedback / lagged / co_occurs / acts_on, plus V1 precedes / follows /
                          granger_precedes
  - KGEdge              — typed, weighted edge with evidence refs and optional effect / lag /
                          reliability
  - PatientKG           — per-patient graph container

Design invariants (locked 2026-05-23; extended for schema 2.0 on 2026-09-15):
  - Ontology grounding is DOWNSTREAM of signal encoders, never an auxiliary head.
  - Every concept node carries evidence pointers back to source windows (first-class property).
  - Every intervention node carries its timestamped actions — the treatment-side counterpart.
  - external_mappings is a future-proof hook for cross-ontology integration.
  - Schema 2.0 only adds fields and enum members: V1 graphs load and validate unchanged and
    report ``schema_version == "1.0"``.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PersistenceTier(str, Enum):
    """Temporal persistence of a concept activation across signal windows.

    Thresholds are data-driven (calibration study in Phase 3).  The enum
    values are stable; only the numeric cutoffs will change.
    """

    TRANSIENT = "transient"
    """Short-lived activation: below the episodic threshold.
    Examples: transient SVT, brief desaturation dip."""

    EPISODIC = "episodic"
    """Recurrent or sustained over minutes–hours but not the full stay.
    Examples: paroxysmal AFib, intermittent tachypnea episodes."""

    PERSISTENT = "persistent"
    """Maintained across the majority of the monitored window.
    Examples: chronic hypertension, persistent hypoxemia."""


class EdgeType(str, Enum):
    """Edge types, each grounded in signal evidence rather than an external ontology.

    Schema 2.0 (two-timescale patient KG): ``FEEDBACK``, ``LAGGED``, ``CO_OCCURS``, ``ACTS_ON``.
    V1 edges, kept so old graphs load and as baselines: ``PRECEDES``, ``FOLLOWS``,
    ``GRANGER_PRECEDES``.
    """

    PRECEDES = "precedes"
    """Temporal ordering: source-concept windows consistently precede target."""

    FOLLOWS = "follows"
    """Temporal ordering: source-concept windows consistently follow target.
    Inverse of PRECEDES; stored explicitly for directed-graph traversal."""

    CO_OCCURS = "co_occurs"
    """Co-evidence: source- and target-concept window sets overlap above a
    Jaccard threshold."""

    GRANGER_PRECEDES = "granger_precedes"
    """Causal-suggestive: Granger causality / transfer entropy from source
    signal modality to target.  Framed as *suggestive*, not proof."""

    FEEDBACK = "feedback"
    """Equilibrium equation coefficient; not necessarily a finite-window or causal effect.
    Loops are allowed — a feedback loop is a pair of opposite FEEDBACK edges.
    ``provenance`` states the evidence and interpretation in schema 2.1."""

    LAGGED = "lagged"
    """Cross-window regression coefficient, not necessarily a causal effect.
    ``lag_sec`` carries the delay and ``effect`` the signed coefficient."""

    ACTS_ON = "acts_on"
    """Treatment-to-concept coefficient; causal identification is not implied.
    The source is always an intervention node and the target always a concept node."""


class InterventionKind(str, Enum):
    """How a clinician acts on the patient through a set-point."""

    TCI_TARGET = "tci_target"
    """Target-controlled infusion target concentration (e.g. propofol, remifentanil)."""

    INFUSION_RATE = "infusion_rate"
    """Manually titrated infusion rate (vasopressors, vasodilators, inotropes)."""

    VENTILATOR_SETTING = "ventilator_setting"
    """A ventilator set-point, named by ``InterventionNode.parameter`` (e.g. ``set_rr``)."""


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidencePointer(BaseModel):
    """Back-pointer from a KG node to the source signal window that activated it.

    Evidence traceability is a first-class architectural property of CLBench,
    not an afterthought.  Every KGNode must carry at least one EvidencePointer.
    """

    record_id: str = Field(
        ...,
        description="Source record: MIMIC-IV-WDB record id (e.g. '3000063') or VitalDB patient id",
    )
    modality: str = Field(
        ..., description="Signal modality that produced this activation: ECG | PPG | RESP | ABP"
    )
    window_start_sec: float = Field(
        ..., ge=0.0, description="Window start, seconds from record start"
    )
    window_end_sec: float = Field(..., gt=0.0, description="Window end, seconds from record start")
    activation_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="CBM sigmoid output for this concept in this window",
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class KGNode(BaseModel):
    """A (SNOMED concept, persistence tier) node in the patient KG.

    Node identity is the (snomed_code, persistence_tier) pair — the same
    SNOMED concept at different persistence tiers becomes two distinct nodes.
    node_id is a stable UUID assigned at construction time.
    """

    node_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable UUID for this node instance",
    )
    snomed_code: str = Field(..., description="SNOMED CT concept code, e.g. '3424008'")
    preferred_term: str = Field(..., description="SNOMED preferred term, e.g. 'Tachycardia'")
    persistence_tier: PersistenceTier
    evidence: list[EvidencePointer] = Field(
        default_factory=list,
        description="Source signal windows that support this node",
    )
    external_mappings: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Future-proof hook for V2 cross-ontology integration "
            "(e.g. {'ICD10CM': 'R00.0', 'CUI': 'C0039231'}).  "
            "Populated only with {'SNOMEDCT': snomed_code} in V1."
        ),
    )

    @property
    def identity_key(self) -> tuple[str, PersistenceTier]:
        """Canonical identity: (snomed_code, persistence_tier)."""
        return (self.snomed_code, self.persistence_tier)


class InterventionAction(BaseModel):
    """One timestamped clinician action on a set-point (the treatment-side evidence pointer)."""

    record_id: str = Field(..., description="Record the action belongs to")
    t_sec: float = Field(..., ge=0.0, description="Time of the action, seconds from record start")
    value_before: float = Field(..., description="Set-point before the action (0 = off)")
    value_after: float = Field(..., description="Set-point after the action (0 = stopped)")

    model_config = {"frozen": True}


class InterventionNode(BaseModel):
    """A treatment the patient received, identified by (agent, kind, parameter).

    Intervention nodes live in ``PatientKG.interventions``, separate from concept nodes, and
    connect to concepts only through ACTS_ON edges.
    """

    node_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable UUID for this node instance",
    )
    agent: str = Field(..., description="Drug or device, e.g. 'remifentanil', 'ventilator'")
    kind: InterventionKind
    parameter: str | None = Field(
        None, description="Set-point name when the agent has several, e.g. 'set_rr'"
    )
    unit: str | None = Field(None, description="Unit of the set-point values, e.g. 'ng/mL'")
    actions: list[InterventionAction] = Field(
        default_factory=list,
        description="Timestamped actions, in time order",
    )
    external_mappings: dict[str, str] = Field(
        default_factory=dict,
        description="Cross-ontology codes for the agent (e.g. {'RXNORM': '...'})",
    )

    @property
    def identity_key(self) -> tuple[str, InterventionKind, str | None]:
        """Canonical identity: (agent, kind, parameter)."""
        return (self.agent, self.kind, self.parameter)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class InputArtifact(BaseModel):
    """Content-addressed input table; row indices refer to its stored order."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_rows: int = Field(ge=0)
    model_config = {"frozen": True}


class EvidenceRecord(BaseModel):
    """Replayable fit or coactivation calculation, shared by multiple edges.

    Half-open row ranges name exact response rows, not node activation windows.
    Lagged predictor rows are determined by the stored lag count and row order.
    """

    record_id: str
    method: Literal["lagged_ols", "pairwise_complete_jaccard"]
    input_artifact: InputArtifact
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_ranges: dict[str, list[tuple[int, int]]]
    settings: dict[str, Any]
    estimates: dict[str, Any]

    @model_validator(mode="after")
    def check_rows(self):
        for name, ranges in self.row_ranges.items():
            previous = 0
            for start, stop in ranges:
                if start < previous or stop <= start or stop > self.input_artifact.n_rows:
                    raise ValueError(f"invalid or overlapping row range for {name}: {(start, stop)}")
                previous = stop
        return self


class EdgeProvenance(BaseModel):
    """Meaning and evidence of an association; not a causal certificate."""

    evidence_record_id: str
    refit_record_ids: list[str] = Field(default_factory=list)
    evidence_kind: Literal["observational_fit", "coactivation"]
    causal_status: Literal["not_identified"] = "not_identified"
    source_variable: str
    target_variable: str
    coefficient_kind: Literal["equilibrium", "lagged", "input_equilibrium", "jaccard"]
    raw_coefficient: float = Field(allow_inf_nan=False)
    multiplier: float = Field(allow_inf_nan=False)
    source_unit: str
    effect_semantics: str
    time_semantics: str
    reliability_semantics: str
    model_config = {"frozen": True}


class KGEdge(BaseModel):
    """A typed, weighted edge between two KG nodes with evidence references."""

    edge_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_node_id: str = Field(
        ..., description="node_id of the source KGNode (or InterventionNode for ACTS_ON)"
    )
    target_node_id: str = Field(..., description="node_id of the target KGNode")
    edge_type: EdgeType
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised edge strength [0, 1]; the magnitude of ``effect`` when present",
    )
    evidence_window_ids: list[str] = Field(
        default_factory=list,
        description="References to EvidencePointer record_id+window combos that support this edge",
    )
    effect: float | None = Field(
        None,
        description="Signed fitted coefficient; interpretation is in provenance when available",
    )
    lag_sec: float | None = Field(
        None, ge=0.0, description="Regressor lag in seconds (LAGGED / ACTS_ON)"
    )
    reliability: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of split-half / bootstrap refits that reproduce this edge",
    )
    provenance: EdgeProvenance | None = Field(
        None, description="Schema 2.1: exact fit/coactivation evidence and association semantics"
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Patient graph
# ---------------------------------------------------------------------------


class PatientKG(BaseModel):
    """Container for one patient's concept-grounded knowledge graph.

    Invariants:
      - ACTS_ON edges run from an intervention node to a concept node.
      - Every other edge connects two concept nodes in ``nodes``.
      - concept_vocabulary_version tracks which SNOMED vocabulary was used.
    """

    patient_id: str = Field(..., description="MIMIC-IV subject_id, VitalDB patient id, or equivalent")
    hadm_id: str | None = Field(None, description="Hospital admission ID if available")
    nodes: list[KGNode] = Field(default_factory=list)
    edges: list[KGEdge] = Field(default_factory=list)
    interventions: list[InterventionNode] = Field(
        default_factory=list,
        description="Treatments the patient received (schema 2.0)",
    )
    evidence_records: dict[str, EvidenceRecord] = Field(default_factory=dict)
    build_metadata: dict[str, Any] = Field(default_factory=dict)
    concept_vocabulary_version: str = Field(
        default="1.0",
        description="Version string of the ConceptVocabulary used to build this graph",
    )
    schema_version: str = Field(
        default="1.0",
        description="'1.0' = V1 concept graph; '2.0' adds intervention nodes and "
        "feedback / lagged / acts_on edges; '2.1' records observational edge provenance",
    )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def node_by_id(self, node_id: str) -> KGNode | None:
        """Return the concept node with the given node_id, or None."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def intervention_by_id(self, node_id: str) -> InterventionNode | None:
        """Return the intervention node with the given node_id, or None."""
        for node in self.interventions:
            if node.node_id == node_id:
                return node
        return None

    def nodes_by_snomed(self, snomed_code: str) -> list[KGNode]:
        """Return all nodes (across tiers) for a given SNOMED code."""
        return [n for n in self.nodes if n.snomed_code == snomed_code]

    def edges_from(self, node_id: str) -> list[KGEdge]:
        """Return all edges originating from the given node."""
        return [e for e in self.edges if e.source_node_id == node_id]

    def edges_to(self, node_id: str) -> list[KGEdge]:
        """Return all edges targeting the given node."""
        return [e for e in self.edges if e.target_node_id == node_id]

    def validate_edge_refs(self) -> list[str]:
        """Return a list of validation errors for dangling or mistyped edge references."""
        node_ids = {n.node_id for n in self.nodes}
        intervention_ids = {n.node_id for n in self.interventions}
        errors: list[str] = []
        for key, record in self.evidence_records.items():
            if key != record.record_id:
                errors.append(f"Evidence key {key!r} differs from record id {record.record_id!r}")
        for edge in self.edges:
            if edge.provenance is not None:
                refs = [edge.provenance.evidence_record_id, *edge.provenance.refit_record_ids]
                for ref in refs:
                    if ref not in self.evidence_records:
                        errors.append(f"Edge {edge.edge_id}: unknown evidence record {ref!r}")
            elif self.schema_version == "2.1":
                errors.append(f"Edge {edge.edge_id}: schema 2.1 requires provenance")
            if edge.edge_type == EdgeType.ACTS_ON:
                if edge.source_node_id not in intervention_ids:
                    errors.append(
                        f"Edge {edge.edge_id}: acts_on source {edge.source_node_id!r} "
                        "is not an intervention node"
                    )
                if edge.target_node_id not in node_ids:
                    errors.append(
                        f"Edge {edge.edge_id}: acts_on target {edge.target_node_id!r} "
                        "is not a concept node"
                    )
                continue
            if edge.source_node_id not in node_ids:
                errors.append(f"Edge {edge.edge_id}: source {edge.source_node_id!r} not in nodes")
            if edge.target_node_id not in node_ids:
                errors.append(f"Edge {edge.edge_id}: target {edge.target_node_id!r} not in nodes")
        return errors
