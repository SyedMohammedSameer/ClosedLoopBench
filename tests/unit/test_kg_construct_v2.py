"""
Unit tests for schema 2.0 KG construction: learned FEEDBACK / LAGGED edges, intervention nodes
and ACTS_ON edges, and unchanged V1 behaviour when none of them are passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.kg_construct import (
    ConceptActivation,
    ConceptCatalogEntry,
    ConceptEffect,
    InterventionEffect,
    build_patient_kg,
    intervention_nodes_from_events,
)
from clbench.persistence import CalibrationConfig, TierThresholds
from clbench.schema import EdgeType, InterventionKind, PatientKG

N_WINDOWS = 60
STRIDE = 15.0
V1_TYPES = {EdgeType.PRECEDES, EdgeType.FOLLOWS, EdgeType.GRANGER_PRECEDES}
V2_TYPES = {EdgeType.FEEDBACK, EdgeType.LAGGED, EdgeType.ACTS_ON}


def _activation(key: str, modality: str, snomed: str, on: slice) -> ConceptActivation:
    probs = np.full(N_WINDOWS, 0.05)
    probs[on] = 0.9
    starts = np.arange(N_WINDOWS) * STRIDE
    return ConceptActivation(
        catalog=ConceptCatalogEntry(modality, key, snomed, key.title()),
        probs=probs,
        window_starts_sec=starts,
        window_ends_sec=starts + 30.0,
        record_id="vdb0010",
    )


@pytest.fixture()
def activations() -> list[ConceptActivation]:
    return [
        _activation("tachycardia", "ECG", "3424008", slice(10, 20)),
        _activation("hypotension", "ABP", "45007003", slice(12, 24)),
    ]


@pytest.fixture()
def calibration() -> CalibrationConfig:
    return CalibrationConfig(
        concepts={
            key: TierThresholds(
                concept=key,
                transient_max_sec=60.0,
                episodic_max_sec=600.0,
                activation_threshold=0.5,
                n_runs_observed=0,
                source="default",
            )
            for key in ("tachycardia", "hypotension")
        },
        window_stride_sec=STRIDE,
        tier_percentiles=(50.0, 90.0),
    )


def _nodes_by_term(kg: PatientKG) -> dict[str, str]:
    return {n.preferred_term: n.node_id for n in kg.nodes}


class TestV1Unchanged:
    def test_defaults_build_a_v1_graph(self, activations, calibration):
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False)
        assert kg.schema_version == "1.0"
        assert kg.interventions == []
        assert not {e.edge_type for e in kg.edges} & V2_TYPES
        assert all(e.effect is None for e in kg.edges)

    def test_legacy_edges_off_keeps_only_co_evidence(self, activations, calibration):
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              legacy_edges=False)
        assert kg.schema_version == "2.0"
        types = {e.edge_type for e in kg.edges}
        assert not types & V1_TYPES
        assert types <= {EdgeType.CO_OCCURS}


class TestLearnedEdges:
    def test_feedback_loop_is_two_signed_edges(self, activations, calibration):
        effects = [
            ConceptEffect("hypotension", "tachycardia", EdgeType.FEEDBACK, -0.6, reliability=0.9),
            ConceptEffect("tachycardia", "hypotension", EdgeType.FEEDBACK, 0.4),
        ]
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, concept_effects=effects, legacy_edges=False)
        ids = _nodes_by_term(kg)
        feedback = {(e.source_node_id, e.target_node_id): e for e in kg.edges
                    if e.edge_type == EdgeType.FEEDBACK}
        back = feedback[(ids["Hypotension"], ids["Tachycardia"])]
        forth = feedback[(ids["Tachycardia"], ids["Hypotension"])]
        assert (back.effect, back.weight, back.reliability) == (-0.6, 0.6, 0.9)
        assert (forth.effect, forth.weight) == (0.4, 0.4)
        assert kg.validate_edge_refs() == []

    def test_lagged_edge_carries_lag(self, activations, calibration):
        effects = [ConceptEffect("hypotension", "tachycardia", EdgeType.LAGGED, -0.2, lag_sec=15.0)]
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, concept_effects=effects)
        (lagged,) = [e for e in kg.edges if e.edge_type == EdgeType.LAGGED]
        assert lagged.lag_sec == 15.0

    def test_self_effects_skipped(self, activations, calibration):
        effects = [ConceptEffect("tachycardia", "tachycardia", EdgeType.LAGGED, 0.7, lag_sec=15.0)]
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, concept_effects=effects)
        assert not [e for e in kg.edges if e.edge_type == EdgeType.LAGGED]

    def test_effects_on_absent_concepts_ignored(self, activations, calibration):
        effects = [ConceptEffect("bradypnea", "tachycardia", EdgeType.FEEDBACK, 0.3)]
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, concept_effects=effects)
        assert not [e for e in kg.edges if e.edge_type == EdgeType.FEEDBACK]

    def test_non_learned_type_rejected(self, activations, calibration):
        effects = [ConceptEffect("hypotension", "tachycardia", EdgeType.CO_OCCURS, 0.3)]
        with pytest.raises(ValueError, match="FEEDBACK or LAGGED"):
            build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                             enable_granger=False, concept_effects=effects)


class TestInterventionEdges:
    EVENTS = [
        {"agent": "remifentanil", "kind": "tci_target", "t_sec": 270.8, "value_before": 0.0, "value_after": 3.0},
        {"agent": "remifentanil", "kind": "tci_target", "t_sec": 521.7, "value_before": 3.0, "value_after": 0.0},
    ]

    def test_acts_on_only_for_received_treatments(self, activations, calibration):
        interventions = intervention_nodes_from_events(self.EVENTS, record_id="vdb0010")
        effects = [
            InterventionEffect("remifentanil", InterventionKind.TCI_TARGET, "tachycardia", -3.0, lag_sec=60.0),
            InterventionEffect("norepinephrine", InterventionKind.INFUSION_RATE, "hypotension", -2.0),
        ]
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, interventions=interventions,
                              intervention_effects=effects, legacy_edges=False)
        (acts_on,) = [e for e in kg.edges if e.edge_type == EdgeType.ACTS_ON]
        assert acts_on.source_node_id == kg.interventions[0].node_id
        assert acts_on.target_node_id == _nodes_by_term(kg)["Tachycardia"]
        assert (acts_on.effect, acts_on.weight, acts_on.lag_sec) == (-3.0, 1.0, 60.0)
        assert kg.validate_edge_refs() == []

    def test_interventions_alone_mark_schema_v2(self, activations, calibration):
        interventions = intervention_nodes_from_events(self.EVENTS, record_id="vdb0010")
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, interventions=interventions)
        assert kg.schema_version == "2.0"
        assert len(kg.interventions) == 1 and len(kg.interventions[0].actions) == 2

    def test_json_round_trip(self, activations, calibration):
        interventions = intervention_nodes_from_events(self.EVENTS, record_id="vdb0010")
        effects = [InterventionEffect("remifentanil", InterventionKind.TCI_TARGET, "tachycardia", -3.0)]
        kg = build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, interventions=interventions,
                              intervention_effects=effects)
        assert PatientKG.model_validate_json(kg.model_dump_json()) == kg


class TestPrimaryNodes:
    def _two_tier_activations(self):
        # tachycardia: one short run (transient) and one long run (persistent) → two tier nodes
        probs = np.full(N_WINDOWS, 0.05)
        probs[2:3] = 0.9
        probs[20:50] = 0.9
        starts = np.arange(N_WINDOWS) * STRIDE
        tachy = ConceptActivation(ConceptCatalogEntry("ECG", "tachycardia", "3424008", "Tachycardia"),
                                  probs, starts, starts + 30.0, "vdb0010")
        return [tachy, _activation("hypotension", "ABP", "45007003", slice(12, 24))]

    def test_primary_attaches_once_to_the_longest_tier(self, calibration):
        acts = self._two_tier_activations()
        effects = [ConceptEffect("hypotension", "tachycardia", EdgeType.FEEDBACK, 0.6)]
        every = build_patient_kg("vdb0010", acts, calibration, window_stride_sec=STRIDE,
                                 enable_granger=False, concept_effects=effects, legacy_edges=False)
        primary = build_patient_kg("vdb0010", acts, calibration, window_stride_sec=STRIDE,
                                   enable_granger=False, concept_effects=effects, legacy_edges=False,
                                   learned_edges_on="primary")
        count = lambda kg: sum(e.edge_type == EdgeType.FEEDBACK for e in kg.edges)  # noqa: E731
        assert count(every) == 2 and count(primary) == 1
        (edge,) = [e for e in primary.edges if e.edge_type == EdgeType.FEEDBACK]
        target = primary.node_by_id(edge.target_node_id)
        assert target.persistence_tier == max(
            (n for n in primary.nodes if n.preferred_term == "Tachycardia"),
            key=lambda n: sum(e.window_end_sec - e.window_start_sec for e in n.evidence),
        ).persistence_tier

    def test_primary_co_evidence_is_one_pair_of_edges(self, calibration):
        acts = self._two_tier_activations()
        kg = build_patient_kg("vdb0010", acts, calibration, window_stride_sec=STRIDE,
                              enable_granger=False, legacy_edges=False, learned_edges_on="primary")
        co = [e for e in kg.edges if e.edge_type == EdgeType.CO_OCCURS]
        assert len(co) in (0, 2)  # both directions between the two primary nodes, or none

    def test_bad_option_rejected(self, activations, calibration):
        with pytest.raises(ValueError, match="learned_edges_on"):
            build_patient_kg("vdb0010", activations, calibration, window_stride_sec=STRIDE,
                             enable_granger=False, learned_edges_on="some")
