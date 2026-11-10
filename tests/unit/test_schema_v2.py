"""
Unit tests for schema 2.0: intervention nodes, feedback / lagged / acts_on edges, and
backward compatibility with V1 graphs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.kg_construct import intervention_nodes_from_events
from clbench.schema import (
    EdgeType,
    InterventionAction,
    InterventionKind,
    InterventionNode,
    KGEdge,
    KGNode,
    PatientKG,
    PersistenceTier,
)

V1_KG = {
    "patient_id": "80030009",
    "hadm_id": None,
    "nodes": [
        {
            "node_id": "n-tachy",
            "snomed_code": "3424008",
            "preferred_term": "Tachycardia",
            "persistence_tier": "episodic",
            "evidence": [
                {
                    "record_id": "80030009_0002",
                    "modality": "ECG",
                    "window_start_sec": 0.0,
                    "window_end_sec": 30.0,
                    "activation_score": 0.9,
                }
            ],
            "external_mappings": {"SNOMEDCT": "3424008"},
        },
        {
            "node_id": "n-hypo",
            "snomed_code": "45007003",
            "preferred_term": "Hypotension",
            "persistence_tier": "transient",
            "evidence": [],
            "external_mappings": {"SNOMEDCT": "45007003"},
        },
    ],
    "edges": [
        {
            "edge_id": "e1",
            "source_node_id": "n-hypo",
            "target_node_id": "n-tachy",
            "edge_type": "granger_precedes",
            "weight": 0.4,
            "evidence_window_ids": [],
        }
    ],
    "concept_vocabulary_version": "1.0",
}


def _concept(node_id: str) -> KGNode:
    return KGNode(node_id=node_id, snomed_code="3424008", preferred_term="Tachycardia",
                  persistence_tier=PersistenceTier.EPISODIC)


def _remifentanil(node_id: str = "i-remi") -> InterventionNode:
    return InterventionNode(node_id=node_id, agent="remifentanil", kind=InterventionKind.TCI_TARGET,
                            unit="ng/mL")


class TestBackwardCompatibility:
    def test_v1_graph_loads(self):
        kg = PatientKG.model_validate(V1_KG)
        assert kg.schema_version == "1.0"
        assert kg.interventions == []
        assert kg.edges[0].effect is None and kg.edges[0].lag_sec is None
        assert kg.validate_edge_refs() == []

    def test_v1_edge_type_values_unchanged(self):
        assert [e.value for e in (EdgeType.PRECEDES, EdgeType.FOLLOWS, EdgeType.CO_OCCURS,
                                  EdgeType.GRANGER_PRECEDES)] == [
            "precedes", "follows", "co_occurs", "granger_precedes"]

    def test_round_trip(self):
        kg = PatientKG.model_validate(V1_KG)
        assert PatientKG.model_validate_json(kg.model_dump_json()) == kg


class TestNewEdges:
    def test_new_edge_type_values(self):
        assert EdgeType.FEEDBACK.value == "feedback"
        assert EdgeType.LAGGED.value == "lagged"
        assert EdgeType.ACTS_ON.value == "acts_on"

    def test_feedback_loop_is_two_edges(self):
        a, b = _concept("a"), _concept("b")
        kg = PatientKG(
            patient_id="vdb0010",
            nodes=[a, b],
            edges=[
                KGEdge(source_node_id="a", target_node_id="b", edge_type=EdgeType.FEEDBACK, weight=0.4, effect=0.4),
                KGEdge(source_node_id="b", target_node_id="a", edge_type=EdgeType.FEEDBACK, weight=0.6, effect=-0.6),
            ],
            schema_version="2.0",
        )
        assert kg.validate_edge_refs() == []

    def test_lag_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            KGEdge(source_node_id="a", target_node_id="b", edge_type=EdgeType.LAGGED, weight=0.1, lag_sec=-15.0)

    def test_reliability_bounded(self):
        with pytest.raises(ValidationError):
            KGEdge(source_node_id="a", target_node_id="b", edge_type=EdgeType.LAGGED, weight=0.1, reliability=1.2)


class TestInterventions:
    def test_acts_on_valid(self):
        kg = PatientKG(
            patient_id="vdb0010",
            nodes=[_concept("c")],
            interventions=[_remifentanil()],
            edges=[KGEdge(source_node_id="i-remi", target_node_id="c", edge_type=EdgeType.ACTS_ON,
                          weight=0.8, effect=-3.0, lag_sec=60.0)],
        )
        assert kg.validate_edge_refs() == []
        assert kg.intervention_by_id("i-remi").agent == "remifentanil"
        assert kg.node_by_id("i-remi") is None

    def test_acts_on_from_concept_is_an_error(self):
        kg = PatientKG(
            patient_id="vdb0010",
            nodes=[_concept("c"), _concept("d")],
            edges=[KGEdge(source_node_id="c", target_node_id="d", edge_type=EdgeType.ACTS_ON, weight=0.5)],
        )
        assert any("is not an intervention node" in e for e in kg.validate_edge_refs())

    def test_concept_edge_to_intervention_is_an_error(self):
        kg = PatientKG(
            patient_id="vdb0010",
            nodes=[_concept("c")],
            interventions=[_remifentanil()],
            edges=[KGEdge(source_node_id="c", target_node_id="i-remi", edge_type=EdgeType.FEEDBACK, weight=0.5)],
        )
        assert any("not in nodes" in e for e in kg.validate_edge_refs())

    def test_action_time_non_negative(self):
        with pytest.raises(ValidationError):
            InterventionAction(record_id="vdb0010", t_sec=-1.0, value_before=0.0, value_after=1.0)


class TestInterventionNodesFromEvents:
    ROWS = [
        {"agent": "remifentanil", "kind": "tci_target", "t_sec": 970.9, "value_before": 0.0, "value_after": 1.5},
        {"agent": "ventilator", "kind": "vent_set_rr", "t_sec": 509.9, "value_before": 0.0, "value_after": 12.0},
        {"agent": "remifentanil", "kind": "tci_target", "t_sec": 270.8, "value_before": 0.0, "value_after": 3.0},
        {"agent": "nitroglycerin", "kind": "infusion_rate", "t_sec": 739.9, "value_before": 0.0, "value_after": 7.0},
    ]

    def test_groups_one_node_per_treatment(self):
        nodes = intervention_nodes_from_events(self.ROWS, record_id="vdb0010")
        keys = [n.identity_key for n in nodes]
        assert keys == [
            ("nitroglycerin", InterventionKind.INFUSION_RATE, None),
            ("remifentanil", InterventionKind.TCI_TARGET, None),
            ("ventilator", InterventionKind.VENTILATOR_SETTING, "set_rr"),
        ]

    def test_actions_sorted_and_units(self):
        nodes = {n.agent: n for n in intervention_nodes_from_events(self.ROWS, record_id="vdb0010")}
        assert [a.t_sec for a in nodes["remifentanil"].actions] == [270.8, 970.9]
        assert nodes["remifentanil"].unit == "ng/mL"
        assert nodes["nitroglycerin"].unit == "mL/h"
        assert nodes["ventilator"].unit == "/min"
        assert all(a.record_id == "vdb0010" for a in nodes["remifentanil"].actions)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="Unknown intervention kind"):
            intervention_nodes_from_events([{**self.ROWS[0], "kind": "bolus"}], record_id="vdb0010")
