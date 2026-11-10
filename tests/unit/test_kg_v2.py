"""Unit tests for the CLBench v2 graph builder (schema 3.0): pointer rules, tiers, identification."""

from __future__ import annotations

import numpy as np
import pytest

from clbench import kg_construct as kc
from clbench import kg_v2 as kv
from clbench.edges import typed_temporal as tt
from clbench.persistence import CalibrationConfig, TierThresholds


def _calib() -> CalibrationConfig:
    th = {c: TierThresholds(c, 15.0, 45.0, 0.5, 10, "test") for c in tt.CONCEPTS if "capnia" not in c}
    return CalibrationConfig(concepts=th, window_stride_sec=15.0, tier_percentiles=(33.0, 67.0))


def _series(T: int, on: list[tuple[int, int]], unobs: list[int] = ()) -> np.ndarray:
    x = np.zeros(T, np.int8)
    for s, n in on:
        x[s:s + n] = 1
    for i in unobs:
        x[i] = -1
    return x


def test_valid_onsets_need_two_observed_off_steps():
    x = _series(20, [(0, 2), (5, 1), (10, 3)], unobs=[8])
    # run at 0 has no history; run at 5 has x[3]=x[4]=0; run at 10 has x[9]=0 but x[8]=-1
    assert kv.valid_onsets(x).tolist() == [5]


def test_association_pairs_state_and_onset_bins():
    T = 100
    a = _series(T, [(10, 30)])  # A on 10..39, valid onset at 10
    b = _series(T, [(20, 2), (60, 2)])  # B onsets at 20 and 60
    p = kv.association_pairs(a, b)
    got = {(int(s), int(t), int(lag), lab) for s, t, lag, lab in zip(p.src, p.tgt, p.lag, p.label)}
    # B onset 20: A covers step 18 (state) and A onset 10 steps earlier (onset_1-3min)
    # B onset 60: A not on at 58; A onset 50 steps earlier (> 40): no pair
    assert got == {(0, 0, 10, "state+onset_1-3min")}


def test_association_onset_window_edges():
    T = 120
    a = _series(T, [(10, 1)])
    for d, expect in ((1, False), (2, True), (4, True), (40, True), (41, False)):
        b = _series(T, [(10 + d, 1)]) if d > 1 else _series(T, [(11, 1)])
        p = kv.association_pairs(a, b)
        assert (len(p) > 0) == expect, d


def test_policy_and_response_pairs():
    T = 100
    a = _series(T, [(30, 5)])
    bins = np.array([31, 36, 80])
    p = kv.policy_pairs(a, bins)
    # bin 31: A on at 30 (state) and onset 1 step earlier; bin 36: A off at 35, onset 6 steps earlier
    assert sorted(zip(p.tgt.tolist(), p.label)) == [(31, "state+onset_0-1min"), (36, "onset_1-3min")]
    b = _series(T, [(40, 2), (90, 1)])
    r = kv.response_pairs(np.array([30, 34, 36]), b)
    # onset 40: bins 30 (lag 10) and 34 (lag 6); 36 is lag 4 < 5. onset 90: bins too far (>= 54)
    assert sorted(zip(r.src.tolist(), r.lag.tolist())) == [(30, 10), (34, 6)]


def test_tiers_match_kg_construct_nodes():
    calib = _calib()
    T = 200
    S = np.zeros((T, 8), np.int8)
    S[:, 0] = _series(T, [(5, 1), (20, 3), (50, 10)])  # 15 s, 45 s, 150 s -> transient, episodic, persistent
    case = kv.CaseInput(pid="p", S=S, ws=np.arange(T) * 15.0, chg=np.zeros((T, 4, 2), np.int8), events=[],
                        vitals={})
    nodes, eps, run_ids = kv.concept_nodes(case, calib)
    tach = [n for n in nodes if n.snomed_code == kv.SNOMED["tachycardia"]]
    assert sorted(n.persistence_tier.value for n in tach) == ["episodic", "persistent", "transient"]
    assert [e.tier.value for e in eps if e.concept == "tachycardia"] == ["transient", "episodic", "persistent"]
    # identical to kg_construct's own node construction
    act = kc.ConceptActivation(kc.ConceptCatalogEntry("NUM", "tachycardia", "x", "x"), (S[:, 0] == 1).astype(float),
                               np.arange(T) * 15.0, np.arange(T) * 15.0 + 30.0, "p")
    ref = kc._nodes_for_concept(act, calib.concepts["tachycardia"], 15.0)
    assert sorted((n.persistence_tier, len(n.evidence)) for n in ref) == sorted(
        (n.persistence_tier, len(n.evidence)) for n in tach)
    # fallback thresholds for the EtCO2 concepts, as kg_construct
    assert kv.thresholds_for("hypocapnia", calib).episodic_max_sec == 600.0


def _rel(**kw) -> kv.PopulationRelation:
    base = dict(relation_id="r", edge_type="association", level="concept", source="hypertension",
                target="tachycardia", source_kind="concept", target_kind="concept", sign=1, estimand="omega",
                estimate=0.4, ci_lo=0.3, ci_hi=0.5, identification_status="identified", identification_basis="t",
                source_file="x")
    base.update(kw)
    return kv.PopulationRelation(**base)


def test_build_case_graph_pointers_and_refs():
    calib = _calib()
    T = 200
    S = np.zeros((T, 8), np.int8)
    S[:, tt.CONCEPTS.index("hypertension")] = _series(T, [(50, 20)])
    S[:, tt.CONCEPTS.index("tachycardia")] = _series(T, [(60, 3)])
    chg = np.zeros((T, 4, 2), np.int8)
    ws = np.arange(T) * 15.0
    # remifentanil up at 55*15+20 s -> bin 55 ([ws+15, ws+30))
    t_ev = 55 * 15.0 + 20.0
    chg[55, tt.ACTIONS.index("remifentanil"), 0] = 1
    events = [{"agent": "remifentanil", "kind": "tci_target", "t_sec": t_ev, "value_before": 2.0, "value_after": 3.0}]
    rels = [_rel(),
            _rel(relation_id="p", edge_type="policy", target="remifentanil_up", target_kind="action",
                 action="remifentanil", direction="up"),
            _rel(relation_id="q", edge_type="response", source="remifentanil_up", source_kind="action",
                 target="tachycardia", action="remifentanil", direction="up", identification_status="not identified")]
    case = kv.CaseInput(pid="p", S=S, ws=ws, chg=chg, events=events, vitals={})
    kg = kv.build_case_graph(case, rels, calib)
    assert kg.validate_refs() == []
    by = {e.relation_id: e for e in kg.edges}
    assert set(by) == {"r", "p", "q"}
    assert all(e.pointers for e in kg.edges)
    assert by["q"].pointers[0].lag_sec == 5 * 15.0
    assert by["p"].pointers[0].target_ref == "act0.0"
    # population effect carried, no patient weight
    assert by["r"].population_estimate == 0.4 and not hasattr(by["r"], "weight")
    d = kv.graph_counts(kg)
    assert d["edges_with_pointers"] == d["edges_node_level"] and d["relation_instances"] == 3


def test_concept_identification_rule():
    e = {"omega": 0.5}
    h_same = {"association|a->b": {"bh": True, "omega": 0.4}}
    h_not = {"association|a->b": {"bh": False, "omega": 0.4}}
    h_opp = {"association|a->b": {"bh": True, "omega": -0.4}}
    k = "association|a->b"
    assert kv.concept_identification(k, e, h_same, h_same, set())[0] == "identified"
    assert kv.concept_identification(k, e, h_same, h_not, set())[0] == "not identified"
    assert kv.concept_identification(k, e, h_same, h_opp, set())[0] == "sign-unresolved"
    assert kv.concept_identification(k, e, h_same, h_same, {k})[0] == "sign-unresolved"
    kr = "response|x_up->b"
    assert kv.concept_identification(kr, e, {kr: {"bh": True, "omega": 1}}, {kr: {"bh": True, "omega": 1}},
                                     set())[0] == "not identified"


def test_vital_identification_rule():
    assert kv.vital_identification({"30s": "correct", "2s": None})[0] == "identified"
    assert kv.vital_identification({"30s": "wrong", "2s": "correct"})[0] == "sign-unresolved"


@pytest.mark.parametrize("dirn,expect", [("up", 1), ("down", 0)])
def test_vital_event_pointers_itt(dirn, expect):
    t_c = np.arange(0, 3000, 15.0) + 15.0
    vital = np.full(len(t_c), 80.0)
    rel = _rel(level="vital", edge_type="response", source="remifentanil_up", source_kind="action",
               target="hr", target_kind="variable", action="remifentanil", direction=dirn, horizon_sec=120.0)
    # two ups 60 s apart: the second is not ITT-clean (change in the previous 180 s)
    t = np.array([1000.0, 1060.0])
    ptr = kv.vital_event_pointers(rel, t, np.array([2.0, 3.0]), np.array([3.0, 4.0]), t_c, vital)
    assert len(ptr) == expect
    if expect:
        assert ptr[0][0] == 0
