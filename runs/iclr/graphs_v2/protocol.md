# Track A pre-registration: CLBench v2 graph builder (schema 3.0) and graph-level evaluation

Written 2026-09-24 ~22:30 local time, before any v2 graph was built or evaluated. The sha256 of this file is
recorded in `provenance.json`, `results.json` and `report.md`. Code: `src/clbench/kg_v2.py`,
`scripts/50_build_kgs_v3.py`, tests `tests/unit/test_kg_v2.py` (code sha256 values go to `provenance.json`).

What was already known when this was written (disclosure): the wave-3 summaries in
`runs/iclr/typed_edges/confirmatory/report.md` (admitted pooled edges, family-level H1->H2 pooled
replication 12/28, 23/33, 9/21; mean full-rule null admissions per replicate 0.05-0.32 per family; v1
false edges per graph at null replicate 0), `runs/iclr/closed_loop_v2/*/typed_edges.csv` and
`verify/verification.md` (admission under 9 calibrators, null-vs-null admissions), and the v1 MIMIC graph
totals (20,192 edges, 0 with evidence). Per-edge half-split admissions (used below for identification
status) had not been looked at.

## 1. Inputs (no re-estimation of any population relation)

* **Concept-level relations**: `runs/iclr/typed_edges/confirmatory/results.json` (`pooled_edges_full`,
  `admitted == True`: 28 association, 33 policy, 20 response) and the H1/H2 fits in `cache/fits.jsonl`.
* **Vital-level relations**: `runs/iclr/closed_loop_v2/{30s_all,2s_fine}/typed_edges.csv`, arm `real`,
  `admitted == True`; robustness across calibrators from
  `closed_loop_v2/verify/typed_admission_by_null_{30s,2s}.csv`; null-vs-null admissions from
  `verify/null_vs_null_{30s,2s}.csv`.
* **VitalDB cohort**: the wave-3 cohort (`typed_edges/confirmatory/cache/cohort.npz`, 3,442 cases, 30 s
  windows at a 15 s stride, same concept definitions: 6 label concepts + hypercapnia EtCO2 > 45 and
  hypocapnia EtCO2 < 30, EtCO2 valid in [15, 100]). Window start times and action events are re-read
  from `data/processed_vitaldb/{numerics,events}`. The re-derived concept and action-change arrays are
  compared with `cohort.npz` (mismatching cases are reported, and the cohort arrays are used).
* **MIMIC**: the 167 v1 graphs in `data/kgs/` (built by `kg_construct` from v1 learned CBM activations).
  Their concept x tier nodes and per-run evidence pointers are reused verbatim; the model is NOT re-run
  (`fused_inference.py` over 167 patients' waveforms does not fit the ~60 min budget). Population
  relations for MIMIC are the association edges admitted by the exploratory MIMIC fit
  (`typed_edges/mimic_external/results.json`, 51 patients, v1 learned activations); MIMIC has no action
  data, so it has association edges only. Concept mapping: tachycardia = ECG tachycardia node (the PPG
  tachycardia node is not an endpoint), hypotension/hypertension = ABP nodes, tachypnea/bradypnea = RESP.

## 2. Nodes

* **Concept x tier nodes, exactly as `kg_construct` emits them**: `kg_construct._nodes_for_concept` on the
  0/1 concept series (unobserved counts as off, as in v1), `checkpoints/cbm/persistence_calibration.json`,
  `kg_construct`'s fallback (60 s / 600 s, threshold 0.5) for hypercapnia and hypocapnia, 15 s stride,
  window end = start + 30 s. Each activation run is one **episode** with an id, its tier node, start/end
  step and time, and an `onset_valid` flag (the run starts after two observed-off steps, the wave-3 onset
  definition). SNOMED codes: v1 registry codes; hypercapnia/hypocapnia get `local:` codes (no SNOMED code
  in the repository; not invented).
* **Action nodes**: `kg_construct.intervention_nodes_from_events` on the case's event rows (one node per
  agent/kind/parameter, every timestamped action kept). Actions are referenced by (node, index).
* **Variable nodes**: one per vital (hr, map, etco2) that an admitted vital-level relation uses and that is
  observed in the case.

## 3. Edges

Only population-admitted relations are instantiated. An edge carries its type (policy / response /
association), the population sign, estimand, estimate and 95% CI, and an identification status. **No
patient-specific weight** is estimated or stored (the pre-registered patient-specificity test was negative).

**Per-patient instantiation (evidence pointers).** A relation is instantiated in a patient iff it has at
least one evidence pair; each pair is a pointer (source ref, target ref, lag, exposure label). Pairs are
the patient's exposed outcome events in the wave-3 estimator's exposure/lag structure (step = 15 s):
* association A -> B: B episode starting at a valid onset i; A episode covering step i-2 (`state`), or
  starting at a valid A onset s with 2 <= i-s <= 40 (`onset_0-1min` [2,4], `onset_1-3min` [5,12],
  `onset_3-10min` [13,40]).
* policy A -> (action a, direction d): a change of (a, d) in bin i (wave-3 action definition: TCI and
  ventilator starts/stops from/to 0 excluded, 15 s bins); A episode covering step i-1 (`state`) or a valid
  A onset s with 1 <= i-s <= 40.
* response (a, d) -> B: a change of (a, d) in bin j; B valid onset at i with 5 <= i-j <= 40.
* vital-level response (closed-loop v2 relation): intention-to-treat events of the relation's action and
  direction (`EventSet.usable`, no change of the action in [t-180 s, t)), with the vital observed in the
  baseline window [t-30 s, t) and the outcome window [t+h-15 s, t+h+15 s] (30 s numerics). Lag = h.
* vital-level policy: the same event set, with the vital observed in [t-150 s, t-120 s] and [t-30 s, t).

Node-level edges connect the endpoint nodes of the pointers (tier node of each episode, the action node, or
the variable node); a relation instance can therefore produce several node-level edges, one per endpoint
node pair. For negative-signed relations the pointers are exposed events that occurred despite the lower
population hazard; pointers locate evidence, they do not make an edge patient-specific.

**Identification status** (of the sign under the edge type's interpretation), fixed now:
* vital-level: `identified` if admitted with the expected sign at >= 1 scale and never admitted with the
  opposite sign at either scale (run's own calibrator); `sign-unresolved` if admitted with the opposite
  sign at any scale. Attributes: scales admitted, per-scale call, fraction of the 9 calibrators admitting.
* concept-level association and policy: `identified` if the edge is BH-admitted (rule 1, q = 0.05 within
  family) in both the H1 and the H2 pooled fits with the full-data sign; `sign-unresolved` if either half is
  BH-admitted with the opposite sign, or the sign contradicts a pre-specified wave-3 sign check; otherwise
  `not identified`.
* concept-level response: never `identified` (the family failed its pre-specified pharmacological checks,
  0/6 correct, 2 wrong); `sign-unresolved` under the same two conditions as above; otherwise
  `not identified`.
* MIMIC association edges: the same half-split rule on the MIMIC fits.

## 4. Serialization

A new model `PatientKGv3` (schema_version "3.0") in `kg_v2.py`; `schema.py` is not modified. Top-level
`nodes` (concept x tier nodes as `schema.KGNode`), `interventions` (`schema.InterventionNode`) and
`edges` keep the names the FastAPI loader reads. One gzip JSON per patient. Check: `api.data._summarize_kg`
node counts equal the graph's, and `GET /patients/{id}/kg` returns 200 for sample graphs (api/ untouched;
`api.data.KG_DIR` is pointed at a scratch copy in the test process only).

## 5. Evaluation (pre-specified)

**E1 False edges per patient graph under the nulls (primary).** VitalDB concept-level relations: for each
of the 38 wave-3 null replicates (19 transplant, 19 circular shift; the same row maps, seeds 10000+r and
20000+r), the null population admission is recomputed with the full rule (BH within family and z^2 above
the maximum over the other 37 null replicates of both kinds, leave-one-out, as in wave 3), and each
null-admitted relation is instantiated with the pointer rule on the null-mapped series (association and
policy: all concept sources mapped; response: action-change series mapped). Every null edge is false.
Reported: mean false relation instances per graph and node-level edges per graph, averaged over all
replicates of each kind, with a patient-bootstrap 95% CI (1,000 draws; replicate-averaged per-patient
counts), and the fraction of (graph, replicate) pairs with any false edge. Prediction: < 0.5 false
relation instances per graph under both kinds. Comparators: v1 (xcorr precedes/follows union
granger_precedes) concept pairs among the 48 cross-vital pairs at replicate 0 (wave-3 per_patient.jsonl);
secondary, if compute allows, v1 on replicates 0-4 of both kinds for a random 300-case subset (seed 7).
Vital-level relations: the cross-patient null-vs-null admissions (56 pairs per scale) are the null; each
null-admitted relation would be instantiated per patient; secondary: the nc_random input at 30 s (8
calibrator pairs), instantiated on seeded random-time streams (`nc_random_times`, seed 5000 + case index).
MIMIC: the same computation on the 51-patient MIMIC cohort arrays and fits (38 replicates).

**E2 Edge-evidence coverage.** Fraction of emitted v2 edges with non-empty pointers (1.0 by construction,
stated as such), against v1 (MIMIC 0/20,192; VitalDB v1 edges carry no pointers by construction). The
informative quantity: among relation instances under wave-3's presence rule (source has an episode or
change anywhere in the case, target has an onset or change anywhere), the fraction with >= 1 in-window
evidence pair. **Pointer audit**: on 100 random VitalDB cases (seed 11) the concept series are re-derived
independently from the raw parquet columns, and every pointer is checked (source on / onset at the claimed
steps, target onset valid, lag inside the relation's window, action present in the event table at the
claimed time, vital observed in the claimed windows). Reported: fraction valid.

**E3 Graph size.** Per patient: concept x tier nodes, action nodes, variable nodes, relation instances and
node-level edges by type and by identification status. v1 recounted exactly as `kg_construct` emits it
(defaults of `scripts/05_build_kgs.py`: legacy edges and co-evidence on): nodes, PRECEDES, FOLLOWS,
GRANGER_PRECEDES, CO_OCCURS. VitalDB: v1 pair sets from wave-3 `per_patient.jsonl` (verified identical to
`kg_construct` in 40/40 cases), node tiers from `kg_construct` itself, co-evidence from
`co_evidence_relation`; the counts are verified against `build_patient_kg` on 30 random cases (seed 3).
MIMIC: counted from the 167 v1 graph files.

**E4 Case studies.** Three VitalDB cases drawn with `numpy.random.default_rng(2027)` from cases (sorted by
id) with >= 1 identified policy edge, >= 1 identified vital-level response edge and >= 1 association edge.
Full JSON plus a short readable summary.

## 6. Guardrails

* Policy edges describe the recorded decision rule. Association edges are conditional temporal
  associations. Response edges are not causal claims unless `identified`.
* Everything here is post hoc relative to the population estimates it reuses; the typed vital-level
  estimator is itself not confirmatory (closed_loop_v2 verification).
* Failures are reported as they come out.
