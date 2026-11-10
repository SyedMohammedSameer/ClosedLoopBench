# Track B pre-registration: typed, pooled temporal edges between concepts

Status: written 2026-09-24 before the confirmatory run. The sha256 of this file is written to
`confirmatory/provenance.json`, `confirmatory/results.json` and `confirmatory/report.md`.
Code: `src/clbench/edges/typed_temporal.py`, `scripts/47_typed_temporal_edges.py`, tests in
`tests/unit/test_typed_temporal.py`. The code sha256 values are also recorded in `provenance.json`.

## Pilot disclosure (exploratory, not confirmatory)
We ran exploratory pilots on random subsets of 250 and 400 VitalDB cases (seed 0; output in `runs/iclr/typed_edges/pilot/`).
They changed the design in four ways before this protocol was frozen:
1. **Fixed effects.** Pooled hazards are fit with patient fixed effects, i.e. within-patient hazard ratios. Without them, the
   circular-shift null admitted edges: in the 250-case pilot, 19-41 % of null p-values were below 0.05.
2. **Primary test.** The primary test is now a 1-df Wald test of the scalar summary omega. The pre-pilot choice was a joint
   Wald test over the lag-bin block. That test was unstable on sparse cells, e.g. p < 1e-7 with 8 exposed events.
3. **Minimum support.** An edge is testable only with at least 20 exposed events from at least 10 patients.
4. **Exposure.** The exposure for concept sources is "source concept on at the source lag".

We also added two secondary policy sign checks after seeing the pilot. They are labelled `secondary_post_pilot` and are not
primary. The confirmatory run uses **all** cases, including the pilot cases; there is no held-out case set.

## Data and concepts
* **Data.** VitalDB 30 s numerics at a 15 s stride (`data/processed_vitaldb/numerics`), every case with at least
  120 rows, plus the event table (`data/processed_vitaldb/events`).
* **Concepts.** There are 8 tri-state concepts (1 on, 0 off, -1 unobserved):
  * label_tachycardia, label_bradycardia (HR);
  * label_hypotension, label_hypertension (MAP);
  * label_tachypnea, label_bradypnea (RR);
  * hypercapnia EtCO2 > 45 and hypocapnia EtCO2 < 30 (EtCO2). EtCO2 counts as valid only in [15, 100]; otherwise it is
    unobserved.
* **Onsets.** An onset at step i means S[i] = 1 and S[i-1] = S[i-2] = 0, all observed. Step i is at risk for B when S_B[i]
  is observed and S_B[i-1] = S_B[i-2] = 0.
* **Actions.** There are 4 actions (propofol TCI Ct, remifentanil TCI Ct, phenylephrine rate, set ventilator RR), each with
  an up and a down direction:
  * A change is assigned to the 15 s bin [ws_i + 15, ws_i + 30).
  * TCI and ventilator starts and stops from or to 0 are excluded.
  * Phenylephrine starts are included.
* **Doses.** Dose covariates are the numerics columns as recorded. NaN means not recorded (value 0 plus a "recorded"
  indicator), so doses are never forward-filled past the end of a track.

## Edge families and designs
All families use one pooled discrete-time logistic hazard per outcome with patient fixed effects. Non-event rows are
subsampled at 0.2 with weight 5, and every event row is kept. There is a ridge of 1 on the shared coefficients.

All designs share two sets of adjustment covariates:
* **Phase:** time since anaesthesia start (pre, 0-15 min, 15-60 min, 1-3 h), time since the recording started
  (0-10 min, 10-30 min, 30-60 min, 1-2 h, 2-4 h) and time to the end of the recording (<=15 min, 15-30 min, 30-60 min).
* **B history:** steps since B was last on (never, <=2 min, 2-10 min, 10-30 min).

The three families:
* **Association A -> B** (48 ordered pairs; A and B on different vitals).
  * Outcome: B onset hazard.
  * Sources: all 6 cross-vital concepts jointly. Each source has 5 features: unobserved and on at lag 2 steps (30 s,
    no window overlap), plus onsets in [2, 4], [5, 12] and [13, 40] steps (0-1, 1-3 and 3-10 min).
  * Adjustments: phase, B history, doses at lag 1, and indicators of any recent up or down change per action in the
    last 10 min.
* **Policy A -> action change (a, d)** (8 concepts x 8 action-directions = 64).
  * Outcome: a change of a in direction d in bin i.
  * At risk: TCI or ventilator running (recorded dose > 0) at i-1. For phenylephrine "up", every row is at risk; for
    phenylephrine "down", rate > 0 at i-1.
  * Sources: all 8 concepts, with the same 5 features at lag 1.
  * Adjustments: phase, time since the last change of a (never, <=1 min, 1-5 min, 5-15 min) and doses at lag 1.
* **Response action change (a, d) -> B** (8 action-directions x 8 concepts = 64).
  * Outcome: B onset hazard.
  * Sources: indicators of a change of (a, d) in [5, 12] and [13, 40] steps. These lags start at 1 min, at or beyond
    the latency.
  * Adjustments: phase, B history, doses at lag 41, each concept's state at lag 41, and B's underlying vital (z-scored)
    at lag 41. These are pre-treatment covariates.

## Edge statistic and admission
* **Omega.** omega is the average log hazard ratio of the source's lag-pattern coefficients over its exposed rows.
  For a concept source, exposed means the source is on at the source lag. For an action source, exposed means a change
  in [5, 40].
* **Uncertainty.** A patient-cluster score bootstrap with B = 500 multinomial patient resamples. Each replicate is a
  one-step Newton update from the full fit, with the Hessian held fixed (Kline & Santos 2012).
* **Primary test.** p_omega = 2 Phi(-|omega / SE_boot|). It equals 1 when the edge falls below the support minimum.
* **Rule 1.** Benjamini-Hochberg at q = 0.05 within each family (48, 64 or 64 edges).
* **Negative controls.** There are R = 19 cross-patient transplant replicates and R = 19 within-patient circular-shift
  replicates.
  * Transplant: a partner case (a derangement) supplies the source series at the same elapsed step. It is missing past
    the partner's end.
  * Shift: a random non-zero multiple of g, with g = max(41, ceil(max over concepts of the integrated autocorrelation
    time)), capped at T/2. This is the subgroup-lattice logic of `edges/controlled.py`.
  * In the association and policy families the null replaces all concept source series. In the response family it
    replaces the action-change series.
* **Admission.** A pooled edge is admitted iff rule 1 holds **and** z_omega^2 exceeds the maximum z_omega^2 of the same
  edge over all 19 transplant and all 19 shift replicates.
* **Typing.** Every edge carries its family as its type: association, policy or response.
* **Patient level.** A patient-level edge is only a deviation from the pooled edge.
  * Test: a score test of delta_p on the exposure indicator, with the pooled linear predictor as an offset and the
    patient intercept profiled out. It uses all at-risk rows, not the subsample.
  * Multiplicity: Benjamini-Yekutieli at q = 0.05 over all patient x edge tests of all three families.
  * Shrinkage: DerSimonian-Laird empirical-Bayes shrinkage of penalized per-patient MLEs of delta (prior sd 3).

## Comparators (same concept series; missing counts as off, as in v1)
* **v1 xcorr precedes/follows.** `clbench.edges.temporal.temporal_relation` with threshold 0.5, stride 15 s, maximum
  lag +-30 min, minimum xcorr 0.15 and minimum lag 15 s, applied to every unordered pair.
* **v1 granger_precedes.** `clbench.edges.granger.granger_precedes_score` with lags (1, 4, 12) and alpha 0.05, applied
  to every ordered pair. "v1" denotes the union of the two.
* **Pairwise Granger + BH (gbh).** An OLS F-test on the 15 s grid of B_t on its own lags (1, 2, 3-4, 5-12, 13-40)
  plus A's lags (2, 3-4, 5-12, 13-40), with BH at 0.05 over the patient's 48-pair family.
* **PCMCI+.** Tigramite with ParCorr on 1-min block means, tau 1-10 and pc_alpha 0.01. The p-value per pair is the
  Bonferroni-over-lag minimum, then BH at 0.05 over the 48 pairs. It runs on a random 400-case subset (seed 1).
  Its null is a group null: the HR and EtCO2 concepts are transplanted or shifted as a block, and the 32 cross-group
  pairs are the null pairs.
* **Comparison family.** All comparisons use the 48 cross-vital ordered pairs.

## Pre-specified evaluations
**(a) Held-out-future replication within patient.**
* **Splits.** H1 is the first half of each case's steps and H2 the second half.
* **Per-patient methods.** The metric is the fraction of H1 edges also present in H2 with the same direction. gbh
  additionally requires the same sign.
* **Chance baseline.** Random edges density-matched to the patient's H1 edge count, drawn from the patient's candidate
  pairs (both concepts' vitals observed in H1). The chance level is
  sum_p |E1_p| |E2_p| / m_p / sum_p |E1_p|. Both the rate and the chance level get patient-bootstrap CIs (1000 draws).
* **Typed pooled edges.** Pooled edges admitted by rule 1 on the pooled H1 fit are instantiated in patient p's H1 when
  the source has at least 1 episode (or at least 1 change for an action) and the target has at least 1 onset (or at
  least 1 change) in that half. They replicate when the same instantiation, with the same sign, holds using the pooled
  H2 fit. Rule 2 is not applied on halves: there are no half-specific null replicates.
* **Also reported.** Pooled-level H1 -> H2 replication, and replication of the patient-level deviation edges
  (BY computed on each half).

**(b) Held-out-future predictive log-likelihood of concept onsets.** This uses the association family on H2 rows.
* **Models.** All are fit on H1 with a non-fixed-effects pooled logistic, the same subsample and the same covariates:
  * base: no sources;
  * pooled: all 6 sources;
  * typed: only the sources admitted on H1 (rule 1);
  * v1-edge features: source lag features gated by the patient's v1 H1 edges (xcorr or Granger);
  * shrunk-patient: pooled plus EB-shrunk patient deviations estimated on H1;
  * per-patient: pooled plus unshrunk penalized per-patient deviations.
* **Metric.** Delta log-likelihood versus base in nats per H2 onset. There is a 95% patient-bootstrap CI (2000 draws),
  per target and over all targets. Paired differences: shrunk - pooled, shrunk - per-patient, typed - v1,
  pooled - v1, shrunk - v1.

**(c) False edges under the nulls.**
* **Pooled.** Per null replicate: the count admitted by rule 1 and by the full rule, where the full rule uses the other
  37 replicates leave-one-out. Also the fraction of testable null p_omega values below 0.05, and a KS test against
  uniform.
* **Per-patient graphs.** Transplant and shift replicate 0: false edges per graph among null pairs, for v1x, v1g, v1,
  gbh, PCMCI+ and the typed instantiation. Null-pooled admissions under the full rule are instantiated per patient.
* **Deviations.** Patient-level deviation edges (BY) under null replicates 0-1.

**(d) Pharmacological sign checks**, reported with the closed-loop caveat.
* **Call rule.** Correct if the 95% CI of omega excludes 0 in the expected direction; WRONG if it excludes 0 in the
  opposite direction; inconclusive otherwise.
* **Response checks:**
  * propofol up -> hypotension (+);
  * remifentanil up -> bradycardia (+);
  * remifentanil up -> hypotension (+);
  * vent RR up -> hypocapnia (+);
  * vent RR up -> hypercapnia (-);
  * phenylephrine up -> hypotension (-).
* **Policy checks:**
  * hypertension -> remifentanil up (+);
  * tachycardia -> remifentanil up (+);
  * hypotension -> phenylephrine up (+);
  * hypotension -> propofol down (+);
  * hypercapnia -> vent RR up (+).
* **Secondary, post-pilot:**
  * hypocapnia -> vent RR down (+);
  * hypertension -> propofol up (+).

**Graph size.**
* **Nodes.** Nodes per patient are concept x persistence tier. Tier thresholds come from
  `checkpoints/cbm/persistence_calibration.json`; hypercapnia and hypocapnia use the fallback 60 s / 600 s. Episode
  duration is (n - 1) x 15 + 30 s.
* **v1 edges.** v1 temporal edges are counted as emitted by `kg_construct`: 2 per tier pair for precedes/follows and
  1 per tier pair for granger_precedes, over all 28 concept pairs. The concept-pair counts are also reported.
* **Typed edges.** Typed edges per patient are the admitted pooled edges instantiated on the full case, by type, plus
  the deviation edges.

## Interpretation guardrails (fixed in advance)
* Association edges are conditional, within-patient temporal associations, not causal claims. Policy edges describe
  the recorded decision rule.
* Response-edge signs are confounded by indication in a closed loop. A WRONG or inconclusive call is reported as such,
  not tuned away.
* If the typed edges do not beat v1 on (a) or (b), we report that.
