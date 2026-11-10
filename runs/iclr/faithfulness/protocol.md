# Track C: node faithfulness tests that can fail. Pre-registered protocol

Written 2026-09-24 22:45 local time, before any test or val patient was run through the edit pipeline. The
runner (`scripts/51_faithfulness_cf.py`) refuses to run unless this file's sha256 equals the value in
`protocol.sha256`, and it records the sha256 in `run_info.json`, `summary.json` and `report.md`. Any
later deviation is listed in the report with its reason.

**Seen before registration.**
- Wave-1 reference-standard results (`runs/iclr/reference/report.md`): learned heads vs monitor
  numerics, the PPG-tachycardia "ECG access" finding, and the finding that pressure concepts are not
  inferred without ABP.
- A development run on 2 TRAIN-split patients (82924339, 81515145; outputs in `dev/`). It was used to
  debug the code. Its outcome tables were seen. Design changes made after that run:
  1. Rule-detector orientation bug in C1 fixed.
  2. ABP "neutral" edits restricted to edits that leave the ground-truth label unchanged. A pulse-pressure
     change moves the sub-window means that define the ABP rule.
  3. ABP normalisation factor bounded to the warp-factor range.
  4. A secondary coherent-normalisation pointer operation added (`norm_coh`).
  5. Artifact amplitude raised from 1.5 to 3.0 SD, with the interference frequency randomised per key.
     The autocorrelation rule is not moved by interference that is a harmonic of the true rate, so a
     fixed 150/min tone on a 75/min rhythm is not a positive control.
  6. A rank-based variant of v1 ERASER added.

  No criterion threshold was changed after the dev run. The thresholds below were coded before it.

## 1. Question

v1 reported ERASER comprehensiveness and sufficiency for CLBench nodes. `src/clbench/eval/eraser.py`
re-ranks *stored* activations with rationale = the above-threshold windows, so both metrics hold by
construction: sufficiency is identically 0 and comprehensiveness is >= 0. Track C replaces them with
tests that edit the waveform, re-run the model, and have known physiological ground truth. Every test
is shown to fail for at least one baseline.

A node is **faithful** when three things hold:
- (a) its concept probability follows the physiological quantity the concept names (rate or pressure
  level) in the expected direction and across the clinical threshold;
- (b) the signal inside its evidence pointers is what drives it;
- (c) the modality it claims as its source is the one it depends on.

## 2. Data and models

- **Data.** MIMIC-IV-WDB test and val patients with processed windows: 30-s windows, 15-s hop,
  62.47 Hz. ECG, PPG and RESP are z-scored per segment. ABP input is (mmHg - 85)/20. Only v1's
  windows are used: `quality_ok` and v1 channel count.
- **Per-window index.** Rule estimates, monitor references and stored v1 probabilities come from
  `runs/iclr/reference/per_window.parquet` (wave 1, alignment validated).
- **Trained model.** v1 seed-42 checkpoints (`checkpoints/seed_42`, md5-identical to `checkpoints/`):
  encoders -> CrossModalFusion (present-mask) -> 7 linear CBM heads. It is re-run on MPS for every
  edit, exactly as `clbench.fused_inference` does. Operating thresholds are v1's calibrated
  activation thresholds (`persistence_calibration.json`).
- **Baselines:**
  - `randhead`: every CBM head's weight vector is replaced by a random Gaussian direction of equal
    norm. Bias and feature standardisation are kept. This is the Adebayo et al. 2018 model-randomisation
    check on the head.
  - `randfull`: encoders, fusion and heads are all re-initialised.
  - `rule`: the v1 autocorrelation and sub-window rule detectors, applied to the edited home-modality
    signal.
  - Random models use *matched-rate* thresholds: the same fraction of active windows as the trained
    model on the same windows.

## 3. Design

### Test 1: counterfactual edits (dose-response)

**Keys.** Per patient and family, up to 25 keys sampled at random (seed 20260924) from windows whose
baseline is verified normal against the independent monitor:

| Family | Rule estimate | Agreement with monitor | Other conditions |
|---|---|---|---|
| ECG | 65-95 bpm | within 5 bpm of monitor HR | SQI >= 0.5 |
| PPG | 65-95 bpm | within 5 bpm of SpO2 pulse | SQI >= 0.5 |
| RESP | 13-19 /min | within 2 of monitor RR | SQI >= 0.5 |
| ABP | window mean 70-100 mmHg | within 5 mmHg of ABPm | sustained 10-s sub-window means in [67, 103]; SQI >= 0.5 |

Rate families also need the contiguous successor windows (+30 s and +60 s) as warp source.

**Edits.** A time-warp by factor f = target/baseline (linear interpolation, zero-phase anti-alias FIR
when f > 1, random sub-sample phase) multiplies every rate in the window by f. f must lie in
[0.3, 3.0].
- **HR targets:** 40, 45, 50, 55, 65, 70, 75, 85, 90, 95, 105, 110, 120, 130, 140, 150 bpm.
- **RR targets:** 6, 8, 9, 10, 11, 13, 14, 16, 18, 19, 21, 22, 24, 27, 30, 35 /min.
- **ABP:** MAP set to 45, 50, 55, 60, 62, 68, 70, 75, 85, 95, 100, 102, 108, 110, 120, 130 or
  140 mmHg.
  - Primary `mult`: the waveform is scaled in mmHg.
  - Secondary `add`: the level is shifted, keeping pulse pressure.
- **Modes:**
  - HR families: primary `coherent`, where every pulsatile modality at the key (ECG, PPG, ABP) is
    warped by the same f. This is one heart rate, as physiology requires. Secondary `home`: only the
    home modality is warped.
  - RESP: `home` only.
- **Sham:** the same operation with factor 1. For warps this is a fractional-delay resampling through
  the same interpolation. For ABP it is the identity: level edits have no resampling step, so the ABP
  sham is exactly 0 by construction.
- **Neutral:** the home-modality amplitude is multiplied by 0.6 and by 1.6, which leaves the rate
  unchanged. For ABP the pulsatile component is scaled by 0.6 and 1.6 while the window mean is kept.
  Only ABP edits whose sub-window-mean truth is unchanged are counted.

**Ground truth.**
- HR and RR: the target (baseline x f). This is valid because the baseline was verified against the
  monitor. A manipulation check reports the rule estimate on the edited window against the target.
- ABP: the rule's sustained sub-window statistics recomputed on the edited signal, i.e. min < 65
  (hypotension) and max > 105 (hypertension).

**Primary family and mode per concept:**
- tachycardia and bradycardia: ECG coherent;
- tachycardia_ppg: PPG coherent;
- tachypnea and bradypnea: RESP home;
- hypotension and hypertension: ABP mult.

**Criteria** (per concept and model; 95 % patient-cluster bootstrap CI, 2000 resamples):
- **C1 direction:** per key, Spearman rho(target, probability) x expected sign. The statistic is the
  mean over patients of per-patient means. PASS if CI lower bound > 0.
- **C2 threshold:** balanced accuracy of (probability >= operating threshold) against the
  counterfactual truth over all target edits. PASS if CI lower bound > 0.70.
- **C3 sham:** mean |p_sham - p_orig| has CI upper bound < 0.05, and the decision-flip fraction under
  sham has CI upper bound < 0.05.
- **C4 neutral:** decision-flip fraction under neutral edits has CI upper bound < 0.10.
- **Test-1 PASS = C1 and C2 and C3 and C4.**

For the rule baseline:
- C1 uses the detector's own estimate, oriented so larger means more concept-positive.
- C2 uses its binary decision.
- C3 and C4 use decision flips only.

**Secondary, reported but not part of pass/fail:**
- dose-response curves (per patient: `dose_response_per_patient.csv`);
- crossing value, where the patient-mean curve crosses the operating threshold;
- the secondary modes;
- off-target |dp| of concepts whose quantity was not edited;
- the manipulation check.

### Artifact arm (test 2b, positive control that the tests can fail)

- The first 10 selected keys per patient in each of the ECG, PPG and RESP families.
- Physiology is warped to targets 50, 70, 85, 120 and 140 bpm, or 8, 10, 16, 24 and 30 /min.
- A sinusoidal interference is then added to the home channel:
  - amplitude 3.0 x channel SD;
  - frequency ~ U(2.1, 2.7) Hz (126-162 /min) for ECG and PPG, and U(0.40, 0.60) Hz (24-36 /min) for
    RESP, drawn per key;
  - random phase.
- Truth stays physiological. C1 and C2 are computed for the rule detector and for the trained model,
  along with P(fire | truth negative).
- Prediction: the rule detector fails C2 for the tachy concepts, because it reads the interference as
  the rate. The trained model's result is a robustness finding, with no prediction.

### Test 2: evidence pointers (edit and re-run)

**Pointers** are v1 activation runs, the windows v1 turns into a node's `EvidencePointer`. A run is a
maximal run of home-modality windows with stored probability >= threshold, in window order.
- Per patient and concept, up to 6 runs of length 2-40 windows, sampled at random.
- Up to 12 pointer windows per run.
- Span = the run +- 8 windows.

**Operations** on each pointer window. Only the home-modality signal at the key is changed; the other
modalities are left as they are and the model is re-run.
- `neg`: home content replaced by a random same-patient window that is rule-negative, has stored
  probability < threshold/2 and lies outside any run.
- `pos` (matched sham swap): home content replaced by a rule-positive window from another run of the
  same concept.
- `mask`: home modality removed from fusion.
- `norm`: physiological normalisation of the home window. Warp to 80 bpm or 16 /min with f from that
  window's rule estimate, or ABP level scaled to MAP 85. f must lie in [0.3, 3].
- `sham`: factor-1 warp (identity for ABP).
- `norm_coh` (secondary, HR concepts): every pulsatile modality at the key normalised coherently.
- **Same-count random control:** negative content placed into the same number of random span windows
  outside the run.

**Outcomes:**
- R_op = fraction of pointer windows still >= threshold after op (pooled; patient cluster).
- Span presence = mean probability over the span windows.

**Criteria:**
- **P1 content:** R_pos - R_neg, paired per window. PASS if CI lower bound > 0.5.
- **P2 physiology:** R_sham - R_norm, paired. PASS if CI lower bound > 0.5.
- **P3 localisation:** span-presence drop from pointer `neg` minus drop from the random same-count
  control. PASS if CI lower bound > 0.
- **Test-2 PASS = P1 and P2 and P3.**

**Baseline:** `randhead` pointers, i.e. runs of the randomised head at its matched-rate threshold.
- These are drawn from 2 random blocks of 160 home windows per patient and concept, with at most 3
  runs per block.
- `neg` and `pos` pools use the same rule condition with the randomised head's own activation.

**v1 ERASER** (comprehensiveness, sufficiency, insertion - deletion) is reported on the same blocks
for trained and `randhead`, on probabilities and on within-block percentile ranks. The expectation
from its construction is that sufficiency is 0 for every model and that rank-ERASER depends only on
the rationale size.

### Test 3: modality masking

- **Keys:** per patient and concept, up to 10 keys active under the stored v1 probability plus 10
  random home keys.
- **Operations:** each present modality is masked in turn (removed from the fusion present-mask, as
  v1 does for an absent modality). A keep-only-home variant is also run.

**Criteria:**
- **M1 home dominance:** for every other modality m with >= 5 patients, mean paired
  (|dp_mask_home| - |dp_mask_m|) has CI lower bound > 0.
- **M2 home sufficiency:** agreement of the keep-only-home decision with the full-model decision has
  CI lower bound > 0.85.
- **Test-3 PASS = M1 and M2.** Models: trained, randhead, randfull.
- Wave 1 suggested that tachycardia_ppg reads the ECG through fusion. M1 tests this directly.

## 4. Can-fail requirement and multiplicity

- A test counts as able to fail if at least one baseline (randhead, randfull, rule under artifact)
  fails its composite criterion for at least one concept. This is reported per concept.
- Criteria are per concept with unadjusted 95 % CIs. There are 7 concepts x 3 tests. This is a
  measurement study, and the tables report every CI.
- Patients with no eligible keys for a family simply do not contribute. Counts are reported.

## 5. Compute

MPS, single process, about 15 s per patient (53 patients).
