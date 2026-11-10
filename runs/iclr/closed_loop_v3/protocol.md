# ClosedLoopBench 1.0 (closed-loop benchmark v3): pre-registered protocol

Written 2026-09-24 23:18 local time, before any v3 run on the full case sets. Before registration the v3 code
was run only on (a) synthetic unit tests, (b) the clinician-in-the-loop simulator (§9, development), and
(c) real-data smoke tests on 24 cases at 2 s and 60 cases at 30 s (PCMCI+ on 20), used to check run time
and errors. Their outputs went to the session scratchpad. They were not analysed, but two things were seen:
the 2 s smoke run's console printed its 24-case leaderboard counts, and a slice of the 60-case main table
(ventilator rows) was opened to check the format. The runner records this
file's sha256 in every output (`meta.json`, table headers). Any later change to this file changes the
hash. `protocol.sha256` holds the hash at registration.

Code at registration (sha256; byte copies in `registered_code/`, so the hashes can be checked later):
- `src/clbench/eval/closed_loop_bench.py` dc10e4843ac9b58b35fefe8e46092237a246bc35209cbb7a8fec83cdd003a806
- `src/clbench/edges/typed_temporal_pooled.py` 9d8c09b9cb4d26e4602ec6cc1abf087789221b92f8cc0e189007b5d49ed5e7e9
- `scripts/44_closed_loop_bench.py` c12ee4079024aea1bb1cc7aa0be0a62f2aab92664fe952ce873cd79ff33df61b
- `tests/unit/test_closed_loop_bench.py` c76f03b2cbff92db45d6fb2382b702d5d271b0966a00f02745b3df07d6304f32
- `src/clbench/bench/__init__.py` d31949fdfd6098087624b97612f12805f8b93adf44096604d6cf07e0e473936e
- `src/clbench/bench/ground_truth.py` 5cc1c6cecc5df34ebdf5b733ee14c2b469b2d81e6672d50edc08995132747904
- `src/clbench/bench/loader.py` 1f25a98effaab379aeab73c9b2f1f99cc84ae1f066160372b073c9db149ea19e
- `src/clbench/bench/nulls.py` 77c332b610306d161230079d2c998e38ced96e4c5e613e3f3b5b54cb0f7b28a1
- `src/clbench/bench/scoring.py` f05b9d56c0d80ace3db247fc81810b5a0eb68c78b662730f6dbc9aea7ee9b5f9
- `src/clbench/bench/sim.py` 687e31d7f5ee8dd3eec45800a85cc16316f7a6a7ddbd4c44e03774355d6c20f3
- `src/clbench/sim/clinician_loop.py` 47c5d932b59ee038a1a82e4b8fd775975b455085b8c2db2bcaf528a060e58dc6

(`src/clbench/sim/clinician_loop.py` is not owned by this track and was not modified; it is hashed because the simulator validation uses it.)

## 0. What v3 fixes (wave-3 verifier, runs/iclr/closed_loop_v2/verify/verification.md and
## runs/iclr/sim_closed_loop/verify/v3/)

1. Single-derangement calibration was fragile (2 s: 6–9 admitted and 0–1 wrong across 8 draws; one draw's
   null estimate of R08 was itself significant). v3 calibrates against K = 8 derangements, adds the
   between-draw variance to the SE, and requires stability across draws (§6).
2. Within-patient matched control pools keep a policy-dependent bias under feedback (simulator). v3 adds
   the pooled sequential-trial contrast `seqtrial` (§5), validated on the simulator first (§9). It did not
   pass the pre-stated gate, so it is a benchmark method, not the recommended estimator.
3. The v2 co-intervention sensitivities counted other actions' changes in [t−60, t+60] (post-treatment).
   v3 uses [t−60 s, t) only, in `*_coint`, `*_coadj` and in the `seqtrial` state.
4. Patient-specificity bootstrap CIs were not reproducible (rows appended in arrival order). v3 sorts every
   per-case list by case before it is stacked or resampled, and consumes per-case results in submission
   order (`imap`), so floating-point accumulations are reproducible too.
5. Registered code hashes could not be checked. v3 records code hashes in `meta.json` and keeps byte
   copies of the registered code.
6. The typed estimator's null rate was a different statistic from the other methods'. v3 applies one
   admission rule to every method and reports, for every method, the same null false-admission rates
   (§7) next to the uncalibrated single-arm null discovery counts.

## 1. Data and runs

- VitalDB numerics at 30 s windows (15 s stride), **all 3,442 cases** (`30s_all`), and the **500 cases**
  with 2 s numerics at **2 s** (`2s_fine`). Action events from `data/processed_vitaldb/events` (same
  agents as v1/v2). Dose masking after a track stops, as in v2.
- Seed 0; 200 case-bootstrap replicates (one weight matrix shared by every arm); 1,000 patient-bootstrap
  replicates for the patient-specificity test; VAR-LiNGAM 50 bootstrap replicates; PCMCI+ on the same
  seeded case subsample in every arm (30 s: 1,200 cases, pressor cases first; 2 s: all 500), as in v2.
- Commands (8 workers):
  ```
  scripts/44_closed_loop_bench.py --scale 2s  --cases fine --workers 8 --protocol runs/iclr/closed_loop_v3/protocol.md
  scripts/44_closed_loop_bench.py --scale 30s --cases all  --workers 8 --protocol runs/iclr/closed_loop_v3/protocol.md
  scripts/44_closed_loop_bench.py --combine-v3
  ```
- Outputs: `runs/iclr/closed_loop_v3/{30s_all,2s_fine}/` and `tables.md`. v1/v2 outputs are not modified.

## 2. Ground truth

Unchanged from v1/v2: 12 response relations (pharmacology / physiology sign) and 6 policy relations
(clinical-practice sign); primary horizons 60 s (ventilator → EtCO2) and 120 s (drugs); policy relations
use the 2-min pre-action trend. The table with sources is `clbench.bench.ground_truth`
(`ground_truth_table()`).

## 3. Arms (negative controls)

- `real`.
- `nc_random`: each action's steps moved to uniformly random times within the case (seeded per case as in v2).
- `nc_cross`, `nc_cross2` … `nc_cross8`: K = 8 cross-patient derangements, salts 991, 7919, 101, 202, 303,
  404, 505, 606 (draws 1–2 = v2's nc_cross / nc_cross2; 3–8 = the verifier's fresh draws).
Every method runs on every arm (PCMCI+ on the shared subsample in every arm).

## 4. Methods (table columns)

naive, naive_lvl, xcorr_v1 (Wave-2 signed |peak|), xcorr_dir (cause-leads peak), xcorr_pos (the v1
CLBench positive-peak rule), granger (pooled FE), varx, pcmciplus, varlingam, matched, kink — all
exactly as in v2 except the co-intervention windows of the `*_coint` / `*_coadj` sensitivities (§0.3) —
plus `seqtrial` (§5) and `typed_pooled` (§6). Sensitivity variants (never used for primary claims): v2's
`*_clean`, `*_v1`, `*_nobc`, `*_coint`, `*_coadj`, `kink_prealt`, and `seqtrial_nobc` (no bias
correction), `seqtrial_noco` (no co-intervention state).

## 5. Pooled sequential-trial contrast (`seqtrial`)

- Events: intention-to-treat titrations of the action in the relation's direction with no change of the
  action in the pre window [t − P, t), P = max(150 s, kink pre-window) (the v2 event set).
- Control bank: per case and action, the anchors of the v2 pool (eligible: no change in [g − P, g); untreated
  at g: no change within the decision interval [g, g + grid/2]; pre-t information only), thinned to at
  most 40 (30 s) / 120 (2 s) evenly spaced anchors per case.
- Pre-t state: per vital (MAP, HR, EtCO2) the last level [t−30 s, t), the recent level [t−10 s, t) (2 s
  scale; missing at 30 s and set to the bank mean), and the 2-min trend; the dose in force; elapsed case
  time; other actions' up / down changes in [t−60 s, t). Policy relations match on dose, elapsed time and
  co-interventions only (the vital trend is the outcome).
- Each event is matched to its k = 10 nearest controls (Euclidean, features z-scored on the bank; a missing
  non-target feature is set to the bank mean) **from other patients**, with Abadie–Imbens regression bias
  correction fitted on the pooled bank. Outcome: post-window mean at t + h minus the last pre-level
  (response) / the 2-min pre-trend (policy), times the event direction.
- Pooled estimate: event-weighted mean; case-bootstrap CI from the per-case sums of the event contrasts
  (matched controls held fixed; see limitations).

## 6. The v3 admission rule (every method) and the typed edge

For a method m and relation r at its primary horizon, with real estimate θ_R and null estimates θ_1..θ_K
(one per cross draw), all with paired case-bootstrap replicates:
- d = θ_R − mean_k θ_k;
- SE² = Var_boot(θ_R,b − mean_k θ_k,b) + Var_k(θ_k)/K (patient sampling + choice of derangement);
- CI = d ± 1.96 SE; p = two-sided normal;
- stable: the single-draw paired percentile CI of θ_R − θ_k excludes 0 with the sign of d in ≥ 75% of draws;
- **admitted** iff CI excludes 0 AND Benjamini–Yekutieli adjusted p ≤ 0.05 (family = the method's
  relations) AND stable AND ≥ 5 contributing cases. Call: admitted with the expected sign = correct,
  opposite = wrong.
`typed_pooled` (CLBench typed edge, v3): VARX response step responses and VARX policy equations with the v3 rule
(BY over its 18 relations). The registered plan was to take response relations from `seqtrial` if it passed the
simulator gate (§9). It did not, so `typed_pooled` equals `varx` under the v3 rule.

## 7. Outputs and metrics

- Method × relation table: calibrated (v3 rule) and uncalibrated (single-arm CI) calls, per scale.
- Uncalibrated null discovery per method: relations with single-arm CI excluding 0 (≥ 5 cases) on each of
  the 8 cross draws (mean, min–max) and on nc_random, also counted over distinct estimates.
- Null false-admission rates of the v3 rule, per method: (a) cross leave-one-out: each cross draw as input
  calibrated against the other 7 (mean admissions per draw, admissions / tests, draws with any);
  (b) random-time: nc_random as input calibrated against the 8 cross draws. (b) is a different null: the
  cross draws keep the schedule of care, nc_random destroys it, so schedule-sensitive relations can be
  "admitted" in (b) without the rule being miscalibrated for the cross null (v2 verifier: P05 at 30 s).
- **Leaderboard** (one line per method): score = calibrated correct-sign admissions − wrong-sign
  admissions on real data, with the null false-admission rates (a) and (b) and the uncalibrated counts.
- Also kept from v2: EB patient-level edges and the patient-specificity split-half test (reproducible now).

## 8. Pre-specified predictions (checked and reported whatever the outcome)

P2–P4 involve estimators whose v2 single-draw results are known, so they are replications under the new rule,
not independent confirmations. P1 and P5 concern the new parts.
- **P1 (rule on the cross null).** For every primary method, cross leave-one-out false admissions average ≤ 1
  of 18 per held-out draw, at both scales.
- **P2 (uncalibrated methods on the cross null).** At 30 s, naive, xcorr_pos, granger and pcmciplus each have
  mean uncalibrated null discoveries ≥ 4 of 18 per cross draw.
- **P3 (policy edges).** At 30 s, `typed_pooled` admits ≥ 2 of 6 policy relations with the expected sign and 0
  with the wrong sign.
- **P4 (titration paradox under the v3 rule).** At 30 s, `typed_pooled` admits at least one remifentanil-decrease
  relation (R07 or R08) with the wrong sign.
- **P5 (pooled contrast vs VARX, response relations).** At each scale `seqtrial` admits fewer wrong-sign
  response relations than `varx`. No prediction about the number of correct-sign admissions.

## 9. Simulator validation (development, before registration)

Development evidence for the pooled sequential-trial contrast, obtained before this registration
(`sim_validation/`: `GATE.md` written before the validation outputs existed, `GATE_OUTCOME.md`, `summary.md`,
`sim_validation.csv`). 800 simulated patients per cohort; recorded / reversed clinician x inert / active
drug (+ an unrecorded-confounder stress test), 2 seeds, 2 s and 30 s, real arm + nc_random + 4 cross draws,
exact ITT truth by twin simulation.

- Design change made during development, before the validation run: the recent 10 s level of each vital was
  added to the matching state (without it, remifentanil -> MAP had a large positive bias at 2 s in a
  500-patient development cohort).
- **Gate result: FAILED** (G1: inert cells with a CI excluding 0 = 6/24 at 2 s and 19/24 at 30 s, limit 10%; G2: 2
  wrong-signed active cells at 30 s). Therefore, as stated in GATE.md before the results: `seqtrial` is scored
  as a method but is not the recommended response estimator, and **`typed_pooled` keeps the VARX base for
  response relations** (it is then identical to `varx` under the v3 rule; it is kept as a named row for
  continuity with v2).
- Descriptive: under the v3 rule, seqtrial at 2 s had 24/24 correct active admissions, 0 wrong and 3/24 spurious inert
  admissions (the fewest among event designs); at 30 s it was worse than the within-patient pre-trend design.
  With an unrecorded confounder every event design is biased.

## 10. Deviations

Any change after registration is listed in `deviations.md` with its reason and which outputs it affects;
the registered code stays in `registered_code/`.
