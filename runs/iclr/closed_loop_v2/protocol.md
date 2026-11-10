# Closed-loop benchmark v2: pre-registered protocol

Written 2026-09-24 17:17 local time, before any confirmatory v2 run on the full case sets. Only smoke tests
(40 cases at 30 s, 24 cases at 2 s, written to a scratch directory and not kept) and synthetic unit tests
were run with this code. The runner records this file's sha256 in every output (`meta.json`, table
headers). Any later change to this file changes the hash.

Code at registration time (sha256):
- `src/clbench/eval/closed_loop_bench.py` fbb899d44b9f4d516293222cc584d6c54b546bb23aea08ba517b8962a0730c6a
- `src/clbench/edges/typed_temporal_pooled.py` 1db1b66f8004346b33d350f2013eaff017f26aade152645f875cb9391bd4f78a
- `scripts/44_closed_loop_bench.py` ef6f0b7ef28d694798935b454ed0d520ee1b85d052a9589a7b08dedcdf1c51df
- `tests/unit/test_closed_loop_bench.py` c1ee9f45edb32ea2c56e7ee03a2463596a8796a003f30b4c92c8d52aae6cde08

## 1. Data and runs

- VitalDB numerics at 30 s windows with a 15 s stride, **all 3,442 cases** (`30s_all`).
- The 500 cases that also have 2 s numerics, at **2 s** (`2s_fine`).
- Action events come from `data/processed_vitaldb/events`, with the same agents as v1.
- Commands. Seed 0, 200 case-bootstrap replicates, 1,000 patient-bootstrap replicates for §6.
  ```
  scripts/44_closed_loop_bench.py --scale 30s --cases all  --protocol runs/iclr/closed_loop_v2/protocol.md
  scripts/44_closed_loop_bench.py --scale 2s  --cases fine --protocol runs/iclr/closed_loop_v2/protocol.md
  scripts/44_closed_loop_bench.py --combine-v2
  ```
- Outputs go to `runs/iclr/closed_loop_v2/`. `runs/iclr/closed_loop/` (v1) is not modified.

## 2. Ground truth

Unchanged from v1 (`RELATIONS`, citations in the module):
- 12 **response** relations (action -> vital) with a pharmacology or physiology sign.
- 6 **policy** relations (vital -> action) with a clinical-practice sign.

Primary response horizons: 60 s for ventilator rate -> EtCO2 and 120 s for every drug. Policy relations
use the 2-min pre-action trend. Estimates are in coefficient space, so a correct estimate has the sign in
`expected`.

## 3. Estimators

Each estimator is primary in v2 unless marked otherwise.

| method | definition |
|---|---|
| naive | post-window mean at t+h minus the mean over [t-120 s, t) (v1 definition) |
| naive_lvl | post-window mean at t+h minus the last pre-level [t-30 s, t) (flag 9) |
| xcorr_v1 | Wave-2 xcorr: signed value at argmax \|c\| over ±30 min, shared by both directions of a pair |
| xcorr_dir | argmax \|c\| only over lags where the relation's cause leads (≥ 1 step) (flag 5) |
| xcorr_pos | v1 rule: argmax c, the positive peak (flag 5) |
| granger | pairwise VAR(p) in levels, pooled FE (as v1); per-patient F-tests |
| varx | pooled direction-split VARX (as v1) |
| pcmciplus | PCMCI+ (ParCorr, α = 0.01) on the **same case subsample in every arm** (30 s: 1,200 cases, pressor cases first then seeded random; 2 s: all 500). Lags cover 240 s: 30 s grid with τmax 8 at 30 s; 10 s block means with τmax 24 at 2 s (flag 6) |
| varlingam | pooled FE-VAR + DirectLiNGAM (as v1, 50 bootstrap replicates) |
| matched | ITT events, v2 control pool, bias-corrected NN matching on level, 2-min trend and dose |
| kink | ITT events, v2 control pool, bias-corrected NN matching on pre-trend line (a, b) and dose; outcome = deviation from the local pre-trend (30 s: 180 s pre-fit; 2 s: 60 s) |
| typed_pooled | CLBench typed-edge estimator, §5 |

**v2 control pool (flag 1).** A control anchor g on the placebo grid must meet two conditions:
- Eligible: no change of the action in [g − P, g), where P = max(150 s, kink pre-window). Treated events meet the same condition.
- Untreated at g: no change within the decision interval [g, g + grid/2]. That is ±7.5 s at 30 s and ±5 s at 2 s.

Nothing after g + grid/2 is used; this is sequential-trial-emulation logic. v1 also excluded anchors
followed by a change up to g + max(h) (future inaction). That v1 pool is kept as `*_v1`.

**Bias correction** (added after the synthetic unit test, before this registration). The matched and
kink contrasts add (z_event − mean z_matched) · β, where β comes from the within-case OLS of the control
outcome on the z-scored matching features (Abadie & Imbens 2011). This applies only to response
relations. Policy matching (dose, phase) is unchanged.

**Pre-specified sensitivity variants.** These are reported in `sensitivity.md` and in the combined
tables. They are never used for the primary claims.
- `*_clean`: v1 post-treatment selection. Events with no same-action change in (t, t+h] (flag 2).
- `*_v1`: exact v1, i.e. the clean selection with the future-inaction pool and no bias correction.
- `*_nobc`: v2 without bias correction.
- `*_coint`: excludes events and controls with any other action's change within ±60 s (flag 3).
- `*_coadj`: adds other-action up/down counts within ±60 s to the matching features (flag 3).
- `kink_prealt`: kink with a 60 s pre-fit at 30 s and 180 s at 2 s (flag 10).

**Dose masking (flag 7).** A positive dose after the last recorded value of the action's numerics track
(`tci_*_ct`, `vent_set_rr`, `inf_*_rate_ml_h`) is NaN. Only the tail is masked; intermittent gaps are
not.

## 4. Nulls and calls

- **nc_cross** is the primary null for every method (flag 4). It places another patient's action stream (a per-action derangement, seed 0) on this patient's clock.
- **nc_random** keeps timing only: steps moved to uniform random times. It is reported alongside nc_cross.
- **nc_cross2** is a second derangement (salt 7919), run for the VARX only. It is the null input for the typed estimator's false-admission rate.
- **Calibrated estimate**: real minus nc_cross, from a paired case bootstrap (shared weights, 200 replicates). The main table reports it for every method, with the % of eligible patients (≥ 3 events, ≥ 50% vital coverage) whose own real-data estimate has the expected sign.
- **Call**: R/X = expected/wrong sign with the 95% percentile CI excluding 0; r/x = sign only; sign-only when fewer than 5 cases contribute.
- **Null discovery rate**: the number of relations with the CI excluding 0 under nc_cross and under nc_random. It is also counted over **distinct estimates** (identical est, lo, hi collapse; flag 8).

## 5. CLBench typed-edge estimator (`typed_pooled`)

1. The population edge estimate comes from the pooled direction-split VARX. For response relations it is the step response at the primary horizon to a unit up- or down-increment. For policy relations it is the up/down increment regressed on the vital's 2-min trend (plus levels and own lags).
2. The calibrated estimate is d = real − nc_cross, with a paired case-bootstrap SE and a two-sided p = 2Φ(−|d|/SE).
3. **Admission**: an edge is admitted iff the paired 95% percentile CI excludes 0 **and** its Benjamini–Yekutieli adjusted p ≤ 0.05. The BY family is all 18 relations of the run. The admitted edge carries sign(d) and is scored against `expected`: correct, wrong, or not admitted.
4. **False-admission rate**: the same pipeline, with nc_random in place of real and with nc_cross2 in place of real, both calibrated against nc_cross.
5. **Patient-level response edges.** Take the per-patient kink (and matched) estimates at the primary horizon, with their per-patient SE. Shrink them toward the pooled estimate of the same method with a normal-normal EB prior; τ² is estimated by DerSimonian–Laird. A patient-level edge is admitted if |shrunk| > 1.96 posterior SD. This is reported for real and nc_cross.

## 6. Patient-specificity test (pre-registered)

- **Eligibility**: a patient with ≥ 6 ITT (clean-pre) events of the relation's direction and ≥ 50% coverage of the vital. Events are split by time into an early half (first ⌊n/2⌋) and a late half, and each half needs ≥ 3 finite outcomes.
- **Outcomes per event**, at the relation's primary horizon, in coefficient space:
  - `obs` (**primary**): the observed post-event change, post-window mean minus the last pre-level [t−30 s, t).
  - `kinkraw`: deviation from the local pre-trend line.
  - `kink`: the bias-corrected placebo-calibrated kink contrast. It removes patient-level drift.
- **Predictions**, all fitted on early halves only:
  - per-patient-only: the patient's early mean.
  - pooled-only: the event-weighted mean of all eligible patients' early events.
  - shrunk: the EB combination μ + B_i(m_i − μ), with B_i = τ²/(τ² + σ²_w/n_i). σ²_w is the pooled within-patient variance and τ² the moment estimate of between-patient variance.
- **Metrics**, patient-averaged over the late halves:
  - MSE of each prediction against the late-half event outcomes.
  - Sign agreement between the prediction and the patient's late-half mean.
  - The early–late correlation of patient means.
- **Inference**: a patient bootstrap with 1,000 replicates. Patients are resampled and μ, σ²_w, τ² and B_i are refitted in every replicate. The paired differences are MSE(per-patient) − MSE(pooled) and MSE(shrunk) − MSE(pooled).
- **Decision rule**: patient-specific response edges "carry information beyond the population edge" for relation r iff the 95% CI of MSE(shrunk) − MSE(pooled) is entirely < 0 on real data for **both** `obs` and `kink`. The same statistics on the nc_cross arm are the negative control. An improvement that also appears under nc_cross is patient-level drift, not response heterogeneity. No multiplicity correction applies; all 12 relations are reported.

## 7. Runtime and deviations

- The runtime budget is about 60 min in total, with PCMCI+ subsampled as stated in §3.
- Any deviation from this protocol will be listed in the report as a deviation, with its reason.
- Nothing is tuned after looking at the confirmatory outputs.
