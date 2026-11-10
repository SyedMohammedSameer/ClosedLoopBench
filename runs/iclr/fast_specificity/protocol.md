# Track D: is patient-specific structure real at the fast (beat) timescale? Pre-registered protocol

Written 2026-09-24 22:25 local time, before any split-half, ICC, early-late or MSE statistic was computed on
the fast edges. The runner (`scripts/52_fast_edge_specificity.py`) refuses to run the analysis stage
unless this file's sha256 equals the value in `protocol.sha256`, and records the sha256 in every
output (`meta.json`, table headers). Any later edit to this file changes the hash; deviations are
listed in the report with their reason.

**Seen before registration**: the wave-2 population-level tables (`runs/iclr/fast_coupling/tables.md`:
patient-median summaries of each edge, sign-correctness rates) and data availability counts (usable
epochs per patient, finite epochs per edge, covariate coverage). The wave-3 slow-edge results
(`runs/iclr/closed_loop_v2/*/patient_specificity.csv`: early-late r ~ 0) motivate the test. No
per-patient split-half quantity of any fast edge has been computed.

## 1. Question and hypotheses

Wave 3 found that patient-specific slow drug-response edges carry no out-of-sample information
beyond the population edge (0/11 relations; early-late r ~ 0). Track D asks whether patient-specific
structure exists at the beat timescale, where reflex and mechanical couplings act within seconds and
the clinician's control loop contributes little.

- H1: for fast coupling edges, a patient's own early estimate predicts that patient's late estimate
  better than the population mean (wave-3 decision rule, epochs as units).
- H2: this survives adjustment for the concurrent mean HR and mean SBP (so it is not only a stable
  operating point).
- Secondary H3: it survives adjustment for a coupling-destroyed "signal-property twin" of the same
  statistic computed on the same epoch (so it is not only stable variance or spectral properties).

## 2. Data

- Beat tables from wave 2 (`runs/iclr/fast_coupling/cache/{mimic,vitaldb}/*_beats.parquet`, built
  by `clbench.beats` in `scripts/45_fast_coupling.py`): R-peak times, RRI, beat SBP/DBP/MAP/PP,
  pulse arrival time (PAT, R-peak to PPG foot), respiratory phase per beat, beat validity.
- Epoch grid: the exact wave-2 5-min grid and `usable` flag from
  `runs/iclr/fast_coupling/epochs.parquet` (usable: >= 100 beats, >= 80 % valid, >= 150 valid beats).
- **MIMIC-IV-WDB (primary cohort)**: 50 ICU patients with usable epochs, up to 4 x 60-min blocks
  spread over the stay (median 36 usable epochs). **VitalDB (secondary)**: 24 local surgical cases
  (single anaesthetic, median 43 usable epochs). PAT in VitalDB exists in few cases (truncated
  downloads lack PLETH), and respiration in 14 of 24.
- No new waveform extraction. The Colab native-rate VitalDB tables are not used.

## 3. Edges (per 5-min epoch)

Primary (one statistic per edge; functions from `clbench.edges.fast_coupling` where they exist, so
epoch values must equal the wave-2 `epochs.parquet` values, which is checked and reported):

| id | edge | statistic | unit |
|---|---|---|---|
| E1 | pulse arrival time vs SBP | `pat_sbp_slope`: slope of 31-beat-detrended PAT on 31-beat-detrended SBP (`pat_coupling`) | ms/mmHg |
| E2 | respiratory pulse-pressure variation | `ppv`: median over breaths of 100 (PPmax - PPmin)/mean (`respiratory_pressure_variation`, breaths from respiratory-phase wraps; requires finite phase for > 80 % of beats) | % |
| E3 | respiratory sinus arrhythmia | `rsa_h`: 15-beat-detrended RRI regressed by OLS on [1, cos phi, sin phi] over valid beats with finite phase (>= 60 beats, phase finite > 80 %); amplitude = 2 sqrt(a^2 + b^2) (peak-to-trough of the fitted sinusoid) | ms |
| E4 | beat-level baroreflex sequence slope | `seq_pooled_brs`: mean slope of concordant sequences pooled over lags 0, 1, 2 (>= 3 beats, SBP steps >= 1 mmHg, RRI steps >= 5 ms, r >= 0.85; `sequence_method`) | ms/mmHg |
| E5 | heart-period (RRI) variability | `rmssd`: root mean square of successive RRI differences over pairs of consecutive valid beats | ms |

E5 is a signal property, not a coupling. It is the reference for how patient-specific a plain
stable signal property is.

Secondary statistics (reported, not part of the decision): `spv` (mmHg), `rsa_amp` (wave-2
phase-binned peak-to-trough, ms), `reg_brs` (ARX SBP(n..n-2) -> RRI(n), ms/mmHg), `sdnn` (ms),
`pat_sbp_corr` (r).

Covariates per epoch: `hr_mean` = 60000 / mean valid RRI, `sbp_mean` = mean valid beat SBP (the
wave-2 definitions). VitalDB only: `remifentanil_ce`, `propofol_ce` (NaN -> 0) and an indicator
that propofol Ce is recorded.

Signal-property twins (secondary; the same statistic on the same epoch with the coupling destroyed
but the marginal signal properties kept):
- T1 (E1): SBP and its validity circularly shifted against PAT by u ~ U{30, ..., n - 30} beats;
  K = 5 shifts; twin = mean of the K slopes.
- T2 (E2): PPV recomputed on the same breaths after subtracting from PP its respiratory-phase-locked
  component (OLS of PP on [1, cos phi, sin phi, cos 2phi, sin 2phi] over valid beats; intercept kept).
- T3 (E3): harmonic amplitude recomputed with an independent random phase offset delta_b ~ U(0, 2 pi)
  added to every breath (breaths delimited by phase wraps); K = 5; mean.
- T4 (E4): SBP and its validity circularly shifted against RRI by u ~ U{20, ..., n - 20} beats;
  K = 5; twin = nan-mean of the K pooled sequence slopes.
- E5 has no twin.

RNG: `numpy.random.default_rng([20260924, dataset_index, patient_index, epoch_index])`.

Winsorisation: each epoch-level quantity (edge, twin, covariate) is winsorised at the 1st and 99th
percentiles of its values pooled over all usable epochs of that dataset, before splitting.

## 4. Splits

A patient is eligible for an edge and split if each half has >= 3 finite epoch values.

- **S1 early/late (primary)**: the patient's finite epochs in time order; early = first floor(n/2),
  late = the rest. In MIMIC this usually spans different days of the ICU stay.
- **S2 interleaved with gap**: each epoch's index j on the contiguous 5-min grid of its block
  (unusable epochs included in the count). A = {j = 0 mod 4}, B = {j = 2 mod 4}; odd j are
  discarded, so every A epoch is >= 5 min from every B epoch. A takes the role of "early".
- **S3 within-patient epoch-order shuffle**: epochs randomly permuted within each patient, then
  split as in S1; 200 draws; the median over draws is reported. This is the stationary reliability
  ceiling: S1 below S3 means the patient-specific component drifts over time.
- **S4 data-limited**: the first 3 finite epochs of each S1 half (15 min each), to approach the
  per-patient information of the wave-3 slow edges (>= 3 events per half).

## 5. Estimators and metrics (per edge x dataset x split x adjustment)

Units are epochs (the wave-3 units were events). With per-half counts, sums and sums of squares of
the epoch values, the wave-3 function `clbench.edges.typed_temporal_pooled.patient_specificity`
is called unchanged (2,000 patient-bootstrap replicates, seed 0; every quantity including the
pooled mean and EB prior refitted per replicate):
- predictions from early halves: per-patient early mean, pooled epoch-weighted early mean, EB shrunk
  mu + B_i (m_i - mu);
- MSE against the late-half epoch values, averaged within patients, then across patients;
- `d_shrunk_minus_pooled`, `d_pp_minus_pooled`, `rel_shrunk_vs_pooled`, `r_early_late`, `mean_B`.

Also computed (own code, 2,000 patient-bootstrap replicates, seed 0):
- Pearson r and Spearman rho of patient early vs late half means;
- ICC(1,1) (one-way random, absolute agreement, k = 2 halves) of the half means;
- half-mean MSE ratio: mean_i (early_i - late_i)^2 / mean_i (mu - late_i)^2, mu = mean of early
  half means (< 1: own early estimate predicts own late estimate better than the population mean);
- **cross-patient pairing null**: late halves permuted across patients (1,000 permutations, seed 1);
  one-sided permutation p-values for r (large) and for d_shrunk_minus_pooled (small).

Adjustment levels (epoch-level residuals; OLS fitted once on the early-half epochs of all eligible
patients of that edge and dataset, applied to all epochs; not refitted per bootstrap replicate,
since it has 3 to 5 parameters fitted on hundreds of epochs):
- `raw`;
- `adj_hs`: residual after mean HR and mean SBP (**primary adjusted**);
- `adj_hst`: residual after mean HR, mean SBP and the twin (E1 to E4 only);
- `adj_hsd` (VitalDB only): residual after mean HR, mean SBP, remifentanil Ce, propofol Ce and the
  propofol-recorded indicator.
The ICC of each twin itself (raw) is also reported.

## 6. Decision rule (per edge and dataset)

Patient-specific fast structure is declared **real** for edge e in dataset d iff, on S1:
- (P1) the 95 % patient-bootstrap CI of MSE(shrunk) - MSE(pooled) lies entirely below 0 on `raw`
  values (the wave-3 rule, unchanged), and
- (P2) the same holds on `adj_hs` residuals.

Reported with it: whether it also holds on `adj_hst` (H3, "beyond signal properties") and on S2;
the cross-patient pairing p for r on `adj_hs` (if P2 holds but this p >= 0.05, the result is flagged
as discordant); S3 vs S1 (drift); S4 (data-limited). MIMIC is the primary cohort; VitalDB is a
replication with fewer patients. All 5 primary edges x 2 datasets are reported; no multiplicity
correction is applied to the decision (as in wave 3), but Holm-adjusted permutation p-values over
the 10 primary (edge, dataset) cells are reported.

Interpretation fixed in advance:
- passes P1 and P2 while wave-3 slow edges show r ~ 0: patient-specific structure exists at the fast
  timescale for that edge;
- passes P1 but fails P2: its patient-specificity is carried by the HR/SBP operating point;
- fails `adj_hst`: carried by signal properties captured by the twin;
- S4 failing where S1 passes: the contrast with the slow edges is partly information per patient,
  not timescale.

## 7. Contrast with wave 3 (descriptive)

From `runs/iclr/closed_loop_v2/{30s_all,2s_fine}/patient_specificity.csv`, arm `real`, outcomes
`obs` (wave-3 primary) and `kink`: n patients, events per patient, r(early, late) [CI],
rel_shrunk_vs_pooled [CI], and whether the wave-3 rule held. They are placed in one table and one
figure with the fast edges. Different cohorts and units, so no formal test between them.

## 8. Validation, runtime, deviations

- Integrity: recomputed epoch values of `pat_sbp_slope`, `ppv`, `spv`, `rsa_amp`, `seq_pooled_brs`,
  `sdnn`, `reg_brs`, `hr_mean`, `sbp_mean` are compared with wave-2 `epochs.parquet` (max abs diff).
- Synthetic check of the analysis code before the real analysis: a random-effects generator with
  known between-patient variance (ICC 0 and ICC > 0), checking that the cross-patient pairing null is
  calibrated at ICC 0 and that ICC(1,1) is recovered.
- Commands (`PYTHONPATH=src:.`):
  ```
  python scripts/52_fast_edge_specificity.py --stage extract --workers 4
  python scripts/52_fast_edge_specificity.py --stage synthetic
  python scripts/52_fast_edge_specificity.py --stage analyse
  python scripts/52_fast_edge_specificity.py --stage report
  ```
- Budget: under 60 min of compute in total, 4 worker processes (shared machine).
- Nothing is tuned after the confirmatory outputs are seen. Any deviation is listed in the report.
