# Pre-registration: closed-loop simulator v2 (Track C)

Registered 2026-09-24 17:15 local time, before the confirmatory run. The sha256 of this file is written to
`v2/manifest.json` and to the header of `v2/results.md` by `scripts/46_sim_closed_loop.py`.

## Provenance and what was seen before registration

- Code at registration (sha256): `src/clbench/sim/clinician_loop.py` d88e58b227615a8d0a383b4a98f45398c41f5006dea24e2f6c75c0a072a6d759;
  `scripts/46_sim_closed_loop.py` 3f9b2e005f2dce8ac68219abf804c589d0cf7807bbd4e4bf037213f5372631e8. Git HEAD f64f4b9 (changes uncommitted).
- Before writing this file I saw: the verifier outputs in `runs/iclr/sim_closed_loop/verify/` and `verify/v2/`
  (own seeds 9100–9950), one smoke run (inert/active, recorded clinician, 200 patients, seed 777), and a dev run of
  every experiment below with small n and seeds offset by +9,000,000 (`v2/dev/`, not reported as results).
  The hypotheses below were written after those runs; the confirmatory run uses fresh seeds and larger n.
- Any deviation from this protocol is reported in `v2/results.md` / the theory note as a deviation.

## Common settings

Simulator defaults (`SimConfig()`), 1 s step, 7,200 s per patient after a 600 s burn-in, patient gains
log-normal jitter σ = 0.2. Remifentanil step 0.5 ng/mL, refractory 120 s, display = 10 s trailing mean,
alarm check after a 10 s reaction delay. CIs: 95% percentile patient-cluster bootstrap, 1,000 draws
(`cluster_bootstrap_mean` for event rows, `bootstrap_median` for per-patient fits). "CI excludes 0" is the
only significance criterion; no multiplicity correction (all cells are reported).

Clinicians (controller keyword arguments):

| name | gain | anticipation_s | w_z (field) | w_y (vitals) |
|---|---|---|---|---|
| recorded | 1 | 0 | 0 | +1 |
| recorded_reversed | 1 | 0 | 0 | −1 |
| recorded+field | 1 | 0 | 1 | +1 |
| recorded+field+plan60 | 1 | 60 | 1 | +1 |

Drug: `active` (defaults: D_HR = −4, D_MAP = −5 per ng/mL) or `inert` (D_HR = D_MAP = 0; every u→y estimate is bias).

Seeds: SEED_V2 = 270000. lemma: SEED_V2 + 1000 + 10·(clinician index in the table order) + (0 inert, 1 active).
headline2: SEED_V2 + 2000 + 10·(index in [recorded, recorded+field, recorded+field+plan60]) + (0 active, 1 inert).
beta_grid: SEED_V2 + 3000 + 10·(β index) + (τ_d index). monitor_delay: SEED_V2 + 4000 + (β index).

## Experiment `lemma` (Lemma FI), n = 1,000 patients per clinician × drug

Design (`decision_contrasts`): 1 s recorded series; event = change of remifentanil in the event direction
(remi_up_hr, remi_down_hr), recorded at i = t + 1. Contrast = mean HR_obs over [i+h−1, i+h] minus mean over
[i−60, i−1]; h ∈ {10, 30, 60, 110} s. Eligible times: no change in the previous 120 s, dose not at the bound
in the event direction. Controls: eligible, no change at i, not an alarm-triggered check, every 3rd second.
Pools: `eligible` (pre-t information only) and `future_inaction` (additionally no change in (i, i + 111]).
Each event: mean contrast of the 5 nearest same-patient controls in the recorded decision statistic
(0.5·(display HR/8 + display MAP/10)). Paths: `routine` (not an alarm-triggered check; overlap region) and `alarm`.
Truth: inert = 0 exactly; active = ITT effect from paired twin runs (`itt_truth`, k_max = 3 decisions per
patient, same seed, single decision vetoed), paired to events by (patient, i); also the sustained-step estimand.

Hypotheses (routine path unless stated):

- **L1** inert, recorded, remi_up: eligible-pool CI covers 0 at all four horizons.
- **L2** inert, recorded, remi_up: future-inaction estimate > 0 with CI excluding 0 at h ∈ {30, 60, 110}.
- **L3** (sign set by the policy) paired difference future-inaction − eligible, coefficient space (× event
  direction), inert: CI > 0 for recorded remi_up at h ∈ {30, 60, 110}; CI < 0 for recorded_reversed remi_up at
  h ∈ {30, 60, 110}. Secondary: recorded remi_down (coefficient space) > 0; recorded_reversed remi_down < 0.
- **L4** active, recorded, remi_up: eligible-pool bias vs ITT truth has a CI covering 0 at all four horizons;
  future-inaction bias vs ITT truth has a CI > 0 at ≥ 2 of h ∈ {30, 60, 110}.
- **L5** inert, recorded+field and recorded+field+plan60, remi_up: eligible-pool estimate CI > 0 at ≥ 2 of 4
  horizons (residual confounding by unrecorded information) and the paired difference (FI − eligible) CI > 0
  at ≥ 2 of 4 horizons.
- **L6** (secondary) alarm path, inert, recorded, remi_up: eligible-pool CI excludes 0 at all horizons (no overlap).

## Experiment `headline2`, n = 200 patients per clinician × drug

`event_effects` at 30 s (naive, matched), 2 s (naive, pretrend, kink, kink_rich, naive_exog) and 1 s (kink,
kink_rich); horizons 30 s scale {30, 60, 120}, 2 s and 1 s {10, 30, 60, 120}; specs remi_up_hr, remi_up_map,
vent_up_co2, vent_down_co2 (inert drug: remifentanil specs only); pools for matched/kink/kink_rich:
`eligible` (ITT events), `future_inaction` (ITT events), `v1_future_inaction_clean` (v1 behaviour). Truth:
sustained-step estimand of each estimator's functional (naive/matched/naive_exog: 2-sample post mean minus
pre-mean; pretrend/kink/kink_rich: 3-sample post mean minus extrapolated pre-trend). Plus the 30 s direct-method
ARX (2 lags of HR, dose, MAP; step response × step size), per-patient, bootstrap median.

- **H1** inert, recorded, 2 s kink and 1 s kink, eligible pool: CI covers 0 at every h.
- **H2** inert, remi_up_hr, 2 s kink and 1 s kink: paired difference FI − eligible CI > 0 at h ∈ {10, 30, 60}
  for all three clinicians.
- **H3** inert, 30 s direct-method ARX, remi_up_hr: CI excludes 0 at ≥ 1 horizon for each clinician.
- Everything else in `headline2` is descriptive.

## Experiment `beta_grid`, n = 60 patients per cell

Headline controller (gain 1, anticipation 60 s, w_z = 1), β ∈ {−0.4, −0.2, −0.1, −0.05} × τ_d ∈ {20, 60, 200, 600,
2000} s (τ_s = τ_d/10) × scale {beat (ARX 6 lags), 2 s (ARX 4 lags)} × W ∈ {15, 31, 61, 121, 241} s and raw.
Estimate: median over patients of the static gain; truth: median planted β; bias = estimate − truth.

- **B1** residualised at W = 61 s (beat) / 62 s (2 s): for each (τ_d, scale), max − min of the bias across the
  four β values ≤ 0.06 bpm/mmHg.
- **B2** sign rule: predicted sign of the estimate = sign(β + B̂), B̂ = bias at β = −0.4 in the same (τ_d, scale, W).
  Over the 150 non-reference residualised cells, excluding cells with |β + B̂| < 0.02, the point estimate has the
  predicted sign in ≥ 90% of cells.

## Experiment `monitor_delay`, n = 60 patients per β

Headline controller, τ_d = 200 s, β ∈ {−0.4, −0.2}; 2 s numerics with HR trailing average 2 or 8 s, MAP 2 s,
HR display delay d ∈ {0, 2, 4, 6, 8, 12} s; ARX(4) raw and residualised (31 samples = 62 s).

- **M1** HR averaging 2 s, residualised: CI < 0 at d = 0 and CI > 0 at d = 12, for both β.
- **M2** residualised: Spearman correlation between d and the median estimate ≥ 0.9 in each of the four
  (β, HR averaging) sequences.

## Experiment `theory`

Deterministic numerics: Cor. 1(b) crossover horizon h*(T) for P = 120 s, display 10 s; Prop. 4 leakage
exponent for m = 1..4 (prediction min(2m − 1, 4)).
