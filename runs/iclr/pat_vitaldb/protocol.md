# Addendum A1 to track-B protocol af14df06: PAT analyses at VitalDB scale with delay-aware PAT. Pre-registered

Written 2026-09-25 ~15:20 local time, before any PAT outcome (PAT-SBP slope or correlation, epoch or case PAT level,
PAT vs age / MAP / drugs, any null or split-half quantity) was computed on the script-56 PAT. Runner:
`scripts/57_pat_vitaldb.py`. Its analysis stages refuse to run unless this file's sha256 equals
`protocol.sha256`, and every output records the hash. Parent protocol: `runs/iclr/fast_vitaldb/protocol.md`
(sha256 af14df0605c8...; "af14df06"). Everything not stated here is af14df06 unchanged. Deviations found while
running are listed in the report with their reason.

**Seen before registration** (availability and plumbing only, no PAT outcome):
- the ledger (K3, K4), af14df06 and its results (PAT-SBP on censored native PAT 56.5 %, "not a test"; CV7
  declared not assessable), the eab2d583 fast registry (F4 abstained, "population sign not assessable");
- `scripts/56_pat_native.py` (method: per-case modal foot - R delay D, foot nearest R + D within
  +-min(0.15 s, 0.4 x median RR)); column layout of `native/pat/{pid}.parquet` and row alignment with
  `native/beats/{pid}.parquet` for case vdb0001 (t_r identical); one summary file (vdb0001);
- the QC fields of all 2,954 `native/pat_summaries/*.json` (clock_warning, delay_mode_s, coverage_valid), used
  only to count the gates of section 2: 24 cases with clock_warning; of the other 2,930, 13 with
  delay_mode_s outside [0.45, 0.90] s; of the remaining 2,917, 24 with coverage_valid < 0.80; **2,893 pass**.
  Median coverage 0.990 (brief). No `median_pat_ms` was related to age or to any other variable.
- the exploratory delay-aware analysis on 9 local files (slope negative in 8/9; af14df06 report).

## 0. Deviation from af14df06 (the reason for this addendum)

af14df06 used the native `pat` column of the beats tables, censored by the 0.08-0.60 s foot window (median
coverage 3.1 %). Here, for every beat, `pat` := `pat_ms` from `data/processed_vitaldb/native/pat/{pid}.parquet`
(script 56, rows aligned 1:1 with the beats table; alignment is checked per case: identical row count and
identical `t_r`, otherwise the case is excluded and counted). **PAT includes a per-case PLETH device delay**
(modal foot - R delay ~0.655 s), so absolute PAT and between-case PAT differences are not physiologically
interpretable; within-case, beat-to-beat and epoch-to-epoch PAT variation is (the device delay is assumed
constant within a case).

Unchanged from af14df06: the beats tables (`sbp`, `map_beat`, `valid`), the epoch grid and the `usable` flag
(taken from track B's `runs/iclr/fast_vitaldb/epochs.parquet`: same `t0`, `t1`, `epoch_index`; `case_index` from
`cases.parquet`), the statistic `clbench.edges.fast_coupling.pat_coupling` (31-beat running-mean detrend over
valid beats, >= 30 valid beats with finite PAT and SBP; slope of detrended PAT on detrended SBP, ms/mmHg; Pearson
r), `pat_median` from the same call, and the wave-4 `epoch_stats` (script 52, unchanged) with RNG
`default_rng([20260925, case_index, epoch_index])`, which gives `pat_sbp_slope`, `pat_sbp_corr`, `n_pat` and the
registered T1 twin `twin_pat` (SBP circularly shifted against PAT by u ~ U{30..n-30} beats, mean of 5).
Integrity: the non-PAT columns of `epoch_stats` that do not depend on the PAT RNG draws (`hr_mean`, `sbp_mean`,
`sdnn`, `rmssd`, `seq_pooled_brs`, `twin_seq`, `reg_brs`, `rsa_h`, `ppv`, `spv`) must equal track B's values on
every usable epoch; mismatches are reported.

## 1. Cases: PAT QC rule (fixed in advance)

A case enters the PAT analyses iff it is one of track B's 2,954 QC cases (`g4_20min`), its script-56 summary
exists, and
- no `clock_warning` (PLETH not pinned to casestart: misaligned),
- `delay_mode_s` in [0.45, 0.90] s (outside: implausible, e.g. feet matched to the next or previous beat),
- `coverage_valid >= 0.80` (share of valid beats with a PAT).
Counts at each gate are reported. **Sensitivity (QC relaxed)**: R1, all 2,954 track-B QC cases with a PAT file
(no PAT QC); R2, only the clock_warning exclusion.

## 2. Population results (af14df06 section 3 estimators, unchanged)

Per case: median over its usable epochs (finite values) of `pat_sbp_slope` and of `pat_sbp_corr`. Reported:
share of cases whose median is negative (expected), case bootstrap; share of epochs negative, case-cluster
bootstrap; median and IQR of case medians with a case-bootstrap CI of the median; coverage (epochs with a
finite slope, i.e. >= 30 valid PAT beats; cases with >= 1). All CIs: 2,000 replicates, seed 0, script 54's
`boot_cases` / `boot_cluster` imported unchanged. `pat_median` (ms) is reported as a distribution with the
device-delay caveat, descriptive only.

**Reading (registered here; af14df06 registered no bar for PAT because it declared censored PAT not a test).**
The af14df06 section-3 readings registered for the baroreflex ARX are applied unchanged to the PAT-SBP case
share of the expected (negative) sign: "above chance" iff its 95 % CI lies above 0.5; "reliable enough to be a
positive control" iff the CI lower bound is >= 0.90. Primary = the PAT-QC set; R1 / R2 are sensitivity only.

## 3. Within-case nulls: is the sign share a mechanical artefact of the estimator?

For every usable epoch of a PAT-QC case with a finite observed slope, K = 5 draws of each null, RNG
`default_rng([20260925, case_index, epoch_index, 57, null_id])`, `pat_coupling` applied unchanged:
- **N-a, within-epoch circular shift >= 60 s**: s ~ U(60, 240) s; u = number of the epoch's beats with
  t_r - t_r[first] < s; SBP and its validity mask rolled by u beats against PAT
  (`pat_coupling(pat, roll(sbp, u), valid & roll(valid, u))`). On the 300-s epoch circle the shift is >= 60 s
  both ways.
- **N-b, across-epoch pairing within case**: PAT of epoch e paired with SBP of another usable epoch e' of the
  same case (uniform over the case's other usable epochs; >= 300 s apart by construction), aligned by beat index
  from each epoch's first beat and truncated to the shorter epoch; validity = valid_e & valid_e'. Cases with one
  usable epoch have no N-b draw.
Per draw: per-case median of the null slopes, case share negative, epoch share negative. Reported: the mean over
the 5 draws of the null case share and epoch share, and the paired case-bootstrap CI (2,000, seed 0) of
(observed case share negative - null case share negative), the null share averaged over draws within each
replicate. The T1 twin (`twin_pat`) sign share is reported as a third, descriptive null.
**Reading**: the population sign is "not a mechanical artefact of the estimator" iff the difference CI lies
above 0 under both N-a and N-b. Expectation stated in advance: null shares near 0.5. Under controlled
ventilation a circular shift can keep PAT and SBP co-modulated by the (periodic) ventilator (eab2d583 N1
limitation), so an N-a share far from 0.5 would be read as respiratory co-modulation, with N-b as the check that
is not locked to the ventilator phase; both must hold for the reading above.

## 4. Construct validity

**CV7 (af14df06, unchanged)**: Spearman rho, age (manifest) vs case-median `pat_median`, adults (age >= 18),
case bootstrap; expected < 0; verdict rule of af14df06 section 4 (holds / fails (reversed) / not shown).
**Pre-stated caveat**: between-case PAT includes the per-case PLETH device delay; `delay_mode_s` is estimated
from the same foot - R lags, so it cannot be subtracted independently. A between-case PAT difference, and
therefore CV7, is physiologically interpretable only if the device delay is identical across cases, which these
data cannot verify. The CV7 verdict is reported with that label and is **not counted** in the construct-validity
tally (ledger K3d) in either direction.

**CV7w (within-case alternative, declared here)**: slope of epoch `pat_median` (ms) on epoch `map_mean` (mmHg),
both demeaned within case (case fixed effects), pooled OLS; cases with SD of `map_mean` >= 3 mmHg over their
usable epochs with finite `pat_median` (script 54 `within_slope`, `log_y=False`, case bootstrap 2,000 seed 0).
Expected < 0 (higher arterial pressure, shorter pulse arrival; Mukkamala et al. 2015, IEEE Trans Biomed Eng
62:1879). The device delay cancels within case. Caveats stated now: (i) PAT is assigned only within
+-min(0.15 s, 0.4 RR) of R + D, which truncates large within-case PAT changes and biases the slope toward 0;
(ii) CV7w is the slow (epoch) face of the same PAT-pressure mechanism as the beat-level slope, so it is a
consistency check, not independent construct validity. Verdict rule as CV7.
**Descriptive only (no direction registered, no verdict)**: the same within-case estimator for `propofol_ce`
(cases with SD >= 0.1 ug/ml) and `bis` (SD >= 3).

## 5. Patient specificity of the PAT-SBP slope (af14df06 section 6 = wave-4 6d88096d sections 3-6, unchanged)

Input frame: track B's `prepare_vdb` frame (usable epochs of the 2,954 QC cases, 1/99 % winsorisation over
usable epochs) with `pat_sbp_slope`, `pat_sbp_corr`, `twin_pat`, `n_pat` replaced by the script-56 values for
PAT-QC cases and set to NaN for the other cases, before winsorisation (so all non-PAT columns are identical to
track B's). Edge E1 `pat_sbp_slope`: adjustments `raw`, `adj_hs` (hr_mean, sbp_mean), `adj_hst` (+ `twin_pat`),
`adj_hsd` (+ remifentanil Ce, propofol Ce, propofol-recorded); splits S1 (primary), S2, S4, S3 (200 within-patient
shuffles); >= 3 finite epochs per half; script 52 `build_groups` / `split_stats` (2,000 patient bootstraps seed 0;
cross-patient pairing null, 1,000 permutations seed 1); script 54's vectorised grouping checked for equality on
S1. Secondary `pat_sbp_corr` (raw, adj_hs, adj_hsd, S1); twin `twin_pat` (raw, S1); operating points `hr_mean`,
`sbp_mean` (raw, S1) recomputed on the same frame.
Decision (unchanged): REAL iff on S1 the CI of MSE(shrunk) - MSE(pooled) lies below 0 on `raw` (P1) and on
`adj_hs` (P2). Holm over the 5 primary edges on the `adj_hs` S1 permutation p, with E2-E5 taken unchanged from
track B's `specificity_all.csv` and E1 replaced. Twin-adjusted (H3) reported. Operating-point comparison
(unchanged): paired case bootstrap of r(E1, adj_hs) - r(hr_mean, raw) and - r(sbp_mean, raw) on common patients;
CI below 0 "less persistent", covering 0 "as persistent", above 0 "more persistent".

## 6. Fast-edge admission reading for F4 (input to the next stage; the decision is taken there)

In the eab2d583 registry (`scripts/55_graphs_fast.py::registry`), the other sign-bearing fast relation, F3
(baroreflex), was abstained because the af14df06 positive-control bar (case-share CI lower bound >= 0.90) was
not met; F1 / F2 (magnitude couplings) were admitted on construct validity plus twin-surviving patient
specificity; F4 was abstained because its population sign was not assessable. Fixed here: **F4's population
sign meets the admission rule iff** (a) on the PAT-QC set the case-share CI of negative case-median slopes has
lower bound >= 0.90 (the F3 rule), **and** (b) the section-3 artefact reading holds (N-a and N-b). CV7 / CV7w and
patient specificity are reported as the basis items, as for F1-F3, but do not change (a)-(b).

## 7. Runtime, integrity

`PYTHONPATH=src:. python scripts/57_pat_vitaldb.py --stage extract --workers 8`, then `--stage specificity`,
then `--stage report`. Budget <= 40 min per stage, <= 8 processes. No PAT outcome is inspected before the
extract stage has run under this hash. Nothing is tuned after outputs are seen.
