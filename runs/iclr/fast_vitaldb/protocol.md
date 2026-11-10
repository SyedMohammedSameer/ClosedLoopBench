# Track B (final wave): fast edges at VitalDB scale and monitor display latency. Pre-registered protocol

Written 2026-09-25 ~12:00 local time, before any edge statistic, lag, fit or split-half quantity was computed on
the VitalDB native tables. Runner: `scripts/54_fast_edges_vitaldb.py`. Its analysis stages refuse to run
unless this file's sha256 equals `protocol.sha256`; every output records the hash. Deviations are listed
in the report with their reason.

**Seen before registration** (availability and plumbing only, no effect estimates):
- wave-2 (`runs/iclr/fast_coupling/`) and wave-4 (`runs/iclr/fast_specificity/`) results, as in the ledger
  (E1-E3, C3, D4, D5);
- the per-case QC fields in `data/processed_vitaldb/native/summaries/*.json` (3,227 cases with beats;
  2,954 pass the valid-fraction and HR-bias gates before the 20-min rule);
- per-beat coverage on 8 random QC cases: respiratory phase finite 94-100 %, **PAT finite in only 3-12 %
  of beats**; a timing probe of the wave-4 `epoch_stats` (about 6 ms per epoch; values not inspected);
- a plumbing diagnosis on local case 0010: the SNUADC/PLETH pulse foot follows the ECG R-peak by
  ~0.71 s, beyond the 0.08-0.60 s PAT search window of `clbench.beats.build_beat_table`. The native
  PAT column is therefore **censored** (only beats whose delayed foot falls under 0.6 s survive, plus
  mismatches), and its absolute level includes a device/filter delay of the PLETH output.
- `clinical_data.csv` fields: anestart (median -583 s from casestart), intraop_phe / intraop_eph /
  intraop_epi (bolus totals, no times).

## 1. Cases (pre-registered QC)

A VitalDB case is analysed iff: beats file present; `beats_valid_frac >= 0.5`; `|hr_bias_beat_minus_solar_bpm|
<= 5` (finite); >= 20 min of valid beats (sum of RRI over `valid` beats >= 1,200 s; `valid` already
requires a matched arterial pulse, `gate_beats(require_pressure=True)`). Counts are reported at each gate.

Clock: beats, breaths and numerics_1s all use seconds from clinical casestart (script 48 pins the
VitalFile origin), so they are joined without correction. clinical_data.csv (anestart) is on the same
clock. No older product (fine_2s, numerics, events: loader-origin clock) is used.

## 2. Epochs (identical to waves 2/4)

Blocks: maximal runs of valid beats with gaps <= 60 s, kept if >= 600 s. Consecutive 5-min epochs inside
each block; **usable** iff >= 100 beats, >= 80 % valid, >= 150 valid beats. Per usable epoch, the wave-4
function `epoch_stats` (imported unchanged from `scripts/52_fast_edge_specificity.py`; RNG
`default_rng([20260925, case_index, epoch_index])`) gives `hr_mean, sbp_mean, sdnn, rmssd, seq_pooled_brs,
twin_seq, reg_brs, pat_sbp_slope, pat_sbp_corr, n_pat, twin_pat, rsa_amp, spv, ppv, n_breaths, rsa_h,
twin_rsa, twin_ppv`. Added: `pat_median` (`pat_coupling`), `map_mean`, and 1 s numerics epoch means:
`bis` (seconds with `bis_sqi >= 50` only), `propofol_ce` (`ppf20_ce`), `remifentanil_ce` (`rftn20_ce`, else
`rftn50_ce`), `set_rr`, `tv`; breath-table rate `60 / median(ttot)` and CV of `ttot`.

**Controlled ventilation** (per epoch): >= 30 breaths in the breath table with `t_exp_start` in the
epoch, CV(ttot) <= 0.15, median `set_rr` finite and > 0, and |60/median(ttot) - median set_rr| <= 2 /min.
PPV, SPV and RSA population summaries and construct validity use controlled-ventilation epochs only.

## 3. Population results (part 2), case-cluster bootstrap (2,000 replicates, seed 0) for every CI

Per case: the median over its usable epochs (finite values) of each statistic.
- **PAT-SBP** (`pat_sbp_slope`, expected < 0): share of cases whose median is negative; share of epochs
  negative (cluster CI). Because native PAT is censored (section 0), this is reported with coverage
  (epochs with >= 30 valid PAT beats) and flagged as censored; it is not a test of the coupling.
- **Baroreflex ARX** (`reg_brs`, wave-2 primary: 31-beat detrended RRI(n) on RRI(n-1, n-2) and SBP(n,
  n-1, n-2), summed SBP coefficient, expected > 0): share of cases positive; share of epochs positive.
  Reading fixed in advance: "above chance" iff the case-share CI lies above 0.5; "reliable enough to be a
  positive control" iff the CI lower bound is >= 0.90.
- Distributions (median and IQR of case medians, CI for the median): `ppv`, `spv`, `rsa_h`, `rsa_amp`,
  `rsa_h / tv` (ms per litre; RSA gain, secondary), `seq_pooled_brs`, `reg_brs`, `pat_median` (censored).

## 4. Construct validity (part 2), directions fixed from physiology

Each item **holds** iff its 95 % case-cluster CI excludes 0 in the expected direction, **fails
(reversed)** iff it excludes 0 in the other direction, and is **not shown** otherwise.

| id | relation | estimator | expected | basis |
|---|---|---|---|---|
| CV1 | age vs BRS | Spearman rho, age vs case-median `seq_pooled_brs`, adults (age >= 18) | < 0 | Gribbin et al. 1971 Circ Res 29:424; Laitinen et al. 1998 J Appl Physiol 84:576 |
| CV2 | age vs RSA | Spearman rho, age vs case-median `rsa_h` (controlled-ventilation epochs), adults | < 0 | Hrushesky et al. 1984 Science 224:1001; Pfeifer et al. 1983 Am J Med 75:249 |
| CV3 | propofol Ce vs BRS (within case) | slope of log `seq_pooled_brs` on `propofol_ce`, both demeaned within case (case fixed effects), pooled OLS; cases with SD of propofol_ce >= 0.1 ug/ml over their epochs | < 0 | Cullen et al. 1987 Anesth Analg 66:1115; Ebert et al. 1992 Anesthesiology 76:725 |
| CV4 | BIS vs BRS (within case) | slope of log `seq_pooled_brs` on epoch BIS, case fixed effects; cases with SD of BIS >= 3 | > 0 (lower BIS, lower BRS) | as CV3 (depth-dependent baroreflex depression) |
| CV5 | vasopressor cases vs PPV | difference in the median of case-median PPV (controlled ventilation) between cases with any vasopressor (intraop_phe, intraop_eph or intraop_epi > 0, or an Orchestra PHEN/NEPI/EPI/VASO/DOPA pump in the manifest) and cases with none; bootstrap resamples cases within group | > 0 | Michard et al. 2000 AJRCCM 162:134 (PPV indexes preload dependence); hypotension needing pressors is enriched for preload dependence |
| CV6 | early post-induction vs later PPV (within case) | per case: median PPV of controlled-ventilation epochs starting in [anestart, anestart + 1800 s] minus median PPV of those starting >= anestart + 3600 s; median of the case differences | > 0 | Muzi et al. 1992 Anesth Analg 74:877 (propofol venodilation); Michard 2000 |
| CV7 | age vs PAT | Spearman rho, age vs case-median `pat_median`, adults | < 0 (**shorter** with age) | Arterial stiffening raises pulse-wave velocity with age (Reference Values for Arterial Stiffness Collaboration 2010 Eur Heart J 31:2338); finger pulse transit time shortens with age (Allen & Murray 2002 J Hum Hypertens 16:711) |

Note on CV7: the track brief stated "PAT longer with higher age". The registered direction follows the
cited physiology (shorter); the pre-ejection period, which lengthens with some cardiac disease, is
included in PAT, so the direction for PAT (not PTT) is less certain. Because native PAT is censored at
0.6 s with a ~0.4-0.5 s device delay (section 0), CV7 is **declared not assessable** on the native
tables; it is computed and reported only as a censored descriptive number.

## 5. Monitor display latency (part 3)

Per QC case, on its blocks:
- Beat HR: 60000 / RRI placed at t_r + RRI/1000 (the first moment the interval is known); beat MAP:
  `map_beat` at `t_sys`. Each averaged into 1-s bins [k, k+1). Monitor: numerics_1s `hr` and `art_mbp`
  at `t_sec` (values <= 0 set to NaN; NaN between records).
- **Primary**: both series high-passed (minus a centred 61-s moving mean over finite values, kept where
  >= 20 of the 61 seconds are finite; the monitor records HR and ART_MBP about every 2 s, so a 50 %
  rule would drop about half of its samples). For lag L in -30..+30 s (positive L = monitor reports later), Pearson
  correlation of beat(t) with monitor(t + L) over all finite pairs inside blocks (>= 300 pairs). The lag
  is the argmax, refined by parabolic interpolation on the neighbouring lags. A case enters the lag
  distribution for a signal iff its peak correlation is >= 0.5. Relative lag = HR lag - MAP lag (cases
  that pass for both). Secondary: the same on raw (not high-passed) series.
- Reported: median, IQR, 5-95 % of the HR lag, MAP lag and relative lag, with case-bootstrap CIs of
  the medians.

2 s lags-only MAP -> HR fit (the September / wave-2 estimator: `lagged_timescale_fit(blocks, n_lags=5)`,
HR on its lags 1-5 and MAP lags 1-5 on 2-s bins, block intercepts, summed MAP coefficient, **expected
< 0**), on the same blocks of each case:
- (a) monitor numerics as is (2-s `bin_means` of `hr` and `art_mbp`, `min_frac=0`, as wave 2);
- (b) monitor numerics with the HR series moved earlier by the case's relative lag (HR time t - rel);
  cases without a case-specific relative lag use the median relative lag of the included cases;
  (b2, sensitivity) the population median relative lag for every case;
- (c) beat-derived 2-s series (`bin_means` of beat HR and beat MAP as above; wave-2 `2s raw`).
Same cases: those with finite coefficients in (a), (b) and (c). Reported: share with the expected
(negative) sign, case-bootstrap CI; exact McNemar tests (a vs b, a vs c, b vs c). Secondary: the 30-s
high-passed versions of (a)-(c).

Reading fixed in advance (ledger D5/E3): if (b) and (c) have a clearly higher correct-sign share than (a)
(McNemar p < 0.05), the monitor's relative HR delay is a real cause of the September sign failure at
2 s; if (a), (b), (c) are all near or below 50 %, the 2 s failure is not explained by display latency.

## 6. Patient specificity at scale (part 4): the wave-4 registered test, unchanged

Protocol `runs/iclr/fast_specificity/protocol.md` (sha256 6d88096d...) sections 3-6 applied to the
epochs of section 2 of all QC cases (VitalDB only; one dataset): primary edges E1 `pat_sbp_slope`, E2
`ppv`, E3 `rsa_h`, E4 `seq_pooled_brs`, E5 `rmssd`; secondary `spv, rsa_amp, reg_brs, sdnn,
pat_sbp_corr`; twins; winsorisation at 1/99 % over usable epochs; covariates `hr_mean, sbp_mean`
(adj_hs), + twin (adj_hst), + remifentanil Ce, propofol Ce (NaN -> 0) and a propofol-recorded indicator
(adj_hsd); splits S1 (primary), S2, S3 (200 within-patient shuffles), S4; >= 3 finite epochs per half;
`patient_specificity` (2,000 patient bootstraps, seed 0) and the cross-patient pairing null (1,000
permutations, seed 1). The same functions are imported from script 52; only the grouping step is
re-implemented vectorised for speed and checked for equality with script 52's `build_groups` on S1.

Decision (unchanged): REAL for an edge iff on S1 the CI of MSE(shrunk) - MSE(pooled) lies below 0 on
`raw` (P1) and on `adj_hs` (P2). Holm-adjusted permutation p over the 5 primary edges.

Operating-point comparison (the wave-4 verifier found mean HR as persistent, r 0.81): the same S1
statistics for `hr_mean` and `sbp_mean` (raw). Paired case-bootstrap difference r(edge, adj_hs) -
r(hr_mean, raw) on the patients eligible for both. Reading: CI below 0, "less persistent than mean HR";
covering 0, "as persistent"; above 0, "more persistent".

Secondary: E2/E3 restricted to controlled-ventilation epochs.

## 7. Runtime, integrity, deviations

- `PYTHONPATH=src:. python scripts/54_fast_edges_vitaldb.py --stage extract --workers 4`, then
  `--stage specificity --workers 4`, then `--stage report`. Budget <= 60 min compute, <= 4 processes.
- Integrity: the vectorised grouping is checked against script 52's `build_groups` (S1, all adjustments)
  on the real data; the epoch grid reproduces the wave-2 rule; counts at every QC gate are reported.
- No surrogate p-values are computed (runtime); sign shares are descriptive with case-cluster CIs.
- Nothing is tuned after outputs are seen. Deviations are listed in the report.
