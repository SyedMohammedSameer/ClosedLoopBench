# Addendum B1 to track-D protocol eab2d583: the PAT fast edge (F4) and `prolonged_pat` in the v2-fast graphs. Pre-registered

Written 2026-09-25 ~15:35 local time, before any per-patient F4 statistic (per-epoch surrogate p, CI, patient p, Holm,
pointer, instantiation, null rate) or any `prolonged_pat` window, threshold value or episode was computed.
Runner: `scripts/55_graphs_fast.py` (extended); library: `src/clbench/kg_v2.py` (extended, backward compatible;
`tests/unit/test_kg_v2.py` must pass). The build and report stages refuse to run unless **both** `protocol.md`
(eab2d583, unchanged) and this file match their registered sha256 (`protocol.sha256`, `protocol_pat.sha256`);
every output records both hashes. Everything not stated here is eab2d583 unchanged, including the clock fix of
commit 800595a (`grid_offset` = 0: native casestart clock = v2 grid clock). Deviations found while running are
listed in the report with their reason.

**Seen before registration** (no per-patient F4 or `prolonged_pat` quantity):
- the ledger (K3, K4), eab2d583 and its clock-fixed outputs (now frozen in `pre_pat/`), addendum A1
  (`runs/iclr/pat_vitaldb/protocol.md`, sha256 230b57c5...) and its reported population results
  (`summary.json`, `tables.md`): PAT-QC set 2,893 cases; case share of negative PAT-SBP slopes 94.3 % [93.4, 95.1]
  (n = 2,884); within-case nulls N-a 49.1 %, N-b 52.3 % negative, observed - null +45.2 [44.0, 46.4] and
  +42.0 [40.8, 43.2] pp; CV7w -0.652 [-0.673, -0.631] ms/mmHg; E1 patient specificity REAL, twin r 0.03;
  `f4_admission_reading` = met; exploratory PPG-amplitude adjustment -0.491 ms/mmHg, 92.4 % negative;
- column layouts of `runs/iclr/pat_vitaldb/epochs_pat.parquet` and `cases.parquet` (gate counts 2,954 / 2,930 /
  2,917 / 2,893), `native/pat/{pid}.parquet`, and the vdb0001 PAT summary. No per-epoch or per-case PAT value was
  inspected in this stage.

## 1. F4 admission decision (eab2d583 registry, reading fixed in A1 section 6)

F4 (PAT <-> beat SBP, expected sign negative) is **admitted for per-patient testing iff** A1's
`summary.json` field `f4_admission_reading["population sign meets the F4 admission rule"]` is true, i.e.
(a) the F3 rule of the eab2d583 registry (case-share CI lower bound of negative case-median slopes >= 0.90) and
(b) A1's artefact reading (difference CI above 0 under N-a and N-b). The build reads the field; if false, F4 stays
abstained and its stored reason is updated with the A1 numbers. The registry entry stores the A1 population sign,
the nulls, CV7w (consistency check), CV7 (labelled not physiologically interpretable, not counted), the E1
specificity decision and the device-delay caveat.

## 2. Cases

F4, the `pat` variable node and `prolonged_pat` are built only for track-B QC cases (`g4_20min`, 2,954) that pass
the A1 PAT-QC rule (`runs/iclr/pat_vitaldb/cases.parquet`, `q3_cover == True`: aligned PAT file, no
clock_warning, `delay_mode_s` in [0.45, 0.90] s, `coverage_valid` >= 0.80; 2,893 cases). Other QC graphs record
the gate they fail. PAT for every beat := `pat_ms` of `native/pat/{pid}.parquet` (script 56; rows aligned 1:1 with
the beats table, re-checked per case: same length and identical `t_r`, otherwise the case is treated as failing
PAT QC and counted). Absolute PAT includes a constant per-case PLETH device delay; only within-case variation is
used.

## 3. `pat` variable node and `prolonged_pat` beat concept (within case)

`prolonged_pat` **can be defined within case** and is built (replacing eab2d583's abstention for PAT-QC cases):
- Window i = [ws_i, ws_i + 30) on the v2 grid (15-s stride), beats with R-peak in the window. Observed iff >= 20
  valid beats (`valid`) have a finite PAT; otherwise -1. The same rule defines the `pat` variable node's observed
  windows (`{pid}:var:pat`, unit "ms, within-case only").
- Statistic d_i = median PAT of those beats - B, where B = the case's own baseline = median over the case's
  observed windows of the window median PAT. The device delay is constant within case and cancels.
- On iff d_i >= Delta_pat. Delta_pat = 90th percentile, over the usable epochs of the PAT-QC cases in A1's
  `epochs_pat.parquet`, of (epoch `pat_median` - the case's median of epoch `pat_median`) (finite values only).
  Cohort-relative and within-case, fixed from A1's table before any window is computed (as `low_hrv`'s T_low);
  its value is recorded. No clinical cut-off exists for a within-case PAT rise.
- Meaning: a **signal concept** ("pulse arrival slower than this patient's typical value"; e.g. lower arterial
  pressure or longer pre-ejection period), not a diagnosis. Caveat: script 56 assigns a foot only within
  +-min(0.15 s, 0.4 RR) of R + D, so rises beyond that window are truncated (missing), biasing toward "off".
- Tiers, episodes, ids as eab2d583 section 2 (fallback 60 s / 600 s tiers; node `{pid}:b:prolonged_pat:{tier}`;
  episode `b3.{r}`; beat row range, n beats, median statistic "PAT - case baseline (ms)"); code
  `local:prolonged_pat`. The three eab2d583 beat concepts are computed exactly as before (must be unchanged).

## 4. F4 per-patient test (same structure as F1 / F2, eab2d583 section 3)

**Epochs**: all of track B's usable 5-min epochs of the case (`epochs.parquet`, `usable`; any ventilation mode, as
in A1), beats with t_r in [t0, t1). **Statistic**: s = `pat_coupling(pat, sbp, valid)["pat_sbp_slope"]` (31-beat
running-mean detrend over valid beats with finite PAT and SBP, >= 30 pairs; OLS slope of detrended PAT on detrended
SBP, ms/mmHg); must equal A1's `epochs_pat.parquet` `pat_sbp_slope` on every epoch (checked). An epoch is tested iff
s is finite and the epoch has >= 60 beats.

**Per-epoch test and CI**: K = 499 surrogates: SBP with its validity mask circularly shifted against PAT by
u_k ~ U{30, ..., n - 30} beats (the registered T1-twin range), slope recomputed with the same estimator (validity
valid & roll(valid, u_k)); one-sided p_e = (1 + #{s_k <= s}) / (K + 1) (expected sign negative). 95 % CI: block-
cluster bootstrap (B = 200) of the epoch's detrended pairs, clusters = consecutive 31-beat blocks of the epoch
(the detrend window), resampled with replacement, weighted least-squares slope, percentile. Seeds:
`default_rng([20260926, case_index, epoch_index, 3, stream])` (relation index 3; stream 0 real, 1 N1, 2 N2, 5 N-b).
A vectorised surrogate estimator is used; it must reproduce `pat_coupling` on shifted input (checked on a sample).

**Patient level**: >= 3 tested epochs. T = mean of the epoch slopes; T_k = mean over epochs of the k-th surrogate;
p = (1 + #{T_k <= T}) / (K + 1). **Holm over the case's admitted fast relations at family-wise 0.05**, as
eab2d583 section 3 states: the family is now {F1, F2, F4} (those eligible in the case), so F1 / F2 instantiation
can change for PAT-QC cases relative to `pre_pat/`; every change is reported. **Pointers**: epochs whose p_e passes
BH at q = 0.05 within (case, F4). Instantiated iff Holm passes and >= 1 pointer. Each pointer carries epoch start /
end (v2 clock), first / last beats row, slope with its CI, p_e, p_bh, n PAT-SBP pairs, n blocks, and track B's
epoch `sbp_mean`, `map_mean`, `hr_mean` plus the epoch PAT-SBP correlation. The edge (source `{pid}:var:pat`,
target `{pid}:var:sbp_beat`, expected sign -1) carries the patient mean slope with an epoch-cluster bootstrap CI
(1,000; seed `[20260926, case_index, 999, 3, 3]`), p and Holm p. Note attached: association between beat PAT and
beat SBP (bidirectional, not causal); the patient estimate is descriptive; population split-half persistence of the
slope is REAL and survives the twin adjustment (A1) but is below that of mean HR, so no per-patient specificity
claim is attached; absolute PAT is not interpretable (device delay).

## 5. Nulls for F4 (the whole pipeline of section 4 re-run on null data; eligible = PAT-QC cases with >= 3 tested
epochs; case-bootstrap 95 % CI, 2,000 replicates, seed 0)

- **N1** (eab2d583 construction): in each epoch SBP with validity rolled against PAT by u ~ U[ceil(n/4),
  floor(3n/4)] beats.
- **N2** (implementation check): SBP with validity rolled by u0 ~ U{30, ..., n - 30} (independent stream), i.e. the
  data replaced by one draw of the test's own surrogate distribution; approximately exchangeable with the
  surrogates, so the false-edge rate must be <= 0.05 up to Monte Carlo error; a rate above 0.08 flags a bug.
- **N-b** (A1's across-epoch pairing; not exchangeable with the surrogates, secondary): PAT of epoch e with SBP of
  another usable epoch e' of the same case (uniform over the case's other usable epochs), aligned by beat index
  from each epoch's first beat, truncated to the shorter, validity valid_e & valid_e'. Expectation: <= 0.05
  approximately; reported, no bug flag.
Under N1 and N2 all admitted relations (F1, F2, F4) are re-run on their own null data and Holm is taken over the
null p-values, as in the real pipeline; under N-b (F4 only) the Holm family uses F1 / F2 under N2. The share of
eligible cases with null patient p <= 0.05 (family-free) is also reported. F1 / F2 N1 / N2 use eab2d583's seeds, so
their per-epoch values are identical to `pre_pat/`; only Holm (family now including F4) can change their rates.

## 6. Reports (added to eab2d583 section 5)

Graph size per patient (all 3,442 and QC 2,954; plus the PAT-QC subset) with the new `pat` variable node,
`prolonged_pat` nodes / episodes and F4 edges; evidence coverage (F4 edges with >= 1 pointer; pointer audit on the
same 100 random QC graphs, seed 2027: beat range inside its epoch, slope recomputed from the beats + PAT files);
F4 instantiation rate (share of eligible PAT-QC cases, and of all 2,954 QC cases), case-bootstrap CIs, pooled
share of tested epochs that are pointers, median pointers; N1 / N2 / N-b rates; `prolonged_pat` prevalence;
integrity (v2 layer unchanged; F1 / F2 per-epoch values identical to `pre_pat/`; the three eab2d583 beat concepts'
episodes identical to `pre_pat/`; F2 = `rsa_h`; F4 epoch slope = A1 `pat_sbp_slope`; `validate_refs` empty;
FastAPI 200 for 5 graphs; unit tests pass); **changes vs `pre_pat/`** per graph (nodes, variables, beat episodes,
fast edges by relation, pointers; F1 / F2 instantiation flips). Case study (iii), rule fixed now: the PAT-QC case
with F4 instantiated and the most `prolonged_pat` episode time (ties: lower case id); (i) and (ii) as eab2d583.

## 7. Runtime

`PYTHONPATH=src:. python scripts/55_graphs_fast.py --stage all --workers 8`; budget <= 40 min, <= 8 processes
(eab2d583 said <= 4; relaxed for the deadline, no effect on results: all seeds are per case / epoch). Nothing is
tuned after outputs are seen.
