# ClosedLoopBench 1.1 (closed-loop benchmark v4): pre-registered protocol

Written 2026-09-25 ~12:30 local time, before the v4 run on the full case set. Before registration the v4 code was run
only on (a) the existing unit tests (31/31 pass; v3 behaviour unchanged), (b) a 16-case real-data smoke run
(PCMCI+ on 4 cases, 50 bootstrap replicates) and a 12-case smoke run, both to check run time and errors, and
(c) run-time probes of the pooled matching on the real-arm control banks of all 3,442 cases (no outcomes looked
at). The 16-case smoke run printed its leaderboard and a per-method domain summary to the console and it was
seen (16 cases; not used for any decision below except run time). The runner records this file's sha256 in
`meta.json` and in the table headers; `protocol.sha256` holds the hash at registration. Code at registration is
hashed below, with byte copies in `registered_code/`. v3 outputs (`runs/iclr/closed_loop_v3/`) are not modified.

## 0. What is new in v4

1. **Data.** A 2 s grid for **all 3,442 VitalDB cases** from the native 1 s numerics
   (`clbench.bench.native_grid`, output `data/processed_vitaldb/native_2s_v4/`); v3 had 2 s data for 500 cases.
2. **Relations.** The v3 table (18) plus 11 relations (29 in total) with physics / physiology and policy signs
   (§2), on two new vitals (SpO2, BIS) and four new action streams (set tidal volume, set PEEP, set FiO2, derived
   MAC steps).
3. **Interpretive axis.** Every relation has a domain: `gas_exchange` (ventilation -> EtCO2, FiO2 -> SpO2),
   `cardiopulmonary` (PEEP -> MAP), `drug` (recorded drug set-points / rates), `drug_derived` (MAC steps derived
   from the measured MAC), `policy`. The question is whether the calibrated methods recover the physics relations
   while the drug responses are not identified (§8).

## 1. Data

- Grid: bins [2k, 2k + 2) s on the casestart clock of the native tables; value = median of the valid records
  in the bin (MAP = Solar8000 ART_MBP 20-200, HR 20-250, EtCO2 = Solar8000 ETCO2 5-100 (Primus EtCO2 only for a
  case with no Solar ETCO2), SpO2 50-100, BIS 1-100 with SQI >= 50; zero = disconnected, as v1-v3).
- Actions (one clock):
  - v3 agents (propofol and remifentanil TCI targets, ventilator set rate, phenylephrine, norepinephrine,
    nitroglycerin) from the v1-v3 events file, and their track-stop times from the 30 s tables (dose masking);
  - set tidal volume, set PEEP and set FiO2 from the native Primus set-point records with the v1-v3 set-point
    step detector (`clbench.data.vitaldb.step_changes`, debounce 10 s); the first record of a track is
    initialisation, not an event;
  - MAC: steps of >= 0.2 MAC of the 30 s rolling median of the measured Primus MAC (levels < 0.2 = off, reset after
    a 120 s gap). The vaporiser dial is not recorded, so MAC events are **derived exposure steps**, lagging the dial
    by the wash-in time (domain `drug_derived`).
- **Clock (deviation from the data notes, decided on evidence before registration).** The notes say the older
  tables (events, 30 s numerics) run 0-3 s later than the native casestart clock. Checked against the native
  Primus SET_RR record: events-file ventilator steps coincide with the native set-rate changes WITHOUT a shift
  (3,388 cases; median |dt| 0.50 s = the native table's 1 s rounding; 100% within 1 s) and are off by the offset
  WITH it (median 1.10 s; cases with offsets of 2.5-6.5 s are off by 3.0-7.1 s). So the events file and the 30 s
  tables are joined **without a shift**. `native_2s_v4/qc.csv` keeps both checks for every case.
- Counts (all events, both directions, incl. non-titration endpoints): ventilator rate 24,237 (3,388 cases),
  remifentanil 53,727 (2,871), propofol 23,576 (1,836), set TV 10,705 (2,786), set FiO2 11,965 (3,404),
  set PEEP 1,612 (856), MAC steps 44,138 (1,998), phenylephrine 999 (103), norepinephrine 668 (61),
  nitroglycerin 131 (30). Coverage (median share of 2 s bins): MAP 0.89, HR 0.94, EtCO2 0.91, SpO2 0.95, BIS 0.89;
  3,067 cases have BIS in >= 50% of bins.
- No case is excluded. (The beat-level QC flag |HR bias| > 5 bpm, 127 cases, concerns beat extraction, which v4
  does not use.)

## 2. Ground truth (29 relations; `clbench.bench.ground_truth`, citations in the code)

v3 relations R01-R12, P01-P06 unchanged. New:

| id | relation | expected | horizon | domain |
|---|---|---|---|---|
| R13 | set TV up -> EtCO2 down | - | 60 s | gas_exchange |
| R14 | set TV down -> EtCO2 up | - | 60 s | gas_exchange |
| R15 | set FiO2 up -> SpO2 up, events with pre-level SpO2 < 97% | + | 120 s | gas_exchange |
| R16 | PEEP up -> MAP down (modest, venous return) | - | 60 s | cardiopulmonary |
| R17 | MAC up -> MAP down (derived MAC steps) | - | 120 s | drug_derived |
| R18 | MAC up -> BIS down (derived MAC steps) | - | 120 s | drug_derived |
| R19 | propofol Ct up -> BIS down | - | 120 s | drug |
| P07 | SpO2 falling -> FiO2 up | - | 120 s trend | policy |
| P08 | EtCO2 rising -> set TV up | + | 120 s trend | policy |
| P09 | BIS rising -> propofol Ct up | + | 120 s trend | policy |
| P10 | BIS rising -> MAC up (derived) | + | 120 s trend | policy |

Domains of v3 relations: R01, R02 gas_exchange; R03-R12 drug; P01-P06 policy.
R15's ceiling restriction (pre-level SpO2 < 97%, last 30 s before the event / anchor) applies to the event designs
(naive, naive_lvl, matched, kink) and seqtrial, for events and control anchors alike. The regression methods
(xcorr, Granger, VARX, PCMCI+, VAR-LiNGAM) have no event selection and estimate R15 unrestricted (the ceiling can
only attenuate, not reverse, the sign). BIS enters every method as a recorded vital and the seqtrial pre-t state
(last level, recent level, 2-min trend) as a covariate.

## 3. Arms (negative controls)

As v3: `real`, `nc_random`, and K = 8 cross-patient derangements (salts 991, 7919, 101, 202, 303, 404, 505, 606).
The derangement is per action among the cases that use it; the new actions are appended after the v3 actions, so
the v3 actions consume the seeded generator in the same order as in v3.

## 4. Methods

The v3 primary methods, unchanged in definition: naive, naive_lvl, xcorr_v1, xcorr_dir, xcorr_pos, granger,
varx, pcmciplus, varlingam, matched, kink, seqtrial, typed_pooled (= varx under the v3 rule). Changes needed for the
larger table, or for run time on 3,442 cases with <= 4 worker processes on a shared machine:

1. **VARX with missing vitals.** v3 dropped every row with any vital missing. With BIS and SpO2 in the system that
   would drop cases without BIS from every equation. v4 fits each equation on its own rows with a finite outcome;
   missing lagged changes enter the regressors as 0; in the step-response propagation an equation with no rows
   contributes 0. VARX policy equations: a missing trend is 0 and a missing level is the case mean.
2. **VAR-LiNGAM sets.** v3 sets `core`, `pressor` plus `vent` (set TV, FiO2, PEEP, EtCO2, SpO2), `peepmap` (PEEP,
   MAP, HR) and `depth` (MAC, propofol, BIS, MAP). A relation is scored by the first set holding its cause and
   effect (for v3 relations this is exactly the v3 set). Pooled residual rows for DirectLiNGAM capped at
   **20,000** (v3: 50,000); 50 bootstrap replicates as v3.
3. **PCMCI+**: one seeded subsample of **60 cases** (pressor cases first, as v3), the same cases in every arm, on
   **30 s block means with lags to 240 s (tau_max 8)** (v3 at 2 s: 10 s blocks, tau_max 24 on 500 cases). Probe:
   ~12 s per arm per case with 10 s blocks vs ~3 s with 30 s blocks at v4's variable count.
4. **seqtrial**: pooled control bank capped at **10 anchors per case and action** (v3 2 s: 120 per case on 500
   cases; the v4 bank is still ~34,000 anchors per action), 19 pre-t features (L / R / T of five vitals + dose,
   elapsed time, co-interventions). Neighbours by exact brute force (torch, float32, CPU) instead of the KD-tree;
   on a test set the two give identical contrasts.
5. Not computed in v4: the pre-specified sensitivity variants (`*_clean`, `*_v1`, `*_nobc`, `*_coint`, `*_coadj`,
   `kink_prealt`, `seqtrial_nobc`, `seqtrial_noco`), the patient-specificity split-half test and the per-patient
   Granger fits (per-patient rows for display only). These were v3 results (runs/iclr/closed_loop_v3/).
6. xcorr curves stored as float16 per case (memory); pooled curves and peaks are computed in float64.

Everything else (horizons, windows, matching, bias correction, dose masking, ITT events, pre-t control pools,
co-intervention windows) is the v3 2 s configuration (`SCALES["2s_v4"]` = `SCALES["2s"]` with items 3-4).

## 5. Admission rule, metrics, outputs

Unchanged from v3 §6-7: calibrated estimate = real minus the mean of the 8 cross-draw estimates; SE adds the
between-draw variance; admitted iff CI excludes 0, BY-adjusted p <= 0.05 (family = the method's 29 relations),
stable in >= 75% of draws, >= 5 contributing cases. 200 case-bootstrap replicates (one weight matrix for every
arm), seed 0. Null false-admission rates: cross leave-one-out and random-time input. Leaderboard score =
correct-sign minus wrong-sign admissions on real data. New output: `domain_summary.csv` (per method and domain:
correct / wrong admissions) and the per-relation calibrated call grid in `leaderboard.md`.

Command (4 workers):
```
scripts/44_closed_loop_bench.py --bench-version v4 --workers 4 --lingam-max-rows 20000 \
    --protocol runs/iclr/closed_loop_v4/protocol.md
```
Output: `runs/iclr/closed_loop_v4/2s_all/`.

## 6. Primary methods for the predictions

The 12 scored primary methods: naive, naive_lvl, xcorr_v1, xcorr_dir, xcorr_pos, granger, varx, pcmciplus,
varlingam, matched, kink, seqtrial (typed_pooled duplicates varx). "Calibrated" = admitted by the v3 rule.
Domain sets: GAS = {R01, R02, R13, R14, R15}; DRUG = {R03-R12, R19}.

## 7. Pre-specified predictions (checked and reported whatever the outcome)

- **Q1 (rule on the cross null, replication of v3 P1 on the larger table).** Every primary method averages <= 1
  false admission (of 29) per held-out cross draw (leave-one-out).
- **Q2 (uncalibrated methods on the cross null).** naive, xcorr_pos and granger each have mean uncalibrated null
  discoveries >= 4 of 29 per cross draw.
- **Q3 (physics vs drug, the key axis).** For at least 7 of the 12 primary methods, the share of GAS relations
  admitted with the expected sign is larger than the share of DRUG relations admitted with the expected sign.
- **Q4 (physics calls are right when made).** Summed over the 12 primary methods, at least 80% of the admitted GAS
  cells have the expected sign, and there are at least 5 such correct cells.
- **Q5 (FiO2 -> SpO2).** R15 is admitted with the expected sign by at least one of the event designs (naive_lvl,
  matched, kink, seqtrial) and with the wrong sign by none of them.
- **Q6 (titration paradox at 2 s).** At least one primary method admits a DRUG relation with the wrong sign.

No prediction for `cardiopulmonary` (R16), `drug_derived` (R17, R18) or the new policy relations; they are
reported descriptively.

## 8. Limitations stated in advance

MAC steps are derived from a measured concentration, not recorded actions. FiO2 steps cluster at induction and
emergence (pre-oxygenation / before extubation), where the ceiling restriction removes most events. PEEP changes
are rare (856 cases). PCMCI+ runs on 60 cases only. The regression methods' R15 estimate is unrestricted.

## 9. Deviations

Any change after registration goes to `deviations.md` with its reason and the outputs it affects.

## Code at registration (sha256; byte copies in `registered_code/`)

- `src/clbench/eval/closed_loop_bench.py` 4896062b3605f2c86404dd0e4a67d954b98cc7b396b4b01eb629c9c3ef670da8
- `src/clbench/edges/typed_temporal_pooled.py` 9d8c09b9cb4d26e4602ec6cc1abf087789221b92f8cc0e189007b5d49ed5e7e9
- `scripts/44_closed_loop_bench.py` fa3be9a49701c0fb1c94efec4a4104ccbcb6e63495464f7dee4e5f2374586d11
- `tests/unit/test_closed_loop_bench.py` c76f03b2cbff92db45d6fb2382b702d5d271b0966a00f02745b3df07d6304f32
- `src/clbench/bench/__init__.py` d31949fdfd6098087624b97612f12805f8b93adf44096604d6cf07e0e473936e
- `src/clbench/bench/ground_truth.py` 287d112853969ef04ba845cc9379d2a11c4a4e95054c7d94fe4f0932dd56c34a
- `src/clbench/bench/loader.py` 1f25a98effaab379aeab73c9b2f1f99cc84ae1f066160372b073c9db149ea19e
- `src/clbench/bench/nulls.py` 1346db209f0790c9f834040cd894fb670a41b5b904c63c649aec88fb4565f3ba
- `src/clbench/bench/scoring.py` f05b9d56c0d80ace3db247fc81810b5a0eb68c78b662730f6dbc9aea7ee9b5f9
- `src/clbench/bench/sim.py` 687e31d7f5ee8dd3eec45800a85cc16316f7a6a7ddbd4c44e03774355d6c20f3
- `src/clbench/bench/native_grid.py` 9f943080f9709afd2a6208944a6a0a6264277c43342231db16dcd8402ab570c9
- `src/clbench/data/vitaldb.py` e988e3b742c3e691af871116d299c39a588c3f2763ed3efe9ca13adce3b68e38

(`typed_temporal_pooled.py` and `data/vitaldb.py` are not owned by this track and were not modified; they are hashed because the admission rule and the set-point step detector come from them.)
