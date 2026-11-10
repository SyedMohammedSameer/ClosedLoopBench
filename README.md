# ClosedLoopBench and feedback-robust patient graphs (anonymized copy)

This package accompanies a submission under double-blind review. It contains ClosedLoopBench (benchmark
version 1.1), the reference implementations of the 12 estimators, the negative-control constructions, the
calibrated admission rule, the Submission API for scoring a new estimator, the clinician-in-the-loop simulator,
the patient-graph construction used for Table 2, the VitalDB preprocessing scripts, the pre-registered protocols
with their hashes, and 13 unit-test modules. Names, affiliations, paths and the project's working name were
removed mechanically (`ANONYMIZATION.md`); no code logic was changed. Python package name: `clbench`.

## What ClosedLoopBench is

Anesthesia is a closed loop: clinicians titrate drugs and ventilation in response to the vitals, so a temporal
relation estimated from monitoring data mixes the patient's *response* with the clinician's *policy*.
ClosedLoopBench scores temporal relation estimators on real data where the sign of each relation is fixed in
advance:

* **Data.** 3,442 VitalDB surgical cases (12,653 recorded hours), vitals (MAP, HR, EtCO2, SpO2, BIS, MAC) on a
  2 s grid built from the monitor numerics at their native time stamps, and 171,758 action events: propofol and
  remifentanil infusion targets, four ventilator settings (set rate, tidal volume, PEEP, FiO2), three vasoactive
  infusion rates and derived volatile-anesthetic (MAC) steps.
* **29 relations** (`clbench.bench.ground_truth`, `runs/iclr/closed_loop_v4/BENCHMARK_CARD.txt`): 19 responses
  (five gas-exchange relations with signs from respiratory physics, eleven drug responses with signs from
  pharmacology, two volatile-exposure relations, PEEP->MAP) scored at 60 s or 120 s, and 10 policies with signs from
  documented practice (e.g. rising MAP -> remifentanil increase). Increases and decreases are separate relations.
* **12 estimators** (`clbench.eval.closed_loop_bench`): positive-peak, signed-peak and direction-aware lagged
  cross-correlation; pairwise Granger; PCMCI+ (tigramite); VAR-LiNGAM (lingam); VARX (= the typed pooled
  estimator, `typed_pooled`); naive event and naive level contrasts; a matched event design with bias-corrected
  placebo times; an interrupted-trend (kink) design; a sequential-trial contrast.
* **Two negative-control streams** (`clbench.bench.nulls`): *transplant* (K = 8 seeded cross-patient derangements:
  each case receives another case's action streams, aligned by elapsed time) and *random-time* (each case's own
  steps at sorted uniform random times).
* **Admission rule** (`clbench.edges.typed_temporal_pooled.admit_multi`): the calibrated estimate is the real-data
  estimate minus the mean over the 8 transplant draws; its SE adds the paired case-bootstrap variance (200
  replicates, one weight matrix for every arm) and the between-draw variance. A relation is admitted iff the 95%
  CI excludes 0, the Benjamini-Yekutieli adjusted p over the estimator's relations is <= 0.05, the single-draw
  calibrated CI excludes 0 with the same sign in >= 75% of draws, and >= 5 cases contribute. Metrics: uncalibrated
  null discoveries per transplant draw, calibrated admissions with the expected / wrong sign per domain, false
  admissions with a held-out transplant draw or the random-time stream as input.

## Layout

```
src/clbench/            library (modules listed below)
scripts/                numbered pipeline scripts (download, ingest, native extraction, benchmark, graphs, simulator)
tests/unit/             unit tests (pytest)
runs/iclr/<analysis>/   pre-registered protocol texts, their hashes, the v4 registered-code snapshot, benchmark card
checkpoints/cbm/persistence_calibration.json   persistence-tier cut-offs used by the graph construction
data/vitaldb_case_index.csv, data/pat_cases.csv   small case lists (no patient data)
colab/                  the Colab drivers under which the long CPU jobs were run (download, ingest, native extraction, PAT)
scripts/extended/       the five extended analyses (A-E, see "Extended analyses"); library code in src/clbench/extended/
PROTOCOLS.md            registered vs shipped protocol hashes;  ANONYMIZATION.md  what was substituted
```

Library modules included (the import closure of the scripts above):

| Module | Role |
|---|---|
| `clbench` |  |
| `clbench.beats` | beat and breath tables from raw waveforms |
| `clbench.bench` | ClosedLoopBench package (benchmark name, version, relation table) |
| `clbench.bench.ground_truth` | the relations, expected signs, horizons, domains and the sources for each sign |
| `clbench.bench.loader` | VitalDB processed numerics + events -> `CaseData` (dose masking after a track stops) |
| `clbench.bench.native_grid` | v4 (1.1): 2 s grid and one-clock action streams from the native 1 s numerics |
| `clbench.bench.nulls` | random-time null and K = 8 cross-patient derangements (registered salts) |
| `clbench.bench.scoring` | Submission API: pooled estimates + paired case-bootstrap replicates -> admission rule, leaderboard line |
| `clbench.bench.sim` | clinician-in-the-loop simulator adapter with exact intention-to-treat truth |
| `clbench.cbm` | concept registry and concept-bottleneck heads (needed by the tier calibration script only) |
| `clbench.data` |  |
| `clbench.data.download` | download and SHA-256 verification helpers |
| `clbench.data.preprocess` | shared windowing / resampling / filtering pipeline |
| `clbench.data.split` | patient splits |
| `clbench.data.vitaldb` | VitalDB ingest: windows, numerics, action events, set-point step detector |
| `clbench.edges` |  |
| `clbench.edges.co_evidence` | CKG baseline: Jaccard co-occurrence edges |
| `clbench.edges.controlled` | controlled (adjusted) relation estimates used by the typed estimator |
| `clbench.edges.fast_coupling` | beat-level couplings (respiration->SBP, RSA, baroreflex, PAT-SBP) |
| `clbench.edges.granger` | CKG baseline: pairwise Granger edges |
| `clbench.edges.temporal` | CKG baseline: lagged cross-correlation PRECEDES / FOLLOWS edges |
| `clbench.edges.typed_temporal` | typed pooled hazard relations between concepts (policy / response / association) |
| `clbench.edges.typed_temporal_pooled` | the calibrated admission rule (`admit_multi`) shared by the benchmark and the graphs |
| `clbench.encoders` | signal encoders (imported by the tier calibration script; training code not included) |
| `clbench.encoders.abp` |  |
| `clbench.encoders.base` |  |
| `clbench.encoders.cache` |  |
| `clbench.encoders.ecg` |  |
| `clbench.encoders.linear_probe` |  |
| `clbench.encoders.ppg` |  |
| `clbench.encoders.pretraining` |  |
| `clbench.encoders.resp` |  |
| `clbench.eval` | evaluation helpers (bootstrap CIs, tier-1 metrics; imported by the package `__init__`) |
| `clbench.eval.closed_loop_bench` | the 12 estimators, scale configurations, bootstrap, calibrated admission tables, leaderboard |
| `clbench.eval.eraser` |  |
| `clbench.eval.stats` |  |
| `clbench.eval.tier1_intrinsic` |  |
| `clbench.fused_inference` | concept probabilities from the concept models (tier calibration script only) |
| `clbench.fusion` | multi-modal fusion module (tier calibration script only) |
| `clbench.kg_construct` | v1 graph construction (correlational baseline CKG) |
| `clbench.kg_v2` | patient graph schema 3.x construction (concept nodes, evidence pointers, typed relations) |
| `clbench.persistence` | persistence-tier calibration (transient / episodic / persistent concept runs) |
| `clbench.extended` | extended analyses (see "Extended analyses") |
| `clbench.extended._dcor_shim` | pure-numpy distance correlation used in place of the `dcor` package when it is absent (tigramite's GPDC test) |
| `clbench.extended._numba_shim` | identity `jit` used in place of `numba` when it is absent (tigramite's CMIknn test) |
| `clbench.extended.c_sim_grid_summary` | extended C: metrics, tables and figures of the simulation grid (power, wrong sign, false positives, bias against the ITT truth) |
| `clbench.extended.matched_ckg` | extended A: CKG statistics (cross-correlation, Granger, Jaccard, PCMCI+) on the graph construction's candidate relations and null replicates |
| `clbench.extended.nonlinear` | extended D: the paper's cohort / arms / seeds / bootstrap weights and the nonlinear estimators (rank cross-correlation, PCMCI+ with GPDC or CMIknn, boosted-tree Granger, dose-binned VARX, monotone PAT-SBP) |
| `clbench.schema` | graph schema (pydantic models) |
| `clbench.sim` |  |
| `clbench.sim.clinician_loop` | the clinician-in-the-loop simulator (physiology, controller, unrecorded driver) |

## Install

Python 3.11.14 (3.11 or newer). Versions used for the paper's runs: numpy 2.4.6, pandas 3.0.3, pyarrow 24.0.0, scipy 1.17.1, statsmodels 0.14.6, scikit-learn 1.9.0, torch 2.12.0, pydantic 2.13.4, tigramite 5.2.10.1, lingam 1.13.0, pygam 0.12.0, matplotlib 3.10.9, tqdm 4.67.3, vitaldb 1.7.2, wfdb 4.3.1 (`requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                      # exact versions of the paper's environment
pip install --no-deps -r requirements-nodeps.txt     # lingam and pygam (see the note in that file)
pytest tests -q                                      # unit tests (conftest.py puts src/ on sys.path)
```

`pip install -e .` (version floors from `pyproject.toml`) also works: a resolver then backs off to an older
scipy / lingam pair (with pip 25 / uv: scipy 1.16.3 and lingam 1.12.2) because of lingam's declared scipy cap, which
is not the environment used for the paper's runs. The two-step route above reproduces the paper's environment exactly.

The scripts insert `src/` into `sys.path` themselves; running them as `PYTHONPATH=src python scripts/<name>.py`
from the package root also works without installing. Scripts 50, 55 and 57 load sibling scripts by path and need
`PYTHONPATH=src:.`.

## Data: VitalDB from PhysioNet

VitalDB is public and openly licensed (Creative Commons Attribution 4.0). Nothing from it is redistributed here.

* Lee H-C, Park Y, Yoon SB, Yang SM, Park D, Jung C-W. VitalDB, a high-fidelity multi-parameter vital signs
  database in surgical patients. *Scientific Data* 9, 279 (2022). doi:10.1038/s41597-022-01411-5
* Lee H-C, Jung C-W. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients
  (version 1.0.0). PhysioNet (2022). doi:10.13026/czw8-9p62 - https://physionet.org/content/vitaldb/1.0.0/
* Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215-e220 (2000).

```bash
# 1. download the .vital files (open PhysioNet S3 mirror, SHA-256 verified, resumable; ~94 GB for all 6,388
#    cases, --max-tier 2 keeps the 3,442 cases with ECG, PPG, arterial and capnogram waveforms)
python scripts/01c_download_vitaldb.py --max-tier 2
#    -> data/raw/physionet.org/files/vitaldb/1.0.0/vital_files/*.vital, data/vitaldb_case_index.csv
#    clinical_data.csv from the same PhysioNet page goes next to vital_files/

# 2. 30 s windows, 30 s numerics, action events, cohort manifest (CPU-bound; ran on Colab, colab/vitaldb_ingest_colab.py)
python scripts/14_vitaldb_ingest.py --workers 8
#    -> data/processed_vitaldb/{numerics,events,summaries}/<pid>.parquet|json, data/vitaldb_manifest.csv

# 3. native-rate tables: beats, breaths and 1 s numerics on the clinical case-start clock
#    (about 12 CPU-hours; ran on Colab, colab/run_all_colab.py section A)
PYTHONPATH=src python scripts/48_vitaldb_native_extract.py --workers 12
#    -> data/processed_vitaldb/native/{beats,breaths,numerics_1s,summaries}/

# 4. only for ClosedLoopBench 1.0 (--bench-version v3 --cases fine): 2 s numerics for 500 cases
python scripts/18_vitaldb_fine_numerics.py --step-sec 2 --limit 500 --workers 8     # -> data/processed_vitaldb/fine_2s/

# 5. only for the beat layer of the graphs and the PAT analyses: delay-aware pulse arrival time
PYTHONPATH=src python scripts/56_pat_native.py --workers 8                           # -> data/processed_vitaldb/native/pat/
```

Expected layout (`clbench.bench.loader`, `clbench.bench.native_grid`):

```
data/processed_vitaldb/numerics/<pid>.parquet      30 s windows (15 s stride): map, hr, etco2, ... and the action tracks
data/processed_vitaldb/events/<pid>.parquet        patient_id, t_sec, agent, value_before, value_after
data/processed_vitaldb/native/numerics_1s/<pid>.parquet   1 s monitor / ventilator / pump numerics (case-start clock)
data/processed_vitaldb/native/summaries/<pid>.json
data/processed_vitaldb/native/{beats,breaths,pat}/<pid>.parquet   beat layer (graphs with fast couplings, PAT analyses)
```

## Reproduce Table 1 (ClosedLoopBench 1.1, 2 s grid, 3,442 cases)

```bash
# build the 2 s grid and the one-clock action streams for every case (67 s with 4 workers)
PYTHONPATH=src python scripts/44_closed_loop_bench.py --bench-version v4 --build-grid --workers 4
#    -> data/processed_vitaldb/native_2s_v4/{grid,events}/, stops.parquet, qc.csv

# the registered run: 12 estimators x (real + random-time + 8 transplant draws), 200 case-bootstrap replicates,
# PCMCI+ on a seeded 60-case subsample; 48 min with 4 worker processes on a 14-core laptop (2,874 s wall)
PYTHONPATH=src python scripts/44_closed_loop_bench.py --bench-version v4 --scale 2s --cases all --workers 4 \
    --lingam-max-rows 20000 --protocol runs/iclr/closed_loop_v4/protocol.md
#    -> runs/iclr/closed_loop_v4/2s_all/: main_table.md (real-data estimates), nc_tables.md and null_discovery.csv
#       (uncalibrated null discoveries, "Null" column), calibrated.csv and domain_summary.csv (calibrated admissions,
#       expected / wrong sign per domain), leaderboard.md / leaderboard.csv (held-out transplant and random-time
#       false admissions, "RT" column), meta.json (arguments, protocol and code hashes, seeds, timings)

# smoke test (12 cases, a few minutes)
PYTHONPATH=src python scripts/44_closed_loop_bench.py --bench-version v4 --limit 12 --workers 2 --out-dir /tmp/clb_smoke

# ClosedLoopBench 1.0 (18 relations; 30 s all cases, 2 s on the 500 fine cases) stays reproducible:
PYTHONPATH=src python scripts/44_closed_loop_bench.py --bench-version v3 --scale 30s --cases all --workers 8 \
    --protocol runs/iclr/closed_loop_v3/protocol.md
```

Seeds are fixed (`--seed 0`; per-case seeds `seed * 100003 + i`; transplant salts 991, 7919, 101, 202, 303, 404,
505, 606). Worker count does not change the results; the pooled sequential-trial matching uses exact brute-force
nearest neighbours (torch, CPU).

## Reproduce Table 2 (patient graphs: false relation instances, replication, graph size)

```bash
# concept-level typed relations: pooled hazard model with patient fixed effects, 19 transplant + 19 shift null
# replicates, split-half replication, held-out likelihood; and the CKG baselines on the same concept series
# (~30 min with 12 workers)
PYTHONPATH=src:. python scripts/47_typed_temporal_edges.py --workers 12 --protocol runs/iclr/typed_edges/protocol.md
#    -> runs/iclr/typed_edges/confirmatory/{results.json,report.md,...}

# graphs for all 3,442 cases (concept layer), false instances under the 38 null replicates, CKG comparison,
# pointer audit (~2 min for the graphs; the null stage dominates)
PYTHONPATH=src:. python scripts/50_build_kgs_v3.py --workers 8
#    -> runs/iclr/graphs_v2/{results.json,report.md,null_replicates_vitaldb.csv,graph_size_vitaldb.csv,kgs_vitaldb/}

# beat-level concepts and fast couplings (respiration->SBP, RSA, PAT-SBP) added to every graph, with their nulls
# (needs data/processed_vitaldb/native/{beats,breaths,pat}; ~11 min with 8 processes)
PYTHONPATH=src:. python scripts/55_graphs_fast.py --stage all --workers 8
#    -> runs/iclr/graphs_v2_fast/{results.json,report.md,kgs/}

# PAT analyses at scale (slope sign in 94.3% of cases, patient specificity); 57 reads epoch tables written by 54
PYTHONPATH=src:. python scripts/54_fast_edges_vitaldb.py --stage extract --workers 8
PYTHONPATH=src:. python scripts/57_pat_vitaldb.py --stage extract --workers 8
PYTHONPATH=src:. python scripts/57_pat_vitaldb.py --stage specificity --workers 8
PYTHONPATH=src:. python scripts/57_pat_vitaldb.py --stage report
```

The persistence-tier cut-offs consumed by the graph construction are shipped
(`checkpoints/cbm/persistence_calibration.json`); `scripts/04b_calibrate_persistence.py` is the script that produced
them from the concept models (model training code is not part of this package). The MIMIC-IV-WDB columns of Table 2
need the MIMIC waveform pipeline and credentialed clinical tables, which are outside this package; script 50 skips
the MIMIC stage when `data/kgs/` is absent.

## Simulator

`clbench.sim.clinician_loop` simulates a cohort with linear physiology, a clinician controller (gain, anticipation,
alarms, reaction to an unrecorded driver) and drug / ventilator actions, with an exact intention-to-treat truth by
paired twin simulation.

```bash
PYTHONPATH=src python scripts/44_closed_loop_bench.py --sim-validate --workers 4   # every estimator on simulated cohorts
PYTHONPATH=src python scripts/46_sim_closed_loop.py --exp v2                        # the registered simulation study
PYTHONPATH=src python scripts/46_sim_closed_loop.py --exp v2report                  # -> runs/iclr/sim_closed_loop/v2/results.md
```

## Submission API: scoring a new estimator

A method is scored from its pooled estimates, one per (arm, relation) at the relation's primary horizon, each
with case-bootstrap replicates computed with the benchmark's shared weight matrix (so replicates of the real arm
and of the null arms are paired by patient). The admission rule and the leaderboard line are then identical to
those of the 12 reference estimators.

```python
from clbench.bench import scoring
from clbench.eval import closed_loop_bench as clb

clb.use_table("v4")                                            # the 29-relation table (default "v3": 18 relations)
W = scoring.bootstrap_weights(n_cases)                         # (n_cases, 200) multiplicities, seed 0, sorted case ids
sub = scoring.Submission("my_method", scale="2s_v4")
for arm in scoring.ARMS:                                       # "real", "nc_random", "nc_cross", ..., "nc_cross8"
    for rid in scoring.RELATION_IDS:                           # "R01" ... "R19", "P01" ... "P10"
        # per-case sums and counts of your estimator's contrasts on this arm (or your own replicates)
        est, reps = scoring.event_estimate(case_sums, case_counts, W)
        sub.add(arm, rid, est, reps, n_cases=int((case_counts > 0).sum()))
calibrated, line = scoring.score(sub)                          # calibrated admission table, leaderboard row
print(line["text"])
```

The arms' action streams come from `clbench.bench.loader` (real) and `clbench.bench.nulls` (`nc_random_times`,
`nc_cross_patient` with the registered salts); `clb.CaseData` holds a case's vitals and events. Reference
implementations of all 12 estimators are in `clbench.eval.closed_loop_bench` (`run_case` and the `Aggregator`).

## Extended analyses

`scripts/extended/` holds the five extended analyses; their library code is `clbench.extended`.
They reuse the benchmark's cohort, arms, seeds, bootstrap weights and admission rule. Three library files gained
optional keyword arguments for them (`bench/sim.py`: controller gain and a sigmoid Emax remifentanil model;
`eval/closed_loop_bench.py`: `pcmci_case(cond_ind_test=..., sign_fn=...)`, `varx_step_response(cols=...)`;
`sim/clinician_loop.py`: `emax_deviation`); every default is unchanged, the unit tests pass, and the 12-case
benchmark smoke run is identical with and without the additions. Every script writes under
`runs/extended/<analysis>/` and never into `runs/iclr/` or `data/`. Runtimes are wall-clock on a 14-core laptop
with the stated number of worker processes (one BLAS thread each).

**Inputs.** B, E and the validation stages of A and D read the outputs of the registered runs, which are not
shipped (their `meta.json` files hold local paths). Regenerate them first with the commands of the Table 1 and
Table 2 sections: `runs/iclr/closed_loop_v4/2s_all/` and `runs/iclr/closed_loop_v3/30s_all/` (script 44),
`runs/iclr/typed_edges/confirmatory/` (47), `runs/iclr/graphs_v2/` (50), `runs/iclr/pat_vitaldb/` (54, 57).
A and D also read `data/processed_vitaldb/` (D: the 2 s grid of `44 --build-grid`). C needs no data.

```bash
# A. Matched construction ladder: the correlational KG (cross-correlation, Granger, Jaccard, PCMCI+, per patient
#    and pooled) and the feedback-robust construction on the same 176 candidate relations, the same 38 null
#    replicates (19 transplant + 19 shift) and the same pointer rule; false relation instances per graph at every
#    rung, null calibration, time-split replication. Stages cache; run them in this order (about 3 h, 6 workers:
#    validate 5 s, perpatient 42 min, pcmci 6 min, pooled 29 min, pooled_adj 1-2 h, instances 2 min, summary 15 s)
for s in validate perpatient pcmci pooled pooled_adj instances summary; do
    PYTHONPATH=src python scripts/extended/A_matched_ckg.py --stage $s --workers 6; done
#    -> runs/extended/A_matched_ckg/{results.json,validation.json,ladder*.md}
#    smoke test of the whole ladder on 300 cases (about 11 min): add --n-cases 300 --out runs/extended/A_matched_ckg/smoke300

# B. ClosedLoopBench results by evidence tier (A: respiratory physics, B: pharmacology / physiology, C: documented
#    practice): re-partitions the published run files, no estimator is re-run (2-10 s, 1 process)
PYTHONPATH=src python scripts/extended/B_tiers.py
#    -> runs/extended/B_tiers/{2s,30s}/{tier_table.csv,tier_tables.tex,call_matrix_*.csv,...}, extended/results/B_tiers.md

# C. Simulated closed-loop systems with known ground truth through the full benchmark pipeline (12 estimators,
#    random-time and 8 transplant nulls, calibrated admission rule): feedback gain x unrecorded confounding x
#    Emax nonlinearity x sampling rate x policy type, an active and an inert drug, exact ITT truth by paired twin
#    simulation. 76 cells (38 configurations x 2 replicates, n = 600 patients): 4.5 h with 4 workers; the four
#    strong-Emax cells (--configs emax_strong,emax_strong_wz1) another 30 min; one cell (smoke --n 100): 3 min
PYTHONPATH=src python scripts/extended/C_sim_grid.py run --n 600 --workers 4 --reps 2 --wave2
PYTHONPATH=src python scripts/extended/C_sim_grid.py summarize                      # 17 s
#    -> runs/extended/C_sim_grid/{cells/<cell>/*.csv, cells_long.csv, cells_summary.csv, tables.md, figures/}

# D. Nonlinear robustness of the Table 1 conclusions: Spearman (rank) cross-correlation, dose-binned VARX,
#    PCMCI+ with the ParCorr path check, boosted-tree Granger (500-case subsample), monotone PAT-SBP; every stage
#    uses the paper's cohort, arms, seeds and bootstrap weights (about 2.5 h with 4 workers: xcorr 3 min,
#    varx 1 bin 3 min and 3 bins 70 min, pcmci parcorr 3 min, gbm 55 min, pat 2 min, report 1 s)
for a in "--stage xcorr" "--stage varx --dose-bins 1 --arms real" "--stage varx --dose-bins 3" \
         "--stage pcmci --test parcorr" "--stage gbm --n-cases 500 --stride-sec 40 --folds 3" "--stage pat" "--stage report"; do
    PYTHONPATH=src python scripts/extended/D_nonlinear.py $a --workers 4; done
#    -> runs/extended/D_nonlinear/<stage>/{pooled,calibrated,null_discovery,leaderboard}.csv, report_tables.md
#    PCMCI+ with the nonlinear tests (GPDC with a random-Fourier-feature GP, CMIknn with a precomputed null):
#    60-85 s per case-arm for GPDC when run alone (the full 60 cases x 10 arms is 10-14 h; a few cases run for
#    hours, there is no per-case time cap) and a median of 27 s per case-arm for CMIknn; the authors ran a
#    20-case GPDC pilot and a reduced CMIknn design within the time available
PYTHONPATH=src python scripts/extended/D_nonlinear.py --stage pcmci --test gpdc --workers 4 --limit 20
PYTHONPATH=src python scripts/extended/D_nonlinear.py --stage pcmci --test cmiknn --workers 4 --arms real,nc_random,nc_cross,nc_cross2 --limit 12

# E. Sensitivity of the admissions to the fixed thresholds: part 1 the benchmark rule (q, BY vs BH, stability
#    >= 0.75, >= 5 cases, the CI condition; one factor at a time and a 400-setting grid), part 2 the concept-level
#    rule (q, null rule, testability thresholds) with the false instances per graph re-instantiated; re-admission
#    from the run files, no estimator is re-run (37 s and 31 s, 1 process; --figures-only re-renders the figures)
PYTHONPATH=src python scripts/extended/E_sensitivity.py --part 1
PYTHONPATH=src python scripts/extended/E_sensitivity.py --part 2
#    -> runs/extended/E_sensitivity/{part1_oat_groups.csv,part1_grid_*.csv,part2_*.csv,fig_*.png}
```

## Pre-registered protocols

Each confirmatory analysis was specified in a written protocol hashed before its run (first eight hex digits as in
the paper's appendix). The shipped texts are anonymized copies; `PROTOCOLS.md` lists the registered and the shipped
hash of each file.

| Analysis | Registered hash | Shipped file |
|---|---|---|
| ClosedLoopBench 1.1: 29 relations, 2 s, 3,442 cases (Table 1) | `418281f1` | `runs/iclr/closed_loop_v4/protocol.md` |
| ClosedLoopBench 1.0: 18 relations, 30 s (3,442 cases) and 2 s (500 cases) | `5e578794` | `runs/iclr/closed_loop_v3/protocol.md` |
| Slow-response specificity (early/late split, 11 relations) | `482db32f` | `runs/iclr/closed_loop_v2/protocol.md` |
| Clinician-in-the-loop simulator (14 hypotheses) | `7b75ce95` | `runs/iclr/sim_closed_loop/v2/protocol.md` |
| Simulator amendment: descriptive phase-grid analysis | `e9e25964` | `runs/iclr/sim_closed_loop/v2/protocol_amendment1.md` |
| Concept-level typed relations (hazard model, nulls, sign checks) | `43947eaf` | `runs/iclr/typed_edges/protocol.md` |
| Patient graphs, concept layer (pointer rule, false instances; Table 2) | `c767688d` | `runs/iclr/graphs_v2/protocol.md` |
| Beat concepts and fast couplings in graphs | `eab2d583` | `runs/iclr/graphs_v2_fast/protocol.md` |
| Addendum: PAT-SBP edge, prolonged PAT | `20f873f5` | `runs/iclr/graphs_v2_fast/protocol_pat.md` |
| Fast couplings at scale (gates, epoch statistics, baroreflex bar) | `af14df06` | `runs/iclr/fast_vitaldb/protocol.md` |
| PAT analyses (delay-aware PAT, slope sign bar, specificity rule) | `230b57c5` | `runs/iclr/pat_vitaldb/protocol.md` |
| Fast-coupling specificity, MIMIC-IV-WDB and VitalDB | `6d88096d` | `runs/iclr/fast_specificity/protocol.md` |
| Concept heads, six models (not reproducible from this package: training code not included) | `ce7ed6fe` | `runs/iclr/nodes_multiseed/protocol.md` |
| Counterfactual faithfulness (not reproducible from this package: concept-model code not included) | `6037da73` | `runs/iclr/faithfulness/protocol.md` |

## Tests

13 unit-test modules exercise the included modules: `test_clinician_loop.py`, `test_closed_loop_bench.py`, `test_edges_controlled.py`, `test_encoders.py`, `test_fast_coupling.py`, `test_kg_construct_v2.py`, `test_kg_v2.py`, `test_linear_probe.py`, `test_preprocess.py`, `test_schema_v2.py`, `test_split.py`, `test_typed_temporal.py`, `test_vitaldb.py`. `pytest tests -q` runs them in about a
minute (the PCMCI+ and VAR-LiNGAM tests are skipped if tigramite or lingam are not installed).

## License

MIT (see `LICENSE`). VitalDB is distributed by its authors under CC BY 4.0 and must be obtained from PhysioNet.
