# Track D (final wave): beat-level concepts and mechanistic fast edges in the CLBench v2 graphs. Pre-registered protocol

Written 2026-09-25 ~12:50 local time, before any beat-level concept, fast-edge statistic, surrogate p-value or null
graph was computed. Runner: `scripts/55_graphs_fast.py`; library: `src/clbench/kg_v2.py` (extended, backward
compatible). The build stage refuses to run unless this file's sha256 equals `protocol.sha256`, and every
output records the hash. Deviations are listed in the report with their reason.

**Seen before registration** (no concept or edge statistic of this track):
- the results ledger (waves 1-4) and track B's final-wave report (`runs/iclr/fast_vitaldb/`, protocol af14df06):
  QC 3,227 -> 2,962 -> 2,954 cases; baroreflex ARX positive in 63.3 % [61.5, 64.9] of cases (registered
  reliability bar 90 %: not met); native PAT censored (PLETH foot 0.60-0.72 s after the R-peak, search window
  ends at 0.60 s; `t_ppg_foot` missing for ~91 % of beats in case vdb0001), PAT-SBP 56.5 % (not a test);
  PPV / SPV / RSA distributions; construct validity CV2, CV3 (weak), CV5, CV6 hold, CV1 fails (reversed), CV4
  not shown, CV7 not assessable; patient specificity REAL for PPV, RSA, sequence BRS, RMSSD, with PPV and RSA
  surviving the twin adjustment and sequence BRS not; every edge less persistent than mean HR;
- column layouts of `beats/`, `numerics/` and track B's `epochs.parquet` / `cases.parquet`, and their first
  three rows for case vdb0001; the wave-4 v2 graph files and `graph_size_vitaldb.csv` (sizes, as in ledger G5).

## 1. Cases and clocks

- Graphs are rebuilt for all 3,442 VitalDB cases. The wave-4 v2 content (concept x tier nodes, episodes, action
  and variable nodes, population relations, typed edges, pointers) is copied unchanged from
  `runs/iclr/graphs_v2/kgs_vitaldb/{pid}.json.gz` and checked field by field.
- Beat-level content is added only for track B's QC cases (`cases.parquet`, `g4_20min == True`: 2,954 cases).
  The other 488 graphs carry the fast layer's population registry with the reason they have no beat content.
- Clock: v2 graphs use the VitalFile loader origin (`numerics/window_start_sec`); native tables use clinical
  casestart. A native time t_cs maps to the v2 clock as t = t_cs - `loader_origin_minus_casestart_s` (per-case
  summary). Every time stored in a graph is on the v2 clock; beat pointers also carry row indices into
  `data/processed_vitaldb/native/beats/{pid}.parquet`.

## 2. Beat-level concept nodes (per 30-s window on the v2 grid, 15-s stride)

Window i covers [ws_i, ws_i + 30) on the v2 clock. Beats of window i: rows whose R-peak time (v2 clock) lies in
the window and whose interval ends inside it (t_r + rri/1000 < ws_i + 30). RR intervals pass an artefact filter
300 <= rri <= 2000 ms. A rhythm window is **observed** iff >= 20 RR intervals pass and they cover >= 24 s;
otherwise the concept is unobserved (-1) in that window.

| concept | on in an observed window iff | basis |
|---|---|---|
| `irregular_rr` (signal concept: AF-like RR irregularity; VitalDB has no rhythm labels, so this is **not** a rhythm diagnosis) | after removing the 2 largest and 2 smallest RR intervals (ectopy / missed-beat robust trimming): RMSSD / mean RR >= 0.10 **and** normalised Shannon entropy of the RR histogram (16 equal bins between min and max) >= 0.70 **and** the turning-point count within (2N-4)/3 +/- 2 sqrt((16N-29)/90) (random-series range) | Dash, Chon, Lu, Raeder 2009, Ann Biomed Eng 37:1701 (nRMSSD, Shannon entropy, turning-point ratio) |
| `low_hrv` | RMSSD of successive differences of consecutive passing intervals, after dropping the 2 largest absolute differences, < T_low, where T_low = 10th percentile of `rmssd` over track B's usable 5-min epochs of the QC cases (a cohort-relative reference fixed from track B's table before any window is computed; its value is recorded). Requires >= 15 successive differences | cohort-relative; no ultra-short-term clinical cut-off exists under anaesthesia |
| `high_ppv` | observed only if the window lies entirely inside one of track B's usable controlled-ventilation 5-min epochs (`usable & controlled_vent`), `irregular_rr` is observed and off in the window, and >= 3 breaths qualify under `respiratory_pressure_variation` (valid beats, >= 3 per breath and >= 70 % of the breath); on iff that function's PPV (median over the window's breaths) > 13 % | Michard et al. 2000, AJRCCM 162:134 (PPV 13 %); validity needs controlled ventilation and a regular rhythm (De Backer et al. 2005, Intensive Care Med 31:517). Tidal volume per kg is not enforced (limitation) |
| `prolonged_pat` | **not built**: track B found native PAT censored (3-12 % of beats; foot beyond the 0.6-s window). Recorded as an abstained concept | track B QC |

Tiers and episodes follow v2 exactly: persistence tiers from `kv.thresholds_for` (these concepts are not in the
persistence calibration, so the fallback 60 s / 600 s tiers apply; stated in the graph), node ids
`{pid}:b:{concept}:{tier}`, one episode per activation run (`b{k}.{r}`). Each beat episode carries its time
span, the first and last beats row index of its windows, the number of beats and the median of the defining
statistic over its windows. Local codes (`local:irregular_rr`, `local:low_hrv`, `local:high_ppv`) are used; no
SNOMED id is invented.

## 3. Mechanistic fast edges

Population registry (identical in every graph):

| id | relation | status | basis (track B / ledger) |
|---|---|---|---|
| F1 | respiration -> beat SBP (respiratory systolic modulation; PPV/SPV) | **admitted for per-patient testing** | PPV construct validity CV5, CV6 hold; PPV patient specificity REAL and survives its twin adjustment |
| F2 | respiration -> RR interval (RSA) | **admitted for per-patient testing** | CV2 holds (weak); RSA patient specificity REAL and survives its twin adjustment |
| F3 | beat SBP -> RR interval (baroreflex) | **abstained** | ARX positive in 63.3 % [61.5, 64.9] of 2,944 cases; registered positive-control bar 90 % not met; sequence-BRS persistence is a signal property (twin-adjusted r 0.16) |
| F4 | PAT <-> SBP | **abstained in VitalDB** | native PAT censored; population sign not assessable at scale (56.5 %, not a test). Recovered in MIMIC (E2, 100 % of 50) and in 8/9 local delay-aware VitalDB files (exploratory); needs re-extraction |

F1 and F2 are magnitude couplings; they have no sign to recover. What is admitted per patient is phase locking
of the target to the respiratory cycle beyond a per-breath phase-randomisation null.

**Per-epoch statistic.** On each of track B's usable controlled-ventilation epochs of the case (5-min, casestart
clock, beats with t_r in [t0, t1)): harmonic amplitude A = 2 sqrt(b^2 + c^2) of the OLS fit of the 15-beat
running-mean-detrended target (valid beats; F1: `sbp`; F2: `rri`) on [1, cos phi, sin phi], phi = `resp_phase`;
>= 60 usable beats (script 52's `harmonic_rsa`, which for F2 must reproduce track B's `rsa_h` exactly; checked).

**Per-epoch test and CI.** K = 499 surrogates: phi + delta_breath, delta_breath ~ U(0, 2 pi) independently per
breath (breaths from phase wraps, script 52 `breath_ids`); p_e = (1 + #{A_k >= A}) / (K + 1). 95 % CI: breath-
cluster bootstrap (B = 200, resample breaths with replacement, weighted least squares), percentile.
Seeds: `default_rng([20260926, case_index, epoch_index, relation_index, stream])`.

**Patient-level admission.** Needs >= 3 tested epochs. T = mean over the case's tested epochs of A; T_k = mean
over epochs of the k-th surrogate amplitude; p = (1 + #{T_k >= T}) / (K + 1). Holm over the case's admitted
fast relations (F1, F2) at family-wise 0.05. **Pointers**: epochs whose per-epoch p passes BH at q = 0.05
within (case, relation). An edge is instantiated iff Holm passes **and** it has >= 1 pointer. Each pointer
carries: epoch start / end (v2 clock), first / last beats row index, A with its CI and p_e, n beats, n breaths,
and track B's epoch PPV and SPV (F1) or `rsa_h` (F2). The edge carries the patient mean A with an epoch-cluster
bootstrap CI (1,000 replicates) and the population reference from track B. No patient-specific coupling claim
is attached (ledger C3 / track B: persistence is not coupling-specific); the edge states this.

## 4. Null evaluation of fast edges (relation edges of the v2 layer are unchanged and are not re-evaluated)

The whole pipeline of section 3 (its own surrogates, patient test, Holm, BH pointers, fresh seeds) is re-run on
null data for every QC case with >= 3 tested epochs:
- **N1 within-patient circular shift (primary, as requested):** in each epoch the target series (with its
  validity mask) is rolled against the respiratory phase by u beats, u uniform on [ceil(n/4), floor(3n/4)].
  False-edge rate = share of eligible cases with the edge instantiated, per relation and per graph (F1 + F2),
  case-bootstrap 95 % CI (2,000 replicates, seed 0).
  **Expectation stated in advance:** controlled ventilation is nearly periodic, so a circular shift rotates
  the respiratory phase by a nearly constant angle and can leave phase locking intact; if the N1 rate is far
  above 0.05 it will be read as the known limitation of shift nulls for periodic drivers (the shifted target
  is still ventilator-locked), not as the error rate of the rule.
- **N2 per-breath phase randomisation of the data (secondary, implementation check):** phi replaced by
  phi + delta_breath with an independent seed before the pipeline runs. Exchangeable with the test's surrogates,
  so the false-edge rate must be <= 0.05 up to Monte Carlo error; a rate above 0.08 flags a bug.

## 5. Reports

- Per-patient graph size, all 3,442 and the 2,954 QC cases: nodes by family (v1 concepts x tier, beat concepts x
  tier, action, variable), episodes (concept, beat), edges by type (policy / response / association / fast F1 /
  fast F2), pointers; against the wave-4 v2 graphs (same files) and v1 (`graph_size_vitaldb.csv` columns
  `v1_nodes`, `v1_edges`). Means with case-bootstrap CIs.
- Evidence coverage: fast edges with >= 1 pointer; beat episodes with a valid beat index range. Pointer audit on
  100 random QC graphs: every pointer's beat range lies in its epoch, and A is recomputed from the beats file.
- Concept prevalence: share of observed windows on and of cases with >= 1 episode, per beat concept.
- Fast-edge instantiation rates (case-bootstrap CIs) and N1 / N2 false-edge rates.
- Integrity: v2 layer unchanged; F2 epoch A equals track B `rsa_h`; `validate_refs` empty for every graph;
  FastAPI loader returns 200 for 5 graphs; `tests/unit/test_kg_v2.py` passes.
- Two case studies chosen by rule: (i) the QC case with the most `high_ppv` episode time among cases with both
  F1 and F2 instantiated; (ii) the QC case with the most `irregular_rr` episode time. Ties: lower case id.

## 6. Runtime

`PYTHONPATH=src:. python scripts/55_graphs_fast.py --stage all --workers 4`; budget <= 45 min, <= 4 processes.
Nothing is tuned after outputs are seen.
