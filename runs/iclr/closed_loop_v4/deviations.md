# ClosedLoopBench 1.1 (v4): deviations from the registered protocol

Registered protocol sha256 `418281f18a34c6c9c66111250b0d17d8f9296e8718c469a08515ba386793d10e` (protocol.sha256).
The full run started 2026-09-25 12:25:53 local time, right after registration.

1. **Unit tests added after registration** (`tests/unit/test_closed_loop_bench.py`, 7 v4 tests: table switch and exact
   v3 restore, v3 derangements unchanged under the v4 table, per-equation sufficient statistics, torch vs KD-tree
   neighbours, the R15 pre-level restriction, MAC-step hysteresis, native set-point initialisation). The test file's
   hash therefore differs from the registered one; the tests are not imported by the run. No other registered file
   was modified while the run was in progress.
2. **Analysis scripts written after registration, before any v4 result was seen:** `check_predictions.py`
   (checks Q1-Q6 exactly as worded in protocol §7) and `make_card.py` (BENCHMARK_CARD.txt from the outputs).
3. **After the run** (13:14 local time; 2,874 s total, 0 errors): `src/clbench/bench/__init__.py` docstring extended to
   describe 1.1 (documentation only; its hash now differs from the registered one). `make_card.py` gained a
   post hoc "reading the results" section (5b of the card), labelled as not pre-registered, and
   `summarize_axis.py` (descriptive per-relation tables, `2s_all/axis.md`) and `2s_all/v3_vs_v4_shared18.csv`
   (descriptive comparison with the v3 500-case 2 s run on the 18 shared relations) were added. No estimate,
   rule or prediction was changed.
