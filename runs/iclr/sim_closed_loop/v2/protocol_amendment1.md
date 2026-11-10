# Protocol amendment 1: phase-grid re-run with the corrected control pool

Registered 2026-09-24 17:17 local time, after the confirmatory `lemma` and `headline2` runs of `protocol.md` had
started and before this re-run. No phase-grid output with the v2 estimators had been computed.

Code: `scripts/46_sim_closed_loop.py` sha256 96dfa3d8108ded4ffad78814d179a03519683638ec672dfb7b13107f360b040d
(`uy_rows` now uses the eligible ITT pool, estimator-specific truths, and optionally the v1 pool);
`src/clbench/sim/clinician_loop.py` unchanged from `protocol.md`.

Run: `--exp phase --fresh --v1-pool --out-dir runs/iclr/sim_closed_loop/v2 --n-patients 60 --seed 0`, then
`--exp report --out-dir runs/iclr/sim_closed_loop/v2`. Same grid, seeds and cohorts as v1
(gain {0, 0.5, 1, 2, 4} × anticipation {0, 60, 180} s × τ_d {20, 60, 200, 600, 2000} s; seed 0 + 100 + cell index),
so v1 and v2 estimators are compared on identical simulated data.

Purely descriptive (no new hypotheses): report, for the closed-loop cells, the share of cells correct / wrong /
correct-and-within-2× for every u→y estimator with the eligible pool and with the v1 pool
(`*_v1pool`), and the y→y table (unchanged estimators, recomputed on identical cohorts as a reproducibility check).
