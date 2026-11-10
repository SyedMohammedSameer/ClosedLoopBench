"""
ClosedLoopBench 1.0: a real-data temporal causal-discovery benchmark with a clinician in the loop.

VitalDB perioperative monitoring (3,442 cases at 30 s; 500 at 2 s): vitals (MAP, HR, EtCO2) and the
clinician's actions (propofol / remifentanil TCI targets, ventilator rate, vasopressor and nitroglycerin
infusion rates). Anaesthesia is a closed loop, so a temporal edge can be the drug's *response* or the
clinician's *policy*; the benchmark scores whether a method recovers 12 response and 6 policy relations
with the sign known from pharmacology, physiology or practice, and how often it "discovers" relations in
negative-control arms where none exist.

Modules
-------
``ground_truth``  the 18 relations, their primary horizons and the sources for each sign
``loader``        VitalDB processed numerics + events -> ``CaseData`` (dose masking after a track stops)
``nulls``         random-time null and K = 8 cross-patient derangements (registered salts)
``scoring``       submission format (pooled estimate + paired case-bootstrap replicates per arm and
                  relation), the calibrated admission rule and the one-line leaderboard metric
``sim``           clinician-in-the-loop simulator adapter with exact ITT truth (validation)
``native_grid``   v4: 2 s grid + one-clock action streams from the native 1 s numerics (all cases)

ClosedLoopBench 1.1 (v4; ``closed_loop_bench.use_table("v4")``): 29 relations (+ tidal volume, FiO2, PEEP,
derived MAC steps; SpO2 and BIS as vitals; a physics / drug / policy domain per relation) on a 2 s native grid for
all 3,442 cases (``native_grid``); card ``runs/iclr/closed_loop_v4/BENCHMARK_CARD.txt``. ``use_table("v3")`` (the
import default) keeps 1.0 byte-reproducible (runs/iclr/closed_loop_v4/v3_reproducibility.md).

Reference implementations of eleven methods (naive event pre/post, v1 cross-correlation rules, Granger,
PCMCI+, VAR-LiNGAM, VARX, within-patient matched and pre-trend designs, the pooled sequential-trial
contrast, and the CLBench typed estimator) live in ``clbench.eval.closed_loop_bench``; the runner is
``scripts/44_closed_loop_bench.py`` and the benchmark card is ``runs/iclr/closed_loop_v3/BENCHMARK_CARD.txt``.
"""

from clbench.bench.ground_truth import GROUND_TRUTH_VERSION, RELATIONS, ground_truth_table

BENCH_NAME = "ClosedLoopBench"
BENCH_VERSION = "1.0"

__all__ = ["BENCH_NAME", "BENCH_VERSION", "GROUND_TRUTH_VERSION", "RELATIONS", "ground_truth_table"]
