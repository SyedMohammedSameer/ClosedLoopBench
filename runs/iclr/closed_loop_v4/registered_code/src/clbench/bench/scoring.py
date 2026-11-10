"""
ClosedLoopBench scoring API.

A method is scored from its pooled estimates, one per (arm, relation) at the relation's primary
horizon, each with case-bootstrap replicates computed with the benchmark's shared weight matrix
(``bootstrap_weights``: entry W[i, b] is the multiplicity of case i in replicate b). Because every arm
uses the same W, replicates of the real arm and of the null arms are paired by patient.

    from clbench.bench import scoring
    W = scoring.bootstrap_weights(n_cases)                       # (n_cases, 200), seed 0
    sub = scoring.Submission("my_method", scale="30s")
    for arm in scoring.ARMS:                                       # real, nc_random, nc_cross ... nc_cross8
        for rid in scoring.RELATION_IDS:
            est, reps = scoring.event_estimate(case_sums, case_counts, W)   # or your own replicates
            sub.add(arm, rid, est, reps, n_cases=int((case_counts > 0).sum()))
    calibrated, line = scoring.score(sub)
    print(line["text"])

Rule (identical for every method; ``clbench.edges.typed_temporal_pooled.admit_multi``): the
calibrated estimate is real minus the mean of the K = 8 cross-patient null estimates; SE = paired
bootstrap SE plus the between-draw SE; an edge is admitted iff the CI excludes 0, the
Benjamini-Yekutieli adjusted p over the method's relations is <= 0.05, the edge is stable (single-draw
calibrated CI excludes 0 with the same sign in >= 75% of the draws) and >= 5 cases contribute.

Leaderboard metric: ``score`` = correct-sign admissions - wrong-sign admissions on real data, reported
with the null false-admission rates of the same rule (each cross draw as input against the other seven;
the random-time null as input against all eight).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from clbench.eval import closed_loop_bench as clb

ARMS: tuple[str, ...] = clb.ARMS_V3
RELATION_IDS: tuple[str, ...] = tuple(r.rid for r in clb.RELATIONS)
N_BOOT: int = 200


def bootstrap_weights(n_cases: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """The benchmark's case-bootstrap weight matrix (cases in sorted case-id order)."""
    return clb.case_bootstrap_weights(n_cases, n_boot, seed)


def event_estimate(case_sums: np.ndarray, case_counts: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray]:
    """Event-weighted mean (sum of per-case sums / sum of per-case counts) and its paired replicates."""
    return clb.ratio_boot(np.asarray(case_sums, float), np.asarray(case_counts, float), W)


@dataclass
class Submission:
    method: str
    scale: str = "30s"
    store: dict = field(default_factory=dict)

    @property
    def cfg(self) -> clb.ScaleConfig:
        return clb.SCALES[self.scale]

    def add(self, arm: str, rid: str, est: float, reps: np.ndarray, n_cases: int) -> None:
        r = clb.REL_BY_ID[rid]
        h = clb.primary_h(r, self.method, self.cfg)
        self.store[(arm, self.method, rid, h)] = (float(est), np.asarray(reps, float), int(n_cases))


def _pooled_frame(sub: Submission):
    import pandas as pd

    rows = []
    for (arm, m, rid, h), (est, reps, n) in sub.store.items():
        r = clb.REL_BY_ID[rid]
        lo, hi = clb.ci(reps)
        rows.append(dict(arm=arm, method=m, rid=rid, h=h, est=est, lo=lo, hi=hi,
                         call=clb.sign_call(est, lo, hi, r.expected, n), n_cases=n))
    return pd.DataFrame(rows)


def score(sub: Submission, q: float = 0.05) -> tuple[object, dict]:
    """(calibrated admission table, leaderboard row with a one-line ``text``)."""
    calib = clb.calibrated_admission_table(sub.store, sub.cfg, q=q)
    if calib is not None:
        calib = calib[calib.method == sub.method]
    nulls = clb.null_discovery_table(_pooled_frame(sub), sub.cfg, (sub.method,))
    row = clb.leaderboard(calib, nulls, (sub.method,)).iloc[0].to_dict()
    row["text"] = leaderboard_line(row)
    return calib, row


def leaderboard_line(row: dict) -> str:
    def g(k, fmt="{}"):
        v = row.get(k, np.nan)
        return "na" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)

    return (f"{row['method']}: score {g('score')} ({g('cal_correct')} correct - {g('cal_wrong')} wrong of "
            f"{g('n_relations')}, calibrated vs {g('null_cross_draws')} cross draws); null false admissions "
            f"{g('null_cross_fa_per_draw', '{:.2f}')} per cross draw (LOO; {g('null_cross_draws_any')} of "
            f"{g('null_cross_draws')} draws with any), {g('null_random_fa')} random-time; uncalibrated null "
            f"discoveries {g('uncal_null_cross_sig_mean', '{:.1f}')} per cross draw, {g('uncal_null_random_sig')} "
            f"random-time")


__all__ = ["ARMS", "N_BOOT", "RELATION_IDS", "Submission", "bootstrap_weights", "event_estimate",
           "leaderboard_line", "score"]
