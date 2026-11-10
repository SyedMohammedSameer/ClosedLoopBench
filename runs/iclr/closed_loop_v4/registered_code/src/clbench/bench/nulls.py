"""
ClosedLoopBench null generators (negative controls). Each keeps the vitals of a case and replaces its
action streams, so any "effect" an estimator reports on a null arm is produced by the method, the
closed-loop structure of the data, or both.

* ``nc_random``: each action's steps (same sizes, same order) moved to uniformly random times within
  the case. Destroys timing and the within-case schedule.
* ``nc_cross`` (K = 8 draws): another patient's action streams, time-aligned by elapsed time from case
  start, one derangement per action among the cases that use it. Keeps the typical schedule of care
  (induction titrations early, fewer changes late) and the per-action event counts, but not the
  coupling to this patient's vitals. Draw k uses ``CROSS_SALTS[k]``; draws 1-2 are v2's nc_cross /
  nc_cross2.

Calibration uses the cross draws (they keep the schedule, which level-based methods pick up); the
random-time null is scored as a second, different null (distinct estimates are reported).
"""

from __future__ import annotations

import numpy as np

from clbench.eval import closed_loop_bench as _clb
from clbench.eval.closed_loop_bench import (CROSS_ARMS, CROSS_SALTS, CaseData, EventSet, nc_cross_patient,
                                              nc_random_times)


def cross_partners(pids: list[str], ev: dict, seed: int = 0, salt: int = 991) -> dict[str, dict[str, str]]:
    """Per action, a derangement among the cases that use that action (keeps per-action event counts).
    ``ev``: case id -> events DataFrame (``loader.load_events``). Returns case -> {agent: partner case}."""
    rng = np.random.default_rng(seed + salt)
    out: dict[str, dict[str, str]] = {p: {} for p in pids}
    for key, spec in _clb.ACTIONS.items():  # looked up at call time: follows ``use_table`` (v3 / v4)
        users = [p for p in pids if p in ev and (ev[p].agent == spec.agent).any()]
        if len(users) < 2:
            continue
        perm = rng.permutation(len(users))
        for i, p in enumerate(users):
            j = perm[i] if perm[i] != i else perm[(i + 1) % len(users)]
            if users[j] == p:
                j = (i + 1) % len(users)
            out[p][spec.agent] = users[j]
    return out


def cross_partner_sets(pids: list[str], ev: dict, seed: int = 0, k: int = len(CROSS_SALTS)) -> dict[str, dict]:
    """cross arm name -> ``cross_partners`` for the first ``k`` registered salts."""
    return {arm: cross_partners(pids, ev, seed, salt) for arm, salt in list(zip(CROSS_ARMS, CROSS_SALTS))[:k]}


def random_time_null(case: CaseData, seed: int) -> CaseData:
    return nc_random_times(case, np.random.default_rng(seed))


def cross_null(case: CaseData, partner_events: dict[str, EventSet]) -> CaseData:
    return nc_cross_patient(case, partner_events)


__all__ = ["CROSS_ARMS", "CROSS_SALTS", "cross_null", "cross_partner_sets", "cross_partners", "random_time_null"]
