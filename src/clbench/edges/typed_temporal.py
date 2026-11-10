"""
Typed, pooled temporal edges between clinical concepts (policy / response / association).

v1 built one temporal edge per concept pair and patient from a lagged cross-correlation or a bivariate
Granger test, with no error control and no notion of *who* drives the relation. Under closed-loop
clinical monitoring three kinds of temporal regularity coexist and need different identification:

* **association**  concept A -> concept B: the discrete-time hazard of a B onset while B is off, given
  A's recent state and onsets in lag bins, adjusted for B's own history, the case phase and the
  recorded actions (doses and recent changes). Descriptive and conditional; not causal.
* **policy**       concept A -> action change (a, direction): the clinician's reaction hazard.
  Identified from observational data as a description of the recorded decision rule.
* **response**     action change -> concept B onset at lags >= the pharmacological latency.
  Confounded by indication in a closed loop (the action was chosen *because* of the state), so its
  sign is checked against pharmacology and reported with that caveat.

Everything lives on one step grid (VitalDB 30 s windows at a 15 s stride). A concept is a tri-state
series ``S`` (1 on, 0 off, -1 unobserved). An *onset* at step i is ``S[i] = 1`` after at least two
observed off steps (``S[i-1] = S[i-2] = 0``); step i is *at risk* for B when ``S_B[i]`` is observed and
``S_B[i-1] = S_B[i-2] = 0``. Rows of all cases are concatenated; ``cs[i]`` is the first row of row i's
case, so every lag stays inside its case.

Estimation. One pooled logistic discrete-time hazard per outcome (target concept, or action
direction), all candidate sources entering jointly. Non-event rows are subsampled with inverse-
probability weights (events are all kept). Uncertainty is a patient-cluster score bootstrap
(Kline & Santos 2012): replicate ``b`` is the one-step Newton update ``H^{-1} sum_c (w_bc - 1) g_c``
from the full-sample fit with multinomial case weights ``w_bc`` (first-order equivalent to the
pairs cluster bootstrap; its covariance is the cluster-robust sandwich). An edge's exposure block
(state and onset lag bins for concept sources; lag bins for action sources) gets a joint Wald test
with the bootstrap covariance, and a scalar summary ``omega`` = the average log hazard ratio of the
source's lag pattern over its *exposed* rows (concept source: rows where A is on at the source lag,
so exp(omega) is the hazard ratio "while A is present", onset lag bins included; action source: rows
with a change in the 1-10 min window).

Patient level. Per-patient edges are deviations from the pooled edge: with the pooled linear
predictor as an offset and a patient intercept, a score test of ``delta_p`` on the edge's exposure
indicator; BY over patient x edge. Empirical-Bayes (DerSimonian-Laird) shrinkage of the one-step
per-patient deviations gives the shrunk patient-level log hazard ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import expit

# ---------------------------------------------------------------------------
# Concepts, groups, lag bins
# ---------------------------------------------------------------------------

CONCEPTS: tuple[str, ...] = (
    "tachycardia", "bradycardia", "hypercapnia", "hypocapnia",
    "hypotension", "hypertension", "tachypnea", "bradypnea",
)
CONCEPT_VITAL: dict[str, str] = {
    "tachycardia": "hr", "bradycardia": "hr", "hypercapnia": "etco2", "hypocapnia": "etco2",
    "hypotension": "map", "hypertension": "map", "tachypnea": "rr", "bradypnea": "rr",
}
# Null "blocks": group-1 concepts (HR, EtCO2) are transplanted / shifted as one block against the
# real group-2 concepts (MAP, RR); cross-group ordered pairs are then null pairs.
GROUP1: tuple[str, ...] = ("tachycardia", "bradycardia", "hypercapnia", "hypocapnia")
GROUP2: tuple[str, ...] = ("hypotension", "hypertension", "tachypnea", "bradypnea")

ACTIONS: tuple[str, ...] = ("propofol", "remifentanil", "phenylephrine", "vent_rr")
DOSES: tuple[str, ...] = ("propofol", "remifentanil", "phenylephrine", "norepinephrine", "vent_rr")
RECORDED: tuple[str, ...] = ("propofol", "remifentanil", "vent_rr")
DIRS: tuple[str, ...] = ("up", "down")

STEP_SEC: float = 15.0
WINDOW_SEC: float = 30.0
# Onset lag bins in steps (15 s). Association sources start at lag 2 (the source window then ends
# where the target window starts: no overlap); policy sources at lag 1 (the action bin i is
# [ws_i + 15, ws_i + 30), after window i-1 ends).
ASSOC_LAG0: int = 2
POLICY_LAG0: int = 1
ONSET_BINS_HI: tuple[tuple[int, int], ...] = ((5, 12), (13, 40))  # 1-3 min, 3-10 min
RESPONSE_BINS: tuple[tuple[int, int], ...] = ((5, 12), (13, 40))  # >= 1 min (latency), to 10 min
PRE_LAG: int = 41  # pre-treatment covariate lag for the response model (10 min + 1 step)


def cross_vital_pairs() -> list[tuple[int, int]]:
    """Ordered (source, target) concept index pairs measured on different vitals (48 pairs)."""
    return [
        (a, b)
        for a in range(len(CONCEPTS))
        for b in range(len(CONCEPTS))
        if CONCEPT_VITAL[CONCEPTS[a]] != CONCEPT_VITAL[CONCEPTS[b]]
    ]


def null_pairs() -> list[tuple[int, int]]:
    """Cross-group ordered pairs (32): null under the group transplant / shift controls."""
    g1 = {CONCEPTS.index(c) for c in GROUP1}
    return [(a, b) for a, b in cross_vital_pairs() if (a in g1) != (b in g1)]


# ---------------------------------------------------------------------------
# Within-case lag primitives on concatenated rows
# ---------------------------------------------------------------------------


def lag_value(x: np.ndarray, cs: np.ndarray, k: int, fill) -> np.ndarray:
    """``x[i-k]`` when row i-k is in the same case, else ``fill``."""
    i = np.arange(len(x))
    j = i - k
    return np.where(j >= cs, x[np.maximum(j, 0)], fill)


def lagged_count(x: np.ndarray, cs: np.ndarray, k0: int, k1: int) -> np.ndarray:
    """Sum of ``x[i-k1 .. i-k0]`` restricted to the case (k0 <= k1)."""
    n = len(x)
    c = np.concatenate([[0], np.cumsum(np.asarray(x, dtype=np.int64))])
    i = np.arange(n)
    hi = i - k0
    lo = np.maximum(i - k1, cs)
    ok = hi >= lo
    return np.where(ok, c[np.clip(hi + 1, 0, n)] - c[np.clip(lo, 0, n)], 0)


def onsets_and_risk(S: np.ndarray, cs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Onset indicator and at-risk indicator of a tri-state concept series."""
    p1 = lag_value(S, cs, 1, -1)
    p2 = lag_value(S, cs, 2, -1)
    risk = (S >= 0) & (p1 == 0) & (p2 == 0)
    return risk & (S == 1), risk


def last_on_steps(S: np.ndarray, cs: np.ndarray, lag: int = 1) -> np.ndarray:
    """Steps since the last row <= i-lag with S == 1 in the same case; -1 if none."""
    n = len(S)
    i = np.arange(n)
    idx = np.where(S == 1, i, -1)
    run = np.maximum.accumulate(idx)
    prev = lag_value(run, cs, lag, -1)
    prev = np.where(prev >= cs, prev, -1)
    return np.where(prev >= 0, i - prev, -1)


def source_block(S: np.ndarray, cs: np.ndarray, lag0: int) -> np.ndarray:
    """(n, 5): [unobserved at lag0, on at lag0, onset in [lag0,4], onset in [5,12], onset in [13,40]]."""
    on_, _ = onsets_and_risk(S, cs)
    s0 = lag_value(S, cs, lag0, -1)
    cols = [s0 < 0, s0 == 1, lagged_count(on_, cs, lag0, 4) > 0]
    cols += [lagged_count(on_, cs, k0, k1) > 0 for k0, k1 in ONSET_BINS_HI]
    return np.column_stack(cols).astype(np.float64)


SOURCE_EXPOSURE_COLS: tuple[int, ...] = (1, 2, 3, 4)  # of source_block (col 0 is an adjustment)
SOURCE_STATE_COL: int = 1  # exposure indicator of a concept source: A on at the source lag
SOURCE_LAG_LABELS: tuple[str, ...] = ("state", "onset_0-1min", "onset_1-3min", "onset_3-10min")


def episodes(S: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of runs of S == 1 in one case."""
    on = np.concatenate([[0], (S == 1).astype(np.int8), [0]])
    d = np.diff(on)
    st = np.nonzero(d == 1)[0]
    en = np.nonzero(d == -1)[0]
    return list(zip(st.tolist(), (en - st).tolist()))


# ---------------------------------------------------------------------------
# Cohort container
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    """Concatenated per-step arrays of all cases (row order: case, then step)."""

    case: np.ndarray  # (n,) case index 0..C-1
    step: np.ndarray  # (n,) step within case
    S: np.ndarray  # (n, 8) int8 tri-state concepts, CONCEPTS order
    dose: np.ndarray  # (n, 5) float, DOSES order, NaN = not recorded
    chg: np.ndarray  # (n, 4, 2) int8 action changes in bin i (ACTIONS x DIRS)
    vital_z: np.ndarray  # (n, 4) standardized hr, map, rr, etco2 (NaN missing)
    anes_start: np.ndarray  # (C,) first step with an anaesthetic / IPPV running (-1 none)
    T: np.ndarray  # (C,) steps per case
    ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.case = np.asarray(self.case, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(self.T)[:-1]]).astype(np.int64)
        self.case_start = starts
        self.cs = starts[self.case]

    @property
    def n(self) -> int:
        return len(self.case)

    @property
    def n_cases(self) -> int:
        return len(self.T)

    def half_mask(self, half: int) -> np.ndarray:
        """Rows in the first (half=1) or second (half=2) half of their case."""
        mid = self.T[self.case] // 2
        return self.step < mid if half == 1 else self.step >= mid


VITAL_COLS: tuple[str, ...] = ("hr", "map", "rr", "etco2")


# ---------------------------------------------------------------------------
# Null row maps (transplant / circular shift) and their application
# ---------------------------------------------------------------------------


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def transplant_map(co: Cohort, rng: np.random.Generator) -> np.ndarray:
    """Row map: row i of case c <- the same elapsed step of a partner case (-1 beyond its end)."""
    partner = derangement(co.n_cases, rng)
    p = partner[co.case]
    ok = co.step < co.T[p]
    return np.where(ok, co.case_start[p] + co.step, -1)


def shift_map(co: Cohort, rng: np.random.Generator, spacing: np.ndarray) -> np.ndarray:
    """Row map: circular shift within case by a random non-zero multiple of ``spacing[c]``
    (the subgroup-lattice offsets of ``clbench.edges.controlled``); T < 2 g falls back to T // 2."""
    off = np.zeros(co.n_cases, dtype=np.int64)
    for c in range(co.n_cases):
        T, g = int(co.T[c]), int(max(1, spacing[c]))
        m = T // g
        off[c] = g * int(rng.integers(1, m)) if m >= 2 else max(1, T // 2)
    j = (co.step + off[co.case]) % co.T[co.case]
    return co.case_start[co.case] + j


def apply_map_S(S: np.ndarray, rowmap: np.ndarray) -> np.ndarray:
    out = S[np.maximum(rowmap, 0)].copy()
    out[rowmap < 0] = -1
    return out


def apply_map_chg(chg: np.ndarray, rowmap: np.ndarray) -> np.ndarray:
    out = chg[np.maximum(rowmap, 0)].copy()
    out[rowmap < 0] = 0
    return out


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------


@dataclass
class Design:
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    case: np.ndarray
    rows: np.ndarray
    names: list[str]
    blocks: dict[str, list[int]]  # edge source name -> exposure column indices
    exposure: dict[str, np.ndarray]  # edge source name -> (n_rows,) bool exposure indicator


def _bins_since(x: np.ndarray, edges: tuple[int, ...]) -> list[np.ndarray]:
    """Dummies for 'never' (x < 0) and (edges[k-1], edges[k]]; the last open bin is the reference."""
    cols = [x < 0]
    lo = 0
    for hi in edges:
        cols.append((x > lo) & (x <= hi))
        lo = hi
    return cols


def _phase_cols(co: Cohort, rows: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    """Case phase: time since anaesthesia start (4 bins), time since recording start (5 bins) and
    time to the end of the recording (3 bins, emergence); the open-ended bins are the references."""
    c = co.case[rows]
    st = co.step[rows]
    a0 = co.anes_start[c]
    el = np.where(a0 >= 0, st - a0, -1)
    cols = [(a0 < 0) | (el < 0), (el >= 0) & (el < 60), (el >= 60) & (el < 240), (el >= 240) & (el < 720)]
    names = ["phase_pre", "phase_0-15m", "phase_15-60m", "phase_1-3h"]
    for lo, hi, nm in ((0, 40, "rec_0-10m"), (40, 120, "rec_10-30m"), (120, 240, "rec_30-60m"),
                       (240, 480, "rec_1-2h"), (480, 960, "rec_2-4h")):
        cols.append((st >= lo) & (st < hi))
        names.append(nm)
    left = co.T[c] - st
    for lo, hi, nm in ((0, 60, "end_<15m"), (60, 120, "end_15-30m"), (120, 240, "end_30-60m")):
        cols.append((left > lo) & (left <= hi))
        names.append(nm)
    return cols, names


def _dose_cols(co: Cohort, rows_lagged: np.ndarray, valid: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    D = co.dose[np.maximum(rows_lagged, 0)]
    D = np.where(valid[:, None], D, np.nan)
    cols, names = [], []
    for j, a in enumerate(DOSES):
        v = np.nan_to_num(D[:, j], nan=0.0)
        cols.append(v / 10.0 if a == "vent_rr" else np.log1p(np.maximum(v, 0.0)))
        names.append(f"dose_{a}")
    for a in RECORDED:
        j = DOSES.index(a)
        cols.append(np.isfinite(D[:, j]))
        names.append(f"rec_{a}")
    return cols, names


def subsample_rows(y: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Keep all events and a ``frac`` Bernoulli sample of non-events (weight 1/frac)."""
    keep = y | (rng.random(len(y)) < frac)
    w = np.where(y, 1.0, 1.0 / frac)
    return keep, w


def _finish(cols, names, blocks_named, y, w, co, rows, expo_named=None) -> Design:
    """``blocks_named``: source -> columns of its joint Wald test; ``expo_named``: source -> columns
    whose non-zero rows define the exposure indicator (default: the Wald columns)."""
    X = np.column_stack([np.ones(len(rows))] + [np.asarray(c, dtype=np.float64) for c in cols])
    names = ["intercept"] + names
    blocks, exposure = {}, {}
    expo_named = expo_named or {}
    for src, idx in blocks_named.items():
        ii = [i + 1 for i in idx]
        blocks[src] = ii
        ee = [i + 1 for i in expo_named.get(src, idx)]
        exposure[src] = np.any(X[:, ee] != 0, axis=1)
    return Design(X, y.astype(np.float64), w, co.case[rows], rows, names, blocks, exposure)


def association_design(
    co: Cohort, b: int, rows: np.ndarray, S_src: np.ndarray | None = None
) -> Design:
    """Hazard of concept ``b`` onset on at-risk ``rows``; sources = cross-vital concepts (from
    ``S_src`` when given, e.g. a null-mapped copy) with state / onset lag bins from lag 2."""
    S = co.S
    Ssrc = S if S_src is None else S_src
    cs = co.cs
    onset_b, _ = onsets_and_risk(S[:, b], cs)
    y = onset_b[rows]
    cols, names = [], []
    tl = last_on_steps(S[:, b], cs, 1)[rows]
    for k, cnm in zip(_bins_since(tl, (8, 40, 120)), ("never", "<=2m", "2-10m", "10-30m")):
        cols.append(k)
        names.append(f"b_since_off_{cnm}")
    pc, pn = _phase_cols(co, rows)
    cols += pc
    names += pn
    lagged = rows - 1
    dc, dn = _dose_cols(co, lagged, lagged >= cs[rows])
    cols += dc
    names += dn
    for ai, a in enumerate(ACTIONS):
        for di, d in enumerate(DIRS):
            cnt = lagged_count(co.chg[:, ai, di], cs, 1, 40)[rows]
            cols.append(cnt > 0)
            names.append(f"recent_{a}_{d}")
    blocks, expo = {}, {}
    vb = CONCEPT_VITAL[CONCEPTS[b]]
    for a in range(len(CONCEPTS)):
        if CONCEPT_VITAL[CONCEPTS[a]] == vb:
            continue
        blk = source_block(Ssrc[:, a], cs, ASSOC_LAG0)[rows]
        base = len(cols)
        for j, lab in enumerate(("unobs",) + SOURCE_LAG_LABELS):
            cols.append(blk[:, j])
            names.append(f"{CONCEPTS[a]}:{lab}")
        blocks[CONCEPTS[a]] = [base + j for j in SOURCE_EXPOSURE_COLS]
        expo[CONCEPTS[a]] = [base + SOURCE_STATE_COL]
    return _finish(cols, names, blocks, y, np.ones(len(rows)), co, rows, expo)


def association_rows(co: Cohort, b: int) -> np.ndarray:
    _, risk = onsets_and_risk(co.S[:, b], co.cs)
    return np.nonzero(risk)[0]


def policy_rows(co: Cohort, ai: int, di: int) -> np.ndarray:
    """At risk for a change of action ``ai`` in direction ``di`` at bin i (pre-i information only)."""
    a = ACTIONS[ai]
    cs = co.cs
    prev = lag_value(np.arange(co.n), cs, 1, -1)
    ok = prev >= 0
    if a == "phenylephrine" and DIRS[di] == "up":
        return np.nonzero(ok)[0]
    dj = DOSES.index(a)
    d1 = co.dose[np.maximum(prev, 0), dj]
    return np.nonzero(ok & np.isfinite(d1) & (d1 > 0))[0]


def policy_design(
    co: Cohort, ai: int, di: int, rows: np.ndarray, S_src: np.ndarray | None = None
) -> Design:
    Ssrc = co.S if S_src is None else S_src
    cs = co.cs
    y = co.chg[rows, ai, di] > 0
    cols, names = [], []
    any_chg = (co.chg[:, ai, 0] + co.chg[:, ai, 1]) > 0
    tl = last_on_steps(any_chg.astype(np.int8), cs, 1)[rows]
    for k, cnm in zip(_bins_since(tl, (4, 20, 60)), ("never", "<=1m", "1-5m", "5-15m")):
        cols.append(k)
        names.append(f"since_change_{cnm}")
    pc, pn = _phase_cols(co, rows)
    cols += pc
    names += pn
    lagged = rows - 1
    dc, dn = _dose_cols(co, lagged, lagged >= cs[rows])
    cols += dc
    names += dn
    blocks, expo = {}, {}
    for a in range(len(CONCEPTS)):
        blk = source_block(Ssrc[:, a], cs, POLICY_LAG0)[rows]
        base = len(cols)
        for j, lab in enumerate(("unobs",) + SOURCE_LAG_LABELS):
            cols.append(blk[:, j])
            names.append(f"{CONCEPTS[a]}:{lab}")
        blocks[CONCEPTS[a]] = [base + j for j in SOURCE_EXPOSURE_COLS]
        expo[CONCEPTS[a]] = [base + SOURCE_STATE_COL]
    return _finish(cols, names, blocks, y, np.ones(len(rows)), co, rows, expo)


def response_design(
    co: Cohort, b: int, rows: np.ndarray, chg_src: np.ndarray | None = None
) -> Design:
    """Hazard of concept ``b`` onset given action changes (per action x direction) in the lag bins
    [5,12] and [13,40] steps (>= 1 min latency), adjusted for pre-treatment (lag-41) state."""
    S = co.S
    chg = co.chg if chg_src is None else chg_src
    cs = co.cs
    onset_b, _ = onsets_and_risk(S[:, b], cs)
    y = onset_b[rows]
    cols, names = [], []
    tl = last_on_steps(S[:, b], cs, 1)[rows]
    for k, cnm in zip(_bins_since(tl, (8, 40, 120)), ("never", "<=2m", "2-10m", "10-30m")):
        cols.append(k)
        names.append(f"b_since_off_{cnm}")
    pc, pn = _phase_cols(co, rows)
    cols += pc
    names += pn
    pre = rows - PRE_LAG
    pre_ok = pre >= cs[rows]
    dc, dn = _dose_cols(co, pre, pre_ok)
    cols += dc
    names += [f"pre_{x}" for x in dn]
    Spre = np.where(pre_ok[:, None], S[np.maximum(pre, 0)], -1)
    for a in range(len(CONCEPTS)):
        cols.append(Spre[:, a] == 1)
        names.append(f"pre_{CONCEPTS[a]}_on")
        cols.append(Spre[:, a] < 0)
        names.append(f"pre_{CONCEPTS[a]}_unobs")
    vj = VITAL_COLS.index(CONCEPT_VITAL[CONCEPTS[b]])
    vpre = np.where(pre_ok, co.vital_z[np.maximum(pre, 0), vj], np.nan)
    cols.append(np.nan_to_num(vpre, nan=0.0))
    names.append("pre_target_vital_z")
    cols.append(~np.isfinite(vpre))
    names.append("pre_target_vital_missing")
    blocks = {}
    for ai, a in enumerate(ACTIONS):
        for di, d in enumerate(DIRS):
            base = len(cols)
            for k0, k1 in RESPONSE_BINS:
                cols.append(lagged_count(chg[:, ai, di], cs, k0, k1)[rows] > 0)
                names.append(f"{a}_{d}:lag_{k0}-{k1}")
            blocks[f"{a}_{d}"] = [base, base + 1]
    return _finish(cols, names, blocks, y, np.ones(len(rows)), co, rows)


# ---------------------------------------------------------------------------
# Weighted logistic regression, cluster score bootstrap, Wald tests
# ---------------------------------------------------------------------------


@dataclass
class Fit:
    beta: np.ndarray
    H: np.ndarray  # penalized observed information (sum scale)
    G: np.ndarray  # (n_cases, p) per-case score contributions at beta
    keep: np.ndarray  # columns kept (non-constant)
    converged: bool
    n_iter: int


def fit_logistic(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, case: np.ndarray, n_cases: int,
    ridge: float = 1.0, max_iter: int = 30, tol: float = 1e-7,
) -> Fit:
    """Weighted logistic regression by Newton-Raphson with a small ridge on non-intercept terms.
    Columns that are constant zero on the rows are dropped (their coefficient is reported as 0)."""
    p_all = X.shape[1]
    keep = np.ones(p_all, dtype=bool)
    keep[1:] = np.any(X[:, 1:] != 0, axis=0)
    Xk = X[:, keep]
    p = Xk.shape[1]
    pen = np.full(p, ridge)
    pen[0] = 0.0
    beta = np.zeros(p)
    ybar = float(np.sum(w * y) / np.sum(w))
    beta[0] = np.log(max(ybar, 1e-8) / max(1 - ybar, 1e-8))
    conv = False
    it = 0
    for it in range(1, max_iter + 1):
        mu = expit(Xk @ beta)
        g = Xk.T @ (w * (y - mu)) - pen * beta
        H = (Xk * (w * mu * (1 - mu))[:, None]).T @ Xk + np.diag(pen)
        step = np.linalg.solve(H, g)
        # damp very large steps (quasi-separation)
        m = np.max(np.abs(step))
        if m > 5:
            step *= 5 / m
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            conv = True
            break
    mu = expit(Xk @ beta)
    H = (Xk * (w * mu * (1 - mu))[:, None]).T @ Xk + np.diag(pen)
    r = w * (y - mu)
    G = np.zeros((n_cases, p))
    for j in range(p):
        G[:, j] = np.bincount(case, weights=r * Xk[:, j], minlength=n_cases)
    G -= (pen * beta)[None, :] / n_cases
    full = np.zeros(p_all)
    full[keep] = beta
    return Fit(full, H, G, keep, conv, it)


def fit_logistic_fe(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, case: np.ndarray, n_cases: int,
    ridge: float = 1.0, max_iter: int = 40, tol: float = 1e-7,
) -> Fit:
    """Patient fixed-effects (within-patient) weighted logistic regression.

    Column 0 of ``X`` (the intercept) is replaced by one intercept per case; cases without both an
    event and a non-event carry no within-case information and are dropped. Newton-Raphson with the
    case intercepts profiled out through the Schur complement; the returned ``H`` is the profiled
    information for the shared coefficients and ``G`` holds per-case efficient scores (covariates
    centred on their information-weighted case means), so ``score_bootstrap`` resamples patients.
    ``beta[0]`` is reported as 0 (no common intercept)."""
    ev = np.bincount(case, weights=y, minlength=n_cases)
    nn = np.bincount(case, weights=1 - y, minlength=n_cases)
    use = (ev[case] > 0) & (nn[case] > 0)
    Xs, ys, ws, cs = X[use, 1:], y[use], w[use], case[use]
    p_all = X.shape[1]
    keep = np.zeros(p_all, dtype=bool)
    keep[1:] = np.any(Xs != 0, axis=0) & (np.ptp(Xs, axis=0) > 0)
    Xk = Xs[:, keep[1:]]
    p = Xk.shape[1]
    beta = np.zeros(p)
    alpha = np.zeros(n_cases)
    ybar = np.bincount(cs, weights=ws * ys, minlength=n_cases) / np.maximum(
        np.bincount(cs, weights=ws, minlength=n_cases), 1e-300)
    alpha[:] = np.log(np.clip(ybar, 1e-6, 1 - 1e-6) / (1 - np.clip(ybar, 1e-6, 1 - 1e-6)))
    conv, it = False, 0

    def pieces(beta, alpha):
        mu = expit(Xk @ beta + alpha[cs])
        v = ws * mu * (1 - mu)
        r = ws * (ys - mu)
        Haa = np.bincount(cs, weights=v, minlength=n_cases) + 1e-8
        Hba = np.stack([np.bincount(cs, weights=v * Xk[:, j], minlength=n_cases) for j in range(p)])
        Hbb = (Xk * v[:, None]).T @ Xk + ridge * np.eye(p)
        gb = Xk.T @ r - ridge * beta
        ga = np.bincount(cs, weights=r, minlength=n_cases)
        return mu, v, r, Haa, Hba, Hbb, gb, ga

    for it in range(1, max_iter + 1):
        mu, v, r, Haa, Hba, Hbb, gb, ga = pieces(beta, alpha)
        A = Hbb - (Hba / Haa[None, :]) @ Hba.T
        rhs = gb - Hba @ (ga / Haa)
        db = np.linalg.solve(A, rhs)
        m = np.max(np.abs(db)) if p else 0.0
        if m > 5:
            db *= 5 / m
        da = (ga - Hba.T @ db) / Haa
        beta += db
        alpha += np.clip(da, -5, 5)
        if m < tol and np.max(np.abs(da)) < 1e-5:
            conv = True
            break
    mu, v, r, Haa, Hba, Hbb, gb, ga = pieces(beta, alpha)
    A = Hbb - (Hba / Haa[None, :]) @ Hba.T
    xbar = (Hba / Haa[None, :]).T  # (n_cases, p)
    Xc = Xk - xbar[cs]
    G = np.zeros((n_cases, p))
    for j in range(p):
        G[:, j] = np.bincount(cs, weights=r * Xc[:, j], minlength=n_cases)
    G -= (ridge * beta)[None, :] / n_cases
    full = np.zeros(p_all)
    full[keep] = beta
    return Fit(full, A, G, keep, conv, it)


def patient_delta_mle(eta: np.ndarray, y: np.ndarray, E: np.ndarray, case: np.ndarray, n_cases: int,
                      prior_sd: float = 3.0, n_iter: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Per-case penalized MLE of (alpha_c, delta_c) in logit = eta + alpha_c + delta_c E with a
    N(0, prior_sd^2) penalty on delta_c (weak ridge 0.1 on alpha_c); returns (delta, se)."""
    Ef = E.astype(np.float64)
    a = np.zeros(n_cases)
    d = np.zeros(n_cases)
    lam = 1.0 / prior_sd**2
    for _ in range(n_iter):
        mu = expit(eta + a[case] + d[case] * Ef)
        v = mu * (1 - mu)
        ga = np.bincount(case, weights=y - mu, minlength=n_cases) - 0.1 * a
        gd = np.bincount(case, weights=(y - mu) * Ef, minlength=n_cases) - lam * d
        haa = np.bincount(case, weights=v, minlength=n_cases) + 0.1
        had = np.bincount(case, weights=v * Ef, minlength=n_cases)
        hdd = np.bincount(case, weights=v * Ef * Ef, minlength=n_cases) + lam
        det = haa * hdd - had**2
        sa = (hdd * ga - had * gd) / det
        sd = (haa * gd - had * ga) / det
        a += np.clip(sa, -3, 3)
        d += np.clip(sd, -3, 3)
    mu = expit(eta + a[case] + d[case] * Ef)
    v = mu * (1 - mu)
    haa = np.bincount(case, weights=v, minlength=n_cases) + 0.1
    had = np.bincount(case, weights=v * Ef, minlength=n_cases)
    hdd = np.bincount(case, weights=v * Ef * Ef, minlength=n_cases) + lam
    se = np.sqrt(haa / (haa * hdd - had**2))
    return d, se


def bootstrap_weights(n_cases: int, n_boot: int, seed: int) -> np.ndarray:
    """(B, n_cases) multinomial patient-resampling multiplicities."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_cases, np.full(n_cases, 1.0 / n_cases), size=n_boot).astype(np.float64)


def score_bootstrap(fit: Fit, W: np.ndarray) -> np.ndarray:
    """(B, p_all) bootstrap draws of beta (zeros for dropped columns)."""
    D = (W - 1.0) @ fit.G  # (B, p_kept)
    Dk = np.linalg.solve(fit.H, D.T).T
    out = np.zeros((W.shape[0], len(fit.keep)))
    out[:, fit.keep] = fit.beta[fit.keep][None, :] + Dk
    return out


@dataclass
class EdgeStat:
    source: str
    chi2: float
    df: int
    p: float
    omega: float  # average exposure log hazard ratio
    omega_lo: float
    omega_hi: float
    coefs: list[float]
    coef_lo: list[float]
    coef_hi: list[float]
    n_exposed_rows: float
    n_exposed_events: float
    n_event_patients: int = 0  # patients contributing an exposed event
    se_omega: float = float("nan")
    z_omega: float = 0.0
    p_omega: float = 1.0  # two-sided, bootstrap SE; 1.0 when below the support minimum
    testable: bool = False


MIN_EXPOSED_EVENTS: int = 20
MIN_EVENT_PATIENTS: int = 10


def edge_stats(des: Design, fit: Fit, draws: np.ndarray, level: float = 0.95) -> list[EdgeStat]:
    """Per source: joint Wald over the exposure block (secondary) and the primary 1-df test of
    omega with its patient-bootstrap SE; edges with fewer than MIN_EXPOSED_EVENTS exposed events or
    MIN_EVENT_PATIENTS contributing patients are not testable (p_omega = 1)."""
    out = []
    qlo, qhi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    for src, idx in des.blocks.items():
        idx_k = [i for i in idx if fit.keep[i]]
        E = des.exposure[src]
        nexp = float(np.sum(des.w[E]))
        nev = float(np.sum(des.y[E]))
        if not idx_k or E.sum() == 0:
            out.append(EdgeStat(src, 0.0, 0, 1.0, 0.0, 0.0, 0.0, [0.0] * len(idx),
                                [0.0] * len(idx), [0.0] * len(idx), nexp, nev))
            continue
        b = fit.beta[idx_k]
        Db = draws[:, idx_k] - fit.beta[idx_k][None, :]
        V = np.atleast_2d(np.cov(Db, rowvar=False))
        Vi = np.linalg.pinv(V)
        rank = int(np.linalg.matrix_rank(V))
        chi2 = float(b @ Vi @ b)
        pval = float(stats.chi2.sf(chi2, max(rank, 1)))
        wv = np.average(des.X[E][:, idx], axis=0, weights=des.w[E])
        om = float(wv @ fit.beta[idx])
        om_d = draws[:, idx] @ wv
        cl = np.percentile(draws[:, idx], qlo, axis=0)
        ch = np.percentile(draws[:, idx], qhi, axis=0)
        npat = int(len(np.unique(des.case[E & (des.y > 0)])))
        se = float(np.std(om_d, ddof=1))
        ok = nev >= MIN_EXPOSED_EVENTS and npat >= MIN_EVENT_PATIENTS and se > 0
        z = om / se if ok else 0.0
        out.append(EdgeStat(
            src, chi2, rank, pval, om, float(np.percentile(om_d, qlo)), float(np.percentile(om_d, qhi)),
            fit.beta[idx].tolist(), cl.tolist(), ch.tolist(), nexp, nev, npat, se, float(z),
            float(2 * stats.norm.sf(abs(z))) if ok else 1.0, bool(ok),
        ))
    return out


# ---------------------------------------------------------------------------
# Patient-level deviations and empirical-Bayes shrinkage
# ---------------------------------------------------------------------------


def patient_intercepts(eta: np.ndarray, y: np.ndarray, case: np.ndarray, n_cases: int,
                       n_iter: int = 25) -> np.ndarray:
    """Per-case intercept shift alpha_c solving sum_i (y_i - expit(eta_i + alpha_c)) = 0 (ridge 0.1).
    Cases without events get the ridge-limited solution (information near 0)."""
    alpha = np.zeros(n_cases)
    for _ in range(n_iter):
        mu = expit(eta + alpha[case])
        g = np.bincount(case, weights=y - mu, minlength=n_cases) - 0.1 * alpha
        h = np.bincount(case, weights=mu * (1 - mu), minlength=n_cases) + 0.1
        alpha += np.clip(g / h, -3, 3)
    return alpha


@dataclass
class Deviation:
    """Per-case score tests of a deviation delta_c on an edge's exposure indicator."""

    U: np.ndarray  # (n_cases,)
    I: np.ndarray  # (n_cases,) efficient information (intercept profiled out)
    n_exp_events: np.ndarray  # (n_cases,)
    delta: np.ndarray | None = None  # penalized per-case MLE (patient_delta_mle)
    se: np.ndarray | None = None

    @property
    def z(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.I > 1e-9, self.U / np.sqrt(np.maximum(self.I, 1e-300)), np.nan)

    @property
    def p(self) -> np.ndarray:
        return 2 * stats.norm.sf(np.abs(self.z))

    def delta_hat(self, cap: float = 5.0) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(self.I > 1e-9, self.U / np.maximum(self.I, 1e-300), 0.0)
        return np.clip(d, -cap, cap)


def deviation_scores(mu: np.ndarray, y: np.ndarray, E: np.ndarray, case: np.ndarray,
                     n_cases: int) -> Deviation:
    v = mu * (1 - mu)
    Ef = E.astype(np.float64)
    U = np.bincount(case, weights=(y - mu) * Ef, minlength=n_cases)
    s_v = np.bincount(case, weights=v, minlength=n_cases)
    s_ve = np.bincount(case, weights=v * Ef, minlength=n_cases)
    s_ve2 = np.bincount(case, weights=v * Ef * Ef, minlength=n_cases)
    with np.errstate(invalid="ignore", divide="ignore"):
        I = s_ve2 - np.where(s_v > 0, s_ve**2 / np.maximum(s_v, 1e-300), 0.0)
    nev = np.bincount(case, weights=y * Ef, minlength=n_cases)
    return Deviation(U, np.maximum(I, 0.0), nev)


def eb_shrink(d: np.ndarray, se2: np.ndarray) -> tuple[np.ndarray, float]:
    """DerSimonian-Laird tau^2 around 0 (deviations from the pooled effect); shrunk deviations."""
    ok = np.isfinite(d) & np.isfinite(se2) & (se2 > 0) & (se2 < 1e6)
    if ok.sum() < 3:
        return np.zeros_like(d), 0.0
    wv = 1.0 / se2[ok]
    dm = np.sum(wv * d[ok]) / np.sum(wv)
    Q = np.sum(wv * (d[ok] - dm) ** 2)
    k = ok.sum()
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(wv) - np.sum(wv**2) / np.sum(wv)))
    shr = np.zeros_like(d)
    shr[ok] = tau2 / (tau2 + se2[ok]) * d[ok]
    return shr, float(tau2)


# ---------------------------------------------------------------------------
# Multiple testing and replication helpers
# ---------------------------------------------------------------------------


def bh(p: np.ndarray, q: float) -> np.ndarray:
    from clbench.edges.controlled import benjamini_hochberg

    return benjamini_hochberg(np.asarray(p, dtype=np.float64), q)


def by(p: np.ndarray, q: float) -> np.ndarray:
    from clbench.edges.controlled import benjamini_yekutieli

    return benjamini_yekutieli(np.asarray(p, dtype=np.float64), q)


def replication_with_chance(E1: list[set], E2: list[set], family_sizes: list[int]) -> dict:
    """Pooled within-patient replication |E1 & E2| / |E1| and the density-matched random-edge
    chance level sum_p |E1_p| |E2_p| / m_p / sum_p |E1_p| (m_p = candidate pairs of patient p)."""
    n1 = sum(len(a) for a in E1)
    rep = sum(len(a & b) for a, b in zip(E1, E2))
    chance = sum(len(a) * len(b) / m for a, b, m in zip(E1, E2, family_sizes) if m > 0)
    return {
        "n_edges_h1": int(n1),
        "n_replicated": int(rep),
        "replication": rep / n1 if n1 else float("nan"),
        "chance": chance / n1 if n1 else float("nan"),
        "n_patients_with_h1_edges": int(sum(1 for a in E1 if a)),
    }


def patient_bootstrap_ratio(num: np.ndarray, den: np.ndarray, W: np.ndarray) -> tuple[float, float, float]:
    """Ratio sum(num)/sum(den) with a patient-bootstrap 95% percentile CI (W: (B, n) multiplicities)."""
    est = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = (W @ num) / (W @ den)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
