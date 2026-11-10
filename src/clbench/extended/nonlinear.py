"""
Extended analysis D: nonlinear robustness of the ClosedLoopBench conclusions.

Everything here re-uses ``clbench.eval.closed_loop_bench`` (the v4 cohort, arms, bootstrap weights and the
calibrated admission rule) and only adds estimators:

    D1a  pcmciplus_gpdc    PCMCI+ with the GPDC conditional-independence test (GP regression + distance corr.)
    D1b  pcmciplus_cmiknn  PCMCI+ with the CMIknn test (kNN conditional mutual information, shuffle test)
    D1c  granger_gbm       nonlinear multivariate Granger: HistGradientBoosting predictive gain + average
                           marginal effect (pooled across cases, case-grouped folds)
    D1d  xcorr_rank_*      Spearman (rank) lagged cross-correlation in place of Pearson (3 peak rules)
    D2   varx_dosebins     the typed pooled VARX with the dose exposure entered through dose-size bins
    D3   PAT-SBP           per-epoch Spearman and isotonic (monotone) coupling in place of the OLS slope

Cohort, arms and seeds are those of ``scripts/44_closed_loop_bench.py --bench-version v4`` (``run_v4``): the
same 3,442 cases, the same per-case ``nc_random`` seed, the same 8 cross-patient derangements (salts
``CROSS_SALTS``), the same 60-case PCMCI+ subsample and the same 200 case-bootstrap weights, so every new
estimator is scored by ``closed_loop_bench.calibrated_admission_table`` exactly as the paper's 12.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from clbench.bench import loader as bl
from clbench.bench import nulls as bn
from clbench.eval import closed_loop_bench as clb

ROOT = Path(__file__).resolve().parents[3]
GRID_V4 = ROOT / "data" / "processed_vitaldb" / "native_2s_v4"
PAPER_RUN = ROOT / "runs" / "iclr" / "closed_loop_v4" / "2s_all"
SCALE = "2s_v4"


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 0. Optional-dependency shims (dcor, numba are not installed; tigramite's GPDC / CMIknn import them)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def install_shims() -> dict[str, str]:
    """Register pure-numpy stand-ins for ``dcor`` and ``numba`` when the real packages are absent (see the
    shim modules). Returns {package: 'real' | 'shim'}."""
    out = {}
    for name, shim in (("dcor", "clbench.extended._dcor_shim"), ("numba", "clbench.extended._numba_shim")):
        try:
            __import__(name)
            out[name] = "real"
        except ImportError:
            import importlib

            sys.modules[name] = importlib.import_module(shim)
            out[name] = "shim"
    # tigramite 5.2.10 calls np.corrcoef(..., ddof=0) in the block-shuffle path (CMIknn's unconditional
    # tests); the keyword was removed in numpy 2. Same numbers (ddof never affected corrcoef).
    try:
        from tigramite.independence_tests import independence_tests_base as itb

        if not getattr(itb.CondIndTest, "_acf_patched", False):
            def _get_acf(self, series, max_lag=None):
                if max_lag is None:
                    max_lag = int(max(5, 0.1 * len(series)))
                autocorr = np.ones(max_lag + 1)
                for lag in range(1, max_lag + 1):
                    autocorr[lag] = np.corrcoef(series[lag:], series[: len(series) - lag])[0, 1]
                return autocorr

            itb.CondIndTest._get_acf = _get_acf
            itb.CondIndTest._acf_patched = True
            out["tigramite_acf"] = "patched"
    except ImportError:
        pass
    return out


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 1. The v4 cohort exactly as the paper runner builds it
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


@dataclass
class Cohort:
    pids: list[str]
    files: list[Path]
    ev: dict
    stops: dict
    part_sets: dict  # cross arm -> {pid: {agent: partner pid}}
    seed: int
    K: int
    pc_set: set
    cfg: clb.ScaleConfig
    W: np.ndarray = field(repr=False)  # (n_cases, 200) shared case-bootstrap weights
    pos: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.pos = {p: i for i, p in enumerate(self.pids)}

    def job(self, pid: str) -> dict:
        """The per-case job dict of ``run_v4`` (same seed, same partner streams)."""
        i = self.pos[pid]

        def prt_for(parts):
            out, st_ = {}, {}
            for agent, pp_ in parts[pid].items():
                rec = bl.ev_records(self.ev.get(pp_))
                if agent in rec:
                    out[agent] = rec[agent]
                    st_[agent] = self.stops.get(pp_, {}).get(agent, np.inf)
            return out, st_

        return dict(path=str(self.files[i]), pid=pid, scale=SCALE, seed=self.seed * 100_003 + i, version="v4",
                    events=bl.ev_records(self.ev.get(pid)), stops=self.stops.get(pid, {}),
                    cross={arm: prt_for(parts) for arm, parts in self.part_sets.items()}, pcmci=pid in self.pc_set)

    @property
    def arms(self) -> tuple[str, ...]:
        return ("real", "nc_random") + tuple(self.part_sets)


def load_cohort(seed: int = 0, K: int = 8, n_pcmci: int = 60, n_boot: int = 200, grid: Path = GRID_V4,
                limit: int = 0) -> Cohort:
    """Switches the module table to v4 and rebuilds the paper run's cohort objects (cases in sorted order,
    seeded cross partners, seeded PCMCI+ subsample with pressor cases first, bootstrap weights)."""
    import pandas as pd

    clb.use_table("v4")
    files = sorted((grid / "grid").glob("*.parquet"))
    if limit:
        files = files[:limit]
    pids = [f.stem for f in files]
    ev = bl.load_events(set(pids), grid)
    st = pd.read_parquet(grid / "stops.parquet")
    stops = {pid: dict(zip(g.agent, g.t_stop)) for pid, g in st.groupby("pid")}
    part_sets = bn.cross_partner_sets(pids, ev, seed, K)
    press_agents = {clb.ACTIONS[a].agent for a in clb.PRESSORS}
    rng = np.random.default_rng(seed + 5)
    press = [p for p in pids if p in ev and ev[p].agent.isin(press_agents).any()]
    rest = [p for p in pids if p not in set(press)]
    order = list(rng.permutation(press)) + list(rng.permutation(rest))
    pc_set = set(order[:n_pcmci])
    W = clb.case_bootstrap_weights(len(pids), n_boot, seed)
    return Cohort(pids, files, ev, stops, part_sets, seed, K, pc_set, clb.SCALES[SCALE], W)


def case_arms(job: dict) -> dict[str, clb.CaseData]:
    """``run_job`` (v4 branch) of the paper runner: the case's vitals with the real, random-time and the K
    cross-patient action streams."""
    import pandas as pd

    cfg = clb.SCALES[job["scale"]]
    cols = ["window_start_sec", *clb.VITALS]
    d = pd.read_parquet(job["path"], columns=cols)
    events = bl.to_events(job["events"], job.get("stops"))
    case = clb.CaseData(job["pid"], d.window_start_sec.to_numpy(float) + cfg.window_sec / 2, cfg.window_sec,
                        {v: d[v].to_numpy(float) for v in clb.VITALS}, events)
    cross = {arm: bl.to_events(rec, st) for arm, (rec, st) in job["cross"].items()}
    return clb.build_arms(case, None, seed=job["seed"], arms=("real", "nc_random") + tuple(job["cross"]),
                          cross_events=cross)


def eligibility(case: clb.CaseData) -> dict[str, tuple[int, float]]:
    return {r.rid: (case.n_usable(r.action, r.event_dir), case.coverage(r.vital)) for r in clb.RELATIONS}


def is_eligible(n_events: int, coverage: float) -> bool:
    return n_events >= clb.MIN_EVENTS_PER_PATIENT and coverage >= clb.MIN_VITAL_COVERAGE


def worker_init() -> None:
    """Pool initializer: v4 table, shims, single-threaded BLAS."""
    import os

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    clb.use_table("v4")
    install_shims()


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 2. D1a / D1b: PCMCI+ with nonlinear conditional-independence tests
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

PCMCI_TESTS: tuple[str, ...] = ("parcorr", "gpdc", "gpdc_exact", "cmiknn", "cmiknn_shuffle")


RFF_FEATURES: int = 100
RFF_LENGTH_SCALE: float = 1.0
RFF_RIDGE: float = 1.0


def _rff_gpdc_class():
    """GPDC whose GP regression is replaced by its random-Fourier-feature approximation (Rahimi & Recht 2007):
    on the standardised conditions z, phi(z) = sqrt(2/D) cos(z W + b) with W ~ N(0, 1/l^2), b ~ U(0, 2 pi),
    D = ``RFF_FEATURES`` features (seeded by the number of conditions), ridge regression with penalty
    ``RFF_RIDGE`` = the GP noise variance (the weight-space view of an RBF GP with length-scale l =
    ``RFF_LENGTH_SCALE`` and unit prior variance, tigramite's kernel initialisation). The distance-correlation
    statistic, the uniform transform and the analytic null of GPDC are unchanged. Cost O(T D^2) per
    regression instead of O(T^3) with hyperparameter optimisation (exact GPDC: ~200 s per 165-block case)."""
    install_shims()
    from tigramite.independence_tests.gpdc import GPDC

    class RFFGPDC(GPDC):
        def __init__(self, n_features: int = RFF_FEATURES, length_scale: float = RFF_LENGTH_SCALE,
                     ridge: float = RFF_RIDGE, rff_seed: int = 0, **kw):
            super().__init__(**kw)
            self.n_features, self.length_scale, self.ridge, self.rff_seed = n_features, length_scale, ridge, rff_seed
            self._rff_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def _features(self, d: int) -> tuple[np.ndarray, np.ndarray]:
            if d not in self._rff_cache:
                rng = np.random.default_rng([self.rff_seed, d])
                Wm = rng.normal(0.0, 1.0 / self.length_scale, size=(d, self.n_features))
                b = rng.uniform(0.0, 2 * np.pi, self.n_features)
                self._rff_cache[d] = (Wm, b)
            return self._rff_cache[d]

        def _get_single_residuals(self, array, target_var, return_means=False, standardize=True,
                                  return_likelihood=False):
            dim, T = array.shape
            if dim <= 2:
                if return_likelihood:
                    return array[target_var, :], -np.inf
                return array[target_var, :]
            arr = np.array(array, float)
            if standardize:
                arr -= arr.mean(axis=1).reshape(dim, 1)
                std = arr.std(axis=1)
                for i in range(dim):
                    if std[i] != 0.0:
                        arr[i] /= std[i]
            y = arr[target_var]
            z = arr[2:].T
            key = (z.shape, float(z[0, 0]), float(z[-1, -1]), float(z.sum()))
            if getattr(self, "_last_key", None) == key:  # x and y are regressed on the same conditions
                Phi, Ainv = self._last_phi
            else:
                Wm, b = self._features(z.shape[1])
                Phi = np.sqrt(2.0 / self.n_features) * np.cos(z @ Wm + b)
                A = Phi.T @ Phi + self.ridge * np.eye(self.n_features)
                Ainv = np.linalg.inv(A)
                self._last_key, self._last_phi = key, (Phi, Ainv)
            w = Ainv @ (Phi.T @ y)
            mean = Phi @ w
            resid = y - mean
            if return_likelihood:
                return (resid, mean, -np.inf) if return_means else (resid, -np.inf)
            return (resid, mean) if return_means else resid

    return RFFGPDC


def _cmiknn_null_class():
    """CMIknn (Frenzel-Pompe kNN conditional mutual information) with a GPDC-style analytic significance: the
    p-value of a value is its upper tail probability in a null distribution of the same estimator on
    ``sig_samples`` independent samples (x, y) of the test's sample size (uniform marginals), generated once per
    sample size and cached. tigramite's own local-permutation shuffle test (``sig_samples`` kNN-CMI evaluations
    per test, ~10^5 tests per case here) is out of the compute budget; the precomputed null ignores the
    conditioning set (as GPDC's analytic null does), which is stated as a caveat."""
    install_shims()
    from tigramite.independence_tests.cmiknn import CMIknn

    class CMIknnNull(CMIknn):
        def __init__(self, null_seed: int = 0, **kw):
            super().__init__(**kw)
            self.significance = "analytic"
            self.null_seed = null_seed
            self._nulls: dict[int, np.ndarray] = {}

        def _null(self, T: int) -> np.ndarray:
            if T not in self._nulls:
                rng = np.random.default_rng([self.null_seed, T])
                vals = np.empty(self.sig_samples)
                xyz = np.array([0, 1])
                for b in range(self.sig_samples):
                    arr = rng.uniform(size=(2, T))
                    vals[b] = self.get_dependence_measure(arr, xyz)
                self._nulls[T] = np.sort(vals)
            return self._nulls[T]

        def get_analytic_significance(self, value, T, dim, xyz=None):
            null = self._null(int(T))
            return float((np.sum(null >= value) + 1) / (len(null) + 1))

    return CMIknnNull


def make_ci_test(name: str, sig_samples: int = 500, knn: float = 0.2, seed: int = 0):
    """ParCorr (the paper), GPDC (exact: GP regression on the conditions with hyperparameter optimisation),
    GPDC-RFF (the GP replaced by its random-Fourier-feature approximation, see ``_rff_gpdc_class``) — both
    with the distance correlation of the residuals and the analytic null from ``sig_samples`` draws per sample
    size — or CMIknn (Frenzel-Pompe kNN CMI, local restricted-shuffle test with ``sig_samples`` permutations)."""
    install_shims()
    if name == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr

        return ParCorr(significance="analytic")
    if name == "gpdc_exact":
        from tigramite.independence_tests.gpdc import GPDC

        return GPDC(significance="analytic", sig_samples=sig_samples)
    if name == "gpdc":
        return _rff_gpdc_class()(significance="analytic", sig_samples=sig_samples, rff_seed=seed)
    if name == "cmiknn_shuffle":
        from tigramite.independence_tests.cmiknn import CMIknn

        return CMIknn(knn=knn, shuffle_neighbors=5, significance="shuffle_test", sig_samples=sig_samples,
                      workers=1, seed=seed)
    if name == "cmiknn":
        return _cmiknn_null_class()(knn=knn, shuffle_neighbors=5, significance="shuffle_test", sig_samples=sig_samples,
                                    workers=1, seed=seed, null_seed=seed)
    raise ValueError(name)


def _mci_conditions(pc, i: int, j: int, tau: int) -> list[tuple[int, int]]:
    """Lagged-parent conditioning set of the MCI test of the link (i, -tau) -> (j, 0): the lagged parents of
    j found in the PC1 phase (without the link itself) and those of i shifted by tau."""
    P = getattr(pc, "all_lagged_parents", None) or {}
    Z = [(k, -t) for (k, t) in [(a, -b) for (a, b) in P.get(j, [])] if not (k == i and t == tau)]
    Z += [(k, -(t + tau)) for (k, t) in [(a, -b) for (a, b) in P.get(i, [])]]
    return list(dict.fromkeys(Z))


def _spearman_sign(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    r = stats.spearmanr(x, y).correlation
    return float(np.sign(r)) if np.isfinite(r) else 0.0


def _partial_spearman_sign(arr: np.ndarray) -> float:
    """Sign of the partial Spearman correlation of rows 0 and 1 given the remaining rows (rank transform, then
    linear adjustment on the conditions)."""
    from scipy import stats

    R = np.vstack([stats.rankdata(row) for row in arr])
    x, y = R[0], R[1]
    if R.shape[0] > 2:
        Xz = np.column_stack([np.ones(R.shape[1]), R[2:].T])
        bx = np.linalg.lstsq(Xz, x, rcond=None)[0]
        by = np.linalg.lstsq(Xz, y, rcond=None)[0]
        x, y = x - Xz @ bx, y - Xz @ by
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.sign(np.corrcoef(x, y)[0, 1]))


def sign_of_link(pc, r, names, i, j, tau, tau_max) -> float:
    """Sign given to an unsigned dependence statistic (GPDC: distance correlation; CMIknn: CMI) of the link
    (i, -tau) -> (j, 0): the sign of the partial Spearman correlation of x_{t-tau} and y_t given the MCI
    lagged-parent conditions (identical rule for every test). For GPDC the Spearman sign of the test's own
    (GP / RFF) residuals is recorded as a secondary sign in ``pc.cond_ind_test._signs2[(i, j, tau)]``."""
    test = pc.cond_ind_test
    Z = _mci_conditions(pc, i, j, tau)
    array, xyz, _, _ = test._get_array(X=[(i, -tau)], Y=[(j, 0)], Z=Z, tau_max=tau_max)
    arr = np.array(array, float)
    s1 = _partial_spearman_sign(arr)
    if hasattr(test, "_get_single_residuals"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                rx = test._get_single_residuals(arr.copy(), target_var=0)
                ry = test._get_single_residuals(arr.copy(), target_var=1)
                s2 = _spearman_sign(np.asarray(rx), np.asarray(ry))
            except Exception:  # noqa: BLE001
                s2 = np.nan
        if not hasattr(test, "_signs2"):
            test._signs2 = {}
        test._signs2[(i, j, tau)] = s2
    return s1


SIGN_FN = {"parcorr": None, "gpdc": sign_of_link, "gpdc_exact": sign_of_link, "cmiknn": sign_of_link,
           "cmiknn_shuffle": sign_of_link}


def pcmci_job(args) -> tuple[str, dict, float]:
    """One case, the requested arms, one CI test: {arm: pcmci dict (as ``clb.pcmci_case``)} + eligibility."""
    job, test_name, arms, sig_samples, knn = args
    t0 = time.time()
    cfg = clb.SCALES[job["scale"]]
    all_arms = case_arms(job)
    out = {}
    for arm in arms:
        case = all_arms[arm]
        rec = {"elig": eligibility(case), "pcmci": {}, "error": None}
        try:
            test = make_ci_test(test_name, sig_samples=sig_samples, knn=knn, seed=job["seed"])
            rec["pcmci"] = clb.pcmci_case(case, cfg, opts=clb.V4, cond_ind_test=test, sign_fn=SIGN_FN[test_name])
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
        out[arm] = rec
    return job["pid"], out, time.time() - t0



# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 3. D1d: rank (Spearman) lagged cross-correlation
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

XCORR_RULES: tuple[tuple[str, str], ...] = (("v1", "abs"), ("dir", "dir"), ("pos", "pos"))


def rank_transform(x: np.ndarray) -> np.ndarray:
    """Average ranks of the finite values (NaN kept), so ``clb.xcorr_curve`` of two ranked series is the
    Spearman cross-correlation at every lag (ranks over the whole case, as in the Pearson version the
    centring is over the whole case)."""
    from scipy import stats

    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if ok.any():
        out[ok] = stats.rankdata(x[ok])
    return out


def xcorr_job(args) -> tuple[str, dict, float]:
    """One case, all arms: rank cross-correlation curves per (action, vital) pair (float16) plus, when
    ``with_pearson``, the paper's Pearson curves on the real arm (validation of the aggregation path)."""
    job, with_pearson = args
    t0 = time.time()
    cfg = clb.SCALES[job["scale"]]
    L = int(round(cfg.xcorr_max_lag_sec / cfg.row_stride_sec))
    out = {}
    for arm, case in case_arms(job).items():
        rec = {"elig": eligibility(case), "rank": {}, "pearson": {}}
        for a, v in clb.PAIRS:
            ev = case.events.get(a)
            if ev is None or not ev.usable.any():
                continue
            x = case.dose_series(a)
            y = case.vitals[v]
            c = clb.xcorr_curve(rank_transform(x), rank_transform(y), L)
            if c is not None:
                rec["rank"][(a, v)] = c.astype(np.float16)
            if with_pearson and arm == "real":
                c = clb.xcorr_curve(x, y, L)
                if c is not None:
                    rec["pearson"][(a, v)] = c.astype(np.float16)
        out[arm] = rec
    return job["pid"], out, time.time() - t0


class XcorrAccumulator:
    """Streams per-case curves into full and bootstrap-weighted sums per (arm, kind, relation) — the same
    numbers as stacking the eligible cases' curves and applying ``W`` (the paper's aggregator) without
    holding every curve in memory."""

    def __init__(self, cohort: Cohort):
        self.W = cohort.W
        self.B = cohort.W.shape[1]
        self.cfg = cohort.cfg
        self.L = int(round(self.cfg.xcorr_max_lag_sec / self.cfg.row_stride_sec))
        self.full: dict[tuple, np.ndarray] = {}
        self.boot: dict[tuple, np.ndarray] = {}
        self.den: dict[tuple, np.ndarray] = {}
        self.n: dict[tuple, int] = {}
        self.pos = cohort.pos

    def add(self, pid: str, res: dict) -> None:
        c = self.pos[pid]
        w = self.W[c]
        for arm, rec in res.items():
            for kind in ("rank", "pearson"):
                for (a, v), curve in rec[kind].items():
                    cv = np.nan_to_num(curve.astype(float))
                    for r in clb.RELATIONS:
                        if (r.action, r.vital) != (a, v) or not is_eligible(*rec["elig"][r.rid]):
                            continue
                        key = (arm, kind, r.rid)
                        if key not in self.full:
                            self.full[key] = np.zeros(len(cv))
                            self.boot[key] = np.zeros((self.B, len(cv)))
                            self.den[key] = np.zeros(self.B)
                            self.n[key] = 0
                        self.full[key] += cv
                        nz = np.nonzero(w)[0]
                        self.boot[key][nz] += w[nz, None] * cv[None, :]
                        self.den[key] += w
                        self.n[key] += 1

    def estimates(self, kind: str, prefix: str) -> dict[tuple, tuple[float, np.ndarray, int]]:
        """rep_store entries (arm, method, rid, 0.0) -> (est, reps, n_cases) for the three peak rules."""
        store = {}
        for (arm, k, rid), full in self.full.items():
            if k != kind:
                continue
            r = clb.REL_BY_ID[rid]
            orient = 1 if r.kind == "response" else -1
            n = self.n[(arm, k, rid)]
            mean = full / n
            den = self.den[(arm, k, rid)]
            ok = den > 0
            bc = self.boot[(arm, k, rid)][ok] / den[ok, None]
            for suffix, mode in XCORR_RULES:
                val, lag = clb.xcorr_peak(mean, self.L, mode, orient)
                reps = np.full(self.B, np.nan)
                reps[ok] = [clb.xcorr_peak(b, self.L, mode, orient)[0] for b in bc]
                store[(arm, f"{prefix}_{suffix}", rid, 0.0)] = (val, reps, n)
        return store


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 4. D1c: nonlinear Granger by gradient-boosted predictive gain
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

GBM_STRIDE_SEC: float = 20.0  # anchor spacing
GBM_POLICY_H: float = 10.0  # policy targets: the action's change over the next 10 s
GBM_TARGETS_RESPONSE: dict[str, float] = {}  # filled by gbm_targets()


def gbm_bins(cfg: clb.ScaleConfig) -> list[tuple[int, int]]:
    return cfg.bin_lags()


def gbm_series_names() -> list[str]:
    return list(clb.VITALS) + list(clb.ACTIONS)


def gbm_feature_names(cfg: clb.ScaleConfig) -> list[str]:
    names = []
    for s in gbm_series_names():
        names.append(f"{s}:L0")
        for k0, k1 in gbm_bins(cfg):
            names.append(f"{s}:B{k0}-{k1}")
    return names


def gbm_targets() -> list[tuple[str, str, float]]:
    """(kind, series, horizon) targets implied by the relation table: each effect vital at its primary
    horizon (response relations) and each policy action over the next ``GBM_POLICY_H`` s."""
    tg = []
    for r in clb.RELATIONS:
        key = ("response", r.vital, float(r.horizon_s)) if r.kind == "response" else ("policy", r.action, GBM_POLICY_H)
        if key not in tg:
            tg.append(key)
    return tg


def gbm_case_design(case: clb.CaseData, cfg: clb.ScaleConfig) -> dict | None:
    """Anchor rows every ``GBM_STRIDE_SEC``: within-case demeaned level and lag-bin means of every series
    (vitals, dose series; NaN after a track stops) and, per target, the series' change over its horizon.
    Returns {'X': (n, F) float32, 'Y': {target: (n,)}, 'present': {series: bool}} or None."""
    gi = clb.reg_grid(case, cfg)
    tg = case.t[gi]
    step = cfg.reg_step_sec
    n = len(tg)
    if n < 200:
        return None
    stride = max(1, int(round(GBM_STRIDE_SEC / step)))
    anchors = np.arange(0, n, stride)
    bins = gbm_bins(cfg)
    series, present = {}, {}
    for v in clb.VITALS:
        series[v] = case.vitals[v][gi].astype(float)
        present[v] = case.coverage(v) >= clb.MIN_VITAL_COVERAGE
    for a in clb.ACTIONS:
        ev = case.events.get(a)
        series[a] = ev.dose_at(tg) if ev is not None and len(ev.t) else np.zeros(n)
        present[a] = ev is not None and bool(ev.usable.any())
    blocks = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in gbm_series_names():
            x = series[s]
            mu = np.nanmean(x) if np.isfinite(x).any() else 0.0
            xd = x - mu
            ok = np.isfinite(xd)
            cs = np.concatenate([[0.0], np.cumsum(np.where(ok, xd, 0.0))])
            cn = np.concatenate([[0.0], np.cumsum(ok)])
            cols = [xd[anchors]]
            for k0, k1 in bins:  # mean of x over rows [t-k1, t-k0]
                hi = np.clip(anchors - k0 + 1, 0, n)
                lo = np.clip(anchors - k1, 0, n)
                cnt = cn[hi] - cn[lo]
                cols.append(np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan))
            blocks.append(np.column_stack(cols))
    X = np.hstack(blocks).astype(np.float32)
    Y = {}
    for kind, s, h in gbm_targets():
        kh = int(round(h / step))
        x = series[s]
        fut = np.full(len(anchors), np.nan)
        m = anchors + kh < n
        fut[m] = x[anchors[m] + kh] - x[anchors[m]]
        Y[(kind, s, h)] = fut.astype(np.float32)
    return {"X": X, "Y": Y, "present": present, "n": len(anchors)}


def gbm_job(args) -> tuple[str, dict, float]:
    job, arms = args
    t0 = time.time()
    cfg = clb.SCALES[job["scale"]]
    all_arms = case_arms(job)
    out = {}
    for arm in arms:
        d = gbm_case_design(all_arms[arm], cfg)
        out[arm] = d
        if d is not None:
            d["elig"] = eligibility(all_arms[arm])
    return job["pid"], out, time.time() - t0


GBM_MODEL: str = "gbm"  # "gbm" (HistGradientBoostingRegressor) | "linear" (ridge on the same design: the linear twin)


class _LinearTwin:
    """Ridge regression (alpha = 1) on the same design as the GBM; NaN features (missing lag bins) are set to 0
    = the within-case mean, which is what the GBM's missing-value branch approximates. Its average marginal
    effect is the sum of the source coefficients, i.e. the Granger statistic of this design."""

    def __init__(self):
        from sklearn.linear_model import Ridge

        self.m = Ridge(alpha=1.0)

    def fit(self, X, y):
        self.m.fit(np.nan_to_num(X, nan=0.0), y)
        return self

    def predict(self, X):
        return self.m.predict(np.nan_to_num(X, nan=0.0))


def gbm_fit_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int,
                    max_iter: int = 200, threads: int = 4) -> tuple[np.ndarray, list]:
    """Case-grouped K-fold out-of-fold predictions of a HistGradientBoostingRegressor (or, with
    ``GBM_MODEL == "linear"``, of the ridge twin); returns (oof, models with their fold masks)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    oof = np.full(len(y), np.nan)
    models = []
    gkf = GroupKFold(n_splits=n_folds)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if GBM_MODEL == "linear":
            m = _LinearTwin()
        else:
            m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                                              random_state=seed + k)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        models.append((m, te))
    return oof, models


def gbm_relation_stats(X: np.ndarray, y: np.ndarray, case_idx: np.ndarray, n_cases: int, W: np.ndarray,
                       src_cols: np.ndarray, delta: float, n_folds: int, seed: int, threads: int = 4,
                       models_full=None, oof_full=None) -> dict:
    """Predictive gain and average marginal effect of one source feature block.

    gain  = out-of-fold R^2(full) - R^2(without the source columns), case bootstrap of the per-case
            squared-error sums; AME = mean over anchor rows of [f(x + d 1_S) - f(x - d 1_S)] / (2 d) with the
            out-of-fold model of the row's fold (d = ``delta``, half the pooled within-case SD of the source
            series; the AME is in effect units per source unit), case bootstrap of per-case sums."""
    ok = np.isfinite(y)  # NaN features are handled natively by the histogram GBM (missing-value branch)
    Xo, yo, co = X[ok], y[ok].astype(float), case_idx[ok]
    if oof_full is None:
        oof_full, models_full = gbm_fit_predict(Xo, yo, co, n_folds, seed, threads=threads)
    keep = np.ones(X.shape[1], bool)
    keep[src_cols] = False
    oof_wo, _ = gbm_fit_predict(Xo[:, keep], yo, co, n_folds, seed, threads=threads)
    # per-case sums for the bootstrap of the gain
    sse_f = np.bincount(co, weights=(yo - oof_full) ** 2, minlength=n_cases)
    sse_w = np.bincount(co, weights=(yo - oof_wo) ** 2, minlength=n_cases)
    sst = np.bincount(co, weights=(yo - yo.mean()) ** 2, minlength=n_cases)
    cnt = np.bincount(co, minlength=n_cases).astype(float)
    gain = float((sse_w.sum() - sse_f.sum()) / max(sst.sum(), 1e-12))
    with np.errstate(invalid="ignore", divide="ignore"):
        g_reps = (W.T @ sse_w - W.T @ sse_f) / (W.T @ sst)
    r2_full = float(1 - sse_f.sum() / max(sst.sum(), 1e-12))
    # AME with the out-of-fold model
    ame = np.full(len(yo), np.nan)
    for m, te in models_full:
        Xp = Xo[te].copy()
        Xm = Xo[te].copy()
        Xp[:, src_cols] += delta
        Xm[:, src_cols] -= delta
        ame[te] = (m.predict(Xp) - m.predict(Xm)) / (2 * delta)
    s_ame = np.bincount(co, weights=np.nan_to_num(ame), minlength=n_cases)
    est, reps = clb.ratio_boot(s_ame, cnt, W)
    return dict(est=est, reps=reps, n_cases=int((cnt > 0).sum()), n_rows=int(len(yo)), gain=gain, gain_reps=g_reps,
                r2_full=r2_full, oof_full=oof_full, models_full=models_full)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 5. D2: typed pooled VARX with the dose exposure through dose-size bins
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def scored_channels() -> list[tuple[str, str]]:
    """(action, 'up'|'down') channels that carry a scored response relation."""
    out = []
    for r in clb.RELATIONS:
        if r.kind == "response":
            ch = (r.action, "up" if r.event_dir > 0 else "down")
            if ch not in out:
                out.append(ch)
    return out


def dose_bin_edges(ev_by_pid: dict, n_bins: int) -> dict[tuple[str, str], np.ndarray]:
    """Per scored channel, the |step| quantile edges (tertiles for n_bins = 3) of the real cohort's usable
    titration steps (starts / stops excluded as in ``EventSet.build``)."""
    import pandas as pd

    if n_bins <= 1:
        return {}
    sizes: dict[tuple[str, str], list] = {ch: [] for ch in scored_channels()}
    for pid, df in ev_by_pid.items():
        for agent, g in df.groupby("agent"):
            a = clb.AGENT_TO_ACTION.get(agent)
            if a is None:
                continue
            ev = clb.EventSet.build(g["t_sec"], g["value_before"], g["value_after"], clb.ACTIONS[a])
            d = ev.after - ev.before
            for dirn, m in (("up", ev.usable & (d > 0)), ("down", ev.usable & (d < 0))):
                if (a, dirn) in sizes and m.any():
                    sizes[(a, dirn)].extend(np.abs(d[m]).tolist())
    edges = {}
    for ch, vals in sizes.items():
        v = np.asarray(vals, float)
        if len(v) < 3 * n_bins:
            continue
        q = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
        edges[ch] = q
    return edges


def dosebin_columns(cfg: clb.ScaleConfig, edges: dict, n_bins: int) -> tuple[dict[str, slice], list[tuple]]:
    """Column map (same layout as ``clb.varx_columns``: AR block, then one slice per channel key) where each
    scored channel with edges is split into ``n_bins`` dose-size sub-channels '<action>:<dir>#<k>'. Returns
    (cols, channel specs [(action, dir, key, lo, hi)]) in ``clb.VARX_CHANNELS`` order."""
    nb = len(cfg.bin_lags())
    k = len(clb.VITALS)
    cols = {"ar": slice(0, k * cfg.varx_p)}
    off = k * cfg.varx_p
    specs = []
    for a, c in clb.VARX_CHANNELS:
        if (a, c) in edges:
            e = np.concatenate([[0.0], edges[(a, c)], [np.inf]])
            for j in range(n_bins):
                key = f"{a}:{c}#{j}"
                cols[key] = slice(off, off + nb)
                specs.append((a, c, key, float(e[j]), float(e[j + 1])))
                off += nb
        else:
            key = f"{a}:{c}"
            cols[key] = slice(off, off + nb)
            specs.append((a, c, key, 0.0, np.inf))
            off += nb
    cols["_total"] = slice(0, off)
    return cols, specs


def varx_dosebin_design(case: clb.CaseData, cfg: clb.ScaleConfig, specs: list[tuple]) -> tuple | None:
    """``clb.varx_response_design_v4`` with the scored channels split by |step| into the dose bins of
    ``specs`` (with no split it is identical to the paper's design)."""
    gi = clb.reg_grid(case, cfg)
    tg = case.t[gi]
    Wg = cfg.reg_step_sec
    V = np.column_stack([case.vitals[v][gi] for v in clb.VITALS])
    dV = np.vstack([np.full((1, V.shape[1]), np.nan), np.diff(V, axis=0)])
    p = cfg.varx_p
    if len(tg) <= p + 10:
        return None
    Xar, _ = clb.lag_matrix(np.nan_to_num(dV, nan=0.0), p)
    Y = dV[p:]
    bins = cfg.bin_lags()
    blocks = [Xar]
    for a, c, key, lo, hi in specs:
        ev = case.events.get(a, clb.EventSet.empty())
        if len(ev.t):
            m = clb._channel_masks(ev)[c]
            if np.isfinite(hi) or lo > 0:
                size = np.abs(ev.after - ev.before)
                m = m & (size > lo) & (size <= hi) if lo > 0 else m & (size <= hi)
        else:
            m = np.zeros(0, bool)
        u = clb.increment_series(ev, tg, Wg, m)
        blocks.append(clb.binned_lags(u, bins)[p:])
    X = np.hstack(blocks)
    if np.isfinite(Y).any(1).sum() < 50:
        return None
    return X, Y


def varx_job(args) -> tuple[str, dict, float]:
    """One case, the requested arms: per-equation sufficient statistics of the dose-bin VARX."""
    job, arms, specs = args
    t0 = time.time()
    cfg = clb.SCALES[job["scale"]]
    all_arms = case_arms(job)
    out = {}
    for arm in arms:
        d = varx_dosebin_design(all_arms[arm], cfg, specs)
        out[arm] = {} if d is None else {clb.VITALS[j]: S for j, S in clb.suff_stats_per_equation(*d).items()}
    return job["pid"], out, time.time() - t0


class VarxAccumulator:
    """Full and bootstrap-weighted sufficient statistics per (arm, equation) (``Aggregator._acc``)."""

    def __init__(self, cohort: Cohort):
        self.W = cohort.W
        self.B = cohort.W.shape[1]
        self.pos = cohort.pos
        self.full: dict[tuple, np.ndarray] = {}
        self.boot: dict[tuple, np.ndarray] = {}
        self.count: dict[tuple, int] = {}

    def add(self, pid: str, res: dict) -> None:
        c = self.pos[pid]
        w = self.W[c]
        nz = np.nonzero(w)[0]
        for arm, eqs in res.items():
            for v, S in eqs.items():
                key = (arm, v)
                if key not in self.full:
                    self.full[key] = np.zeros_like(S)
                    self.boot[key] = np.zeros((self.B, len(S)))
                    self.count[key] = 0
                self.full[key] += S
                self.count[key] += 1
                self.boot[key][nz] += w[nz, None] * S[None, :]

    def step_responses(self, arm: str, cfg: clb.ScaleConfig, cols: dict, specs: list[tuple], n_bins: int,
                       prefix: str = "varx_dosebins") -> dict[tuple, tuple[float, np.ndarray, int]]:
        """rep_store entries for the response relations: per dose bin ('<prefix>_b<k>', per-unit step response
        of the bin's sub-channel) and their mean ('<prefix>'); with no split, '<prefix>' is the paper's VARX."""
        kx = cols["_total"].stop
        nv = len(clb.VITALS)
        keys = [(arm, v) for v in clb.VITALS]
        if not any(k in self.full for k in keys):
            return {}

        def beta_of(Ss):
            Bm = np.zeros((kx, nv))
            for j, S_ in enumerate(Ss):
                Bm[:, j] = np.nan if (S_ is None or S_[-1] <= 0) else clb.solve_suff(S_, kx, 1)[:, 0]
            return Bm

        bf = beta_of([self.full.get(k) for k in keys])
        bb = [beta_of([self.boot[k][b] if k in self.boot else None for k in keys]) for b in range(self.B)]
        store = {}
        for r in clb.RELATIONS:
            if r.kind != "response":
                continue
            vi = clb.VITALS.index(r.vital)
            k_ = keys[vi]
            if k_ not in self.full:
                continue
            dirn = "up" if r.event_dir > 0 else "down"
            chans = [key for a, c, key, lo, hi in specs if a == r.action and c == dirn]
            h = clb.primary_h(r, "varx", cfg)
            ests, repss = [], []
            for key in chans:
                est = clb.varx_step_response(np.nan_to_num(bf), cfg, key, h, cols)[vi]
                reps = np.array([clb.varx_step_response(np.nan_to_num(b), cfg, key, h, cols)[vi]
                                 if np.isfinite(b[:, vi]).all() else np.nan for b in bb])
                ests.append(est)
                repss.append(reps)
            n_c = self.count[k_]
            if len(chans) == 1:
                store[(arm, prefix, r.rid, h)] = (float(ests[0]), repss[0], n_c)
            else:
                for j, (e, rp) in enumerate(zip(ests, repss)):
                    store[(arm, f"{prefix}_b{j + 1}", r.rid, h)] = (float(e), rp, n_c)
                store[(arm, prefix, r.rid, h)] = (float(np.mean(ests)), np.mean(np.vstack(repss), 0), n_c)
        return store


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 6. D3: PAT-SBP beat-level coupling, rank and monotone versions of the OLS slope
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def pat_epoch_coupling(pat: np.ndarray, sbp: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """The paper's epoch statistic (``fast_coupling.pat_coupling``: 31-beat running-mean detrended PAT on
    detrended SBP, OLS slope and Pearson r) plus the Spearman rank correlation of the same detrended series
    and an isotonic (monotone) fit: PAT regressed on SBP under an increasing and a decreasing constraint
    (pool-adjacent-violators), the better-fitting direction kept; ``iso_slope`` is its fitted change between
    the 5th and 95th SBP percentiles per mmHg (0 when both fits are constant), ``iso_r2`` its R^2."""
    from scipy import stats
    from sklearn.isotonic import IsotonicRegression

    from clbench.edges import fast_coupling as fc

    base = fc.pat_coupling(pat, sbp, valid)
    out = {**base, "spearman": np.nan, "iso_slope": np.nan, "iso_dir": 0.0, "iso_r2": np.nan}
    if not np.isfinite(base["pat_sbp_slope"]):
        return out
    ok = np.asarray(valid, bool) & np.isfinite(pat) & np.isfinite(sbp)
    a = fc.running_mean_detrend(pat, ok, 31)
    b = fc.running_mean_detrend(sbp, ok, 31)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    out["spearman"] = float(stats.spearmanr(b, a).correlation)
    sst = float(((a - a.mean()) ** 2).sum())
    best = None
    for inc in (True, False):
        iso = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(b, a)
        f = iso.predict(b)
        sse = float(((a - f) ** 2).sum())
        if best is None or sse < best[0] - 1e-12:
            best = (sse, inc, iso)
    sse, inc, iso = best
    q05, q95 = np.quantile(b, [0.05, 0.95])
    rng_ = float(iso.predict([q95])[0] - iso.predict([q05])[0])
    out["iso_slope"] = rng_ / (q95 - q05) if q95 > q05 else 0.0
    out["iso_dir"] = float(np.sign(rng_))
    out["iso_r2"] = 1 - sse / sst if sst > 0 else np.nan
    return out


def pat_case_job(args) -> dict:
    """Per-epoch couplings of one case from the cached beat and PAT tables (no waveform access)."""
    pid, eps, native = args
    import pandas as pd

    out = {"patient_id": pid, "rows": [], "error": None}
    try:
        beats = pd.read_parquet(Path(native) / "beats" / f"{pid}.parquet")
        pf = pd.read_parquet(Path(native) / "pat" / f"{pid}.parquet")
        if len(pf) != len(beats) or not np.array_equal(pf["t_r"].to_numpy(float), beats["t_r"].to_numpy(float)):
            out["error"] = "pat/beats misaligned"
            return out
        tr = beats["t_r"].to_numpy(float)
        pat_all = pf["pat_ms"].to_numpy(float)
        sbp_all = beats["sbp"].to_numpy(float)
        v_all = beats["valid"].to_numpy(bool)
        for ei, t0, t1 in eps:
            lo, hi = np.searchsorted(tr, t0, "left"), np.searchsorted(tr, t1, "left")
            st = pat_epoch_coupling(pat_all[lo:hi], sbp_all[lo:hi], v_all[lo:hi])
            out["rows"].append({"patient_id": pid, "epoch_index": int(ei), "t0": float(t0), "t1": float(t1), **st})
    except Exception as ex:  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 7. Scoring through the shared admission rule
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def score_store(store: dict, cfg: clb.ScaleConfig, methods: tuple[str, ...] | None = None, q: float = 0.05):
    """(pooled single-arm table, calibrated admission table, null-discovery table, leaderboard) of every
    method in a rep_store {(arm, method, rid, h): (est, reps, n_cases)} with the paper's rule."""
    import pandas as pd

    rows = []
    for (arm, m, rid, h), (est, reps, n) in store.items():
        r = clb.REL_BY_ID[rid]
        lo, hi = clb.ci(reps)
        rows.append(dict(arm=arm, method=m, rid=rid, kind=r.kind, label=r.label, h=h, expected=r.expected, est=est,
                         lo=lo, hi=hi, call=clb.sign_call(est, lo, hi, r.expected, n), n_cases=n,
                         n_boot_ok=int(np.isfinite(reps).sum())))
    pooled = pd.DataFrame(rows)
    methods = methods or tuple(sorted({k[1] for k in store}))
    calib = clb.calibrated_admission_table(store, cfg, q=q)
    if calib is not None:
        calib = calib[calib.method.isin(methods)].reset_index(drop=True)
    nulls = clb.null_discovery_table(pooled, cfg, methods)
    lb = clb.leaderboard(calib, nulls, methods)
    return pooled, calib, nulls, lb


__all__ = ["Cohort", "PCMCI_TESTS", "SIGN_FN", "VarxAccumulator", "XcorrAccumulator", "case_arms", "dose_bin_edges",
           "dosebin_columns", "eligibility", "gbm_case_design", "gbm_job", "gbm_relation_stats", "gbm_targets",
           "install_shims", "is_eligible", "load_cohort", "make_ci_test", "pat_case_job", "pat_epoch_coupling",
           "pcmci_job", "rank_transform", "score_store", "varx_dosebin_design", "varx_job", "worker_init",
           "xcorr_job"]
