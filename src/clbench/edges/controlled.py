"""
Error-controlled concept edges: surrogate-null p-values + BH / BY within a patient.

For every ordered pair (a, b) of concept-activation series we test whether a's past carries
information about b's present, with a p-value from surrogates that break the a→b alignment while
keeping each series' own autocorrelation / persistence, then select edges with Benjamini–Hochberg
(or Benjamini–Yekutieli, valid under arbitrary dependence) across all candidate pairs of the
patient. The selected edge set then has FDR ≤ α for the null hypotheses the test targets.

Test statistics (``statistic=``)
--------------------------------
``"xcorr"``      max over lags k = 1..K of |corr(a_{t-k}, b_t)| on the raw series.
``"pwxcorr"``    the same on AR(K)-prewhitened series (Haugh 1976); removes the reverse-direction
                 and common-persistence leakage of the raw cross-correlation.
``"granger"``    bivariate Granger: partial R² of a's lags 1..K for b_t given [1, b's lags 1..K].
``"cgranger"``   conditional Granger: partial R² of a's lags given [1, lags 1..K of every other
                 series, b's own included]. Targets *direct* lagged edges among the observed
                 concepts (no unobserved driver assumed).
``"pwcgranger"`` conditional Granger whose test block is a's AR(K) *innovations* (prewhitened
                 source) while the conditioning set stays raw. Same null as ``cgranger``; the
                 surrogates no longer re-capture an edge through the source's own persistence.

Every statistic is a deterministic function of (source series, the other series); a surrogate
replaces only the source series.

Surrogates (``surrogate=``)
---------------------------
``"shift"``  circular shifts of the source by multiples of a spacing g (a *subgroup* of the cyclic
             group; the working series is trimmed to a multiple of g). Under independence and
             circular stationarity of the source, the observed statistic is exchangeable with the
             surrogates, so the p-value is exact (Hemerik & Goeman 2018 for random subsets of a
             group). ``spacing=None`` picks ``g = max(K + 1, ceil((1 + r)/(1 − r)))`` from the
             source's lag-1 autocorrelation r, so surrogates do not overlap the observed alignment
             through persistence. Resolution is 1 / (T/g): a short, persistent stay has few
             independent alignments and therefore little power — by design.
             (A minimum-gap set of offsets is *not* a group and is anti-conservative in the far
             tail; see tests.)
``"block"``  circular block bootstrap of the source, block length from the lag-1 autocorrelation.
             Many distinct surrogates for short series but only approximately valid;
             under-represents memory longer than a block.

p-values (``pvalue=``)
---------------------
``"empirical"``  ``p = (1 + #{s : T_s >= T_obs}) / (B + 1)`` (exact with shift surrogates; its floor
                 is 1/(B+1) ≈ g/T, so BH over m pairs needs about m·g/(α·T) simultaneous
                 discoveries before it can reject anything).
``"gamma"``      a gamma distribution matched to the surrogates' mean and variance (of the
                 statistic, or of its square for correlations), evaluated at T_obs — the
                 moment-matched tail approximation of permutation p-values (Winkler et al. 2016).
                 Continuous, so it has power below the empirical floor; approximate only.
Both are always computed; ``pvalue`` chooses which one ``pvalues`` holds.

Working scale and covariates
----------------------------
``transform="auto"`` (default) analyses probability-valued concept series on the log-odds scale
(clipped to [1e-4, 1 − 1e-4]) and leaves binary or unbounded series as they are; linear
conditioning on a saturated probability does not block a path through the concept, its logit
nearly does. ``covariates`` (T × q, e.g. recorded drug infusions / ventilator settings) enter
every conditional-Granger conditioning set at lags 0..K and are never tested themselves.

What the guarantee covers
-------------------------
Under independence of a and b the shift p-value is valid, so BH over the candidate pairs controls
the expected fraction of *independent* pairs among selected edges (BY: under any dependence
between the p-values). The conditional statistics narrow the null to "no direct lagged effect of
a on b given the observed concepts' past"; for that composite null the surrogate distribution is
an approximation (exact when b's residual is Gaussian white noise), and it fails when an
unobserved driver acts on several concepts. ``latent_factors > 0`` adds a leave-pair-out estimate
of slow shared drivers (principal components of the *other* series' trailing moving average) to
the conditioning set: a heuristic with no guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma as _gamma_dist

STATISTICS: tuple[str, ...] = ("xcorr", "pwxcorr", "granger", "cgranger", "pwcgranger")
SURROGATES: tuple[str, ...] = ("shift", "block")
PROCEDURES: tuple[str, ...] = ("bh", "by")

DEFAULT_MAX_LAG: int = 4
DEFAULT_N_SURROGATES: int = 999


# ---------------------------------------------------------------------------
# Multiple-testing procedures
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """
    Benjamini–Hochberg step-up at level ``alpha``. NaN entries are not hypotheses.

    Returns a boolean array of ``pvalues``' shape: True where the hypothesis is rejected.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.zeros(p.shape, dtype=bool)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out
    flat = p[finite]
    order = np.argsort(flat, kind="mergesort")
    below = np.nonzero(flat[order] <= alpha * np.arange(1, m + 1) / m)[0]
    rejected = np.zeros(m, dtype=bool)
    if below.size:
        rejected[order[: below[-1] + 1]] = True
    out[finite] = rejected
    return out


def benjamini_yekutieli(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini–Yekutieli: BH at ``alpha / H_m``; controls FDR under arbitrary dependence."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = int(np.isfinite(p).sum())
    if m == 0:
        return np.zeros(p.shape, dtype=bool)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    return benjamini_hochberg(p, alpha / harmonic)


def fdr_select(pvalues: np.ndarray, alpha: float, procedure: str = "bh") -> np.ndarray:
    if procedure == "bh":
        return benjamini_hochberg(pvalues, alpha)
    if procedure == "by":
        return benjamini_yekutieli(pvalues, alpha)
    raise ValueError(f"procedure must be one of {PROCEDURES}, got {procedure!r}")


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------


def lag1_autocorr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    den = float(x @ x)
    return float(x[:-1] @ x[1:] / den) if den > 0 else 0.0


def integrated_time(x: np.ndarray, r_max: float = 0.995) -> float:
    """AR(1) integrated autocorrelation time ``(1 + r)/(1 − r)`` with r = lag-1 autocorrelation ≥ 0."""
    r = float(np.clip(lag1_autocorr(x), 0.0, r_max))
    return (1.0 + r) / (1.0 - r)


def auto_spacing(x: np.ndarray, max_lag: int) -> int:
    return int(max(max_lag + 1, np.ceil(integrated_time(x))))


def lattice_offsets(
    T: int, spacing: int, n_surrogates: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Non-identity elements {g, 2g, ...} of the subgroup of Z_T generated by g (T a multiple of g);
    all of them, or ``n_surrogates`` drawn without replacement.
    """
    if T % spacing:
        raise ValueError(f"T={T} is not a multiple of spacing={spacing}")
    offsets = spacing * np.arange(1, T // spacing)
    if offsets.size == 0:
        raise ValueError(f"No surrogate: T={T}, spacing={spacing}")
    if offsets.size <= n_surrogates:
        return offsets
    return np.sort(rng.choice(offsets, size=n_surrogates, replace=False))


def shift_index(T: int, offsets: np.ndarray) -> np.ndarray:
    """(B, T) index: row b is ``np.roll(x, offsets[b])`` expressed as ``x[idx[b]]``."""
    return (np.arange(T)[None, :] - np.asarray(offsets)[:, None]) % T


def default_block_len(x: np.ndarray, T: int, max_lag: int) -> int:
    tau = integrated_time(x, 0.99)
    return int(np.clip(np.ceil(2.0 * tau), 2 * max_lag, max(2 * max_lag, T // 4)))


def block_bootstrap_index(
    T: int, n_surrogates: int, block_len: int, rng: np.random.Generator
) -> np.ndarray:
    """(B, T) circular block-bootstrap index with blocks of ``block_len``."""
    block_len = max(1, min(int(block_len), T))
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T, size=(n_surrogates, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)[None, None, :]) % T
    return idx.reshape(n_surrogates, n_blocks * block_len)[:, :T]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    ok = sd > 1e-10
    Z = np.zeros_like(X)
    Z[:, ok] = (X[:, ok] - mu[ok]) / sd[ok]
    return Z, ok


def _lags(W: np.ndarray, K: int) -> np.ndarray:
    """(n, p, K) with [:, j, k] = W[t - (k + 1), j] for t = K..T-1."""
    T = W.shape[0]
    return np.stack([W[K - 1 - k : T - 1 - k] for k in range(K)], axis=2)


def prewhiten(X: np.ndarray, order: int) -> np.ndarray:
    """Residuals (T - order, p) of a per-column AR(order) OLS fit with intercept."""
    T, p = X.shape
    out = np.empty((T - order, p))
    for j in range(p):
        x = X[:, j]
        design = np.column_stack(
            [np.ones(T - order)] + [x[order - k - 1 : T - k - 1] for k in range(order)]
        )
        y = x[order:]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        out[:, j] = y - design @ coef
    return out


def _orth(Z: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of Z (rank-revealing via SVD)."""
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    if s.size == 0:
        return U
    return U[:, s > s[0] * 1e-10]


def _trailing_mean(W: np.ndarray, width: int) -> np.ndarray:
    """Mean of rows t-width..t-1 (strictly past; shorter at the start; row 0 is zero)."""
    c = np.vstack([np.zeros((1, W.shape[1])), np.cumsum(W, axis=0)])
    T = W.shape[0]
    t = np.arange(T)
    lo = np.maximum(0, t - width)
    n = np.maximum(1, t - lo)
    return (c[t] - c[lo]) / n[:, None]


def _factor_columns(W: np.ndarray, others: list[int], n_factors: int, width: int) -> np.ndarray:
    """First ``n_factors`` principal components of the others' trailing moving average."""
    if n_factors <= 0 or len(others) == 0:
        return np.zeros((W.shape[0], 0))
    S = _trailing_mean(W[:, others], width)
    S = S - S.mean(axis=0)
    U, s, _ = np.linalg.svd(S, full_matrices=False)
    r = min(n_factors, U.shape[1])
    return U[:, :r] * s[:r]


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------


@dataclass
class EdgeTestResult:
    """Surrogate p-values for every ordered pair; ``pvalues[a, b]`` tests a → b."""

    pvalues: np.ndarray  # (p, p); NaN on the diagonal and for non-candidate pairs
    pvalues_empirical: np.ndarray  # (p, p) exact surrogate-rank p-values
    pvalues_gamma: np.ndarray  # (p, p) moment-matched gamma tail approximation
    statistics: np.ndarray  # (p, p) observed statistic
    best_lag: np.ndarray  # (p, p) lag (windows) of the largest per-lag association; -1 if none
    n_surrogates: np.ndarray  # (p,) surrogates used per source
    spacing: np.ndarray  # (p,) shift spacing per source (0 for block surrogates)
    statistic: str
    surrogate: str
    max_lag: int


def _pvalue(stats: np.ndarray) -> float:
    """``stats[0]`` observed, ``stats[1:]`` surrogates."""
    obs = stats[0]
    tol = 1e-10 * max(1.0, abs(obs))
    return float((1 + np.sum(stats[1:] >= obs - tol)) / stats.shape[0])


def _pvalue_gamma(stats: np.ndarray, squared: bool) -> float:
    """Gamma fitted to the surrogates' first two moments, upper tail at the observed statistic."""
    x = stats**2 if squared else stats
    sur = x[1:]
    m, v = float(sur.mean()), float(sur.var(ddof=1)) if sur.size > 1 else 0.0
    if not (m > 0 and v > 0):
        return _pvalue(stats)
    return float(_gamma_dist.sf(x[0], m * m / v, scale=v / m))


def apply_transform(X: np.ndarray, transform: str = "auto", eps: float = 1e-4) -> np.ndarray:
    """``"logit"`` of clipped probabilities; ``"auto"`` = logit for non-binary columns in [0, 1]."""
    X = np.asarray(X, dtype=np.float64)
    if transform == "none":
        return X
    if transform not in ("auto", "logit"):
        raise ValueError(f"transform must be 'auto', 'logit' or 'none', got {transform!r}")
    out = X.copy()
    for j in range(X.shape[1]):
        x = X[:, j]
        in_unit = np.all((x >= 0.0) & (x <= 1.0))
        binary = np.all((x == 0.0) | (x == 1.0))
        if transform == "logit" or (in_unit and not binary):
            c = np.clip(x, eps, 1.0 - eps)
            out[:, j] = np.log(c) - np.log1p(-c)
    return out


def _working_series(W: np.ndarray, statistic: str, K: int) -> tuple[np.ndarray, np.ndarray]:
    """(source matrix, conditioning / target matrix), same length."""
    if statistic in ("xcorr", "granger", "cgranger"):
        return W, W
    E, _ = _standardize(prewhiten(W, K))
    if statistic == "pwxcorr":
        return E, E
    return E, W[K:]  # pwcgranger: innovations as the test block, raw conditioning


def edge_pvalues(
    X: np.ndarray,
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    statistic: str = "cgranger",
    surrogate: str = "shift",
    n_surrogates: int = DEFAULT_N_SURROGATES,
    spacing: int | None = None,
    block_len: int | None = None,
    candidates: np.ndarray | None = None,
    latent_factors: int = 0,
    factor_window: int | None = None,
    pvalue: str = "empirical",
    transform: str = "auto",
    covariates: np.ndarray | None = None,
    seed: int | np.random.Generator | None = 0,
    chunk: int = 256,
) -> EdgeTestResult:
    """
    Surrogate p-values for every candidate ordered pair of the columns of ``X`` (T × p).

    ``candidates`` is an optional (p, p) boolean mask of ordered pairs to test; the diagonal is
    never tested. Constant columns get p = 1 for every pair they are in.
    """
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}")
    if surrogate not in SURROGATES:
        raise ValueError(f"surrogate must be one of {SURROGATES}, got {surrogate!r}")
    if pvalue not in ("empirical", "gamma"):
        raise ValueError(f"pvalue must be 'empirical' or 'gamma', got {pvalue!r}")
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be (T, p), got shape {X.shape}")
    K = int(max_lag)
    T0, p = X.shape
    if T0 < 6 * K + 10:
        raise ValueError(f"series too short (T={T0}) for max_lag={K}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    cand = (
        np.ones((p, p), dtype=bool)
        if candidates is None
        else np.asarray(candidates, dtype=bool).copy()
    )
    np.fill_diagonal(cand, False)

    W, ok = _standardize(apply_transform(X, transform))
    SRC, CON = _working_series(W, statistic, K)
    Tw = SRC.shape[0]
    regression = statistic in ("granger", "cgranger", "pwcgranger")
    COV = None
    if covariates is not None:
        if statistic not in ("cgranger", "pwcgranger"):
            raise ValueError("covariates are only used by the conditional statistics")
        cov = np.nan_to_num(np.asarray(covariates, dtype=np.float64).reshape(T0, -1))
        cov, cov_ok = _standardize(cov)
        COV = cov[:, cov_ok][T0 - Tw :]

    pvals = np.full((p, p), np.nan)
    pvals_g = np.full((p, p), np.nan)
    stats_obs = np.full((p, p), np.nan)
    best_lag = np.full((p, p), -1, dtype=np.int64)
    n_used = np.zeros(p, dtype=np.int64)
    spacing_used = np.zeros(p, dtype=np.int64)

    for a in range(p):
        targets = [b for b in range(p) if cand[a, b]]
        if not targets:
            continue
        dead = [b for b in targets if not (ok[a] and ok[b])]
        for b in dead:
            pvals[a, b], pvals_g[a, b], stats_obs[a, b] = 1.0, 1.0, 0.0
        targets = [b for b in targets if b not in dead]
        if not targets:
            continue

        # Surrogate index on the (possibly trimmed) working window.
        if surrogate == "shift":
            g = auto_spacing(SRC[:, a], K) if spacing is None else int(spacing)
            T = g * (Tw // g)
            if T // g < 2:
                for b in targets:
                    pvals[a, b], pvals_g[a, b], stats_obs[a, b] = 1.0, 1.0, 0.0
                continue
            idx = shift_index(T, lattice_offsets(T, g, n_surrogates, rng))
            spacing_used[a] = g
        else:
            T = Tw
            bl = default_block_len(SRC[:, a], T, K) if block_len is None else int(block_len)
            idx = block_bootstrap_index(T, n_surrogates, bl, rng)
        idx = np.vstack([np.arange(T)[None, :], idx])  # row 0 = observed
        S = idx.shape[0]
        n_used[a] = S - 1

        src = SRC[Tw - T :, a]
        C = CON[Tw - T :]
        n = T - K
        Y = C[K:]
        Yc = Y - Y.mean(axis=0)
        yn = np.sqrt((Yc**2).sum(axis=0))

        bases: dict[int, np.ndarray] = {}
        resid: dict[int, tuple[np.ndarray, float]] = {}
        if regression:
            L_all = _lags(C, K)  # (n, p, K)
            fw = factor_window if factor_window is not None else max(4 * K, T // 20)
            shared = None
            for b in targets:
                if statistic == "granger":
                    cols = [np.ones((n, 1)), L_all[:, b, :]]
                else:
                    rest = [j for j in range(p) if j != a]
                    cols = [np.ones((n, 1)), L_all[:, rest, :].reshape(n, -1)]
                    if COV is not None and COV.shape[1]:
                        Cv = COV[Tw - T :]
                        cols += [Cv[K - k : T - k] for k in range(K + 1)]
                    if latent_factors > 0:
                        others = [j for j in range(p) if j not in (a, b)]
                        cols.append(_factor_columns(C, others, latent_factors, fw)[K:])
                    elif shared is not None:
                        bases[b] = shared
                        continue
                Q = _orth(np.hstack(cols))
                bases[b] = Q
                if statistic != "granger" and latent_factors == 0:
                    shared = Q
            for b in targets:
                Q = bases[b]
                y = Y[:, b]
                r = y - Q @ (Q.T @ y)
                resid[b] = (r, float(r @ r))

        stat_all = np.empty((S, p))
        lag_obs = np.full(p, -1, dtype=np.int64)
        for lo in range(0, S, chunk):
            hi = min(S, lo + chunk)
            s_cnt = hi - lo
            A_src = src[idx[lo:hi]]  # (s, T)
            Xm3 = np.empty((n, s_cnt, K))
            for k in range(K):
                Xm3[:, :, k] = A_src[:, K - 1 - k : T - 1 - k].T
            Xm = Xm3.reshape(n, s_cnt * K)
            if not regression:
                Xc = Xm - Xm.mean(axis=0)
                num = Xc.T @ Yc  # (sK, p)
                den = np.sqrt((Xc**2).sum(axis=0))[:, None] * yn[None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
                absc = np.abs(corr).reshape(s_cnt, K, p)
                stat_all[lo:hi] = absc.max(axis=1)
                if lo == 0:
                    lag_obs = absc[0].argmax(axis=0) + 1
                continue
            # Partial R^2 of the source lags given basis Q: v'G^{-1}v / r'r with
            # G = X'(I - QQ')X and v = X'r (r is orthogonal to Q).
            XtX = np.matmul(Xm3.transpose(1, 2, 0), Xm3.transpose(1, 0, 2))  # (s, K, K)
            eye = np.eye(K)[None]
            gram: dict[int, np.ndarray] = {}
            for b in targets:
                Q = bases[b]
                if id(Q) not in gram:
                    Cq = (Q.T @ Xm).reshape(Q.shape[1], s_cnt, K)
                    G = XtX - np.einsum("rsk,rsl->skl", Cq, Cq)
                    G = G + (1e-10 * np.trace(G, axis1=1, axis2=2) + 1e-12)[:, None, None] * eye
                    gram[id(Q)] = G
                G = gram[id(Q)]
                r, rss0 = resid[b]
                v = (Xm.T @ r).reshape(s_cnt, K)
                sol = np.linalg.solve(G, v[:, :, None])[:, :, 0]
                stat_all[lo:hi, b] = np.einsum("sk,sk->s", v, sol) / max(rss0, 1e-300)
                if lo == 0:
                    pc = v[0] / np.sqrt(np.maximum(np.diag(G[0]) * rss0, 1e-300))
                    lag_obs[b] = int(np.argmax(np.abs(pc))) + 1
        for b in targets:
            pvals[a, b] = _pvalue(stat_all[:, b])
            pvals_g[a, b] = _pvalue_gamma(stat_all[:, b], squared=not regression)
            stats_obs[a, b] = stat_all[0, b]
            best_lag[a, b] = lag_obs[b]

    return EdgeTestResult(
        pvalues=pvals if pvalue == "empirical" else pvals_g,
        pvalues_empirical=pvals,
        pvalues_gamma=pvals_g,
        statistics=stats_obs,
        best_lag=best_lag,
        n_surrogates=n_used,
        spacing=spacing_used,
        statistic=statistic,
        surrogate=surrogate,
        max_lag=K,
    )


@dataclass
class ControlledEdges:
    """Edge set selected at FDR level ``alpha`` within one patient."""

    adjacency: np.ndarray  # (p, p) bool; [a, b] = edge a → b
    tests: EdgeTestResult
    alpha: float
    procedure: str
    n_candidates: int

    def edges(self) -> list[tuple[int, int, float, int]]:
        """(source, target, p-value, best lag) for every selected edge."""
        src, tgt = np.nonzero(self.adjacency)
        return [
            (int(a), int(b), float(self.tests.pvalues[a, b]), int(self.tests.best_lag[a, b]))
            for a, b in zip(src, tgt)
        ]


def candidate_mask(p: int, admitted: np.ndarray | list[bool] | None = None) -> np.ndarray:
    """Ordered pairs of distinct admitted nodes (all nodes when ``admitted`` is None)."""
    adm = np.ones(p, dtype=bool) if admitted is None else np.asarray(admitted, dtype=bool)
    if adm.shape != (p,):
        raise ValueError(f"admitted must have length {p}")
    mask = np.outer(adm, adm)
    np.fill_diagonal(mask, False)
    return mask


def select_edges(tests: EdgeTestResult, alpha: float, procedure: str = "bh") -> ControlledEdges:
    """BH / BY over the finite p-values of ``tests``."""
    return ControlledEdges(
        adjacency=fdr_select(tests.pvalues, alpha, procedure),
        tests=tests,
        alpha=float(alpha),
        procedure=procedure,
        n_candidates=int(np.isfinite(tests.pvalues).sum()),
    )


def controlled_edges(
    X: np.ndarray,
    alpha: float = 0.1,
    *,
    procedure: str = "bh",
    admitted: np.ndarray | list[bool] | None = None,
    **test_kwargs,
) -> ControlledEdges:
    """
    Error-controlled edge set for one patient's concept series ``X`` (T × p).

    With ``admitted`` (length-p booleans, e.g. nodes admitted by conformal selection) only pairs
    of admitted nodes are candidates, so BH runs over those pairs alone.
    """
    X = np.asarray(X, dtype=np.float64)
    tests = edge_pvalues(X, candidates=candidate_mask(X.shape[1], admitted), **test_kwargs)
    return select_edges(tests, alpha, procedure)
