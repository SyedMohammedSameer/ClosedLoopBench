"""
CLBench typed temporal edges from closed-loop monitoring data: pooled, null-calibrated, multiplicity-
controlled admission of *policy* (vital -> action) and *response* (action -> vital) edges, plus
empirical-Bayes patient-level response edges and a pre-registered patient-specificity test.

Estimator (used by ``clbench.eval.closed_loop_bench`` as the ``typed_pooled`` method)
-----------------------------------------------------------------------------------
1. Population edge estimate: the pooled direction-split VARX of the benchmark (response equations: step
   response of a vital at the relation's primary horizon to a unit up- or down-increment of the action;
   policy equations: the up/down increment of the action on the vital's 2-min trend and level).
2. Null calibration: the same estimator on a cross-patient transplanted action stream (``nc_cross``),
   with the case-bootstrap weights shared between the two arms, so ``d = est_real - est_null`` has a
   paired case-bootstrap distribution.
3. Calibrated two-sided p-value from the paired bootstrap standard error (normal approximation; the
   percentile bootstrap cannot resolve the p-values that a multiplicity correction needs with 200 reps).
4. Admission: an edge is admitted iff (a) its paired 95% percentile CI excludes 0 and (b) its
   Benjamini-Yekutieli adjusted p-value (family = all relations scored in the run) is <= q. The admitted
   edge carries the sign of ``d``. Two-sided BY with sign declaration also bounds the directional
   (type-S) false discovery rate under arbitrary dependence (Benjamini & Yekutieli 2001, 2005).
5. Patient-level response edges: per-patient event estimates shrunk toward the pooled estimate with a
   method-of-moments normal-normal empirical-Bayes prior (``eb_shrink``).

v3 (ClosedLoopBench 1.0; ``calibrated_test_multi`` / ``admit_multi``). A single derangement can be an
unlucky draw (wave-3 verifier: at 2 s one draw's null estimate of remi-down -> MAP was itself
significant and doubled the calibrated effect). The v3 rule calibrates against the mean of K = 8
derangements, adds the between-draw variance of that mean to the paired bootstrap variance, and admits
an edge only if (a) the total-SE CI excludes 0, (b) BY-adjusted p <= q over the method's 18 relations,
and (c) the single-draw calibrated CI excludes 0 with the same sign in >= 75% of the draws. The v3
typed edge (``typed_pooled``) keeps the VARX base for response and policy relations: the pooled
sequential-trial contrast (``seqtrial``) was to replace it for response relations only if it passed a
simulator gate stated in advance, and it did not (``closed_loop_bench.TYPED_RESPONSE_BASE``). Every
method is scored with the same rule.

All functions are numpy-only and deterministic given their inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Multiplicity and calibrated admission
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def benjamini_yekutieli(p: np.ndarray) -> np.ndarray:
    """BY step-up adjusted p-values (valid under arbitrary dependence). NaN p-values stay NaN and are
    not counted in the family size."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    pv = p[ok]
    o = np.argsort(pv)
    ranked = pv[o] * m * c_m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    res = np.empty(m)
    res[o] = adj
    out[ok] = res
    return out


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass
class CalibratedTest:
    est: float  # real - null
    lo: float
    hi: float
    se: float
    z: float
    p: float
    est_real: float
    est_null: float
    n_boot_ok: int


def calibrated_test(est_real: float, reps_real: np.ndarray, est_null: float, reps_null: np.ndarray,
                    level: float = 0.95) -> CalibratedTest:
    """Paired (shared-weight) case-bootstrap test of real - null."""
    d = float(est_real - est_null)
    reps = np.asarray(reps_real, float) - np.asarray(reps_null, float)
    r = reps[np.isfinite(reps)]
    if len(r) < 10 or not np.isfinite(d):
        return CalibratedTest(d, np.nan, np.nan, np.nan, np.nan, np.nan, est_real, est_null, len(r))
    a = (1 - level) / 2
    lo, hi = float(np.quantile(r, a)), float(np.quantile(r, 1 - a))
    se = float(np.std(r, ddof=1))
    if se > 0:
        z = d / se
        p = float(2 * _norm_sf(abs(z)))
    else:  # degenerate bootstrap (no variation): no evidence either way
        z, p = np.nan, np.nan
    return CalibratedTest(d, lo, hi, se, float(z), p, float(est_real), float(est_null), len(r))


def admit(tests: list[CalibratedTest], q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(admitted flags, BY-adjusted p) for a family of calibrated tests."""
    p = np.array([t.p for t in tests], float)
    p_by = benjamini_yekutieli(p)
    ci_ex = np.array([np.isfinite(t.lo) and np.isfinite(t.hi) and (t.lo > 0 or t.hi < 0) for t in tests])
    adm = ci_ex & np.isfinite(p_by) & (p_by <= q)
    return adm, p_by


# ── v3: calibration against K null draws (cross-patient derangements) ─────────────────────────────────

MIN_STABLE: float = 0.75  # share of single-draw calibrations that must agree for admission
_Z975: float = 1.959963984540054


@dataclass
class MultiCalibratedTest:
    est: float  # real - mean of the K null estimates
    lo: float  # normal CI with the total SE
    hi: float
    se: float  # sqrt(se_boot^2 + se_draw^2)
    se_boot: float  # paired case-bootstrap SE of real - mean(null)
    se_draw: float  # between-draw SE of mean(null): sd_k(null_k) / sqrt(K)
    z: float
    p: float
    est_real: float
    est_null_mean: float
    null_sd: float  # sd over draws of the null estimates
    n_draws: int
    stable_frac: float  # share of draws whose single-draw paired 95% CI excludes 0 with the sign of est
    draw_est: tuple  # real - null_k, per draw
    n_boot_ok: int


def calibrated_test_multi(est_real: float, reps_real: np.ndarray, est_nulls, reps_nulls,
                          level: float = 0.95) -> MultiCalibratedTest:
    """Calibration of one estimate against K null draws that share its case-bootstrap weights.

    d = est_real - mean_k est_null_k. Two sources of uncertainty: (i) patient sampling, the paired
    case-bootstrap SD of real_b - mean_k null_k,b; (ii) the choice of derangement, the variance of the
    mean of K draws, var_k(null_k) / K. SE = sqrt(se_boot^2 + se_draw^2); CI d +- z SE; two-sided normal p.
    ``stable_frac``: share of the K single-draw paired percentile CIs (real - null_k) that exclude 0 with
    the sign of d. With K = 1 this reduces to ``calibrated_test`` except for the normal (not percentile) CI.
    """
    zq = _Z975 if abs(level - 0.95) < 1e-12 else float(__import__("statistics").NormalDist().inv_cdf(0.5 + level / 2))
    pairs = [(float(e), np.asarray(r, float)) for e, r in zip(est_nulls, reps_nulls)
             if e is not None and np.isfinite(e) and r is not None]
    K = len(pairs)
    nan = np.nan
    if K == 0 or not np.isfinite(est_real):
        return MultiCalibratedTest(nan, nan, nan, nan, nan, nan, nan, nan, float(est_real), nan, nan, K, nan, (), 0)
    e0 = np.array([e for e, _ in pairs])
    R0 = np.vstack([r for _, r in pairs])
    rr = np.asarray(reps_real, float)
    d = float(est_real - e0.mean())
    reps = rr - R0.mean(0)
    r = reps[np.isfinite(reps)]
    draw_est = tuple(float(est_real - x) for x in e0)
    null_sd = float(np.std(e0, ddof=1)) if K >= 2 else nan
    if len(r) < 10:
        return MultiCalibratedTest(d, nan, nan, nan, nan, nan, nan, nan, float(est_real), float(e0.mean()), null_sd,
                                   K, nan, draw_est, len(r))
    se_boot = float(np.std(r, ddof=1))
    se_draw = null_sd / math.sqrt(K) if K >= 2 else 0.0
    se = math.sqrt(se_boot**2 + se_draw**2)
    if se > 0:
        z = d / se
        p = float(2 * _norm_sf(abs(z)))
    else:
        z, p = nan, nan
    a = (1 - level) / 2
    stab = 0
    for k in range(K):
        rk = rr - R0[k]
        rk = rk[np.isfinite(rk)]
        if len(rk) < 10:
            continue
        lk, hk = float(np.quantile(rk, a)), float(np.quantile(rk, 1 - a))
        if (lk > 0 or hk < 0) and np.sign(draw_est[k]) == np.sign(d):
            stab += 1
    return MultiCalibratedTest(d, d - zq * se, d + zq * se, se, se_boot, se_draw, float(z), p, float(est_real),
                               float(e0.mean()), null_sd, K, stab / K, draw_est, len(r))


def admit_multi(tests: list[MultiCalibratedTest], q: float = 0.05, min_stable: float = MIN_STABLE,
                eligible: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(admitted, BY-adjusted p) for a family of multi-draw calibrated tests. Admitted iff the total-SE CI
    excludes 0, the BY-adjusted p (family = the tests passed, NaN p-values dropped) is <= q, and the edge
    is stable across draws (``stable_frac >= min_stable``); ``eligible`` (e.g. enough contributing
    cases) is an extra per-test requirement."""
    p = np.array([t.p for t in tests], float)
    p_by = benjamini_yekutieli(p)
    ci_ex = np.array([np.isfinite(t.lo) and np.isfinite(t.hi) and (t.lo > 0 or t.hi < 0) for t in tests], bool)
    stable = np.array([np.isfinite(t.stable_frac) and t.stable_frac >= min_stable - 1e-12 for t in tests], bool)
    adm = ci_ex & np.isfinite(p_by) & (p_by <= q) & stable
    if eligible is not None:
        adm &= np.asarray(eligible, bool)
    return adm, p_by


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Empirical-Bayes shrinkage of per-patient estimates
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


@dataclass
class EBResult:
    mu: float
    tau2: float
    shrunk: np.ndarray
    post_sd: np.ndarray
    weight: np.ndarray  # B_i = tau2 / (tau2 + se_i^2): weight on the patient's own estimate


def eb_shrink(est: np.ndarray, se2: np.ndarray, mu: float | None = None, mu_var: float | None = None) -> EBResult:
    """Normal-normal EB: est_i ~ N(theta_i, se_i^2), theta_i ~ N(mu, tau^2).

    ``mu`` defaults to the precision-weighted mean; tau^2 by DerSimonian-Laird (method of moments).
    The posterior variance includes the uncertainty of mu: B_i se_i^2 + (1 - B_i)^2 var(mu), with
    ``mu_var`` (e.g. the pooled estimate's bootstrap variance) or, by default, the random-effects
    variance 1 / sum 1/(se_i^2 + tau^2). (Without the second term a fully shrunk patient, tau^2 = 0,
    would get posterior SD 0.)
    """
    y = np.asarray(est, float)
    v = np.asarray(se2, float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    shr = np.full_like(y, np.nan)
    psd = np.full_like(y, np.nan)
    B = np.full_like(y, np.nan)
    if ok.sum() < 3:
        return EBResult(np.nan, np.nan, shr, psd, B)
    w = 1.0 / v[ok]
    mu_fe = float(np.sum(w * y[ok]) / np.sum(w))
    Q = float(np.sum(w * (y[ok] - mu_fe) ** 2))
    k = int(ok.sum())
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    wr = 1.0 / (v[ok] + tau2)
    if mu is None:
        mu = float(np.sum(wr * y[ok]) / np.sum(wr))
    if mu_var is None or not np.isfinite(mu_var):
        mu_var = float(1.0 / np.sum(wr))
    b = tau2 / (tau2 + v[ok]) if tau2 > 0 else np.zeros(k)
    shr[ok] = mu + b * (y[ok] - mu)
    psd[ok] = np.sqrt(b * v[ok] + (1 - b) ** 2 * mu_var)
    B[ok] = b
    return EBResult(float(mu), float(tau2), shr, psd, B)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# Patient-specificity test (early/late split of each patient's events)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════


def split_half_sums(tau: np.ndarray, y: np.ndarray, min_per_half: int = 3):
    """Split events by time into early (first floor(n/2)) and late halves; per half (n, sum, sumsq) of the
    finite outcomes. Returns None when either half has fewer than ``min_per_half`` finite outcomes."""
    tau = np.asarray(tau, float)
    y = np.asarray(y, float)
    o = np.argsort(tau, kind="stable")
    y = y[o]
    n = len(y)
    if n < 2 * min_per_half:
        return None
    e, l_ = y[: n // 2], y[n // 2 :]
    e, l_ = e[np.isfinite(e)], l_[np.isfinite(l_)]
    if len(e) < min_per_half or len(l_) < min_per_half:
        return None
    return (len(e), float(e.sum()), float((e**2).sum()), len(l_), float(l_.sum()), float((l_**2).sum()))


def _spec_stats(ne, se, sse, nl, sl, ssl, w):
    """Weighted (patient weights w) patient-specificity statistics for one bootstrap replicate."""
    me = se / ne
    ml = sl / nl
    # pooled-only: event-weighted mean of the early halves
    mu = np.sum(w * se) / np.sum(w * ne)
    # pooled within-patient variance of the early events
    sw2 = np.sum(w * (sse - ne * me**2)) / max(np.sum(w * (ne - 1)), 1e-12)
    # between-patient variance of the true patient means (method of moments, patient-weighted)
    W = np.sum(w)
    mbar = np.sum(w * me) / W
    var_m = np.sum(w * (me - mbar) ** 2) / max(W - 1, 1e-12)
    tau2 = max(0.0, var_m - np.sum(w * sw2 / ne) / W)
    B = tau2 / (tau2 + sw2 / ne) if tau2 > 0 else np.zeros_like(me)
    shr = mu + B * (me - mu)

    def mse(c):
        return ssl / nl - 2 * c * ml + c**2

    m_pp, m_pool, m_shr = mse(me), mse(mu), mse(shr)
    out = {
        "mse_pp": np.sum(w * m_pp) / W,
        "mse_pooled": np.sum(w * m_pool) / W,
        "mse_shrunk": np.sum(w * m_shr) / W,
        "sign_pp": np.sum(w * (np.sign(me) == np.sign(ml))) / W,
        "sign_pooled": np.sum(w * (np.sign(mu) == np.sign(ml))) / W,
        "sign_shrunk": np.sum(w * (np.sign(shr) == np.sign(ml))) / W,
        "mu": mu,
        "tau2": tau2,
        "sw2": sw2,
        "mean_B": np.sum(w * B) / W,
    }
    # early-late correlation of patient means (patient-weighted)
    cm = me - mbar
    mlb = np.sum(w * ml) / W
    cl = ml - mlb
    den = math.sqrt(max(np.sum(w * cm**2) * np.sum(w * cl**2), 1e-300))
    out["r_early_late"] = float(np.sum(w * cm * cl) / den)
    out["d_pp_minus_pooled"] = out["mse_pp"] - out["mse_pooled"]
    out["d_shrunk_minus_pooled"] = out["mse_shrunk"] - out["mse_pooled"]
    out["d_pp_minus_shrunk"] = out["mse_pp"] - out["mse_shrunk"]
    out["rel_shrunk_vs_pooled"] = out["mse_shrunk"] / out["mse_pooled"] - 1.0
    return out


def patient_specificity(ne, se, sse, nl, sl, ssl, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """Patient-bootstrap (resampling patients with replacement; every quantity including the pooled mean
    and the EB prior is refitted in each replicate) of per-patient-only vs pooled-only vs EB-shrunk
    predictions of each patient's late-half event responses."""
    ne, se, sse, nl, sl, ssl = (np.asarray(x, float) for x in (ne, se, sse, nl, sl, ssl))
    P = len(ne)
    if P < 5:
        return {"n_patients": P}
    est = _spec_stats(ne, se, sse, nl, sl, ssl, np.ones(P))
    rng = np.random.default_rng(seed)
    reps = {k: np.empty(n_boot) for k in est}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, P, P), minlength=P).astype(float)
        r = _spec_stats(ne, se, sse, nl, sl, ssl, w)
        for k in est:
            reps[k][b] = r[k]
    a = (1 - level) / 2
    out = {"n_patients": P, "n_events_early": int(ne.sum()), "n_events_late": int(nl.sum())}
    for k, v in est.items():
        rr = reps[k][np.isfinite(reps[k])]
        out[k] = float(v)
        out[k + "_lo"] = float(np.quantile(rr, a)) if len(rr) > 10 else np.nan
        out[k + "_hi"] = float(np.quantile(rr, 1 - a)) if len(rr) > 10 else np.nan
    return out
