"""
Matched construction ladder (extended analysis A): correlational-KG (CKG) statistics under the pooling,
multiple-testing, negative-control calibration and instantiation of the feedback-robust construction.

Every rung of the ladder shares the 176 concept-level candidate relations, the 38 null row maps of
``clbench.edges.typed_temporal`` (transplant / circular shift, as drawn by script 47), the pointer rule
of ``clbench.kg_v2`` for instantiation and one false-edge metric. The functions here are the pieces
that do not already exist:

* pooled lagged cross-correlation with patient fixed effects (within-case centring, per-case sufficient
  statistics, patient score bootstrap, Bonferroni over lags);
* pooled fixed-effects OLS Granger with the lag bins of script 47's ``granger_bh_pvals`` (patient score
  bootstrap, joint Wald over the source block and the 1-df exposure-weighted ``omega`` test of ours);
* a per-patient saturated hazard (2 x 2 G-test on the patient's at-risk rows, our exposure definition);
* admission (BH within family, rule (ii) against the other null replicates), instantiation counts for all
  candidates, false-edge summaries (replicate bootstrap and two-stage bootstrap), presence-matched
  replication and density-matched top-k selection.

All functions are numpy-only and deterministic given their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from clbench import kg_v2 as kv
from clbench.edges import typed_temporal as tt
from clbench.edges.co_evidence import co_evidence_relation
from clbench.edges.granger import granger_precedes_score
from clbench.edges.temporal import is_directed_temporal_edge, temporal_relation

FAMILIES: tuple[str, ...] = ("association", "policy", "response")
LAG_WINDOW: dict[str, tuple[int, int]] = {"association": (2, 40), "policy": (1, 40), "response": (5, 40)}
SOURCE_LAG0: dict[str, int] = {"association": tt.ASSOC_LAG0, "policy": tt.POLICY_LAG0}
# Source lag bins of the pooled Granger regression: the bins of script 47 ``_lag_feats`` (lag 2, 3-4, 5-12,
# 13-40) for association; shifted to start at lag 1 for policy; the response window for action sources.
SRC_BINS: dict[str, tuple[tuple[int, int], ...]] = {
    "association": ((2, 2), (3, 4), (5, 12), (13, 40)),
    "policy": ((1, 1), (2, 4), (5, 12), (13, 40)),
    "response": ((5, 12), (13, 40)),
}
OWN_BINS: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (3, 4), (5, 12), (13, 40))
MIN_HISTORY: int = 40  # rows with full lag history, as in ``_lag_feats`` (t >= 40)
CKG_MIN_XCORR: float = 0.15
CKG_ALPHA: float = 0.05
CKG_MIN_JACCARD: float = 0.10


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def action_dirs() -> list[str]:
    return [f"{a}_{d}" for a in tt.ACTIONS for d in tt.DIRS]


def action_index(ad: str) -> tuple[int, int]:
    a, d = ad.rsplit("_", 1)
    return tt.ACTIONS.index(a), tt.DIRS.index(d)


def candidates() -> list[tuple[str, str, str]]:
    """(family, source, outcome) of the 176 concept-level candidates (48 + 64 + 64), fixed order."""
    out = [("association", tt.CONCEPTS[a], tt.CONCEPTS[b]) for a, b in tt.cross_vital_pairs()]
    out += [("policy", tt.CONCEPTS[a], ad) for a in range(len(tt.CONCEPTS)) for ad in action_dirs()]
    out += [("response", ad, tt.CONCEPTS[b]) for ad in action_dirs() for b in range(len(tt.CONCEPTS))]
    return out


def key(family: str, source: str, outcome: str) -> str:
    """Same key as script 47 ``edge_key`` / script 50 ``null_admissions``."""
    return f"{family}|{source}->{outcome}"


def parse_key(k: str) -> tuple[str, str, str]:
    fam, rest = k.split("|")
    src, out = rest.split("->")
    return fam, src, out


# ---------------------------------------------------------------------------
# Cohort views and series
# ---------------------------------------------------------------------------


def sub_cohort(co: tt.Cohort, mask: np.ndarray | None) -> tt.Cohort:
    """The cohort restricted to ``mask`` rows (a contiguous block per case, e.g. a half); every case keeps
    its index (cases with no rows get T = 0)."""
    if mask is None:
        return co
    T = np.bincount(co.case[mask], minlength=co.n_cases)
    return tt.Cohort(case=co.case[mask], step=co.step[mask], S=co.S[mask], dose=co.dose[mask], chg=co.chg[mask],
                     vital_z=co.vital_z[mask], anes_start=co.anes_start, T=T, ids=list(co.ids))


def concept_on(S: np.ndarray, c: int) -> np.ndarray:
    return (S[:, c] == 1).astype(np.float64)


def change_series(chg: np.ndarray, ad: str) -> np.ndarray:
    ai, di = action_index(ad)
    return (chg[:, ai, di] > 0).astype(np.float64)


def source_series(family: str, source: str, S_src: np.ndarray, chg_src: np.ndarray) -> np.ndarray:
    if family == "response":
        return change_series(chg_src, source)
    return concept_on(S_src, tt.CONCEPTS.index(source))


def target_series(family: str, outcome: str, S: np.ndarray, chg: np.ndarray) -> np.ndarray:
    if family == "policy":
        return change_series(chg, outcome)
    return concept_on(S, tt.CONCEPTS.index(outcome))


def exposure_indicator(family: str, source: str, S_src: np.ndarray, chg_src: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Our exposure: a concept source on at the source lag; an action source changed in the response
    window [5, 40] (``typed_temporal`` SOURCE_STATE_COL / response blocks)."""
    if family == "response":
        ai, di = action_index(source)
        return tt.lagged_count(chg_src[:, ai, di], cs, *LAG_WINDOW["response"]) > 0
    return tt.lag_value(S_src[:, tt.CONCEPTS.index(source)], cs, SOURCE_LAG0[family], -1) == 1


def evidence_exposure(family: str, source: str, S_src: np.ndarray, chg_src: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """Exposure as the pointer rule sees it: source on at the source lag or a source onset inside the lag
    window (``source_block`` columns 1-4); for an action source, a change in [5, 40]."""
    if family == "response":
        return exposure_indicator(family, source, S_src, chg_src, cs)
    blk = tt.source_block(S_src[:, tt.CONCEPTS.index(source)], cs, SOURCE_LAG0[family])
    return blk[:, 1:].any(axis=1)


# ---------------------------------------------------------------------------
# Pooled lagged cross-correlation with patient fixed effects
# ---------------------------------------------------------------------------


def xcorr_suff(x: np.ndarray, y: np.ndarray, co: tt.Cohort, lags: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-case sufficient statistics of the pooled within-case-centred lagged cross-correlation
    r_k = sum_p sum_t xc_p(t) yc_p(t+k) / sqrt(sum_p sum_t xc_p^2 sum_p sum_t yc_p^2), both rows of a
    product in the same case (the fixed-effects form of CKG's per-patient ``temporal_relation``).
    Returns N (n_cases, L), Dx (n_cases,), Dy (n_cases,)."""
    lags = np.asarray(lags, dtype=np.int64)
    if (lags < 0).any():  # N_k(x, y) = N_{-k}(y, x)
        neg = lags < 0
        Nn, Dy_, Dx_ = xcorr_suff(y, x, co, -lags[neg])
        Np, Dx_, Dy_ = xcorr_suff(x, y, co, lags[~neg])
        N = np.zeros((co.n_cases, len(lags)))
        N[:, neg], N[:, ~neg] = Nn, Np
        return N, Dx_, Dy_
    case, n_cases, T = co.case, co.n_cases, co.T
    start = co.case_start
    n = len(x)
    Tf = np.maximum(T, 1).astype(np.float64)
    sx = np.bincount(case, weights=x, minlength=n_cases)
    sy = np.bincount(case, weights=y, minlength=n_cases)
    xm, ym = sx / Tf, sy / Tf
    Dx = np.bincount(case, weights=x * x, minlength=n_cases) - sx * sx / Tf
    Dy = np.bincount(case, weights=y * y, minlength=n_cases) - sy * sy / Tf
    cx = np.concatenate([[0.0], np.cumsum(x)])
    cy = np.concatenate([[0.0], np.cumsum(y)])
    nz_x, nz_y = np.nonzero(x)[0], np.nonzero(y)[0]
    use_x = len(nz_x) <= len(nz_y)
    end = start + T
    N = np.zeros((n_cases, len(lags)))
    for j, k in enumerate(lags):
        k = int(k)
        if use_x:
            i = nz_x[nz_x + k < n]
            jj = i + k
            ok = case[jj] == case[i]
            i, jj = i[ok], jj[ok]
        else:
            jj = nz_y[nz_y - k >= 0]
            i = jj - k
            ok = case[i] == case[jj]
            i, jj = i[ok], jj[ok]
        A = np.bincount(case[i], weights=x[i] * y[jj], minlength=n_cases)
        nk = np.maximum(T - k, 0).astype(np.float64)
        Bx = cx[np.maximum(end - k, start)] - cx[start]  # sum x over [start, end - k)
        By = cy[end] - cy[np.minimum(start + k, end)]  # sum y over [start + k, end)
        N[:, j] = A - ym * Bx - xm * By + nk * xm * ym
    return N, Dx, Dy


def xcorr_test(N: np.ndarray, Dx: np.ndarray, Dy: np.ndarray, W: np.ndarray, lags: np.ndarray) -> dict:
    """Pooled r_k with patient-bootstrap SEs (multiplicities W), per-lag z, Bonferroni-over-lags p of the
    most significant lag; ``stat`` = max_k z_k^2 is the rule-(ii) statistic; ``lag_peak`` / ``r_peak`` is
    CKG's own choice (largest |r|)."""
    L = len(lags)
    den = float(np.sqrt(max(Dx.sum(), 0.0) * max(Dy.sum(), 0.0)))
    r = N.sum(axis=0) / den if den > 0 else np.zeros(L)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = (W @ N) / np.sqrt(np.maximum((W @ Dx) * (W @ Dy), 1e-300))[:, None]
    se = np.std(rb, axis=0, ddof=1)
    z = np.where(se > 0, r / np.maximum(se, 1e-300), 0.0)
    p_lag = 2 * stats.norm.sf(np.abs(z))
    jz = int(np.argmax(np.abs(z)))
    jr = int(np.argmax(np.abs(r)))
    return {"r_peak": float(r[jr]), "lag_peak": int(lags[jr]), "r": float(r[jz]), "lag": int(lags[jz]),
            "se": float(se[jz]), "z": float(z[jz]), "stat": float(z[jz] ** 2),
            "p": float(min(1.0, L * p_lag.min())) if den > 0 else 1.0, "sign": int(np.sign(r[jz])) or 1}


# ---------------------------------------------------------------------------
# Pooled fixed-effects OLS Granger
# ---------------------------------------------------------------------------


def lag_feats(x: np.ndarray, cs: np.ndarray, bins: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.column_stack([tt.lagged_count(x, cs, k0, k1) for k0, k1 in bins]).astype(np.float64)


def granger_fe(y: np.ndarray, x: np.ndarray, co: tt.Cohort, src_bins: tuple[tuple[int, int], ...], W: np.ndarray,
               expo: np.ndarray, min_events: int = tt.MIN_EXPOSED_EVENTS,
               min_patients: int = tt.MIN_EVENT_PATIENTS, Z: np.ndarray | None = None) -> dict:
    """Pooled OLS of y_t on its own lag bins and the source lag bins with patient fixed effects
    (within-case demeaning), rows with >= MIN_HISTORY steps of history. Joint Wald over the source block
    with the patient score-bootstrap covariance (``tt.score_bootstrap``), the classical F of the block
    (iid, for information), and the 1-df test of omega = mean source-lag pattern over exposed rows times
    the source coefficients (our primary test form). Testability as ours. ``Z`` (n, q): optional
    adjustment covariates entering both the restricted and the unrestricted model (e.g. the phase
    columns of the hazard model)."""
    cs, case = co.cs, co.case
    step = np.arange(len(y)) - cs
    use = step >= MIN_HISTORY
    own = lag_feats(y, cs, OWN_BINS)[use]
    if Z is not None:
        own = np.column_stack([own, np.asarray(Z, dtype=np.float64)[use]])
    src = lag_feats(x, cs, src_bins)[use]
    yy = y[use]
    cu = case[use]
    E = expo[use]
    n_cases = co.n_cases
    cnt = np.bincount(cu, minlength=n_cases).astype(np.float64)
    nev = float(np.sum(yy[E]))
    npat = int(len(np.unique(cu[E & (yy > 0)])))
    p_own, p_src = own.shape[1], src.shape[1]
    X = np.column_stack([own, src])
    p_all = X.shape[1]

    def demean(v: np.ndarray) -> np.ndarray:
        m = np.bincount(cu, weights=v, minlength=n_cases) / np.maximum(cnt, 1)
        return v - m[cu]

    Xc = np.empty_like(X)
    for j in range(p_all):
        Xc[:, j] = demean(X[:, j])
    yc = demean(yy)
    keep = np.ptp(Xc, axis=0) > 0
    out = {"n_rows": int(use.sum()), "n_exposed_events": nev, "n_event_patients": npat, "testable": False,
           "chi2": 0.0, "df": 0, "p_joint": 1.0, "F": 0.0, "p_F": 1.0, "omega": 0.0, "se_omega": float("nan"),
           "z_omega": 0.0, "p_omega": 1.0, "stat": 0.0, "coefs": [0.0] * p_src, "sign": 1}
    src_idx = [j for j in range(p_own, p_all) if keep[j]]
    if yc.std() == 0 or not src_idx:
        return out
    Xk = Xc[:, keep]
    H = Xk.T @ Xk + 1e-9 * np.eye(int(keep.sum()))
    beta_k = np.linalg.solve(H, Xk.T @ yc)
    resid = yc - Xk @ beta_k
    rss1 = float(resid @ resid)
    own_keep = [j for j in range(p_own) if keep[j]]
    if own_keep:
        X0 = Xc[:, own_keep]
        b0 = np.linalg.solve(X0.T @ X0 + 1e-9 * np.eye(len(own_keep)), X0.T @ yc)
        r0 = yc - X0 @ b0
    else:
        r0 = yc
    rss0 = float(r0 @ r0)
    n_used = int(len(np.unique(cu)))
    dfd = len(yc) - n_used - int(keep.sum())
    m = len(src_idx)
    F = ((rss0 - rss1) / m) / (rss1 / dfd) if dfd > 0 and rss1 > 0 else 0.0
    p_F = float(stats.f.sf(F, m, dfd)) if dfd > 0 and np.isfinite(F) else 1.0
    G = np.zeros((n_cases, int(keep.sum())))
    for j in range(G.shape[1]):
        G[:, j] = np.bincount(cu, weights=resid * Xk[:, j], minlength=n_cases)
    beta = np.zeros(p_all)
    beta[keep] = beta_k
    fit = tt.Fit(beta, H, G, keep, True, 1)
    draws = tt.score_bootstrap(fit, W)
    b = beta[src_idx]
    Db = draws[:, src_idx] - b[None, :]
    V = np.atleast_2d(np.cov(Db, rowvar=False))
    rank = int(np.linalg.matrix_rank(V))
    chi2 = float(b @ np.linalg.pinv(V) @ b)
    p_joint = float(stats.chi2.sf(chi2, max(rank, 1)))
    src_all = list(range(p_own, p_all))
    wv = src[E].mean(axis=0) if E.any() else np.zeros(p_src)
    om = float(wv @ beta[src_all])
    om_d = draws[:, src_all] @ wv
    se = float(np.std(om_d, ddof=1))
    ok = nev >= min_events and npat >= min_patients and se > 0
    z = om / se if ok else 0.0
    out.update({"testable": bool(ok), "chi2": chi2, "df": rank, "p_joint": p_joint if ok else 1.0, "F": float(F),
                "p_F": p_F, "omega": om, "se_omega": se, "z_omega": float(z),
                "p_omega": float(2 * stats.norm.sf(abs(z))) if ok else 1.0, "stat": float(z * z),
                "coefs": beta[src_all].tolist(), "sign": int(np.sign(om)) or 1})
    return out


# ---------------------------------------------------------------------------
# Per-patient saturated hazard (rung 7)
# ---------------------------------------------------------------------------


def hazard_tables(family: str, source: str, outcome: str, co: tt.Cohort, S_src: np.ndarray, chg_src: np.ndarray
                  ) -> np.ndarray:
    """(n_cases, 4) counts [n11, n10, n01, n00] of (exposed, event) over the family's at-risk rows, with the
    pointer-rule exposure (``evidence_exposure``): the per-patient saturated hazard of ours."""
    cs = co.cs
    if family == "policy":
        ai, di = action_index(outcome)
        rows = tt.policy_rows(co, ai, di)
        y = co.chg[rows, ai, di] > 0
    else:
        b = tt.CONCEPTS.index(outcome)
        on_b, risk = tt.onsets_and_risk(co.S[:, b], cs)
        rows = np.nonzero(risk)[0]
        y = on_b[rows]
    E = evidence_exposure(family, source, S_src, chg_src, cs)[rows]
    c = co.case[rows]
    n = co.n_cases
    return np.column_stack([np.bincount(c, weights=(y & E), minlength=n), np.bincount(c, weights=(~y & E), minlength=n),
                            np.bincount(c, weights=(y & ~E), minlength=n), np.bincount(c, weights=(~y & ~E), minlength=n)])


def gtest_p(tab: np.ndarray) -> np.ndarray:
    """Likelihood-ratio (G) test of independence of a 2 x 2 table per row; p = 1 when a margin is zero."""
    tab = np.asarray(tab, dtype=np.float64)
    n = tab.sum(axis=1)
    r1 = tab[:, 0] + tab[:, 1]
    c1 = tab[:, 0] + tab[:, 2]
    ok = (n > 0) & (r1 > 0) & (r1 < n) & (c1 > 0) & (c1 < n)
    p = np.ones(len(tab))
    if not ok.any():
        return p
    t = tab[ok]
    nn = n[ok][:, None]
    rows = np.column_stack([r1[ok], r1[ok], nn[:, 0] - r1[ok], nn[:, 0] - r1[ok]])
    cols = np.column_stack([c1[ok], nn[:, 0] - c1[ok], c1[ok], nn[:, 0] - c1[ok]])
    expct = rows * cols / nn
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(t > 0, t * np.log(t / expct), 0.0)
    G = 2 * term.sum(axis=1)
    p[ok] = stats.chi2.sf(np.maximum(G, 0.0), 1)
    return p


# ---------------------------------------------------------------------------
# Per-patient CKG statistics (rungs 1-3) on one case
# ---------------------------------------------------------------------------


def ckg_case_stats(C_src: np.ndarray, C_tgt: np.ndarray, pairs: list[tuple[int, int]], unordered: bool,
                   stride_sec: float = tt.STEP_SEC) -> dict[str, np.ndarray]:
    """CKG's three per-patient statistics on ``pairs`` (ordered (a, b)): peak cross-correlation and its
    signed lag (``temporal_relation``; a real record is tested once per unordered pair, the direction given
    by the lag sign, as ``kg_construct`` / script 47 ``v1_edges`` do; a null source is tested in the a -> b
    direction), Granger min-p over lags {1, 4, 12} (``granger_precedes_score``) and the Jaccard overlap
    (``co_evidence_relation``). Edges follow the published thresholds."""
    n = len(pairs)
    xpk = np.zeros(n, np.float32)
    xlag = np.zeros(n, np.int16)
    xedge = np.zeros(n, bool)
    gp = np.ones(n, np.float32)
    gstat = np.zeros(n, bool)
    jac = np.zeros(n, np.float32)
    cache: dict[tuple[int, int], tuple[float, float]] = {}
    for i, (a, b) in enumerate(pairs):
        if unordered:
            k = (min(a, b), max(a, b))
            if k not in cache:
                rel = temporal_relation(C_src[:, k[0]], C_tgt[:, k[1]], 0.5, 0.5, window_stride_sec=stride_sec)
                cache[k] = (rel.peak_xcorr, rel.lag_sec)
            pk, lag = cache[k]
            lag = lag if a <= b else -lag  # a -> b when the lag of (min, max) points from a to b
        else:
            rel = temporal_relation(C_src[:, a], C_tgt[:, b], 0.5, 0.5, window_stride_sec=stride_sec)
            pk, lag = rel.peak_xcorr, rel.lag_sec
        xpk[i] = pk
        xlag[i] = int(round(lag / stride_sec))
        xedge[i] = pk >= CKG_MIN_XCORR and lag >= 15.0  # is_directed_temporal_edge and a precedes b
        g = granger_precedes_score(C_src[:, a], C_tgt[:, b])
        gp[i] = g.min_p_value
        gstat[i] = g.status == "ok"
        jac[i] = co_evidence_relation(C_src[:, a], C_tgt[:, b], 0.5, 0.5).jaccard
    return {"xcorr_peak": xpk, "xcorr_lag": xlag, "xcorr_edge": xedge, "granger_p": gp, "granger_ok": gstat,
            "jaccard": jac}


def ckg_edges(st: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Edge indicators (rows x pairs) from stacked ``ckg_case_stats`` arrays."""
    x = st["xcorr_edge"]
    g = st["granger_ok"] & (st["granger_p"] < CKG_ALPHA)
    j = st["jaccard"] >= CKG_MIN_JACCARD
    return {"xcorr": x, "granger": g, "jaccard": j, "union": x | g, "union_jaccard": x | g | j}


def ckg_weights(st: dict[str, np.ndarray]) -> np.ndarray:
    """CKG's own edge weight per pair: normalised peak cross-correlation of a precedes edge, 1 - min p of a
    Granger-precedes edge, the larger when both exist; 0 when the pair has no CKG edge."""
    e = ckg_edges(st)
    wx = np.where(e["xcorr"], st["xcorr_peak"], 0.0)
    wg = np.where(e["granger"], 1.0 - st["granger_p"], 0.0)
    return np.maximum(wx, wg).astype(np.float64)


def topk_mask(weights: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest positive weights per row (ties by column order)."""
    n, m = weights.shape
    out = np.zeros((n, m), bool)
    if k <= 0:
        return out
    order = np.argsort(-weights, axis=1, kind="stable")[:, :k]
    rows = np.repeat(np.arange(n), order.shape[1])
    cols = order.ravel()
    out[rows, cols] = weights[rows, cols] > 0
    return out


def choose_k(weights: np.ndarray, target: float, k_max: int) -> tuple[int, dict[int, float]]:
    """Integer k whose mean kept edges per graph is closest to ``target``; all means returned."""
    means = {k: float(topk_mask(weights, k).sum(axis=1).mean()) for k in range(0, k_max + 1)}
    k = min(means, key=lambda kk: (abs(means[kk] - target), kk))
    return k, means


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def bh_within_family(p: dict[str, float], q: float) -> set[str]:
    out: set[str] = set()
    for fam in FAMILIES:
        keys = [k for k in p if k.startswith(fam + "|")]
        if not keys:
            continue
        rej = tt.bh(np.array([p[k] for k in keys]), q)
        out |= {k for k, r in zip(keys, rej) if r}
    return out


def admit(tables: dict[tuple[str, int], dict[str, tuple[float, float]]], variant: tuple[str, int], q: float,
          calibrate: bool, null_variants: list[tuple[str, int]]) -> set[str]:
    """Rule (i): BH at q within family on ``tables[variant]`` (key -> (p, stat)); rule (ii) when
    ``calibrate``: stat > max of the same key's stat over the other null variants (and > 0), the
    leave-one-out rule of script 50 ``null_admissions``."""
    tab = tables[variant]
    adm = bh_within_family({k: v[0] for k, v in tab.items()}, q)
    if not calibrate:
        return adm
    out = set()
    others = [v for v in null_variants if v != variant]
    for k in adm:
        mx = max([tables[v][k][1] for v in others if k in tables[v]] + [0.0])
        if tab[k][1] > mx:
            out.add(k)
    return out


# ---------------------------------------------------------------------------
# Instantiation (pointer rule) for all candidates
# ---------------------------------------------------------------------------


def instance_matrix(co: tt.Cohort, cands: list[tuple[str, str, str]], S_src: np.ndarray, chg_src: np.ndarray,
                    calib, cases: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For every case and candidate: whether the pointer rule finds >= 1 evidence pair (source from the
    null-mapped or real ``S_src`` / ``chg_src``, target from the real arrays) and the number of node-level
    edges (distinct tier-node endpoint pairs; script 50 ``null_counts``)."""
    n = co.n_cases
    inst = np.zeros((n, len(cands)), bool)
    node = np.zeros((n, len(cands)), np.int16)
    cidx = {c: i for i, c in enumerate(tt.CONCEPTS)}
    for c in (range(n) if cases is None else cases):
        s0, T = int(co.case_start[c]), int(co.T[c])
        if T == 0:
            continue
        S, Sn = co.S[s0:s0 + T], S_src[s0:s0 + T]
        chg, chgn = co.chg[s0:s0 + T], chg_src[s0:s0 + T]
        tiers_null = tiers_real = None
        for j, (fam, src, out) in enumerate(cands):
            if fam == "association":
                p = kv.association_pairs(Sn[:, cidx[src]], S[:, cidx[out]])
                if len(p):
                    tiers_null = tiers_null or kv.case_tiers(Sn, calib)
                    tiers_real = tiers_real or kv.case_tiers(S, calib)
                    ta, tb = tiers_null[src], tiers_real[out]
                    node[c, j] = len({(ta[s], tb[t]) for s, t in zip(p.src, p.tgt)})
            elif fam == "policy":
                ai, di = action_index(out)
                p = kv.policy_pairs(Sn[:, cidx[src]], np.nonzero(chg[:, ai, di])[0])
                if len(p):
                    tiers_null = tiers_null or kv.case_tiers(Sn, calib)
                    node[c, j] = len({tiers_null[src][s] for s in p.src})
            else:
                ai, di = action_index(src)
                p = kv.response_pairs(np.nonzero(chgn[:, ai, di])[0], S[:, cidx[out]])
                if len(p):
                    tiers_real = tiers_real or kv.case_tiers(S, calib)
                    node[c, j] = len({tiers_real[out][t] for t in p.tgt})
            inst[c, j] = len(p) > 0
    return inst, node


# ---------------------------------------------------------------------------
# False-edge summaries and replication
# ---------------------------------------------------------------------------


def fe_summary(per_rep: list[np.ndarray], n_boot: int = 2000, seed: int = 0, n_two: int = 1000, seed_two: int = 1,
               n_inner: int = 200) -> dict:
    """FE-bar over replicates (each an (n_graphs,) count array): mean, percentile bootstrap over the
    replicate means (the paper's CI), two-stage bootstrap (replicates, then patients within replicate),
    share of (graph, replicate) pairs with >= 1 false edge, replicates with any."""
    R = len(per_rep)
    means = np.array([float(r.mean()) for r in per_rep])
    rng = np.random.default_rng(seed)
    bm = means[rng.integers(0, R, size=(n_boot, R))].mean(axis=1)
    rng2 = np.random.default_rng(seed_two)
    inner = np.empty((R, n_inner))
    for i, r in enumerate(per_rep):
        idx = rng2.integers(0, len(r), size=(n_inner, len(r)))
        inner[i] = r[idx].mean(axis=1)
    pick_r = rng2.integers(0, R, size=(n_two, R))
    pick_i = rng2.integers(0, n_inner, size=(n_two, R))
    two = inner[pick_r, pick_i].mean(axis=1)
    return {"mean": float(means.mean()), "ci_replicate": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "ci_two_stage": [float(np.percentile(two, 2.5)), float(np.percentile(two, 97.5))],
            "per_rep_mean": means.tolist(), "reps_with_any": int(np.sum(means > 0)), "n_reps": R,
            "frac_graphs_any": float(np.mean([float((r > 0).mean()) for r in per_rep])),
            "max_any_graph": float(max(float(r.max()) for r in per_rep)) if per_rep else 0.0}


def replication(E1: list[set], E2: list[set], P1: list[set], P2: list[set], W: np.ndarray) -> dict:
    """Rep = sum |E1 & E2| / sum |E1|; presence-matched chance Rep0 of App. relations-replication
    (Eq. replication); patient-bootstrap CIs (multiplicities W) of both and of their difference."""
    num = np.array([len(a & b) for a, b in zip(E1, E2)], float)
    den = np.array([len(a) for a in E1], float)
    ch = np.array([len(a) * len(b) * len(p1 & p2) / (len(p1) * len(p2)) if len(p1) and len(p2) else 0.0
                   for a, b, p1, p2 in zip(E1, E2, P1, P2)], float)
    est, lo, hi = tt.patient_bootstrap_ratio(num, den, W)
    c0, clo, chi = tt.patient_bootstrap_ratio(ch, den, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (W @ num) / (W @ den) - (W @ ch) / (W @ den)
    d = d[np.isfinite(d)]
    return {"n_edges_h1": int(den.sum()), "edges_per_patient_h1": float(den.mean()),
            "replication": [est, lo, hi], "chance": [c0, clo, chi],
            "diff": [est - c0, float(np.percentile(d, 2.5)) if d.size else float("nan"),
                     float(np.percentile(d, 97.5)) if d.size else float("nan")]}
