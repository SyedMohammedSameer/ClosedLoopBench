"""Unit tests for the closed-loop measurement benchmark estimators (synthetic data, known signs)."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest

from clbench.eval import closed_loop_bench as clb

CFG = clb.SCALES["2s"]
# Coarser grid for the slow graph methods (PCMCI+, VAR-LiNGAM): 10 s steps, 1 min of lags.
CFG_GRAPH = dataclasses.replace(CFG, reg_every=5, pcmci_tau_max=6, lingam_p=6)


def _first_order(target: np.ndarray, rng, step: float, tau_s: float, noise: float, y0: float) -> np.ndarray:
    y = np.empty(len(target))
    y[0] = y0
    a = step / tau_s
    for i in range(1, len(target)):
        y[i] = y[i - 1] + a * (target[i - 1] - y[i - 1]) + noise * rng.normal()
    return y


def synth_open_loop(pid: str, seed: int, n: int = 3600, gain: float = -4.0, mean_gap_s: float = 600.0):
    """Propofol target steps at random (open-loop) times; MAP follows with first-order dynamics and a
    negative gain (mmHg per ug/mL), time constant 40 s. HR and EtCO2 are independent AR(1) noise."""
    rng = np.random.default_rng(seed)
    step = 2.0
    t = np.arange(n) * step + step / 2
    T = n * step
    k = rng.poisson(T / mean_gap_s)
    ev_t = np.sort(rng.uniform(300, T - 300, k))
    dose, before, after = 3.0, [], []
    for _ in ev_t:
        new = float(np.clip(dose + rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0), 1.5, 6.0))
        if abs(new - dose) < 0.1:
            new = dose + 0.5
        before.append(dose)
        after.append(new)
        dose = new
    evs = clb.EventSet.build(ev_t, before, after, clb.ACTIONS["propofol"])
    d = evs.dose_at(t)
    mp = _first_order(80 + gain * (d - 3.0), rng, step, 40.0, 0.4, 80.0)
    hr = _first_order(np.full(n, 70.0), rng, step, 60.0, 0.3, 70.0)
    co2 = _first_order(np.full(n, 38.0), rng, step, 60.0, 0.1, 38.0)
    return clb.CaseData(pid, t, step, {"map": mp, "hr": hr, "etco2": co2}, {"propofol": evs})


def synth_policy(pid: str, seed: int, n: int = 3600):
    """Closed-loop *policy only*: remifentanil is raised when MAP has risen over the last 2 min and
    lowered when it has fallen; remifentanil has no effect on MAP (MAP is a slow random walk)."""
    rng = np.random.default_rng(seed)
    step = 2.0
    t = np.arange(n) * step + step / 2
    mp = 80 + np.cumsum(rng.normal(0, 0.35, n))
    K = int(120 / step)
    ev_t, before, after = [], [], []
    dose, last = 2.0, -1e9
    for i in range(K + 1, n):
        if t[i] - last < 240:
            continue
        tr = mp[i - 1] - mp[i - 1 - K]
        if abs(tr) > 4 and rng.random() < 0.3:
            new = float(np.clip(dose + np.sign(tr) * 0.5, 0.5, 6.0))
            if new != dose:
                ev_t.append(t[i] - 0.5)
                before.append(dose)
                after.append(new)
                dose, last = new, t[i]
    evs = clb.EventSet.build(ev_t, before, after, clb.ACTIONS["remifentanil"])
    hr = 70 + rng.normal(0, 1, n)
    co2 = 38 + rng.normal(0, 0.3, n)
    return clb.CaseData(pid, t, step, {"map": mp, "hr": hr, "etco2": co2}, {"remifentanil": evs})


@pytest.fixture(scope="module")
def open_loop_cases():
    return [synth_open_loop(f"s{i}", 100 + i) for i in range(24)]


def _pooled(cases, cfg, methods, n_boot=100, **kw):
    agg = clb.Aggregator([c.pid for c in cases], cfg, n_boot=n_boot, seed=0)
    for c in cases:
        agg.add(c.pid, "real", clb.analyse_case(c, cfg, methods=methods, **kw))
    pooled, pp = agg.finalize(lingam_boot=12, lingam_max_rows=20_000)
    return pooled, pp


def _get(pooled, method, rid, h=0.0):
    q = pooled[(pooled.method == method) & (pooled.rid == rid) & (pooled.h == h)]
    assert len(q) == 1, (method, rid, h, len(q))
    return q.iloc[0]


# ── building blocks ────────────────────────────────────────────────────────────────────────────────


def test_series_index_matches_numpy():
    rng = np.random.default_rng(0)
    t = np.arange(200) * 2.0 + 1
    y = 3 + 0.2 * t + rng.normal(size=200)
    y[[5, 17, 40]] = np.nan
    S = clb.SeriesIndex(t, y)
    lo, hi, anchor = 30.0, 150.0, 160.0
    m = (t >= lo) & (t <= hi) & np.isfinite(y)
    assert np.isclose(S.mean(lo, hi), y[m].mean())
    a, b = S.linfit(anchor, lo, hi, 5)
    b_ref, a_ref = np.polyfit(t[m] - anchor, y[m], 1)
    assert np.isclose(a, a_ref) and np.isclose(b, b_ref)


def test_binned_lags_and_increments():
    u = np.zeros(20)
    u[5] = 2.0
    B = clb.binned_lags(u, [(1, 1), (2, 4)])
    assert B[6, 0] == 2.0 and B[5, 0] == 0.0
    assert B[7, 1] == 2.0 and B[9, 1] == 2.0 and B[10, 1] == 0.0 and B[6, 1] == 0.0
    ev = clb.EventSet.build([10.5, 11.0, 30.2], [1, 2, 3], [2, 3, 2.5], clb.ACTIONS["propofol"])
    tg = np.arange(20) * 2.0 + 1.0  # windows [2i, 2i+2)
    inc = clb.increment_series(ev, tg, 2.0, ev.usable)
    assert inc[5] == 2.0 and np.isclose(inc[15], -0.5) and inc.sum() == 1.5


def test_event_set_excludes_starts_and_stops():
    ev = clb.EventSet.build([1, 2, 3, 4], [0, 2, 3, 2], [2, 3, 2, 0], clb.ACTIONS["remifentanil"])
    assert ev.usable.tolist() == [False, True, True, False]
    ev = clb.EventSet.build([1, 2], [0, 2], [2, 0], clb.ACTIONS["phenylephrine"])
    assert ev.usable.tolist() == [True, True]
    assert ev.dose_at(np.array([0.5, 1.5, 2.5])).tolist() == [0.0, 2.0, 0.0]


def test_xcorr_curve_matches_v1_direct_formula():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = np.roll(x, 7) + 0.3 * rng.normal(size=300)
    c = clb.xcorr_curve(x, y, 20)
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    for k in (-5, 0, 7, 12):
        ref = (xc[: len(x) - k] * yc[k:]).sum() / den if k >= 0 else (xc[-k:] * yc[: len(y) + k]).sum() / den
        assert np.isclose(c[20 + k], ref)
    val, lag = clb.xcorr_peak(c, 20)
    assert lag == 7 and val > 0.5


def test_granger_matches_statsmodels_var():
    from statsmodels.tsa.api import VAR

    rng = np.random.default_rng(2)
    n = 600
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        y[t] = 0.4 * y[t - 1] - 0.6 * x[t - 1] + 0.2 * x[t - 2] + rng.normal()
    f = clb.granger_fit(x, y, 2)
    res = VAR(np.column_stack([x, y])).fit(2, trend="c")
    ref = res.coefs[0][1, 0] + res.coefs[1][1, 0]  # y equation, x lags
    assert np.isclose(f.est_xy, ref)
    assert f.p_xy < 1e-6 and f.est_xy < 0


def test_negative_controls_preserve_steps():
    case = synth_open_loop("a", 1)
    nc = clb.nc_random_times(case, np.random.default_rng(0))
    e0, e1 = case.events["propofol"], nc.events["propofol"]
    assert len(e0.t) == len(e1.t)
    assert np.allclose(e0.after - e0.before, e1.after - e1.before)
    assert e1.t.min() >= case.t_start and e1.t.max() <= case.t_end
    other = synth_open_loop("b", 2, n=1800)  # shorter partner stream is fine; longer is truncated
    cross = clb.nc_cross_patient(other, case.events)
    assert (cross.events["propofol"].t <= other.t_end).all()


# ── estimators recover a known negative response on open-loop synthetic data ───────────────────────


def test_event_designs_recover_negative_response(open_loop_cases):
    pooled, pp = _pooled(open_loop_cases, CFG, ("naive", "matched", "kink"))
    for m in ("naive", "matched", "kink"):
        for rid in ("R03", "R04"):  # propofol up -> MAP down; propofol down -> MAP up
            row = _get(pooled, m, rid, 120.0)
            assert row.call == "correct", (m, rid, row.est, row.lo, row.hi)
    ps = clb.patient_summary(pp)
    q = ps[(ps.method == "kink") & (ps.rid == "R03") & (ps.h == 120.0)].iloc[0]
    assert q.frac_correct > 0.7


def test_xcorr_granger_varx_recover_negative_response(open_loop_cases):
    pooled, pp = _pooled(open_loop_cases, CFG, ("xcorr_v1", "granger", "varx"))
    x = _get(pooled, "xcorr_v1", "R03")
    assert x.est < 0 and x.lag_s > 0
    assert _get(pooled, "granger", "R03").call == "correct"
    for rid in ("R03", "R04"):
        assert _get(pooled, "varx", rid, 120.0).call == "correct"


@pytest.mark.skipif(importlib.util.find_spec("tigramite") is None, reason="tigramite not installed")
def test_pcmci_recovers_negative_response():
    cases = [synth_open_loop(f"p{i}", 200 + i, n=5400, mean_gap_s=300) for i in range(3)]
    signs = []
    for c in cases:
        res = clb.pcmci_case(c, CFG_GRAPH)
        val, p, det, tau = res[("propofol", "map")]
        signs.append(val < 0 and det)
    assert sum(signs) >= 2


@pytest.mark.skipif(importlib.util.find_spec("lingam") is None, reason="lingam not installed")
def test_varlingam_recovers_negative_response():
    cases = [synth_open_loop(f"l{i}", 300 + i, mean_gap_s=300) for i in range(12)]
    pooled, _ = _pooled(cases, CFG_GRAPH, ("varlingam",), lingam_resid_rows=400)
    row = _get(pooled, "varlingam", "R03")
    assert row.est < 0


# ── policy relation and the closed-loop sign trap ──────────────────────────────────────────────────


def test_policy_sign_recovered_and_response_is_null():
    cases = [synth_policy(f"q{i}", 400 + i) for i in range(30)]
    pooled, _ = _pooled(cases, CFG, ("naive", "matched", "varx"))
    assert _get(pooled, "naive", "P01").call == "correct"  # MAP rising -> remi up
    assert _get(pooled, "matched", "P01").call == "correct"
    assert _get(pooled, "varx", "P01").est > 0
    # remifentanil has no effect here; the placebo-matched design must not report the textbook sign
    assert _get(pooled, "matched", "R06", 120.0).call != "correct"
    # ... whereas the naive pre/post difference reads the policy-driven pre-trend as a *wrong-signed*
    # response (MAP keeps rising after a remifentanil increase): the closed-loop sign trap.
    assert _get(pooled, "naive", "R06", 120.0).call == "wrong"


def test_sign_call_requires_enough_cases():
    assert clb.sign_call(-1.0, -2.0, -0.5, -1, n_cases=3) == "ns_right"
    assert clb.sign_call(-1.0, -2.0, -0.5, -1, n_cases=30) == "correct"
    assert clb.sign_call(1.0, 0.5, 2.0, -1, n_cases=30) == "wrong"


# ── v2: control pool, typed estimator, EB, patient-specificity, masking, xcorr direction ───────────


def synth_inert_loop(pid: str, seed: int, n: int = 3600, thr: float = 3.0, hazard: float = 0.02):
    """Closed loop with an *inert* remifentanil: MAP is a random walk; the clinician raises (lowers) the
    target with a small per-step hazard while the 2-min MAP trend is above +thr (below -thr), with a
    240 s refractory period. Any non-null event estimate is bias."""
    rng = np.random.default_rng(seed)
    step = 2.0
    t = np.arange(n) * step + step / 2
    mp = 80 + np.cumsum(rng.normal(0, 0.35, n))
    K = int(120 / step)
    ev_t, before, after = [], [], []
    dose, last = 2.0, -1e9
    for i in range(K + 1, n):
        if t[i] - last < 240:
            continue
        tr = mp[i - 1] - mp[i - 1 - K]
        if abs(tr) > thr and rng.random() < hazard:
            new = float(np.clip(dose + np.sign(tr) * 0.5, 0.5, 6.0))
            if new != dose:
                ev_t.append(t[i] - 0.5)
                before.append(dose)
                after.append(new)
                dose, last = new, t[i]
    evs = clb.EventSet.build(ev_t, before, after, clb.ACTIONS["remifentanil"])
    hr = 70 + rng.normal(0, 1, n)
    co2 = 38 + rng.normal(0, 0.3, n)
    return clb.CaseData(pid, t, step, {"map": mp, "hr": hr, "etco2": co2}, {"remifentanil": evs})


def test_future_inaction_pool_is_biased_and_pre_only_pool_is_not():
    """The v1 placebo pool drops anchors followed by an action within max(h): with a trend-triggered
    clinician that keeps only anchors whose MAP did *not* go on to rise, so an inert drug looks
    wrong-signed. The v2 pool (eligible and untreated at t, pre-t information only) does not."""
    cases = [synth_inert_loop(f"z{i}", 500 + i) for i in range(60)]
    pooled, _ = _pooled(cases, CFG, ("naive", "matched", "kink"))
    v1 = _get(pooled, "matched_v1", "R06", 120.0)
    v2 = _get(pooled, "matched", "R06", 120.0)
    assert v1.call == "wrong" and v1.est > 0.5, (v1.est, v1.lo, v1.hi)
    assert v2.call in ("ns_right", "ns_wrong"), (v2.est, v2.lo, v2.hi)
    assert abs(v2.est) < 0.4 * abs(v1.est)
    # bias-corrected kink with the v2 pool: CI covers 0 for the inert drug
    k2 = _get(pooled, "kink", "R06", 120.0)
    assert k2.call in ("ns_right", "ns_wrong"), (k2.est, k2.lo, k2.hi)


def test_v1_options_emit_only_v1_rows():
    case = synth_inert_loop("a", 1)
    rel = clb.REL_BY_ID["R06"]
    names_v1 = {m for m, _, _ in clb.event_design_rows(case, rel, CFG, clb.BenchOptions.v1())}
    names_v2 = {m for m, _, _ in clb.event_design_rows(case, rel, CFG, clb.BenchOptions())}
    assert names_v1 <= {"naive", "matched", "kink"}
    assert {"naive_lvl", "matched_v1", "kink_clean", "kink_prealt", "kink_nobc"} <= names_v2


def test_benjamini_yekutieli_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    from clbench.edges import typed_temporal_pooled as tp

    p = np.array([0.001, 0.02, 0.03, 0.2, 0.5, 0.0004, np.nan, 0.04])
    ok = np.isfinite(p)
    ref = multipletests(p[ok], method="fdr_by")[1]
    got = tp.benjamini_yekutieli(p)
    assert np.allclose(got[ok], ref) and np.isnan(got[~ok]).all()


def test_calibrated_admission_rejects_shared_artefact():
    from clbench.edges import typed_temporal_pooled as tp

    rng = np.random.default_rng(0)
    shared = rng.normal(0, 0.1, 200)
    # artefact present in both arms (same schedule): real = null + 0 -> not admitted
    t0 = tp.calibrated_test(-1.0, -1.0 + shared, -1.0, -1.0 + shared + rng.normal(0, 0.05, 200))
    # true effect on top of the artefact -> admitted
    t1 = tp.calibrated_test(-2.0, -2.0 + shared, -1.0, -1.0 + shared + rng.normal(0, 0.05, 200))
    adm, p_by = tp.admit([t0, t1])
    assert adm.tolist() == [False, True] and t1.est == -1.0


def test_eb_shrink_recovers_heterogeneity():
    from clbench.edges import typed_temporal_pooled as tp

    rng = np.random.default_rng(1)
    theta = rng.normal(-1.0, 0.5, 400)
    se = rng.uniform(0.3, 1.5, 400)
    y = theta + rng.normal(0, se)
    r = tp.eb_shrink(y, se**2)
    assert abs(r.mu + 1.0) < 0.15 and 0.12 < r.tau2 < 0.45
    assert np.mean((r.shrunk - theta) ** 2) < np.mean((y - theta) ** 2)
    r0 = tp.eb_shrink(rng.normal(0, se), se**2)  # no heterogeneity -> heavy shrinkage
    assert np.nanmean(r0.weight) < 0.2


def _spec_sums(theta_sd: float, seed: int, P: int = 300):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(P):
        th = -1.0 + theta_sd * rng.normal()
        n = int(rng.integers(6, 16))
        y = th + rng.normal(0, 2.0, n)
        rows.append(__import__("clbench.edges.typed_temporal_pooled", fromlist=["x"]).split_half_sums(np.arange(n), y))
    return np.array(rows).T


def test_patient_specificity_detects_heterogeneity_only_when_present():
    from clbench.edges import typed_temporal_pooled as tp

    het = tp.patient_specificity(*_spec_sums(1.5, 2), n_boot=200)
    assert het["d_shrunk_minus_pooled_hi"] < 0 and het["r_early_late_lo"] > 0
    hom = tp.patient_specificity(*_spec_sums(0.0, 3), n_boot=200)
    assert hom["d_pp_minus_pooled_lo"] > 0  # per-patient-only estimates overfit
    assert hom["d_shrunk_minus_pooled_lo"] < 0.05 * hom["mse_pooled"]


def test_dose_masked_after_track_stop():
    ev = clb.EventSet.build([10.0, 20.0], [0.0, 2.0], [2.0, 3.0], clb.ACTIONS["remifentanil"], t_stop=50.0)
    d = ev.dose_at(np.array([5.0, 15.0, 40.0, 60.0]))
    assert d[:3].tolist() == [0.0, 2.0, 3.0] and np.isnan(d[3])
    ev2 = clb.EventSet.build([10.0, 20.0], [0.0, 2.0], [2.0, 0.0], clb.ACTIONS["phenylephrine"], t_stop=30.0)
    assert ev2.dose_at(np.array([60.0])).tolist() == [0.0]  # stopped at 0: nothing to mask


def test_xcorr_direction_aware_peak():
    c = np.zeros(21)
    c[10 - 4] = 0.8  # vital leads by 4 steps (policy direction)
    c[10 + 6] = -0.3  # action leads by 6 steps (response direction)
    assert clb.xcorr_peak(c, 10, "abs") == (0.8, -4)
    assert clb.xcorr_peak(c, 10, "dir", +1) == (-0.3, 6)
    assert clb.xcorr_peak(c, 10, "dir", -1) == (0.8, -4)
    assert clb.xcorr_peak(c, 10, "pos")[0] == 0.8


def test_eb_posterior_sd_includes_uncertainty_of_the_pooled_mean():
    """Regression: with tau^2 = 0 every patient is shrunk fully to mu; the posterior SD must then be the
    SD of mu, not 0 (otherwise every patient-level edge is 'admitted')."""
    from clbench.edges import typed_temporal_pooled as tp

    rng = np.random.default_rng(4)
    se = np.full(200, 1.0)
    r = tp.eb_shrink(rng.normal(0, se), se**2, mu=0.05, mu_var=0.01)
    assert r.tau2 == 0.0 and np.allclose(r.post_sd, 0.1)
    assert (np.abs(r.shrunk) > 1.96 * r.post_sd).mean() == 0.0


# ── v3 (ClosedLoopBench 1.0): multi-draw calibration, pooled sequential trial, pre-t co-interventions,
#    reproducible aggregation, bench package ──────────────────────────────────────────────────────


def test_coint_counts_pre_only_window():
    ev_other = clb.EventSet.build([50.0, 95.0, 130.0], [1.0, 2.0, 3.0], [2.0, 3.0, 2.0], clb.ACTIONS["propofol"])
    case = clb.CaseData("c", np.arange(0, 300, 2.0) + 1, 2.0, {v: np.zeros(150) for v in clb.VITALS},
                        {"propofol": ev_other, "remifentanil": clb.EventSet.empty()})
    t = np.array([100.0])
    sym = clb._coint_counts(case, "remifentanil", t, 60.0)  # v2: [40, 160] -> 2 up, 1 down
    pre = clb._coint_counts(case, "remifentanil", t, 60.0, pre_only=True)  # v3: [40, 100) -> 2 up
    assert sym.tolist() == [[2.0, 1.0]] and pre.tolist() == [[2.0, 0.0]]
    # a change exactly at t is not pre-t information
    assert clb._coint_counts(case, "remifentanil", np.array([95.0]), 60.0, pre_only=True).tolist() == [[1.0, 0.0]]


def test_calibrated_test_multi_propagates_draw_variance_and_requires_stability():
    from clbench.edges import typed_temporal_pooled as tp

    rng = np.random.default_rng(5)
    B = 200
    shared = rng.normal(0, 0.1, B)
    real = -1.0 + shared
    # eight null draws around 0 (true effect -1): stable, admitted
    nulls = [(float(m), m + shared + rng.normal(0, 0.02, B)) for m in rng.normal(0, 0.05, 8)]
    t = tp.calibrated_test_multi(-1.0, real, [e for e, _ in nulls], [r for _, r in nulls])
    assert t.n_draws == 8 and t.se_draw > 0 and t.se > t.se_boot and t.stable_frac == 1.0
    # one outlying draw shifts the mean; the between-draw SE and the stability share expose it
    bad = [(0.0, shared + rng.normal(0, 0.02, B))] * 6 + [(-2.0, -2.0 + shared)] * 2
    tb = tp.calibrated_test_multi(-1.0, real, [e for e, _ in bad], [r for _, r in bad])
    assert tb.stable_frac == 0.75 and tb.se_draw > 0.3
    adm, _ = tp.admit_multi([t, tb], min_stable=0.8)
    assert adm.tolist() == [True, False]


def test_pooled_nn_contrast_uses_other_groups_and_recovers_effect():
    rng = np.random.default_rng(2)
    nc, ne = 3000, 400
    gc = rng.integers(0, 100, nc)
    fc = rng.normal(0, 1, (nc, 2))
    yc = 2.0 * fc[:, 0] + rng.normal(0, 0.3, nc)
    ge = rng.integers(0, 100, ne)
    fe = rng.normal(1.0, 1, (ne, 2))  # shifted state: bias correction must handle it
    ye = 2.0 * fe[:, 0] - 1.5 + rng.normal(0, 0.3, ne)  # true effect -1.5
    v = clb.pooled_nn_contrast(fe, ye, ge, fc, yc, gc, k=10)[:, 0]
    assert abs(np.nanmean(v) + 1.5) < 0.1
    # a group whose own controls carry a huge offset: excluded, so the estimate does not move
    yc2 = yc + np.where(gc == ge[0], 50.0, 0.0)
    v2 = clb.pooled_nn_contrast(fe[:1], ye[:1], ge[:1], fc, yc2, gc, k=10, bias_correct=False)[0, 0]
    v1 = clb.pooled_nn_contrast(fe[:1], ye[:1], ge[:1], fc, yc, gc, k=10, bias_correct=False)[0, 0]
    assert abs(v2 - v1) < 1e-9


def _v3_pooled(cases, cfg, methods, crosses=4, n_boot=100, order=None):
    pids = [c.pid for c in cases]
    agg = clb.Aggregator(pids, cfg, n_boot=n_boot, seed=0, opts=clb.V3)
    idx = list(range(len(cases))) if order is None else order
    for i in idx:
        c = cases[i]
        cross = {a: cases[(i + 1 + 7 * k) % len(cases)].events for k, a in enumerate(clb.CROSS_ARMS[:crosses])}
        arms = clb.build_arms(c, None, seed=i, arms=("real", "nc_random") + clb.CROSS_ARMS[:crosses], cross_events=cross)
        for arm, cc in arms.items():
            agg.add(c.pid, arm, clb.analyse_case(cc, cfg, methods=methods, opts=clb.V3))
    P, pp = agg.finalize(lingam_boot=6, lingam_max_rows=20_000)
    return P, pp, agg


def test_seqtrial_recovers_open_loop_response_and_calibrated_rule_admits_it(open_loop_cases):
    """Open loop (random step times): the pooled contrast and the v3 rule recover propofol -> MAP."""
    P, _, agg = _v3_pooled(open_loop_cases, CFG, ("naive", "matched", "kink", "seqtrial", "varx"))
    for rid in ("R03", "R04"):
        r = _get(P[P.arm == "real"], "seqtrial", rid, 120.0)
        assert r.call == "correct", (rid, r.est, r.lo, r.hi)
    C = agg.calib_table
    st = C[(C.input == "real") & (C.method == "seqtrial")]
    assert set(st[st.admitted].rid) >= {"R03", "R04"}
    typed = C[(C.input == "real") & (C.method == "typed_pooled")]
    assert set(typed.base) == {clb.TYPED_RESPONSE_BASE, "varx"}
    # null inputs: no admissions (leave-one-out cross draws, random-time input)
    assert not C[C.method.isin(["seqtrial", "typed_pooled"]) & (C.input != "real")].admitted.any()


def test_aggregation_is_reproducible_across_case_order():
    cases = [synth_inert_loop(f"q{i}", 700 + i) for i in range(16)]
    meth = ("naive", "matched", "kink", "seqtrial", "xcorr_v1", "varlingam")
    cfg = dataclasses.replace(CFG, reg_every=5, lingam_p=6)
    P1, _, a1 = _v3_pooled(cases, cfg, meth, crosses=2)
    P2, _, a2 = _v3_pooled(cases, cfg, meth, crosses=2, order=list(range(16))[::-1])
    key = ["arm", "method", "rid", "h"]
    M = P1.merge(P2, on=key, suffixes=("_1", "_2"))
    assert len(M) == len(P1) == len(P2)
    for c in ("est", "lo", "hi"):
        a, b = M[f"{c}_1"].to_numpy(float), M[f"{c}_2"].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), c


def test_bench_scoring_api_scores_a_submission():
    from clbench.bench import scoring

    rng = np.random.default_rng(3)
    n = 50
    W = scoring.bootstrap_weights(n, 200)
    sub = scoring.Submission("toy", scale="30s")
    good = {"R01": -1.0, "R05": -1.0, "P01": 1.0}
    for arm in scoring.ARMS:
        for rid in scoring.RELATION_IDS:
            mu = good.get(rid, 0.0) if arm == "real" else 0.0
            x = mu + rng.normal(0, 0.5, n)
            est, reps = scoring.event_estimate(x, np.ones(n), W)
            sub.add(arm, rid, est, reps, n)
    calib, row = scoring.score(sub)
    assert row["cal_correct"] == 3 and row["cal_wrong"] == 0 and row["score"] == 3
    assert row["null_cross_fa_per_draw"] <= 0.25 and "score 3" in row["text"]


def test_ground_truth_table_has_sources_for_every_relation():
    from clbench.bench import ground_truth_table

    G = ground_truth_table()
    assert len(G) == 18 and set(G.kind) == {"response", "policy"}
    assert (G.sources.str.len() > 20).all() and (G.expected.abs() == 1).all()


def test_sim_adapter_events_and_itt_truth():
    from clbench.bench import sim as bsim
    from clbench.sim import clinician_loop as cl

    scfg = dataclasses.replace(bsim.sim_config("recorded", "active"), duration_s=2400)
    coh = cl.simulate_cohort(scfg, 30, seed=4)
    cases = bsim.cohort_to_cases(coh, "2s")
    e = coh.events["remi"]
    p0 = e["patient"][0]
    ev = cases[p0].events["remifentanil"]
    assert ev.t[0] == e["t"][0] + 0.5 and ev.after[0] - ev.before[0] == e["delta"][0]
    T = bsim.itt_truth_bench(scfg, 30, 4, "2s", k_max=1, base=coh)
    t6 = T[(T.rid == "R06") & (T.h == 120.0)]
    assert len(t6) > 3 and t6.itt.mean() < 0  # remifentanil up lowers MAP


def test_pooled_nn_contrast_matches_brute_force():
    rng = np.random.default_rng(9)
    nc, ne, G = 400, 60, 12
    gc = np.sort(rng.integers(0, G, nc))  # clustered groups: many own-group neighbours
    fc = rng.normal(0, 1, (nc, 3)) + gc[:, None] * 0.3
    yc = fc @ np.array([1.0, -0.5, 0.2]) + rng.normal(0, 0.1, nc)
    ge = rng.integers(0, G, ne)
    fe = rng.normal(0, 1, (ne, 3)) + ge[:, None] * 0.3
    ye = fe @ np.array([1.0, -0.5, 0.2]) + 0.7
    bc, raw = clb.pooled_nn_contrast(fe, ye, ge, fc, yc, gc, k=5, bias_correct="both")
    mu, sd = fc.mean(0), fc.std(0)
    zc, ze = (fc - mu) / sd, (fe - mu) / sd
    beta = np.linalg.lstsq(np.column_stack([np.ones(nc), zc]), yc, rcond=None)[0]
    for i in range(ne):
        d = ((zc - ze[i]) ** 2).sum(1)
        d[gc == ge[i]] = np.inf
        nn = np.argsort(d, kind="stable")[:5]
        assert np.isclose(raw[i, 0], ye[i] - yc[nn].mean())
        assert np.isclose(bc[i, 0], ye[i] - yc[nn].mean() - (ze[i] - zc[nn].mean(0)) @ beta[1:])


def test_patient_specificity_is_independent_of_arrival_order():
    rng = np.random.default_rng(11)
    cfg = CFG
    agg1 = clb.Aggregator([f"p{i}" for i in range(40)], cfg, n_boot=20, seed=0, spec_boot=100)
    agg2 = clb.Aggregator([f"p{i}" for i in range(40)], cfg, n_boot=20, seed=0, spec_boot=100)
    rows = [("real", "R06", "obs", c, 4, -4.0 + rng.normal(), 9.0, 4, -3.0 + rng.normal(), 8.0) for c in range(40)]
    agg1.spec_rows = list(rows)
    agg2.spec_rows = list(rows[::-1])
    agg1._spec()
    agg2._spec()
    a, b = agg1.spec_table, agg2.spec_table
    assert np.allclose(a.select_dtypes(float).to_numpy(), b.select_dtypes(float).to_numpy(), equal_nan=True)
