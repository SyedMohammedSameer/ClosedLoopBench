"""Unit tests for clbench.edges.typed_temporal on synthetic episodes."""

from __future__ import annotations

import numpy as np
import pytest

from clbench.edges import typed_temporal as tt

C_IDX = {c: i for i, c in enumerate(tt.CONCEPTS)}


def _markov(T: int, p_on: float, p_off: float, rng) -> np.ndarray:
    s = np.zeros(T, dtype=np.int8)
    for t in range(1, T):
        s[t] = (rng.random() < p_on) if s[t - 1] == 0 else (rng.random() >= p_off)
    return s


def synth_cohort(n_cases=120, T=600, seed=0, hr_edge=np.log(4.0), policy=np.log(8.0),
                 deviant_cases=()):
    """Planted: tachycardia (on at lag 2..) raises hypotension onset hazard by exp(hr_edge);
    hypertension on raises remifentanil-up hazard by exp(policy) (closed-loop policy); remifentanil up
    lowers the hypertension onset hazard (response) by x0.3 over the next 1-10 min.
    Cases in ``deviant_cases`` have the tachycardia->hypotension effect multiplied by exp(2.5)."""
    rng = np.random.default_rng(seed)
    S_all, chg_all = [], []
    for c in range(n_cases):
        S = np.zeros((T, 8), dtype=np.int8)
        for k in range(8):
            if k in (C_IDX["hypotension"], C_IDX["hypertension"]):
                continue
            S[:, k] = _markov(T, 0.01, 0.1, rng)
        chg = np.zeros((T, 4, 2), dtype=np.int8)
        hyp = np.zeros(T, dtype=np.int8)
        hyper = np.zeros(T, dtype=np.int8)
        last_remi_up = -1000
        tach = S[:, C_IDX["tachycardia"]]
        for t in range(2, T):
            # hypotension
            if hyp[t - 1] == 1:
                hyp[t] = rng.random() >= 0.15
            elif hyp[t - 2] == 0:
                eff = hr_edge * (tach[t - 2] == 1) * (1 + (2.5 / hr_edge if c in deviant_cases else 0))
                h = 1 / (1 + np.exp(-(np.log(0.01 / 0.99) + eff)))
                hyp[t] = rng.random() < h
            # hypertension with remi response
            if hyper[t - 1] == 1:
                hyper[t] = rng.random() >= 0.15
            elif hyper[t - 2] == 0:
                resp = np.log(0.3) if 5 <= t - last_remi_up <= 40 else 0.0
                h = 1 / (1 + np.exp(-(np.log(0.012 / 0.988) + resp)))
                hyper[t] = rng.random() < h
            # policy: remi up at bin t depends on hypertension at t-1
            lp = np.log(0.005 / 0.995) + policy * (hyper[t - 1] == 1)
            if rng.random() < 1 / (1 + np.exp(-lp)):
                chg[t, 1, 0] = 1
                last_remi_up = t
        S[:, C_IDX["hypotension"]] = hyp
        S[:, C_IDX["hypertension"]] = hyper
        S_all.append(S)
        chg_all.append(chg)
    Tv = np.full(n_cases, T)
    n = n_cases * T
    dose = np.full((n, 5), np.nan)
    dose[:, 1] = 2.0  # remifentanil running everywhere
    return tt.Cohort(case=np.repeat(np.arange(n_cases), T), step=np.tile(np.arange(T), n_cases),
                     S=np.vstack(S_all), dose=dose, chg=np.concatenate(chg_all),
                     vital_z=np.zeros((n, 4)), anes_start=np.zeros(n_cases, dtype=np.int64), T=Tv)


@pytest.fixture(scope="module")
def cohort():
    return synth_cohort()


def _fit(co, des):
    fit = tt.fit_logistic(des.X, des.y, des.w, des.case, co.n_cases)
    draws = tt.score_bootstrap(fit, tt.bootstrap_weights(co.n_cases, 400, seed=1))
    return fit, {e.source: e for e in tt.edge_stats(des, fit, draws)}


def test_lag_primitives():
    S = np.array([0, 0, 1, 1, 0, 0, 1, -1, 0, 0, 1], dtype=np.int8)
    cs = np.zeros(len(S), dtype=np.int64)
    on, risk = tt.onsets_and_risk(S, cs)
    assert on.nonzero()[0].tolist() == [2, 6, 10]
    assert not risk[7] and not risk[8] and risk[10]
    x = np.array([1, 0, 0, 1, 0], dtype=np.int8)
    assert tt.lagged_count(x, cs[:5], 1, 2).tolist() == [0, 1, 1, 0, 1]
    # case boundary: second case starts at row 3
    cs2 = np.array([0, 0, 0, 3, 3])
    assert tt.lagged_count(x, cs2, 1, 3).tolist() == [0, 1, 1, 0, 1]
    assert tt.last_on_steps(np.array([1, 0, 0, 1, 0], dtype=np.int8), cs2, 1).tolist() == [-1, 1, 2, -1, 1]


def test_planted_association_recovered(cohort):
    co = cohort
    b = C_IDX["hypotension"]
    rows = tt.association_rows(co, b)
    des = tt.association_design(co, b, rows)
    _, es = _fit(co, des)
    e = es["tachycardia"]
    assert e.p < 1e-6 and e.omega_lo > 0.5  # planted log HR = log 4 = 1.39
    assert 0.8 < e.omega < 2.0
    null_ps = [es[s].p for s in es if s != "tachycardia"]
    assert min(null_ps) > 1e-4


def test_transplant_and_shift_nulls_remove_edge(cohort):
    co = cohort
    b = C_IDX["hypotension"]
    rows = tt.association_rows(co, b)
    for m in (tt.transplant_map(co, np.random.default_rng(3)),
              tt.shift_map(co, np.random.default_rng(4), np.full(co.n_cases, 41))):
        des = tt.association_design(co, b, rows, tt.apply_map_S(co.S, m))
        _, es = _fit(co, des)
        assert es["tachycardia"].p > 1e-3
    # transplant never maps a case onto itself; shift stays inside the case
    m = tt.transplant_map(co, np.random.default_rng(3))
    assert np.all(co.case[m[m >= 0]] != co.case[m >= 0])
    m = tt.shift_map(co, np.random.default_rng(4), np.full(co.n_cases, 41))
    assert np.all(co.case[m] == co.case)
    assert np.all(((m - co.cs) - co.step) % co.T[co.case] % 41 == 0)


def test_closed_loop_policy_typed_as_policy(cohort):
    co = cohort
    ai, di = tt.ACTIONS.index("remifentanil"), 0
    rows = tt.policy_rows(co, ai, di)
    des = tt.policy_design(co, ai, di, rows)
    _, es = _fit(co, des)
    e = es["hypertension"]
    assert e.p < 1e-6 and e.omega_lo > 1.0  # planted log 8 = 2.08 (state at lag 1)
    others = [es[s].p for s in es if s != "hypertension"]
    assert min(others) > 1e-4
    # response: remifentanil up lowers the hypertension onset hazard (planted x0.3)
    b = C_IDX["hypertension"]
    rdes = tt.response_design(co, b, tt.association_rows(co, b))
    _, rs = _fit(co, rdes)
    assert rs["remifentanil_up"].omega_hi < 0


def test_patient_deviation_detected():
    dev_cases = (0, 1, 2)
    co = synth_cohort(n_cases=80, T=800, seed=5, deviant_cases=dev_cases)
    b = C_IDX["hypotension"]
    rows = tt.association_rows(co, b)
    des = tt.association_design(co, b, rows)
    fit = tt.fit_logistic(des.X, des.y, des.w, des.case, co.n_cases)
    eta = des.X @ fit.beta
    alpha = tt.patient_intercepts(eta, des.y, des.case, co.n_cases)
    mu = 1 / (1 + np.exp(-(eta + alpha[des.case])))
    d = tt.deviation_scores(mu, des.y, des.exposure["tachycardia"], des.case, co.n_cases)
    z = d.z
    assert np.nanmean(z[list(dev_cases)]) > 2.0
    assert np.nanmean(np.abs(z[3:])) < 1.5
    se2 = np.where(d.I > 0, 1 / np.maximum(d.I, 1e-12), np.inf)
    shr, tau2 = tt.eb_shrink(d.delta_hat(), se2)
    assert tau2 > 0
    assert np.all(np.abs(shr) <= np.abs(d.delta_hat()) + 1e-12)


def test_fixed_effects_fit_recovers_edge_and_ignores_patient_prevalence():
    co = synth_cohort(n_cases=100, T=600, seed=9)
    b = C_IDX["hypotension"]
    rows = tt.association_rows(co, b)
    des = tt.association_design(co, b, rows)
    fit = tt.fit_logistic_fe(des.X, des.y, des.w, des.case, co.n_cases)
    draws = tt.score_bootstrap(fit, tt.bootstrap_weights(co.n_cases, 400, seed=1))
    es = {e.source: e for e in tt.edge_stats(des, fit, draws)}
    assert es["tachycardia"].p < 1e-6 and 0.8 < es["tachycardia"].omega < 2.0
    # a source whose per-patient prevalence tracks the target's per-patient baseline, with no
    # within-patient timing link: shift-null style confounding that the fixed effects remove
    rng = np.random.default_rng(2)
    S = co.S.copy()
    k = C_IDX["hypercapnia"]
    rate = np.bincount(co.case, weights=(S[:, b] == 1), minlength=co.n_cases) / co.T
    for c in range(co.n_cases):
        s0 = co.case_start[c]
        S[s0:s0 + co.T[c], k] = (rng.random(co.T[c]) < min(0.9, 3 * rate[c])).astype(np.int8)
    des2 = tt.association_design(co, b, rows, S)
    fit2 = tt.fit_logistic_fe(des2.X, des2.y, des2.w, des2.case, co.n_cases)
    d2 = tt.score_bootstrap(fit2, tt.bootstrap_weights(co.n_cases, 400, seed=1))
    es2 = {e.source: e for e in tt.edge_stats(des2, fit2, d2)}
    assert es2["hypercapnia"].p > 1e-3


def test_replication_chance():
    E1 = [{(0, 4), (1, 5)}, set(), {(2, 6)}]
    E2 = [{(0, 4)}, {(1, 4)}, {(2, 6), (3, 7)}]
    r = tt.replication_with_chance(E1, E2, [48, 48, 48])
    assert r["replication"] == pytest.approx(2 / 3)
    assert r["chance"] == pytest.approx((2 * 1 / 48 + 1 * 2 / 48) / 3)


def test_family_sizes():
    assert len(tt.cross_vital_pairs()) == 48
    assert len(tt.null_pairs()) == 32
