"""
Unit tests for clbench.sim.clinician_loop — the clinician-in-the-loop simulator and the
estimators used in paper/closed_loop_theory.md. Synthetic only; small cohorts so the suite is fast.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.sim.clinician_loop import (
    EventSpec,
    SimConfig,
    _pretrend,
    _upcoming,
    arx_fit,
    bootstrap_median,
    cluster_bootstrap_mean,
    decision_contrasts,
    delay,
    event_effects,
    exog_labels,
    itt_truth,
    observe,
    residualize,
    sign_status,
    simulate_cohort,
    step_truth,
    trailing_mean,
    with_controller,
    with_drivers,
    with_physiology,
)


def _small(cfg: SimConfig | None = None, **ctrl) -> SimConfig:
    cfg = cfg or SimConfig()
    cfg = replace(cfg, duration_s=3600, burn_in_s=300)
    return with_controller(cfg, **ctrl) if ctrl else cfg


@pytest.fixture(scope="module")
def cohort():
    return simulate_cohort(_small(gain=1.0, anticipation_s=60.0), 6, seed=3)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class TestSimulator:
    def test_shapes_and_burn_in(self, cohort):
        assert cohort.hr.shape == (6, 3600)
        for name in ("map", "co2", "hr_obs", "remi", "remi_ce", "vent", "z", "s", "q"):
            assert getattr(cohort, name).shape == (6, 3600)
        assert set(cohort.gains) >= {"beta", "kappa", "remi_hr", "vent_co2"}

    def test_reproducible(self):
        a = simulate_cohort(_small(gain=1.0), 3, seed=11)
        b = simulate_cohort(_small(gain=1.0), 3, seed=11)
        np.testing.assert_array_equal(a.hr, b.hr)
        np.testing.assert_array_equal(a.remi, b.remi)

    def test_gain_signs_kept_under_jitter(self, cohort):
        assert np.all(cohort.gains["beta"] < 0)
        assert np.all(cohort.gains["kappa"] > 0)
        assert np.all(cohort.gains["remi_hr"] < 0)
        assert np.all(cohort.gains["vent_co2"] < 0)

    def test_event_log_matches_input_changes(self, cohort):
        C = cohort.config.controller
        for name, step in (("remi", C.remi_step), ("vent", C.vent_step)):
            u = getattr(cohort, name)
            ev = cohort.events[name]
            n_changes = int((np.diff(u, axis=1) != 0).sum())
            # a decision at second t is recorded at t + 1; one on the last second is not visible
            visible = ev[ev["t"] + 1 < cohort.T]
            assert len(visible) == n_changes
            assert np.all(np.isclose(np.abs(ev["delta"]), step) | (np.abs(ev["delta"]) < step))
            p, t = visible["patient"], visible["t"]
            np.testing.assert_allclose(u[p, t + 1] - u[p, t], visible["delta"])

    def test_inputs_respect_bounds_and_refractory(self, cohort):
        C = cohort.config.controller
        assert cohort.remi.min() >= C.remi_bounds[0] - 1e-9 and cohort.remi.max() <= C.remi_bounds[1] + 1e-9
        ev = cohort.events["remi"]
        for p in np.unique(ev["patient"]):
            t = np.sort(ev["t"][ev["patient"] == p])
            assert np.all(np.diff(t) >= C.refractory_s)

    def test_open_loop_changes_are_all_exogenous(self):
        c = simulate_cohort(_small(gain=0.0), 6, seed=4)
        for k in ("remi", "vent"):
            if len(c.events[k]):
                assert c.events[k]["exog"].all()

    def test_effect_site_lags_target(self, cohort):
        """First-order PK: CE moves toward the target and never overshoots it after a step."""
        u, ce = cohort.remi[0], cohort.remi_ce[0]
        t = np.flatnonzero(np.diff(u) != 0)
        if len(t) == 0:
            pytest.skip("no change in patient 0")
        i = t[0] + 1
        assert abs(ce[i + 60] - u[i]) < abs(ce[i] - u[i])

    def test_upcoming_window_sum(self):
        imp = np.zeros((1, 10))
        imp[0, 5] = 2.0
        up = _upcoming(imp, 3)
        # impulse at 5 is within (t, t+3] for t = 2, 3, 4
        np.testing.assert_allclose(up[0], [0, 0, 2, 2, 2, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(_upcoming(imp, 0), 0.0)

    def test_anticipation_moves_increases_before_episodes(self):
        """With a plan horizon, remifentanil increases cluster in the minutes before stimulation."""

        def share_before_onset(H: float) -> float:
            cfg = with_drivers(_small(gain=1.0, anticipation_s=H, w_y=0.0, w_z=0.0), episode_rate_per_h=8.0)
            c = simulate_cohort(cfg, 12, seed=9)
            # recover onsets: s rises after an onset; use z jumps
            dz = np.diff(c.z, axis=1)
            ev = c.events["remi"]
            ev = ev[(ev["delta"] > 0) & ~ev["exog"]]
            if len(ev) == 0:
                return 0.0
            hits = [np.any(dz[p, t : t + 180] > 0.3) for p, t in zip(ev["patient"], ev["t"])]
            return float(np.mean(hits))

        assert share_before_onset(180.0) > 0.8


# ---------------------------------------------------------------------------
# Observation operators
# ---------------------------------------------------------------------------


class TestObserve:
    def test_scales(self, cohort):
        b, f, w = observe(cohort, "beat"), observe(cohort, "2s"), observe(cohort, "30s")
        assert b.hr.shape[1] == cohort.T and b.step_s == 1.0
        assert f.hr.shape[1] == cohort.T // 2 and f.step_s == 2.0
        assert w.step_s == 15.0 and np.all(np.diff(w.t_end) == 15)
        # 30 s input value is the value at the window end
        np.testing.assert_array_equal(w.remi, cohort.remi[:, w.t_end])

    def test_trailing_mean(self):
        x = np.arange(10, dtype=float)[None]
        np.testing.assert_allclose(trailing_mean(x, 3)[0, 2:], [1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_allclose(trailing_mean(x, 3)[0, :2], [0, 0.5])

    def test_exog_labels_on_grid(self, cohort):
        o = observe(cohort, "2s")
        lab = exog_labels(cohort, o, "remi")
        du = np.diff(o.remi, axis=1, prepend=o.remi[:, :1]) != 0
        assert not np.any(lab & ~du)  # labels sit on visible changes only


# ---------------------------------------------------------------------------
# Estimator building blocks
# ---------------------------------------------------------------------------


class TestResidualize:
    def test_removes_linear_trend_in_interior(self):
        x = 3.0 + 0.5 * np.arange(200.0)
        r = residualize(x, 21)
        assert np.all(np.isnan(r[:10])) and np.all(np.isnan(r[-10:]))
        np.testing.assert_allclose(r[10:-10], 0.0, atol=1e-9)

    def test_nan_aware(self):
        x = np.sin(np.arange(300) / 5.0)
        x[100] = np.nan
        r = residualize(x, 21)
        assert np.isnan(r[100])
        assert np.isfinite(r[95]) and np.isfinite(r[105])

    def test_none_demeans(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(residualize(x, None), [-1, 0, 1])


class TestARX:
    def test_recovers_open_loop_arx(self):
        rng = np.random.default_rng(0)
        n = 20000
        x = rng.normal(size=n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.5 * y[t - 1] - 0.3 * x[t - 1] + 0.2 * x[t - 2] + rng.normal()
        r = arx_fit(y, x, 2)
        assert r["coupling"] == pytest.approx(-0.1, abs=0.03)
        assert r["persistence"] == pytest.approx(0.5, abs=0.03)
        assert r["static"] == pytest.approx(-0.2, abs=0.06)

    def test_slow_common_driver_flips_raw_but_not_residualised(self):
        """Proposition 4 in miniature: a smooth latent driver of both series makes the raw lagged
        coefficient wrong-signed; residualising at a width between the loop and driver timescales
        restores the sign."""
        rng = np.random.default_rng(1)
        n = 40000
        s = np.convolve(rng.normal(size=n + 2000), np.ones(400) / 20.0, "same")[:n]  # smooth driver
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + 0.4 * (2.0 * s[t]) + rng.normal()
            y[t] = 0.6 * y[t - 1] + 0.4 * (-0.4 * x[t - 1] + 2.0 * s[t]) + rng.normal()
        raw = arx_fit(y, x, 3)["static"]
        res = arx_fit(y, x, 3, width=31)["static"]
        assert raw > 0
        assert res < 0

    def test_pretrend_exact_on_line(self):
        y = 2.0 + 0.25 * np.arange(50.0)
        a, b = _pretrend(y, 10)
        assert np.isnan(a[9])
        # intercept is the extrapolated value at index i (time 0), slope per sample
        np.testing.assert_allclose(a[10:], y[10:])
        np.testing.assert_allclose(b[10:], 0.25)


class TestEventEffects:
    def test_noise_free_step_equals_truth(self):
        cfg = _small(gain=1.0)
        spec = EventSpec("vent", "co2", +1)
        truth = step_truth(cfg, spec, "beat", (30, 120), n_phase=1)
        # vent effect on EtCO2: G_v · step · (1 − e^{−h/τ}) up to the 3-sample post average
        P = cfg.physiology
        expected = P.vent_co2 * cfg.controller.vent_step * (1 - np.exp(-np.array([29.0, 119.0]) / P.tau_co2))
        assert truth[30] == pytest.approx(expected[0], rel=0.05)
        assert truth[120] == pytest.approx(expected[1], rel=0.05)
        assert truth[30] < 0 and truth[120] < truth[30]

    def test_truth_signs_for_remifentanil(self):
        cfg = _small()
        for scale in ("beat", "2s", "30s"):
            tr = step_truth(cfg, EventSpec("remi", "hr", +1), scale, (60, 120), n_phase=3)
            assert tr[60] < 0 and tr[120] < 0
            tr = step_truth(cfg, EventSpec("remi", "hr", -1), scale, (60,), n_phase=3)
            assert tr[60] > 0

    def test_all_event_estimators_reported(self, cohort):
        o = observe(cohort, "2s")
        rows = event_effects(o, EventSpec("remi", "hr", +1), (30, 60))
        assert rows, "expected remifentanil increases in the fixture cohort"
        for key in ("naive", "matched", "pretrend", "kink", "kink_rich", "exog", "patient", "h"):
            assert key in rows[0]
        assert {r["h"] for r in rows} <= {30, 60}

    def test_open_loop_estimators_unbiased_on_average(self):
        """With exogenous changes only, the naive contrast is centred on the truth (loose check)."""
        cfg = with_drivers(_small(gain=0.0), ou_sd=0.0, episode_rate_per_h=0.0, insuff_rate_per_h=0.0)
        cfg = with_controller(cfg, exog_vent_per_h=6.0)
        cfg = with_physiology(cfg, noise_co2=0.05, meas_co2=0.05)
        c = simulate_cohort(cfg, 10, seed=2)
        o = observe(c, "2s")
        spec = EventSpec("vent", "co2", +1)
        rows = event_effects(o, spec, (60,))
        tr = step_truth(cfg, spec, "2s", (60,))[60]
        est = np.mean([r["pretrend"] for r in rows])
        assert len(rows) > 10
        assert est == pytest.approx(tr, rel=0.25)


class TestStats:
    def test_cluster_bootstrap(self):
        v = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
        g = np.array([0, 0, 1, 1, 2])
        est, lo, hi = cluster_bootstrap_mean(v, g, n_boot=500)
        assert est == pytest.approx(2.6)
        assert lo <= est <= hi
        assert lo >= 1.0 and hi <= 5.0

    def test_bootstrap_median(self):
        est, lo, hi = bootstrap_median(np.array([1.0, 2.0, 3.0, np.nan]), n_boot=200)
        assert est == 2.0 and lo <= 2.0 <= hi

    def test_sign_status(self):
        assert sign_status(-2.0, -1.0, -1) == "correct"
        assert sign_status(1.0, 2.0, -1) == "wrong"
        assert sign_status(-1.0, 1.0, -1) == "inconclusive"
        assert sign_status(np.nan, 1.0, -1) == "undefined"


# ---------------------------------------------------------------------------
# v2: control pools, twin (veto) simulation, HR display latency, estimand functionals
# ---------------------------------------------------------------------------


class TestV2:
    def test_event_path_labels(self, cohort):
        ev = cohort.events["remi"]
        assert "alarm" in ev.dtype.names
        assert not np.any(ev["alarm"] & ev["exog"])  # exogenous changes are never on the alarm path
        assert cohort.alarm_check is not None and cohort.alarm_check.shape == cohort.hr.shape

    def test_veto_is_a_paired_counterfactual(self):
        cfg = _small(gain=1.0, w_z=0.0)
        a = simulate_cohort(cfg, 4, seed=21)
        ev = a.events["remi"]
        p, t = int(ev["patient"][0]), int(ev["t"][0])
        v = np.full(4, -1)
        v[p] = t
        b = simulate_cohort(cfg, 4, seed=21, veto={"remi": v})
        # identical up to and including the decision second; no change recorded at t + 1 in the twin
        np.testing.assert_array_equal(a.hr[:, : t + 1], b.hr[:, : t + 1])
        assert b.remi[p, t + 1] == b.remi[p, t]
        assert a.remi[p, t + 1] != a.remi[p, t]
        # other patients are untouched
        others = [i for i in range(4) if i != p]
        np.testing.assert_array_equal(a.hr[others], b.hr[others])

    def test_itt_truth_is_zero_for_inert_drug(self):
        cfg = with_physiology(_small(gain=1.0, w_z=0.0), remi_hr=0.0, remi_map=0.0)
        rows = itt_truth(cfg, 6, EventSpec("remi", "hr", +1), (30, 60), seed=5, k_max=2)
        assert rows
        np.testing.assert_allclose([r["itt_effect"] for r in rows], 0.0, atol=1e-12)

    def test_itt_truth_negative_for_active_drug(self):
        cfg = _small(gain=1.0, w_z=0.0)
        rows = itt_truth(cfg, 20, EventSpec("remi", "hr", +1), (110,), seed=5, k_max=2)
        assert np.mean([r["itt_effect"] for r in rows]) < 0

    def test_future_inaction_pool_biased_eligible_pool_not(self):
        """Lemma FI in miniature: inert drug, recorded-only clinician, routine-path decisions matched on
        the recorded decision statistic. The future-inaction pool gives a positive (wrong-direction)
        contrast; the eligible pool is centred near zero."""
        cfg = with_physiology(_small(gain=1.0, w_z=0.0, anticipation_s=0.0), remi_hr=0.0, remi_map=0.0)
        c = simulate_cohort(cfg, 150, seed=8)
        rows = [r for r in decision_contrasts(c, EventSpec("remi", "hr", +1), (60,)) if r["path"] == "routine"]
        fi = np.mean([r["effect"] for r in rows if r["pool"] == "future_inaction"])
        el = np.mean([r["effect"] for r in rows if r["pool"] == "eligible"])
        assert fi > el + 0.3
        assert abs(el) < 0.5

    def test_event_effects_pools(self, cohort):
        o = observe(cohort, "2s")
        a = event_effects(o, EventSpec("remi", "hr", +1), (30,), control_pool="eligible")
        b = event_effects(o, EventSpec("remi", "hr", +1), (30,), control_pool="future_inaction")
        assert [(r["patient"], r["i"]) for r in a] == [(r["patient"], r["i"]) for r in b]  # same events
        with pytest.raises(ValueError):
            event_effects(o, EventSpec("remi", "hr", +1), (30,), control_pool="bogus")

    def test_hr_display_delay(self, cohort):
        d = delay(cohort.hr_obs, 4)
        np.testing.assert_array_equal(d[:, 4:], cohort.hr_obs[:, :-4])
        o0 = observe(cohort, "2s")
        o4 = observe(cohort, "2s", monitor_hr_delay_s=4)
        np.testing.assert_array_equal(o4.map, o0.map)
        np.testing.assert_allclose(o4.hr[:, 2:], o0.hr[:, :-2])

    def test_hr_display_latency_flips_residualised_sign(self):
        """A displayed-HR latency longer than the ARX lag span moves the HR → MAP path (κ > 0) into the
        lagged MAP → HR coefficient: the residualised 2 s estimate turns positive."""
        cfg = with_drivers(_small(gain=1.0), timescale_s=2000.0)
        c = simulate_cohort(cfg, 8, seed=12)
        est = {}
        for d in (0, 12):
            o = observe(c, "2s", monitor_hr_delay_s=d)
            est[d] = np.median([arx_fit(o.hr[i], o.map[i], 4, width=31)["static"] for i in range(c.n)])
        assert est[0] < 0 < est[12]

    def test_naive_and_pretrend_functionals_differ_at_30s(self):
        cfg = _small()
        spec = EventSpec("remi", "hr", +1)
        n = step_truth(cfg, spec, "30s", (30,), n_phase=5, functional="naive")[30]
        p = step_truth(cfg, spec, "30s", (30,), n_phase=5, functional="pretrend")[30]
        assert n < p < 0  # the 2-sample post mean sits later than the 3-sample one
        with pytest.raises(ValueError):
            step_truth(cfg, spec, "30s", (30,), functional="bogus")
