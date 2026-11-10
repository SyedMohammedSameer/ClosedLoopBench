"""
Unit tests for clbench.data.vitaldb and the shared windowing helper.

All tests use synthetic NumPy arrays — no .vital files or the vitaldb package required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.data.preprocess import (
    ABP_POPULATION_MU_MMHG,
    ABP_POPULATION_SIGMA_MMHG,
    windows_from_signal,
)
from clbench.data.vitaldb import (
    TARGET_FS,
    _window_last,
    _window_median,
    numeric_concept_labels,
    patient_id_for,
    resample_to_target,
    step_changes,
    window_grid,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# resample_to_target
# ---------------------------------------------------------------------------


class TestResampleToTarget:
    def test_500_hz_to_target_length(self):
        out = resample_to_target(np.zeros((5000, 2)), 500.0)
        assert out.shape == (625, 2)
        assert out.dtype == np.float32

    def test_one_dimensional_input_becomes_one_channel(self):
        assert resample_to_target(np.zeros(800), 500.0).shape == (100, 1)

    def test_in_band_sine_preserved(self):
        fs = 500.0
        t = np.arange(20_000) / fs
        out = resample_to_target(np.sin(2 * np.pi * 1.2 * t), fs)[:, 0]
        ref = np.sin(2 * np.pi * 1.2 * np.arange(len(out)) / TARGET_FS)
        core = slice(200, len(out) - 200)
        assert np.max(np.abs(out[core] - ref[core])) < 0.02

    def test_nan_gap_stays_a_gap(self):
        x = np.ones(8000)
        x[2000:4000] = np.nan  # samples 250..500 at 62.5 Hz
        out = resample_to_target(x, 500.0)[:, 0]
        assert np.isnan(out[260:490]).all()
        assert not np.isnan(out[:240]).any()
        assert not np.isnan(out[510:]).any()

    def test_all_nan_channel_stays_nan(self):
        x = np.stack([np.ones(800), np.full(800, np.nan)], axis=1)
        out = resample_to_target(x, 500.0)
        assert np.isnan(out[:, 1]).all()
        assert not np.isnan(out[:, 0]).any()

    def test_same_rate_is_passthrough(self):
        x = RNG.normal(size=(100, 1))
        np.testing.assert_allclose(resample_to_target(x, TARGET_FS), x.astype(np.float32))


# ---------------------------------------------------------------------------
# step_changes
# ---------------------------------------------------------------------------


class TestStepChanges:
    def test_first_setting_from_off_is_an_event(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        v = np.array([0.0, 0.0, 3.0, 3.0])
        assert step_changes(t, v, tol=0.05, lo=0.0, hi=12.0) == [(2.0, 0.0, 3.0)]

    def test_burst_of_adjustments_is_one_action(self):
        t = np.array([0.0, 100.0, 103.0, 106.0, 200.0])
        v = np.array([2.0, 3.0, 3.5, 4.0, 4.0])
        events = step_changes(t, v, tol=0.05, lo=0.0, hi=12.0, debounce_sec=10.0)
        assert events == [(0.0, 0.0, 2.0), (100.0, 2.0, 4.0)]

    def test_implausible_values_dropped(self):
        t = np.array([0.0, 10.0, 20.0])
        v = np.array([3.0, 20972.0, 3.0])
        assert step_changes(t, v, tol=0.05, lo=0.0, hi=12.0) == [(0.0, 0.0, 3.0)]

    def test_jitter_below_tolerance_ignored(self):
        t = np.arange(4, dtype=float)
        v = np.array([3.0, 3.01, 2.99, 3.02])
        assert step_changes(t, v, tol=0.05, lo=0.0, hi=12.0) == [(0.0, 0.0, 3.0)]

    def test_reversal_within_debounce_cancels(self):
        t = np.array([0.0, 100.0, 104.0])
        v = np.array([2.0, 3.0, 2.0])
        assert step_changes(t, v, tol=0.05, lo=0.0, hi=12.0) == [(0.0, 0.0, 2.0)]

    def test_nan_and_unsorted_records(self):
        t = np.array([50.0, 0.0, 20.0])
        v = np.array([4.0, 2.0, np.nan])
        assert step_changes(t, v, tol=0.05, lo=0.0, hi=12.0) == [(0.0, 0.0, 2.0), (50.0, 2.0, 4.0)]

    def test_stopping_an_infusion_is_an_event(self):
        t = np.array([0.0, 300.0])
        v = np.array([5.0, 0.0])
        assert step_changes(t, v, tol=0.05, lo=0.0, hi=500.0) == [(0.0, 0.0, 5.0), (300.0, 5.0, 0.0)]


# ---------------------------------------------------------------------------
# Window aggregates
# ---------------------------------------------------------------------------


class TestWindowAggregates:
    def test_window_grid_matches_stride(self):
        np.testing.assert_array_equal(window_grid(60.0, 30.0, 15.0), [0.0, 15.0, 30.0])

    def test_window_grid_short_recording_is_empty(self):
        assert window_grid(29.0, 30.0, 15.0).size == 0

    def test_median_per_window(self):
        t = np.array([1.0, 2.0, 16.0, 17.0])
        v = np.array([80.0, 90.0, 100.0, 110.0])
        starts = np.array([0.0, 15.0, 60.0])
        out = _window_median(t, v, starts, starts + 30.0)
        np.testing.assert_allclose(out[:2], [95.0, 105.0])
        assert np.isnan(out[2])

    def test_last_value_respects_hold(self):
        t = np.array([5.0, 50.0])
        v = np.array([1.0, 2.0])
        out = _window_last(t, v, np.array([30.0, 60.0, 200.0, 2.0]), hold_sec=60.0)
        np.testing.assert_allclose(out[:2], [1.0, 2.0])
        assert np.isnan(out[2])  # last record 150 s old
        assert np.isnan(out[3])  # before any record

    def test_empty_tracks_give_nan(self):
        starts = np.array([0.0, 15.0])
        assert np.isnan(_window_median(np.empty(0), np.empty(0), starts, starts + 30)).all()
        assert np.isnan(_window_last(np.empty(0), np.empty(0), starts + 30)).all()


# ---------------------------------------------------------------------------
# numeric_concept_labels
# ---------------------------------------------------------------------------


class TestNumericConceptLabels:
    @pytest.fixture()
    def numerics(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "hr": [120.0, 50.0, np.nan, 100.0],
                "pleth_hr": [121.0, 49.0, 80.0, 100.0],
                "rr": [25.0, 10.0, 15.0, 20.0],
                "map": [60.0, 110.0, 80.0, 65.0],
            }
        )

    def test_thresholds(self, numerics):
        labels = numeric_concept_labels(numerics)
        assert labels["label_tachycardia"].tolist()[:2] == [1, 0]
        assert labels["label_bradycardia"].tolist()[:2] == [0, 1]
        assert labels["label_tachypnea"].tolist()[:3] == [1, 0, 0]
        assert labels["label_bradypnea"].tolist()[:3] == [0, 1, 0]
        assert labels["label_hypotension"].tolist()[:3] == [1, 0, 0]
        assert labels["label_hypertension"].tolist()[:3] == [0, 1, 0]

    def test_missing_numeric_is_na_not_negative(self, numerics):
        labels = numeric_concept_labels(numerics)
        assert pd.isna(labels.loc[2, "label_tachycardia"])
        assert labels.loc[2, "label_tachycardia_ppg"] == 0

    def test_boundaries_are_strict(self, numerics):
        labels = numeric_concept_labels(numerics)
        row = labels.loc[3]
        assert row["label_tachycardia"] == 0
        assert row["label_tachypnea"] == 0
        assert row["label_hypotension"] == 0

    def test_nullable_int_dtype(self, numerics):
        assert all(str(d) == "Int8" for d in numeric_concept_labels(numerics).dtypes)


# ---------------------------------------------------------------------------
# windows_from_signal (shared with MIMIC-IV-WDB preprocessing)
# ---------------------------------------------------------------------------


class TestWindowsFromSignal:
    def test_geometry_and_indices(self):
        sig = RNG.normal(size=(int(TARGET_FS * 61), 2))
        windows = windows_from_signal(
            sig, TARGET_FS, "ECG", ["II", "V"], "vdb0001", "vdb0001_0000", start_idx=5
        )
        assert [w.window_idx for w in windows] == [5, 6, 7]
        assert all(w.n_samples == 1875 and w.n_channels == 2 for w in windows)
        assert all(w.channel_names == ["II", "V"] for w in windows)
        assert all(w.quality_ok for w in windows)
        assert windows[1].window_start_sec == pytest.approx(938 / TARGET_FS)

    def test_abp_keeps_population_scale(self):
        sig = np.full((int(TARGET_FS * 30), 1), 70.0)
        (window,) = windows_from_signal(sig, TARGET_FS, "ABP", ["ART"], "vdb0001", "vdb0001_0000")
        assert window.signal_mu_raw[0] == pytest.approx(ABP_POPULATION_MU_MMHG)
        assert window.signal_sigma_raw[0] == pytest.approx(ABP_POPULATION_SIGMA_MMHG)
        mmhg = window.signal * ABP_POPULATION_SIGMA_MMHG + ABP_POPULATION_MU_MMHG
        np.testing.assert_allclose(mmhg, 70.0, atol=1e-4)

    def test_nan_heavy_window_flagged(self):
        sig = RNG.normal(size=(int(TARGET_FS * 30), 1))
        sig[: int(TARGET_FS * 10)] = np.nan
        (window,) = windows_from_signal(sig, TARGET_FS, "PPG", ["Pleth"], "vdb0001", "vdb0001_0000")
        assert not window.quality_ok


def test_patient_id_prefix():
    assert patient_id_for(10) == "vdb0010"
    assert patient_id_for(6388) == "vdb6388"
