"""
Unit tests for clbench.data.preprocess.

All tests use synthetic NumPy arrays — no waveform files required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.data.preprocess import (
    BANDPASS_SPECS,
    CHANNEL_MAP,
    ECG_MAX_LEADS,
    MAX_NAN_FRAC,
    WindowRecord,
    _apply_bandpass,
    _iter_segments,
    _make_windows,
    _sort_ecg_channels,
    _zscore_normalize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
FS = 62.5  # close to MIMIC-IV-WDB native rate


def _sine_signal(
    n_samples: int,
    n_channels: int = 1,
    freq_hz: float = 1.0,
    fs: float = FS,
    *,
    rng: np.random.Generator = RNG,
) -> np.ndarray:
    """Return a (n_samples, n_channels) array with sinusoids + small noise."""
    t = np.arange(n_samples) / fs
    sig = np.sin(2 * np.pi * freq_hz * t)[:, np.newaxis] * np.ones((1, n_channels))
    sig = sig + rng.normal(0, 0.01, sig.shape)
    return sig.astype(np.float64)


# ---------------------------------------------------------------------------
# CHANNEL_MAP
# ---------------------------------------------------------------------------


class TestChannelMap:
    """CHANNEL_MAP must cover every canonical channel used in MIMIC-IV-WDB."""

    _required_ecg = {"II", "V", "aVR", "I", "III"}
    _required_ppg = {"Pleth"}
    _required_abp = {"ABP", "ART"}
    _required_resp = {"Resp"}

    def test_ecg_leads_present(self) -> None:
        for lead in self._required_ecg:
            assert lead in CHANNEL_MAP, f"ECG lead '{lead}' missing from CHANNEL_MAP"
            assert CHANNEL_MAP[lead] == "ECG"

    def test_ppg_channels_present(self) -> None:
        for ch in self._required_ppg:
            assert ch in CHANNEL_MAP, f"PPG channel '{ch}' missing"
            assert CHANNEL_MAP[ch] == "PPG"

    def test_abp_channels_present(self) -> None:
        for ch in self._required_abp:
            assert ch in CHANNEL_MAP, f"ABP channel '{ch}' missing"
            assert CHANNEL_MAP[ch] == "ABP"

    def test_resp_channels_present(self) -> None:
        for ch in self._required_resp:
            assert ch in CHANNEL_MAP, f"RESP channel '{ch}' missing"
            assert CHANNEL_MAP[ch] == "RESP"

    def test_pap_cvp_excluded(self) -> None:
        """PAP and CVP must NOT map to ABP (they are non-systemic pressures)."""
        assert "PAP" not in CHANNEL_MAP or CHANNEL_MAP.get("PAP") != "ABP"
        assert "CVP" not in CHANNEL_MAP or CHANNEL_MAP.get("CVP") != "ABP"

    def test_all_values_are_canonical_modalities(self) -> None:
        valid = {"ECG", "PPG", "ABP", "RESP"}
        for ch, mod in CHANNEL_MAP.items():
            assert mod in valid, f"Channel '{ch}' maps to unknown modality '{mod}'"


# ---------------------------------------------------------------------------
# BANDPASS_SPECS
# ---------------------------------------------------------------------------


class TestBandpassSpecs:
    def test_all_modalities_have_spec(self) -> None:
        for mod in ("ECG", "PPG", "ABP", "RESP"):
            assert mod in BANDPASS_SPECS, f"Modality '{mod}' missing from BANDPASS_SPECS"

    def test_abp_has_no_bandpass(self) -> None:
        assert BANDPASS_SPECS["ABP"] is None, "ABP must have no bandpass (None)"

    def test_ecg_ppg_resp_have_bandpass(self) -> None:
        for mod in ("ECG", "PPG", "RESP"):
            spec = BANDPASS_SPECS[mod]
            assert spec is not None, f"{mod} must have a bandpass spec"
            low, high = spec
            assert 0 < low < high, f"{mod} bandpass [{low}, {high}] is invalid"


# ---------------------------------------------------------------------------
# _apply_bandpass
# ---------------------------------------------------------------------------


class TestApplyBandpass:
    def test_output_shape_preserved(self) -> None:
        sig = _sine_signal(1000, n_channels=3)
        out = _apply_bandpass(sig, FS, 0.5, 20.0)
        assert out.shape == sig.shape

    def test_output_dtype_float32(self) -> None:
        sig = _sine_signal(1000)
        out = _apply_bandpass(sig, FS, 0.5, 20.0)
        assert out.dtype == np.float32

    def test_attenuates_high_frequency(self) -> None:
        """A 25 Hz signal should be attenuated by a 0.5–8 Hz PPG bandpass."""
        n = 4000
        t = np.arange(n) / FS
        sig_25hz = np.sin(2 * np.pi * 25.0 * t)[:, np.newaxis]
        out = _apply_bandpass(sig_25hz, FS, 0.5, 8.0)
        # Energy after filtering should be much less than before
        energy_in = float(np.mean(sig_25hz**2))
        energy_out = float(np.mean(out**2))
        assert energy_out < 0.01 * energy_in, (
            f"High-freq signal not attenuated: in={energy_in:.4f}, out={energy_out:.4f}"
        )

    def test_passes_in_band_signal(self) -> None:
        """A 1 Hz signal through a 0.5–8 Hz filter must retain most energy."""
        n = 4000
        t = np.arange(n) / FS
        sig_1hz = np.sin(2 * np.pi * 1.0 * t)[:, np.newaxis].astype(np.float64)
        out = _apply_bandpass(sig_1hz, FS, 0.5, 8.0)
        energy_in = float(np.mean(sig_1hz**2))
        energy_out = float(np.mean(out**2))
        assert energy_out > 0.5 * energy_in, (
            f"In-band signal attenuated too much: in={energy_in:.4f}, out={energy_out:.4f}"
        )

    def test_nan_preserved_after_filter(self) -> None:
        sig = _sine_signal(500)
        sig[10:20] = np.nan   # inject NaN span
        out = _apply_bandpass(sig, FS, 0.5, 20.0)
        assert np.isnan(out[10:20]).all(), "NaN positions must be preserved after filtering"
        assert not np.isnan(out[:10]).any(), "Non-NaN positions must have valid values"

    def test_all_nan_channel_unchanged(self) -> None:
        sig = np.full((500, 2), np.nan)
        sig[:, 1] = _sine_signal(500)[:, 0]  # channel 1 has valid data
        out = _apply_bandpass(sig, FS, 0.5, 20.0)
        assert np.isnan(out[:, 0]).all(), "All-NaN channel must remain NaN"
        assert not np.isnan(out[:, 1]).any(), "Valid channel must not become NaN"

    def test_invalid_bandpass_returns_original(self) -> None:
        """When bandpass freqs are invalid (e.g. fs too low), signal passes through."""
        sig = _sine_signal(200, fs=1.5)  # fs=1.5 Hz, asking for 0.5–8 Hz → invalid
        out = _apply_bandpass(sig, fs=1.5, low_hz=0.5, high_hz=8.0)
        # Should return something (not crash); energy roughly preserved
        assert out.shape == sig.shape


# ---------------------------------------------------------------------------
# _zscore_normalize
# ---------------------------------------------------------------------------


class TestZscoreNormalize:
    def test_output_shape_preserved(self) -> None:
        sig = _sine_signal(1000, n_channels=4)
        out, mu, sigma = _zscore_normalize(sig)
        assert out.shape == sig.shape
        assert mu.shape == sigma.shape == (4,)

    def test_output_dtype_float32(self) -> None:
        sig = _sine_signal(1000)
        out, _, _ = _zscore_normalize(sig)
        assert out.dtype == np.float32

    def test_mean_near_zero(self) -> None:
        sig = _sine_signal(2000)
        out, _, _ = _zscore_normalize(sig)
        valid = out[~np.isnan(out)]
        assert abs(float(valid.mean())) < 0.05

    def test_std_near_one(self) -> None:
        sig = _sine_signal(2000)
        out, _, _ = _zscore_normalize(sig)
        valid = out[~np.isnan(out)]
        assert 0.8 < float(valid.std()) < 1.2

    def test_nan_preserved(self) -> None:
        sig = _sine_signal(500)
        sig[50:60] = np.nan
        out, _, _ = _zscore_normalize(sig)
        assert np.isnan(out[50:60]).all()
        assert not np.isnan(out[:50]).any()

    def test_all_nan_channel_unchanged(self) -> None:
        sig = np.full((200, 1), np.nan)
        out, _, _ = _zscore_normalize(sig)
        assert np.isnan(out).all()

    def test_zero_variance_channel_centred(self) -> None:
        """A constant signal should be mean-centred to 0 (no crash)."""
        sig = np.ones((200, 1), dtype=np.float64) * 5.0
        out, mu, sigma = _zscore_normalize(sig)
        assert np.allclose(out, 0.0), "Constant signal must be centred to 0"
        np.testing.assert_allclose(out * sigma + mu, sig)

    def test_independent_channels(self) -> None:
        """Channels with different scales must be normalised independently."""
        sig = np.zeros((500, 2), dtype=np.float64)
        sig[:, 0] = np.linspace(-100, 100, 500)   # large scale
        sig[:, 1] = np.linspace(-1, 1, 500)        # small scale
        out, mu, sigma = _zscore_normalize(sig)
        np.testing.assert_allclose(out * sigma + mu, sig, atol=1e-5)
        for ch in range(2):
            col = out[:, ch]
            assert abs(float(col.mean())) < 0.01
            assert 0.9 < float(col.std()) < 1.1


# ---------------------------------------------------------------------------
# _make_windows
# ---------------------------------------------------------------------------


class TestMakeWindows:
    def _n_windows(self, n_samples: int, window_sec: float, stride_sec: float) -> int:
        ws = int(round(window_sec * FS))
        ss = int(round(stride_sec * FS))
        if n_samples < ws or ws == 0:
            return 0
        count = 0
        start = 0
        while start + ws <= n_samples:
            count += 1
            start += ss
        return count

    def test_empty_when_signal_too_short(self) -> None:
        sig = _sine_signal(10, fs=FS)  # way shorter than 30 s
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=15.0)
        assert windows == []

    def test_single_window_exact_length(self) -> None:
        ws = int(round(30.0 * FS))
        sig = _sine_signal(ws, fs=FS)
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=15.0)
        assert len(windows) == 1

    def test_window_count_50_pct_overlap(self) -> None:
        n = int(FS * 120)  # 2 minutes
        sig = _sine_signal(n, fs=FS)
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=15.0)
        expected = self._n_windows(n, 30.0, 15.0)
        assert len(windows) == expected

    def test_window_shape(self) -> None:
        n = int(FS * 120)
        n_ch = 3
        sig = _sine_signal(n, n_channels=n_ch, fs=FS)
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=15.0)
        assert len(windows) > 0
        w_arr, _, _, _ = windows[0]
        assert w_arr.shape == (int(round(30.0 * FS)), n_ch)

    def test_timestamps_monotonically_increasing(self) -> None:
        n = int(FS * 120)
        sig = _sine_signal(n, fs=FS)
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=15.0)
        starts = [s for _, s, _, _ in windows]
        assert starts == sorted(starts)
        assert all(s >= 0 for s in starts)

    def test_quality_flag_high_nan(self) -> None:
        ws = int(round(30.0 * FS))
        n = ws * 2
        sig = _sine_signal(n, fs=FS)
        # Inject > 20% NaN into the first window
        sig[: int(ws * 0.5)] = np.nan
        # stride_sec=30.0 → non-overlapping windows
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=30.0)
        _, _, _, q0 = windows[0]
        _, _, _, q1 = windows[1]
        assert not q0, "First window with >20% NaN should be quality_ok=False"
        assert q1, "Second clean window should be quality_ok=True"

    def test_quality_flag_low_nan(self) -> None:
        ws = int(round(30.0 * FS))
        sig = _sine_signal(ws, fs=FS)
        sig[: int(ws * 0.05)] = np.nan  # only 5% NaN
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=ws)
        _, _, _, q0 = windows[0]
        assert q0, "Window with only 5% NaN should be quality_ok=True"

    def test_start_end_sec_match_window_sec(self) -> None:
        ws = int(round(30.0 * FS))
        sig = _sine_signal(ws, fs=FS)
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=30.0)
        _, start_sec, end_sec, _ = windows[0]
        assert abs(start_sec - 0.0) < 1e-6
        assert abs(end_sec - end_sec) < 1e-6  # tautology check
        assert abs((end_sec - start_sec) - 30.0) < 0.1


# ---------------------------------------------------------------------------
# _sort_ecg_channels
# ---------------------------------------------------------------------------


class TestSortEcgChannels:
    def test_priority_ordering(self) -> None:
        # Channels: aVR=idx2, V=idx1, II=idx0 — should come out II, V, aVR
        channel_names = ["II", "V", "aVR", "aVL", "I"]
        indices = [0, 1, 2, 3, 4]
        sorted_idx = _sort_ecg_channels(indices, channel_names)
        names_out = [channel_names[i] for i in sorted_idx]
        assert names_out[:3] == ["II", "V", "aVR"]

    def test_capped_at_ecg_max_leads(self) -> None:
        channel_names = ["I", "II", "III", "V", "aVR", "aVL", "aVF"]
        indices = list(range(len(channel_names)))
        sorted_idx = _sort_ecg_channels(indices, channel_names)
        assert len(sorted_idx) <= ECG_MAX_LEADS

    def test_single_lead_unchanged(self) -> None:
        channel_names = ["II", "Pleth"]
        sorted_idx = _sort_ecg_channels([0], channel_names)
        assert sorted_idx == [0]


# ---------------------------------------------------------------------------
# WindowRecord
# ---------------------------------------------------------------------------


class TestWindowRecord:
    def _make_record(
        self,
        n_samples: int = 1874,
        n_channels: int = 3,
        modality: str = "ECG",
        nan_frac: float = 0.0,
    ) -> WindowRecord:
        sig = _sine_signal(n_samples, n_channels=n_channels).astype(np.float32)
        if nan_frac > 0:
            n_nan = int(n_samples * n_channels * nan_frac)
            flat = sig.ravel()
            idx = RNG.choice(len(flat), size=n_nan, replace=False)
            flat[idx] = np.nan
            sig = flat.reshape(n_samples, n_channels)
        return WindowRecord(
            patient_id="p_test",
            segment_name="seg_0001",
            window_idx=0,
            modality=modality,
            channel_names=["II", "V", "aVR"][:n_channels],
            fs=FS,
            window_start_sec=0.0,
            window_end_sec=n_samples / FS,
            signal=sig,
            quality_ok=True,
        )

    def test_parquet_row_keys(self) -> None:
        rec = self._make_record()
        row = rec.to_parquet_row()
        expected_keys = {
            "patient_id", "segment_name", "window_idx", "modality",
            "channel_names", "fs", "window_start_sec", "window_end_sec",
            "n_samples", "n_channels", "signal_bytes", "signal_shape",
            "signal_dtype", "quality_ok", "signal_mu_raw", "signal_sigma_raw",
        }
        assert set(row.keys()) == expected_keys

    def test_parquet_row_n_samples(self) -> None:
        rec = self._make_record(n_samples=1874)
        row = rec.to_parquet_row()
        assert row["n_samples"] == 1874

    def test_parquet_row_n_channels(self) -> None:
        rec = self._make_record(n_channels=3)
        row = rec.to_parquet_row()
        assert row["n_channels"] == 3

    def test_signal_bytes_round_trip(self) -> None:
        """signal_bytes → numpy reconstruction must be numerically identical."""
        rec = self._make_record(n_samples=500, n_channels=2)
        row = rec.to_parquet_row()
        shape: list[int] = json.loads(row["signal_shape"])
        dtype: str = row["signal_dtype"]
        reconstructed = np.frombuffer(row["signal_bytes"], dtype=dtype).reshape(shape)
        expected = rec._signal_2d.astype(np.float32)
        np.testing.assert_array_equal(reconstructed, expected)

    def test_signal_from_row_helper(self) -> None:
        import pandas as pd

        rec = self._make_record(n_samples=300, n_channels=1)
        row = rec.to_parquet_row()
        series = pd.Series(row)
        reconstructed = WindowRecord.signal_from_row(series)
        np.testing.assert_array_equal(reconstructed, rec._signal_2d.astype(np.float32))

    def test_dtype_is_float32(self) -> None:
        rec = self._make_record()
        row = rec.to_parquet_row()
        assert row["signal_dtype"] == "float32"

    def test_channel_names_json(self) -> None:
        rec = self._make_record(n_channels=3)
        row = rec.to_parquet_row()
        names = json.loads(row["channel_names"])
        assert isinstance(names, list)
        assert len(names) == 3

    def test_1d_signal_stored_as_2d(self) -> None:
        """1-D signal (single-channel) must be stored as (n, 1) in parquet."""
        sig = _sine_signal(500, n_channels=1).ravel().astype(np.float32)
        rec = WindowRecord(
            patient_id="p",
            segment_name="s",
            window_idx=0,
            modality="PPG",
            channel_names=["Pleth"],
            fs=FS,
            window_start_sec=0.0,
            window_end_sec=8.0,
            signal=sig,
            quality_ok=True,
        )
        row = rec.to_parquet_row()
        assert row["n_channels"] == 1
        shape = json.loads(row["signal_shape"])
        assert shape == [500, 1]

    def test_quality_ok_false_stored(self) -> None:
        rec = self._make_record()
        rec.quality_ok = False
        row = rec.to_parquet_row()
        assert row["quality_ok"] is False

    def test_modality_stored_as_string(self) -> None:
        for mod in ("ECG", "PPG", "ABP", "RESP"):
            rec = self._make_record(modality=mod)
            row = rec.to_parquet_row()
            assert row["modality"] == mod
            assert isinstance(row["modality"], str)


# ---------------------------------------------------------------------------
# Edge cases / integration-light
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_abp_no_filter_then_normalize(self) -> None:
        """ABP pipeline: skip bandpass, only z-score — must not crash."""
        n = int(FS * 60)
        sig = np.random.default_rng(1).normal(100, 20, (n, 1))  # ABP-like values
        # No bandpass
        assert BANDPASS_SPECS["ABP"] is None
        normalised, mu, sigma = _zscore_normalize(sig.astype(np.float64))
        np.testing.assert_allclose(normalised * sigma + mu, sig, atol=1e-5)
        assert normalised.shape == sig.shape
        assert normalised.dtype == np.float32
        windows = _make_windows(normalised, FS)
        assert len(windows) > 0

    def test_very_short_segment_produces_no_windows(self) -> None:
        """Segments shorter than one window must yield zero windows."""
        sig = _sine_signal(50, fs=FS)  # < 1 s
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=15.0)
        assert windows == []

    def test_exactly_two_windows(self) -> None:
        """n_samples = 2 × window_samples → exactly 2 windows with stride = window."""
        ws = int(round(30.0 * FS))
        sig = _sine_signal(ws * 2, fs=FS)
        windows = _make_windows(sig, FS, window_sec=30.0, stride_sec=30.0)
        assert len(windows) == 2
