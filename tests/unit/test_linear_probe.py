"""
Unit tests for clbench.encoders.linear_probe and clbench.encoders.cache.

All tests use synthetic data — no real waveform files required.
Tests that need torch are automatically skipped in environments where torch
is not installed (e.g. the CI sandbox).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

torch = pytest.importorskip("torch")  # skip whole module if torch absent

from clbench.encoders.base import SignalEncoder
from clbench.encoders.linear_probe import (
    ProbeConfig,
    LinearProbeResult,
    _auroc,
    _auprc,
    _detect_heart_rate,
    _detect_resp_rate,
    _detect_abp_threshold,
    _CONCEPT_DETECTORS,
    extract_embeddings,
    label_windows,
    run_linear_probe,
)
from clbench.encoders.cache import (
    cache_encoder_features,
    load_cached_features,
    list_cached_patients,
)

DEVICE = torch.device("cpu")
T = 128   # short window; encoder works on any T
D = 64    # small d_model for fast tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder(in_channels: int = 3) -> SignalEncoder:
    """Tiny encoder for fast unit tests."""
    return SignalEncoder(
        in_channels=in_channels,
        d_model=D,
        n_heads=4,
        n_layers=1,
        ff_dim=128,
    )


def _sine_signal(fs: float, duration_s: float, freq_hz: float, n_ch: int = 1) -> np.ndarray:
    """Return (n_ch, n_samples) float32 sine wave."""
    n = int(fs * duration_s)
    t = np.arange(n) / fs
    sig = np.tile(np.sin(2 * np.pi * freq_hz * t), (n_ch, 1)).astype(np.float32)
    return sig


def _make_parquet(
    tmp_path: Path,
    patient_id: str,
    modality: str,
    n_windows: int,
    n_ch: int,
    n_samples: int,
    quality_ok: bool = True,
) -> Path:
    """Write a minimal synthetic parquet that the probe / cache modules can load."""
    import pandas as pd

    rows = []
    for i in range(n_windows):
        sig = np.random.randn(n_samples, n_ch).astype(np.float32)
        rows.append({
            "patient_id": patient_id,
            "segment_name": "seg0",
            "window_idx": i,
            "modality": modality,
            "channel_names": json.dumps([f"ch{c}" for c in range(n_ch)]),
            "fs": 62.5,
            "window_start_sec": float(i * 30),
            "window_end_sec": float(i * 30 + 30),
            "n_samples": n_samples,
            "n_channels": n_ch,
            "signal_bytes": sig.tobytes(),
            "signal_shape": json.dumps([n_samples, n_ch]),
            "signal_dtype": "float32",
            "quality_ok": quality_ok,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / f"{patient_id}.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Metric helpers: _auroc, _auprc
# ---------------------------------------------------------------------------

class TestAUROC:
    def test_perfect_separation(self) -> None:
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        labels = np.array([1, 1, 1, 0, 0, 0])
        assert _auroc(scores, labels) == pytest.approx(1.0)

    def test_random_guess(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        labels = rng.integers(0, 2, 200)
        # With 200 samples, AUROC should be near 0.5
        assert 0.35 < _auroc(scores, labels) < 0.65

    def test_inverted_returns_near_zero(self) -> None:
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])
        assert _auroc(scores, labels) < 0.2

    def test_no_positive_returns_nan(self) -> None:
        assert np.isnan(_auroc(np.array([0.5, 0.6]), np.array([0, 0])))

    def test_no_negative_returns_nan(self) -> None:
        assert np.isnan(_auroc(np.array([0.5, 0.6]), np.array([1, 1])))


class TestAUPRC:
    def test_perfect_separation(self) -> None:
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        assert _auprc(scores, labels) == pytest.approx(1.0, abs=0.05)

    def test_no_positive_returns_nan(self) -> None:
        assert np.isnan(_auprc(np.array([0.5]), np.array([0])))

    def test_positive_auprc(self) -> None:
        rng = np.random.default_rng(1)
        scores = rng.random(100)
        labels = (scores > 0.5).astype(int)
        auprc = _auprc(scores, labels)
        assert 0 < auprc <= 1.0


# ---------------------------------------------------------------------------
# Rule-based detectors
# ---------------------------------------------------------------------------

class TestDetectHeartRate:
    def test_tachycardia_sine_at_150bpm(self) -> None:
        """150 bpm = 2.5 Hz exceeds the lower tachycardia bound."""
        fs = 250.0
        sig = _sine_signal(fs, 10.0, freq_hz=2.5, n_ch=1)
        assert _detect_heart_rate(sig, fs, low_bpm=100) is True
        assert _CONCEPT_DETECTORS["tachycardia"][1](sig, fs) is True

    def test_normal_rhythm_not_tachycardia(self) -> None:
        """75 bpm = 1.25 Hz → should NOT trigger tachycardia detector."""
        fs = 250.0
        sig = _sine_signal(fs, 10.0, freq_hz=1.25, n_ch=1)
        assert _detect_heart_rate(sig, fs, low_bpm=100) is False
        assert _CONCEPT_DETECTORS["tachycardia"][1](sig, fs) is False

    def test_all_nan_returns_false(self) -> None:
        sig = np.full((1, 500), np.nan, dtype=np.float32)
        assert _detect_heart_rate(sig, 250.0, high_bpm=100) is False

    def test_constant_signal_returns_false(self) -> None:
        sig = np.ones((1, 500), dtype=np.float32)
        assert _detect_heart_rate(sig, 250.0, high_bpm=100) is False


class TestDetectRespRate:
    def test_tachypnea_at_30brpm(self) -> None:
        """30 brpm = 0.5 Hz exceeds the lower tachypnea bound."""
        fs = 50.0
        sig = _sine_signal(fs, 30.0, freq_hz=0.5, n_ch=1)
        assert _detect_resp_rate(sig, fs, low_brpm=20) is True
        assert _CONCEPT_DETECTORS["tachypnea"][1](sig, fs) is True

    def test_normal_rate_not_tachypnea(self) -> None:
        """15 brpm = 0.25 Hz → should NOT trigger tachypnea."""
        fs = 50.0
        sig = _sine_signal(fs, 30.0, freq_hz=0.25, n_ch=1)
        assert _detect_resp_rate(sig, fs, low_brpm=20) is False
        assert _CONCEPT_DETECTORS["tachypnea"][1](sig, fs) is False


class TestConceptDetectors:
    def test_all_named_concepts_have_detectors(self) -> None:
        required = {"tachycardia", "bradycardia", "tachypnea", "hypotension", "hypertension"}
        assert required.issubset(set(_CONCEPT_DETECTORS.keys()))

    def test_detectors_return_bool(self) -> None:
        fs = 250.0
        sig = _sine_signal(fs, 10.0, freq_hz=1.5, n_ch=1)
        for name, (_, detector) in _CONCEPT_DETECTORS.items():
            result = detector(sig, fs)
            assert isinstance(result, bool), f"Detector '{name}' did not return bool"


# ---------------------------------------------------------------------------
# extract_embeddings
# ---------------------------------------------------------------------------

class TestExtractEmbeddings:
    def test_output_shape(self) -> None:
        enc = _make_encoder(in_channels=3)
        enc.eval()
        signals = [np.random.randn(3, T).astype(np.float32) for _ in range(10)]
        embs = extract_embeddings(enc, signals, DEVICE, batch_size=4)
        assert embs.shape == (10, D)

    def test_output_dtype_float32(self) -> None:
        enc = _make_encoder(in_channels=1)
        enc.eval()
        signals = [np.random.randn(1, T).astype(np.float32) for _ in range(5)]
        embs = extract_embeddings(enc, signals, DEVICE)
        assert embs.dtype == np.float32

    def test_single_window(self) -> None:
        enc = _make_encoder(in_channels=1)
        enc.eval()
        signals = [np.random.randn(1, T).astype(np.float32)]
        embs = extract_embeddings(enc, signals, DEVICE)
        assert embs.shape == (1, D)

    def test_frozen_encoder_no_grad(self) -> None:
        """extract_embeddings must not accumulate gradients."""
        enc = _make_encoder()
        enc.eval()
        signals = [np.random.randn(3, T).astype(np.float32) for _ in range(3)]
        extract_embeddings(enc, signals, DEVICE)
        for p in enc.parameters():
            assert p.grad is None


# ---------------------------------------------------------------------------
# LinearProbeResult
# ---------------------------------------------------------------------------

class TestLinearProbeResult:
    def test_passed_when_auroc_above_gate(self) -> None:
        r = LinearProbeResult(
            concept="tachycardia", modality="ECG",
            n_windows=100, n_positive=20, positive_frac=0.2,
            auroc=0.80, auprc=0.65, min_auroc_gate=0.70, passed=True,
        )
        assert r.passed is True

    def test_failed_when_auroc_below_gate(self) -> None:
        r = LinearProbeResult(
            concept="tachycardia", modality="ECG",
            n_windows=100, n_positive=20, positive_frac=0.2,
            auroc=0.55, auprc=0.40, min_auroc_gate=0.70, passed=False,
        )
        assert r.passed is False

    def test_str_contains_auroc(self) -> None:
        r = LinearProbeResult(
            concept="tachycardia", modality="ECG",
            n_windows=50, n_positive=10, positive_frac=0.2,
            auroc=0.75, auprc=0.60, min_auroc_gate=0.70, passed=True,
        )
        assert "0.750" in str(r)

    def test_to_dict_has_all_fields(self) -> None:
        r = LinearProbeResult(
            concept="tachycardia", modality="ECG",
            n_windows=50, n_positive=10, positive_frac=0.2,
            auroc=0.75, auprc=0.60, min_auroc_gate=0.70, passed=True,
        )
        d = r.to_dict()
        assert "auroc" in d and "passed" in d and "concept" in d


# ---------------------------------------------------------------------------
# run_linear_probe (end-to-end, synthetic data)
# ---------------------------------------------------------------------------

class TestRunLinearProbe:
    def _make_processed_dir(self, tmp_path: Path, n_patients: int = 10) -> Path:
        """Write synthetic parquets with tachycardia-compatible signal shapes."""
        for i in range(n_patients):
            _make_parquet(
                tmp_path, f"p{i:04d}", "ECG",
                n_windows=20, n_ch=3, n_samples=T,
            )
        return tmp_path

    def test_runs_without_error(self, tmp_path: Path) -> None:
        proc_dir = self._make_processed_dir(tmp_path)
        enc = _make_encoder(in_channels=3)
        cfg = ProbeConfig(concept="tachycardia", min_auroc=0.0, epochs=2, batch_size=16)
        result = run_linear_probe(
            encoder=enc,
            processed_dir=proc_dir,
            patient_ids=[f"p{i:04d}" for i in range(10)],
            modality="ECG",
            concept="tachycardia",
            device=DEVICE,
            cfg=cfg,
        )
        assert isinstance(result, LinearProbeResult)

    def test_result_has_valid_auroc_range(self, tmp_path: Path) -> None:
        proc_dir = self._make_processed_dir(tmp_path)
        enc = _make_encoder(in_channels=3)
        cfg = ProbeConfig(concept="tachycardia", min_auroc=0.0, epochs=2, batch_size=16)
        result = run_linear_probe(
            encoder=enc, processed_dir=proc_dir,
            patient_ids=[f"p{i:04d}" for i in range(10)],
            modality="ECG", concept="tachycardia",
            device=DEVICE, cfg=cfg,
        )
        if not np.isnan(result.auroc):
            assert 0.0 <= result.auroc <= 1.0

    def test_too_few_windows_returns_nan(self, tmp_path: Path) -> None:
        """With no parquets, probe should return NaN result (not crash)."""
        enc = _make_encoder(in_channels=3)
        cfg = ProbeConfig(concept="tachycardia", min_auroc=0.70, epochs=2)
        result = run_linear_probe(
            encoder=enc, processed_dir=tmp_path,
            patient_ids=["nonexistent"],
            modality="ECG", concept="tachycardia",
            device=DEVICE, cfg=cfg,
        )
        assert np.isnan(result.auroc)
        assert result.passed is False

    def test_unknown_concept_raises(self, tmp_path: Path) -> None:
        proc_dir = self._make_processed_dir(tmp_path, n_patients=2)
        enc = _make_encoder()
        cfg = ProbeConfig(concept="unknown_xyz", min_auroc=0.0, epochs=1)
        with pytest.raises(ValueError, match="Unknown concept"):
            run_linear_probe(
                encoder=enc, processed_dir=proc_dir,
                patient_ids=["p0000"],
                modality="ECG", concept="unknown_xyz",
                device=DEVICE, cfg=cfg,
            )


# ---------------------------------------------------------------------------
# cache_encoder_features / load_cached_features
# ---------------------------------------------------------------------------

class TestEncoderCache:
    def test_cache_creates_pt_files(self, tmp_path: Path) -> None:
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        _make_parquet(proc_dir, "p001", "ECG", n_windows=5, n_ch=3, n_samples=T)

        cache_dir = tmp_path / "cache"
        enc = _make_encoder(in_channels=3)
        written = cache_encoder_features(
            encoder=enc,
            processed_dir=proc_dir,
            patient_ids=["p001"],
            modality="ECG",
            device=DEVICE,
            cache_dir=cache_dir,
        )
        assert "p001" in written
        assert written["p001"].exists()

    def test_cached_tensor_shape(self, tmp_path: Path) -> None:
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        _make_parquet(proc_dir, "p001", "ECG", n_windows=5, n_ch=3, n_samples=T)

        cache_dir = tmp_path / "cache"
        enc = _make_encoder(in_channels=3)
        cache_encoder_features(
            encoder=enc, processed_dir=proc_dir,
            patient_ids=["p001"], modality="ECG",
            device=DEVICE, cache_dir=cache_dir,
        )
        features = load_cached_features("p001", "ECG", cache_dir=cache_dir)
        assert features.shape == (5, D)
        assert features.dtype == torch.float32

    def test_overwrite_false_skips_existing(self, tmp_path: Path) -> None:
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        _make_parquet(proc_dir, "p001", "ECG", n_windows=3, n_ch=3, n_samples=T)

        cache_dir = tmp_path / "cache"
        enc = _make_encoder(in_channels=3)
        # First call writes
        cache_encoder_features(
            encoder=enc, processed_dir=proc_dir,
            patient_ids=["p001"], modality="ECG",
            device=DEVICE, cache_dir=cache_dir, overwrite=False,
        )
        pt_files_before = list(cache_dir.rglob("*.pt"))
        mtime_before = pt_files_before[0].stat().st_mtime

        # Second call (overwrite=False) must NOT re-write
        import time
        time.sleep(0.05)
        cache_encoder_features(
            encoder=enc, processed_dir=proc_dir,
            patient_ids=["p001"], modality="ECG",
            device=DEVICE, cache_dir=cache_dir, overwrite=False,
        )
        mtime_after = pt_files_before[0].stat().st_mtime
        assert mtime_before == mtime_after, "File was re-written despite overwrite=False"

    def test_load_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cached_features("nonexistent", "ECG", cache_dir=tmp_path)

    def test_list_cached_patients(self, tmp_path: Path) -> None:
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        for pid in ["p001", "p002", "p003"]:
            _make_parquet(proc_dir, pid, "ECG", n_windows=2, n_ch=3, n_samples=T)

        cache_dir = tmp_path / "cache"
        enc = _make_encoder(in_channels=3)
        cache_encoder_features(
            encoder=enc, processed_dir=proc_dir,
            patient_ids=["p001", "p002", "p003"], modality="ECG",
            device=DEVICE, cache_dir=cache_dir,
        )
        patients = list_cached_patients("ECG", cache_dir=cache_dir)
        assert set(patients) == {"p001", "p002", "p003"}

    def test_missing_patient_skipped_gracefully(self, tmp_path: Path) -> None:
        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        _make_parquet(proc_dir, "p001", "ECG", n_windows=2, n_ch=3, n_samples=T)

        cache_dir = tmp_path / "cache"
        enc = _make_encoder(in_channels=3)
        written = cache_encoder_features(
            encoder=enc, processed_dir=proc_dir,
            patient_ids=["p001", "p_does_not_exist"], modality="ECG",
            device=DEVICE, cache_dir=cache_dir,
        )
        assert "p001" in written
        assert "p_does_not_exist" not in written
