"""
Unit tests for CLBench Phase 1.5 signal encoders.

All tests use synthetic tensors — no real data files required.

Coverage
--------
TestSinusoidalPosEnc      (3 tests)
TestConvStem               (4 tests)
TestSignalEncoder          (7 tests)
TestPerModalityEncoders    (5 tests)
TestMaskedSignalEncoder    (6 tests)
TestWindowDataset          (4 tests)
TestCheckpointIO           (3 tests)
TestSplitLogic             (6 tests)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.data.split import _stratified_split
from clbench.encoders.abp import ABPEncoder
from clbench.encoders.base import (
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    ConvStem,
    SignalEncoder,
    SinusoidalPosEnc,
)
from clbench.encoders.ecg import ECG_IN_CHANNELS, ECGEncoder
from clbench.encoders.ppg import PPGEncoder
from clbench.encoders.pretraining import (
    MASK_RATIO,
    MaskedSignalEncoder,
    WindowDataset,
    load_encoder,
    save_encoder,
)
from clbench.encoders.resp import RESPEncoder

# ---------------------------------------------------------------------------
# Constants for synthetic signals
# ---------------------------------------------------------------------------

BATCH = 2
T = 128         # short window so tests run fast; T' ≈ 8 after 4x stride-2 blocks
ECG_CH = 3
MONO_CH = 1
D = D_MODEL     # 256


# ---------------------------------------------------------------------------
# Helper: compute T' for a given T through the ConvStem
# ---------------------------------------------------------------------------

def _t_prime(t: int) -> int:
    """Pass a dummy tensor through ConvStem and return the sequence length."""
    stem = ConvStem(in_channels=1, d_model=D)
    stem.eval()
    with torch.no_grad():
        out = stem(torch.zeros(1, 1, t))
    return int(out.shape[-1])


# ---------------------------------------------------------------------------
# TestSinusoidalPosEnc
# ---------------------------------------------------------------------------


class TestSinusoidalPosEnc:
    def test_output_shape_preserved(self) -> None:
        """Shape (B, T, d) should be unchanged."""
        T_seq = 32
        enc = SinusoidalPosEnc(d_model=D)
        x = torch.zeros(BATCH, T_seq, D)
        out = enc(x)
        assert out.shape == (BATCH, T_seq, D)

    def test_no_learnable_params(self) -> None:
        """Sinusoidal encoding must be parameter-free."""
        enc = SinusoidalPosEnc(d_model=D)
        assert len(list(enc.parameters())) == 0

    def test_different_positions_differ(self) -> None:
        """Output at position 0 and position 1 should differ (encoding is position-dependent)."""
        enc = SinusoidalPosEnc(d_model=D)
        x = torch.zeros(1, 10, D)
        out = enc(x)
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# TestConvStem
# ---------------------------------------------------------------------------


class TestConvStem:
    def test_output_shape_channels(self) -> None:
        """Channel dim must equal d_model."""
        stem = ConvStem(in_channels=ECG_CH, d_model=D)
        x = torch.randn(BATCH, ECG_CH, T)
        out = stem(x)
        assert out.shape[0] == BATCH
        assert out.shape[1] == D

    def test_output_shape_time_reduced(self) -> None:
        """Time dim must be strictly less than T (4x stride-2 blocks)."""
        stem = ConvStem(in_channels=1, d_model=D)
        x = torch.randn(BATCH, 1, T)
        out = stem(x)
        assert out.shape[2] < T // 2

    def test_mono_and_multi_channel_input(self) -> None:
        """Works for in_channels=1 and in_channels=3."""
        for ch in (1, 3):
            stem = ConvStem(in_channels=ch, d_model=D)
            out = stem(torch.randn(BATCH, ch, T))
            assert out.shape[1] == D

    def test_parameters_exist(self) -> None:
        """ConvStem must have learnable parameters."""
        stem = ConvStem(in_channels=1, d_model=D)
        assert sum(p.numel() for p in stem.parameters()) > 0


# ---------------------------------------------------------------------------
# TestSignalEncoder
# ---------------------------------------------------------------------------


class TestSignalEncoder:
    @pytest.fixture
    def ecg_encoder(self) -> SignalEncoder:
        return SignalEncoder(in_channels=ECG_CH, d_model=D, n_heads=N_HEADS, n_layers=N_LAYERS)

    @pytest.fixture
    def mono_encoder(self) -> SignalEncoder:
        return SignalEncoder(in_channels=MONO_CH, d_model=D, n_heads=N_HEADS, n_layers=N_LAYERS)

    def test_forward_output_shape(self, ecg_encoder: SignalEncoder) -> None:
        """forward() → (B, d_model)."""
        x = torch.randn(BATCH, ECG_CH, T)
        out = ecg_encoder(x)
        assert out.shape == (BATCH, D)

    def test_encode_equals_forward(self, ecg_encoder: SignalEncoder) -> None:
        """encode() must be an alias for forward()."""
        x = torch.randn(BATCH, ECG_CH, T)
        ecg_encoder.eval()
        with torch.no_grad():
            assert torch.allclose(ecg_encoder.forward(x), ecg_encoder.encode(x))

    def test_extract_tokens_shape(self, ecg_encoder: SignalEncoder) -> None:
        """extract_tokens() → (B, T', d_model) with T' < T."""
        x = torch.randn(BATCH, ECG_CH, T)
        tokens = ecg_encoder.extract_tokens(x)
        assert tokens.dim() == 3
        assert tokens.shape[0] == BATCH
        assert tokens.shape[2] == D
        assert tokens.shape[1] < T

    def test_mono_encoder_forward(self, mono_encoder: SignalEncoder) -> None:
        """Single-channel encoder works end-to-end."""
        x = torch.randn(BATCH, MONO_CH, T)
        out = mono_encoder(x)
        assert out.shape == (BATCH, D)

    def test_config_keys(self, ecg_encoder: SignalEncoder) -> None:
        """config() must return all expected constructor keys."""
        cfg = ecg_encoder.config()
        required = {"in_channels", "d_model", "n_heads", "n_layers", "ff_dim", "dropout", "max_seq_len"}
        assert required <= set(cfg.keys())

    def test_config_in_channels(self, ecg_encoder: SignalEncoder) -> None:
        """config() in_channels must match constructor arg."""
        assert ecg_encoder.config()["in_channels"] == ECG_CH

    def test_output_is_float32(self, ecg_encoder: SignalEncoder) -> None:
        """Encoder output must be float32."""
        x = torch.randn(BATCH, ECG_CH, T)
        out = ecg_encoder(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestPerModalityEncoders
# ---------------------------------------------------------------------------


class TestPerModalityEncoders:
    def test_ecg_encoder_in_channels(self) -> None:
        enc = ECGEncoder()
        assert enc.in_channels == ECG_IN_CHANNELS == 3

    def test_ppg_encoder_in_channels(self) -> None:
        enc = PPGEncoder()
        assert enc.in_channels == 1

    def test_abp_encoder_in_channels(self) -> None:
        enc = ABPEncoder()
        assert enc.in_channels == 1

    def test_resp_encoder_in_channels(self) -> None:
        enc = RESPEncoder()
        assert enc.in_channels == 1

    def test_all_encoders_forward(self) -> None:
        """All four modality encoders must produce (B, d_model) output."""
        encoders_and_channels = [
            (ECGEncoder(), ECG_CH),
            (PPGEncoder(), MONO_CH),
            (ABPEncoder(), MONO_CH),
            (RESPEncoder(), MONO_CH),
        ]
        for enc, ch in encoders_and_channels:
            x = torch.randn(BATCH, ch, T)
            out = enc(x)
            assert out.shape == (BATCH, D), f"{enc.__class__.__name__} forward shape wrong"


# ---------------------------------------------------------------------------
# TestMaskedSignalEncoder
# ---------------------------------------------------------------------------


class TestMaskedSignalEncoder:
    @pytest.fixture
    def mae(self) -> MaskedSignalEncoder:
        enc = SignalEncoder(in_channels=ECG_CH, d_model=D, n_heads=N_HEADS, n_layers=N_LAYERS)
        return MaskedSignalEncoder(enc, mask_ratio=MASK_RATIO)

    def test_forward_returns_loss_and_pred(self, mae: MaskedSignalEncoder) -> None:
        """forward() must return (scalar loss, pred tensor)."""
        x = torch.randn(BATCH, ECG_CH, T)
        loss, pred = mae(x)
        assert loss.ndim == 0   # scalar
        assert pred.ndim == 3   # (B, T', d_model)

    def test_loss_is_finite(self, mae: MaskedSignalEncoder) -> None:
        """Loss must be finite (not NaN or Inf)."""
        x = torch.randn(BATCH, ECG_CH, T)
        loss, _ = mae(x)
        assert torch.isfinite(loss).item()

    def test_loss_is_positive(self, mae: MaskedSignalEncoder) -> None:
        """MSE loss must be non-negative (and > 0 for random inputs)."""
        x = torch.randn(BATCH, ECG_CH, T)
        loss, _ = mae(x)
        assert float(loss.item()) >= 0.0

    def test_pred_shape_matches_tokens(self, mae: MaskedSignalEncoder) -> None:
        """Pred shape must match extract_tokens() shape."""
        x = torch.randn(BATCH, ECG_CH, T)
        tokens = mae.encoder.extract_tokens(x)
        _, pred = mae(x)
        assert pred.shape == tokens.shape

    def test_mask_token_is_learnable(self, mae: MaskedSignalEncoder) -> None:
        """mask_token must be a nn.Parameter."""
        assert isinstance(mae.mask_token, nn.Parameter)
        assert mae.mask_token.shape == (D,)

    def test_grad_flows_through_loss(self, mae: MaskedSignalEncoder) -> None:
        """Backward pass must not raise and must populate gradients."""
        x = torch.randn(BATCH, ECG_CH, T)
        loss, _ = mae(x)
        loss.backward()
        # At least one encoder parameter must have a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mae.encoder.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# TestWindowDataset
# ---------------------------------------------------------------------------


def _make_synthetic_parquet(tmp_path: Path, patient_id: str, modality: str, n_windows: int, n_ch: int, n_samples: int) -> Path:
    """Write a minimal synthetic parquet that WindowDataset can load."""
    rows = []
    for i in range(n_windows):
        sig = np.random.randn(n_samples, n_ch).astype(np.float32)
        rows.append({
            "patient_id": patient_id,
            "segment_name": "seg_0",
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
            "quality_ok": True,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / f"{patient_id}.parquet"
    df.to_parquet(path, index=False)
    return path


class TestWindowDataset:
    def test_len_correct(self, tmp_path: Path) -> None:
        _make_synthetic_parquet(tmp_path, "p001", "ECG", n_windows=5, n_ch=ECG_CH, n_samples=T)
        ds = WindowDataset(tmp_path, ["p001"], "ECG", quality_only=True)
        assert len(ds) == 5

    def test_item_shape(self, tmp_path: Path) -> None:
        """Each item should be (n_channels, n_samples) float32."""
        _make_synthetic_parquet(tmp_path, "p001", "PPG", n_windows=3, n_ch=1, n_samples=T)
        ds = WindowDataset(tmp_path, ["p001"], "PPG")
        item = ds[0]
        assert item.shape == (1, T)
        assert item.dtype == torch.float32

    def test_missing_patient_skipped(self, tmp_path: Path) -> None:
        """Patients whose parquets don't exist should be silently skipped."""
        _make_synthetic_parquet(tmp_path, "p001", "ECG", n_windows=2, n_ch=ECG_CH, n_samples=T)
        ds = WindowDataset(tmp_path, ["p001", "p_missing"], "ECG")
        assert len(ds) == 2

    def test_wrong_modality_excluded(self, tmp_path: Path) -> None:
        """Only rows with the requested modality should be included."""
        # p001 has ECG windows, but we request PPG
        _make_synthetic_parquet(tmp_path, "p001", "ECG", n_windows=4, n_ch=ECG_CH, n_samples=T)
        ds = WindowDataset(tmp_path, ["p001"], "PPG")
        assert len(ds) == 0

    def test_n_channels_filter_excludes_wrong_channel_count(self, tmp_path: Path) -> None:
        """n_channels filter must exclude windows whose channel count doesn't match."""
        import pandas as pd
        # Write a parquet with mixed channel counts: 3 windows with 3ch, 2 with 1ch
        rows = []
        for i in range(3):
            sig = np.random.randn(T, ECG_CH).astype(np.float32)
            rows.append({
                "patient_id": "p001", "segment_name": "s", "window_idx": i,
                "modality": "ECG", "channel_names": json.dumps(["II","V","aVR"]),
                "fs": 62.5, "window_start_sec": 0.0, "window_end_sec": 30.0,
                "n_samples": T, "n_channels": ECG_CH,
                "signal_bytes": sig.tobytes(), "signal_shape": json.dumps([T, ECG_CH]),
                "signal_dtype": "float32", "quality_ok": True,
            })
        for i in range(2):
            sig = np.random.randn(T, 1).astype(np.float32)
            rows.append({
                "patient_id": "p001", "segment_name": "s", "window_idx": i + 3,
                "modality": "ECG", "channel_names": json.dumps(["II"]),
                "fs": 62.5, "window_start_sec": 0.0, "window_end_sec": 30.0,
                "n_samples": T, "n_channels": 1,
                "signal_bytes": sig.tobytes(), "signal_shape": json.dumps([T, 1]),
                "signal_dtype": "float32", "quality_ok": True,
            })
        df = pd.DataFrame(rows)
        df.to_parquet(tmp_path / "p001.parquet", index=False)

        # Without filter: all 5 windows loaded (would crash in DataLoader)
        ds_all = WindowDataset(tmp_path, ["p001"], "ECG")
        assert len(ds_all) == 5

        # With n_channels=3: only the 3-channel windows
        ds_filtered = WindowDataset(tmp_path, ["p001"], "ECG", n_channels=ECG_CH)
        assert len(ds_filtered) == 3

        # All returned tensors must have exactly 3 channels
        for i in range(len(ds_filtered)):
            assert ds_filtered[i].shape[0] == ECG_CH


# ---------------------------------------------------------------------------
# TestCheckpointIO
# ---------------------------------------------------------------------------


class TestCheckpointIO:
    def test_save_creates_files(self, tmp_path: Path) -> None:
        enc = SignalEncoder(in_channels=1, d_model=D)
        save_encoder(enc, tmp_path)
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "config.json").exists()

    def test_load_encoder_round_trip(self, tmp_path: Path) -> None:
        """Loaded encoder must produce the same output as the original."""
        enc = SignalEncoder(in_channels=ECG_CH, d_model=D, n_heads=N_HEADS, n_layers=N_LAYERS)
        enc.eval()
        x = torch.randn(1, ECG_CH, T)
        with torch.no_grad():
            expected = enc(x)

        save_encoder(enc, tmp_path)
        loaded = load_encoder(tmp_path)
        loaded.eval()
        with torch.no_grad():
            actual = loaded(x)

        assert torch.allclose(expected, actual, atol=1e-5)

    def test_load_encoder_eval_mode(self, tmp_path: Path) -> None:
        """load_encoder() must return the encoder in eval mode."""
        enc = SignalEncoder(in_channels=1, d_model=D)
        save_encoder(enc, tmp_path)
        loaded = load_encoder(tmp_path)
        assert not loaded.training


# ---------------------------------------------------------------------------
# TestSplitLogic
# ---------------------------------------------------------------------------


class TestSplitLogic:
    def _make_ids_strata(self, n: int) -> tuple[list[str], list[int]]:
        """Generate n patient IDs with 4 balanced strata (0-3)."""
        pids = [f"p{i:04d}" for i in range(n)]
        strata = [i % 4 for i in range(n)]
        return pids, strata

    def test_train_val_test_sizes(self) -> None:
        pids, strata = self._make_ids_strata(200)
        train, val, test = _stratified_split(pids, strata, n_val=30, n_test=30)
        assert len(val) <= 30
        assert len(test) <= 30
        assert len(train) + len(val) + len(test) == 200

    def test_no_overlap_val_test(self) -> None:
        pids, strata = self._make_ids_strata(200)
        train, val, test = _stratified_split(pids, strata, n_val=30, n_test=30)
        assert set(val).isdisjoint(set(test))

    def test_no_overlap_train_val(self) -> None:
        pids, strata = self._make_ids_strata(200)
        train, val, test = _stratified_split(pids, strata, n_val=30, n_test=30)
        assert set(train).isdisjoint(set(val))

    def test_no_overlap_train_test(self) -> None:
        pids, strata = self._make_ids_strata(200)
        train, val, test = _stratified_split(pids, strata, n_val=30, n_test=30)
        assert set(train).isdisjoint(set(test))

    def test_train_only_patients_in_train(self) -> None:
        """Stratum 4 (no LOS) patients must appear only in train."""
        pids = [f"p{i:04d}" for i in range(100)]
        # Half with strata 0-3, half with stratum 4
        strata = [i % 4 for i in range(50)] + [4] * 50
        train, val, test = _stratified_split(pids, strata, n_val=10, n_test=10)
        train_only_set = set(f"p{i:04d}" for i in range(50, 100))
        # None of the stratum-4 patients appear in val or test
        assert train_only_set.isdisjoint(set(val))
        assert train_only_set.isdisjoint(set(test))

    def test_all_ids_covered(self) -> None:
        """Every input patient ID must appear in exactly one split."""
        pids, strata = self._make_ids_strata(120)
        train, val, test = _stratified_split(pids, strata, n_val=20, n_test=20)
        assert sorted(train + val + test) == sorted(pids)
