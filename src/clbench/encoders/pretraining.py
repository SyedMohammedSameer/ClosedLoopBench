"""
Masked autoencoder (MAE) self-supervised pre-training for CLBench encoders.

Objective
---------
Following MAE for time series (He et al. 2022; Dong et al. 2023 for biosignals):

1. Pass input (B, C, T) through the conv stem + positional encoding
   -> token sequence (B, T', d_model).
2. Randomly mask ``mask_ratio`` fraction of tokens (replace with learnable
   [MASK] embedding).
3. Run all tokens (including masked) through the transformer.
4. A lightweight prediction head projects each position to d_model.
5. MSE loss on masked positions only, between predictions and the
   *original unmasked* token features (stop-gradient target).

Only the :class:`SignalEncoder` weights are used at inference time.
The prediction head and mask token are discarded after pre-training.

Dataset
-------
:class:`WindowDataset` loads per-patient parquet files from
``data/processed/`` and yields (C, T) float32 tensors.

Checkpoint format
-----------------
``checkpoints/encoders/{modality}/``
  best.pt      — SignalEncoder state dict
  config.json  — SignalEncoder constructor kwargs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from clbench.encoders.base import SignalEncoder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MAE pre-training module
# ---------------------------------------------------------------------------

MASK_RATIO: float = 0.40  # fraction of content tokens masked during pre-training
# Patch size at the encoder input = 2 ** 4 = 16 raw samples per output token
# (four stride-2 conv blocks in :class:`ConvStem`).  Used by the
# ``target='signal'`` MAE objective to slice the input into target patches.
PATCH_SAMPLES: int = 16


class MaskedSignalEncoder(nn.Module):  # type: ignore[misc]
    """
    MAE-style self-supervised wrapper around :class:`SignalEncoder`.

    Masking is applied at the token level (after the conv stem and positional
    encoding), so the transformer sees the full sequence context.  This is
    simpler than the asymmetric ViT-MAE design and works well for the small
    sequence lengths we have (~117 content tokens per window, plus 1 CLS).

    Reconstruction targets
    ----------------------
    ``target='feature'`` (default, legacy) — reconstruct each masked token's
    own pre-transformer feature.  Cheap for the encoder to satisfy and tends
    to encourage "predict the baseline" solutions when the input has sparse
    high-amplitude events (e.g. QRS spikes in ECG).  Phase 2.1 diagnostics
    showed mean-pooled ECG embeddings under this objective carry no rate
    information at any first/second moment.

    ``target='signal'`` — reconstruct the raw signal patch under each masked
    token (16 samples × in_channels per patch).  The encoder cannot minimise
    this loss without encoding *where* QRS spikes are, so rate is preserved
    in the learned representation.  Pair with ``use_cls_token=True`` on the
    encoder for an explicit aggregation slot.

    CLS-aware masking
    -----------------
    When the wrapped encoder has ``use_cls_token=True``, position 0 of the
    token sequence is the CLS token and is **never masked** — masking it
    would deprive the model of its global aggregation slot during training.
    Masking and loss are restricted to the content tokens (positions ≥ 1).

    Parameters
    ----------
    encoder : SignalEncoder
        The encoder to pre-train.
    mask_ratio : float
        Fraction of *content* tokens to mask per sample.
    target : {'feature', 'signal'}
        Reconstruction target.  See above.
    """

    def __init__(
        self,
        encoder: SignalEncoder,
        mask_ratio: float = MASK_RATIO,
        target: str = "feature",
    ) -> None:
        super().__init__()
        if target not in {"feature", "signal"}:
            raise ValueError(f"Unknown target='{target}'.  Expected 'feature' or 'signal'.")
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.target = target
        d = encoder.d_model
        # Learnable mask token (one vector shared across all masked positions)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, std=0.02)
        if target == "feature":
            self.pred_head = nn.Linear(d, d)
        else:  # signal
            # Predict a (PATCH_SAMPLES × in_channels)-dim raw-signal patch.
            self.pred_head = nn.Linear(d, PATCH_SAMPLES * encoder.in_channels)
        self.patch_samples = PATCH_SAMPLES

    @staticmethod
    def _signal_to_patches(x: Tensor, patch_samples: int) -> Tensor:
        """
        Slice (B, C, T) input into non-overlapping (B, n_patches, C*P) patches.

        Patches that exceed the largest multiple of ``patch_samples`` are
        truncated — same convention as the conv stem's strided downsampling.
        """
        b, c, t = x.shape
        n_patches = t // patch_samples
        t_clip = n_patches * patch_samples
        patches = x[:, :, :t_clip].reshape(b, c, n_patches, patch_samples)
        # (B, n_patches, C, P) -> (B, n_patches, C*P)
        return patches.permute(0, 2, 1, 3).reshape(b, n_patches, c * patch_samples)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute MAE pre-training loss.

        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        loss : scalar Tensor — MSE on masked content positions only.
        pred : (B, T'', d_out) — full prediction sequence (for logging).
               ``T''`` includes the CLS slot when enabled.
        """
        # Guard against NaN/Inf in raw signal data (rare but possible on edge windows)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        # Pre-compute signal targets BEFORE the encoder when needed.
        signal_targets: Tensor | None = None
        if self.target == "signal":
            signal_targets = self._signal_to_patches(x, self.patch_samples).detach()

        # 1. Obtain unmasked token sequence from conv stem (+ CLS if enabled)
        tokens = self.encoder.extract_tokens(x)  # (B, T'', d_model)
        b, t_total, d = tokens.shape

        # 2. Build random binary mask over CONTENT tokens only.
        content_start = 1 if self.encoder.use_cls_token else 0
        n_content = t_total - content_start
        n_mask = max(1, int(n_content * self.mask_ratio))
        # Vectorised per-sample masking: draw random scores for the content
        # tokens and take the top-``n_mask`` per row.  Replaces a Python
        # ``for i in range(b)`` loop of ``torch.randperm`` calls, which issued
        # B sequential GPU kernel launches per step and left the device idle.
        scores = torch.rand(b, n_content, device=x.device)
        sel = scores.topk(n_mask, dim=1).indices + content_start  # (B, n_mask)
        mask = torch.zeros(b, t_total, dtype=torch.bool, device=x.device)
        mask.scatter_(1, sel, True)

        # 3. Replace masked positions with the learnable mask token.  Use
        #    torch.where (not in-place) for MPS compatibility.
        mask_3d = mask.unsqueeze(-1).expand_as(tokens)
        mask_fill = self.mask_token.view(1, 1, d).expand_as(tokens)
        masked_tokens = torch.where(mask_3d, mask_fill, tokens)

        # 4. Run through transformer
        z = self.encoder.transformer(masked_tokens)  # (B, T'', d_model)

        # 5. Predict + loss on masked positions only
        pred = self.pred_head(z)  # (B, T'', d_out)
        if self.target == "feature":
            target = tokens.detach()
            loss: Tensor = F.mse_loss(pred[mask], target[mask])
        else:  # signal
            assert signal_targets is not None
            # Align the prediction with content positions only — the CLS slot
            # of ``pred`` doesn't correspond to any input patch.
            if self.encoder.use_cls_token:
                pred_content = pred[:, 1:]  # (B, T', C*P)
                mask_content = mask[:, 1:]
            else:
                pred_content = pred  # (B, T', C*P)
                mask_content = mask
            # n_patches from _signal_to_patches should equal T'; if T % 16 != 0
            # the trailing input samples were truncated, which is matched by
            # the conv-stem padding behaviour and won't desynchronise tokens
            # for the typical T=1874 (T'=117, n_patches=117).
            n_patches = signal_targets.size(1)
            t_content = pred_content.size(1)
            if n_patches != t_content:
                # Use the overlapping prefix to stay robust to off-by-one cases.
                k = min(n_patches, t_content)
                pred_content = pred_content[:, :k]
                signal_targets = signal_targets[:, :k]
                mask_content = mask_content[:, :k]
            loss = F.mse_loss(pred_content[mask_content], signal_targets[mask_content])

        return loss, pred


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WindowDataset(Dataset):  # type: ignore[misc]
    """
    Loads windowed, preprocessed signals from ``data/processed/`` parquets.

    Each item is a float32 tensor of shape ``(n_channels, n_samples)``,
    i.e. the standard PyTorch convention (channels first).

    Parameters
    ----------
    processed_dir : Path
        Directory containing ``{patient_id}.parquet`` files.
    patient_ids : list[str]
        Which patients to include (train, val, or test subset).
    modality : str
        One of ``ECG | PPG | ABP | RESP``.
    quality_only : bool
        If True (default), only include windows flagged ``quality_ok=True``.
    n_channels : int | None
        If provided, only include windows whose channel count equals this value.
        **Required for ECG** (some patients have 1-lead recordings; the 3-channel
        ECGEncoder cannot collate mixed shapes).  Pass ``encoder.in_channels``.
        For single-channel modalities (PPG/ABP/RESP) this is always 1 and the
        filter is a no-op.
    """

    def __init__(
        self,
        processed_dir: Path,
        patient_ids: list[str],
        modality: str,
        quality_only: bool = True,
        n_channels: int | None = None,
    ) -> None:
        self.modality = modality
        self._arrays: list[np.ndarray] = []  # each: (n_channels, n_samples) float32

        n_patients_loaded = 0
        n_filtered_ch = 0
        for pid in patient_ids:
            fpath = processed_dir / f"{pid}.parquet"
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath)
            df = df[df["modality"] == modality]
            if quality_only:
                df = df[df["quality_ok"].astype(bool)]
            if n_channels is not None:
                before = len(df)
                df = df[df["n_channels"] == n_channels]
                n_filtered_ch += before - len(df)
            if df.empty:
                continue
            n_patients_loaded += 1
            for _, row in df.iterrows():
                shape: list[int] = json.loads(row["signal_shape"])
                arr = np.frombuffer(row["signal_bytes"], dtype="float32").reshape(shape)
                # arr shape: (n_samples, n_channels) -> transpose to (n_channels, n_samples)
                self._arrays.append(arr.T.copy())

        if n_filtered_ch > 0:
            log.info(
                "WindowDataset[%s]: dropped %d windows with n_channels != %d",
                modality,
                n_filtered_ch,
                n_channels,
            )
        log.info(
            "WindowDataset[%s]: %d windows from %d patients",
            modality,
            len(self._arrays),
            n_patients_loaded,
        )

    def __len__(self) -> int:
        return len(self._arrays)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.from_numpy(self._arrays[idx])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: MaskedSignalEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        x: Tensor = batch.to(device)
        optimizer.zero_grad()
        loss, _ = model(x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()  # type: ignore[misc]
def evaluate(
    model: MaskedSignalEncoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return mean validation loss (no gradient)."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        x: Tensor = batch.to(device)
        loss, _ = model(x)
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_encoder(
    encoder: SignalEncoder,
    checkpoint_dir: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save encoder weights and config to *checkpoint_dir*.

    Writes two files:
      ``best.pt``     — ``encoder.state_dict()``
      ``config.json`` — constructor kwargs from ``encoder.config()``

    Parameters
    ----------
    encoder : SignalEncoder
    checkpoint_dir : Path
        Directory will be created if it does not exist.
    extra : dict | None
        Additional metadata to merge into ``config.json`` (e.g. val_loss).

    Returns
    -------
    Path
        Path to the written ``best.pt``.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pt_path = checkpoint_dir / "best.pt"
    cfg_path = checkpoint_dir / "config.json"

    torch.save(encoder.state_dict(), pt_path)

    cfg = encoder.config()
    if extra:
        cfg.update(extra)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    log.info("Encoder checkpoint → %s", checkpoint_dir)
    return pt_path


def load_encoder(checkpoint_dir: Path) -> SignalEncoder:
    """
    Load a :class:`SignalEncoder` from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : Path
        Must contain ``best.pt`` and ``config.json``.

    Returns
    -------
    SignalEncoder  (in eval mode, weights loaded)
    """
    cfg_path = checkpoint_dir / "config.json"
    pt_path = checkpoint_dir / "best.pt"

    if not cfg_path.exists() or not pt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint incomplete in {checkpoint_dir} " f"(need best.pt + config.json)"
        )

    cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Strip non-constructor keys that may have been added via `extra`
    _encoder_keys = {
        "in_channels",
        "d_model",
        "n_heads",
        "n_layers",
        "ff_dim",
        "dropout",
        "max_seq_len",
        "use_cls_token",  # added Phase 2.1 fix-up; absent on legacy ckpts
        "norm_type",  # added Phase 2.1 BatchNorm-DC-erasure fix
    }
    encoder_kwargs = {k: v for k, v in cfg.items() if k in _encoder_keys}
    encoder = SignalEncoder(**encoder_kwargs)
    encoder.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    encoder.eval()
    log.info("Loaded encoder from %s  (%s)", checkpoint_dir, encoder)
    return encoder
