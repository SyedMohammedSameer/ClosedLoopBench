"""
Per-modality signal encoder: 1D Conv stem + small Transformer.

Architecture
------------
Input shape:  (B, C, T)
  B  -- batch size
  C  -- input channels (leads/sensors; 3 for ECG, 1 for PPG/ABP/RESP)
  T  -- time samples  (1874 for a 30 s window at 62.5 Hz)

Stage 1 -- 1D Convolutional Stem (4 blocks)
  Each block: Conv1d(stride=2) -> BatchNorm1d -> GELU
  Channels : C -> 64 -> 128 -> 256 -> d_model
  Stride   : x2 per block -> T -> T/2 -> T/4 -> T/8 -> T/16
  For T=1874: token sequence length T' == 117

Stage 2 -- Sinusoidal Positional Encoding
  Added to the (B, T', d_model) token sequence.

Stage 3 -- Transformer Encoder
  n_layers=4, n_heads=8, d_model=256, ff_dim=1024, dropout=0.1
  Implemented with PyTorch's built-in nn.TransformerEncoderLayer (Pre-LN).

Stage 4 -- Mean Pooling
  Mean over T' -> (B, d_model) patient-level embedding.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Hyperparameter defaults (referenced in paper)
# ---------------------------------------------------------------------------

D_MODEL: int = 256
N_HEADS: int = 8
N_LAYERS: int = 4
FF_DIM: int = 1024
DROPOUT: float = 0.1
MAX_SEQ_LEN: int = 512

_STEM_CHANNELS = (64, 128, 256)
_STEM_KERNELS = (7, 5, 5, 3)
_STEM_STRIDES = (2, 2, 2, 2)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class SinusoidalPosEnc(nn.Module):  # type: ignore[misc]
    """Fixed sinusoidal positional encoding (no learnable parameters)."""

    def __init__(self, d_model: int, max_seq_len: int = MAX_SEQ_LEN) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        t = x.size(1)
        pe: Tensor = self.pe
        return x + pe[:, :t, :]


# ---------------------------------------------------------------------------
# Convolutional stem
# ---------------------------------------------------------------------------


def _make_norm(out_ch: int, norm_type: str) -> nn.Module:
    """
    Build a normalisation layer for the conv stem.

    * ``"batch"`` (legacy default) — :class:`nn.BatchNorm1d`.  Strips the
      batch-wise per-channel mean from the feature maps, which makes the
      encoder *invariant to additive offsets in the input*.  Fine for
      modalities whose target concepts are periodicity-based (rate,
      rhythm) but FATAL for modalities whose target concepts depend on
      absolute signal level (e.g. ABP hypotension / hypertension, where
      MAP is encoded in the DC component).
    * ``"group"`` — :class:`nn.GroupNorm` with one group, i.e. layer-norm
      across channels at each spatial position.  Preserves per-sample
      additive offsets, which is what ABP needs after the Phase 2.1
      population-scale normalization put MAP back into the signal.
    """
    if norm_type == "batch":
        return nn.BatchNorm1d(out_ch)
    if norm_type == "group":
        return nn.GroupNorm(num_groups=1, num_channels=out_ch)
    raise ValueError(f"Unknown norm_type='{norm_type}'. Expected 'batch' or 'group'.")


class _ConvBlock(nn.Module):  # type: ignore[misc]
    """Conv1d(stride=2) -> Norm -> GELU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel // 2),
            _make_norm(out_ch, norm_type),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.block(x)
        return result


class ConvStem(nn.Module):  # type: ignore[misc]
    """
    Four-block strided conv stem: (B, C, T) -> (B, d_model, T//16).

    Parameters
    ----------
    in_channels : int
        Number of input signal channels.
    d_model : int
        Output channel dimension.
    norm_type : {'batch', 'group'}
        Normalisation layer inside each conv block.  See :func:`_make_norm`
        for the trade-off; modalities where absolute signal level matters
        (currently only ABP) MUST use 'group'.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = D_MODEL,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        ch = (in_channels,) + _STEM_CHANNELS + (d_model,)
        blocks: list[nn.Module] = []
        for i, (k, s) in enumerate(zip(_STEM_KERNELS, _STEM_STRIDES)):
            blocks.append(
                _ConvBlock(
                    int(ch[i]),
                    int(ch[i + 1]),
                    kernel=k,
                    stride=s,
                    norm_type=norm_type,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.blocks(x)
        return result


# ---------------------------------------------------------------------------
# Signal encoder
# ---------------------------------------------------------------------------


class SignalEncoder(nn.Module):  # type: ignore[misc]
    """
    Per-modality signal encoder for CLBench.

    Takes a batch of pre-processed windows and returns a fixed-dimensional
    embedding per window.  During SSL pre-training, ``extract_tokens``
    exposes the token sequence (before the transformer) for MAE masking.

    Parameters
    ----------
    in_channels : int
        Input channels (3 for ECG, 1 for PPG / ABP / RESP).
    d_model : int
        Transformer model dimension and output embedding size.
    n_heads : int
        Number of attention heads (must divide d_model evenly).
    n_layers : int
        Number of transformer encoder layers.
    ff_dim : int
        Feed-forward inner dimension.
    dropout : float
        Dropout probability.
    max_seq_len : int
        Maximum supported token sequence length.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN,
        use_cls_token: bool = False,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.dropout_p = dropout
        self.max_seq_len = max_seq_len
        self.use_cls_token = use_cls_token
        self.norm_type = norm_type

        self.conv_stem = ConvStem(in_channels, d_model, norm_type=norm_type)
        self.pos_enc = SinusoidalPosEnc(d_model, max_seq_len)

        if use_cls_token:
            # Learnable [CLS] token prepended to the token sequence.  After
            # the transformer, its final hidden state is the pooled embedding.
            # This gives the transformer a dedicated slot to aggregate global
            # information (rate, rhythm regularity) that mean-pooling erases.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def extract_tokens(self, x: Tensor) -> Tensor:
        """
        Conv stem + positional encoding only (pre-transformer).

        Used by MaskedSignalEncoder to obtain the token sequence before
        masking is applied.

        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        (B, T'', d_model) where T'' = T // 16 (+1 if ``use_cls_token``).
        Position 0 is the CLS token when enabled.
        """
        h: Tensor = self.conv_stem(x)  # (B, d_model, T')
        h = h.transpose(1, 2)  # (B, T', d_model)
        if self.use_cls_token:
            cls = self.cls_token.expand(h.size(0), -1, -1)  # (B, 1, d_model)
            h = torch.cat([cls, h], dim=1)  # (B, 1+T', d_model)
        return self.pos_enc(h)

    def forward(self, x: Tensor) -> Tensor:
        """
        End-to-end forward pass: window -> pooled embedding.

        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        (B, d_model).  When ``use_cls_token`` is True, this is the final
        CLS hidden state; otherwise it is the mean over content tokens.
        """
        h = self.extract_tokens(x)  # (B, T'', d_model)
        h = self.transformer(h)  # (B, T'', d_model)
        if self.use_cls_token:
            return h[:, 0]  # CLS hidden state
        return h.mean(dim=1)  # legacy mean pool over all positions

    def encode(self, x: Tensor) -> Tensor:
        """Alias for ``forward``."""
        return self.forward(x)

    def config(self) -> dict[str, Any]:
        """Return constructor kwargs for checkpoint serialisation."""
        return {
            "in_channels": self.in_channels,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_p,
            "max_seq_len": self.max_seq_len,
            "use_cls_token": self.use_cls_token,
            "norm_type": self.norm_type,
        }

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (
            f"SignalEncoder(in_channels={self.in_channels}, "
            f"d_model={self.d_model}, n_layers={self.n_layers}, "
            f"params={n:,})"
        )
