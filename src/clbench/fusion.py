"""
Cross-modal temporal fusion for CLBench Phase 2.2.

Architecture
------------
``CrossModalFusion`` takes per-modality encoder embeddings and fuses them
into a single representation via multi-head cross-attention.  Missing
modalities (patients that don't have all four signals) are handled with
learned mask tokens so the module always produces a full-rank output.

Design choices
--------------
- **Embedding-level fusion** (after encoder mean-pooling): each modality
  contributes one (B, d_model) vector per window.  Avoids the complexity
  of token-level cross-attention while still learning cross-modal structure.
- **Modality type embeddings**: learnable per-modality offset (analogous to
  token-type embeddings in BERT) so the attention can distinguish ECG from
  ABP context even after the shared projection.
- **Mask-token fill-in**: when a modality is absent for a patient, its slot
  is filled with a learned mask token.  The attention key-padding mask
  excludes absent modalities from softmax so they contribute no gradient
  signal.  This is cleaner than zero-padding (which would pull the attention
  weights toward a zero vector that happens to be in the value space).
- **Two-layer self-attention over modalities**: depth-2 is sufficient for
  four modalities; deeper would overfit on the small N=200 cohort.
- **Masked mean-pool readout**: output is the mean of present-modality
  hidden states (not the full 4-modality sequence), so missing modalities
  don't dilute the aggregate.

Forward contract
----------------
    fusion = CrossModalFusion(d_model=256, n_modalities=4)
    # embeddings: dict[str → Tensor (B, d_model)], subset of modality_order
    # mask: BoolTensor (B, n_modalities), True = modality present
    out = fusion(embeddings, mask)   # (B, d_model)

Checkpoint format
-----------------
Saved / loaded via torch.save / torch.load on the state_dict.
Config JSON sits alongside: ``fusion_config.json``.

References
----------
  Vaswani et al. (2017) Attention Is All You Need.
  Liang et al. (2022) Foundations and Trends in Multimodal Machine Learning.
  Kiyasseh et al. (2021) CLOCS — Contrastive Learning of Cardiac Signals.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical modality order (must be consistent across training + inference)
# ---------------------------------------------------------------------------

MODALITY_ORDER: list[str] = ["ECG", "PPG", "ABP", "RESP"]
MODALITY_INDEX: dict[str, int] = {m: i for i, m in enumerate(MODALITY_ORDER)}


# ---------------------------------------------------------------------------
# Helper: scaled dot-product attention with key-padding mask
# ---------------------------------------------------------------------------


class _MultiHeadSelfAttention(nn.Module):  # type: ignore[misc]
    """
    Standard pre-LN multi-head self-attention over a sequence of length S.

    Parameters
    ----------
    d_model : int
    n_heads : int
    dropout : float
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : (B, S, d_model)
        key_padding_mask : (B, S) BoolTensor — True for positions to IGNORE

        Returns
        -------
        (B, S, d_model)
        """
        B, S, D = x.shape  # noqa: N806

        # Pre-LN
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        # (B, n_heads, S, d_head)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, S, S)

        if key_padding_mask is not None:
            # Expand mask to (B, 1, 1, S) then broadcast across heads and queries
            mask_4d = key_padding_mask[:, None, None, :]  # (B, 1, 1, S)
            attn = attn.masked_fill(mask_4d, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        # Rows that are all -inf (fully masked) → nan after softmax → zero them
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_proj(out)
        x = x + self.drop(out)

        # Feed-forward (pre-LN)
        x = x + self.drop(self.ff(self.ff_norm(x)))
        return x


# ---------------------------------------------------------------------------
# CrossModalFusion
# ---------------------------------------------------------------------------


class CrossModalFusion(nn.Module):  # type: ignore[misc]
    """
    Cross-modal fusion over per-modality encoder embeddings.

    Parameters
    ----------
    d_model : int
        Dimension of each modality's encoder output AND the fused output.
        Must match the encoder d_model (default 256).
    n_modalities : int
        Number of modalities (default 4: ECG, PPG, ABP, RESP).
    n_heads : int
        Attention heads in each self-attention layer (default 4).
    n_layers : int
        Number of self-attention layers over the modality sequence (default 2).
    dropout : float
    modality_order : list[str]
        Canonical ordering of modality names; maps names to token positions.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_modalities: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        modality_order: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.modality_order = modality_order or MODALITY_ORDER[:n_modalities]
        self.modality_index = {m: i for i, m in enumerate(self.modality_order)}

        # Learnable modality type embeddings: one per slot.
        # Added BEFORE the attention layers so the model can distinguish
        # "ECG representation" from "ABP representation" even if they happen
        # to be similar vectors.
        self.modality_embeddings = nn.Embedding(n_modalities, d_model)
        nn.init.normal_(self.modality_embeddings.weight, std=0.02)

        # Mask token: fills absent modality slots.
        # Excluded from attention via key_padding_mask so it doesn't
        # corrupt the gradient of present modalities.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Attention stack over the modality sequence (length = n_modalities)
        self.layers = nn.ModuleList(
            [_MultiHeadSelfAttention(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(
            "CrossModalFusion: d_model=%d, n_modalities=%d, " "n_heads=%d, n_layers=%d, params=%s",
            d_model,
            n_modalities,
            n_heads,
            n_layers,
            f"{n_params:,}",
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: dict[str, Tensor],
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Fuse per-modality embeddings into a single representation.

        Parameters
        ----------
        embeddings : dict[modality_name → Tensor (B, d_model)]
            Per-modality encoder outputs.  Only present modalities need
            to be included; absent modalities are filled with mask tokens.
        mask : BoolTensor (B, n_modalities) or None
            True = modality PRESENT (note: opposite of key_padding_mask
            convention).  If None, inferred from the keys of ``embeddings``.

        Returns
        -------
        (B, d_model)
        """
        # Infer batch size from any present embedding
        B = next(iter(embeddings.values())).shape[0]  # noqa: N806
        device = next(iter(embeddings.values())).device

        # Build the (B, n_modalities, d_model) input sequence
        seq = self.mask_token.expand(B, self.n_modalities, self.d_model).clone()

        present_flags = torch.zeros(B, self.n_modalities, dtype=torch.bool, device=device)

        for mod, emb in embeddings.items():
            if mod not in self.modality_index:
                log.warning("CrossModalFusion: unknown modality '%s' — skipped", mod)
                continue
            idx = self.modality_index[mod]
            seq[:, idx, :] = emb.to(device)
            present_flags[:, idx] = True

        # Override with explicit mask if provided
        if mask is not None:
            present_flags = mask.to(device)

        # Add modality type embeddings
        mod_idx = torch.arange(self.n_modalities, device=device)
        seq = seq + self.modality_embeddings(mod_idx).unsqueeze(0)  # (1, S, D) broadcast

        # key_padding_mask for attention: True = IGNORE (absent slots)
        key_pad = ~present_flags  # (B, n_modalities)

        # Attention stack
        h = seq
        for layer in self.layers:
            h = layer(h, key_padding_mask=key_pad)
        h = self.norm(h)  # (B, n_modalities, d_model)

        # Masked mean-pool over PRESENT modalities only
        # present_flags: (B, S) → expand to (B, S, d_model) for weighted sum
        present_float = present_flags.float().unsqueeze(-1)  # (B, S, 1)
        n_present = present_float.sum(dim=1).clamp(min=1.0)  # (B, 1)
        fused = (h * present_float).sum(dim=1) / n_present  # (B, d_model)

        return fused

    # ------------------------------------------------------------------
    # Convenience: build mask from modality key set
    # ------------------------------------------------------------------

    def make_mask(
        self, present_modalities: list[str], batch_size: int, device: torch.device
    ) -> Tensor:
        """
        Build a (B, n_modalities) BoolTensor for a uniform batch
        (all samples have the same modalities present).
        """
        flags = torch.zeros(batch_size, self.n_modalities, dtype=torch.bool, device=device)
        for mod in present_modalities:
            if mod in self.modality_index:
                flags[:, self.modality_index[mod]] = True
        return flags

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_modalities": self.n_modalities,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "modality_order": self.modality_order,
        }

    def save(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_dir / "fusion.pt")
        (checkpoint_dir / "fusion_config.json").write_text(
            json.dumps(self.config(), indent=2), encoding="utf-8"
        )
        log.info("CrossModalFusion saved -> %s", checkpoint_dir)

    @classmethod
    def load(cls, checkpoint_dir: Path) -> CrossModalFusion:
        cfg_path = checkpoint_dir / "fusion_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No fusion_config.json in {checkpoint_dir}")
        cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
        fusion = cls(**cfg)
        state = torch.load(checkpoint_dir / "fusion.pt", map_location="cpu", weights_only=True)
        fusion.load_state_dict(state)
        fusion.eval()
        log.info("CrossModalFusion loaded from %s", checkpoint_dir)
        return fusion


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    B, D = 8, 256
    fusion = CrossModalFusion(d_model=D, n_modalities=4, n_heads=4, n_layers=2)
    fusion.eval()

    # All 4 modalities present
    embs = {m: torch.randn(B, D) for m in MODALITY_ORDER}
    out = fusion(embs)
    assert out.shape == (B, D), f"Expected ({B},{D}), got {out.shape}"
    print(f"All-modalities: {out.shape}  ✓")

    # Only ECG + ABP present (PPG + RESP absent)
    embs_partial = {"ECG": torch.randn(B, D), "ABP": torch.randn(B, D)}
    out_partial = fusion(embs_partial)
    assert out_partial.shape == (B, D)
    print(f"Partial (ECG+ABP only): {out_partial.shape}  ✓")

    # Gradient flows through fusion
    fusion.train()
    embs_grad = {m: torch.randn(B, D, requires_grad=True) for m in MODALITY_ORDER}
    loss = fusion(embs_grad).sum()
    loss.backward()
    assert all(v.grad is not None for v in embs_grad.values())
    print("Gradient flow through fusion:  ✓")

    # Checkpoint round-trip
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        fusion.save(p)
        fusion2 = CrossModalFusion.load(p)
        fusion2.eval()
        with torch.no_grad():
            out2 = fusion2({m: embs[m] for m in MODALITY_ORDER})
        assert torch.allclose(out, out2, atol=1e-5), "Checkpoint mismatch"
    print("Checkpoint round-trip:  ✓")

    print("\nAll smoke-tests passed.")
