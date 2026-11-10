"""
Concept Bottleneck Model (CBM) for CLBench Phase 2.2.

Architecture
------------
Independent CBM: one sigmoid head per (modality, concept) pair, trained
on rule-derived binary labels.  Each head is a single Linear + Sigmoid,
keeping the bottleneck maximally interpretable.

The bottleneck enforces that all information flowing to the KG must pass
through SNOMED concept activations — no latent bypass.

Design choices
--------------
- **Independent heads**: each concept trained separately (no cross-concept
  gradient coupling).  Simpler to debug, easier to replace one concept's
  labeller without retraining the others.
- **Frozen encoder**: encoder weights are NEVER updated during CBM training.
  Representations come from MAE pre-training; CBM only learns a linear
  alignment from that representation to the concept space.
- **Cached features**: training operates on pre-cached (N_windows, d_model)
  tensors, not raw waveforms, so CBM training is O(minutes) not O(hours).

Concept coverage (Tier-1 rule-defined)
---------------------------------------
  ECG  : tachycardia, bradycardia
  PPG  : tachycardia_ppg
  RESP : tachypnea, bradypnea
  ABP  : hypotension, hypertension

Checkpoint format
-----------------
  checkpoints/cbm/
    config.json          -- CBM constructor kwargs + concept list
    {modality}_{concept}.pt  -- state_dict for each concept head
    training_report.txt
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept registry
# ---------------------------------------------------------------------------

# (modality, concept_key, snomed_code)
CONCEPT_REGISTRY: list[dict[str, str]] = [
    {"modality": "ECG", "concept": "tachycardia", "snomed": "3424008", "rule": "HR > 100 bpm"},
    {"modality": "ECG", "concept": "bradycardia", "snomed": "48867003", "rule": "HR < 60 bpm"},
    {
        "modality": "PPG",
        "concept": "tachycardia_ppg",
        "snomed": "3424008",
        "rule": "HR > 100 bpm (PPG)",
    },
    {"modality": "RESP", "concept": "tachypnea", "snomed": "271823003", "rule": "RR > 20 brpm"},
    {"modality": "RESP", "concept": "bradypnea", "snomed": "426813001", "rule": "RR < 12 brpm"},
    {
        "modality": "ABP",
        "concept": "hypotension",
        "snomed": "45007003",
        "rule": "MAP < 65 mmHg (proxy for SBP < 90)",
    },
    {
        "modality": "ABP",
        "concept": "hypertension",
        "snomed": "38341003",
        "rule": "MAP > 105 mmHg (proxy for SBP >= 140)",
    },
]


@dataclass
class CBMConfig:
    """Training hyperparameters for the CBM heads."""

    d_model: int = 128
    epochs: int = 30
    lr: float = 5e-4
    batch_size: int = 512
    weight_decay: float = 1e-4
    val_frac: float = 0.15
    seed: int = 42
    min_positive_frac: float = 0.02
    pos_weight_cap: float = 20.0


# ---------------------------------------------------------------------------
# Single concept head
# ---------------------------------------------------------------------------


class ConceptHead(nn.Module):  # type: ignore[misc]
    """Linear concept head: d_model -> 1 logit (sigmoid applied externally)."""

    def __init__(self, d_model: int, concept: str, snomed_code: str) -> None:
        super().__init__()
        self.concept = concept
        self.snomed_code = snomed_code
        self.linear = nn.Linear(d_model, 1)
        # Feature standardization, fitted from the training set and stored as
        # buffers so the SAME transform is applied at inference (and survives
        # save/load). Without it, large-magnitude encoder features (e.g. the
        # ECG weighted-MAE embeddings) make the optimizer diverge to NaN.
        self.register_buffer("feat_mean", torch.zeros(d_model))
        self.register_buffer("feat_std", torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, d_model) -> (B,) logit"""
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = (x - self.feat_mean) / self.feat_std
        result: Tensor = self.linear(x).squeeze(-1)
        return result

    def predict_proba(self, x: Tensor) -> Tensor:
        """x: (B, d_model) -> (B,) sigmoid activation in [0, 1]"""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def __repr__(self) -> str:
        return f"ConceptHead(concept={self.concept}, snomed={self.snomed_code})"


# ---------------------------------------------------------------------------
# ConceptCBM
# ---------------------------------------------------------------------------


class ConceptCBM:
    """
    Full concept bottleneck model.

    heads[modality][concept] = ConceptHead

    Each modality's heads are sized to that modality's encoder ``d_model``.
    This matters because encoders are allowed to have heterogeneous widths
    (e.g. after Phase 2.1's fix-up the ECG encoder is d_model=256 while a
    legacy PPG checkpoint may still be 128).  Callers pass
    ``modality_d_models`` keyed by modality string; the legacy ``d_model``
    argument is honoured as a uniform fallback for backward compatibility
    with older checkpoints / tests.
    """

    def __init__(
        self,
        d_model: int = 128,
        concept_registry: list[dict[str, str]] | None = None,
        modality_d_models: dict[str, int] | None = None,
    ) -> None:
        registry = concept_registry or CONCEPT_REGISTRY
        # Resolve per-modality d_model.  If ``modality_d_models`` is provided
        # it wins; otherwise every modality falls back to the uniform
        # ``d_model`` argument (backwards-compatible behaviour).
        resolved: dict[str, int] = {}
        modalities = {entry["modality"] for entry in registry}
        for mod in modalities:
            if modality_d_models is not None and mod in modality_d_models:
                resolved[mod] = int(modality_d_models[mod])
            else:
                resolved[mod] = int(d_model)
        self.modality_d_models = resolved
        # ``self.d_model`` is kept for legacy callers; it reports the most
        # common width across modalities and is purely informational.
        self.d_model = int(d_model)

        self.heads: dict[str, dict[str, ConceptHead]] = {}
        for entry in registry:
            mod = entry["modality"]
            con = entry["concept"]
            if mod not in self.heads:
                self.heads[mod] = {}
            self.heads[mod][con] = ConceptHead(resolved[mod], con, entry.get("snomed", ""))

        n_heads = sum(len(v) for v in self.heads.values())
        log.info(
            "ConceptCBM: %d concept heads across %d modalities (d_models=%s)",
            n_heads,
            len(self.heads),
            resolved,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        modality: str,
        embeddings: Tensor,
        device: torch.device | None = None,
    ) -> dict[str, Tensor]:
        """
        Returns dict[concept -> (N,) sigmoid activation].
        """
        if modality not in self.heads:
            return {}
        out: dict[str, Tensor] = {}
        for concept, head in self.heads[modality].items():
            if device is not None:
                head = head.to(device)
            head.eval()
            out[concept] = head.predict_proba(embeddings.to(next(head.parameters()).device))
        return out

    def concepts_for_modality(self, modality: str) -> list[str]:
        return list(self.heads.get(modality, {}).keys())

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, checkpoint_dir: Path, extra: dict[str, Any] | None = None) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        registry_out = []
        for mod, concepts in self.heads.items():
            for con, head in concepts.items():
                pt_name = f"{mod}_{con}.pt"
                torch.save(head.state_dict(), checkpoint_dir / pt_name)
                registry_out.append(
                    {
                        "modality": mod,
                        "concept": con,
                        "snomed": head.snomed_code,
                        "pt_file": pt_name,
                    }
                )
        cfg: dict[str, Any] = {
            "d_model": self.d_model,
            "modality_d_models": self.modality_d_models,
            "concepts": registry_out,
        }
        if extra:
            cfg.update(extra)
        (checkpoint_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        n_heads = len(registry_out)
        log.info("ConceptCBM saved -> %s  (%d heads)", checkpoint_dir, n_heads)

    @classmethod
    def load(cls, checkpoint_dir: Path) -> ConceptCBM:
        cfg_path = checkpoint_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No config.json in {checkpoint_dir}")
        cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
        registry = [
            {"modality": e["modality"], "concept": e["concept"], "snomed": e["snomed"], "rule": ""}
            for e in cfg["concepts"]
        ]
        cbm = cls(
            d_model=cfg["d_model"],
            concept_registry=registry,
            modality_d_models=cfg.get("modality_d_models"),  # absent on legacy
        )
        for entry in cfg["concepts"]:
            mod, con = entry["modality"], entry["concept"]
            pt_path = checkpoint_dir / entry["pt_file"]
            cbm.heads[mod][con].load_state_dict(
                torch.load(pt_path, map_location="cpu", weights_only=True)
            )
            cbm.heads[mod][con].eval()
        log.info("ConceptCBM loaded from %s  (%d heads)", checkpoint_dir, len(cfg["concepts"]))
        return cbm


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@dataclass
class ConceptTrainResult:
    modality: str
    concept: str
    n_train: int
    n_val: int
    n_positive_train: int
    val_auroc: float
    val_loss: float
    skipped: bool = False
    skip_reason: str = ""

    def __str__(self) -> str:
        if self.skipped:
            return f"[SKIP] {self.modality}/{self.concept}: {self.skip_reason}"
        status = "PASS" if self.val_auroc >= 0.60 else "LOW "
        pos_pct = self.n_positive_train / max(self.n_train, 1)
        return (
            f"[{status}] {self.modality}/{self.concept}  "
            f"AUROC={self.val_auroc:.3f}  loss={self.val_loss:.4f}  "
            f"n={self.n_train}  pos%={pos_pct:.1%}"
        )


def _auroc_numpy(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC. O(n log n) time, O(n) memory."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Ascending sort: rank 1 = lowest score, rank N = highest.
    # Good classifier → positives at top → high ranks → large u_stat → AUROC near 1.
    order = np.argsort(scores, kind="stable")
    ranks = np.arange(1, len(labels) + 1, dtype=np.float64)
    u_stat = float(ranks[labels[order] == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    return u_stat / (float(n_pos) * float(n_neg))


def train_concept_head(
    head: ConceptHead,
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg: CBMConfig,
    device: torch.device,
    modality: str = "",
) -> ConceptTrainResult:
    """
    Train one ConceptHead on pre-extracted embeddings.

    Parameters
    ----------
    head : ConceptHead
    embeddings : (N, d_model) float32 ndarray
    labels : (N,) int ndarray {0, 1}
    cfg : CBMConfig
    device : torch.device
    modality : str  (used for logging only)
    """
    n_total = len(labels)
    if n_total < 10:
        return ConceptTrainResult(
            modality=modality,
            concept=head.concept,
            n_train=0,
            n_val=0,
            n_positive_train=0,
            val_auroc=float("nan"),
            val_loss=float("nan"),
            skipped=True,
            skip_reason="fewer than 10 windows",
        )

    pos_frac = float(labels.mean())
    if pos_frac < cfg.min_positive_frac or (1 - pos_frac) < cfg.min_positive_frac:
        return ConceptTrainResult(
            modality=modality,
            concept=head.concept,
            n_train=0,
            n_val=0,
            n_positive_train=0,
            val_auroc=float("nan"),
            val_loss=float("nan"),
            skipped=True,
            skip_reason=f"pos_frac={pos_frac:.2%} outside balance window",
        )

    rng = np.random.default_rng(cfg.seed)
    val_n = max(1, int(n_total * cfg.val_frac))
    idx = rng.permutation(n_total)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    x_train = torch.from_numpy(embeddings[train_idx]).float()
    y_train = torch.from_numpy(labels[train_idx]).float()
    x_val = torch.from_numpy(embeddings[val_idx]).float()
    y_val = labels[val_idx]

    # Fit standardization on the train split and store on the head so forward()
    # (train, val, and inference) applies the identical transform. Sanitize the
    # stats so a stray NaN feature can't poison the whole head.
    x_train = torch.nan_to_num(x_train, nan=0.0, posinf=1e4, neginf=-1e4)
    feat_mean = x_train.mean(0)
    feat_std = x_train.std(0).clamp_min(1e-6)
    head.feat_mean.copy_(feat_mean)
    head.feat_std.copy_(feat_std)

    n_pos = int(y_train.sum().item())
    n_neg = len(y_train) - n_pos
    pos_weight = min(cfg.pos_weight_cap, n_neg / max(n_pos, 1))
    pw = torch.tensor([pos_weight], device=device)

    ds = TensorDataset(x_train, y_train)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    head.train()
    for _epoch in range(cfg.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            F.binary_cross_entropy_with_logits(head(xb), yb, pos_weight=pw).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

    head.eval()
    with torch.no_grad():
        val_logits = head(x_val.to(device)).cpu().numpy()

    val_scores = 1.0 / (1.0 + np.exp(-val_logits))
    val_loss_tensor = F.binary_cross_entropy_with_logits(
        torch.from_numpy(val_logits),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    val_loss = float(val_loss_tensor.item())
    val_auroc = _auroc_numpy(val_scores, y_val)

    return ConceptTrainResult(
        modality=modality,
        concept=head.concept,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_positive_train=n_pos,
        val_auroc=val_auroc,
        val_loss=val_loss,
    )
