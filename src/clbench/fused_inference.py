"""
Shared fused-inference stack: encoder → CrossModalFusion → CBM.

Both ``scripts/04b_calibrate_persistence.py`` (persistence-tier calibration)
and ``scripts/05_build_kgs.py`` (KG construction) must compute concept
probabilities the SAME way, or the calibration is fitted on a different
distribution than the graphs are built from.  Centralising the path here
removes the cached-feature dependency (which can go stale when an encoder is
re-trained at the same architecture) and guarantees the two stages agree.

Window alignment mirrors ``scripts/colab_export.py::_collect_joint_patient``:
two windows fuse iff they share ``(segment_name, round(window_start_sec, 2))``.
Each modality's fused representation is sliced back onto that modality's own
window order, so a concept's probability sequence stays on its home-modality
timeline (e.g. tachycardia → ECG windows) and ``len(probs)`` equals the number
of quality-OK windows for that modality — exactly what the rule labeller and
the KG builder expect.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clbench.cbm import ConceptCBM
from clbench.encoders import load_encoder
from clbench.encoders.base import SignalEncoder
from clbench.fusion import MODALITY_ORDER, CrossModalFusion

log = logging.getLogger(__name__)

# Rounding used to match windows across modalities (must match colab_export.py).
ALIGN_DECIMALS = 2


# ---------------------------------------------------------------------------
# Checkpoint discovery / loading
# ---------------------------------------------------------------------------


def discover_encoder_in_channels(encoder_ckpt_dir: Path) -> dict[str, int]:
    """Channel count each encoder was trained on (e.g. 3-lead ECG)."""
    out: dict[str, int] = {}
    for mod in MODALITY_ORDER:
        cfg_path = encoder_ckpt_dir / mod / "config.json"
        if cfg_path.exists():
            out[mod] = int(json.loads(cfg_path.read_text(encoding="utf-8"))["in_channels"])
    return out


@dataclass
class FusedStack:
    encoders: dict[str, SignalEncoder]
    fusion: CrossModalFusion
    cbm: ConceptCBM
    d_model: int
    in_channels: dict[str, int]


def load_fused_stack(
    encoder_ckpt_dir: Path,
    fusion_ckpt_dir: Path,
    cbm_ckpt_dir: Path,
    device: torch.device,
) -> FusedStack:
    """Load encoders, fusion, and CBM, all moved to ``device`` and eval()'d."""
    if not (cbm_ckpt_dir / "config.json").exists():
        raise FileNotFoundError(f"No CBM checkpoint at {cbm_ckpt_dir}")
    cbm = ConceptCBM.load(cbm_ckpt_dir)
    d_model = int(json.loads((cbm_ckpt_dir / "config.json").read_text())["d_model"])

    if not (fusion_ckpt_dir / "fusion_config.json").exists():
        raise FileNotFoundError(f"No fusion checkpoint at {fusion_ckpt_dir}")
    fusion = CrossModalFusion.load(fusion_ckpt_dir).to(device)
    fusion.eval()

    encoders: dict[str, SignalEncoder] = {}
    for mod in MODALITY_ORDER:
        mod_dir = encoder_ckpt_dir / mod
        if (mod_dir / "best.pt").exists():
            enc = load_encoder(mod_dir).to(device)
            enc.eval()
            encoders[mod] = enc
        else:
            log.warning("No encoder checkpoint for %s — modality skipped", mod)
    if not encoders:
        raise FileNotFoundError(f"No encoder checkpoints under {encoder_ckpt_dir}")

    in_channels = discover_encoder_in_channels(encoder_ckpt_dir)
    return FusedStack(encoders, fusion, cbm, d_model, in_channels)


# ---------------------------------------------------------------------------
# Per-patient fused representations
# ---------------------------------------------------------------------------


@dataclass
class ModalityFused:
    """Fused representations sliced onto one modality's window order.

    ``df`` carries the exact (modality, quality_ok, n_channels) filtered rows in
    the same order as ``fused``/``starts``/``ends`` so callers can derive rule
    labels that align 1:1 with the probabilities.
    """

    df: pd.DataFrame
    fused: torch.Tensor  # (N_modality_windows, d_model)
    starts: np.ndarray
    ends: np.ndarray
    record_id: str


def _decode_window(row: pd.Series) -> np.ndarray:
    """parquet row -> (C, T) float32 (channels-first, as the encoder expects)."""
    shape = json.loads(row["signal_shape"])
    arr = np.frombuffer(row["signal_bytes"], dtype="float32").reshape(shape)
    return arr.T.copy()


@dataclass
class _RawWindows:
    df: pd.DataFrame
    keys: list[tuple[str, float]]
    starts: np.ndarray
    ends: np.ndarray
    emb: torch.Tensor
    record_id: str


def _encode_modality(
    patient_id: str,
    modality: str,
    encoder: SignalEncoder,
    processed_dir: Path,
    device: torch.device,
    batch_size: int,
    in_channels: int | None,
) -> _RawWindows | None:
    parquet = processed_dir / f"{patient_id}.parquet"
    if not parquet.exists():
        return None
    df = pd.read_parquet(parquet)
    df = df[(df["modality"] == modality) & df["quality_ok"].astype(bool)]
    if in_channels is not None and "n_channels" in df.columns:
        df = df[df["n_channels"] == in_channels]
    if df.empty:
        return None
    df = df.reset_index(drop=True)

    starts = df["window_start_sec"].to_numpy(dtype=np.float64)
    ends = df["window_end_sec"].to_numpy(dtype=np.float64)
    segs = df["segment_name"].astype(str).to_numpy()
    keys = [(str(s), round(float(w), ALIGN_DECIMALS)) for s, w in zip(segs, starts)]
    record_id = str(segs[0]) if len(segs) else patient_id

    arrays = [_decode_window(df.iloc[i]) for i in range(len(df))]
    embs: list[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(arrays), batch_size):
            batch = np.stack(arrays[i : i + batch_size])
            x = torch.from_numpy(batch).to(device)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            embs.append(encoder(x).cpu())
    emb = torch.cat(embs, dim=0)
    return _RawWindows(df, keys, starts, ends, emb, record_id)


def patient_fused_by_modality(
    patient_id: str,
    stack: FusedStack,
    processed_dir: Path,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, ModalityFused]:
    """Encode → align → fuse for one patient.

    Returns ``{modality: ModalityFused}`` for every modality with usable
    windows.  An empty dict means the patient has no usable signal.
    """
    per_mod: dict[str, _RawWindows] = {}
    for mod in MODALITY_ORDER:
        enc = stack.encoders.get(mod)
        if enc is None:
            continue
        raw = _encode_modality(
            patient_id,
            mod,
            enc,
            processed_dir,
            device,
            batch_size,
            stack.in_channels.get(mod),
        )
        if raw is not None:
            per_mod[mod] = raw
    if not per_mod:
        return {}

    # Union of aligned window keys across present modalities.
    key_to_idx: dict[tuple[str, float], int] = {}
    for raw in per_mod.values():
        for k in raw.keys:
            if k not in key_to_idx:
                key_to_idx[k] = len(key_to_idx)
    n_keys = len(key_to_idx)

    embeddings: dict[str, torch.Tensor] = {
        mod: torch.zeros(n_keys, stack.d_model) for mod in MODALITY_ORDER
    }
    present = torch.zeros(n_keys, len(MODALITY_ORDER), dtype=torch.bool)
    mod_axis = {m: i for i, m in enumerate(MODALITY_ORDER)}
    for mod, raw in per_mod.items():
        rows = torch.tensor([key_to_idx[k] for k in raw.keys], dtype=torch.long)
        embeddings[mod][rows] = raw.emb.float()
        present[rows, mod_axis[mod]] = True

    stack.fusion.eval()
    with torch.no_grad():
        fused_all = stack.fusion(
            {m: embeddings[m].to(device) for m in MODALITY_ORDER},
            mask=present.to(device),
        )  # (n_keys, d_model)

    out: dict[str, ModalityFused] = {}
    for mod, raw in per_mod.items():
        rows = torch.tensor([key_to_idx[k] for k in raw.keys], dtype=torch.long)
        out[mod] = ModalityFused(
            df=raw.df,
            fused=fused_all[rows.to(device)],
            starts=raw.starts,
            ends=raw.ends,
            record_id=raw.record_id,
        )
    return out


def concept_probs(
    stack: FusedStack, modality: str, concept: str, fused: torch.Tensor
) -> np.ndarray:
    """Run one CBM head over a modality's fused representations -> (N,) probs."""
    head = stack.cbm.heads.get(modality, {}).get(concept)
    if head is None:
        return np.empty((0,), dtype=np.float64)
    head = head.to(fused.device)
    head.eval()
    with torch.no_grad():
        return head.predict_proba(fused).cpu().numpy().astype(np.float64)
