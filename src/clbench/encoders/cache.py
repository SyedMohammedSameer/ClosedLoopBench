"""
Encoder feature caching for CLBench Phase 2.1.

After pre-training, the encoders are frozen and their forward-pass outputs
are cached to disk so that downstream modules (fusion, CBM) can load
pre-computed features without re-running the encoder on every training step.

Cache layout
------------
cache/encoder_features/
  {patient_id}/
    {modality}_{run_id}.pt    # torch.Tensor of shape (N_windows, d_model)
    {modality}_{run_id}.meta.json  # {patient_id, modality, run_id, n_windows, d_model}

Cache invalidation
------------------
The ``run_id`` is derived from a hash of:
  - The encoder checkpoint directory (contains config.json with architecture)
  - The processed parquet mtime for the patient
  - The modality string

If the encoder checkpoint changes or the parquet is updated, a new run_id is
produced and the old cache entry is superseded (but not deleted automatically).
A separate LRU eviction script handles cleanup.

Usage
-----
From Python::

    from clbench.encoders.cache import cache_encoder_features, load_cached_features

    # Cache features for all patients (run once after encoder pretraining)
    cache_encoder_features(
        encoder=ecg_encoder,
        processed_dir=Path("data/processed"),
        patient_ids=all_patient_ids,
        modality="ECG",
        device=torch.device("mps"),
        cache_dir=Path("cache/encoder_features"),
    )

    # Load cached features for one patient
    features = load_cached_features(
        patient_id="80057524",
        modality="ECG",
        cache_dir=Path("cache/encoder_features"),
    )  # shape: (N_windows, d_model)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from clbench.encoders.base import SignalEncoder

log = logging.getLogger(__name__)

CACHE_DIR_DEFAULT = Path(__file__).parent.parent.parent.parent / "cache" / "encoder_features"


# ---------------------------------------------------------------------------
# Run-ID derivation (content-hash-based invalidation)
# ---------------------------------------------------------------------------


def _compute_run_id(
    checkpoint_dir: Path,
    parquet_path: Path,
    modality: str,
) -> str:
    """
    Derive an 8-char run_id from a hash of the checkpoint dir + parquet mtime + modality.

    This ensures that changing the checkpoint or reprocessing the parquet
    invalidates the cache entry automatically.
    """
    cfg_path = checkpoint_dir / "config.json"
    cfg_bytes = cfg_path.read_bytes() if cfg_path.exists() else b""
    mtime = str(parquet_path.stat().st_mtime) if parquet_path.exists() else "0"
    h = hashlib.sha256(cfg_bytes + mtime.encode() + modality.encode())
    return h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@torch.no_grad()  # type: ignore[misc]
def cache_encoder_features(
    encoder: SignalEncoder,
    processed_dir: Path,
    patient_ids: list[str],
    modality: str,
    device: torch.device,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    checkpoint_dir: Path | None = None,
    batch_size: int = 128,
    quality_only: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Run the frozen encoder over all windows for each patient and cache the results.

    Parameters
    ----------
    encoder : SignalEncoder
        Must be pre-trained.  Will be set to eval().
    processed_dir : Path
        Directory of ``{patient_id}.parquet`` files.
    patient_ids : list[str]
    modality : str
    device : torch.device
    cache_dir : Path
        Root of the cache tree.
    checkpoint_dir : Path | None
        Used for run_id derivation.  If None, uses a hash of encoder.config().
    batch_size : int
    quality_only : bool
        If True (default), skip windows flagged quality_ok=False.
    overwrite : bool
        If True, re-compute even if cached file exists.

    Returns
    -------
    dict mapping patient_id → path of the cached .pt file
    """
    encoder.eval()
    encoder.to(device)

    # Fallback run_id when no checkpoint dir is given
    if checkpoint_dir is None:
        cfg_bytes = json.dumps(encoder.config(), sort_keys=True).encode()
        run_id = hashlib.sha256(cfg_bytes).hexdigest()[:8]

    written: dict[str, Path] = {}
    n_skipped = 0

    for pid in patient_ids:
        parquet_path = processed_dir / f"{pid}.parquet"
        if not parquet_path.exists():
            continue

        if checkpoint_dir is not None:
            run_id = _compute_run_id(checkpoint_dir, parquet_path, modality)

        patient_cache_dir = cache_dir / pid
        pt_path = patient_cache_dir / f"{modality}_{run_id}.pt"
        meta_path = patient_cache_dir / f"{modality}_{run_id}.meta.json"

        if pt_path.exists() and not overwrite:
            n_skipped += 1
            written[pid] = pt_path
            continue

        # Load parquet and filter
        df = pd.read_parquet(parquet_path)
        df = df[df["modality"] == modality]
        if quality_only:
            df = df[df["quality_ok"].astype(bool)]
        # The encoder was pre-trained on a fixed channel count (e.g. 3-lead ECG).
        # Some rows of the same modality carry a different channel count (e.g.
        # single-lead ECG windows); feeding those to the conv stem raises a
        # channel-mismatch RuntimeError. Filter to the encoder's in_channels,
        # mirroring the n_channels filter used at export/training time.
        if "n_channels" in df.columns:
            df = df[df["n_channels"] == encoder.in_channels]
        if df.empty:
            log.debug("No usable %s windows for patient %s", modality, pid)
            continue

        # Build signal arrays
        arrays: list[np.ndarray] = []
        for _, row in df.iterrows():
            shape: list[int] = json.loads(row["signal_shape"])
            arr = np.frombuffer(row["signal_bytes"], dtype="float32").reshape(shape)
            arrays.append(arr.T.copy())  # (C, T)

        # Extract embeddings in batches
        all_embs: list[Tensor] = []
        for start in range(0, len(arrays), batch_size):
            batch_np = np.stack(arrays[start : start + batch_size])
            x = torch.from_numpy(batch_np).to(device)
            # Sanitize NaN/Inf before the encoder: a single NaN propagates to a
            # NaN embedding, which silently corrupts the cached features (the
            # same failure mode that pinned the linear probes at ~0.5). The MAE
            # forward sanitizes, but the plain encoder forward does not.
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            emb = encoder(x)  # (B, d_model)
            all_embs.append(emb.cpu())

        features = torch.cat(all_embs, dim=0)  # (N_windows, d_model)

        # Save
        patient_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(features, pt_path)
        meta = {
            "patient_id": pid,
            "modality": modality,
            "run_id": run_id,
            "n_windows": int(features.shape[0]),
            "d_model": int(features.shape[1]),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written[pid] = pt_path

    log.info(
        "cache_encoder_features [%s]: %d patients written, %d skipped (cached)",
        modality,
        len(written) - n_skipped,
        n_skipped,
    )
    return written


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_cached_features(
    patient_id: str,
    modality: str,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    run_id: str | None = None,
) -> Tensor:
    """
    Load cached encoder features for one patient.

    If ``run_id`` is None, loads the most recently modified ``.pt`` file
    for the given patient and modality.

    Parameters
    ----------
    patient_id : str
    modality : str
    cache_dir : Path
    run_id : str | None

    Returns
    -------
    Tensor of shape (N_windows, d_model)

    Raises
    ------
    FileNotFoundError
        If no cached features exist for the patient/modality.
    """
    patient_cache_dir = cache_dir / patient_id

    if run_id is not None:
        pt_path = patient_cache_dir / f"{modality}_{run_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"No cached features for {patient_id}/{modality}/{run_id}")
        return torch.load(pt_path, map_location="cpu", weights_only=True)

    # Find most recent match
    candidates = sorted(
        patient_cache_dir.glob(f"{modality}_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No cached {modality} features for patient {patient_id} in {cache_dir}"
        )
    return torch.load(candidates[0], map_location="cpu", weights_only=True)


def list_cached_patients(
    modality: str,
    cache_dir: Path = CACHE_DIR_DEFAULT,
) -> list[str]:
    """Return patient IDs that have at least one cached feature file for ``modality``."""
    if not cache_dir.exists():
        return []
    return sorted(
        pid_dir.name
        for pid_dir in cache_dir.iterdir()
        if pid_dir.is_dir() and any(pid_dir.glob(f"{modality}_*.pt"))
    )
