"""
Stratified 140 / 30 / 30 data split for CLBench V1.

Stratification variable: ICU LOS quartile (from manifest.csv).
Patients without a MIMIC-IV link (mortality_label == -1 / no stay_id)
are assigned to the training set only — they contribute to self-supervised
encoder pre-training but cannot be used for outcome evaluation.

The test split is frozen at first generation (the "day-one" freeze policy).
Once splits.json is committed, it must not be regenerated unless the manifest
itself changes.

Usage
-----
From Python::

    from clbench.data.split import make_splits, load_splits

    splits = make_splits()          # generate and save
    splits = load_splits()          # load existing

Output format (data/splits.json)
---------------------------------
{
  "version": "1.0",
  "n_train": 140,
  "n_val":    30,
  "n_test":   30,
  "stratified_by": "icu_los_quartile",
  "random_seed": 42,
  "train": ["80030009", ...],
  "val":   ["...", ...],
  "test":  ["...", ...]
}
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent.parent.parent
MANIFEST_PATH = _REPO / "data" / "manifest.csv"
SPLITS_PATH = _REPO / "data" / "splits.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_VERSION = "1.0"
N_TRAIN = 140
N_VAL = 30
N_TEST = 30
N_QUARTILES = 4
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------


def _assign_quartile(los_values: pd.Series) -> pd.Series:
    """
    Bin ICU LOS into quartiles (0-3).  Patients with no LOS (value <= 0)
    are assigned to stratum 4 (train-only bucket).
    """
    q = pd.Series(np.full(len(los_values), 4, dtype=int), index=los_values.index)
    has_los = los_values > 0
    if has_los.sum() >= N_QUARTILES:
        q[has_los] = pd.qcut(
            los_values[has_los],
            q=N_QUARTILES,
            labels=False,
            duplicates="drop",
        ).astype(int)
    return q


def _stratified_split(
    patient_ids: list[str],
    strata: list[int],
    n_val: int,
    n_test: int,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """
    Manually stratified split without scikit-learn.

    Patients in stratum 4 (no LOS / no MIMIC link) go entirely to train.
    Remaining strata are sampled proportionally for val and test.

    Returns
    -------
    (train_ids, val_ids, test_ids)
    """
    rng = np.random.default_rng(seed)

    train_only: list[str] = []
    stratified: dict[int, list[str]] = defaultdict(list)
    for pid, s in zip(patient_ids, strata):
        if s == 4:
            train_only.append(pid)
        else:
            stratified[s].append(pid)

    pool_ids = [p for pids in stratified.values() for p in pids]
    n_pool = len(pool_ids)

    val_ids: list[str] = []
    test_ids: list[str] = []

    for _stratum, pids in sorted(stratified.items()):
        shuffled = rng.permutation(pids).tolist()
        n_s_test = max(1, round(n_test * len(pids) / n_pool))
        n_s_val = max(1, round(n_val * len(pids) / n_pool))
        test_ids.extend(shuffled[:n_s_test])
        val_ids.extend(shuffled[n_s_test : n_s_test + n_s_val])

    # Trim if rounding overshot
    val_ids = val_ids[:n_val]
    test_ids = test_ids[:n_test]

    held_out: set[str] = set(val_ids + test_ids)
    train_ids = train_only + [pid for pid in pool_ids if pid not in held_out]

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_splits(
    manifest_path: Path | None = None,
    splits_path: Path | None = None,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Generate and save the stratified train / val / test split.

    Parameters
    ----------
    manifest_path:
        Path to ``data/manifest.csv``.
    splits_path:
        Destination for ``splits.json``.
    n_val, n_test:
        Target sizes (actual sizes may vary slightly due to rounding).
    seed:
        RNG seed for reproducibility.
    overwrite:
        If False and splits.json already exists, load and return it.

    Returns
    -------
    dict
        The full splits dict (same structure as the JSON file).
    """
    mpath = manifest_path or MANIFEST_PATH
    spath = splits_path or SPLITS_PATH

    if spath.exists() and not overwrite:
        log.info("splits.json already exists at %s — loading existing split", spath)
        return load_splits(spath)

    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    manifest = pd.read_csv(mpath)
    usable = manifest[manifest["usable"].astype(bool)].copy()
    usable["los"] = pd.to_numeric(
        usable.get("icu_los_days", pd.Series(0.0, index=usable.index)),
        errors="coerce",
    ).fillna(0.0)

    patient_ids = usable["patient_id"].astype(str).tolist()
    strata = _assign_quartile(usable["los"]).tolist()

    train_ids, val_ids, test_ids = _stratified_split(
        patient_ids, strata, n_val=n_val, n_test=n_test, seed=seed
    )

    splits: dict[str, Any] = {
        "version": SPLIT_VERSION,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "stratified_by": "icu_los_quartile",
        "random_seed": seed,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    log.info(
        "Splits written → %s  (train=%d  val=%d  test=%d)",
        spath,
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    return splits


def load_splits(splits_path: Path | None = None) -> dict[str, Any]:
    """
    Load an existing splits.json.

    Raises
    ------
    FileNotFoundError
        If splits.json does not exist.  Run ``make_splits()`` first.
    """
    spath = splits_path or SPLITS_PATH
    if not spath.exists():
        raise FileNotFoundError(
            f"splits.json not found: {spath}\n"
            'Run: python -c "from clbench.data.split import make_splits; make_splits()"'
        )
    data: dict[str, Any] = json.loads(spath.read_text(encoding="utf-8"))
    return data


def get_split_ids(
    split: str,
    splits_path: Path | None = None,
) -> list[str]:
    """
    Return patient IDs for a named split (``'train'``, ``'val'``, or ``'test'``).
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")
    data = load_splits(splits_path)
    ids: list[str] = data[split]
    return ids
