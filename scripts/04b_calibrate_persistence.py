"""
Phase 2.3 — Persistence-tier calibration.

Runs a trained ConceptCBM over the train + val splits, extracts per-concept
activation runs from each patient's chronological window sequence, and
fits the {transient, episodic, persistent} tier thresholds from the
pooled run-duration distribution.

Output
------
checkpoints/cbm/persistence_calibration.json

Consumed by Phase 3.1 KG construction to type each (concept, run) into
(concept, tier) node identities.

Usage
-----
    uv run python scripts/04b_calibrate_persistence.py

    # Use a non-default activation threshold instead of best-F1 calibration:
    uv run python scripts/04b_calibrate_persistence.py --threshold 0.5

    # Restrict the cohort for a smoke test:
    uv run python scripts/04b_calibrate_persistence.py --limit-patients 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from clbench.cbm import CONCEPT_REGISTRY  # noqa: E402
from clbench.data.split import get_split_ids  # noqa: E402
from clbench.encoders.linear_probe import label_windows  # noqa: E402
from clbench.fused_inference import (  # noqa: E402
    concept_probs,
    load_fused_stack,
    patient_fused_by_modality,
)
from clbench.persistence import (  # noqa: E402
    CalibrationConfig,
    TIER_PERCENTILES_DEFAULT,
    WINDOW_STRIDE_SEC_DEFAULT,
    calibrate_concept,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("04b_calibrate_persistence")

SEPARATOR = "=" * 70
PROCESSED_DIR = _REPO / "data" / "processed"
CBM_CKPT_DIR = _REPO / "checkpoints" / "cbm"
ENCODER_CKPT_DIR = _REPO / "checkpoints" / "encoders"
FUSION_CKPT_DIR = _REPO / "checkpoints" / "fusion"
OUTPUT_PATH_DEFAULT = CBM_CKPT_DIR / "persistence_calibration.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2.3 — persistence-tier calibration")
    p.add_argument("--device", default=None)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--cbm-ckpt-dir", type=Path, default=CBM_CKPT_DIR)
    p.add_argument("--encoder-ckpt-dir", type=Path, default=ENCODER_CKPT_DIR)
    p.add_argument("--fusion-ckpt-dir", type=Path, default=FUSION_CKPT_DIR)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Encoder forward batch size (default: 256)")
    p.add_argument("--out", type=Path, default=OUTPUT_PATH_DEFAULT,
                   help=f"Output JSON path (default: {OUTPUT_PATH_DEFAULT})")
    p.add_argument("--limit-patients", type=int, default=None)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override activation threshold for all concepts.  When omitted, "
            "the threshold is calibrated per concept by best-F1 against the "
            "rule-derived labels."
        ),
    )
    p.add_argument(
        "--percentiles",
        nargs=2,
        type=float,
        default=list(TIER_PERCENTILES_DEFAULT),
        metavar=("LOW", "HIGH"),
        help=(
            "Lower / upper percentile cutoffs used to split the run-duration "
            f"distribution into the three tiers (default: {TIER_PERCENTILES_DEFAULT})."
        ),
    )
    p.add_argument(
        "--window-stride-sec",
        type=float,
        default=WINDOW_STRIDE_SEC_DEFAULT,
        help=f"Seconds between consecutive windows (default: {WINDOW_STRIDE_SEC_DEFAULT})",
    )
    return p.parse_args()


def _select_device(s: str | None) -> torch.device:
    if s:
        return torch.device(s)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Activation extraction per patient
# ---------------------------------------------------------------------------


def collect_all_activations(
    stack,
    patient_ids: list[str],
    processed_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Run encoder → fusion → CBM once per patient and collect probabilities
    and rule labels for every concept.

    Computing through the fused stack (instead of the per-modality cache)
    makes calibration use the SAME probabilities that ``05_build_kgs.py``
    builds the graphs from, and avoids the stale-cache d_model mismatch.

    Returns
    -------
    probs_by_concept  : {concept: {pid: (N_pid,) float ndarray}}
    labels_by_concept : {concept: {pid: (N_pid,) int ndarray}} — same order.
    """
    concepts_by_modality: dict[str, list[str]] = {}
    for entry in CONCEPT_REGISTRY:
        concepts_by_modality.setdefault(entry["modality"], []).append(entry["concept"])

    probs_by_concept: dict[str, dict[str, np.ndarray]] = {
        e["concept"]: {} for e in CONCEPT_REGISTRY
    }
    labels_by_concept: dict[str, dict[str, np.ndarray]] = {
        e["concept"]: {} for e in CONCEPT_REGISTRY
    }

    for i, pid in enumerate(patient_ids, 1):
        try:
            by_mod = patient_fused_by_modality(
                pid, stack, processed_dir, device, batch_size=batch_size,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Patient %s fused inference failed: %s — skipping", pid, exc)
            by_mod = {}
        for modality, concept_keys in concepts_by_modality.items():
            mf = by_mod.get(modality)
            if mf is None:
                continue
            for concept in concept_keys:
                probs = concept_probs(stack, modality, concept, mf.fused)
                if probs.size == 0:
                    continue
                # Rule labels are concept-specific; recompute per concept.
                try:
                    _, labs_c = label_windows(mf.df, concept)
                except ValueError:
                    continue
                labs_arr = np.asarray(labs_c, dtype=np.int32)
                if len(labs_arr) != len(probs):
                    log.warning(
                        "Length mismatch %s/%s/%s: probs=%d labels=%d — skipping",
                        modality, concept, pid, len(probs), len(labs_arr),
                    )
                    continue
                probs_by_concept[concept][pid] = probs
                labels_by_concept[concept][pid] = labs_arr
        if i % 20 == 0:
            log.info("  fused activations: %d/%d patients", i, len(patient_ids))

    return probs_by_concept, labels_by_concept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)

    log.info(SEPARATOR)
    log.info("CLBench Phase 2.3 — Persistence-tier Calibration")
    log.info(SEPARATOR)
    log.info("Device: %s", device)

    try:
        stack = load_fused_stack(
            args.encoder_ckpt_dir, args.fusion_ckpt_dir, args.cbm_ckpt_dir, device,
        )
    except FileNotFoundError as exc:
        log.error("%s — run scripts/03_train_e2e.py / 04_train_cbm.py first", exc)
        sys.exit(1)
    log.info("Encoder in_channels: %s | cbm/fusion d_model=%d",
             stack.in_channels, stack.d_model)

    train_ids = list(get_split_ids("train")) + list(get_split_ids("val"))
    if args.limit_patients:
        train_ids = train_ids[: args.limit_patients]
    log.info("Patients (train+val): %d", len(train_ids))

    # Single fused pass over all patients → probs + labels for every concept.
    log.info("")
    log.info("Running encoder → fusion → CBM over the cohort ...")
    t_fused = time.time()
    probs_by_concept, labels_by_concept = collect_all_activations(
        stack, train_ids, args.processed_dir, device, args.batch_size,
    )
    log.info("Fused activations done in %.1f min", (time.time() - t_fused) / 60)

    concepts_cfg: dict[str, dict] = {}
    for entry in CONCEPT_REGISTRY:
        modality = entry["modality"]
        concept = entry["concept"]
        log.info("")
        log.info("-- %s / %s --", modality, concept)
        t0 = time.time()

        probs_per_pid = probs_by_concept.get(concept, {})
        labels_per_pid = labels_by_concept.get(concept, {})
        if not probs_per_pid:
            log.warning("No data for %s/%s — emitting fallback thresholds", modality, concept)
            from clbench.persistence import TierThresholds
            concepts_cfg[concept] = TierThresholds(
                concept=concept,
                transient_max_sec=60.0,
                episodic_max_sec=600.0,
                activation_threshold=0.5,
                n_runs_observed=0,
                source="fallback",
            )
            continue

        # If a uniform --threshold override was passed, use it; otherwise let
        # calibrate_concept pick per-concept via best-F1 against rule labels.
        if args.threshold is not None:
            log.info("  Using override activation threshold %.3f", args.threshold)
            # Drop labels so calibrate_concept skips the F1 search.
            labels_per_pid = {}

        thresholds = calibrate_concept(
            concept=concept,
            probs_per_patient=probs_per_pid,
            labels_per_patient=labels_per_pid if labels_per_pid else None,
            window_stride_sec=args.window_stride_sec,
            percentiles=tuple(args.percentiles),
        )
        # Optional override
        if args.threshold is not None:
            thresholds.activation_threshold = float(args.threshold)
            thresholds.source = "fixed-threshold"

        concepts_cfg[concept] = thresholds
        log.info(
            "  → threshold=%.3f, transient_max=%.1fs, episodic_max=%.1fs, "
            "n_runs=%d, source=%s  (%.1fs)",
            thresholds.activation_threshold,
            thresholds.transient_max_sec,
            thresholds.episodic_max_sec,
            thresholds.n_runs_observed,
            thresholds.source,
            time.time() - t0,
        )

    cfg = CalibrationConfig(
        concepts=concepts_cfg,
        window_stride_sec=args.window_stride_sec,
        tier_percentiles=tuple(args.percentiles),
    )
    cfg.save(args.out)

    log.info("")
    log.info(SEPARATOR)
    log.info("Phase 2.3 complete  →  %s", args.out)
    log.info(SEPARATOR)
    log.info("")
    log.info("Next: uv run python scripts/05_build_kgs.py")


if __name__ == "__main__":
    main()
