"""
VitalDB → CLBench signal windows, window numerics, and intervention events.

Runs :func:`clbench.data.vitaldb.ingest_case` over downloaded cases (see
``scripts/01c_download_vitaldb.py``) and writes, per case, under ``--out-dir``:

    {patient_id}.parquet             signal windows — same schema as data/processed/
    numerics/{patient_id}.parquet    monitor numbers, TCI / infusion / ventilator state per window
    events/{patient_id}.parquet      timestamped clinician interventions
    summaries/{patient_id}.json      per-case counts (written last; marks the case as done)

then a cohort manifest joining the summaries with the case index and ``clinical_data.csv``.

Resumable: cases that already have a summary are skipped unless ``--no-skip-existing``.
Cases whose ``.vital`` file is not on disk are skipped with a count, so a partial local copy works.

USAGE
─────
  # Local smoke test on the .vital files present locally
  uv run python scripts/14_vitaldb_ingest.py --limit 3

  # Colab, reading the Drive copy (raw data stays on Drive)
  NG=/content/drive/MyDrive/CLBench
  python scripts/14_vitaldb_ingest.py \\
      --raw-dir $NG/data/raw/physionet.org/files/vitaldb/1.0.0 \\
      --out-dir $NG/data/processed_vitaldb \\
      --case-index $NG/data/vitaldb_case_index.csv \\
      --manifest $NG/data/vitaldb_manifest.csv \\
      --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from clbench.data.preprocess import STRIDE_SEC_DEFAULT, WINDOW_SEC_DEFAULT
from clbench.data.vitaldb import EVENT_KINDS, MODALITIES, ingest_case, patient_id_for

# ─── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("14_vitaldb_ingest")

# ─── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "physionet.org" / "files" / "vitaldb" / "1.0.0"
OUT_DIR = ROOT / "data" / "processed_vitaldb"
CASE_INDEX_PATH = ROOT / "data" / "vitaldb_case_index.csv"
MANIFEST_PATH = ROOT / "data" / "vitaldb_manifest.csv"

CLINICAL_COLUMNS = [
    "caseid", "subjectid", "age", "sex", "height", "weight", "bmi", "asa", "emop",
    "department", "optype", "ane_type", "preop_htn", "preop_dm", "icu_days", "death_inhosp",
]


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest VitalDB cases into CLBench windows, numerics and events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                   help=f"VitalDB 1.0.0 root with vital_files/ (default: {RAW_DIR})")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help=f"Output root (default: {OUT_DIR})")
    p.add_argument("--case-index", type=Path, default=CASE_INDEX_PATH,
                   help=f"Case index from 01c_download_vitaldb.py (default: {CASE_INDEX_PATH})")
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                   help=f"Cohort manifest to write (default: {MANIFEST_PATH})")
    p.add_argument("--max-tier", type=int, default=2, choices=(0, 1, 2, 3),
                   help="Ingest tiers 0..N (default: 2 = every four-modality case)")
    p.add_argument("--caseids", type=int, nargs="*", default=None,
                   help="Ingest only these case ids")
    p.add_argument("--limit", type=int, default=None,
                   help="Ingest at most N cases (smoke test)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel processes; each holds one case at 500 Hz in memory (default: 4)")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC_DEFAULT,
                   help=f"Window length in seconds (default: {WINDOW_SEC_DEFAULT})")
    p.add_argument("--stride-sec", type=float, default=STRIDE_SEC_DEFAULT,
                   help=f"Stride in seconds (default: {STRIDE_SEC_DEFAULT})")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-ingest cases that already have a summary")
    return p.parse_args()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _run_one(job: tuple[str, int, str, float, float]) -> tuple[int, dict | None, str | None]:
    vital_path, caseid, out_dir, window_sec, stride_sec = job
    try:
        summary = ingest_case(Path(vital_path), caseid, Path(out_dir), window_sec, stride_sec)
        return caseid, summary, None
    except Exception as exc:  # one unreadable case must not stop the cohort
        return caseid, None, f"{type(exc).__name__}: {exc}"


def build_manifest(out_dir: Path, case_index: pd.DataFrame, clinical_path: Path,
                   manifest_path: Path) -> pd.DataFrame:
    """Join every case summary with its tier and clinical data."""
    summaries = [json.loads(p.read_text()) for p in sorted((out_dir / "summaries").glob("*.json"))]
    if not summaries:
        return pd.DataFrame()
    df = pd.json_normalize(summaries)
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.merge(case_index[["caseid", "tier", "vasoactive_pumps", "tci_pumps"]],
                  on="caseid", how="left")
    if clinical_path.exists():
        clinical = pd.read_csv(clinical_path, encoding="utf-8-sig")
        df = df.merge(clinical[[c for c in CLINICAL_COLUMNS if c in clinical.columns]],
                      on="caseid", how="left")
    else:
        log.warning("clinical_data.csv not found at %s; manifest has no clinical columns",
                    clinical_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("caseid").to_csv(manifest_path, index=False)
    return df


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    vital_dir = args.raw_dir / "vital_files"

    log.info("=" * 60)
    log.info("VitalDB ingest — windows, numerics, interventions")
    log.info("=" * 60)
    log.info("Raw dir : %s", args.raw_dir)
    log.info("Out dir : %s", args.out_dir)

    index = pd.read_csv(args.case_index)
    selected = index[index["tier"] <= args.max_tier]
    if args.caseids:
        selected = selected[selected["caseid"].isin(args.caseids)]
    log.info("[STEP 1] %d cases in tiers 0..%d", len(selected), args.max_tier)

    on_disk = [int(c) for c in selected["caseid"] if (vital_dir / f"{int(c):04d}.vital").exists()]
    if len(on_disk) < len(selected):
        log.info("  %d selected cases have no .vital file here — skipped", len(selected) - len(on_disk))

    todo = on_disk
    if not args.no_skip_existing:
        done = {p.stem for p in (args.out_dir / "summaries").glob("*.json")}
        todo = [c for c in on_disk if patient_id_for(c) not in done]
        log.info("  %d already ingested", len(on_disk) - len(todo))
    if args.limit is not None:
        todo = todo[: args.limit]
    log.info("[STEP 2] Ingesting %d cases with %d workers", len(todo), args.workers)

    jobs = [(str(vital_dir / f"{c:04d}.vital"), c, str(args.out_dir), args.window_sec, args.stride_sec)
            for c in todo]
    failed: list[tuple[int, str]] = []
    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                caseid, summary, error = fut.result()
                if error:
                    failed.append((caseid, error))
                    log.warning("  case %d failed: %s", caseid, error)
                if i % 10 == 0 or i == len(jobs):
                    per_case = (time.time() - t0) / i
                    log.info("  %d/%d cases · %.1f s/case · %d failed · ETA %.1f h",
                             i, len(jobs), per_case, len(failed), (len(jobs) - i) * per_case / 3600)

    log.info("[STEP 3] Building manifest ...")
    manifest = build_manifest(args.out_dir, index, args.raw_dir / "clinical_data.csv", args.manifest)

    log.info("")
    log.info("=" * 60)
    log.info("VitalDB ingest complete")
    if not manifest.empty:
        win_cols = [f"n_windows_ok_{m}" for m in MODALITIES if f"n_windows_ok_{m}" in manifest]
        evt_cols = [f"n_events_{k}" for k in EVENT_KINDS if f"n_events_{k}" in manifest]
        gb = manifest["window_parquet_bytes"].sum() / 1e9
        log.info("  Cases ingested     : %d", len(manifest))
        log.info("  Quality windows    : %s", {c.rsplit("_", 1)[-1]: int(manifest[c].sum()) for c in win_cols})
        log.info("  Intervention events: %s", {c.replace("n_events_", ""): int(manifest[c].sum()) for c in evt_cols})
        if "death_inhosp" in manifest:
            log.info("  In-hospital deaths : %d", int(manifest["death_inhosp"].fillna(0).sum()))
        log.info("  Window parquet     : %.2f GB (≈ %.0f GB for all %d selected cases)",
                 gb, gb / len(manifest) * len(selected), len(selected))
        log.info("  Manifest           : %s", args.manifest)
    log.info("  Failed this run    : %d", len(failed))
    log.info("=" * 60)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
