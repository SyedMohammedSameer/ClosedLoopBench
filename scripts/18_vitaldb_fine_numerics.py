"""
Export VitalDB monitor numerics at a fine time step (default 2 s) for a subset of cases.

The 30 s windows the KG is built on average away the fast loops: the baroreflex settles in 1–5 s
and respiratory sinus arrhythmia acts within a breath. At a 2 s step those dynamics are visible as
*lagged* effects, with thousands of transitions per patient instead of a handful of treatment
changes — enough to fit them without instruments, and to test whether aggregating a fine fit up to
30 s reproduces the within-window (feedback) structure the KG claims.

Reuses :func:`clbench.data.vitaldb.window_numerics` with non-overlapping windows of ``--step-sec``,
so the columns match the 30 s tables exactly.

USAGE
─────
  # Colab, over the Drive copy — pressor cases first, then fill up to --limit
  NG=/content/drive/MyDrive/CLBench
  python scripts/18_vitaldb_fine_numerics.py \\
      --raw-dir $NG/data/raw/physionet.org/files/vitaldb/1.0.0 \\
      --out-dir $NG/data/processed_vitaldb/fine_2s \\
      --manifest $NG/data/vitaldb_manifest.csv --limit 500 --workers 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from clbench.data.vitaldb import patient_id_for, window_numerics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("18_vitaldb_fine_numerics")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "physionet.org" / "files" / "vitaldb" / "1.0.0"
OUT_DIR = ROOT / "data" / "processed_vitaldb" / "fine_2s"
MANIFEST = ROOT / "data" / "vitaldb_manifest.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export fine-step VitalDB numerics.")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--step-sec", type=float, default=2.0, help="Sampling step (default: 2 s)")
    p.add_argument("--limit", type=int, default=500, help="Cases to export (default: 500)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-skip-existing", action="store_true")
    return p.parse_args()


def _run_one(job: tuple[str, int, str, float]) -> tuple[int, int, str | None]:
    vital_path, caseid, out_dir, step = job
    try:
        import vitaldb

        vf = vitaldb.VitalFile(str(vital_path))
        df = window_numerics(vf, caseid, window_sec=step, stride_sec=step)
        df.to_parquet(Path(out_dir) / f"{patient_id_for(caseid)}.parquet", index=False)
        return caseid, len(df), None
    except Exception as exc:
        return caseid, 0, f"{type(exc).__name__}: {exc}"


def select_cases(manifest: Path, limit: int) -> list[int]:
    """Vasopressor cases first (they carry a pressure-only instrument), then the longest recordings."""
    m = pd.read_csv(manifest)
    pressor = m[m["vasoactive_pumps"].notna()].sort_values("duration_h", ascending=False)
    rest = m[m["vasoactive_pumps"].isna()].sort_values("duration_h", ascending=False)
    ordered = pd.concat([pressor, rest])["caseid"].astype(int).tolist()
    return ordered[:limit]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vital_dir = args.raw_dir / "vital_files"

    cases = select_cases(args.manifest, args.limit)
    on_disk = [c for c in cases if (vital_dir / f"{c:04d}.vital").exists()]
    todo = on_disk
    if not args.no_skip_existing:
        done = {p.stem for p in args.out_dir.glob("*.parquet")}
        todo = [c for c in on_disk if patient_id_for(c) not in done]
    log.info("Exporting %d cases at %.1f s (%d already done, %d missing locally) with %d workers",
             len(todo), args.step_sec, len(on_disk) - len(todo), len(cases) - len(on_disk), args.workers)

    jobs = [(str(vital_dir / f"{c:04d}.vital"), c, str(args.out_dir), args.step_sec) for c in todo]
    rows, failed, t0 = 0, [], time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                caseid, n, error = fut.result()
                rows += n
                if error:
                    failed.append((caseid, error))
                    log.warning("  case %d failed: %s", caseid, error)
                if i % 50 == 0 or i == len(jobs):
                    per = (time.time() - t0) / i
                    log.info("  %d/%d · %.1f s/case · ETA %.1f min", i, len(jobs), per,
                             (len(jobs) - i) * per / 60)
    log.info("Done: %d cases, %,d rows, %d failed → %s".replace(",d", "d"),
             len(jobs) - len(failed), rows, len(failed), args.out_dir)


if __name__ == "__main__":
    main()
