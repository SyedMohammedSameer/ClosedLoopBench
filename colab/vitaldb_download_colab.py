# VitalDB → Google Drive, run on Colab (CPU runtime is enough; no GPU needed).
# Downloads all 6,388 VitalDB cases (~94 GB) from the PhysioNet open S3 mirror straight
# into Drive, in priority order, verifying every file's SHA256. Resumable: if the
# session drops, reconnect and re-run Cell 1 and Cell 3 — verified files are skipped.
#
# Later, pull only what you need to the laptop, e.g.:
#   rsync -av --progress "~/Library/CloudStorage/GoogleDrive-.../My Drive/CLBench/data/raw/physionet.org/files/vitaldb/" data/raw/physionet.org/files/vitaldb/

# %% ── Cell 1: mount Drive ─────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
import os
NG = '/content/drive/MyDrive/CLBench'
os.makedirs(f'{NG}/scripts', exist_ok=True)
print(os.listdir(NG))

# %% ── Cell 2: write the download script into Drive (paste as its own cell) ────
%%writefile /content/drive/MyDrive/CLBench/scripts/01c_download_vitaldb.py
"""
VitalDB v1.0.0 download, verification, and case index.

VitalDB is open access (CC BY 4.0): 6,388 surgical cases, one ``.vital`` file each
(~94 GB total), mirrored on the PhysioNet open-data S3 bucket — no credentials.
https://physionet.org/content/vitaldb/1.0.0/

Files land in the same layout wget/aws would produce, next to MIMIC-IV-WDB:
    data/raw/physionet.org/files/vitaldb/1.0.0/vital_files/0001.vital

Cases are downloaded in priority order so the sprint-critical cohort arrives first:
    tier 0  all four modalities (ECG_II, PLETH, ART, CO2) + a vasoactive infusion pump
    tier 1  all four modalities + propofol / remifentanil TCI
    tier 2  all four modalities, no infusion tracks
    tier 3  everything else

The script is resumable: finished files are recorded in ``vital_files/.verified.txt``
after their SHA256 matches, partial downloads resume from ``*.part`` via HTTP Range,
and a re-run skips verified files without re-hashing them.

USAGE
─────
  uv run python scripts/01c_download_vitaldb.py                    # everything, priority order
  uv run python scripts/01c_download_vitaldb.py --max-tier 2       # four-modality cases only
  uv run python scripts/01c_download_vitaldb.py --limit 20         # smoke test

OUTPUTS
───────
  data/raw/physionet.org/files/vitaldb/1.0.0/vital_files/*.vital
  data/vitaldb_case_index.csv   (caseid, tier, modality flags, infusion tracks)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import logging
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from clbench.data.download import _sha256
except ImportError:  # standalone copy (e.g. on Colab without the package)
    def _sha256(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

# ─── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("01c_download_vitaldb")

# ─── Paths & sources ───────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "physionet.org" / "files" / "vitaldb" / "1.0.0"
CASE_INDEX_PATH = ROOT / "data" / "vitaldb_case_index.csv"

S3_BASE = "https://physionet-open.s3.amazonaws.com/vitaldb/1.0.0"
TRKS_URL = "https://api.vitaldb.net/trks"
METADATA_FILES = (
    "LICENSE.txt",
    "SHA256SUMS.txt",
    "clinical_data.csv",
    "clinical_parameters.csv",
    "lab_data.csv",
    "lab_parameters.csv",
    "track_names.csv",
)

FOUR_MODALITIES = {
    "ecg": "SNUADC/ECG_II",
    "pleth": "SNUADC/PLETH",
    "art": "SNUADC/ART",
    "co2": "Primus/CO2",
}
VASOACTIVE_PUMPS = ("PHEN", "NEPI", "EPI", "VASO", "DOPA", "DOBU", "MRN", "NTG", "NPS", "DTZ")
TCI_PUMPS = ("PPF20", "RFTN20", "RFTN50")

CHUNK = 1 << 20
MAX_ATTEMPTS = 3


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and verify VitalDB v1.0.0 .vital files in priority order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                   help=f"Destination root (default: {RAW_DIR})")
    p.add_argument("--case-index", type=Path, default=CASE_INDEX_PATH,
                   help=f"Where to write the case index CSV (default: {CASE_INDEX_PATH})")
    p.add_argument("--workers", type=int, default=12,
                   help="Parallel downloads (default: 12)")
    p.add_argument("--max-tier", type=int, default=3, choices=(0, 1, 2, 3),
                   help="Download tiers 0..N only (default: 3 = all cases)")
    p.add_argument("--limit", type=int, default=None,
                   help="Download at most N cases (smoke test)")
    return p.parse_args()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_bytes(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def ensure_metadata(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        log.info("  fetching %s", name)
        dest.write_bytes(_fetch_bytes(f"{S3_BASE}/{name}"))


def load_expected_hashes(raw_dir: Path) -> dict[str, str]:
    """rel_path ('vital_files/0001.vital') → sha256, vital files only."""
    expected: dict[str, str] = {}
    with open(raw_dir / "SHA256SUMS.txt") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            digest, rel_path = parts
            rel_path = rel_path.lstrip("./")
            if rel_path.startswith("vital_files/") and rel_path.endswith(".vital"):
                expected[rel_path] = digest
    return expected


def build_case_index(caseids: list[int]) -> list[dict]:
    """Assign each case a download tier from the VitalDB track list."""
    raw = _fetch_bytes(TRKS_URL, timeout=300)
    text = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else raw.decode()
    tracks: dict[int, set[str]] = defaultdict(set)
    for row in csv.DictReader(io.StringIO(text)):
        tracks[int(row["caseid"])].add(row["tname"])

    rows = []
    for cid in caseids:
        names = tracks.get(cid, set())
        flags = {mod: int(track in names) for mod, track in FOUR_MODALITIES.items()}
        vaso = [d for d in VASOACTIVE_PUMPS if f"Orchestra/{d}_RATE" in names]
        tci = [d for d in TCI_PUMPS if f"Orchestra/{d}_RATE" in names]
        four = all(flags.values())
        tier = 0 if four and vaso else 1 if four and tci else 2 if four else 3
        rows.append({"caseid": cid, "tier": tier, **flags,
                     "vasoactive_pumps": ";".join(vaso), "tci_pumps": ";".join(tci)})
    rows.sort(key=lambda r: (r["tier"], r["caseid"]))
    return rows


def write_case_index(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def download_one(rel_path: str, digest: str, raw_dir: Path) -> tuple[str, int, str]:
    """Download (resuming) and verify one file. Returns (rel_path, bytes_fetched, status)."""
    dest = raw_dir / rel_path
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"{S3_BASE}/{rel_path}"
    fetched = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if dest.exists():
                if _sha256(dest) == digest:
                    return rel_path, fetched, "ok"
                dest.unlink()

            offset = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url)
            if offset:
                req.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(req, timeout=120) as resp:
                # Server ignored the Range header → start over.
                mode = "ab" if offset and resp.status == 206 else "wb"
                with open(part, mode) as out:
                    while block := resp.read(CHUNK):
                        out.write(block)
                        fetched += len(block)

            part.rename(dest)
            if _sha256(dest) == digest:
                return rel_path, fetched, "ok"
            log.warning("SHA256 mismatch on %s (attempt %d) — re-downloading", rel_path, attempt)
            dest.unlink()
        except Exception as exc:  # network errors: keep the .part and retry
            log.warning("%s attempt %d failed: %s", rel_path, attempt, exc)
            time.sleep(2 * attempt)

    return rel_path, fetched, "failed"


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    vital_dir = raw_dir / "vital_files"
    vital_dir.mkdir(parents=True, exist_ok=True)
    verified_log = vital_dir / ".verified.txt"

    log.info("=" * 60)
    log.info("VitalDB v1.0.0 — download & verify")
    log.info("=" * 60)

    log.info("[STEP 1] Metadata + checksums ...")
    ensure_metadata(raw_dir)
    expected = load_expected_hashes(raw_dir)
    caseid_to_rel = {int(Path(rel).stem): rel for rel in expected}
    log.info("  %d vital files listed in SHA256SUMS.txt", len(expected))

    log.info("[STEP 2] Case index from VitalDB track list ...")
    rows = build_case_index(sorted(caseid_to_rel))
    args.case_index.parent.mkdir(parents=True, exist_ok=True)
    write_case_index(rows, args.case_index)
    tier_counts = defaultdict(int)
    for r in rows:
        tier_counts[r["tier"]] += 1
    log.info("  tiers: %s → %s", dict(sorted(tier_counts.items())), args.case_index)

    verified = set(verified_log.read_text().split()) if verified_log.exists() else set()
    queue = [caseid_to_rel[r["caseid"]] for r in rows
             if r["tier"] <= args.max_tier and caseid_to_rel[r["caseid"]] not in verified]
    if args.limit is not None:
        queue = queue[: args.limit]
    log.info("[STEP 3] %d already verified; %d to download with %d workers",
             len(verified), len(queue), args.workers)
    if not queue:
        log.info("Nothing to do.")
        return

    lock = Lock()
    t0 = time.time()
    total_bytes = n_ok = n_fail = 0
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, rel, expected[rel], raw_dir) for rel in queue]
        for i, fut in enumerate(as_completed(futures), 1):
            rel, nbytes, status = fut.result()
            with lock:
                total_bytes += nbytes
                if status == "ok":
                    n_ok += 1
                    with open(verified_log, "a") as vf:
                        vf.write(rel + "\n")
                else:
                    n_fail += 1
                    failed.append(rel)
            if i % 25 == 0 or i == len(queue):
                elapsed = time.time() - t0
                rate = total_bytes / max(elapsed, 1e-9)
                eta_h = (len(queue) - i) * (elapsed / i) / 3600
                log.info("  %d/%d done · %.1f GB · %.1f MB/s · %d failed · ETA %.1f h",
                         i, len(queue), total_bytes / 1e9, rate / 1e6, n_fail, eta_h)

    log.info("")
    log.info("=" * 60)
    log.info("VitalDB download pass complete")
    log.info("  Verified this run : %d", n_ok)
    log.info("  Failed            : %d", n_fail)
    log.info("  Downloaded        : %.1f GB in %.1f h", total_bytes / 1e9, (time.time() - t0) / 3600)
    log.info("=" * 60)
    if failed:
        log.error("Re-run the script to retry %d failed files, e.g. %s", len(failed), failed[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()

# %% ── Cell 3: smoke test (3 files) ────────────────────────────────────────────
!python /content/drive/MyDrive/CLBench/scripts/01c_download_vitaldb.py \
    --raw-dir /content/drive/MyDrive/CLBench/data/raw/physionet.org/files/vitaldb/1.0.0 \
    --case-index /content/drive/MyDrive/CLBench/data/vitaldb_case_index.csv \
    --limit 3 --workers 3

# %% ── Cell 4: full download (re-run this cell after any disconnect) ───────────
!python /content/drive/MyDrive/CLBench/scripts/01c_download_vitaldb.py \
    --raw-dir /content/drive/MyDrive/CLBench/data/raw/physionet.org/files/vitaldb/1.0.0 \
    --case-index /content/drive/MyDrive/CLBench/data/vitaldb_case_index.csv \
    --workers 8

# %% ── Cell 5: progress check (safe to run any time, even from another notebook) ─
!wc -l < /content/drive/MyDrive/CLBench/data/raw/physionet.org/files/vitaldb/1.0.0/vital_files/.verified.txt
!du -sh /content/drive/MyDrive/CLBench/data/raw/physionet.org/files/vitaldb/1.0.0/vital_files
