"""
MIMIC-IV-WDB download, SHA256 verification, and record metadata extraction.

Dataset: MIMIC-IV Waveform Database v0.1.0
URL: https://physionet.org/content/mimic4wdb/0.1.0/

Actual file structure (confirmed from data):
    waves/
      p{prefix}/              e.g. p100
        p{subject_id}/        e.g. p10014354
          RECORDS             contains: "81739927/81739927"
          {record_id}/        e.g. 81739927/
            {record_id}.hea   top-level multi-segment header
                              has comment lines: # subject_id 10014354
                                                 # hadm_id 29600294
            {record_id}_NNNN.hea  per-segment headers (with sig_name, fs)
            {record_id}_NNNN*.dat
            {record_id}n.csv.gz   numerics table

Top-level RECORDS file: lists patient directories, e.g. "waves/p100/p10014354/"
Per-patient RECORDS:    lists record paths relative to patient dir, e.g. "81739927/81739927"
Full wfdb record path:  "waves/p100/p10014354/81739927/81739927"
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import wfdb

log = logging.getLogger(__name__)

# Canonical modality names → signal label substrings (case-insensitive)
MODALITY_PATTERNS: dict[str, list[str]] = {
    "ECG": ["ii", "iii", "avr", "avl", "avf", " v", "ecg", "ekg"],
    "PPG": ["pleth", "ppg", "spo2"],
    "ABP": ["abp", "art", "arterial"],
    "RESP": ["resp", "co2", "etco2"],
}


@dataclass
class RecordMeta:
    """Metadata for one patient waveform record."""

    record_path: str  # full relative path, e.g. "waves/p100/p10014354/81739927/81739927"
    subject_id: int  # from path, e.g. 10014354
    record_id: str  # e.g. "81739927"
    hadm_id: int = -1  # from .hea comment — no MIMIC-IV join needed
    modalities: list[str] = field(default_factory=list)
    signal_labels: list[str] = field(default_factory=list)
    sampling_rates: dict[str, int] = field(default_factory=dict)
    duration_hr: float = 0.0
    n_segments: int = 0
    usable: bool = True
    notes: str = ""


# ─── SHA256 verification ───────────────────────────────────────────────────────


def verify_sha256(raw_dir: Path) -> tuple[int, int]:
    """
    Verify files against SHA256SUMS.txt in raw_dir.
    Returns (n_ok, n_fail). Skips files not yet downloaded.
    """
    sums_file = raw_dir / "SHA256SUMS.txt"
    if not sums_file.exists():
        raise FileNotFoundError(f"SHA256SUMS.txt not found at {sums_file}")

    ok = fail = skipped = 0
    with open(sums_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            expected, rel_path = parts
            fpath = raw_dir / rel_path.lstrip("./")
            if not fpath.exists():
                skipped += 1
                continue
            actual = _sha256(fpath)
            if actual == expected:
                ok += 1
            else:
                log.error("SHA256 MISMATCH: %s", rel_path)
                fail += 1

    log.info("SHA256: %d ok / %d failed / %d not yet downloaded", ok, fail, skipped)
    return ok, fail


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ─── RECORDS parsing (two-level) ──────────────────────────────────────────────


def parse_records_file(raw_dir: Path) -> list[str]:
    """
    Build the full list of wfdb record paths by reading the two-level RECORDS structure.

    Top-level RECORDS:  patient dirs, e.g. "waves/p100/p10014354/"
    Per-patient RECORDS: record paths relative to patient dir, e.g. "81739927/81739927"

    Returns full paths like: "waves/p100/p10014354/81739927/81739927"
    """
    top_records = raw_dir / "RECORDS"
    if not top_records.exists():
        raise FileNotFoundError(f"RECORDS not found at {top_records}")

    patient_dirs = [
        line.strip().rstrip("/") for line in top_records.read_text().splitlines() if line.strip()
    ]

    full_record_paths: list[str] = []
    missing_sub_records = 0

    for patient_dir in patient_dirs:
        sub_records_file = raw_dir / patient_dir / "RECORDS"
        if not sub_records_file.exists():
            log.warning("No RECORDS in %s — skipping", patient_dir)
            missing_sub_records += 1
            continue
        for line in sub_records_file.read_text().splitlines():
            rec = line.strip()
            if rec:
                full_record_paths.append(f"{patient_dir}/{rec}")

    if missing_sub_records:
        log.warning("%d patient dirs had no RECORDS file", missing_sub_records)

    log.info(
        "Parsed %d records from %d patient dirs",
        len(full_record_paths),
        len(patient_dirs),
    )
    return full_record_paths


def extract_subject_id(record_path: str) -> int:
    """'waves/p100/p10014354/...' → 10014354
    Matches the patient directory (7+ digits), not the prefix dir (3 digits).
    """
    m = re.search(r"/p(\d{7,})/", record_path)
    if not m:
        raise ValueError(f"Cannot parse subject_id from: {record_path}")
    return int(m.group(1))


def extract_record_id(record_path: str) -> str:
    """'waves/p100/p10014354/81739927/81739927' → '81739927'"""
    return Path(record_path).name


# ─── Modality detection ────────────────────────────────────────────────────────


def detect_modalities(signal_names: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """
    Map raw signal labels → canonical modality names.
    Returns (present_modalities, {modality: [matched_labels]}).
    """
    present: list[str] = []
    mapping: dict[str, list[str]] = {}
    lower = [s.lower().strip() for s in signal_names]

    for modality, patterns in MODALITY_PATTERNS.items():
        matched = [
            signal_names[i] for i, lbl in enumerate(lower) if any(pat in lbl for pat in patterns)
        ]
        if matched:
            present.append(modality)
            mapping[modality] = matched

    return present, mapping


# ─── Per-record metadata extraction ────────────────────────────────────────────


def _parse_hea_comments(hea_path: Path) -> dict[str, int]:
    """
    Extract subject_id and hadm_id from .hea comment lines.
    e.g.  # subject_id 10014354
          # hadm_id 29600294
    """
    ids: dict[str, int] = {}
    try:
        for line in hea_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#"):
                m = re.match(r"#\s*(subject_id|hadm_id)\s+(\d+)", line)
                if m:
                    ids[m.group(1)] = int(m.group(2))
    except Exception:
        pass
    return ids


def read_record_meta(record_path: str, raw_dir: Path) -> RecordMeta:
    """
    Read one wfdb record header and extract all metadata.

    record_path: full relative path, e.g. "waves/p100/p10014354/81739927/81739927"
    raw_dir:     absolute path to dataset root (contains RECORDS, SHA256SUMS.txt, waves/)
    """
    subject_id = extract_subject_id(record_path)
    record_id = extract_record_id(record_path)
    meta = RecordMeta(
        record_path=record_path,
        subject_id=subject_id,
        record_id=record_id,
    )

    # Extract hadm_id from .hea comments (avoids MIMIC-IV join for basic linkage)
    hea_path = raw_dir / f"{record_path}.hea"
    if hea_path.exists():
        ids = _parse_hea_comments(hea_path)
        meta.hadm_id = ids.get("hadm_id", -1)
        # subject_id from path should match — warn if not
        hea_subject = ids.get("subject_id", subject_id)
        if hea_subject != subject_id:
            log.warning(
                "subject_id mismatch: path=%d hea=%d for %s",
                subject_id,
                hea_subject,
                record_path,
            )

    # Read wfdb header for signal metadata
    try:
        header_path = str(raw_dir / record_path)
        record = wfdb.rdheader(header_path, rd_segments=False)

        signal_names: list[str] = []
        fs_values: list[int] = []

        # Multi-segment layout (standard for MIMIC-IV-WDB)
        if hasattr(record, "seg_name") and record.seg_name:
            meta.n_segments = len([s for s in record.seg_name if s and s != "~"])
            seg_dir = raw_dir / Path(record_path).parent

            for seg_name in record.seg_name:
                if not seg_name or seg_name == "~":
                    continue
                seg_hea = seg_dir / f"{seg_name}.hea"
                if not seg_hea.exists():
                    continue
                try:
                    seg_hdr = wfdb.rdheader(str(seg_dir / seg_name))
                    if getattr(seg_hdr, "sig_name", None):
                        signal_names.extend(seg_hdr.sig_name)
                    if getattr(seg_hdr, "fs", None):
                        fs_values.append(int(seg_hdr.fs))
                except Exception:
                    pass

            # Duration from the multi-segment layout
            # record.sig_len = total samples; record.fs = base fs
            if getattr(record, "sig_len", None) and getattr(record, "fs", None):
                try:
                    base_fs = float(str(record.fs).split("/")[0])
                    meta.duration_hr = record.sig_len / base_fs / 3600
                except Exception:
                    pass
        else:
            meta.n_segments = 1
            signal_names = list(getattr(record, "sig_name", None) or [])
            if getattr(record, "fs", None):
                try:
                    fs_values = [int(float(str(record.fs).split("/")[0]))]
                except Exception:
                    pass
            if getattr(record, "sig_len", None) and fs_values:
                meta.duration_hr = record.sig_len / fs_values[0] / 3600

        # Deduplicate signal names
        seen: set[str] = set()
        unique: list[str] = []
        for s in signal_names:
            if s and s not in seen:
                seen.add(s)
                unique.append(s)
        meta.signal_labels = unique

        modalities, mod_map = detect_modalities(unique)
        meta.modalities = modalities

        # Per-modality sampling rate: use highest fs seen across segments
        for mod in mod_map:
            meta.sampling_rates[mod] = max(fs_values) if fs_values else 0

    except Exception as exc:
        meta.usable = False
        meta.notes = str(exc)
        log.warning("Failed to read header for %s: %s", record_path, exc)

    return meta
