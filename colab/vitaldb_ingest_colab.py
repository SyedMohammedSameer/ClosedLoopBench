# VitalDB ingest on Colab: .vital files on Drive → CLBench windows, numerics, interventions.
# A CPU runtime works; High-RAM helps (each worker holds one case at 500 Hz, ~1–2 GB).
#
# Before running: upload `colab/clbench_code.zip` from the repo to
# My Drive/CLBench/ (drag it into the Drive folder in Finder).
#
# Outputs on Drive:
#   CLBench/data/processed_vitaldb/{patient_id}.parquet           signal windows (same schema as MIMIC)
#   CLBench/data/processed_vitaldb/numerics/{patient_id}.parquet  monitor numbers + treatment state
#   CLBench/data/processed_vitaldb/events/{patient_id}.parquet    timestamped interventions
#   CLBench/data/vitaldb_manifest.csv                             cohort manifest
#
# Resumable: after a disconnect, re-run Cells 1, 2 and 4. Finished cases are skipped.
# Storage: ≈ 83 GB of window parquet for all 3,442 four-modality cases at the default 15 s stride.
# Add `--stride-sec 30` to Cell 4 to roughly halve that.

# %% ── Cell 1: mount Drive, unpack code to local disk ──────────────────────────
from google.colab import drive
drive.mount('/content/drive')
NG = '/content/drive/MyDrive/CLBench'
!rm -rf /content/clbench && mkdir -p /content/clbench
!unzip -q $NG/clbench_code.zip -d /content/clbench
!pip -q install vitaldb
!ls /content/clbench/scripts

# %% ── Cell 2: paths ───────────────────────────────────────────────────────────
RAW = f'{NG}/data/raw/physionet.org/files/vitaldb/1.0.0'
OUT = f'{NG}/data/processed_vitaldb'
INDEX = f'{NG}/data/vitaldb_case_index.csv'
MANIFEST = f'{NG}/data/vitaldb_manifest.csv'
import os, shutil
if not os.path.exists(INDEX):  # the download run writes this; fall back to the bundled copy
    shutil.copy('/content/clbench/data/vitaldb_case_index.csv', INDEX)
W = os.cpu_count() or 4  # ingest is CPU + Drive I/O bound; a GPU does not help
print(len(os.listdir(f'{RAW}/vital_files')), 'files in vital_files/ ·', W, 'workers')
print(len(os.listdir(f'{OUT}/summaries')) if os.path.exists(f'{OUT}/summaries') else 0, 'cases already ingested')

# %% ── Cell 3: smoke test (3 cases) ────────────────────────────────────────────
!cd /content/clbench && python scripts/14_vitaldb_ingest.py \
    --raw-dir "$RAW" --out-dir "$OUT" --case-index "$INDEX" --manifest "$MANIFEST" \
    --limit 3 --workers 3

# %% ── Cell 4: full ingest, tiers 0–2 (re-run after any disconnect) ─────────────
# Finished cases are skipped, so a disconnect costs nothing: re-run Cells 1, 2, 4.
!cd /content/clbench && python scripts/14_vitaldb_ingest.py \
    --raw-dir "$RAW" --out-dir "$OUT" --case-index "$INDEX" --manifest "$MANIFEST" \
    --workers $W

# %% ── Cell 4b (optional): split across two runtimes to halve wall-clock ─────────
# Run this INSTEAD of Cell 4, with HALF = 0 in one Colab session and HALF = 1 in another.
# Each session takes alternate cases, and neither repeats the other's work.
HALF = 0
import pandas as pd
ids = pd.read_csv(INDEX).query('tier <= 2')['caseid'].tolist()[HALF::2]
open('/content/half.txt', 'w').write(' '.join(map(str, ids)))
print(len(ids), 'cases for this session')
!cd /content/clbench && python scripts/14_vitaldb_ingest.py \
    --raw-dir "$RAW" --out-dir "$OUT" --case-index "$INDEX" --manifest "$MANIFEST" \
    --workers $W --caseids $(cat /content/half.txt)

# %% ── Cell 5: progress check ──────────────────────────────────────────────────
!ls $OUT/summaries | wc -l
!du -sh $OUT

# %% ── Cell 6: verify the finished cohort ──────────────────────────────────────
import pandas as pd
m = pd.read_csv(MANIFEST)
print(f"{len(m)} cases · {m.duration_h.sum():,.0f} recorded hours · "
      f"{m.death_inhosp.fillna(0).sum():.0f} in-hospital deaths")
print("quality windows:", {c.split('_')[-1]: int(m[c].sum())
                           for c in m.columns if c.startswith('n_windows_ok_')})
print("interventions  :", {c.replace('n_events_', ''): int(m[c].sum())
                           for c in m.columns if c.startswith('n_events_')})
print("cases with a vasoactive pump:", int(m.vasoactive_pumps.notna().sum()),
      "· with TCI:", int(m.tci_pumps.notna().sum()))
print("ventilated fraction of windows: median",
      round(float(m.controlled_ventilation_frac.median()), 3))
empty = m[m.n_windows_ok_ECG == 0]
print("cases with no usable ECG windows:", len(empty), list(empty.patient_id[:5]))
print("ECG leads:", m.ecg_leads.value_counts().to_dict())
