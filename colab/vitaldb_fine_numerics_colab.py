# Export 2-second monitor numerics for a subset of VitalDB cases (plan D5, fine-resolution test).
#
# Why: the 30 s windows average away the fast loops — the baroreflex settles in 1–5 s and
# respiratory sinus arrhythmia acts within a breath. At 2 s those show up as lagged effects with
# thousands of transitions per patient, instead of the handful of treatment changes a 30 s fit has
# to lean on. Output is small (~50 MB for 500 cases), so it rsyncs to the laptop easily.
#
# Re-upload `colab/clbench_code.zip` to My Drive/CLBench/ first — it now
# includes the model and the exporter. A CPU runtime is enough.

# %% ── Cell 1: mount Drive, unpack the refreshed code ───────────────────────────
from google.colab import drive
drive.mount('/content/drive')
NG = '/content/drive/MyDrive/CLBench'
!rm -rf /content/clbench && mkdir -p /content/clbench
!unzip -q $NG/clbench_code.zip -d /content/clbench
!pip -q install vitaldb
import os
W = os.cpu_count() or 4
print(W, 'workers')

# %% ── Cell 2: export (vasopressor cases first, then the longest recordings) ────
!cd /content/clbench && python scripts/18_vitaldb_fine_numerics.py \
    --raw-dir "$NG/data/raw/physionet.org/files/vitaldb/1.0.0" \
    --out-dir "$NG/data/processed_vitaldb/fine_2s" \
    --manifest "$NG/data/vitaldb_manifest.csv" \
    --step-sec 2 --limit 500 --workers $W

# %% ── Cell 3: check ────────────────────────────────────────────────────────────
!ls $NG/data/processed_vitaldb/fine_2s | wc -l
!du -sh $NG/data/processed_vitaldb/fine_2s
