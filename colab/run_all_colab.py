# CLBench — ONE Colab notebook for everything the GPU / Drive side needs. Runtime → Run all.
#
# What it does (all outputs go to My Drive/CLBench/; every step is skipped if already done):
#   A. VitalDB native extraction, in the BACKGROUND on the CPU cores (reads the 500 Hz .vital files):
#        data/processed_vitaldb/native/{beats,breaths,numerics_1s,summaries}/  → packed as native.tar
#   B. On the GPU, meanwhile, concept-model training:
#        seed 1337, seed 2024 : Phase 1 (MAE encoders) → Phase 2 with RULE labels   (v1 multi-seed)
#                                                       → Phase 2 with MONITOR labels (same encoders)
#        seed 42              : Phase 1 → Phase 2 with MONITOR labels (rule-label seed 42 already exists)
#      Checkpoints: clbench_ckpts/{seed_S, device_seed_S, phase1_seed_S}/  (+ train.log in each)
#      Settings are exactly those of the seed-42 run (batch 4096/1024, fp16+compile / bf16).
#
# Before running (one time):
#   1. On the laptop: `uv run python scripts/make_colab_code_zip.py`, then drag
#      `colab/clbench_code.zip` into My Drive/CLBench/ (replace the old one).
#   2. My Drive/CLBench/clbench_colab_full.zip (the seed-42 training bundle, ~10 GB) must be there.
#   3. Runtime → Change runtime type → A100 GPU + High-RAM. Then Runtime → Run all.
#
# If Colab disconnects or a cell fails: upload the latest code zip if asked, then Runtime → Restart
# session → Run all. Finished steps are skipped (DONE markers); an interrupted training step restarts
# from its beginning; extraction resumes case by case and is never started twice.
# Expect several hours in total; the extraction alone is roughly 30–60 min.
#
# When it finishes, copy to the laptop:
#   My Drive/CLBench/data/processed_vitaldb/native.tar → CLBench/data/processed_vitaldb/ (then `tar -xf native.tar`)
#   My Drive/CLBench/clbench_ckpts/{seed_1337,seed_2024,device_seed_42,device_seed_1337,device_seed_2024}
#       → CLBench/checkpoints/

# %% ── Cell 1: setup ──────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
import os, shutil, subprocess, sys, time, json, glob
NG = '/content/drive/MyDrive/CLBench'
CODE = '/content/clbench'
BUNDLE = '/content/bundle'
CKPTS = f'{NG}/clbench_ckpts'
RAW = f'{NG}/data/raw/physionet.org/files/vitaldb/1.0.0'
NATIVE = f'{NG}/data/processed_vitaldb/native'
MANIFEST = f'{NG}/data/vitaldb_manifest.csv'
os.makedirs(CKPTS, exist_ok=True)

shutil.rmtree(CODE, ignore_errors=True)
subprocess.run(['unzip', '-q', f'{NG}/clbench_code.zip', '-d', CODE], check=True)
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'vitaldb'], check=True)
print(open(f'{CODE}/BUILD_INFO.txt').read())
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'])
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cpus', os.cpu_count())
assert torch.cuda.is_available(), 'Switch the runtime to a GPU (A100) first'
assert os.path.exists(f'{CODE}/data/label_overlay/device_train.npz'), 'code zip is old: rebuild and re-upload it'
ENV = {**os.environ, 'PYTHONPATH': f'{CODE}/src'}


def run(cmd, log_path):
    """Run a command, streaming its output to the notebook and appending it to log_path."""
    print('$', ' '.join(cmd), flush=True)
    with open(log_path, 'a') as log:
        p = subprocess.Popen(cmd, cwd=CODE, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end='')
            log.write(line)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f'command failed (exit {rc}); see {log_path}')


# %% ── Cell 2: A. start the VitalDB extraction in the background (CPU) ────────────
n_vital = len(os.listdir(f'{RAW}/vital_files'))
n_done = len(os.listdir(f'{NATIVE}/summaries')) if os.path.exists(f'{NATIVE}/summaries') else 0
print(n_vital, '.vital files on Drive |', n_done, 'cases already extracted')
EXTRACT_WORKERS = max(2, (os.cpu_count() or 8) // 2)   # leave the other half for GPU data loading
os.makedirs(NATIVE, exist_ok=True)


def extraction_running():
    out = subprocess.run(['pgrep', '-f', '48_vitaldb_native_extract'], capture_output=True, text=True).stdout
    return bool(out.strip())


if extraction_running():   # e.g. still going from before a session restart: do not start a second one
    print('extraction already running in the background; not starting another')
else:
    subprocess.Popen(
        [sys.executable, 'scripts/48_vitaldb_native_extract.py', '--raw-dir', RAW, '--out-dir', NATIVE,
         '--manifest', MANIFEST, '--workers', str(EXTRACT_WORKERS)],
        cwd=CODE, env=ENV, stdout=open(f'{NATIVE}/extract.log', 'a'), stderr=subprocess.STDOUT,
        start_new_session=True)
    print('extraction started in the background with', EXTRACT_WORKERS, 'workers; log:', f'{NATIVE}/extract.log')

# %% ── Cell 3: unpack the training bundle (.npz) to local disk ────────────────────
os.makedirs(BUNDLE, exist_ok=True)
if not os.path.exists(f'{BUNDLE}/joint_train.npz'):
    subprocess.run(['unzip', '-q', '-o', f'{NG}/clbench_colab_full.zip', '*.npz', '-d', BUNDLE], check=True)
    for p in glob.glob(f'{BUNDLE}/**/*.npz', recursive=True):   # flatten if nested
        if os.path.dirname(p) != BUNDLE:
            shutil.move(p, BUNDLE)
required = [f'{m}_{s}.npz' for m in ('ECG', 'PPG', 'ABP', 'RESP') for s in ('train', 'val')]
required += ['joint_train.npz', 'joint_val.npz']
missing = [f for f in required if not os.path.exists(f'{BUNDLE}/{f}')]
assert not missing, f'missing from bundle: {missing}'
print('bundle ready')

# %% ── Cell 4: B. training (GPU) ──────────────────────────────────────────────────
COMMON = ['--bundle-dir', BUNDLE, '--phase1-epochs', '50', '--phase1-batch', '4096', '--phase1-amp', 'fp16',
          '--compile', '--phase2-epochs', '40', '--phase2-batch', '1024', '--phase2-amp', 'bf16']
OVERLAY = ['--label-overlay-dir', f'{CODE}/data/label_overlay']


def step(name, fn):
    marker = f'{CKPTS}/DONE_{name}'
    if os.path.exists(marker):
        print(f'[skip] {name} (done)')
        return
    t = time.time()
    fn()
    open(marker, 'w').write(time.strftime('%Y-%m-%d %H:%M:%S'))
    print(f'[done] {name} in {(time.time() - t) / 60:.1f} min', flush=True)


def phase1(seed):
    d = f'{CKPTS}/phase1_seed_{seed}'
    os.makedirs(d, exist_ok=True)
    run([sys.executable, 'scripts/03_train_e2e.py', *COMMON, '--ckpt-dir', d, '--seed', str(seed), '--phase', '1'],
        f'{d}/train.log')


def phase2(seed, device_labels):
    d = f'{CKPTS}/{"device_seed" if device_labels else "seed"}_{seed}'
    shutil.rmtree(f'{d}/encoders', ignore_errors=True)            # start from this seed's Phase-1 encoders
    shutil.copytree(f'{CKPTS}/phase1_seed_{seed}/encoders', f'{d}/encoders')
    run([sys.executable, 'scripts/03_train_e2e.py', *COMMON, '--ckpt-dir', d, '--seed', str(seed), '--phase', '2',
         *(OVERLAY if device_labels else [])], f'{d}/train.log')


# quick smoke test on this GPU (compile + AMP + overlay), ~6 min; each phase in its own process,
# exactly as the real runs below
SMOKE = [*COMMON, '--ckpt-dir', '/content/smoke', '--seed', '0', '--phase1-epochs', '1', '--phase2-epochs', '1',
         '--debug-steps', '5']
step('smoke_v2', lambda: (run([sys.executable, 'scripts/03_train_e2e.py', *SMOKE, '--phase', '1'], '/content/smoke.log'),
                          run([sys.executable, 'scripts/03_train_e2e.py', *SMOKE, '--phase', '2', *OVERLAY],
                              '/content/smoke.log')))
for seed in (1337, 2024):
    step(f'phase1_seed_{seed}', lambda s=seed: phase1(s))
    step(f'rule_seed_{seed}', lambda s=seed: phase2(s, device_labels=False))
    step(f'device_seed_{seed}', lambda s=seed: phase2(s, device_labels=True))
step('phase1_seed_42', lambda: phase1(42))
step('device_seed_42', lambda: phase2(42, device_labels=True))
print('training finished')

# %% ── Cell 5: wait for the extraction, check it, pack it ─────────────────────────
while extraction_running():
    n = len(os.listdir(f'{NATIVE}/summaries')) if os.path.exists(f'{NATIVE}/summaries') else 0
    print(time.strftime('%H:%M'), n, 'cases extracted so far', flush=True)
    time.sleep(120)
print('extraction finished; last log lines:')
subprocess.run(['tail', '-3', f'{NATIVE}/extract.log'])
S = [json.load(open(p)) for p in glob.glob(f'{NATIVE}/summaries/*.json')]
print(len(S), 'cases | with beats', sum(1 for s in S if s.get('beats')),
      '| with breaths', sum(1 for s in S if s.get('breaths')))
subprocess.run(f'cd {NG}/data/processed_vitaldb && tar -cf native.tar native && ls -lh native.tar', shell=True, check=True)

# %% ── Cell 6: summary ────────────────────────────────────────────────────────────
for d in sorted(glob.glob(f'{CKPTS}/*seed_*')):
    log = f'{d}/train.log'
    last = [l for l in open(log)] if os.path.exists(log) else []
    best = [l.strip() for l in last if 'done' in l and 'best_val' in l]
    print(os.path.basename(d), '|', best[-1] if best else '(no Phase summary yet)')
print(sorted(os.path.basename(p) for p in glob.glob(f'{CKPTS}/DONE_*')))
print('Copy native.tar and the seed_/device_seed_ folders to the laptop (see the header of this file).')
