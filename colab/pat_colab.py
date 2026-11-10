# CLBench — delay-aware PAT re-extraction on Colab. ONE cell: paste into a new notebook and run.
#
# Reads only SNUADC/PLETH from the 500 Hz .vital files on Drive (R-peaks come from the beat tables
# already on Drive), writes native/pat/ and native/pat_summaries/, packs native_pat.tar, writes
# PAT_RUN_REPORT.txt, flushes Drive, then disconnects the runtime by itself — even if a step fails.
# No GPU is used: pick the runtime with the MOST CPU CORES available (a plain 2-core CPU runtime is slow).
#
# Before running: upload the rebuilt `colab/clbench_code.zip` to My Drive/CLBench/.

from google.colab import drive, runtime
drive.mount('/content/drive')
import os, sys, time, json, glob, shutil, subprocess, traceback

NG = '/content/drive/MyDrive/CLBench'
CODE = '/content/clbench'
RAW = f'{NG}/data/raw/physionet.org/files/vitaldb/1.0.0'
PV = f'{NG}/data/processed_vitaldb'
NAT = f'{PV}/native'
report = []


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    report.append(line)


try:
    shutil.rmtree(CODE, ignore_errors=True)
    subprocess.run(['unzip', '-q', f'{NG}/clbench_code.zip', '-d', CODE], check=True)
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'vitaldb'], check=True)
    log(open(f'{CODE}/BUILD_INFO.txt').read().replace('\n', ' '))
    assert os.path.exists(f'{CODE}/scripts/56_pat_native.py'), 'code zip is old: rebuild and re-upload it'
    cores = os.cpu_count() or 2
    workers = 2 * cores                       # I/O-bound on Drive reads: oversubscribe the cores
    log(f'{cores} CPU cores -> {workers} workers; {len(os.listdir(f"{NAT}/beats"))} beat tables on Drive')
    env = {**os.environ, 'PYTHONPATH': f'{CODE}/src'}
    p = subprocess.Popen([sys.executable, 'scripts/56_pat_native.py', '--raw-dir', RAW, '--beats-dir', f'{NAT}/beats',
                          '--out-dir', NAT, '--cases-csv', f'{CODE}/data/pat_cases.csv', '--workers', str(workers)],
                         cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
        if 'finished' in line or 'failed' in line:
            report.append(line.rstrip())
    log(f'extraction exit code {p.wait()}')
    S = [json.load(open(f)) for f in glob.glob(f'{NAT}/pat_summaries/*.json')]
    cov = sorted(s['coverage_valid'] for s in S if s.get('coverage_valid') is not None)
    log(f'{len(S)} cases done; median coverage {cov[len(cov) // 2]:.3f}' if cov else f'{len(S)} cases done')
    r = subprocess.run(f'cd "{PV}" && tar -cf native_pat.tar.tmp native/pat native/pat_summaries '
                       f'&& mv native_pat.tar.tmp native_pat.tar', shell=True, capture_output=True, text=True)
    log(f'tar exit {r.returncode} {r.stderr[-300:]}')
    if os.path.exists(f'{PV}/native_pat.tar'):
        log(f'native_pat.tar: {os.path.getsize(f"{PV}/native_pat.tar") / 1e9:.2f} GB')
except Exception:
    log('ERROR:\n' + traceback.format_exc())
finally:
    try:
        open(f'{NG}/PAT_RUN_REPORT.txt', 'w').write('\n'.join(report) + '\n')
    except Exception as e:
        print('could not write PAT_RUN_REPORT.txt:', e)
    for attempt in range(3):
        try:
            drive.flush_and_unmount()
            print('Drive flushed: all files are saved.', flush=True)
            break
        except Exception as e:
            print('flush attempt failed:', e, flush=True)
            time.sleep(60)
    time.sleep(10)
    runtime.unassign()
