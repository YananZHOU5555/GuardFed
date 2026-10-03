import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

repo = Path('/workspace/GuardFed-celeba-expanded')
stage = repo / 'results/revision_20261003/celeba_baseline_fullcoverage_v1'
previous = datetime.fromisoformat('2026-10-02T00:45:31.557601+00:00').timestamp()
def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None

services = command(['supervisorctl', 'status'])
guard_services = [x for x in services['stdout'].splitlines() if x.startswith('guardfed')]
workers = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid():
        continue
    try:
        cmd = (p / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace').strip()
        cwd = os.readlink(p / 'cwd')
        if (str(repo) in cmd or cwd.startswith(str(repo))) and re.search(r'worker|run_fedaa|run_fullcoverage\.py|run_screen\.py|run_celeba|runner|launch', cmd):
            workers.append({'pid': int(p.name), 'command': cmd[:600], 'cwd': cwd})
    except OSError:
        pass
manifest_path = stage / 'manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
assert hashlib.sha256(manifest_bytes).hexdigest() == '0751a4f7c15cec43bdfcfb1df428683fda895dd4a3d495b30dd2f6ef0088063c'
records = []
failures = []
recent_errors = []
for j in manifest['jobs']:
    out = Path(j['output'])
    result = out / 'result.json'
    progress = out / 'progress.json'
    round_value = None
    if progress.exists():
        d = json.loads(progress.read_text())
        round_value = d.get('round', d.get('completed_rounds', d.get('round_completed')))
    records.append({'id': j['id'], 'result_exists': result.exists(), 'round': round_value, 'progress_mtime': progress.stat().st_mtime if progress.exists() else None})
for p in stage.rglob('*'):
    if not p.is_file() or 'failed_attempts' in p.parts:
        continue
    if p.suffix == '.json' and 'failure' in p.stem:
        failures.append({'path': str(p.relative_to(repo)), 'mtime': p.stat().st_mtime})
    if p.suffix == '.log' and p.stat().st_mtime > previous:
        tail = p.read_text(errors='replace')[-16000:]
        hits = [x[:350] for x in tail.splitlines() if re.search(r'Traceback|CUDA.*error|out of memory|OOM|RuntimeError', x, re.I)]
        if hits:
            recent_errors.append({'path': str(p.relative_to(repo)), 'hits': hits[-5:]})
queue_files = {}
for p in stage.glob('queue*.json'):
    queue_files[p.name] = {'mtime': p.stat().st_mtime, 'data': json.loads(p.read_text())}
gpu = command(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'])
gpu_rows = list(csv.reader(io.StringIO(gpu['stdout']))) if gpu['exit_code'] == 0 else []
gpu_detail = command(['nvidia-smi', '-q'])
recovery = [x.strip() for x in gpu_detail['stdout'].splitlines() if 'Recovery Action' in x]
cg = Path('/sys/fs/cgroup')
quota_path = next((p for p in cg.glob('cpu*/cpu.cfs_quota_us') if p.is_file()), None)
period_path = quota_path.with_name('cpu.cfs_period_us') if quota_path else None
usage_path = next((p for p in cg.glob('cpu*/cpuacct.usage') if p.is_file()), None)
memory_path = cg / 'memory'
start = time.monotonic()
usage0 = int(read(usage_path)) if usage_path else None
time.sleep(2)
used_cores = (int(read(usage_path)) - usage0) / 1e9 / (time.monotonic() - start) if usage_path else None
quota = int(read(quota_path)) / int(read(period_path)) if quota_path and int(read(quota_path)) > 0 else None
memory = {name: read(memory_path / name) for name in ['memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt']}
disk = shutil.disk_usage(repo)
print(json.dumps({
    'checked_utc': datetime.now(timezone.utc).isoformat(), 'server': '213.224.31.105:26712',
    'ssh_connected': True, 'repo_exists': repo.is_dir(), 'service_status': guard_services,
    'service_query_exit': services['exit_code'], 'workers': workers, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'latest_stage': 'celeba_baseline_fullcoverage_v1', 'planned': len(records),
    'result_files': sum(r['result_exists'] for r in records), 'records': records, 'failures': failures,
    'queue_files': queue_files, 'recent_log_errors': recent_errors,
    'gpu': gpu_rows, 'gpu_exit': gpu['exit_code'], 'gpu_recovery': recovery,
    'cpu': {'cgroup': read(Path('/proc/self/cgroup')), 'quota_file': str(quota_path), 'usage_file': str(usage_path), 'quota_cores': quota, 'used_cores_2sec': used_cores},
    'memory': memory, 'disk_free_bytes': disk.free,
    'new_training_started': True, 'strict_reacceptance_rerun': False,
}, ensure_ascii=False))
