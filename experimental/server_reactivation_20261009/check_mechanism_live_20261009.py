"""One read-only measured queue snapshot; not a recurring monitor."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import time

stage = Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
phase = sys.argv[1]
assert phase in ('preflight', 'formal')
progress_path = stage / (phase + '_queue_progress.json')
progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
root = stage / ('preflight/runs' if phase == 'preflight' else 'runs')
rounds = []
for item in progress.get('active', []):
    path = root / item['id'] / 'progress.json'
    row = json.loads(path.read_text()) if path.exists() else {}
    cmd = Path('/proc') / str(item['pid']) / 'cmdline'
    rounds.append(dict(item, progress=row, process=cmd.read_bytes().replace(b'\0', b' ').decode() if cmd.exists() else 'EXITED'))
def stats():
    return dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
before = stats(); start = time.monotonic(); time.sleep(2); after = stats()
cpu = (int(after['usage_usec']) - int(before['usage_usec'])) / 1e6 / (time.monotonic() - start)
quota = Path('/sys/fs/cgroup/cpu.max').read_text().split()
service = 'guardfed_celeba_mechanism_' + ('preflight' if phase == 'preflight' else 'formal')
sv = subprocess.run(['supervisorctl', 'status', service], capture_output=True, text=True)
failures = [str(p) for p in root.rglob('failure*.json')] if root.exists() else []
errors = []
logs = stage / ('preflight/logs' if phase == 'preflight' else 'logs')
for item in progress.get('active', []):
    path = logs / (item['id'] + '.log')
    if path.exists():
        lines = path.read_text(errors='replace').splitlines()[-60:]
        errors += [{'id': item['id'], 'line': line} for line in lines if any(s in line for s in ('Traceback', 'OutOfMemoryError', 'CUDA error', 'out of memory'))]
disk = __import__('shutil').disk_usage(stage)
report = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'server': '89.22.197.55:60350', 'instance': 52183675, 'phase': phase,
          'service': sv.stdout.strip(), 'queue_completed': len(progress.get('completed', [])),
          'failed': progress.get('failed', []), 'failure_files': failures, 'active': rounds,
          'pending': progress.get('pending'), 'queue_updated_unix': progress.get('updated_unix'),
          'gpu_csv': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader'], text=True),
          'gpu_recovery': [s.strip() for s in subprocess.check_output(['nvidia-smi', '-q'], text=True).splitlines() if 'Recovery Action' in s],
          'cpu_used_cores_2sec': cpu, 'cpu_quota_cores': int(quota[0])/int(quota[1]) if quota[0] != 'max' else None,
          'memory_used_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text()),
          'memory_limit': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
          'memory_events': Path('/sys/fs/cgroup/memory.events').read_text(),
          'disk_free_bytes': disk.free, 'recent_active_log_errors': errors}
destination = Path('/workspace/guardfed_checks/server_reactivation_20261009') / ('latest_' + phase + '_live.json')
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
