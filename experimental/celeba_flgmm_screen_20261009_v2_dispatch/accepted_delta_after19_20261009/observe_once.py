"""One stdlib-only read-only snapshot; no scientific acceptance or remote writes."""
from pathlib import Path
from collections import deque
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

R = Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2')
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path('/etc/vast-agents-guide.md')) == '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(R / 'PACKAGE_SHA256.json') == 'aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
manifest = read(R / 'jobs/manifest.json'); queue = read(R / 'queue_progress.json')
assert len(manifest['jobs']) == 32
utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
active_ids = {r['id'] for r in queue['active']}
rows = []
for item in manifest['jobs']:
    out = R / 'runs' / item['id']; progress = read(out / 'progress.json') if (out / 'progress.json').exists() else None
    rows.append({'id': item['id'], 'active': item['id'] in active_ids, 'progress': progress, 'result_exists': (out / 'result.json').is_file(), 'acceptance_exists': (out / 'acceptance.json').is_file(), 'screen_identity_exists': (out / 'screen_identity.json').is_file()})
errors = []
pattern = re.compile(r'Traceback|RuntimeError|CUDA error|out of memory|OutOfMemory|\bnan\b|\binf\b|fatal', re.I)
for log in (R / 'logs').glob('*.log'):
    with log.open(errors='replace') as stream:
        errors.extend({'log': log.name, 'line': line.strip()[:1000]} for line in deque(stream, maxlen=100) if pattern.search(line))
failures = [str(p) for p in R.glob('*FAILURE*.json')]
for item in manifest['jobs']:
    failures.extend(str(p) for p in (R / 'runs' / item['id']).glob('*failure*.json'))
def command(argv):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    return {'returncode': r.returncode, 'stdout': r.stdout.strip(), 'stderr': r.stderr.strip()}
def cpu_stat(): return dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
before = cpu_stat(); start = time.monotonic(); time.sleep(4); after = cpu_stat(); elapsed = time.monotonic() - start
restricted106 = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if argv and 'python' in Path(argv[0]).name:
            cpus = os.sched_getaffinity(int(proc.name))
            if len(cpus) <= 16 and 106 in cpus: restricted106.append({'pid': int(proc.name), 'argv': argv, 'affinity': sorted(cpus)})
    except (OSError, ProcessLookupError): pass
source = {n: sha(R / n) for n in ('PACKAGE_SHA256.json', 'source/protocol.json', 'jobs/manifest.json', 'frozen_score.py', 'screen_common.py', 'source/accept_result.py', 'source/worker.py', 'source/flgmm_adapter.py')}
helper_paths = [R.parent / 'backup_first_two_v2.py', R.parent / 'backups/increment_20261009T1133Z/collect_delta.py']
result = {'status': 'READONLY_SNAPSHOT_NOT_NEW_SCIENTIFIC_ACCEPTANCE', 'utc': utc, 'rows': rows, 'queue': queue, 'source_sha256': source, 'service': command(['supervisorctl', 'status', 'guardfed_celeba_flgmm_screen']), 'sglang': command(['supervisorctl', 'status', 'sglang']), 'formal_service': command(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal']), 'failure_paths': failures, 'recent_log_error_matches': errors, 'CPU_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(), 'effective_CPU_cores_sample4s': (int(after['usage_usec']) - int(before['usage_usec'])) / 1e6 / elapsed, 'cpu_throttled_usec_delta': int(after.get('throttled_usec', 0)) - int(before.get('throttled_usec', 0)), 'RAM_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text()), 'RAM_limit': Path('/sys/fs/cgroup/memory.max').read_text().strip(), 'memory_events': Path('/sys/fs/cgroup/memory.events').read_text(), 'disk_free_bytes': shutil.disk_usage(R).free, 'GPU': command(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader']), 'restricted_CPU106_Python': restricted106, 'existing_backup_helpers': [{'path': str(p), 'exists': p.is_file(), 'sha256': sha(p) if p.is_file() else None} for p in helper_paths], 'no_training_or_inference': True, 'no_remote_writes': True, 'candidate_selection_performed': False, 'final_test': False}
print('SNAPSHOT_JSON=' + json.dumps(result))
