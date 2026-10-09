"""One bounded live observation; no restart, training change or supervisor loop."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
RELEASE = BASE / 'release_v2'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
queue = read(RELEASE / 'queue_progress.json')
active = []
for item in queue['active']:
    out = RELEASE / 'runs' / item['id']
    progress = read(out / 'progress.json') if (out / 'progress.json').exists() else None
    provenance = read(out / 'provenance.json') if (out / 'provenance.json').exists() else None
    argv = Path('/proc', str(item['pid']), 'cmdline').read_bytes().decode().strip('\0').split('\0')
    active.append(dict(**item, progress=progress, environment=provenance['environment'] if provenance else None,
        nice=os.getpriority(os.PRIO_PROCESS, item['pid']), argv=argv,
        job_sha256=provenance['job_sha256'] if provenance else None))
failures = [str(p) for p in RELEASE.glob('QUEUE_FAILURE*.json')]
failures.extend(str(p) for p in (RELEASE / 'runs').rglob('failure*.json'))
formal_stage=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
formal = read(formal_stage / 'formal_queue_progress.json')
def command(args): return subprocess.check_output(args, text=True).strip()
proof = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='RUNNING_NOT_ACCEPTED',
    service=command(['supervisorctl','status','guardfed_celeba_flgmm_screen']),
    package_sha256=sha(RELEASE/'PACKAGE_SHA256.json'),
    pending=queue['pending'], completed=queue['completed'], active=active, failures=failures,
    formal_service=command(['supervisorctl','status','guardfed_celeba_mechanism_formal']),
    formal_queue=dict(completed=len(formal['completed']),active=formal['active'],failed=formal['failed']),
    gpu=command(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader']),
    recovery=[line.strip() for line in command(['nvidia-smi','-q']).splitlines() if 'Recovery Action' in line],
    memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),
    receipt_hashes={str(p.relative_to(BASE)):sha(p) for p in [BASE/'PREFLIGHT_V2.json',BASE/'START_V2_RECEIPT.json',RELEASE/'EXECUTION_AUTHORIZATION.json']})
assert len(active)==2 and not failures
assert {i['gpu'] for i in active} == {0,1}
assert all(i['nice']==10 and i['environment']['cpu_threads']==1 for i in active)
assert all(i['progress'] and i['progress']['round'] >= 1 for i in active)
assert 'RUNNING' in proof['service'] and 'pid 18899' in proof['service']
assert 'RUNNING' in proof['formal_service'] and 'pid 9179' in proof['formal_service']
assert len(formal['active'])==8 and not formal['failed']
(BASE/'FIRST_PROGRESS_V2.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(utc=proof['utc'],service=proof['service'],
    active=[dict(id=i['id'],pid=i['pid'],gpu=i['gpu'],nice=i['nice'],threads=i['environment']['cpu_threads'],round=i['progress']['round']) for i in active],
    failures=failures,receipt_hashes=proof['receipt_hashes'])))
