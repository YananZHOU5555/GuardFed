"""Root-approved one-time deployment of the sealed remaining-seven outer lifecycle."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / 'remaining_seven_prepared_20261009'
APPROVED_SHA = '270721872c1a8a021d72f114c3300addcaceab4aa9435c4307e5471994101a05'
SEAL_SHA = '119418ce0ba509c364ad7f8c82361abd2d806d018ace847f6cd46714ea6b431d'
SERVICE = 'guardfed_celeba_mechanism_valid_remaining7'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, value):
    with (HERE / name).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')

assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
assert digest('/etc/vast-agents-guide.md') == '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert digest(STAGE / 'FILES_SHA256.json') == SEAL_SHA
assert digest(HERE / 'APPROVED.json') == APPROVED_SHA
assert not (HERE / 'preflight.json').exists() and not (HERE / 'start_receipt.json').exists()
sys.path.insert(0, str(STAGE))
import batch
scope, approval = batch.approved(HERE / 'APPROVED.json', APPROVED_SHA)
assert all(not (STAGE / name).exists() for name in ('runs', 'approvals', 'logs', 'batch_resource_before.json', 'batch_failure.json', 'batch.lock', 'compute_worker.lock'))
for output in approval['outputs'].values():
    batch.fresh_output(output)
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        argv = [s.decode(errors='replace') for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
        assert str(STAGE / 'batch.py') not in argv, ('Duplicate remaining-seven process', proc.name, argv)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
from resource_extra import resource_snapshot
resources = resource_snapshot()
assert resources['total_effective_nominal'] <= resources['actual_quota_cores']
assert resources['worst_case_declared_coexistence']['total'] == 114
for source in ('service.sh', 'supervisor.conf'):
    assert digest(STAGE / source) == batch.read(STAGE / 'FILES_SHA256.json')['files'][source]
script = Path('/opt/supervisor-scripts') / (SERVICE + '.sh')
config = Path('/etc/supervisor/conf.d') / (SERVICE + '.conf')
assert not script.exists() and not config.exists(), 'Existing service must not be overwritten'
assert not (STAGE / 'APPROVED.json').exists() and not (STAGE / 'APPROVED.sha256').exists()
save('preflight.json', dict(status='ROOT_APPROVED_SEALED_SEVEN_VALID_REPLAY_DEPLOYMENT_PREFLIGHT',
    utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(), approval_sha256=APPROVED_SHA,
    seal_sha256=SEAL_SHA, source_members_verified=12, original_bridge_members_verified=13,
    empty_outputs=True, no_duplicate_worker=True, resources=resources, new_training=0,
    new_Full_inference=0, new_test_inference=0, installer_sha256=digest(__file__)))
(STAGE / 'APPROVED.json').write_bytes((HERE / 'APPROVED.json').read_bytes())
(STAGE / 'APPROVED.sha256').write_bytes((APPROVED_SHA + '\n').encode('ascii'))
script.write_bytes((STAGE / 'service.sh').read_bytes()); script.chmod(0o755)
config.write_bytes((STAGE / 'supervisor.conf').read_bytes())
assert digest(script) == digest(STAGE / 'service.sh') and digest(config) == digest(STAGE / 'supervisor.conf')
steps = []
for command in (['supervisorctl', 'reread'], ['supervisorctl', 'add', SERVICE], ['supervisorctl', 'start', SERVICE]):
    result = subprocess.run(command, capture_output=True, text=True)
    steps.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        save('installation_failure.json', dict(steps=steps, automatic_retry=False)); raise RuntimeError(steps[-1])
status = subprocess.run(['supervisorctl', 'status', SERVICE], capture_output=True, text=True).stdout.strip()
save('start_receipt.json', dict(utc=datetime.now(timezone.utc).isoformat(), status=status, steps=steps,
    script_sha256=digest(script), config_sha256=digest(config), approval_sha256=APPROVED_SHA,
    seal_sha256=SEAL_SHA, max_new_ids=7, compute_threads=8, cpu_allocation=list(range(112,120)),
    autostart=False, autorestart=False, no_automatic_retry=True, nominal_reservation_not_actual_cpu_usage=True))
print(json.dumps(dict(status=status, observed_nominal_threads=resources['total_effective_nominal'],
    quota=resources['actual_quota_cores'], worst_case_reservation=114)), flush=True)
