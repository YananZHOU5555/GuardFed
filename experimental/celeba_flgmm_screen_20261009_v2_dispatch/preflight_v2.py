"""Read-only runtime checks; extract only this new owned release if absent."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

BASE = Path(__file__).resolve().parent
RELEASE = BASE / 'release_v2'
REPO = Path('/workspace/GuardFed-celeba-expanded')
STAGE = REPO / 'results/revision_20261009/celeba_mechanism_v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cmd(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return dict(exit_code=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())


def cpu_usage():
    values = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    return int(values['usage_usec'])


def observe():
    queue = read(STAGE / 'formal_queue_progress.json')
    progress = []
    for p in STAGE.rglob('progress.json'):
        value = read(p)
        progress.append(dict(path=str(p), id=value.get('job_id', p.parent.name),
                             round=value.get('round'), mtime=p.stat().st_mtime))
    processes = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().strip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        if args and 'python' in Path(args[0]).name and int(path.parent.name) != os.getpid():
            if any('flgmm' in arg.lower() for arg in args):
                processes.append(dict(pid=int(path.parent.name), argv=args))
    start = time.monotonic(); usage = cpu_usage(); time.sleep(2)
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        formal_service=cmd(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal']),
        formal_queue=dict(completed=len(queue['completed']), active=queue['active'],
                          failed=queue['failed'], pending=queue['pending'], updated_unix=queue['updated_unix']),
        formal_progress=progress, existing_flgmm_python_processes=processes,
        gpu=cmd(['nvidia-smi', '--query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader']),
        recovery=[line.strip() for line in cmd(['nvidia-smi', '-q'])['stdout'].splitlines() if 'Recovery Action' in line],
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        cpu_effective_cores=(cpu_usage()-usage)/1e6/(time.monotonic()-start),
        memory_bytes=int(Path('/sys/fs/cgroup/memory.current').read_text()),
        memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),
        disk_free=shutil.disk_usage(REPO).free,
        intended_service=cmd(['supervisorctl', 'status', 'guardfed_celeba_flgmm_screen']))


def failure_receipt(kind, value, tb):
    import traceback
    path = BASE / 'PREFLIGHT_V2_FAILURE.json'
    if not path.exists():
        path.write_text(json.dumps(dict(error=repr(value), traceback=''.join(traceback.format_exception(kind, value, tb)), time=time.time()), indent=2))
    sys.__excepthook__(kind, value, tb)
sys.excepthook = failure_receipt
receipt = read(BASE / 'UPLOAD_V2_SHA256.json')
assert sha(BASE / 'release_v2.tar.gz') == receipt['archive_sha256']
assert not RELEASE.exists(), 'Do not overwrite any release'
with tarfile.open(BASE / 'release_v2.tar.gz') as archive:
    members = archive.getmembers()
    assert len(members) == receipt['members']
    assert len({m.name for m in members}) == len(members)
    for member in members:
        target = (BASE / member.name).resolve()
        assert member.isfile() and target.is_relative_to(RELEASE.resolve())
    archive.extractall(BASE, members=members, filter='data')
assert sha(RELEASE / 'PACKAGE_SHA256.json') == receipt['release_seal_sha256']
sys.path.insert(0, str(RELEASE))
from screen_common import local_identity, repo_identity
before = observe()
assert before['formal_service']['exit_code'] == 0 and 'RUNNING' in before['formal_service']['stdout']
assert len(before['formal_queue']['active']) == 8 and not before['formal_queue']['failed']
assert len(before['recovery']) == 2 and all(line.endswith(': None') for line in before['recovery'])
quota, period = map(int, before['cpu_quota'].split())
assert before['cpu_effective_cores'] + 2 < quota / period
assert before['memory_bytes'] + 16 * 1024**3 < int(before['memory_max'])
assert before['disk_free'] > 50 * 1024**3
assert all(int(line.split(',')[4].split()[0]) < 10000 for line in before['gpu']['stdout'].splitlines())

protocol, manifest = local_identity()
source_hashes = repo_identity(REPO, protocol)
assert not (RELEASE / 'EXECUTION_AUTHORIZATION.json').exists()
assert not (RELEASE / 'runs').exists()
assert not before['existing_flgmm_python_processes']
assert not Path('/etc/supervisor/conf.d/guardfed_celeba_flgmm_screen.conf').exists()
import torch
torch.set_num_threads(1)
environment = dict(python=sys.version, python_executable=sys.executable,
                   torch=torch.__version__, cuda=torch.version.cuda,
                   devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert sys.version_info[:3] == (3, 12, 3) and torch.__version__ == '2.11.0+cu128' and torch.version.cuda == '12.8'
assert environment['devices'] == ['NVIDIA GeForce RTX 5090'] * 2
after = observe()
assert len(after['recovery']) == 2 and all(line.endswith(': None') for line in after['recovery'])
assert after['formal_service']['exit_code'] == 0 and 'pid 9179' in after['formal_service']['stdout']
assert len(after['formal_queue']['active']) == 8 and not after['formal_queue']['failed']
assert not after['existing_flgmm_python_processes']
assert after['cpu_effective_cores'] + 2 < quota / period
mapping = read(RELEASE / 'REPO_SYMLINK_TARGETS.json')['entries']
resolved = {name: str((REPO / name).resolve()) for name in mapping}
assert all(resolved[name] == item['resolved_target'] for name, item in mapping.items())

previous = read(BASE / 'PREVIOUS_FORMAL.json')
current = {r['id']: r['round'] for r in after['formal_progress']}
growth = [dict(id=row['job_id'], previous_round=row['after_round'], current_round=current.get(row['job_id']))
          for row in previous['formal_progress']]
assert all(row['current_round'] is not None and row['current_round'] > row['previous_round'] for row in growth), growth
proof = dict(status='RELEASE_UPLOADED_HASHED_NOT_AUTHORIZED_OR_STARTED',
    package_sha256=receipt['release_seal_sha256'], archive_sha256=receipt['archive_sha256'],
    package_files=66, manifest_jobs=len(manifest['jobs']), repo_source_data_hashes=source_hashes,
    guide_sha256=sha(Path('/etc/vast-agents-guide.md')), environment=environment, exact_data_targets=resolved,
    before=before, after=after, prior_v3_postflight_comparison=growth,
    service_configuration=str(RELEASE / 'dispatch/guardfed_celeba_flgmm_screen.conf'),
    service_wrapper=str(RELEASE / 'dispatch/guardfed_celeba_flgmm_screen.sh'),
    incremental_backup_entry=str(RELEASE / 'README.md') + ': incremental off-server backup section',
    installed=False, execution_authorization_written=False, training_started=False)
(BASE / 'PREFLIGHT_V2.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(dict(status=proof['status'], seal=proof['package_sha256'],
    checked_repo_files=len(source_hashes), completed=after['formal_queue']['completed'],
    active=len(after['formal_queue']['active']), failed=len(after['formal_queue']['failed']),
    cpu_cores=after['cpu_effective_cores'], gpus=after['gpu']['stdout'], growth=growth)))
