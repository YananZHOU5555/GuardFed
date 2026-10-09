"""Preserve completed cu130 gates, then launch the same bounded cu128 gates."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

STAGE = Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1').resolve()
CHECKS = Path('/workspace/guardfed_checks/server_reactivation_20261009')
assert STAGE.is_relative_to(Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009'))
assert not (STAGE / 'dispatch_receipt.json').exists()
proof = json.loads((CHECKS / 'preflight_cu130_offserver_verification.json').read_text())
backup = json.loads((CHECKS / 'preflight_backup_cu130.json').read_text())
assert proof['off_server_verified'] and proof['members_verified'] == len(backup['members'])
assert proof['archive_sha256'] == backup['sha256']
observed = subprocess.run(['supervisorctl', 'status', 'guardfed_celeba_mechanism_preflight'], text=True, capture_output=True)
assert observed.returncode in (0, 3), observed
state = observed.stdout
assert 'EXITED' in state, state
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ')
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        continue
    assert not (b'/workspace/GuardFed-celeba-expanded' in cmd and (b'worker.py' in cmd or b'run_revision_ablation.py' in cmd)), proc.name
manifest = json.loads((STAGE / 'manifest.json').read_text())
history = (STAGE / 'preflight_history/cu130_20261009').resolve()
source = (STAGE / 'preflight').resolve()
assert source.parent == STAGE and history.is_relative_to(STAGE) and not history.exists()
history.parent.mkdir(exist_ok=True)
source.rename(history)
shutil.copytree(history / 'jobs', STAGE / 'preflight/jobs')
(STAGE / 'preflight_queue_progress.json').rename(history / 'preflight_queue_progress.json')
for entry in manifest['preflight_jobs'] + manifest['reference_jobs']:
    assert hashlib.sha256(Path(entry['job']).read_bytes()).hexdigest() == entry['job_sha256']
    assert not Path(entry['output']).exists()
wrapper = Path('/opt/supervisor-scripts/guardfed_celeba_mechanism_preflight.sh')
old = wrapper.read_bytes()
old_python = b'/workspace/GuardFed-celeba-expanded/.venv/bin/python'
new_python = b'/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
assert old.count(old_python) == 1
(CHECKS / 'preflight_wrapper_cu130.sh').write_bytes(old)
wrapper.write_bytes(old.replace(old_python, new_python))
subprocess.run(['supervisorctl', 'start', 'guardfed_celeba_mechanism_preflight'], check=True)
receipt = {'checked_at_unix': time.time(), 'server': '89.22.197.55:60350', 'runtime': 'torch2.11.0+cu128',
           'archived_preflight': str(history), 'original_job_bytes_unchanged': True,
           'previous20_offserver_sha256': proof['archive_sha256'], 'only_wrapper_interpreter_changed': True,
           'scientific_runs_started': 0, 'old_queue_restarted': False,
           'scope': 'same20 three-round real-image gates in the formal runtime; no method/seed/protocol change'}
(CHECKS / 'preflight_launch_cu128.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
