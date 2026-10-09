"""Stage only the next strictly closed evidence delta; never commit or push."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'tmp/revision-publish-20260928'
TRAIN = Path('docs/server_deployment_20260923/training_20260923')
CHECKS = TRAIN / 'server_reactivation_20261009'
PREVIOUS = '34b601f4962e0357911ce36fa7a338bb36de178a'
MAPPING = {}


def git(*args, **kwargs):
    return subprocess.check_output(['git', '-c', 'core.longpaths=true', *args], cwd=REPO, **kwargs)


def safe(path):
    path = Path(path).resolve()
    return Path('\\\\?\\' + str(path)) if len(str(path)) > 235 else path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8-sig'))


def copy(source, target, expected=None):
    target = Path(target)
    destination = REPO / target
    assert destination.resolve().is_relative_to(REPO.resolve())
    digest = sha(source)
    assert expected is None or digest == expected, source
    assert safe(source).stat().st_size < 100_000_000, source
    safe(destination.parent).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(safe(source), safe(destination))
    assert sha(destination) == digest
    prior = MAPPING.setdefault(target.as_posix(), digest)
    assert prior == digest, target


def seal(source, target, name):
    if name.endswith('.json'):
        rows = read(source / name)
        rows = rows.get('files', rows.get('members', rows))
        assert isinstance(rows, dict)
        for path, item in rows.items():
            digest = item['sha256'] if isinstance(item, dict) else item
            if isinstance(item, dict) and 'bytes' in item:
                assert safe(source / path).stat().st_size == item['bytes']
            copy(source / path, target / path, digest)
    else:
        for line in safe(source / name).read_text().splitlines():
            digest, path = line.split('  ', 1)
            copy(source / path, target / path, digest)
    copy(source / name, target / name)


assert git('rev-parse', 'HEAD', text=True).strip() == PREVIOUS
assert not git('status', '--porcelain', text=True).strip()
state = read(ROOT / TRAIN / 'TRAINING_STATE.json')
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'] == 436
assert state['baseline_valid_GPU_recovery_20261009']['remaining464_dispatched']
base = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
collector = read(base / 'cumulative_436_accepted.json')
assert len(set(collector['accepted_ids'])) == collector['accepted_n'] == 436
assert collector['original_failed_CPU_prediction_still_invalid']
copy(base / 'cumulative_436_accepted.json', Path('experimental') / base.name / 'cumulative_436_accepted.json')
imports = base / 'preserved11_import'
checked = read(imports / 'ROOT_OFFSERVER_IMPORT_VERIFICATION.json')
assert checked['new_CNN_inference'] == 0 and checked['original_CPU_failure_still_invalid']
assert checked['status'] == 'ROOT_PRESERVED_IMPORT11_STRICT_AND_OFFSERVER_REPORTS_PASS'
for name, row in checked['members'].items():
    assert sha(imports / name) == row['sha256']
for path in imports.iterdir():
    assert path.is_file()
    copy(path, Path('experimental') / base.name / 'preserved11_import' / path.name)
prepared = ROOT / 'tmp/celeba_valid_gpu_remaining464_prepared_20261009'
assert sha(prepared / 'PACKAGE_SHA256.json') == 'fa5626ad0ab8be12ac501aea531d7b8ad2f2c05b1a18937dd86fb3708acc8d6b'
seal(prepared, Path('experimental') / prepared.name, 'FILES_SHA256.txt')
manifest = read(prepared / 'manifest.json')
inventory = read(ROOT / TRAIN / 'final_evaluation_prepared_20261009/model_inventory.json')
assert set(manifest['ids']) == {r['id'] for r in inventory['records']} - set(collector['accepted_ids'])
assert len(manifest['ids']) == 464 and len(manifest['chunks']) == 43
execution = ROOT / 'tmp/celeba_valid_gpu_remaining464_execution_20261009'
assert read(execution / 'ROOT_LAUNCH.json')['status'] == 'TARGETED464_SERVICE_START_OBSERVED'
for path in execution.iterdir():
    assert path.is_file()
    copy(path, Path('experimental') / execution.name / path.name)
for name in ('RUNNING.md', 'TRAINING_STATE.json', 'REBUTTAL_COMPLETION_20261009.md',
             'celeba_mechanism_v1/EXECUTION.md', 'publication_closed_increment10_verified_20261009.json'):
    copy(ROOT / TRAIN / name, TRAIN / name)
for name in ('MONITOR_HANDOFF.md', 'latest_formal_live.json', 'root_live_20261009T124116Z.json'):
    copy(ROOT / CHECKS / name, CHECKS / name)
for name in ('update_reactivation_state_20261009.py', 'update_completion_current_20261009.py',
             'import_preserved11_valid_replays_root_20261009.py',
             'deploy_remaining464_valid_recovery_root_20261009.py', 'observe_remaining464_gpu_root_20261009.py',
             'verify_publication_increment11_20261009.py', Path(__file__).name):
    copy(ROOT / 'tmp' / name, Path('experimental/server_reactivation_20261009') / name)
receipt_path = TRAIN / 'publication_closed_increment11_20261009.json'
attributes = REPO / '.gitattributes'; content = attributes.read_text()
for pattern in ('experimental/celeba_valid_gpu_remaining464_prepared_20261009/** -text',
                'experimental/celeba_valid_gpu_remaining464_execution_20261009/** -text',
                receipt_path.as_posix() + ' -text'):
    if pattern not in content: content += '\n' + pattern + '\n'
attributes.write_text(content, newline='\n')
receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_commit=PREVIOUS,
               copied_sha256=MAPPING.copy(), baseline_actual_valid_replays_accepted=436,
               preserved11_registered=True, preserved11_new_CNN_inference=0,
               original_CPU_failure_still_invalid=True, old424_425_collectors_unchanged=True,
               remaining464_started=True, queue_worker_cpu105=True, coordinator_cpu106=True,
               remote_queue_closure_not_offserver_acceptance=True,
               mechanism_scientific_records_offserver_verified=40,
               goal_complete=False, final_protocol_frozen=False, test_started=False)
with (ROOT / receipt_path).open('x', encoding='utf8', newline='\n') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
copy(ROOT / receipt_path, receipt_path)
names = [*MAPPING, '.gitattributes']
for start in range(0, len(names), 25):
    git('add', '-f', '--', *names[start:start + 25]); git('add', '--renormalize', '--', *names[start:start + 25])
changed = git('diff', '--cached', '--name-only', '-z').decode().split('\0')[:-1]
assert set(changed) <= set(names)
payload = git('cat-file', '--batch', input=''.join(':' + n + '\n' for n in MAPPING).encode())
position = 0
for name, digest in MAPPING.items():
    end = payload.index(b'\n', position); header = payload[position:end].split()
    assert header[1] == b'blob'; size = int(header[2])
    assert hashlib.sha256(payload[end + 1:end + 1 + size]).hexdigest() == digest, name
    position = end + size + 2
assert position == len(payload)
print(json.dumps(dict(status='SCOPED_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH', changed_files=len(changed),
                      byte_verified_files=len(MAPPING), cumulative436=True, remaining464_started=True)))
