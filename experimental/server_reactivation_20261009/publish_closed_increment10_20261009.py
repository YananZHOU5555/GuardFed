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
PREVIOUS = 'ff61b603f32a22981b8b179afc9026c61c7c9d66'
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
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'] == 425
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified'] == 40
execution = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
assert read(execution / 'cumulative_425_accepted.json')['accepted_n'] == 425
verified = read(execution / 'first1_backup/ROOT_OFFSERVER_VERIFICATION.json')
assert verified['archive_members_verified'] == 73 and verified['native_max_abs_difference'] == 0
assert verified['old424_unchanged'] and verified['constant_native_prediction_retained']
impl = ROOT / 'tmp/celeba_valid_gpu_recovery_implementation_20261009'
assert sha(impl / 'PACKAGE_SHA256.json') == '6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56'
seal(impl, Path('experimental') / impl.name, 'FILES_SHA256.txt')
target = Path('experimental') / execution.name
for path in execution.iterdir():
    if path.is_file():
        copy(path, target / path.name)
for name in ('chunk_evidence.tar.gz', 'remote_archive_inventory.json', 'ROOT_OFFSERVER_VERIFICATION.json'):
    copy(execution / 'first1_backup' / name, target / 'first1_backup' / name)
population = ROOT / 'tmp/celeba_logofair_population_proposal_20261009'
assert sha(population / 'FILES_SHA256.json') == 'f90f8adf938ba65d0ed5ce3205d8ec731da69d87b72e904254bc1b3d551eb28b'
seal(population, Path('experimental') / population.name, 'FILES_SHA256.json')
for name in ('RUNNING.md', 'TRAINING_STATE.json', 'REBUTTAL_COMPLETION_20261009.md',
             'celeba_mechanism_v1/EXECUTION.md', 'publication_closed_increment9_verified_20261009.json'):
    copy(ROOT / TRAIN / name, TRAIN / name)
for name in ('MONITOR_HANDOFF.md', 'latest_formal_live.json', 'root_live_20261009T115816Z.json',
             'root_live_20261009T122152Z.json', 'LOGOFAIR_POPULATION_PROPOSAL_ROOT_VERIFICATION.json'):
    copy(ROOT / CHECKS / name, CHECKS / name)
for name in ('update_reactivation_state_20261009.py', 'update_completion_current_20261009.py',
             'capture_formal_health_20261009.py', 'verify_logofair_population_root_20261009.py',
             'deploy_first_gpu_valid_recovery_root_20261009.py', 'run_first_gpu_valid_recovery_root_20261009.py',
             'verify_first_gpu_valid_recovery_offserver_root_20261009.py',
             'verify_publication_increment10_20261009.py', Path(__file__).name):
    copy(ROOT / 'tmp' / name, Path('experimental/server_reactivation_20261009') / name)
receipt_path = TRAIN / 'publication_closed_increment10_20261009.json'
attributes = REPO / '.gitattributes'
content = attributes.read_text()
for pattern in ('experimental/celeba_valid_gpu_recovery_implementation_20261009/** -text',
                'experimental/celeba_valid_gpu_recovery_execution_20261009/** -text',
                'experimental/celeba_logofair_population_proposal_20261009/** -text',
                receipt_path.as_posix() + ' -text'):
    if pattern not in content:
        content += '\n' + pattern + '\n'
attributes.write_text(content, newline='\n')
receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_commit=PREVIOUS,
               copied_sha256=MAPPING.copy(), baseline_actual_valid_replays_accepted=425,
               new_GPU_replay_accepted_and_offserver=1, native_max_abs_difference=0,
               old424_collector_unchanged=True, mixed_CPU_GPU_provenance=True,
               preserved_original_CPU_failure=True, CPU_partial10_registered=False,
               original_GPU_diagnostic_registered=False, remaining464_dispatched=False,
               LoGoFair_population_status='ROOT_VERIFIED_PROPOSAL_NOT_APPROVED_NO_FIT_OR_PERFORMANCE',
               mechanism_scientific_records_offserver_verified=40,
               goal_complete=False, final_protocol_frozen=False, test_started=False)
with (ROOT / receipt_path).open('x', encoding='utf8', newline='\n') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write('\n')
copy(ROOT / receipt_path, receipt_path)
names = [*MAPPING, '.gitattributes']
for start in range(0, len(names), 25):
    git('add', '-f', '--', *names[start:start + 25])
    git('add', '--renormalize', '--', *names[start:start + 25])
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
                      byte_verified_files=len(MAPPING), new_GPU_offserver=1, cumulative425=True)))


