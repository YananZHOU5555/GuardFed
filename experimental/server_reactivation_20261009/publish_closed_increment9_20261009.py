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
PREVIOUS = '0f82ad80078d614adef666a9cb51f20eeb922317'
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
assert state['celeba_mechanism_v1']['scientific_results_offserver_verified'] == 40
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'] == 424
assert state['native_mismatch_GPU_diagnostic_20261009']['scientific_GPU_executions'] == 1
assert state['native_mismatch_GPU_diagnostic_20261009']['status'] == 'COMPLETE_DIAGNOSTIC_ONLY_NOT_COHORT_ACCEPTED'

checks = ROOT / CHECKS
seal(checks, CHECKS, 'root_live_20261009T113229Z_FILES_SHA256.txt')
backups = Path('mechanism_science_backups_20261009')
for name in ('verified_ledger.json', 'root_live_20261009T113229Z_offserver_verification.json'):
    copy(checks / backups / name, CHECKS / backups / name)
inspection = backups / 'mechanism_inspection_v4_20261009T113229Z'
for path in (checks / inspection).iterdir():
    assert path.is_file()
    copy(path, CHECKS / inspection / path.name)
for name in ('latest_formal_live.json', 'root_live_20261009T113822Z.json',
             'MECHANISM_SCIENCE_TOTAL40_ROOT_VERIFICATION.json',
             'FLGMM_DELTA_FOUR_20261009T1133Z_ROOT_VERIFICATION.json',
             'NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json', 'MONITOR_HANDOFF.md'):
    copy(checks / name, CHECKS / name)

fl = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
ftarget = Path('experimental/celeba_flgmm_screen_20261009_v2_dispatch')
seal(fl / 'backups/increment_20261009T1133Z', ftarget / 'backups/increment_20261009T1133Z', 'FILES_SHA256.json')
for name in ('BACKUP_CHAIN_increment_20261009T1133Z.json', 'LATEST_BACKUP.json'):
    copy(fl / name, ftarget / name)
hybrid = Path('celeba_hybrid_screen_execution_20261009/results_incremental_20261009T1133Z')
seal(ROOT / 'tmp' / hybrid, Path('experimental') / hybrid, 'FILES_SHA256.json')

diag = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009'
dtarget = Path('experimental/celeba_native_mismatch_diagnostic_execution_20261009')
seal(diag, dtarget, 'EXECUTION_FILES_SHA256.json')
seal(diag / 'attempt2', dtarget / 'attempt2', 'EXECUTION_FILES_SHA256.json')
seal(diag / 'offline_comparison', dtarget / 'offline_comparison', 'FILES_SHA256.json')
for folder, names in {
    'preinference_failure_backup': ('failure.tar.gz', 'backup_receipt.json', 'OFFSERVER_VERIFICATION.json'),
    'attempt2_backup': ('evidence.tar.gz', 'MEMBERS.json', 'BACKUP.json', 'OFFSERVER_VERIFICATION.json'),
    'offline_comparison_backup': ('offline_comparison.tar.gz', 'MEMBERS.json', 'BACKUP_VERIFICATION.json'),
}.items():
    for name in names:
        copy(diag / folder / name, dtarget / folder / name)
for name in ('FINAL_DELIVERY.json', 'post_diagnostic_live.json', 'post_live.py', 'archive_attempt2.py'):
    copy(diag / name, dtarget / name)

# Prepared proposal remains separate from all accepted records and running queues.
prepared = ROOT / 'tmp/celeba_valid_recovery_prepared_20261009'
assert sha(prepared / 'PACKAGE_SHA256.json') == '586d4443ae6f074e686f872400dd90b91356f0e504d29faa8c8cb0ff3ea8e26a'
assert sha(prepared / 'FILES_SHA256.txt') == '69f8bfafb88f2ede62f86dd07f9c16dd192dcdedf9e5cc3b5156091cbc7b375f'
manifest = read(prepared / 'manifest.json')
inventory = read(ROOT / TRAIN / 'final_evaluation_prepared_20261009/model_inventory.json')
collector = read(ROOT / 'tmp/celeba_final_valid_replay_20261009/v4/remaining872_execution_20261009/cumulative_424_accepted.json')
inventory_ids = [r['id'] for r in inventory['records']]
accepted = set(collector['accepted_ids'])
remaining = [r['id'] for r in manifest['records']]
assert len(inventory_ids) == len(set(inventory_ids)) == 900
assert len(accepted) == 424 and len(remaining) == len(set(remaining)) == 476
assert set(remaining) == set(inventory_ids) - accepted
assert set(manifest['accepted424_ids']) == accepted
counts = dict(collections.Counter(r['classification'] for r in manifest['records']))
assert counts == {'CPU_STRICT_PARTIAL_10_NOT_REGISTERED': 10,
                  'GPU_DIAGNOSTIC_1_PENDING_COHORT_DECISION': 1, 'UNEXECUTED_465': 465}
assert all(value is None for value in manifest['root_review_decisions'].values())
seal(prepared, Path('experimental/celeba_valid_recovery_prepared_20261009'), 'FILES_SHA256.txt')

for name in ('RUNNING.md', 'TRAINING_STATE.json', 'REBUTTAL_COMPLETION_20261009.md',
             'celeba_mechanism_v1/EXECUTION.md', 'publication_closed_increment8_verified_20261009.json'):
    copy(ROOT / TRAIN / name, TRAIN / name)
for folder in ('interim_tables_20261009T105813Z', 'interim_tables_20261009T113229Z'):
    for name in ('TABLES.md', 'tables.json'):
        path = Path('celeba_mechanism_v1') / folder / name
        copy(ROOT / TRAIN / path, TRAIN / path)
for name in ('update_reactivation_state_20261009.py', 'update_completion_current_20261009.py',
             'render_mechanism_interim_tables_20261009.py', 'verify_flgmm_delta_four_root_20261009.py',
             'verify_native_GPU_diagnostic_root_20261009.py', Path(__file__).name):
    copy(ROOT / 'tmp' / name, Path('experimental/server_reactivation_20261009') / name)

receipt_path = TRAIN / 'publication_closed_increment9_20261009.json'
attributes = REPO / '.gitattributes'
content = attributes.read_text()
for pattern in ('experimental/celeba_native_mismatch_diagnostic_execution_20261009/** -text',
                'experimental/celeba_valid_recovery_prepared_20261009/** -text',
                'experimental/celeba_flgmm_screen_20261009_v2_dispatch/backups/increment_20261009T1133Z/** -text',
                'experimental/celeba_flgmm_screen_20261009_v2_dispatch/BACKUP_CHAIN_increment_20261009T1133Z.json -text',
                'experimental/celeba_flgmm_screen_20261009_v2_dispatch/LATEST_BACKUP.json -text',
                receipt_path.as_posix() + ' -text'):
    if pattern not in content:
        content += '\n' + pattern + '\n'
attributes.write_text(content, newline='\n')
receipt = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_commit=PREVIOUS,
               copied_sha256=MAPPING.copy(), mechanism_scientific_records_offserver_verified=40,
               baseline_actual_valid_replays_accepted=424, mechanism_three_view_replays_offserver_verified=23,
               FLGMM_70round_jobs_offserver_verified=6, Hybrid_70round_jobs_offserver_verified=0,
               diagnostic_GPU_executions=1, diagnostic_not_cohort_accepted=True,
               original_native_tolerance=1e-12, saved_prediction_flips_native_raw_shared=[1, 1, 0],
               historical_unique_cause_proven=False, prepared_recovery_n=476,
               prepared_recovery_counts=counts, prepared_recovery_executed=False,
               goal_complete=False, test_started=False)
with (ROOT / receipt_path).open('x', encoding='utf8', newline='\n') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
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
    end = payload.index(b'\n', position)
    header = payload[position:end].split()
    assert header[1] == b'blob'
    size = int(header[2])
    assert hashlib.sha256(payload[end + 1:end + 1 + size]).hexdigest() == digest, name
    position = end + size + 2
assert position == len(payload)
print(json.dumps(dict(status='SCOPED_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',
                      changed_files=len(changed), byte_verified_files=len(MAPPING), prepared_only_n=476)))
