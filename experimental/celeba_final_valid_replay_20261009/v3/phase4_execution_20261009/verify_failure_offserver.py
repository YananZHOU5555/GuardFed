"""Independent offserver SHA/member and rejected-attempt verification, stdlib only."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA = 'b201a2e5f309d8b6bbd454cd980a40557cf0bd6b8e6ac229a4f57375aee1cc39'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

inventory = read(HERE / 'failure_archive_inventory.json')
archive = HERE / inventory['archive']
assert sha(archive) == inventory['sha256'] == ARCHIVE_SHA
out = HERE / 'remote_receipts'
out.mkdir(exist_ok=True)
with tarfile.open(archive, 'r:gz') as tf:
    members = tf.getmembers()
    assert len(members) == len({m.name for m in members}) == inventory['member_n'] == 40
    assert {m.name for m in members} == set(inventory['members'])
    for m in members:
        relative = PurePosixPath(m.name)
        assert m.isfile() and not relative.is_absolute() and '..' not in relative.parts
        b = tf.extractfile(m).read()
        assert len(b) == inventory['members'][m.name]['bytes'] and hashlib.sha256(b).hexdigest() == inventory['members'][m.name]['sha256']
        p = out / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            assert p.read_bytes() == b
        else:
            p.write_bytes(b)
batch = out / 'phase4_useful'
inputs, execution = read(batch / 'batch_inputs.json'), read(batch / 'batch_execution.json')
assert inputs['source_before'] == execution['source_after'] and len(inputs['source_before']) == 42
assert execution['source_unchanged'] and execution['identity_error'] is None
assert execution['finished_zero_exit_ids'] == [] and execution['stopped_after_failure'] and len(execution['failures']) == 8
stock = read(HERE.parent.parent / 'inputs/model_inventory.json')
stock = {r['id']: r for r in stock['records']}
errors = []
for i in inputs['selected_ids']:
    w = read(batch / 'runs' / (i + '.worker.json'))
    assert w['status'] == 'FAILED' and w['artifact_before'] == w['artifact_after'] and w['artifacts_unchanged']
    assert sorted(x['sha256'] for x in w['artifact_before'].values()) == sorted(stock[i][k]['sha256'] for k in ('checkpoint', 'result', 'raw_job'))
    errors.append({'id': i, 'error_type': w.get('error_type'), 'error': w.get('error')})
assert sum(x['error_type'] == 'KeyError' and x['error'] == "'output'" for x in errors) == 1
assert not any(batch.rglob('receipt.json')) and not any(batch.rglob('*.npz'))
strict = read(HERE / 'strict_acceptance.json')
assert strict['accepted_n'] == 0 and len(strict['invalid']) == 8 and strict['max_abs_native_metric_difference'] is None
assert not strict['all900_native_valid_replayed'] and strict['status'] == 'PARTIAL_OR_INVALID_VALID_REPLAY'
diagnostic = read(HERE / 'resource_diagnostic.json')
report = {'status': 'FAILURE_EVIDENCE_VERIFIED_OFFSERVER_NOT_ACCEPTED', 'phase': 4, 'archive_sha256': ARCHIVE_SHA, 'member_n': 40, 'all_member_sha_verified': True, 'accepted_n': 0, 'shared_sources_unchanged': 42, 'artifact_files_unchanged': 24, 'errors': errors, 'strict_acceptance_sha256': sha(HERE / 'strict_acceptance.json'), 'failure_measurement_sha256': sha(HERE / 'failure_measurement.json'), 'resource_diagnostic_sha256': sha(HERE / 'resource_diagnostic.json'), 'diagnostic_sample_seconds': diagnostic['sample_seconds'], 'diagnostic_cgroup_throttled_usec_delta': diagnostic['cgroup_cpu_stat_delta']['throttled_usec'], 'diagnostic_claim_limit': 'Startup/import sample; not steady-state CNN utilization. No completed8-worker throughput point.', 'supplemental_stage_files': {n: sha(HERE / n) for n in ('strict_acceptance.json.failure.json', 'strict_acceptance.log')}, 'phase5_started': False, 'prior_accepted_replays_valid': 9, 'all900_native_valid_replayed': False, 'formal_training_modified': False}
p = HERE / 'failure_offserver_verification.json'
assert not p.exists()
p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': report['status'], 'receipt_sha256': sha(p), 'archive_sha256': ARCHIVE_SHA, 'member_n': 40}, indent=2))
