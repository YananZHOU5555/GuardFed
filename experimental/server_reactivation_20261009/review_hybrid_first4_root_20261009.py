"""Execute the reviewed record-only bridge; bind source/runtime and every delta byte."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, importlib.util, json, tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_hybrid_screen_execution_20261009'
DELTA = BASE / 'accepted_delta_first_20261009'
BRIDGE = DELTA / 'local_record_bridge_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
assert sha(DELTA/'DELIVERY_FILES_SHA256.json') == '362508dc871bda917ca64b645f792957e7730e55361d2d262d3e8e1c2a034be1'
for name, row in read(DELTA/'DELIVERY_FILES_SHA256.json')['members'].items():
    assert sha(DELTA/name) == row['sha256'] and (DELTA/name).stat().st_size == row['bytes']
assert sha(BRIDGE/'FILES_SHA256.json') == 'dd02f82fefc26f352bc854fd1671e04627907aea57ca6b795fa7f526a53b68a7'
for name, row in read(BRIDGE/'FILES_SHA256.json')['members'].items():
    assert sha(BRIDGE/name) == row['sha256'] and (BRIDGE/name).stat().st_size == row['bytes']
assert len(read(BRIDGE/'FILES_SHA256.json')['members']) == 5
ready = read(DELTA/'ROOT_READY_DELIVERY.json')
server = read(DELTA/'PARTIAL_ACCEPTANCE.json')
tensor = read(DELTA/'OFFSERVER_MEMBER_TENSOR_PROOF.json')
assert ready['old_accepted'] == 0 and ready['new_server_strict_and_offserver_verified'] == 4
assert ready['accepted_new_ids'] == server['accepted_new_ids'] == tensor['accepted_new_ids']
assert tensor['different_host'] and not tensor['local_CUDA_initialized'] and tensor['member_count'] == 53
assert ready['source_seal_sha256'] == sha(BASE/'FILES_SHA256.json')
members = read(DELTA/'MEMBERS.json')['members']
with tarfile.open(DELTA/'hybrid_first_delta.tar.gz') as bundle:
    assert len(bundle.getnames()) == len(set(bundle.getnames())) == 53
    assert set(bundle.getnames()) == set(members) | {'MEMBERS.json'}
    for item in bundle:
        rel = PurePosixPath(item.name)
        assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        expected = sha(DELTA/'MEMBERS.json') if item.name == 'MEMBERS.json' else members[item.name]['sha256']
        assert hashlib.sha256(bundle.extractfile(item).read()).hexdigest() == expected
spec = importlib.util.spec_from_file_location('reviewed_hybrid_record_bridge', BRIDGE/'bridge.py')
bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
proof = bridge.replay()
assert proof['status'] == 'RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS'
assert proof['local_runtime_not_claimed_equal'] and not proof['local_CUDA_initialized']
assert [r['id'] for r in proof['records']] == ready['accepted_new_ids']
for row, identity, accepted in zip(proof['records'], tensor['records'], server['records']):
    assert row['id'] == identity['id'] and row['checkpoint_sha256'] == identity['model_sha256']
    assert row['id'] == accepted['id'] and row['metrics'] == accepted['metrics']
    assert row['checkpoint_sha256'] == accepted['model_sha256'] and accepted['rounds'] == 70
proof.update(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=sha(BASE/'FILES_SHA256.json'),
    bridge_seal_sha256=sha(BRIDGE/'FILES_SHA256.json'),
    delivery_seal_sha256=sha(DELTA/'DELIVERY_FILES_SHA256.json'),
    archive_sha256=sha(DELTA/'hybrid_first_delta.tar.gz'),
    offserver_tensor_proof_sha256=sha(DELTA/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),
    archive_members_verified=53, accepted_before=0, accepted_new=4,
    root_source_reversal_review=True, scientific_checks_and_null_policy_unchanged=True,
    local_CNN_inference=0, selection_performed=False,
    log_limitation=ready['log_note'])
target = DELTA/'ROOT_RECORD_REVIEW.json'
with target.open('x', encoding='utf8') as stream:
    json.dump(proof, stream, indent=2); stream.write('\n')
print(json.dumps({'status':proof['status'], 'accepted_new':4, 'local_torch':proof['local_torch'],
                  'proof_sha256':sha(target), 'CUDA_initialized':proof['local_CUDA_initialized']}))
