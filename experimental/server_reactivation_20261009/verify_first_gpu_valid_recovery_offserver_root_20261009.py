"""Fetch one new evidence archive, verify saved predictions, register a new ledger."""
from pathlib import Path, PurePosixPath
import ast
import copy
import datetime
import hashlib
import json
import subprocess
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
DEST = BASE / 'first1_backup'
REMOTE = '/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/attempt1/chunk_000'
FIRST = 'FairGuard_IID_FedSA_seed91003'
OLD = ROOT / 'tmp/celeba_final_valid_replay_20261009/v4/remaining872_execution_20261009/cumulative_424_accepted.json'
SHA = lambda b: hashlib.sha256(b).hexdigest()
sha = lambda p: SHA(p.read_bytes())
read = lambda p: json.loads(p.read_bytes())
save = lambda p, v: p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
assert read(BASE / 'ROOT_backup_EXIT.json')['returncode'] == 0
assert sha(OLD) == '75ec99bc1eb9fb2c5e20ead68aad7af867d11cb51f2cab991d01712568985fdd'
assert not DEST.exists()
DEST.mkdir()
for name in ['remote_archive_inventory.json', 'chunk_evidence.tar.gz']:
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', '60350',
                    'root@89.22.197.55:' + REMOTE + '/' + name, str(DEST / name)], check=True, timeout=120)
inventory = read(DEST / 'remote_archive_inventory.json')
archive = DEST / 'chunk_evidence.tar.gz'
assert inventory['status'] == 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER'
assert inventory['accepted_ids'] == [FIRST] and inventory['old_model_files_archived'] == 0
assert sha(archive) == inventory['sha256'] and archive.stat().st_size == inventory['bytes']
unpacked = DEST / 'verified_extract'
specs = inventory['members']
with tarfile.open(archive) as bundle:
    assert len(bundle.getmembers()) == len(set(bundle.getnames())) == len(specs) == inventory['member_n']
    assert set(bundle.getnames()) == set(specs)
    for member in bundle.getmembers():
        name = PurePosixPath(member.name)
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in member.name
        assert not member.name.endswith(('.pt', '.pth'))
        data = bundle.extractfile(member).read()
        assert SHA(data) == specs[member.name]['sha256'] and len(data) == specs[member.name]['bytes']
    unpacked.mkdir()
    for member in bundle.getmembers():
        target = unpacked / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(bundle.extractfile(member).read())
strict_path = unpacked / 'execution/strict_acceptance.json'
assert sha(strict_path) == sha(BASE / 'first1_strict_acceptance.json') == '90e3a983116e6b6b7221a44d39164bd942ec141b5547753eef2e4982ab5573ac'
strict = read(strict_path)
assert strict['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED' and strict['accepted_ids'] == [FIRST] and strict['accepted_n'] == 1 and not strict['invalid']
assert strict['max_abs_native_metric_difference'] == 0 and not strict['all900_native_valid_replayed']
deployment = read(BASE / 'ROOT_DEPLOYMENT_VERIFICATION.json')
batch_path = unpacked / 'batch/batch_inputs.json'
batch = read(batch_path)
assert batch['review_sha256'] == deployment['review_sha256'] and batch['implementation_package_sha256'] == deployment['package_sha256']
assert batch['selected_ids'] == [FIRST] and sha(batch_path) == strict['batch_inputs_sha256']
execution = read(unpacked / 'batch/batch_execution.json')
assert not execution['failures'] and execution['source_unchanged'] and execution['source_after'] == batch['source_before']
assert execution['finished_zero_exit_ids'] == [FIRST] and not execution['cohort_registered']
run = unpacked / 'batch/runs' / FIRST
proof_path = run.with_name(FIRST + '.worker.json')
receipt_path, arrays_path = run / 'receipt.json', run / 'validation_predictions.npz'
proof, receipt = read(proof_path), read(receipt_path)
assert proof['status'] == receipt['status'] == 'DIAGNOSTIC_NATIVE_MATCH'
assert proof['artifacts_unchanged'] and proof['artifact_before'] == proof['artifact_after']
assert sha(receipt_path) == proof['receipt_sha256'] and sha(arrays_path) == receipt['prediction_arrays_sha256']
assert receipt['valid_n'] == 19867 and receipt['weights_before'] == receipt['weights_after']
assert receipt['native_comparison']['tolerance'] == 1e-12 and receipt['native_comparison']['max_abs_difference'] == 0
assert receipt['native_comparison']['accepted'] and receipt['runtime']['device'] == 'cuda:0'
assert receipt['runtime']['cuda_device_count'] == 1 and receipt['runtime']['torch'] == '2.11.0+cu128'
assert not any(receipt[k] for k in ['optimizer_created', 'gradients_created', 'test_labels_accessed', 'test_inference_performed', 'final_dispatch_created'])
model_inventory = ROOT / 'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json'
assert sha(model_inventory) == '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
record = next(r for r in read(model_inventory)['records'] if r['id'] == FIRST)
for field, original in [('checkpoint_sha256', 'checkpoint'), ('original_result_sha256', 'result'), ('original_job_sha256', 'raw_job')]:
    assert receipt[field] == record[original]['sha256']
assert receipt['config_canonical_sha256'] == record['config_canonical_sha256']
assert all(receipt[k] == record[k] for k in ['id', 'method', 'distribution', 'attack', 'seed'])
source = ROOT / 'tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py'
assert sha(source) == '805eedf1fb08137cd86a543a80c83b9527e5c8937be02f7d2dca83a33b86e04c'
namespace = {'np': np, 'json': json, 'hashlib': hashlib,
             'METHODS': {'FedAvg', 'Median', 'FLTrust', 'FairFed', 'FairGuard', 'FLTrust+FairGuard', 'GuardFed-AD2+', 'FedAA', 'LASA'},
             'VIEWS': {'native', 'raw', 'shared_calibration'}}
tree = ast.parse(source.read_text(encoding='utf-8'))
for name in ['canonical_sha', 'check_binary', 'group_metrics', 'predict_views', 'evaluate_frozen_predictions']:
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
labels_cache = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009/original_valid_cache.npz'
assert sha(labels_cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
fits = copy.deepcopy(receipt['fits'])
for fit in fits.values():
    if fit['thresholds'] is not None:
        assert set(fit['thresholds']) == {'0', '1'}
        fit['thresholds'] = {int(k): v for k, v in fit['thresholds'].items()}
with np.load(labels_cache, allow_pickle=False) as truth, np.load(arrays_path, allow_pickle=False) as arrays:
    assert len(arrays['root_image_ids']) == len(arrays['root_margins']) == 16277
    assert len(arrays['valid_image_ids']) == len(arrays['valid_margins']) == 19867
    assert SHA(arrays['valid_image_ids'].tobytes()) == receipt['valid_image_ids_sha256'] == strict['valid_image_ids_sha256']
    assert np.isfinite(arrays['root_margins']).all() and np.isfinite(arrays['valid_margins']).all()
    predicted = namespace['predict_views'](arrays['valid_margins'], truth['valid_sensitive'], fits)
    assert all(np.array_equal(predicted[v], arrays['prediction_' + v]) for v in namespace['VIEWS'])
    scored = namespace['evaluate_frozen_predictions'](predicted, truth['valid_y'], truth['valid_sensitive'])
    assert scored == receipt['views']
    assert all(scored['native'][k] == record['prior_validation_metrics'][k] for k in ['accuracy', 'aeod', 'aspd'])
old = read(OLD)
assert old['accepted_n'] == len(set(old['accepted_ids'])) == 424 and FIRST not in old['accepted_ids']
new_ids = old['accepted_ids'] + [FIRST]
verification = {'status': 'ROOT_FIRST_GPU_VALID_REPLAY_OFFSERVER_AND_SAVED_ARRAY_PASS',
                'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'accepted_new_ids': [FIRST],
                'archive_sha256': sha(archive), 'inventory_sha256': sha(DEST / 'remote_archive_inventory.json'),
                'archive_members_verified': len(specs), 'strict_sha256': sha(strict_path),
                'receipt_sha256': sha(receipt_path), 'array_sha256': sha(arrays_path), 'worker_proof_sha256': sha(proof_path),
                'source_package_sha256': deployment['package_sha256'], 'review_sha256': deployment['review_sha256'],
                'saved_metrics_verified': 9, 'saved_confusion_counts_verified': 24, 'saved_prediction_rules_verified': 3,
                'root_refit_verified_in_original_remote_strict': True, 'local_root_refit': False,
                'native_max_abs_difference': 0, 'old424_unchanged': True,
                'constant_native_prediction_retained': len(np.unique(predicted['native'])) == 1,
                'no_CNN_in_offserver_verifier': True, 'all900_complete': False, 'final_test': False}
save(DEST / 'ROOT_OFFSERVER_VERIFICATION.json', verification)
collector = {'scope': 'NINE_METHOD900_VALID_REPLAY_CUMULATIVE_EXPLICIT_CPU_GPU_SOURCE_VERSIONS',
             'status': 'PARTIAL_UNIQUE_VALID_REPLAY_ACCEPTED', 'accepted_n': 425, 'expected_n': 900,
             'accepted_ids': new_ids, 'previous_424_collector_sha256': sha(OLD),
             'previous_424_collector_path': str(OLD.relative_to(ROOT)),
             'new_GPU_id': FIRST, 'new_GPU_verification_sha256': sha(DEST / 'ROOT_OFFSERVER_VERIFICATION.json'),
             'new_GPU_receipt_sha256': sha(receipt_path), 'new_GPU_array_sha256': sha(arrays_path),
             'source_package_sha256': deployment['package_sha256'], 'new_GPU_uuid': proof['GPU_uuid'],
             'mixing_CPU_GPU_disclosed': True, 'all900_native_valid_replayed': False,
             'uniform_device_comparison': False, 'final_protocol_frozen': False, 'test_evaluation_performed': False,
             'CPU_partial10_registered': False, 'diagnostic1_registered': False, 'remaining464_executed': False}
assert len(set(new_ids)) == 425
save(BASE / 'cumulative_425_accepted.json', collector)
assert sha(OLD) == collector['previous_424_collector_sha256']
print(json.dumps(verification))
