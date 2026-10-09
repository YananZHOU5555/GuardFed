"""Independently verify the saved one-model CPU/GPU contrast; no CNN execution."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import importlib.util
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009'
CHECKS = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
delivery = read(BASE / 'FINAL_DELIVERY.json')
assert sha(BASE / 'FINAL_DELIVERY.json') == 'c34f9d45b91fc7f1a91de87dd247cb6a76490f22651febe53d40474aaf9f4e83'
restore = {}
counts = []
for folder, expected, n in [
    ('preinference_failure_backup', delivery['attempt1_archive_sha256'], 28),
    ('attempt2_backup', delivery['attempt2_archive_sha256'], 15),
    ('offline_comparison_backup', 'a9b203418fabe1bfb7c1e8ad3afe53c5d4079ed881a44cca114761945307ded1', 13),
]:
    base = BASE / folder
    archive = next(base.glob('*.tar.gz'))
    assert sha(archive) == expected
    inventory_path = base / ('verified_extract/backup_inventory.json' if folder == 'preinference_failure_backup' else 'MEMBERS.json')
    inventory_name = 'backup_inventory.json' if folder == 'preinference_failure_backup' else 'MEMBERS.json'
    specs = read(inventory_path)['members']
    normalized = {name.replace('\\', '/'): row for name, row in specs.items()}
    assert len(normalized) == len(specs)
    specs = normalized
    mapping = {}
    with tarfile.open(archive) as bundle:
        assert len(bundle.getmembers()) == len(set(bundle.getnames())) == n
        assert set(bundle.getnames()) == set(specs) | {inventory_name}
        for member in bundle.getmembers():
            assert member.isfile()
            data = bundle.extractfile(member).read()
            if member.name == inventory_name:
                assert hashlib.sha256(data).hexdigest() == sha(inventory_path)
                continue
            spec = specs[member.name]
            assert len(data) == spec['bytes'] and hashlib.sha256(data).hexdigest() == spec['sha256']
            portable = PurePosixPath(member.name.replace('\\', '/'))
            assert not portable.is_absolute() and '..' not in portable.parts and ':' not in str(portable)
            mapping[member.name] = str(portable)
    assert len(set(mapping.values())) == len(mapping)
    restore[expected] = mapping
    counts.append(n)
gpu_dir = BASE / 'attempt2_backup/verified_extract/runs/FairGuard_IID_F-Flip_seed91009'
cpu_dir = BASE / 'prepared/cpu_evidence'
gpu, cpu = read(gpu_dir / 'receipt.json'), read(cpu_dir / 'receipt.json')
assert gpu['native_comparison']['max_abs_difference'] == 0 and gpu['native_comparison']['tolerance'] == 1e-12
assert cpu['native_comparison']['accepted'] is False
assert gpu['weights_before'] == gpu['weights_after'] == cpu['weights_before'] == cpu['weights_after']
assert sha(gpu_dir / 'validation_predictions.npz') == gpu['prediction_arrays_sha256']
assert sha(cpu_dir / 'validation_predictions.npz') == cpu['prediction_arrays_sha256']
cache = BASE / 'original_valid_cache.npz'
assert sha(cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
source = ROOT / 'tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py'
assert sha(source) == '805eedf1fb08137cd86a543a80c83b9527e5c8937be02f7d2dca83a33b86e04c'
spec = importlib.util.spec_from_file_location('original_diag_evaluator', source)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
with np.load(cache, allow_pickle=False) as truth:
    y, sensitive = truth['valid_y'], truth['valid_sensitive']
flips = {}
with np.load(cpu_dir / 'validation_predictions.npz', allow_pickle=False) as c, np.load(gpu_dir / 'validation_predictions.npz', allow_pickle=False) as g:
    assert np.array_equal(c['valid_image_ids'], g['valid_image_ids'])
    for view in ['native', 'raw', 'shared_calibration']:
        for arrays, receipt in [(c, cpu), (g, gpu)]:
            predicted = arrays['prediction_' + view]
            metrics = evaluator.group_metrics(y, predicted, sensitive)
            assert metrics == receipt['views'][view]
            tp = [int(np.sum((sensitive == s) & (y == 1) & (predicted == 1))) for s in [0, 1]]
            positives = [int(np.sum((sensitive == s) & (y == 1))) for s in [0, 1]]
            rates = [float(np.mean(predicted[sensitive == s])) for s in [0, 1]]
            assert metrics['accuracy'] == float(np.mean(predicted == y))
            assert metrics['aeod'] == abs(tp[0] / positives[0] - tp[1] / positives[1])
            assert metrics['aspd'] == abs(rates[0] - rates[1])
            fit = dict(receipt['fits'][view])
            if fit['thresholds'] is not None:
                assert set(fit['thresholds']) == {'0', '1'}
                fit['thresholds'] = {int(k): v for k, v in fit['thresholds'].items()}
            assert np.array_equal(evaluator.predict_views(arrays['valid_margins'], sensitive, {view: fit})[view], predicted)
        changed = np.flatnonzero(c['prediction_' + view] != g['prediction_' + view])
        flips[view] = [int(c['valid_image_ids'][i]) for i in changed]
assert flips == {'native': [172599], 'raw': [172599], 'shared_calibration': []}
assert delivery['scientific_GPU_executions'] == 1 and delivery['accepted_cohort_count_unchanged'] == 424
result = dict(status='ROOT_SAVED_CPU_GPU_ARRAYS_AND_ARCHIVES_PASS_DIAGNOSTIC_ONLY',
              verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), archive_members_verified=counts,
              native_GPU_max_abs_difference=0, prediction_flip_image_ids=flips,
              original_CPU_failure_preserved=True, accepted_cohort_unchanged=424,
              historical_GPU_array_available=False, unique_historical_cause_established=False,
              delivery_sha256=sha(BASE / 'FINAL_DELIVERY.json'), new_CNN_or_GPU_inference=0,
              portable_archive_restore_names=restore, test_started=False)
out = CHECKS / 'NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json'
with out.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps({k:v for k,v in result.items() if k != 'portable_archive_restore_names'}))
