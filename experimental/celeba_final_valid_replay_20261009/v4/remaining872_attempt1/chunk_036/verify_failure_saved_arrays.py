"""Preserve chunk036 failure and diagnose saved arrays; never accepts or replays."""
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import io
import json
import tarfile
import numpy as np

BASE = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parent
FAILED = 'FairGuard_IID_F-Flip_seed91009'
ARCHIVE_SHA = 'fe5c3786ead4d4878e6e6f16c89793e3d10163f75559ef7ad7e798798ccfb555'
AUDIT_SHA = 'da276d12084e6950753c18f73857a6dbcd408cdb7fba162222e94aff011e3718'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))

archive = STAGE / 'failure_chunk_evidence.tar.gz'
inventory = read(STAGE / 'failure_remote_archive_inventory.json')
assert sha(archive) == inventory['sha256'] == ARCHIVE_SHA
assert archive.stat().st_size == inventory['bytes']
data = {}
with tarfile.open(archive) as tf:
    entries = tf.getmembers()
    assert len(entries) == len({x.name for x in entries}) == len(inventory['members']) == 65
    for item in entries:
        name = PurePosixPath(item.name)
        assert item.isfile() and not name.is_absolute() and '..' not in name.parts
        value = tf.extractfile(item).read()
        expected = inventory['members'][item.name]
        assert item.size == len(value) == expected['bytes']
        assert hashlib.sha256(value).hexdigest() == expected['sha256']
        data[item.name] = value
j = lambda name: json.loads(data[name])
m = read(BASE / 'v4/remaining872_prepared_v2_20261009/manifest.json')
assert hashlib.sha256(data['execution/manifest.json']).hexdigest() == 'ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43'
additional_execution_sha = {
    'execution_sources/run_remaining.sh': '666f440b728848adc46519c1718ddb35da2cb939d18e9c4b043c3cb6f313971b',
    'execution_sources/guardfed_celeba_valid_remaining872_20261009.conf': '14c6b45c00dba32220800a0ce1298bcff0f36a6483f584b5b9417a96bee9fb34',
}
for member, path in m['source_archive'].items():
    expected = m['input_sha256'].get(path, m['scientific_source_sha256'].get(path, additional_execution_sha.get(member)))
    assert expected and hashlib.sha256(data[member]).hexdigest() == expected, member
batch, execution = j('batch/batch_inputs.json'), j('batch/batch_execution.json')
expected_sources = dict(m['scientific_source_sha256'])
expected_sources['/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/semantic900_inspection.json'] = 'cc93d94d69478a4ee190abff76da4fb185791ece7e4e1d2ee75360482ead80b4'
assert execution['source_unchanged'] and batch['source_before'] == execution['source_after']
assert {k: v['sha256'] for k, v in batch['source_before'].items()} == expected_sources
records = {r['id']: r for r in read(BASE / 'inputs/model_inventory.json')['records']}
strict = read(STAGE / 'strict_acceptance.json')
assert strict['requested_n'] == 11 and strict['accepted_n'] == 10
assert [r['id'] for r in strict['invalid']] == [FAILED]
assert strict['status'] == 'PARTIAL_OR_INVALID_VALID_REPLAY'
audit_path = BASE / 'v4/remaining872_prepared_v2_20261009/audit_remaining.py'
assert sha(audit_path) == AUDIT_SHA
spec = importlib.util.spec_from_file_location('sealed_saved_array_audit', audit_path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
cache = BASE / 'verification_inputs/original_valid_cache.npz'
assert sha(cache) == audit.CACHE_SHA
with np.load(cache, allow_pickle=False) as z:
    y, sensitive = z['valid_y'], z['valid_sensitive']
rows = []
for model_id in m['chunks'][36]['ids']:
    prefix = 'batch/runs/' + model_id
    r, w = j(prefix + '/receipt.json'), j(prefix + '.worker.json')
    assert r['weights_before'] == r['weights_after']
    assert w['artifact_before'] == w['artifact_after'] and w['artifacts_unchanged']
    for kind in ('checkpoint', 'result', 'raw_job'):
        assert records[model_id][kind]['sha256'] in {x['sha256'] for x in w['artifact_before'].values()}
    assert hashlib.sha256(data[prefix + '/validation_predictions.npz']).hexdigest() == r['prediction_arrays_sha256']
    with np.load(io.BytesIO(data[prefix + '/validation_predictions.npz']), allow_pickle=False) as arrays:
        audit.check_array_ids(records[model_id], arrays)
        metric_differences, count_n = audit.direct_checks(r, arrays, y, sensitive)
        assert all(np.isfinite(arrays[k]).all() for k in ('root_margins', 'valid_margins'))
    rows.append({'id': model_id, 'native_comparison': r['native_comparison'], 'worker_status': w['status'], 'saved_view_metric_differences': metric_differences, 'saved_confusion_count_checks': count_n})
r = j('batch/runs/' + FAILED + '/receipt.json')
w = j('batch/runs/' + FAILED + '.worker.json')
assert w['error'] == 'Native metrics exceed fixed tolerance; preserve evidence and stop without retry'
assert not r['native_comparison']['accepted']
original = STAGE / 'input_original_result.json'
assert sha(original) == records[FAILED]['result']['sha256'] == 'd810b15d357a9d31a14f2edda7e18fb780d5f7329ae4202f313daef8da395944'
old = read(original)
last = lambda value: value[-1] if isinstance(value, list) else value
old_metrics = {k: last(v) for k, v in old['metrics'].items()}
assert old_metrics == r['native_comparison']['expected']
old_stats = {k: last(v) for k, v in old['evaluation_stats'].items()}
counts = r['views']['native']['group_confusion_counts']
def scores(c):
    total = sum(v['tp'] + v['fp'] + v['tn'] + v['fn'] for v in c.values())
    tpr = [c[str(g)]['tp'] / (c[str(g)]['tp'] + c[str(g)]['fn']) for g in (0, 1)]
    rates = [(c[str(g)]['tp'] + c[str(g)]['fp']) / sum(c[str(g)][k] for k in ('tp', 'fp', 'tn', 'fn')) for g in (0, 1)]
    return {'accuracy': sum(v['tp'] + v['tn'] for v in c.values()) / total, 'aeod': abs(tpr[0] - tpr[1]), 'aspd': abs(rates[0] - rates[1]), 'positive_rate': sum(v['tp'] + v['fp'] for v in c.values()) / total}
hypotheses = []
for group in (0, 1):
    for label, negative, positive in ((0, 'tn', 'fp'), (1, 'fn', 'tp')):
        for cpu_prediction in (0, 1):
            c = json.loads(json.dumps(counts))
            source, target = (negative, positive) if cpu_prediction == 0 else (positive, negative)
            if c[str(group)][source] < 1:
                continue
            c[str(group)][source] -= 1
            c[str(group)][target] += 1
            actual = scores(c)
            if all(abs(actual[k] - old_metrics[k]) <= 1e-12 for k in old_metrics) and abs(actual['positive_rate'] - old_stats['positive_rate']) <= 1e-12:
                hypotheses.append({'sensitive': group, 'label': label, 'cpu_prediction': cpu_prediction, 'hypothesized_original_prediction': 1 - cpu_prediction})
with np.load(io.BytesIO(data['batch/runs/' + FAILED + '/validation_predictions.npz']), allow_pickle=False) as z:
    margins = z['valid_margins']
    closest = [{'position': int(i), 'image_id': int(z['valid_image_ids'][i]), 'margin': float(margins[i]), 'label': int(y[i]), 'sensitive': int(sensitive[i]), 'cpu_prediction': int(z['prediction_native'][i])} for i in np.argsort(np.abs(margins), kind='stable')[:12]]
    margin_summary = {'min_abs': float(np.abs(margins).min()), 'abs_quantiles': {str(q): float(np.quantile(np.abs(margins), q)) for q in (0, .001, .01, .5, 1)}, 'abs_below': {str(t): int((np.abs(margins) < t).sum()) for t in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)}, 'zero_n': int((margins == 0).sum()), 'closest12': closest}
# Preserve the failed worker's original bytes separately for direct review.
extracted = []
for member in data:
    if member.startswith('batch/runs/' + FAILED):
        destination = STAGE / 'failed_worker_original' / member.removeprefix('batch/runs/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        destination.write_bytes(data[member])
        extracted.append({'path': destination.relative_to(STAGE).as_posix(), 'sha256': sha(destination)})
report = {'status': 'FAILURE_ARCHIVE_PRESERVED_AND_SAVED_ARRAYS_VERIFIED_NOT_ACCEPTED', 'archive_sha256': ARCHIVE_SHA, 'archive_members_verified': 65, 'source_data_identity_unchanged': True, 'source_data_sha_n': len(expected_sources), 'failed_id': FAILED, 'original_training_torch': r['original_training_torch'], 'runtime': r['runtime'], 'original_result_sha256': sha(original), 'native_comparison': r['native_comparison'], 'cpu_native_confusion_counts': counts, 'original_evaluation_stats_terminal': old_stats, 'original_confusion_counts_or_per_image_predictions_present_in_result': False, 'one_flip_hypotheses_consistent_with_original_terminal_metrics_and_positive_rate': hypotheses, 'margin_summary': margin_summary, 'saved_arrays_consistent_models_n': len(rows), 'saved_arrays_metric_checks': len(rows) * 9, 'saved_arrays_confusion_count_checks': sum(x['saved_confusion_count_checks'] for x in rows), 'registered_accepted_n': 0, 'strict_partial_n_not_registered': 10, 'failure_traceback': w['traceback'], 'all_model_result_rawjob_and_weight_identities_unchanged': True, 'models': rows, 'extracted_failed_original_files': extracted, 'verifier_source_sha256': sha(__file__), 'reused_frozen_direct_array_audit_sha256': AUDIT_SHA, 'claim_limit': 'No original per-image predictions or group confusion counts are stored in this original result. A one-flip group/label explanation is an inference conditional on one changed decision; candidate near-zero margins cannot uniquely identify an actual flipped image. CPU-versus-GPU or other numerical root cause is not established. No CNN replay, tolerance change, threshold change or acceptance of this partial chunk.', 'all900_native_valid_replayed': False}
target = STAGE / 'failure_offserver_verification.json'
assert not target.exists()
target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'failure_proof_sha256': sha(target), 'archive_sha256': ARCHIVE_SHA, 'members': 65, 'original_training_torch': r['original_training_torch'], 'one_flip_hypotheses': hypotheses, 'margin_summary': margin_summary, 'registered_accepted_n': 0}, indent=2))

