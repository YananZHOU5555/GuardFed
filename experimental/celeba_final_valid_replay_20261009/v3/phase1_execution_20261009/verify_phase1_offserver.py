"""Verify actual phase1 bytes and direct valid counts; no image inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
MODEL_ID = 'FedAvg_IID_Benign_seed91001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    inventory = read(HERE / 'remote_archive_inventory.json')
    archive = HERE / inventory['archive']
    assert sha(archive) == inventory['sha256'] == 'ff5a67fd9c01724d7778dda83cf7982d7003669eda61a30990adfe5ae5035de8'
    assert archive.stat().st_size == inventory['bytes']
    extracted = HERE / 'remote_receipts'
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        assert len(members) == len({m.name for m in members}) == inventory['member_n'] == 20
        assert {m.name for m in members} == set(inventory['members'])
        for m in members:
            target = (extracted / m.name).resolve()
            assert m.isfile() and target.is_relative_to(extracted.resolve())
            payload = tf.extractfile(m).read()
            expected = inventory['members'][m.name]
            assert hashlib.sha256(payload).hexdigest() == expected['sha256'] and len(payload) == expected['bytes']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == payload, 'Preserve differing raw receipt'
            else:
                target.write_bytes(payload)
    original = read(BASE / 'inputs/model_inventory.json')
    record = next(r for r in original['records'] if r['id'] == MODEL_ID)
    path = extracted / 'phase1_useful/runs' / MODEL_ID
    r = read(path / 'receipt.json')
    proof = read(extracted / 'phase1_useful/runs' / (MODEL_ID + '.worker.json'))
    acceptance_path = extracted / 'phase1_execution_20261009/strict_acceptance.json'
    acceptance = read(acceptance_path)
    assert sha(acceptance_path) == '0497b7fd0c884e0eb6de66ac09affb667291cccdc7bb3fac37be1af3b1fa047e'
    assert acceptance['accepted_ids'] == [MODEL_ID] and not acceptance['invalid']
    assert acceptance['accepted_n'] == acceptance['requested_n'] == 1 and not acceptance['all900_native_valid_replayed']
    batch = read(extracted / 'phase1_useful/batch_inputs.json')
    execution = read(extracted / 'phase1_useful/batch_execution.json')
    assert batch['selected_ids'] == execution['finished_zero_exit_ids'] == [MODEL_ID]
    assert batch['source_before'] == execution['source_after'] and execution['source_unchanged'] and not execution['failures']
    assert len(batch['source_before']) == 42
    assert proof['artifact_before'] == proof['artifact_after'] and proof['artifacts_unchanged']
    assert proof['receipt_sha256'] == sha(path / 'receipt.json')
    assert r['weights_before'] == r['weights_after'] and not r['optimizer_created'] and not r['gradients_created']
    assert r['checkpoint_sha256'] == record['checkpoint']['sha256'] and r['valid_n'] == 19867 and r['root_reconstruction']['root_n'] == 16277
    assert r['root_reconstruction']['root_image_ids_sha256'] == record['data_contract']['root_image_ids_sha256']
    assert sha(path / 'validation_predictions.npz') == r['prediction_arrays_sha256']
    prefix = proof['metadata_read_receipt']
    assert not prefix['full_loader_executed']
    assert all(p['requested_entries'] == 182637 and p['test_entries_requested_or_materialized'] == 0 for p in prefix['prefix_reads'])
    cache = BASE / 'verification_inputs/original_valid_cache.npz'
    assert sha(cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
    with np.load(cache, allow_pickle=False) as z:
        y, s = z['valid_y'], z['valid_sensitive']
    differences, count_checks = {}, 0
    with np.load(path / 'validation_predictions.npz', allow_pickle=False) as z:
        assert len(z['valid_margins']) == len(y) == 19867 and len(z['root_margins']) == 16277
        assert hashlib.sha256(z['valid_image_ids'].tobytes()).hexdigest() == record['data_contract']['evaluation_image_ids_sha256']
        assert hashlib.sha256(z['root_image_ids'].tobytes()).hexdigest() == record['data_contract']['root_image_ids_sha256']
        for view in ('native', 'raw', 'shared_calibration'):
            fit, pred = r['fits'][view], z['prediction_' + view]
            if fit['rule'] == 'argmax_margin_strictly_positive':
                assert fit['thresholds'] is None
                expected = z['valid_margins'] > 0
            else:
                thresholds = fit['thresholds']
                expected = np.where(s == 0, z['valid_margins'] >= thresholds['0'], z['valid_margins'] >= thresholds['1'])
            assert np.array_equal(pred, expected)
            tpr = [int(((s == g) & (y == 1) & (pred == 1)).sum()) / int(((s == g) & (y == 1)).sum()) for g in (0, 1)]
            rates = [int(((s == g) & (pred == 1)).sum()) / int((s == g).sum()) for g in (0, 1)]
            direct = {'accuracy': int((pred == y).sum()) / len(y), 'aeod': abs(tpr[0] - tpr[1]), 'aspd': abs(rates[0] - rates[1])}
            differences[view] = {k: direct[k] - r['views'][view][k] for k in direct}
            assert all(d == 0 for d in differences[view].values())
            for g in (0, 1):
                counts = {'tp': int(((s == g) & (y == 1) & (pred == 1)).sum()), 'fp': int(((s == g) & (y == 0) & (pred == 1)).sum()),
                          'tn': int(((s == g) & (y == 0) & (pred == 0)).sum()), 'fn': int(((s == g) & (y == 1) & (pred == 0)).sum())}
                assert all(r['views'][view]['group_confusion_counts'][str(g)][k] == value for k, value in counts.items())
                count_checks += len(counts)
    native_difference = max(abs(r['views']['native'][k] - record['prior_validation_metrics'][k]) for k in ('accuracy', 'aeod', 'aspd'))
    assert native_difference == r['native_comparison']['max_abs_difference'] == 0.0
    for snapshot in (r['before_resources'], r['after_resources']):
        assert all(cpus == list(range(16, 24)) for cpus in snapshot['thread_cpu_affinities'].values())
        assert not snapshot['training_queue_snapshot']['failed'] and 'RUNNING' in snapshot['service']['stdout']
    during = read(extracted / 'phase1_execution_20261009/impact_during.json')
    after = read(extracted / 'phase1_execution_20261009/impact_after.json')
    a = {x['id']: x['progress']['round'] for x in during['active_rounds'] if x['progress']}
    b = {x['id']: x['progress']['round'] for x in after['active_rounds'] if x['progress']}
    growth = [{'id': i, 'during': a[i], 'after': b[i]} for i in sorted(set(a) & set(b))]
    assert growth and all(x['after'] > x['during'] for x in growth)
    assert not during['training_queue']['failed'] and not after['training_queue']['failed']
    report = {'status': 'PASS', 'id': MODEL_ID, 'archive_sha256': sha(archive), 'archive_members_verified': 20,
              'strict_acceptance_sha256': sha(acceptance_path), 'actual_v3_accepted_n': 1, 'all900_native_valid_replayed': False,
              'shared_input_identities_unchanged': 42, 'model_result_job_hashes_unchanged': 3,
              'independent_view_metric_differences': differences, 'independent_confusion_count_checks': count_checks,
              'native_max_abs_difference': native_difference, 'receipt_sha256': sha(path / 'receipt.json'),
              'prediction_arrays_sha256': r['prediction_arrays_sha256'], 'batch_wall_seconds': acceptance['wall_seconds'],
              'batch_accepted_models_per_second': acceptance['models_per_second'], 'job_elapsed_seconds': r['elapsed_seconds'],
              'cpu_user_seconds': r['cpu_user_seconds'], 'cpu_system_seconds': r['cpu_system_seconds'],
              'effective_cpu_cores': r['effective_cpu_cores'], 'peak_rss_kib': r['peak_rss_kib'],
              'os_threads_before_after': [r[k]['os_threads'] for k in ('before_resources', 'after_resources')],
              'nice': r['runtime']['nice'], 'all_threads_bound_to_cpus': list(range(16, 24)),
              'training_completed_during_after': [len(during['training_queue']['completed']), len(after['training_queue']['completed'])],
              'training_common_active_round_growth': growth, 'training_failed_during_after': [0, 0],
              'gpu_during_after': [during['gpu'], after['gpu']],
              'metadata_prefix_entries_per_member': 182637, 'test_label_entries_returned_or_materialized_as_arrays': 0,
              'full_celeba_loader_executed': False, 'test_image_inference_performed': False,
              'label_access_disclosure_sha256': sha(extracted / 'phase1_execution_20261009/label_access_disclosure.json'),
              'claim_limit': 'Full-file SHA reads include test-containing bytes; no semantic test-label use. Training continued, no controlled zero-slowdown estimate. One worker measurement does not determine optimum concurrency.',
              'verifier_sha256': sha(Path(__file__)), 'final_protocol_status': 'PREPARED_NOT_FROZEN', 'phase2_started': False}
    out = HERE / 'offserver_verification.json'
    assert not out.exists(), 'Preserve every prior verifier result'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
