"""Direct-count acceptance of explicit-source useful v4 phases4..5; no image inference or dispatch."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PLAN_SHA = '80686577e2a92da823865c20b14153a40692c173a00addd14fe586f2eeb9793c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


V4_SHA = '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e'
V3_SHA = 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'
SEMANTIC_SHA = 'cc93d94d69478a4ee190abff76da4fb185791ece7e4e1d2ee75360482ead80b4'

def batch_source_pins_are_v4(acceptance, out, prefix):
    assert sha(out / 'sealed_sources/replay_v4.py') == V4_SHA
    assert sha(out / 'sealed_sources/replay_v3.py') == V3_SHA
    assert acceptance['v2_source_sha256'] == '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'
    b = read(out / (prefix + '_useful/batch_inputs.json'))
    assert b['scope'] == 'VALID_ONLY_IMPLEMENTATION_REPLAY_V4' and b['v3_source_sha256'] == V4_SHA
    assert b['source_before']['/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/semantic900_inspection.json']['sha256'] == SEMANTIC_SHA
    return True


def main(phase, expected_archive_sha, destination):
    assert phase in (4, 5)
    stage = BASE / 'v4' / ('phase' + str(phase) + '_attempt1_execution_20261009')
    assert sha(BASE / 'v3/throughput_plan.json') == PLAN_SHA
    selected = read(BASE / 'v3/throughput_plan.json')['phases'][phase - 1]
    IDS = [r['id'] for r in selected['models']]
    n = selected['workers']
    assert n == len(IDS) == len(set(IDS))
    phase_prefix = 'phase' + str(phase) + '_attempt1'
    member_n = 24 + 4 * n
    m = read(stage / 'remote_archive_inventory.json')
    archive = stage / m['archive']
    assert sha(archive) == m['sha256'] == expected_archive_sha and archive.stat().st_size == m['bytes']
    out = stage / 'remote_receipts'
    with tarfile.open(archive, 'r:gz') as tf:
        entries = tf.getmembers()
        assert len(entries) == len({e.name for e in entries}) == m['member_n'] == member_n
        assert {e.name for e in entries} == set(m['members'])
        for e in entries:
            target = (out / e.name).resolve()
            assert e.isfile() and target.is_relative_to(out.resolve())
            data = tf.extractfile(e).read()
            assert len(data) == m['members'][e.name]['bytes'] and hashlib.sha256(data).hexdigest() == m['members'][e.name]['sha256']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == data, 'Preserve differing raw evidence'
            else:
                target.write_bytes(data)
    acceptance_path = out / (phase_prefix + '_execution_20261009/strict_acceptance.json')
    acceptance = read(acceptance_path)
    assert sha(acceptance_path) == m[phase_prefix + '_strict_acceptance_sha256']
    assert acceptance['scope'] == 'VALID_ONLY_IMPLEMENTATION_REPLAY_V4'
    assert batch_source_pins_are_v4(acceptance, out, phase_prefix)
    assert acceptance['accepted_ids'] == IDS and acceptance['requested_n'] == acceptance['accepted_n'] == n
    assert not acceptance['invalid'] and not acceptance['all900_native_valid_replayed']
    batch, execution = (read(out / (phase_prefix + '_useful') / name) for name in ('batch_inputs.json', 'batch_execution.json'))
    assert batch['selected_ids'] == execution['requested_ids'] == IDS and set(execution['finished_zero_exit_ids']) == set(IDS)
    assert batch['source_before'] == execution['source_after'] and execution['source_unchanged'] and not execution['failures']
    assert len(batch['source_before']) == 44 and batch['workers'] == execution['workers'] == n
    assert sha(BASE / 'inputs/model_inventory.json') == '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
    assert sha(BASE / 'v3/inputs/storage_map.json') == 'e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949'
    inv = read(BASE / 'inputs/model_inventory.json')
    records = {r['id']: r for r in inv['records']}
    storage = {(r['id'], r['kind']): r for r in read(BASE / 'v3/inputs/storage_map.json')['records']}
    cache = BASE / 'verification_inputs/original_valid_cache.npz'
    assert sha(cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
    with np.load(cache, allow_pickle=False) as z:
        y, s = z['valid_y'], z['valid_sensitive']
    semantic = read(out / 'frozen_inputs/semantic900_inspection.json')
    assert sha(out / 'frozen_inputs/semantic900_inspection.json') == SEMANTIC_SHA and semantic['accepted_n'] == 900 and not semantic['invalid']
    semantic_rows = {r['id']: r for r in semantic['accepted']}
    checked = []
    for slot, model_id in enumerate(IDS):
        path = out / (phase_prefix + '_useful/runs') / model_id
        record, r = records[model_id], read(path / 'receipt.json')
        proof = read(path.parent / (model_id + '.worker.json'))
        cpus = list(range(16 + 8 * slot, 24 + 8 * slot))
        assert proof['status'] == r['status'] == 'NATIVE_VALID_REPLAY_PASS' and proof['id'] == r['id'] == model_id
        assert proof['v3_source_sha256'] == V4_SHA and proof['sealed_v3_runner_source_sha256'] == V3_SHA
        assert proof['semantic_inspection_sha256'] == SEMANTIC_SHA
        expected_bridge = semantic_rows[model_id]['schema_bridge']
        assert proof['v4_schema_bridge'] == expected_bridge
        assert proof['slot'] == slot and proof['allowed_cpus'] == cpus and proof['artifact_before'] == proof['artifact_after'] and proof['artifacts_unchanged']
        for kind in ('checkpoint', 'result', 'raw_job'):
            mapped = storage[model_id, kind]
            actual = proof['artifact_before'][mapped['target']]
            assert actual['sha256'] == mapped['sha256'] == record[kind]['sha256'] and actual['bytes'] == record[kind]['bytes']
        assert proof['receipt_sha256'] == sha(path / 'receipt.json') and sha(path / 'validation_predictions.npz') == r['prediction_arrays_sha256']
        assert r['weights_before'] == r['weights_after'] and not r['optimizer_created'] and not r['gradients_created']
        assert r['runtime']['device'] == 'cpu' and r['runtime']['cuda_device_count'] == 0 and r['runtime']['nice'] == 10
        assert r['valid_n'] == 19867 and r['root_reconstruction']['root_n'] == 16277
        prefix = proof['metadata_read_receipt']
        assert not prefix['full_loader_executed'] and all(p['requested_entries'] == 182637 and p['test_entries_requested_or_materialized'] == 0 for p in prefix['prefix_reads'])
        for snapshot in (r['before_resources'], r['after_resources']):
            assert all(c == cpus for c in snapshot['thread_cpu_affinities'].values())
            assert 'RUNNING' in snapshot['service']['stdout'] and not snapshot['training_queue_snapshot']['failed']
        differences, count_checks = {}, 0
        with np.load(path / 'validation_predictions.npz', allow_pickle=False) as z:
            assert len(z['valid_margins']) == len(y) == 19867 and len(z['root_margins']) == 16277
            for kind, array in [('evaluation', 'valid'), ('root', 'root')]:
                assert hashlib.sha256(z[array + '_image_ids'].tobytes()).hexdigest() == record['data_contract'][kind + '_image_ids_sha256']
            for view in ('native', 'raw', 'shared_calibration'):
                pred, fit = z['prediction_' + view], r['fits'][view]
                if fit['rule'] == 'argmax_margin_strictly_positive':
                    assert fit['thresholds'] is None
                    expected = z['valid_margins'] > 0
                else:
                    t = fit['thresholds']
                    expected = np.where(s == 0, z['valid_margins'] >= t['0'], z['valid_margins'] >= t['1'])
                assert np.array_equal(pred, expected)
                tpr = [int(((s == g) & (y == 1) & (pred == 1)).sum()) / int(((s == g) & (y == 1)).sum()) for g in (0, 1)]
                rates = [int(((s == g) & (pred == 1)).sum()) / int((s == g).sum()) for g in (0, 1)]
                direct = {'accuracy': int((pred == y).sum()) / len(y), 'aeod': abs(tpr[0] - tpr[1]), 'aspd': abs(rates[0] - rates[1])}
                differences[view] = {k: direct[k] - r['views'][view][k] for k in direct}
                assert all(d == 0 for d in differences[view].values())
                for g in (0, 1):
                    counts = {'tp': int(((s == g) & (y == 1) & (pred == 1)).sum()), 'fp': int(((s == g) & (y == 0) & (pred == 1)).sum()), 'tn': int(((s == g) & (y == 0) & (pred == 0)).sum()), 'fn': int(((s == g) & (y == 1) & (pred == 0)).sum())}
                    assert all(r['views'][view]['group_confusion_counts'][str(g)][k] == value for k, value in counts.items())
                    count_checks += 4
        native = max(abs(r['views']['native'][k] - record['prior_validation_metrics'][k]) for k in ('accuracy', 'aeod', 'aspd'))
        assert native == r['native_comparison']['max_abs_difference'] and native <= 1e-12
        checked.append({'id': model_id, 'native_max_abs_difference': native, 'independent_view_metric_differences': differences, 'confusion_count_checks': count_checks, 'receipt_sha256': sha(path / 'receipt.json'), 'prediction_arrays_sha256': r['prediction_arrays_sha256'], 'elapsed_seconds': r['elapsed_seconds'], 'cpu_user_seconds': r['cpu_user_seconds'], 'cpu_system_seconds': r['cpu_system_seconds'], 'effective_cpu_cores': r['effective_cpu_cores'], 'peak_rss_kib': r['peak_rss_kib'], 'os_threads_before_after': [r[k]['os_threads'] for k in ('before_resources', 'after_resources')], 'all_threads_bound_to_cpus': cpus})
    during, after = (read(out / (phase_prefix + '_execution_20261009') / name) for name in ('impact_during.json', 'impact_after.json'))
    a, b = ({r['id']: r['progress']['round'] for r in x['active_rounds'] if r['progress']} for x in (during, after))
    growth = [{'id': i, 'during': a[i], 'after': b[i]} for i in sorted(set(a) & set(b))]
    assert growth and all(r['after'] > r['during'] for r in growth) and not during['training_queue']['failed'] and not after['training_queue']['failed']
    report = {'status': 'PASS', 'archive_sha256': sha(archive), 'archive_members_verified': member_n, 'strict_acceptance_sha256': sha(acceptance_path), 'phase': phase, 'accepted_n': n, 'workers': n, 'shared_input_identities_unchanged': 44, 'model_result_job_hashes_unchanged': 3 * n, 'independent_metric_checks': 9 * n, 'independent_confusion_count_checks': 24 * n, 'models': checked, 'batch_wall_seconds': acceptance['wall_seconds'], 'batch_accepted_models_per_second': acceptance['models_per_second'], 'training_round_growth': growth, 'manual_round_snapshots_during_active_workers': bool(during.get('replay_processes')), 'manual_round_snapshot_scope': 'DURING_AND_AFTER_REPLAY' if during.get('replay_processes') else 'POST_EXECUTION_ONLY_NOT_IN_RUN_ROUND_GROWTH_EVIDENCE', 'worker_training_service_gpu_before_after': [{'id': i, 'before': read(out / (phase_prefix + '_useful/runs') / i / 'receipt.json')['before_resources'], 'after': read(out / (phase_prefix + '_useful/runs') / i / 'receipt.json')['after_resources']} for i in IDS], 'training_completed_during_after': [len(x['training_queue']['completed']) for x in (during, after)], 'training_failed_during_after': [0, 0], 'gpu_during_after': [x['gpu'] for x in (during, after)], 'test_label_entries_returned_as_arrays': 0, 'full_celeba_loader_executed': False, 'test_image_inference_performed': False, 'all900_native_valid_replayed': False, 'next_phase_started_at_this_verification': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN', 'claim_limit': 'Full-file SHA reads include test-containing raw bytes; no semantic test-label use. Observed training advancement does not estimate zero slowdown. These bounded useful jobs do not determine a globally optimum concurrency.', 'verifier_sha256': sha(Path(__file__))}
    destination = Path(destination)
    assert not destination.exists()
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'Supply phase, independently observed remote archive SHA, new output path'
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
