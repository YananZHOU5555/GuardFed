"""Bounded acceptance of label-read boundaries, inventory structure, root binding and fail-stop tolerance."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
import replay

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def reject(name, function, rejected):
    try:
        function()
    except (ValueError, AssertionError) as exc:
        rejected.append({'name': name, 'error': str(exc)})
    else:
        raise AssertionError('Failed to reject: ' + name)


def verify_remote_receipts():
    manifest = replay.read(HERE / 'remote_receipts_inventory.json')
    archive = HERE / manifest['archive']
    replay.require(replay.digest(archive) == manifest['sha256'] and archive.stat().st_size == manifest['bytes'], 'Downloaded archive identity failed')
    out = HERE / 'remote_receipts'
    replay.require(not out.exists(), 'Preserve already extracted raw receipts')
    out.mkdir()
    with tarfile.open(archive, 'r:gz') as tar:
        entries = tar.getmembers()
        replay.require(len(entries) == len({e.name for e in entries}) == len(manifest['members']), 'Duplicate/missing receipt archive members')
        replay.require({e.name for e in entries} == set(manifest['members']), 'Archive member inventory differs')
        for e in entries:
            target = (out / e.name).resolve()
            replay.require(e.isfile() and target.is_relative_to(out.resolve()), 'Unsafe receipt archive path/type')
            payload = tar.extractfile(e).read()
            expected = manifest['members'][e.name]
            replay.require(len(payload) == expected['bytes'] and replay.hashlib.sha256(payload).hexdigest() == expected['sha256'], 'Receipt member SHA failed: ' + e.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    run = replay.read(out / 'canary2_cpu_v2/run_receipt.json')
    before = replay.read(out / 'canary2_cpu_v2/input_sha256_before.json')
    after = replay.read(out / 'canary2_cpu_v2/input_sha256_after.json')
    replay.require(run['status'] == 'SELECTED_NATIVE_VALID_REPLAY_PASS' and run['accepted_ids'] == replay.CANARY_IDS, 'Actual two-canary run did not pass')
    replay.require(before == after and run['all_input_sha_unchanged'] and not run['all900_native_valid_replayed'], 'Input drift or all900 completion misclaim')
    replay.require(not run['final_protocol_frozen'] and not run['test_labels_accessed'], 'Final/test scope violated')
    replay.require(replay.digest(out / 'replay.py') == replay.digest(HERE / 'replay.py'), 'Measured replay source differs from delivered bytes')
    cache_path = HERE / 'verification_inputs/original_valid_cache.npz'
    cache_sha = '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
    replay.require(replay.digest(cache_path) == cache_sha, 'Previously accepted valid-cache bytes changed')
    with replay.np.load(cache_path, allow_pickle=False) as z:
        y, s = z['valid_y'], z['valid_sensitive']
    replay.require(len(y) == 19867 and replay.support(s, y) == {'0|0': 5252, '0|1': 6157, '1|0': 5013, '1|1': 3445}, 'Independent accepted valid label support differs')
    checked = []
    for model_id in replay.CANARY_IDS:
        path = out / 'canary2_cpu_v2' / model_id
        receipt = replay.read(path / 'receipt.json')
        replay.require(receipt['native_comparison']['accepted'] and receipt['native_comparison']['max_abs_difference'] == 0, 'Native replay was not exact')
        replay.require(receipt['weights_before'] == receipt['weights_after'] and not receipt['optimizer_created'] and not receipt['gradients_created'], 'Model state or inference-only contract failed')
        replay.require(replay.digest(path / 'validation_predictions.npz') == receipt['prediction_arrays_sha256'], 'Saved prediction bytes changed')
        with replay.np.load(path / 'validation_predictions.npz', allow_pickle=False) as z:
            replay.require(replay.array_sha(z['valid_image_ids']) == replay.VALID_IDS_SHA and len(z['valid_margins']) == 19867, 'Actual valid sample order differs')
            replay.require(replay.array_sha(z['root_image_ids']) == receipt['root_reconstruction']['root_image_ids_sha256'] and len(z['root_margins']) == 16277, 'Actual root order/support differs')
            deltas = {}
            for view in replay.VIEWS:
                pred = z['prediction_' + view]
                if view == 'raw':
                    reconstructed = z['valid_margins'] > 0
                else:
                    t = receipt['fits'][view]['thresholds']
                    reconstructed = replay.np.where(s == 0, z['valid_margins'] >= t['0'], z['valid_margins'] >= t['1'])
                replay.require(replay.np.array_equal(pred, reconstructed), 'Saved predictions contradict declared margin/tie rule')
                # Independent direct counts from accepted validation labels; do not call evaluator scorer.
                tpr = [int(((s == g) & (y == 1) & (pred == 1)).sum()) / int(((s == g) & (y == 1)).sum()) for g in (0, 1)]
                rates = [int(((s == g) & (pred == 1)).sum()) / int((s == g).sum()) for g in (0, 1)]
                direct = {'accuracy': int((pred == y).sum()) / len(y), 'aeod': abs(tpr[0] - tpr[1]), 'aspd': abs(rates[0] - rates[1])}
                deltas[view] = {k: direct[k] - receipt['views'][view][k] for k in direct}
                replay.require(all(v == 0 for v in deltas[view].values()), 'Independent count/scorer mismatch')
                for g in (0, 1):
                    expected = {'tp': int(((s == g) & (y == 1) & (pred == 1)).sum()),
                                'fp': int(((s == g) & (y == 0) & (pred == 1)).sum()),
                                'tn': int(((s == g) & (y == 0) & (pred == 0)).sum()),
                                'fn': int(((s == g) & (y == 1) & (pred == 0)).sum())}
                    replay.require(all(receipt['views'][view]['groups'][str(g)][k] == v for k, v in expected.items()), 'Stored confusion counts differ')
        for snapshot in (receipt['before_resources'], receipt['after_resources']):
            replay.require(all(cpus == list(range(16, 24)) for cpus in snapshot['thread_cpu_affinities'].values()), 'A replay thread escaped its coordinated CPUs')
            replay.require('RUNNING' in snapshot['service']['stdout'] and not snapshot['training_queue_snapshot']['failed'], 'Training health failed during canary')
        checked.append({'id': model_id, 'native_max_abs_difference': 0, 'independent_view_metric_differences': deltas,
                        'elapsed_seconds': receipt['elapsed_seconds'], 'effective_cpu_cores': receipt['effective_cpu_cores'],
                        'peak_rss_kib': receipt['peak_rss_kib'], 'receipt_sha256': replay.digest(path / 'receipt.json'),
                        'prediction_arrays_sha256': receipt['prediction_arrays_sha256']})
    during, final = replay.read(out / 'impact_during.json'), replay.read(out / 'impact_after.json')
    growth = []
    for a, b in zip(during['active_rounds'], final['active_rounds']):
        replay.require(a['id'] == b['id'] and b['progress']['round'] >= a['progress']['round'], 'Training round regression or active identity change')
        growth.append({'id': a['id'], 'during': a['progress']['round'], 'after': b['progress']['round']})
    report = {'status': 'PASS', 'archive_sha256': manifest['sha256'], 'members_verified': len(manifest['members']),
              'run_receipt_sha256': replay.digest(out / 'canary2_cpu_v2/run_receipt.json'), 'unchanged_input_count': len(before),
              'canaries': checked, 'training_round_growth': growth,
              'independent_valid_label_cache': {'sha256': cache_sha, 'source_archive_sha256': 'f7528394e8888163e157323654ad9cee31f4d32a1b65a0ef57cb6894382e352e',
                  'source_member': 'results/revision_20260928/celeba_shared_calibration_v1/runs/FedAvg_IID_Benign_seed91001/margins.npz',
                  'use': 'Previously sealed accepted cache; local independent prediction scoring only, not new image inference or revalidation of all700'},
              'all900_native_valid_replayed': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN', 'test_labels_accessed': False,
              'impact_claim': 'Observed formal service RUNNING, all eight rounds advancing, no failed queue entries; no controlled estimate of zero training slowdown'}
    replay.save(HERE / 'offserver_verification.json', report)
    return report


def main():
    results, rejected = [], []
    inv = replay.read(HERE / 'inputs/model_inventory.json')
    replay.require(replay.digest(HERE / 'inputs/model_inventory.json') == replay.INVENTORY_SHA, 'Original inventory bytes changed')
    replay.validate_inventory(inv)
    for label, change in [
        ('duplicate model ID', lambda x: x['records'][0].update(id=x['records'][1]['id'])),
        ('missing seed cell', lambda x: x['records'].pop()),
        ('test endpoint', lambda x: x['records'][0]['config'].update(celeba_evaluation_split='test')),
        ('source drift', lambda x: x['records'][0]['source_hashes'].update({'scripts/reproduce_paper_tables.py': '0' * 64})),
        ('terminal valid support drift', lambda x: x['records'][0].update(original_n_eval=19962)),
    ]:
        bad = copy.deepcopy(inv)
        change(bad)
        if label == 'test endpoint':
            bad['records'][0]['config_canonical_sha256'] = replay.canonical(bad['records'][0]['config'])
        reject(label, lambda: replay.validate_inventory(bad), rejected)
    results.append({'name': 'independent900_grid_and_identity', 'passed': True, 'records': 900})

    # Test values beyond the allowed prefix are deliberately nonbinary. A full
    # array label read would encounter them; successful prefix materialization does not.
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        tmp = Path(tmp)
        npz = tmp / 'split_boundary.npz'
        replay.np.savez(npz, Smiling=replay.np.array([0, 1, 1, 0, 91, 92], dtype='i8'))
        values, receipt = replay.read_prefix(npz, 'Smiling', total=6, count=4)
        replay.require(values.tolist() == [0, 1, 1, 0] and receipt['test_entries_requested_or_materialized'] == 0, 'Prefix boundary failed')
        reject('request whole label array including test', lambda: replay.read_prefix(npz, 'Smiling', 6, 6), rejected)
        reject('request extends into nonbinary test sentinel', lambda: replay.read_prefix(npz, 'Smiling', 6, 5), rejected)
        npz_object = tmp / 'unsafe.npz'
        replay.np.savez(npz_object, Smiling=replay.np.array([0, 1, None, None], dtype=object))
        reject('object-pickle label payload', lambda: replay.read_prefix(npz_object, 'Smiling', 4, 2), rejected)
        pinned = tmp / 'pinned.txt'
        pinned.write_bytes(b'accepted raw input')
        pins = {pinned: replay.digest(pinned)}
        replay.hash_pins(pins)
        pinned.write_bytes(b'tampered raw input')
        reject('before/after raw byte drift', lambda: replay.hash_pins(pins), rejected)
        reject('undeclared parent-relative path', lambda: replay.inside(tmp, '../elsewhere'), rejected)
        reject('absolute path injection', lambda: replay.inside(tmp, '/elsewhere'), rejected)
    results.append({'name': 'label_prefix_and_input_tamper_boundaries', 'passed': True})

    reference = REPO / 'docs/server_deployment_20260923/training_20260923/celeba_joint_partition_20261009/train_only_audit_metadata.npz'
    core_path = REPO / 'tmp/revision-publish-20260928/scripts/reproduce_paper_tables.py'
    replay.require(replay.digest(core_path) == replay.CORE_SHA, 'Archived core identity mismatch')
    core = replay.load('valid_selfcheck_frozen_core', core_path)
    with replay.np.load(reference, allow_pickle=False) as z:
        ids, y, s = z['image_id'], z['train_Smiling'], z['train_Male']
    # Only original clean train labels participate in this offline root test.
    for model_id in replay.CANARY_IDS:
        record = next(r for r in inv['records'] if r['id'] == model_id)
        cfg = core.ExperimentConfig(**record['config'])
        root, _, _, root_receipt = replay.rebuild_root(core, cfg, record, ids, y, s)
        replay.require(root_receipt['root_n'] == 16277, 'Unexpected clean root count')
        bad = copy.deepcopy(record)
        bad['data_contract']['root_image_ids_sha256'] = '0' * 64
        reject('root ID tamper ' + model_id, lambda: replay.rebuild_root(core, cfg, bad, ids, y, s), rejected)
    results.append({'name': 'actual_train_metadata_original_root_and_client_partition', 'passed': True,
                    'train_metadata_sha256': replay.digest(reference), 'core_sha256': replay.digest(core_path),
                    'canary_root_n': 16277, 'image_inference_performed': False})

    expected = {'accuracy': .875, 'aeod': .125, 'aspd': .25}
    replay.require(replay.check_native(expected, expected)['accepted'], 'Exact native match rejected')
    changed = {**expected, 'accuracy': expected['accuracy'] + 2e-12}
    comparison = replay.check_native(changed, expected)
    replay.require(not comparison['accepted'] and comparison['tolerance'] == 1e-12, 'Native mismatch tolerance was relaxed')
    reject('nonfinite native metric', lambda: replay.check_native({**expected, 'aeod': float('nan')}, expected), rejected)
    results.append({'name': 'strict_native_tolerance_and_nonfinite_rejection', 'passed': True,
                    'measured_mismatch': comparison['max_abs_difference']})
    report = {'status': 'PASS', 'groups': results, 'rejected': rejected,
              'replay_source_sha256': replay.digest(replay.__file__), 'selfcheck_sha256': replay.digest(__file__),
              'test_labels_accessed': False, 'new_training': False, 'real_image_replay': 'Separate remote receipts required'}
    replay.save(HERE / 'selfcheck.json', report)
    print(json.dumps({'status': report['status'], 'groups': len(results), 'rejection_cases': len(rejected)}))


if __name__ == '__main__':
    if '--receipts' in sys.argv:
        report = verify_remote_receipts()
        print(json.dumps({'status': report['status'], 'members_verified': report['members_verified'], 'canaries': len(report['canaries'])}))
    else:
        main()
