"""Offserver chunk verification and descriptive tables from accepted saved arrays."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[2]
COLLECTOR_SHA = '19066f63c341b9ee23b7c6f491802cfdde0c1c0c833c2fe64724a16de9bb2234'
INVENTORY_SHA = '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
CACHE_SHA = '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
SEMANTIC_SHA = 'cc93d94d69478a4ee190abff76da4fb185791ece7e4e1d2ee75360482ead80b4'
V4_SHA = '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e'
V3_SHA = 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'
VIEWS = ('native', 'raw', 'shared_calibration')
METRICS = ('accuracy', 'aeod', 'aspd')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def collector():
    source = BASE / 'v4/execution_20261009/collect_valid_replay.py'
    require(sha(source) == COLLECTOR_SHA, 'Reviewed mixed-source collector changed')
    spec = importlib.util.spec_from_file_location('sealed_valid_replay_collector', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direct_checks(receipt, arrays, y, sensitive):
    import numpy as np
    differences, count_checks = {}, 0
    require(len(arrays['valid_margins']) == len(y) == 19867 and len(arrays['root_margins']) == 16277, 'Missing root/valid margins')
    for view in VIEWS:
        pred, fit = arrays['prediction_' + view], receipt['fits'][view]
        require(pred.shape == y.shape and np.isin(pred, [0, 1]).all(), 'Invalid prediction support')
        if fit['rule'] == 'argmax_margin_strictly_positive':
            require(fit['thresholds'] is None, 'Raw/native identity view has thresholds')
            expected = arrays['valid_margins'] > 0
        else:
            require(fit['rule'] == 'group_margin_greater_equal', 'Unknown calibrated prediction rule')
            t = fit['thresholds']
            require(set(t) == {'0', '1'} and all(np.isfinite(x) for x in t.values()), 'Invalid calibrated threshold')
            expected = np.where(sensitive == 0, arrays['valid_margins'] >= t['0'], arrays['valid_margins'] >= t['1'])
        require(np.array_equal(pred, expected), 'Saved predictions violate strict-zero/greater-equal tie rules')
        tpr, rates = [], []
        for group in (0, 1):
            mask = sensitive == group
            require(int(mask.sum()) > 0 and int((mask & (y == 1)).sum()) > 0, 'Undefined validation denominator')
            counts = {'tp': int((mask & (y == 1) & (pred == 1)).sum()), 'fp': int((mask & (y == 0) & (pred == 1)).sum()), 'tn': int((mask & (y == 0) & (pred == 0)).sum()), 'fn': int((mask & (y == 1) & (pred == 0)).sum())}
            stored = receipt['views'][view]['group_confusion_counts'][str(group)]
            require(all(stored[k] == value for k, value in counts.items()), 'Saved confusion counts disagree with arrays')
            require(stored['n'] == int(mask.sum()) and stored['positives'] == counts['tp'] + counts['fn'] and stored['negatives'] == counts['fp'] + counts['tn'], 'Sensitive group support differs')
            tpr.append(counts['tp'] / (counts['tp'] + counts['fn']))
            rates.append((counts['tp'] + counts['fp']) / int(mask.sum()))
            count_checks += 4
        actual = {'accuracy': int((pred == y).sum()) / len(y), 'aeod': abs(tpr[0] - tpr[1]), 'aspd': abs(rates[0] - rates[1])}
        differences[view] = {k: actual[k] - receipt['views'][view][k] for k in actual}
        require(all(d == 0 for d in differences[view].values()), 'Direct three-view metric mismatch')
    return differences, count_checks


def check_array_ids(record, arrays):
    for kind, array in [('evaluation', 'valid'), ('root', 'root')]:
        require(hashlib.sha256(arrays[array + '_image_ids'].tobytes()).hexdigest() == record['data_contract'][kind + '_image_ids_sha256'], 'Root/valid image IDs changed')


def verify_chunk(stage, expected_archive_sha, manifest_path, expected_manifest_sha, index):
    import numpy as np
    c = collector()
    require(sha(manifest_path) == expected_manifest_sha, 'Reviewed manifest SHA changed')
    m = read(manifest_path)
    audit_path = '/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/remaining872_prepared_v2_20261009/audit_remaining.py'
    require(sha(__file__) == m['input_sha256'][audit_path], 'Independent verifier differs from the reviewed source')
    ids = m['chunks'][index]['ids']
    require(m['chunks'][index]['index'] == index and len(ids) == len(set(ids)) <= 11, 'Invalid frozen chunk')
    archive, data = c.verified_archive(stage / 'offserver_verification.json', {'archive_sha256': expected_archive_sha})
    j = lambda name: json.loads(data[name])
    require(hashlib.sha256(data['execution/manifest.json']).hexdigest() == expected_manifest_sha, 'Archived dispatch input manifest changed')
    for name, expected in [('sealed_sources/replay_v4.py', V4_SHA), ('sealed_sources/replay_v3.py', V3_SHA), ('frozen_inputs/semantic900_inspection.json', SEMANTIC_SHA)]:
        require(hashlib.sha256(data[name]).hexdigest() == expected, 'Archived scientific source changed')
    require(hashlib.sha256(data['execution_sources/audit_remaining.py']).hexdigest() == sha(__file__), 'Archived verifier source differs')
    strict, batch, execution = (j(name) for name in ('execution/strict_acceptance.json', 'batch/batch_inputs.json', 'batch/batch_execution.json'))
    require(strict['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED' and strict['accepted_ids'] == ids and strict['requested_n'] == strict['accepted_n'] == len(ids) and not strict['invalid'], 'Partial/invalid chunk cannot masquerade as complete')
    require(batch['scope'] == 'VALID_ONLY_IMPLEMENTATION_REPLAY_V4' and batch['v3_source_sha256'] == V4_SHA and batch['selected_ids'] == execution['requested_ids'] == ids, 'Batch/source identity drift')
    require(set(execution['finished_zero_exit_ids']) == set(ids) and not execution['failures'] and not execution['stopped_after_failure'] and execution['source_unchanged'] and batch['source_before'] == execution['source_after'], 'Execution/source failed')
    require(len(batch['source_before']) == 44 and batch['workers'] == execution['workers'] == min(11, len(ids)), 'Missing scientific sources or worker bound')
    expected_science = dict(m['scientific_source_sha256'])
    expected_science['/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/semantic900_inspection.json'] = SEMANTIC_SHA
    require({name: r['sha256'] for name, r in batch['source_before'].items()} == expected_science, 'Full scientific source/data SHA set differs from the reviewed manifest')
    inventory = BASE / 'inputs/model_inventory.json'
    require(sha(inventory) == INVENTORY_SHA, 'Original inventory changed')
    records = {r['id']: r for r in read(inventory)['records']}
    storage_path = BASE / 'v3/inputs/storage_map.json'
    require(sha(storage_path) == 'e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949', 'Reviewed storage map changed')
    storage = {(r['id'], r['kind']): r for r in read(storage_path)['records']}
    require(j('execution/chunk_config_inventory_records.json') == [records[i] for i in ids], 'Archived original configs differ')
    cache = BASE / 'verification_inputs/original_valid_cache.npz'
    require(sha(cache) == CACHE_SHA, 'Offserver original valid-label cache changed')
    with np.load(cache, allow_pickle=False) as z:
        y, sensitive = z['valid_y'], z['valid_sensitive']
    semantic_rows = {r['id']: r for r in j('frozen_inputs/semantic900_inspection.json')['accepted']}
    rows = []
    for slot, model_id in enumerate(ids):
        root = 'batch/runs/' + model_id
        r, worker = j(root + '/receipt.json'), j(root + '.worker.json')
        record = records[model_id]
        cpus = batch['inherited_allowed_cpus'][16 + 8 * slot:24 + 8 * slot]
        require(worker['id'] == r['id'] == model_id and worker['status'] == r['status'] == 'NATIVE_VALID_REPLAY_PASS', 'Missing actual accepted model receipt')
        require(worker['slot'] == slot and worker['allowed_cpus'] == cpus and len(cpus) == 8, 'Worker slot changed')
        require(worker['artifact_before'] == worker['artifact_after'] and worker['artifacts_unchanged'], 'Artifact bytes changed')
        require(worker['batch_receipt_sha256'] == hashlib.sha256(data['batch/batch_inputs.json']).hexdigest() and worker['storage_map_sha256'] == sha(storage_path), 'Worker belongs to another batch/map')
        for kind in ('checkpoint', 'result', 'raw_job'):
            mapped = storage[model_id, kind]
            actual = worker['artifact_before'][mapped['target']]
            require(actual['sha256'] == mapped['sha256'] == record[kind]['sha256'] and actual['bytes'] == record[kind]['bytes'], 'Model/result/rawjob does not match original inventory/map')
        require(worker['v3_source_sha256'] == V4_SHA and worker['sealed_v3_runner_source_sha256'] == V3_SHA and worker['semantic_inspection_sha256'] == SEMANTIC_SHA and worker['v4_schema_bridge'] == semantic_rows[model_id]['schema_bridge'], 'Original schema/source gate changed')
        require(worker['receipt_sha256'] == hashlib.sha256(data[root + '/receipt.json']).hexdigest() and r['prediction_arrays_sha256'] == hashlib.sha256(data[root + '/validation_predictions.npz']).hexdigest(), 'Receipt/prediction SHA changed')
        require(r['model_inventory_record_sha256'] == canonical(record) and r['weights_before'] == r['weights_after'] and not r['optimizer_created'] and not r['gradients_created'], 'Original record or weights changed')
        require(not r['test_inference_performed'] and not worker['metadata_read_receipt']['full_loader_executed'] and all(p['test_entries_requested_or_materialized'] == 0 and p['requested_entries'] == 182637 for p in worker['metadata_read_receipt']['prefix_reads']), 'Semantic test access or loader scope changed')
        require(r['runtime']['device'] == 'cpu' and r['runtime']['cuda_device_count'] == 0 and r['runtime']['nice'] == 10 and r['runtime']['torch_threads'] == 8 and r['runtime']['interop_threads'] == 1 and r['runtime']['loader_workers'] == 0, 'Runtime device/priority/thread contract changed')
        for snapshot in (r['before_resources'], r['after_resources']):
            require(all(c == cpus for c in snapshot['thread_cpu_affinities'].values()), 'Worker thread escaped its slot')
            require(not snapshot['training_queue_snapshot']['failed'], 'Observed training failure')
        with np.load(io.BytesIO(data[root + '/validation_predictions.npz']), allow_pickle=False) as z:
            check_array_ids(record, z)
            differences, count_n = direct_checks(r, z, y, sensitive)
        native = max(abs(r['views']['native'][k] - record['prior_validation_metrics'][k]) for k in METRICS)
        require(native == r['native_comparison']['max_abs_difference'] <= 1e-12, 'Original native metrics disagree')
        rows.append({'id': model_id, 'native_max_abs_difference': native, 'independent_view_metric_differences': differences, 'confusion_count_checks': count_n, 'receipt_sha256': hashlib.sha256(data[root + '/receipt.json']).hexdigest(), 'prediction_arrays_sha256': r['prediction_arrays_sha256']})
    report = {'status': 'PASS', 'scope': 'OFFSERVER_VALID_REPLAY_COMPLETE_CHUNK', 'chunk_index': index, 'archive_sha256': sha(archive), 'archive_members_verified': len(data), 'strict_acceptance_sha256': hashlib.sha256(data['execution/strict_acceptance.json']).hexdigest(), 'accepted_n': len(ids), 'workers': min(11, len(ids)), 'models': rows, 'independent_metric_checks': 9 * len(ids), 'independent_confusion_count_checks': 24 * len(ids), 'manifest_sha256': expected_manifest_sha, 'verifier_source_sha256': sha(__file__), 'all900_native_valid_replayed': False, 'test_image_inference': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN'}
    candidate = stage / 'offserver_candidate_not_final.json'
    save(candidate, report)
    c.collect(inventory, [('v4', str(candidate), sha(candidate))], [])
    destination = stage / 'offserver_verification.json'
    require(not destination.exists(), 'Preserve prior offserver proof')
    candidate.rename(destination)
    print(json.dumps({'status': 'PASS', 'accepted_n': len(ids), 'proof_sha256': sha(destination), 'archive_sha256': sha(archive)}))


def summarize(spec_path, output):
    """Only actual complete900; refuse to render missing scientific cells."""
    c, spec = collector(), read(spec_path)
    proofs = [(v, str(BASE / p), h) for v, p, h in spec['proofs']]
    failures = [(str(BASE / p), h) for p, h in spec.get('failure_proofs', [])]
    collection = c.collect(BASE / 'inputs/model_inventory.json', proofs, failures)
    require(collection['accepted_n'] == 900 and collection['all900_native_valid_replayed'] and collection['all900_three_views_valid_replayed'], 'All900 unique actual three-view acceptances required; no missing-value tables')
    output.mkdir()
    save(output / 'unique900_collection.json', collection)
    receipts = {}
    for version, path, h in proofs:
        p = Path(path)
        _, data = c.verified_archive(p, read(p))
        for name, value in data.items():
            if name.endswith('/receipt.json'):
                r = json.loads(value)
                key = (r['method'], r['distribution'], r['attack'], r['seed'])
                require(key not in receipts, 'Duplicate scientific cell while summarizing')
                receipts[key] = r
    cohorts = read(BASE / 'inputs/protocol.json')['descriptive_statistics']
    cohort_names = ('primary_seed_cohort', 'nonselection_nine', 'historical_matching_six')
    methods = sorted({key[0] for key in receipts})
    attacks = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
    tables, cross, paired = [], [], []
    def stats(values):
        return {'n': len(values), 'mean': statistics.mean(values), 'sample_sd_ddof1': statistics.stdev(values) if len(values) > 1 else None}
    for cohort in cohort_names:
        seeds = cohorts[cohort]
        for method in methods:
            for dist in ('IID', 'non-IID'):
                for view in VIEWS:
                    for attack in attacks:
                        row = {'cohort': cohort, 'method': method, 'distribution': dist, 'attack': attack, 'view': view, 'n': len(seeds)}
                        for metric in METRICS:
                            scale = 100 if metric == 'accuracy' else 1
                            scored = stats([scale * receipts[method, dist, attack, seed]['views'][view][metric] for seed in seeds])
                            row[metric + ('_percent' if metric == 'accuracy' else '') + '_mean'] = scored['mean']
                            row[metric + ('_percent' if metric == 'accuracy' else '') + '_sample_sd_ddof1'] = scored['sample_sd_ddof1']
                        tables.append(row)
                    for metric in METRICS:
                        scale = 100 if metric == 'accuracy' else 1
                        values = [statistics.mean(scale * receipts[method, dist, attack, seed]['views'][view][metric] for attack in attacks) for seed in seeds]
                        cross.append({'cohort': cohort, 'method': method, 'distribution': dist, 'view': view, 'metric': metric, 'unit': 'percentage_points' if metric == 'accuracy' else 'absolute_rate_gap', **stats(values)})
    for (method, dist, attack, seed), r in sorted(receipts.items()):
        full = receipts['GuardFed-AD2+', dist, attack, seed]
        for view in VIEWS:
            row = {'method': method, 'distribution': dist, 'attack': attack, 'seed': seed, 'view': view}
            for metric in METRICS:
                row[metric + '_minus_full'] = (100 if metric == 'accuracy' else 1) * (r['views'][view][metric] - full['views'][view][metric])
            paired.append(row)
    for name, rows in [('per_scenario', tables), ('cross_scenario_seed_first', cross), ('per_seed_paired_full_differences', paired)]:
        save(output / (name + '.json'), rows)
        with (output / (name + '.csv')).open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    save(output / 'summary_identity.json', {'unique_n': 900, 'cohorts': {k: cohorts[k] for k in cohort_names}, 'source_versions': collection['source_versions'], 'collector_source_sha256': COLLECTOR_SHA, 'summary_source_sha256': sha(__file__), 'spec_sha256': sha(spec_path), 'validation_only': True, 'old900_values_overwritten': False, 'mechanism800_new_controls_included': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN', 'runtime_disclosure': 'Original training environments vary, including cu128/cu130; new replay is isolated CPU cu128. Seed91001 participated in recipe selection;9/6 cohorts are explicit descriptive sensitivity views.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    v = s.add_parser('verify-chunk')
    v.add_argument('--stage', type=Path, required=True)
    v.add_argument('--archive-sha256', required=True)
    v.add_argument('--manifest', type=Path, required=True)
    v.add_argument('--manifest-sha256', required=True)
    v.add_argument('--chunk-index', type=int, required=True)
    t = s.add_parser('summarize900')
    t.add_argument('--collection-inputs', type=Path, required=True)
    t.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'verify-chunk':
        verify_chunk(a.stage, a.archive_sha256, a.manifest, a.manifest_sha256, a.chunk_index)
    else:
        summarize(a.collection_inputs, a.output)


if __name__ == '__main__':
    main()
