"""Collect SHA-reviewed v2/v3/v4 actual receipts by original inventory identity."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

INVENTORY_SHA = '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
SOURCES = {'v2': ('replay.py', '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'), 'v3': ('sealed_sources/replay_v3.py', 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'), 'v4': ('sealed_sources/replay_v4.py', '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e')}
VALID_IDS_SHA = '64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def require(ok, message):
    if not ok:
        raise ValueError(message)

def aliases(records):
    result = {}
    for r in records:
        for token in {r['id'], r['raw_job']['member'], PurePosixPath(r['raw_job']['member']).stem, r['original_remote_output'], PurePosixPath(r['original_remote_output']).name}:
            require(token not in result or result[token] == r['id'], 'Ambiguous original job/inventory alias')
            result[token] = r['id']
    return result

def verified_archive(proof_path, proof, failure=False):
    name = 'failure_archive_inventory.json' if failure else ('remote_receipts_inventory.json' if 'canaries' in proof else 'remote_archive_inventory.json')
    manifest = read(proof_path.parent / name)
    archive = proof_path.parent / manifest['archive']
    require(sha(archive) == manifest['sha256'] == proof['archive_sha256'], 'Archive differs from reviewed proof')
    data = {}
    with tarfile.open(archive, 'r:gz') as tf:
        entries = tf.getmembers()
        require(len(entries) == len({e.name for e in entries}) == len(manifest['members']), 'Duplicate/missing archive members')
        require({e.name for e in entries} == set(manifest['members']), 'Archive/member manifest differs')
        for e in entries:
            p = PurePosixPath(e.name)
            require(e.isfile() and not p.is_absolute() and '..' not in p.parts, 'Unsafe archive member')
            b = tf.extractfile(e).read()
            require(len(b) == manifest['members'][e.name]['bytes'] and hashlib.sha256(b).hexdigest() == manifest['members'][e.name]['sha256'], 'Archive member SHA mismatch')
            data[e.name] = b
    return archive, data

def collect(inventory, proof_specs, failure_specs):
    require(sha(inventory) == INVENTORY_SHA, 'Original inventory bytes changed')
    records = read(inventory)['records']
    indexed, alias = {r['id']: r for r in records}, aliases(records)
    require(len(indexed) == len(records) == 900, 'Expected exactly900 original inventory records')
    def key(token):
        require(token in alias, 'Foreign original job/inventory alias')
        return alias[token]
    accepted, origins, failures = {}, [], []
    for version, filename, trusted_sha in proof_specs:
        require(version in SOURCES, 'Explicit known source version required')
        path = Path(filename)
        require(sha(path) == trusted_sha, 'Offserver proof differs from externally reviewed SHA')
        proof = read(path)
        require(proof['status'] == 'PASS', 'Prepared/static/failed evidence is not accepted replay')
        archive, data = verified_archive(path, proof)
        source_member, source_sha = SOURCES[version]
        require(hashlib.sha256(data[source_member]).hexdigest() == source_sha, 'Proof source version differs from declared version')
        proof_rows = proof.get('canaries', proof.get('models', [proof]))
        reported = {key(r['id']): r for r in proof_rows}
        require(len(reported) == len(proof_rows), 'Duplicate aliases within one proof')
        receipts = {n: json.loads(b) for n, b in data.items() if n.endswith('/receipt.json')}
        require(len(receipts) == len(reported), 'Proof model count differs from actual archived receipts')
        if version != 'v2':
            strict_names = [n for n in data if n.endswith('/strict_acceptance.json')]
            require(len(strict_names) == 1, 'Missing unique strict acceptance')
            strict = json.loads(data[strict_names[0]])
            require(strict['scope'] == 'VALID_ONLY_IMPLEMENTATION_REPLAY_' + version.upper() and strict['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED', 'Wrong source-version strict acceptance')
            require(strict['accepted_n'] == strict['requested_n'] == len(reported) and not strict['invalid'], 'Incomplete selected acceptance')
            require({key(i) for i in strict['accepted_ids']} == set(reported), 'Strict acceptance/actual proof IDs differ')
            require(hashlib.sha256(data[strict_names[0]]).hexdigest() == proof['strict_acceptance_sha256'], 'Strict acceptance SHA differs from reviewed proof')
        else:
            names = [n for n in data if n.endswith('/run_receipt.json')]
            require(len(names) == 1, 'Missing v2 run receipt')
            run = json.loads(data[names[0]])
            require(run['status'] == 'SELECTED_NATIVE_VALID_REPLAY_PASS' and {key(i) for i in run['accepted_ids']} == set(reported) and run['all_input_sha_unchanged'], 'v2 run did not accept exact canaries')
            require(hashlib.sha256(data[names[0]]).hexdigest() == proof['run_receipt_sha256'], 'v2 run receipt SHA mismatch')
        occurrence = []
        for member, r in receipts.items():
            model_id = key(r['id'])
            require(model_id in reported and model_id not in accepted, 'Duplicate canonical accepted checkpoint across proofs/aliases')
            record, checked = indexed[model_id], reported[model_id]
            require(r['status'] == 'NATIVE_VALID_REPLAY_PASS' and r['native_comparison']['accepted'] and r['native_comparison']['tolerance'] == 1e-12, 'Native acceptance incomplete')
            require(r['native_comparison']['max_abs_difference'] == checked['native_max_abs_difference'] <= 1e-12, 'Native metric mismatch')
            require(hashlib.sha256(data[member]).hexdigest() == checked['receipt_sha256'], 'Actual receipt SHA differs from reviewed proof')
            arrays = member.removesuffix('receipt.json') + 'validation_predictions.npz'
            require(hashlib.sha256(data[arrays]).hexdigest() == checked['prediction_arrays_sha256'] == r['prediction_arrays_sha256'], 'Actual three-view arrays differ from reviewed proof')
            require(r['model_inventory_record_sha256'] == hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(), 'Original inventory record identity differs')
            for field, kind in [('checkpoint_sha256', 'checkpoint'), ('original_result_sha256', 'result'), ('original_job_sha256', 'raw_job')]:
                require(r[field] == record[kind]['sha256'], 'Checkpoint/result/rawjob identity mismatch')
            require(r['method'] == record['method'] and (r['seed'], r['distribution'], r['attack']) == (record['seed'], record['distribution'], record['attack']), 'Scientific cell identity differs')
            require(r['valid_n'] == 19867 and r['valid_image_ids_sha256'] == VALID_IDS_SHA and r['root_reconstruction']['root_n'] == 16277, 'Root/valid support differs')
            require(r['root_reconstruction']['root_image_ids_sha256'] == record['data_contract']['root_image_ids_sha256'] and r['weights_before'] == r['weights_after'], 'Root or unchanged-weight identity differs')
            require(not r['optimizer_created'] and not r['gradients_created'] and not r['test_inference_performed'], 'Inference scope differs')
            require(set(r['views']) == set(r['fits']) == {'native', 'raw', 'shared_calibration'}, 'Missing actual three-view evaluation')
            require(set(checked['independent_view_metric_differences']) == set(r['views']), 'Three-view independent verification missing')
            for view, metrics in r['views'].items():
                require(all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in ('accuracy', 'aeod', 'aspd')), 'Undefined three-view metric')
                require(all(abs(d) <= 1e-12 for d in checked['independent_view_metric_differences'][view].values()), 'Independent three-view metric verification failed')
                counts = metrics['group_confusion_counts']
                require(sum(counts[g][k] for g in ('0', '1') for k in ('tp', 'fp', 'tn', 'fn')) == 19867, 'Incomplete same-checkpoint confusion support')
            accepted[model_id] = {'id': model_id, 'original_job_id': PurePosixPath(record['raw_job']['member']).stem, 'scientific_cell': [record[k] for k in ('method', 'distribution', 'attack', 'seed')], 'source_version': version, 'source_sha256': source_sha, 'proof_sha256': trusted_sha, 'archive_sha256': proof['archive_sha256'], 'receipt_member': member, 'receipt_sha256': checked['receipt_sha256'], 'prediction_arrays_sha256': r['prediction_arrays_sha256'], 'checkpoint_sha256': record['checkpoint']['sha256'], 'original_result_sha256': record['result']['sha256'], 'original_rawjob_sha256': record['raw_job']['sha256'], 'native_max_abs_difference': checked['native_max_abs_difference'], 'actual_three_views_verified': True}
            occurrence.append(model_id)
        origins.append({'source_version': version, 'source_sha256': source_sha, 'proof': str(path.resolve()), 'proof_sha256': trusted_sha, 'archive': str(archive.resolve()), 'archive_sha256': proof['archive_sha256'], 'accepted_ids': sorted(occurrence)})
    for filename, trusted_sha in failure_specs:
        path = Path(filename)
        require(sha(path) == trusted_sha, 'Failure evidence SHA differs')
        proof = read(path)
        require(proof['status'] == 'FAILURE_EVIDENCE_VERIFIED_OFFSERVER_NOT_ACCEPTED' and proof['accepted_n'] == 0, 'Failure proof is not explicitly rejected')
        archive, data = verified_archive(path, proof, failure=True)
        require(hashlib.sha256(data['sealed_sources/replay_v3.py']).hexdigest() == SOURCES['v3'][1], 'Unexpected failed-attempt source')
        failed_ids = sorted(key(r['id']) for r in proof['errors'])
        require(len(failed_ids) == len(set(failed_ids)), 'Duplicate failed-attempt identities')
        failures.append({'proof': str(path.resolve()), 'proof_sha256': trusted_sha, 'archive_sha256': sha(archive), 'source_version': 'v3', 'accepted_n': 0, 'failed_or_interrupted_ids': failed_ids, 'later_accepted_ids': sorted(set(failed_ids) & set(accepted))})
    complete = set(accepted) == set(indexed)
    return {'scope': 'NINE_METHOD900_VALID_REPLAY_CUMULATIVE_EXPLICIT_SOURCE_VERSIONS', 'status': 'ALL900_UNIQUE_NATIVE_AND_THREE_VIEWS_ACCEPTED' if complete else 'PARTIAL_UNIQUE_VALID_REPLAY_ACCEPTED', 'inventory_sha256': INVENTORY_SHA, 'accepted_n': len(accepted), 'expected_n': 900, 'accepted_ids': sorted(accepted), 'accepted': [accepted[i] for i in sorted(accepted)], 'missing_ids': sorted(set(indexed) - set(accepted)), 'source_versions': sorted({r['source_version'] for r in accepted.values()}), 'distinct_checkpoint_sha256_n': len({r['checkpoint_sha256'] for r in accepted.values()}), 'accepted_provenance': origins, 'preserved_nonaccepted_attempts': failures, 'all900_native_valid_replayed': complete, 'all900_three_views_valid_replayed': complete, 'mechanism800_new_controls_included': False, 'test_inference_performed': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN', 'final_dispatch_created': False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--proof', nargs=3, action='append', metavar=('SOURCE_VERSION', 'PATH', 'REVIEWED_SHA256'), required=True)
    p.add_argument('--failure-proof', nargs=2, action='append', default=[], metavar=('PATH', 'REVIEWED_SHA256'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Preserve previous cumulative evidence')
    report = collect(a.inventory, a.proof, a.failure_proof)
    report['collector_source_sha256'] = sha(Path(__file__))
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'accepted_n', 'source_versions', 'all900_native_valid_replayed')}, indent=2))

if __name__ == '__main__':
    main()
