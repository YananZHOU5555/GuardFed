"""Download and verify exactly one closed GPU-valid chunk; append a separate collector."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import traceback
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
TOL = 1e-12


def need(ok, message):
    if not ok: raise ValueError(message)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def extract_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for name in names:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def context():
    pins = read(HERE / 'INPUT_PINS.json')
    for row in pins['files'].values(): need(sha(ROOT / row['path']) == row['sha256'], 'Frozen input drift: ' + row['path'])
    paths = {k: ROOT / row['path'] for k, row in pins['files'].items()}
    c = {'pins': pins, 'paths': paths, 'plan': read(paths['plan']), 'proposal': read(paths['proposal']), 'review': read(paths['review']), 'base': read(paths['base436']), 'records': {r['id']: r for r in read(paths['inventory'])['records']}}
    c['review_sha'] = sha(paths['review']); c['recovery_sha'] = sha(paths['recovery_seal']); c['queue_sha'] = sha(paths['queue_seal'])
    need(c['review']['current436_collector_sha256'] == sha(paths['base436']) and len(set(c['base']['accepted_ids'])) == c['base']['accepted_n'] == 436 and set(c['base']['accepted_ids']).isdisjoint(c['plan']['ids']) and len(c['records']) == 900, '436/464 lineage or inventory differs')
    for folder, seal in ((paths['recovery_seal'].parent, read(paths['recovery_seal'])), (paths['queue_seal'].parent, read(paths['queue_seal']))):
        for name, row in seal['members'].items(): need(sha(folder / name) == row['sha256'] and (folder / name).stat().st_size == row['bytes'], 'Frozen package member drift: ' + name)
    c['evaluator'] = extract_functions(paths['evaluator'], ['canonical_sha', 'check_binary', 'group_metrics', 'predict_views', 'evaluate_frozen_predictions'], {'np': np, 'json': json, 'hashlib': hashlib, 'METHODS': set(r['method'] for r in c['records'].values()), 'VIEWS': {'native', 'raw', 'shared_calibration'}})
    c['original'] = extract_functions(paths['v2'], ['canonical', 'check_native'], {'np': np, 'json': json, 'hashlib': hashlib, 'require': need, 'TOLERANCE': TOL})
    return c


def prior_chain(path, expected, c, chunk):
    need(sha(path) == expected, 'Externally bound prior SHA differs'); current = read(path); latest = current; seen = set()
    while sha(path) != sha(c['paths']['base436']):
        need(sha(path) not in seen, 'Prior chain cycle'); seen.add(sha(path))
        need(current['base436_sha256'] == sha(c['paths']['base436']) and current['evidence_package_sha256'] == sha(HERE / 'PACKAGE_SHA256.json'), 'Wrong436/source lineage')
        proof_path = Path(current['new_proof_path']); need(sha(proof_path) == current['new_proof_sha256'], 'Prior proof changed'); proof = read(proof_path)
        before = Path(current['previous_collector_path']); need(sha(before) == current['previous_collector_sha256'], 'Previous collector changed'); old = read(before)
        need(proof['status'] == 'ROOT_GPU464_CHUNK_OFFSERVER_SAVED_ARRAY_PASS' and proof['evidence_package_sha256'] == current['evidence_package_sha256'] and proof['chunk_index'] == current['last_chunk_index'] and proof['accepted_new_ids'] == current['added_ids'] == c['plan']['chunks'][current['last_chunk_index']]['ids'] and proof['remote_receipt_sha256'] == current['last_remote_receipt_sha256'], 'Prior chunk proof differs')
        need(current['accepted_ids'] == old['accepted_ids'] + current['added_ids'] and len(set(current['accepted_ids'])) == current['accepted_n'], 'Prior duplicate/omitted IDs')
        path, current = before, old
    prefix = [i for item in c['plan']['chunks'][:chunk] for i in item['ids']]
    need(latest['accepted_ids'] == c['base']['accepted_ids'] + prefix, 'Prior must close exactly preceding chunks; no skip/replay')
    return latest, None if chunk == 0 else latest['last_remote_receipt_sha256']


def unpack(archive, inventory, dest):
    need(sha(archive) == inventory['sha256'] and archive.stat().st_size == inventory['bytes'], 'Archive SHA/size differs')
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers(); specs = inventory['members']
        need(len(members) == len(set(bundle.getnames())) == len(specs) == inventory['member_n'] and set(bundle.getnames()) == set(specs), 'Duplicate/missing archive member')
        for item in members:
            name = PurePosixPath(item.name)
            need(item.isfile() and str(name) == item.name and not name.is_absolute() and '..' not in name.parts and ':' not in item.name and '\\' not in item.name and not item.name.endswith(('.pt', '.pth')), 'Unsafe member/old model')
            data = bundle.extractfile(item).read(); need(hashlib.sha256(data).hexdigest() == specs[item.name]['sha256'] and len(data) == specs[item.name]['bytes'], 'Archive member drift')
        dest.mkdir()
        for item in members:
            target = dest / item.name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write(bundle.extractfile(item).read())


def check_run(run, record, c, strict, batch, remote_stage):
    identity = record['id']; r = read(run / 'receipt.json'); p = read(run.with_name(identity + '.worker.json'))
    need(p['id'] == r['id'] == identity and p['status'] == r['status'] == 'DIAGNOSTIC_NATIVE_MATCH' and p['review_sha256'] == c['review_sha'] and p['batch_receipt_sha256'] == strict['batch_inputs_sha256'], 'Worker/receipt/review identity differs')
    need(p['artifacts_unchanged'] and p['artifact_before'] == p['artifact_after'] and sha(run / 'receipt.json') == p['receipt_sha256'] and sha(run / 'validation_predictions.npz') == r['prediction_arrays_sha256'], 'Weights/artifact/array proof drift')
    need(all(p[key] == digest for key, digest in {'v2_source_sha256': sha(c['paths']['v2']), 'sealed_v3_source_sha256': c['proposal']['scientific_source_pins_unchanged']['/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v3/replay_v3.py'], 'sealed_v4_source_sha256': c['proposal']['scientific_source_pins_unchanged']['/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/replay_v4.py'], 'gpu_body_sha256': read(c['paths']['recovery_seal'])['members']['gpu_replay_body.py']['sha256']}.items()), 'Worker source versions differ')
    expected = next(x for x in c['proposal']['records'] if x['id'] == identity)['artifacts']
    need(set(p['artifact_before']) == {v['target'] for v in expected.values()} and all(p['artifact_before'][v['target']]['sha256'] == v['sha256'] and p['artifact_before'][v['target']]['bytes'] == v['bytes'] for v in expected.values()), 'Wrong storage-map artifacts')
    def local(path): return run.parent.parent.parent / Path(path).relative_to(Path(remote_stage)) if Path(path).is_relative_to(Path(remote_stage)) else Path(path)
    guard = extract_functions(c['paths']['recovery'], ['receipt_guard'], {'Path': Path, 'need': need, 'sha': lambda x: sha(local(x)), 'read': lambda x: read(local(x)), '__file__': str(c['paths']['recovery']), 'SCOPE': 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1'})
    guard['receipt_guard'](p, r, c['review'], Path(remote_stage) / 'batch/runs', record)
    need(r['model_inventory_record_sha256'] == c['original']['canonical'](record) and record['terminal_round'] == record['config']['rounds'] == 70 and record['config']['seed'] == r['seed'], 'Original record/config/70round differs')
    need(r['valid_n'] == 19867 and r['weights_before'] == r['weights_after'] and not any(r[k] for k in ('optimizer_created', 'gradients_created', 'test_labels_accessed', 'test_inference_performed', 'final_dispatch_created')), 'Inference-only contract failed')
    root, data = r['root_reconstruction'], record['data_contract']
    need(data['actual_train_rows'] == 162770 and data['actual_evaluation_rows'] == 19867 and data['evaluation_split'] == record['original_split'] == 'valid' and root['root_n'] == 16277 and root['train_eval_disjoint'] and root['root_client_disjoint'] and root['root_image_ids_sha256'] == data['root_image_ids_sha256'] and root['client_sample_counts'] == data['client_sample_counts'], 'Root/train/valid identity differs')
    fits = copy.deepcopy(r['fits'])
    for view, fit in fits.items():
        need(fit['method'] == record['method'] and fit['view'] == view and fit['fit_data'] in ('none', 'clean_train_root_only'), 'Fit identity/split differs')
        if fit['thresholds'] is not None:
            need(set(fit['thresholds']) == {'0', '1'}, 'Threshold keys differ'); fit['thresholds'] = {int(k): v for k, v in fit['thresholds'].items()}
    with np.load(c['paths']['labels'], allow_pickle=False) as truth, np.load(run / 'validation_predictions.npz', allow_pickle=False) as z:
        need(len(z['root_image_ids']) == len(z['root_margins']) == 16277 and len(np.unique(z['root_image_ids'])) == 16277 and np.all((z['root_image_ids'] >= 1) & (z['root_image_ids'] <= 162770)) and hashlib.sha256(z['root_image_ids'].tobytes()).hexdigest() == root['root_image_ids_sha256'], 'Saved root IDs differ')
        need(np.array_equal(z['valid_image_ids'], np.arange(162771, 182638)) and len(z['valid_margins']) == 19867 and hashlib.sha256(z['valid_image_ids'].tobytes()).hexdigest() == r['valid_image_ids_sha256'] == strict['valid_image_ids_sha256'] == data['evaluation_image_ids_sha256'] and all(np.isfinite(z[k]).all() for k in ('root_margins', 'valid_margins')), 'Saved valid IDs/margins differ')
        predicted = c['evaluator']['predict_views'](z['valid_margins'], truth['valid_sensitive'], fits)
        need(all(np.array_equal(predicted[v], z['prediction_' + v]) for v in c['evaluator']['VIEWS']), 'Saved prediction rule differs')
        scored = c['evaluator']['evaluate_frozen_predictions'](predicted, truth['valid_y'], truth['valid_sensitive']); need(scored == r['views'], 'Saved9 metrics/24 counts differ')
    comparison = c['original']['check_native'](scored['native'], record['prior_validation_metrics'])
    need(comparison == r['native_comparison'] and comparison['accepted'], 'Native exceeds unchanged1e-12')
    return {'id': identity, 'device': 'cuda:0', 'GPU_uuid': p['GPU_uuid'], 'receipt_sha256': sha(run / 'receipt.json'), 'array_sha256': sha(run / 'validation_predictions.npz'), 'worker_proof_sha256': sha(run.with_name(identity + '.worker.json')), 'native_max_abs_difference': comparison['max_abs_difference']}


def verify_chunk(folder, c, index, previous):
    ids = c['plan']['chunks'][index]['ids']; chain = read(folder / 'REMOTE_PENDING_OFFSERVER.json'); inv = read(folder / 'remote_archive_inventory.json')
    need(chain['status'] == 'REMOTE_CLOSED_PENDING_OFFSERVER' and chain['chunk_index'] == inv['chunk_index'] == index and chain['remote_closed_ids'] == inv['accepted_ids'] == ids and chain['previous_receipt_sha256'] == previous and chain['queue_package_sha256'] == c['queue_sha'] and chain['review_sha256'] == c['review_sha'] and chain['accepted_new_n'] == 0 and not chain['cohort_registered'] and not chain['offserver_verified'], 'Remote receipt chain differs')
    need(inv['status'] == 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER' and not inv['offserver_verified'] and inv['old_model_files_archived'] == 0 and chain['archive_sha256'] == inv['sha256'] and chain['archive_inventory_sha256'] == sha(folder / 'remote_archive_inventory.json'), 'Remote archive binding differs')
    dest = folder / 'verified_extract'; unpack(folder / 'chunk_evidence.tar.gz', inv, dest)
    strict = read(dest / 'execution/strict_acceptance.json'); batch = read(dest / 'batch/batch_inputs.json'); execution = read(dest / 'batch/batch_execution.json'); binding = read(dest / 'execution/queue_binding.json')
    need(sha(dest / 'execution/strict_acceptance.json') == chain['strict_sha256'] and strict['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED' and strict['scope'] == 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1' and strict['accepted_ids'] == ids and strict['accepted_n'] == strict['requested_n'] == len(ids) and not strict['invalid'] and strict['max_abs_native_metric_difference'] <= TOL and not strict['all900_native_valid_replayed'], 'Original strict incomplete')
    need(batch['selected_ids'] == execution['finished_zero_exit_ids'] == execution['requested_ids'] == ids and batch['workers'] == 1 and batch['review_sha256'] == c['review_sha'] and batch['implementation_package_sha256'] == c['recovery_sha'] and sha(dest / 'batch/batch_inputs.json') == strict['batch_inputs_sha256'] and not execution['failures'] and execution['source_unchanged'] and execution['source_after'] == batch['source_before'] and not execution['cohort_registered'], 'Batch/source binding differs')
    need(strict['inventory_sha256'] == c['pins']['files']['inventory']['sha256'] and strict['storage_map_sha256'] == batch['storage_map_sha256'] == c['proposal']['scientific_cli_bindings_unchanged']['storage-map-sha256'] and strict['v2_source_sha256'] == sha(c['paths']['v2']) and strict['calibration_core_sha256'] == 'cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed' and strict['final_protocol_status'] == 'PREPARED_NOT_FROZEN' and not strict['test_labels_accessed'], 'Strict data/source/protocol differs')
    need(binding == {'queue_package_sha256': c['queue_sha'], 'queue_manifest_sha256': sha(c['paths']['plan']), 'root_review_sha256': c['review_sha'], 'prior425_sha256': c['plan']['accepted425_sha256'], 'cohort_registered': False}, 'Wrapper source binding differs')
    files = dict(c['proposal']['scientific_source_pins_unchanged']); files.update({c['pins']['remote_review']: c['review_sha'], c['pins']['remote_proposal']: sha(c['paths']['proposal'])}); files.update({c['pins']['remote_recovery'] + '/' + n: v['sha256'] for n, v in read(c['paths']['recovery_seal'])['members'].items()})
    expected = {'sourcefreeze/%03d_%s' % (n, PurePosixPath(path).name): digest for n, (path, digest) in enumerate(sorted(files.items())) if not path.startswith('/workspace/GuardFed-celeba-expanded/data/')}
    need({n for n in inv['members'] if n.startswith('sourcefreeze/')} == set(expected) and all(sha(dest / n) == digest for n, digest in expected.items()) and sha(dest / 'execution/manifest.json') == sha(c['paths']['proposal']), 'Sourcefreeze differs')
    need(all(batch['source_before'][path]['sha256'] == digest for path, digest in files.items()), 'Source/data snapshot differs')
    results = [check_run(dest / 'batch/runs' / identity, c['records'][identity], c, strict, batch, c['plan']['output_parent'] + '/chunk_%03d' % index) for identity in ids]
    return {'status': 'ROOT_GPU464_CHUNK_OFFSERVER_SAVED_ARRAY_PASS', 'chunk_index': index, 'accepted_new_ids': ids, 'archive_sha256': inv['sha256'], 'archive_members_verified': inv['member_n'], 'archive_inventory_sha256': sha(folder / 'remote_archive_inventory.json'), 'remote_receipt_sha256': sha(folder / 'REMOTE_PENDING_OFFSERVER.json'), 'strict_sha256': chain['strict_sha256'], 'source_package_sha256': c['recovery_sha'], 'queue_package_sha256': c['queue_sha'], 'review_sha256': c['review_sha'], 'results': results, 'saved_metrics_verified': len(ids) * 9, 'saved_confusion_counts_verified': len(ids) * 24, 'saved_prediction_rules_verified': len(ids) * 3, 'native_max_abs_difference': max(r['native_max_abs_difference'] for r in results), 'root_refit_verified_in_original_remote_strict': True, 'local_root_refit': False, 'new_CNN_inference': 0, 'final_test': False}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--chunk', type=int, required=True); p.add_argument('--package-sha256', required=True); p.add_argument('--prior', type=Path); p.add_argument('--prior-sha256'); p.add_argument('--output', type=Path, required=True); p.add_argument('--bundle', type=Path); args = p.parse_args()
    need(sha(HERE / 'PACKAGE_SHA256.json') == args.package_sha256, 'Verifier source SHA differs')
    for n, row in read(HERE / 'PACKAGE_SHA256.json')['members'].items(): need(sha(HERE / n) == row['sha256'], 'Verifier member drift')
    c = context(); need(0 <= args.chunk < len(c['plan']['chunks']), 'Unknown chunk')
    need(bool(args.prior) == bool(args.prior_sha256), 'Pass prior and external SHA together')
    prior_path = args.prior or c['paths']['base436']; prior_sha = args.prior_sha256 if args.prior else sha(prior_path)
    prior, previous = prior_chain(prior_path, prior_sha, c, args.chunk); ids = c['plan']['chunks'][args.chunk]['ids']; need(not set(ids) & set(prior['accepted_ids']), 'Already accepted ID')
    out = args.output.absolute(); need(out.parent.resolve() == out.parent and out.is_relative_to(HERE) and not out.exists(), 'Fresh owned output required'); out.mkdir()
    try:
        remote = c['plan']['output_parent']; stage = remote + '/chunk_%03d' % args.chunk
        if not args.bundle:
            subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', '60350', 'root@89.22.197.55:/etc/vast-agents-guide.md', str(out / 'vast-agents-guide.md')], check=True, timeout=45)
            need(sha(out / 'vast-agents-guide.md') == c['plan']['guide_sha256'], 'Guide changed; stop and reread')
        sources = {'remote_archive_inventory.json': stage + '/remote_archive_inventory.json', 'chunk_evidence.tar.gz': stage + '/chunk_evidence.tar.gz', 'REMOTE_PENDING_OFFSERVER.json': remote + '/chunk_%03d.REMOTE_PENDING_OFFSERVER.json' % args.chunk}
        for name, source in sources.items():
            if args.bundle: shutil.copyfile(args.bundle / name, out / name)
            else: subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-P', '60350', 'root@89.22.197.55:' + source, str(out / name)], check=True, timeout=180)
        proof = verify_chunk(out, c, args.chunk, previous)
        need(sha(prior_path) == prior_sha, 'Prior changed during verification')
        for row in c['pins']['files'].values(): need(sha(ROOT / row['path']) == row['sha256'], 'Frozen input changed during verification')
        proof.update(evidence_package_sha256=args.package_sha256, model_inventory_sha256=c['pins']['files']['inventory']['sha256'], labels_cache_sha256=c['pins']['files']['labels']['sha256'], base436_sha256=sha(c['paths']['base436'])); proof_path = out / 'ROOT_OFFSERVER_VERIFICATION.json'; save(proof_path, proof)
        combined = prior['accepted_ids'] + ids; need(len(set(combined)) == len(combined), 'Duplicate cohort identity')
        collector = {'scope': 'NINE_METHOD900_VALID_REPLAY_CUMULATIVE_EXPLICIT_CPU_GPU_SOURCE_VERSIONS', 'status': 'PARTIAL_UNIQUE_VALID_REPLAY_ACCEPTED' if len(combined) < 900 else 'ALL900_VALID_REPLAY_ACCEPTED', 'expected_n': 900, 'accepted_n': len(combined), 'accepted_ids': combined, 'base436_sha256': sha(c['paths']['base436']), 'previous_collector_path': str(prior_path.resolve()), 'previous_collector_sha256': prior_sha, 'new_proof_path': str(proof_path), 'new_proof_sha256': sha(proof_path), 'added_ids': ids, 'last_chunk_index': args.chunk, 'last_remote_receipt_sha256': proof['remote_receipt_sha256'], 'evidence_package_sha256': args.package_sha256, 'mixing_CPU_GPU_disclosed': True, 'uniform_device_comparison': False, 'all900_native_valid_replayed': len(combined) == 900, 'final_protocol_frozen': False, 'test_evaluation_performed': False}
        save(out / ('cumulative_%d_accepted.json' % len(combined)), collector); print(json.dumps({'accepted_n': len(combined), 'new_n': len(ids), 'proof_sha256': sha(proof_path)}))
    except BaseException as error:
        save(out / 'FAILURE_PRESERVED.json', {'status': 'NOT_ACCEPTED_PRESERVED_NO_RETRY', 'error': repr(error), 'traceback': traceback.format_exc(), 'chunk': args.chunk}); raise


if __name__ == '__main__': main()
