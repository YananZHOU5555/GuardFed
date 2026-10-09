"""Offline, hash-bound view/schema adapter around the original table functions.

No model loading, inference, threshold fitting, test data or remote operations.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import ast
import copy
import csv
import difflib
import hashlib
import io
import json
import math
from statistics import mean, stdev
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'outputs/guardfed_tables/celeba_nine_method_final_20261004'
EV = ROOT / 'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009'
REC = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
FINAL = EV / 'chunk_039/cumulative_900_accepted.json'
FINAL_SHA = '00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3'
INV = ROOT / 'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json'
INV_SHA = '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
LABEL = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009/original_valid_cache.npz'
LABEL_SHA = '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64'
EVALUATOR = ROOT / 'tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py'
EVALUATOR_SHA = '805eedf1fb08137cd86a543a80c83b9527e5c8937be02f7d2dca83a33b86e04c'
REPLAY = ROOT / 'tmp/celeba_final_valid_replay_20261009/replay.py'
REPLAY_SHA = '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'
RENDERER = OLD / 'build_tables.py'
RENDERER_SHA = '4ae22c85c3fcc1f8e01ac9b3fc4dbf0ce45ea676f39a677e337782cb0d2b3529'
SNAPSHOT_SHA = '9350e8b1fc9e10d5ae587c3cda9ece899c6258c6dbd2cf5e6d86b2d2ff2fe741'
VIEWS = ['raw', 'native', 'shared_calibration']
METHODS = ['FedAvg', 'FairFed', 'Median', 'FLTrust', 'FedAA-DDPG', 'LASA', 'FairGuard', 'FLTrust+FairGuard', 'GuardFed-AD2+']
DISTS = ['IID', 'non-IID']
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
METRICS = ['accuracy', 'aeod', 'aspd']
SEEDS = {'ten': list(range(91001, 91011)), 'nonselection_nine': list(range(91002, 91011)), 'matching_six': list(range(91005, 91011))}
pins, archives, chain, records, member_audit = {}, [], [], {}, []
checks = Counter()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')


def read(p, expected=None):
    p = path(p)
    raw = p.read_bytes()
    h = sha(raw)
    require(expected is None or h == expected, 'File SHA mismatch: ' + str(p))
    pins[rel(p)] = {'sha256': h, 'bytes': len(raw)}
    return raw


def load(p, expected=None):
    return json.loads(read(p, expected).decode('utf-8-sig'))


def write(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def functions(p, expected, names, namespace):
    s = read(p, expected).decode('utf-8-sig')
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Original functions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), namespace)
    return s, nodes


science = {'np': np, 'json': json, 'hashlib': hashlib}
evaluator_source, _ = functions(EVALUATOR, EVALUATOR_SHA, ['canonical_sha', 'check_binary', 'group_metrics', 'predict_views', 'evaluate_frozen_predictions'], science)
for n in ast.parse(evaluator_source).body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ['METHODS', 'VIEWS'] for t in n.targets):
        exec(compile(ast.Module(body=[n], type_ignores=[]), str(EVALUATOR), 'exec'), science)
alias_ns = {}
for n in ast.parse(read(OLD / 'merge_sources.py').decode('utf-8-sig')).body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'names' for t in n.targets):
        exec(compile(ast.Module(body=[n], type_ignores=[]), 'original/merge_sources.py', 'exec'), alias_ns)
require(alias_ns['names'] == {'FedAA-DDPG-adapted-v1': 'FedAA-DDPG', 'LASA-official': 'LASA'}, 'Original display aliases changed')
functions(REPLAY, REPLAY_SHA, ['canonical', 'check_native'], science)
science.update({'require': require, 'TOLERANCE': 1e-12})
canonical = science['canonical']
inventory = load(INV, INV_SHA)
inventory_by_id = {r['id']: r for r in inventory['records']}
require(len(inventory_by_id) == 900, 'Original inventory is not unique900')
old = load(OLD / 'source_acceptance_snapshot.json', SNAPSHOT_SHA)
old_by_id = {r['id']: r for r in old['all_conditions']}
old_by_cell = {(r['method'], r['distribution'], r['attack'], r['seed']): r for r in old['all_conditions']}
require(len(old_by_cell) == 900 and len(old_by_id) == 900, 'Old table is not unique900')
with np.load(io.BytesIO(read(LABEL, LABEL_SHA)), allow_pickle=False) as z:
    valid_y, valid_s = z['valid_y'], z['valid_sensitive']
require(len(valid_y) == len(valid_s) == 19867, 'Wrong valid labels')


def check_record(rid, receipt_bytes, array_bytes, binding):
    require(rid not in records, 'Duplicate accepted ID: ' + rid)
    r = json.loads(receipt_bytes)
    inv = inventory_by_id[rid]
    original_cell = (inv['method'], inv['distribution'], inv['attack'], inv['seed'])
    display_method = alias_ns['names'].get(inv['source_method'], inv['method'])
    cell = (display_method, inv['distribution'], inv['attack'], inv['seed'])
    require(tuple(r[k] for k in ('method', 'distribution', 'attack', 'seed')) == original_cell and r['id'] == rid, 'Receipt identity mismatch')
    require(r['status'] in ['NATIVE_VALID_REPLAY_PASS', 'DIAGNOSTIC_NATIVE_MATCH'], 'Incomplete/failed receipt')
    for key, expected in [('model_inventory_record_sha256', canonical(inv)), ('checkpoint_sha256', inv['checkpoint']['sha256']), ('original_result_sha256', inv['result']['sha256']), ('original_job_sha256', inv['raw_job']['sha256']), ('config_canonical_sha256', inv['config_canonical_sha256']), ('original_training_torch', inv['training_torch'])]:
        require(r[key] == expected, key + ' mismatch: ' + rid)
    require(canonical(inv['config']) == inv['config_canonical_sha256'], 'Original config seal changed')
    require(inv['terminal_round'] == 70 and inv['original_split'] == 'valid' and inv['original_n_eval'] == r['valid_n'] == 19867, 'Wrong round/split/n')
    require(r['valid_image_ids_sha256'] == '64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf', 'Wrong target IDs')
    require(r['weights_before'] == r['weights_after'], 'Weights changed')
    require(not r['optimizer_created'] and not r['gradients_created'] and not r['test_labels_accessed'] and not r['test_inference_performed'], 'Forbidden science operation in receipt')
    require(sha(array_bytes) == r['prediction_arrays_sha256'], 'Array hash mismatch')
    require(set(r['views']) == set(r['fits']) == set(VIEWS), 'Incomplete three views')
    fits = copy.deepcopy(r['fits'])
    for f in fits.values():
        if f.get('thresholds') is not None:
            f['thresholds'] = {int(k): v for k, v in f['thresholds'].items()}
    with np.load(io.BytesIO(array_bytes), allow_pickle=False) as z:
        require(np.array_equal(z['valid_image_ids'], np.arange(162771, 182638)), 'Valid array image identity')
        require(sha(z['valid_image_ids'].astype('<i8').tobytes()) == r['valid_image_ids_sha256'], 'Valid array IDs SHA')
        require(sha(z['root_image_ids'].astype('<i8').tobytes()) == inv['data_contract']['root_image_ids_sha256'], 'Root IDs SHA')
        require(len(z['root_margins']) == len(z['root_image_ids']) == 16277, 'Root dimensions')
        predictions = science['predict_views'](z['valid_margins'], valid_s, fits)
        for view in VIEWS:
            require(np.array_equal(predictions[view], z['prediction_' + view]), 'Saved prediction rule mismatch')
            checks['saved_prediction_rules'] += 1
        actual = science['evaluate_frozen_predictions'](predictions, valid_y, valid_s)
    for view in VIEWS:
        require(actual[view] == r['views'][view], 'Saved metrics/counts/status mismatch: ' + rid + ' ' + view)
        checks['saved_metrics'] += 3
        checks['saved_confusion_counts'] += 8
    comparison = science['check_native'](r['views']['native'], inv['prior_validation_metrics'])
    require(comparison == r['native_comparison'] and comparison['accepted'] and comparison['max_abs_difference'] == 0, 'Native original replay mismatch')
    previous = old_by_cell[cell]
    require(previous['checkpoint_sha256'] == r['checkpoint_sha256'], 'Old table checkpoint mismatch')
    require(all(previous[k] == r['views']['native'][k] for k in METRICS), 'Old table native value mismatch')
    require(previous['candidate'] == inv['candidate'] and previous['torch_version'] == inv['training_torch'], 'Old source config/environment mismatch')
    require(r['runtime']['device'] == 'cpu' or r['runtime']['device'].startswith('cuda'), 'Undeclared inference device')
    records[rid] = {'id': rid, 'scientific_cell': list(cell), 'original_scientific_cell': list(original_cell), 'method': display_method, 'original_method': inv['method'], 'source_method': inv['source_method'], 'distribution': inv['distribution'], 'attack': inv['attack'], 'seed': inv['seed'], 'old_table_id': previous['id'], 'old_table_identity_join': 'original source-method display alias plus exact scientific_cell/checkpoint SHA/candidate/environment/native three metrics', 'model_inventory_record_sha256': canonical(inv), 'original_inventory_path': rel(INV), 'original_inventory_sha256': INV_SHA, 'original_inventory_record': inv, 'receipt_sha256': sha(receipt_bytes), 'prediction_arrays_sha256': sha(array_bytes), 'checkpoint_sha256': r['checkpoint_sha256'], 'runtime': r['runtime'], 'training_torch': inv['training_torch'], 'training_torch_source': inv['training_torch_source'], 'original_config': inv['config'], 'config_canonical_sha256': inv['config_canonical_sha256'], 'root_reconstruction': r['root_reconstruction'], 'views': r['views'], 'fits': r['fits'], 'native_comparison': comparison, 'source_binding': binding, 'same_checkpoint_all_views': True, 'test_evaluation_performed': False}
    checks['IDs'] += 1
    checks['native_original_and_old_table_metrics'] += 3


def group(proof_path, proof_sha, archive_path, archive_sha, ids, receipt_pins=None, strict_path=None, strict_sha=None, accepted_import=None):
    proof_path = path(proof_path)
    proof = load(proof_path, proof_sha)
    require(proof.get('archive_sha256', proof.get('failure_archive_sha256')) == archive_sha, 'Proof archive seal mismatch')
    if accepted_import:
        load(accepted_import[0], accepted_import[1])
    raw = read(archive_path, archive_sha)
    archive_path = path(archive_path)
    member_bytes = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        for m in tar.getmembers():
            require(not m.issym() and not m.islnk(), 'Linked archive member')
            if m.isfile():
                require(m.name not in member_bytes, 'Duplicate archive member')
                member_bytes[m.name] = tar.extractfile(m).read()
                require(len(member_bytes[m.name]) == m.size, 'Truncated archive member')
    member_hashes = {k: {'sha256': sha(v), 'bytes': len(v)} for k, v in member_bytes.items()}
    inv_path = next((p for p in [proof_path.parent / 'remote_archive_inventory.json', proof_path.parent / 'failure_remote_archive_inventory.json', proof_path.parent / 'MEMBERS.json'] if p.exists()), None)
    inv_ref = None
    if inv_path:
        manifest = load(inv_path, proof.get('archive_inventory_sha256', proof.get('inventory_sha256', proof.get('failure_inventory_sha256'))))
        expected = manifest.get('members', manifest)
        if isinstance(expected, dict) and all(isinstance(v, dict) and 'sha256' in v for v in expected.values()):
            require(set(expected) <= set(member_hashes), 'Archive member inventory missing members')
            require(all(member_hashes[k]['sha256'] == v['sha256'] and member_hashes[k]['bytes'] == v.get('bytes', member_hashes[k]['bytes']) for k, v in expected.items()), 'Archive member inventory hash/size mismatch')
        inv_ref = {'path': rel(inv_path), 'sha256': pins[rel(inv_path)]['sha256']}
    strict_sha = strict_sha or proof.get('strict_sha256', proof.get('strict_acceptance_sha256', proof.get('run_receipt_sha256', proof.get('partial_strict_sha256'))))
    require(strict_sha, 'Strict/diagnostic seal missing')
    if strict_path:
        strict = load(strict_path, strict_sha)
        strict_ref = {'path': rel(path(strict_path)), 'sha256': strict_sha}
    else:
        matches = [k for k, v in member_hashes.items() if v['sha256'] == strict_sha]
        require(len(matches) == 1, 'Strict seal not unique in archive')
        strict = json.loads(member_bytes[matches[0]])
        strict_ref = {'member': matches[0], 'sha256': strict_sha}
    if 'accepted_ids' in strict:
        require(set(ids) <= set(strict['accepted_ids']), 'ID not strict accepted')
        if 'inventory_sha256' in strict:
            require(strict['inventory_sha256'] == INV_SHA, 'Strict inventory seal mismatch')
    else:
        require(len(ids) == 1 and strict.get('id') == ids[0] and strict['native_comparison']['accepted'], 'Diagnostic ID/acceptance mismatch')
        require(accepted_import, 'Diagnostic requires explicit accepted import')
    proof_rows = proof.get('models', proof.get('canaries', proof.get('results', [])))
    proof_rows = {r['id']: r for r in proof_rows}
    remote_receipt = None
    if proof.get('remote_receipt_sha256'):
        remote_path = proof_path.parent / 'REMOTE_PENDING_OFFSERVER.json'
        load(remote_path, proof['remote_receipt_sha256'])
        remote_receipt = {'path': rel(remote_path), 'sha256': proof['remote_receipt_sha256']}
    for rid in ids:
        matches = [k for k in member_bytes if k.endswith('/' + rid + '/receipt.json')]
        require(len(matches) == 1, 'Receipt not unique in sealed archive')
        receipt_member = matches[0]
        array_member = receipt_member.rsplit('/', 1)[0] + '/validation_predictions.npz'
        row = (receipt_pins or {}).get(rid, proof_rows.get(rid, proof if proof.get('id') in (None, rid) else {}))
        for key, member in [('receipt_sha256', receipt_member), ('array_sha256', array_member), ('prediction_arrays_sha256', array_member)]:
            if key in row:
                require(row[key] == member_hashes[member]['sha256'], 'Proof/collector individual member mismatch')
        worker_ref = None
        if row.get('worker_proof_sha256'):
            worker_matches = [k for k in member_hashes if k.endswith('/' + rid + '.worker.json')]
            require(len(worker_matches) == 1 and member_hashes[worker_matches[0]]['sha256'] == row['worker_proof_sha256'], 'GPU worker proof mismatch')
            worker_ref = {'member': worker_matches[0], 'sha256': row['worker_proof_sha256']}
        binding = {'offserver_proof': {'path': rel(proof_path), 'sha256': proof_sha, 'status': proof['status']}, 'archive': {'path': rel(archive_path), 'sha256': archive_sha}, 'archive_inventory': inv_ref, 'strict_or_diagnostic': strict_ref, 'strict_or_diagnostic_metadata': strict, 'receipt_member': receipt_member, 'array_member': array_member, 'worker_proof': worker_ref, 'remote_closed_receipt': remote_receipt, 'proof_record_metadata': row, 'implementation_provenance': {k:v for k,v in proof.items() if ('source' in k or 'package' in k or 'review' in k or 'scope' in k)}, 'accepted_import': {'path': rel(path(accepted_import[0])), 'sha256': accepted_import[1]} if accepted_import else None}
        check_record(rid, member_bytes[receipt_member], member_bytes[array_member], binding)
    archives.append({'archive': rel(archive_path), 'sha256': archive_sha, 'member_n': len(member_hashes), 'accepted_IDs_used': ids, 'strict_or_diagnostic': strict_ref, 'proof': rel(proof_path), 'proof_sha256': proof_sha})
    member_audit.append({'archive': rel(archive_path), 'archive_sha256': archive_sha, 'members': member_hashes})
    print(json.dumps({'verified_IDs': len(records), 'archive': rel(archive_path), 'used_IDs': len(ids)}), flush=True)


final = load(FINAL, FINAL_SHA)
require(final['accepted_n'] == 900 and len(set(final['accepted_ids'])) == 900 and set(final['accepted_ids']) == set(inventory_by_id), 'Final collector grid')
pending_groups = []
current, current_path = final, FINAL
while 'previous_collector_path' in current:
    pp = path(current['previous_collector_path'])
    prior = load(pp, current['previous_collector_sha256'])
    require(set(current['accepted_ids']) == set(prior['accepted_ids']) | set(current['added_ids']) and not set(prior['accepted_ids']) & set(current['added_ids']), 'Collector delta/duplicate')
    require(len(current['accepted_ids']) == current['accepted_n'] == prior['accepted_n'] + len(current['added_ids']), 'Collector count mismatch')
    p = path(current['new_proof_path'])
    proof = load(p, current['new_proof_sha256'])
    require(set(proof.get('accepted_new_ids', [])) == set(current['added_ids']), 'Proof accepted membership mismatch')
    pending_groups.append((current, p, proof))
    chain.append({'path': rel(current_path), 'sha256': pins[rel(current_path)]['sha256'], 'accepted_n': current['accepted_n'], 'added_ids': current['added_ids'], 'previous_path': rel(pp), 'previous_sha256': current['previous_collector_sha256'], 'proof_path': rel(p), 'proof_sha256': current['new_proof_sha256']})
    current, current_path = prior, pp
require(current['accepted_n'] == 436, 'Unexpected collector-chain boundary')
base = current
cpu_path = path(base['previous_424_collector_path'])
cpu = load(cpu_path, base['previous_424_collector_sha256'])
require(cpu['accepted_n'] == 424 and len(cpu['accepted']) == 424, 'Original CPU424 collector mismatch')
cpu_rows = {r['id']: r for r in cpu['accepted']}
for g in cpu['accepted_provenance']:
    group(g['proof'], g['proof_sha256'], g['archive'], g['archive_sha256'], g['accepted_ids'], cpu_rows)

import_path = REC / 'preserved11_import/ROOT_OFFSERVER_IMPORT_VERIFICATION.json'
import_sha = base['preserved11_import_verification_sha256']
imp = load(import_path, import_sha)
cpu_import = load(REC / 'preserved11_import/CPU10.json', imp['members']['CPU10.json']['sha256'])
gpu_import = load(REC / 'preserved11_import/GPU1.json', imp['members']['GPU1.json']['sha256'])
load(REC / 'preserved11_import/ROOT_REVIEW_IMPORT11.json', imp['review_sha256'])
cpudir = ROOT / 'tmp/celeba_final_valid_replay_20261009/v4/remaining872_attempt1/chunk_036'
group(cpudir / 'failure_offserver_verification.json', imp['CPU10_source_proof_sha256'], cpudir / 'failure_chunk_evidence.tar.gz', 'fe5c3786ead4d4878e6e6f16c89793e3d10163f75559ef7ad7e798798ccfb555', cpu_import['eligible_ids'], strict_path=REC / 'preserved11_import/CPU10.original_v4_strict.json', strict_sha=imp['members']['CPU10.original_v4_strict.json']['sha256'], accepted_import=(import_path, import_sha))
gpudir = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009/attempt2_backup'
with tarfile.open(gpudir / 'evidence.tar.gz') as tar:
    diagnostic = tar.extractfile('runs/FairGuard_IID_F-Flip_seed91009.diagnostic.json').read()
group(gpudir / 'OFFSERVER_VERIFICATION.json', imp['GPU1_source_proof_sha256'], gpudir / 'evidence.tar.gz', 'fb1c2861429fc03acd0f2304c68a6a0b6a39c147153b7db1eef03d6172a3ec5d', gpu_import['eligible_ids'], strict_sha=sha(diagnostic), accepted_import=(import_path, import_sha))
first = REC / 'first1_backup'
group(first / 'ROOT_OFFSERVER_VERIFICATION.json', base['new_GPU_verification_sha256'], first / 'chunk_evidence.tar.gz', '87fd3fb4e37994b2616c7f3794cc4a8fd6c306e1e4dd3dba83ddbce090f7849b', [base['new_GPU_id']], {base['new_GPU_id']: {'receipt_sha256': base['new_GPU_receipt_sha256'], 'array_sha256': base['new_GPU_array_sha256']}})
require(set(records) == set(base['accepted_ids']), 'Base436 accepted IDs mismatch')
for c, p, proof in reversed(pending_groups):
    if 'failure_archive_sha256' in proof:
        a = ROOT / 'tmp/celeba_valid_gpu_remaining464_execution_20261009/failure_chunk002/failure_chunk_evidence.tar.gz'
        group(p, c['new_proof_sha256'], a, proof['failure_archive_sha256'], c['added_ids'], strict_path=p.parent / 'original_partial_strict_acceptance.json')
    else:
        group(p, c['new_proof_sha256'], p.parent / 'chunk_evidence.tar.gz', proof['archive_sha256'], c['added_ids'])
require(set(records) == set(final['accepted_ids']) and len(records) == 900, 'Actual receipt grid is incomplete')
require({tuple(r['scientific_cell']) for r in records.values()} == {(m,d,a,s) for m in METHODS for d in DISTS for a in ATTACKS for s in SEEDS['ten']}, 'Scientific grid is incomplete')
require({r['old_table_id'] for r in records.values()} == set(old_by_id), 'Old900 join is not one-to-one')
device_counts = Counter('CPU' if r['runtime']['device'] == 'cpu' else 'GPU' for r in records.values())
training_counts = Counter(r['training_torch'] for r in records.values())
require(device_counts == {'CPU': 434, 'GPU': 466}, 'Measured inference-device counts unexpected')
require(training_counts == {'2.11.0+cu128': 886, '2.11.0+cu130': 14}, 'Measured original training environment counts unexpected')
write('records_three_views_900.json', {'scope': 'NINE_METHOD_VALID_ONLY_SAME_CHECKPOINT_THREE_VIEWS', 'final_collector_path': rel(FINAL), 'final_collector_sha256': FINAL_SHA, 'inventory_path': rel(INV), 'inventory_sha256': INV_SHA, 'records': [records[r['id']] for r in inventory['records']]})
write('collector_chain.json', {'final_sha256': FINAL_SHA, 'standard_deltas': chain, 'base436': {'path': rel(current_path), 'sha256': pins[rel(current_path)]['sha256'], 'original_CPU424': {'path': rel(cpu_path), 'sha256': base['previous_424_collector_sha256']}, 'preserved11': {'path': rel(import_path), 'sha256': import_sha}, 'first_GPU': base['new_GPU_id']}, 'archives': archives})
write('archive_member_SHA256_audit.json', member_audit)

# Load the original statistics and rendering functions without running the old
# module's top-level writes or its additional pairwise summaries.
table_ns = {'mean': mean, 'stdev': stdev, 'plt': plt}
renderer_source, renderer_nodes = functions(RENDERER, RENDERER_SHA, ['values_for', 'render'], table_ns)
tree = ast.parse(renderer_source)
for n in tree.body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ['methods', 'categories', 'dists', 'attacks', 'metrics', 'adapted'] for t in n.targets):
        exec(compile(ast.Module(body=[n], type_ignores=[]), str(RENDERER), 'exec'), table_ns)
original_render = next(n for n in renderer_nodes if n.name == 'render')
require(table_ns['methods'] == METHODS and table_ns['dists'] == DISTS and table_ns['attacks'] == ATTACKS, 'Original presentation grid changed')
all_summaries, table_files, diff_blocks = {}, [], []
independent = Counter()
max_mean_diff, max_sd_diff = 0.0, 0.0
checked_utc = datetime.now(timezone.utc).isoformat()


class NoPdf:
    """Original renderer output sink; this delivery uses PNG/Markdown/TeX."""
    def savefig(self, *args, **kwargs):
        pass


plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
view_notes = {
    'raw': 'Raw view: all nine methods use uncalibrated argmax (margin > 0); same saved checkpoint per ID.',
    'native': 'Native view: GuardFed uses saved clean-training-root group calibration; the eight baselines use native argmax.',
    'shared_calibration': 'Shared-calibration view: all nine methods use saved clean-training-root group thresholds; no validation-label fitting.'
}
notes_base = [
    'Round 70; validation only (19,867 images). Mean +/- sample SD (ddof = 1); ACC in %, gaps on [0, 1].',
    'AEOD is the implemented absolute TPR gap. Descriptive values do not establish statistical significance.',
    '* Adaptations: FairFed, FairGuard, hybrid; FedAA-DDPG round/policy adapter; LASA with local-Adam update differences.',
    'Recipes fixed before coverage. Seven methods used non-IID Benign/S-DFA search; FedAA/LASA used both distributions, Benign/S-DFA, seed 91001.',
    '',
    'Nine methods only; eight additional manuscript baselines and final frozen evaluation remain. Native/shared main endpoint is pending.',
    '',
    'All outcomes, including constant predictions, are retained. Inference: 434 CPU / 466 GPU in the 900-ID source; no uniform-device claim.'
]
for view in VIEWS:
    viewdir = OUT / view
    viewdir.mkdir(exist_ok=True)
    rows = []
    groups = defaultdict(list)
    for inv in inventory['records']:
        r = records[inv['id']]
        row = {'id': r['id'], 'method': r['method'], 'distribution': r['distribution'], 'attack': r['attack'], 'seed': r['seed'], 'checkpoint_sha256': r['checkpoint_sha256'], 'torch_version': r['training_torch'], **{k: r['views'][view][k] for k in METRICS}}
        rows.append(row)
        groups[row['method'],row['distribution'],row['attack']].append(row)
    data = {'checked_utc': checked_utc, 'accepted_total': 900, 'complete': True, 'view': view, 'all_conditions': rows}
    notes = notes_base.copy()
    notes[6] = view_notes[view]
    table_ns.update({'OUT': viewdir, 'data': data, 'groups': groups, 'notes': notes, 'pdf_pages': NoPdf()})
    # Presentation-only replacement; the original values_for function and every
    # mean/stdev call remain byte-for-byte/AST unchanged.
    display_render = copy.deepcopy(original_render)
    for n in ast.walk(display_render):
        if isinstance(n, ast.Constant) and n.value == 'CelebA: ':
            n.value = 'CelebA [' + view.replace('_', ' ') + ']: '
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('All 90 method/distribution/scenario cells use the same '):
            n.value = 'All displayed cells use the same '
    ast.fix_missing_locations(display_render)
    exec(compile(ast.Module(body=[display_render], type_ignores=[]), str(RENDERER), 'exec'), table_ns)
    diff_blocks.extend(difflib.unified_diff(ast.unparse(original_render).splitlines(True), ast.unparse(display_render).splitlines(True), fromfile='original/render', tofile=view + '/render'))
    all_summaries[view] = {}
    for subset, seeds in SEEDS.items():
        summaries = []
        for m in METHODS:
            for d in DISTS:
                for a in ATTACKS:
                    selected = [r for r in groups[m,d,a] if r['seed'] in seeds]
                    require(len(selected) == len(seeds), 'Paired seed count mismatch')
                    displayed = table_ns['values_for'](m,d,a,seeds)
                    summary = {'method': m, 'distribution': d, 'attack': a, 'seeds': seeds, 'IDs': [r['id'] for r in selected], 'n': len(seeds)}
                    for i, (k, _, scale, precision) in enumerate(table_ns['metrics']):
                        vals = [r[k] for r in selected]
                        mu, sd = table_ns['mean'](vals), table_ns['stdev'](vals)
                        nmu, nsd = float(np.mean(vals)), float(np.std(vals, ddof=1))
                        max_mean_diff = max(max_mean_diff, abs(mu-nmu))
                        max_sd_diff = max(max_sd_diff, abs(sd-nsd))
                        require(abs(mu-nmu) <= 1e-12 and abs(sd-nsd) <= 1e-12, 'Independent mean/sample SD mismatch')
                        require(displayed[i] == f'{nmu*scale:.{precision}f} ± {nsd*scale:.{precision}f}', 'Independent displayed value mismatch')
                        summary[k] = {'mean': mu, 'sample_sd': sd, 'display': displayed[i]}
                        independent['mean_sampleSD_display_pairs'] += 1
                    summaries.append(summary)
        all_summaries[view][subset] = summaries
    for name, distributions, seeds in [('celeba_iid_ten_seed', ['IID'], SEEDS['ten']), ('celeba_noniid_ten_seed', ['non-IID'], SEEDS['ten']), ('celeba_iid_noniid_ten_seed', DISTS, SEEDS['ten']), ('celeba_iid_noniid_exclude_selection', DISTS, SEEDS['nonselection_nine']), ('celeba_iid_noniid_matching_six', DISTS, SEEDS['matching_six'])]:
        table_ns['render'](name, distributions, seeds, images=True)
        table_files.extend([view + '/' + name + ext for ext in ['.md', '.tex', '.png']])
    write(view + '/source_snapshot.json', data)
write('summary_statistics.json', all_summaries)
(OUT / 'renderer_presentation_only.diff').write_text(''.join(diff_blocks), encoding='utf-8')

# Compare every native full-precision summary to the original source summaries,
# and every numeric table cell to the previous rendered Markdown/TeX tables.
old_summary_checks = 0
old_summary_roundoff = []
for subset, previous in old['summaries'].items():
    ours = {(g['method'],g['distribution'],g['attack']): g for g in all_summaries['native'][{'all_ten':'ten', 'nonselection_nine':'nonselection_nine', 'matching_six':'matching_six'}[subset]]}
    for prev in previous:
        g = ours[prev['method'],prev['distribution'],prev['attack']]
        require(g['seeds'] == prev['seeds'] and g['n'] == prev['n'], 'Original subset membership changed')
        for k in METRICS:
            require(g[k]['mean'] == prev[k]['mean'], 'Original native mean not exactly equal')
            # This is the original renderer's existing summary audit tolerance;
            # original receipt/native replay still requires actual zero above.
            require(math.isclose(g[k]['sample_sd'], prev[k]['sample_sd'], abs_tol=1e-12), 'Original native sample SD mismatch')
            if g[k]['sample_sd'] != prev[k]['sample_sd']:
                old_summary_roundoff.append({'subset':subset, 'method':g['method'], 'distribution':g['distribution'], 'attack':g['attack'], 'metric':k, 'original_snapshot_sample_sd':prev[k]['sample_sd'], 'original_renderer_sample_sd':g[k]['sample_sd'], 'difference':g[k]['sample_sd']-prev[k]['sample_sd']})
            old_summary_checks += 1
numeric_table_checks = 0
for p in (OUT / 'native').glob('*.md'):
    prior = read(OLD / p.name).decode('utf-8-sig')
    ours = p.read_text(encoding='utf-8')
    a = [line for line in prior.splitlines() if line.startswith('|')]
    b = [line for line in ours.splitlines() if line.startswith('|')]
    require(a == b, 'Native displayed table differs: ' + p.name)
    numeric_table_checks += sum('±' in c for line in b for c in line.split('|'))
write('old_native_snapshot_sampleSD_roundoff.json', {'note': 'All original 2700 record metrics, 810 summary means and five displayed native tables are exactly equal. The original table renderer uses statistics.stdev; 94 original snapshot SD values differ only in final floating-point bits. No values are substituted or adjusted.', 'original_renderer_audit_tolerance':1e-12, 'differences':old_summary_roundoff, 'max_absolute_difference':max(abs(r['difference']) for r in old_summary_roundoff) if old_summary_roundoff else 0})

flat_columns = ['id','method','source_method','distribution','attack','seed','view','accuracy','aeod','aspd','positive_rate','constant_predictions','fairness_status','checkpoint_sha256','training_torch','inference_device','inference_torch','receipt_sha256','prediction_arrays_sha256','archive_sha256','offserver_proof_sha256','strict_or_diagnostic_sha256']
with (OUT / 'records_three_views_2700.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=flat_columns)
    w.writeheader()
    for r in records.values():
        for view in VIEWS:
            w.writerow({**{k:r[k] for k in ['id','method','source_method','distribution','attack','seed','checkpoint_sha256','training_torch','receipt_sha256','prediction_arrays_sha256']}, 'view': view, **{k:r['views'][view][k] for k in ['accuracy','aeod','aspd','positive_rate','constant_predictions','fairness_status']}, 'inference_device': r['runtime']['device'], 'inference_torch': r['runtime']['torch'], 'archive_sha256': r['source_binding']['archive']['sha256'], 'offserver_proof_sha256': r['source_binding']['offserver_proof']['sha256'], 'strict_or_diagnostic_sha256': r['source_binding']['strict_or_diagnostic']['sha256']})

constant = {v: [r['id'] for r in records.values() if r['views'][v]['constant_predictions']] for v in VIEWS}
coverage_rows = []
for m in METHODS:
    for d in DISTS:
        for a in ATTACKS:
            rs = [r for r in records.values() if (r['method'],r['distribution'],r['attack']) == (m,d,a)]
            coverage_rows.append({'method': m, 'distribution': d, 'attack': a, 'n': len(rs), 'seeds': sorted(r['seed'] for r in rs), 'IDs': [r['id'] for r in rs], 'inference_devices': dict(Counter(r['runtime']['device'] for r in rs)), 'training_torch': dict(Counter(r['training_torch'] for r in rs))})
write('coverage_alias_environment.json', {'grid': coverage_rows, 'method_aliases': [dict(method=m, source_method=s, n=n) for (m,s),n in sorted(Counter((r['method'],r['source_method']) for r in records.values()).items())], 'old_ID_aliases': [{'inventory_id':r['id'],'old_table_id':r['old_table_id'],'cell':r['scientific_cell'],'checkpoint_sha256':r['checkpoint_sha256']} for r in records.values()], 'inference_device_counts': device_counts, 'original_training_environment_counts': training_counts, 'inference_environment_counts': dict(Counter(r['runtime']['torch'] for r in records.values())), 'constant_predictions_retained': constant})
write('input_files_SHA256.json', pins)
write('verification.json', {'status': 'OFFLINE900_THREE_VIEWS_AND_ORIGINAL_NATIVE_TABLE_EXACT_PASS', 'collector_sha256': FINAL_SHA, 'records': 900, 'unique_IDs': 900, 'scientific_cells': 90, 'seeds_each': 10, 'views': VIEWS, 'metric_definition': 'AEOD = absolute TPR gap, ASPD = absolute positive-rate gap', 'actual_array_checks': checks, 'archive_count': len(archives), 'archive_members_rehashed': sum(a['member_n'] for a in archives), 'inference_device_counts': device_counts, 'original_training_environment_counts': training_counts, 'original900_native_record_metrics_exact': 2700, 'original900_native_summary_mean_exact': old_summary_checks, 'original900_native_sampleSD_checked': old_summary_checks, 'original900_native_sampleSD_last_bit_differences': len(old_summary_roundoff), 'original_native_displayed_numeric_cells_exact': numeric_table_checks, 'independent_numpy_audit': dict(independent), 'independent_max_mean_abs_difference': max_mean_diff, 'independent_max_sampleSD_abs_difference': max_sd_diff, 'statistics': 'Original statistics.mean/stdev via original values_for; independent numpy mean/std(ddof=1)', 'fixed_seed_panels': SEEDS, 'same_checkpoint_all_views': True, 'main_endpoint_status': 'NATIVE_VS_SHARED_CALIBRATION_PENDING_USER_DECISION', 'new_CNN_inference': 0, 'root_refit': False, 'test_evaluation_performed': False, 'server_operations': 0, 'canonical_changed': False, 'Git_operations': 0, 'uniform_device_comparison': False, 'formal_full17_complete': False, 'visual_QA': 'PENDING'})
print(json.dumps({'status': 'NUMERIC_PASS_RENDERED_PENDING_VISUAL_QA', 'records': len(records), 'device_counts': device_counts, 'training_counts': training_counts, 'independent_pairs': independent, 'tables': len(table_files)//3}), flush=True)
