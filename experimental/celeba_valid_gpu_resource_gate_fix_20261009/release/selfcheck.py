"""Local, no-CNN checks against frozen source and actual saved GPU arrays."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('local_recovery_under_check', HERE / 'recovery.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
read = r.read
need = r.need


def function(path, name):
    text = Path(path).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


def main():
    old = ROOT / 'tmp/celeba_final_valid_replay_20261009'
    proposal = ROOT / 'tmp/celeba_valid_recovery_prepared_20261009'
    m = read(proposal / 'manifest.json'); need(r.sha(proposal / 'manifest.json') == r.PROPOSAL_SHA, 'Wrong prepared476')
    first = next(x['id'] for x in m['records'] if x['classification'] == 'UNEXECUTED_465')
    a = read(HERE / 'APPROVAL_TEMPLATE.json')
    a.update(status='ROOT_APPROVED_GPU_VALID_RECOVERY_V1', implementation_package_sha256='f' * 64, approved_ids=[first], gpu_uuid=read(HERE / 'import_inputs/GPU_approval.json')['gpu_uuid'], execute_new465=True)
    r.validate_review(a, 'f' * 64, m, 'run-chunk', [first])
    rejected = []
    def refuses(name, action):
        try: action()
        except (ValueError, KeyError): rejected.append(name)
        else: raise AssertionError('Unexpected acceptance: ' + name)
    refuses('PREPARED_is_not_approval', lambda: r.validate_review(read(HERE / 'APPROVAL_TEMPLATE.json'), 'f' * 64, m, 'run-chunk', [first]))
    for name, key, value in [('tolerance_drift', 'native_tolerance', 1e-8), ('test_split', 'target_split', 'test'), ('two_GPU_workers', 'max_GPU_workers', 2), ('restart_old872', 'restart_old872', True), ('wrong_prior_collector', 'accepted424_collector_sha256', '0' * 64)]:
        bad = copy.deepcopy(a); bad[key] = value
        refuses(name, lambda bad=bad: r.validate_review(bad, 'f' * 64, m, 'run-chunk', [first]))
    refuses('duplicate_selected_ID', lambda: r.validate_review(a, 'f' * 64, m, 'run-chunk', [first, first]))
    partial = [x['id'] for x in m['records'] if x['classification'] == 'CPU_STRICT_PARTIAL_10_NOT_REGISTERED']
    bad = copy.deepcopy(a); bad['approved_ids'] = partial
    refuses('unreviewed_CPU_import', lambda: r.validate_review(bad, 'f' * 64, m, 'import-cpu-partial', partial))
    refuses('CPU_partial_is_not_fresh_execution', lambda: r.validate_review(bad, 'f' * 64, m, 'run-chunk', partial))
    bad = copy.deepcopy(a); bad['approved_ids'] = [r.FAILED_ID]
    refuses('unreviewed_GPU_import', lambda: r.validate_review(bad, 'f' * 64, m, 'import-gpu-diagnostic', [r.FAILED_ID]))
    original = function(old / 'v3/replay_v3.py', 'accept') + '\n'
    actual = (HERE / 'strict_body.py').read_text()
    restored = actual.replace("proof['status'] == r['status'] == 'DIAGNOSTIC_NATIVE_MATCH'", "proof['status'] == r['status'] == 'NATIVE_VALID_REPLAY_PASS'").replace("r['runtime']['device'] == 'cuda:0' and r['runtime']['cuda_device_count'] == 1", "r['runtime']['device'] == 'cpu' and r['runtime']['cuda_device_count'] == 0").replace('            gpu_receipt_guard(proof, r)\n', '')
    need(restored == original, 'Strict scientific code differs beyond reviewed status/device/guard')
    begin = original.index('            paths = mapping_paths(record, bindings)'); end = original.index('            accepted.append(model_id)', begin)
    block = '\n'.join(x[8:] for x in original[begin:end].splitlines()) + '\n'
    expected = 'def check_saved(record, path, r, proof, args, bindings, core, original, evaluator, ids, y, s):\n' + block + '    return comparison\n'
    need((HERE / 'saved_science.py').read_text() == expected, 'Imported saved science no longer exact17 lines')
    need(r.sha(HERE / 'gpu_replay_body.py') == r.GPU_SHA, 'Actual successful GPU body changed')
    boot_node = ast.parse(function(HERE / 'recovery.py', 'bootstrap'))
    need(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'set_num_interop_threads' for n in ast.walk(boot_node)), 'Known interop bootstrap failure returned')
    body_node = ast.parse((HERE / 'gpu_replay_body.py').read_text())
    gate_calls = [n for n in ast.walk(body_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'resource_gate']
    need([n.args[0].id for n in sorted(gate_calls, key=lambda n: n.lineno)] == ['before', 'after'], 'GPU body no longer gates before/after inference')
    fake_torch = SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1, cuda=SimpleNamespace(device_count=lambda: 1, current_device=lambda: 0), are_deterministic_algorithms_enabled=lambda: True, backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False, deterministic=True, allow_tf32=False), cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    gpu_observation = {'uuid': a['gpu_uuid'], 'action': 'None'}
    fake_subprocess = SimpleNamespace(run=lambda command, **kwargs: SimpleNamespace(stdout=gpu_observation['uuid'] if '--query-gpu=uuid' in command else 'GPU Recovery Action : ' + gpu_observation['action']))
    gate_space = {'need': need, 'os': SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': a['gpu_uuid'], 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}, PRIO_PROCESS=0, getpriority=lambda *_args: 10), 'subprocess': fake_subprocess}
    exec(compile(function(HERE / 'recovery.py', 'GPU_gate'), 'GPU_gate_no_runtime_fixture', 'exec'), gate_space)
    snapshot = {'thread_cpu_affinities': {'11': [105], '12': [105]}, 'training_queue_snapshot': {'failed': []}, 'service': {'stdout': 'RUNNING'}, 'cgroup': {'cpu.max': '12288000 100000', 'memory.max': str(32 * 1024**3), 'memory.current': str(24 * 1024**3)}}
    gate = lambda snap: gate_space['GPU_gate'](SimpleNamespace(torch=fake_torch), a, snap, {'other_declared_threads': 120})
    gate(snapshot)  # Exactly 8GiB is the approved passing boundary.
    for name, mutate in [('live_RAM_below_8GiB', lambda x: x['cgroup'].update({'memory.current': str(24 * 1024**3 + 1)})), ('live_quota_below_budget', lambda x: x['cgroup'].update({'cpu.max': '12000000 100000'})), ('live_quota_unbounded', lambda x: x['cgroup'].update({'cpu.max': 'max 100000'})), ('live_thread_affinity_escape', lambda x: x['thread_cpu_affinities'].update({'12': [105, 106]}))]:
        bad = copy.deepcopy(snapshot); mutate(bad)
        refuses(name, lambda bad=bad: gate(bad))
    gpu_observation['uuid'] = 'GPU-FOREIGN'
    refuses('live_GPU0_UUID_drift', lambda: gate(snapshot))
    gpu_observation['uuid'] = a['gpu_uuid']; gpu_observation['action'] = 'Reset'
    refuses('live_GPU_recovery_required', lambda: gate(snapshot))
    gpu_observation['action'] = 'None'
    diag = read(HERE / 'import_inputs/GPU_diagnostic.json'); approval = read(HERE / 'import_inputs/GPU_approval.json'); boot = read(HERE / 'import_inputs/GPU_bootstrap.json'); resource = read(HERE / 'import_inputs/GPU_resource.json'); final = read(HERE / 'import_inputs/GPU_final_delivery.json')
    gpu_root = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009/attempt2_backup/verified_extract/runs' / r.FAILED_ID
    receipt = read(gpu_root / 'receipt.json')
    evidence = next(x['preserved_evidence_not_cohort_acceptance'] for x in m['records'] if x['id'] == r.FAILED_ID)
    for name, identity in evidence['members'].items(): need(r.sha(gpu_root / name) == identity['sha256'], 'Actual saved diagnostic bytes drifted')
    r.diagnostic_guard(receipt, diag, approval, boot, resource, final, a)
    for name, mutate in [('GPU_count_mismatch', lambda x: x['runtime'].update(cuda_device_count=0)), ('GPU_nice_mismatch', lambda x: x['runtime'].update(nice=0)), ('optimizer_claim', lambda x: x.update(optimizer_created=True)), ('gradient_claim', lambda x: x.update(gradients_created=True)), ('incomplete_valid', lambda x: x.update(valid_n=19866))]:
        bad = copy.deepcopy(receipt); mutate(bad)
        refuses(name, lambda bad=bad: r.diagnostic_guard(bad, diag, approval, boot, resource, final, a))
    bad = copy.deepcopy(boot); bad['restored_cuda_visible_devices'] = 'GPU-FOREIGN'
    refuses('GPU_UUID_drift', lambda: r.diagnostic_guard(receipt, diag, approval, bad, resource, final, a))
    namespace = {'np': np, 'json': json, 'hashlib': hashlib, 'METHODS': {'FedAvg', 'Median', 'FLTrust', 'FairFed', 'FairGuard', 'FLTrust+FairGuard', 'GuardFed-AD2+', 'FedAA', 'LASA'}, 'VIEWS': {'native', 'raw', 'shared_calibration'}}
    for name in ('canonical_sha', 'check_binary', 'group_metrics', 'predict_views', 'evaluate_frozen_predictions'):
        exec(compile(function(old / 'inputs/evaluator.py', name), str(old / 'inputs/evaluator.py'), 'exec'), namespace)
    cache = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009/original_valid_cache.npz'
    need(r.sha(cache) == '39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64', 'Original accepted valid cache drifted')
    fits = copy.deepcopy(receipt['fits'])
    # Only this local saved-array regression restores JSON keys; live strict always refits root.
    for fit in fits.values():
        if fit['thresholds'] is not None: fit['thresholds'] = {int(k): v for k, v in fit['thresholds'].items()}
    with np.load(cache, allow_pickle=False) as labels, np.load(gpu_root / 'validation_predictions.npz', allow_pickle=False) as z:
        predictions = namespace['predict_views'](z['valid_margins'], labels['valid_sensitive'], fits)
        need(all(np.array_equal(predictions[v], z['prediction_' + v]) for v in namespace['VIEWS']), 'Actual savedGPU predictions mismatch')
        scored = namespace['evaluate_frozen_predictions'](predictions, labels['valid_y'], labels['valid_sensitive'])
        need(scored == receipt['views'], 'Actual9 metrics/24 confusion counts disagree')
        bad_fits = copy.deepcopy(fits); bad_fits['shared_calibration']['thresholds'][0] += 1e-4
        refuses('threshold_identity_drift', lambda: namespace['predict_views'](z['valid_margins'], labels['valid_sensitive'], bad_fits))
        refuses('incomplete_prediction_vector', lambda: namespace['evaluate_frozen_predictions']({'native': predictions['native'][:-1]}, labels['valid_y'], labels['valid_sensitive']))
    native_space = {'np': np, 'require': need, 'TOLERANCE': 1e-12}
    exec(compile(function(old / 'replay.py', 'check_native'), str(old / 'replay.py'), 'exec'), native_space)
    expected_metrics = receipt['native_comparison']['expected']; changed = dict(expected_metrics, accuracy=expected_metrics['accuracy'] + 1 / 19867)
    need(not native_space['check_native'](changed, expected_metrics)['accepted'], 'Actual fixed native check accepted one-sample mismatch')
    refuses('empty_view_set', lambda: namespace['evaluate_frozen_predictions']({}, np.array([0]), np.array([0])))
    with tempfile.TemporaryDirectory(prefix='structural_', dir=HERE) as temp:
        folder = Path(temp).resolve(); need(folder.is_relative_to(HERE), 'Fixture cleanup outside owned folder')
        boot_file, resource_file = folder / (r.FAILED_ID + '.bootstrap.json'), folder / (r.FAILED_ID + '.resource.json')
        boot_file.write_text(json.dumps({'original_import_observation': {'cuda_initialized': False}, 'restored_CUDA_VISIBLE_DEVICES': a['gpu_uuid'], 'restored_intra_threads': 1, 'interop_threads': 1, 'CUDA_or_CNN_before_restoration': False}))
        resource_file.write_text(json.dumps({'CPU': 105, 'gpu_uuid': a['gpu_uuid'], 'other_declared_threads': 11, 'quota': 122.87999, 'main_failed_n': 0}))
        proof = {'id': r.FAILED_ID, 'scope': r.SCOPE, 'GPU_uuid': a['gpu_uuid'], 'implementation_source_sha256': r.sha(HERE / 'recovery.py'), 'metadata_read_receipt': diag['metadata_read'], 'bootstrap_receipt': str(boot_file), 'bootstrap_receipt_sha256': r.sha(boot_file), 'resource_receipt': str(resource_file), 'resource_receipt_sha256': r.sha(resource_file)}
        records = read(ROOT / 'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json')['records']
        record = next(x for x in records if x['id'] == r.FAILED_ID)
        r.receipt_guard(proof, receipt, a, folder, record)
        for name, field in [('mixed_checkpoint', 'checkpoint_sha256'), ('mixed_original_result', 'original_result_sha256'), ('mixed_original_job', 'original_job_sha256')]:
            bad = copy.deepcopy(receipt); bad[field] = '0' * 64
            refuses(name, lambda bad=bad: r.receipt_guard(proof, bad, a, folder, record))
        bad = copy.deepcopy(proof); bad['implementation_source_sha256'] = '0' * 64
        refuses('worker_source_version_drift', lambda: r.receipt_guard(bad, receipt, a, folder, record))
        review = folder / 'review.json'; review.write_text(json.dumps(read(HERE / 'APPROVAL_TEMPLATE.json')))
        args = type('Args', (), {'review': review, 'review_sha256': r.sha(review), 'package_sha256': 'f' * 64, 'command': 'run-chunk', 'output': folder / 'forbidden_output'})()
        need(r.preserve_outer_failure(args, ValueError('fixture'), 'fixture trace') is None and not args.output.exists(), 'Unapproved failure path created scientific output')
    need('torch' not in sys.modules, 'Local structural/saved-array checks imported Torch')
    result = {'status': 'PASS_LOCAL_NO_CNN_NO_COHORT_ACCEPTANCE', 'first_new_ID': first, 'exact_v3_strict_science_except_declared_GPU_guards': True, 'exact_saved_scientific_lines': 17, 'GPU_body_SHA': r.GPU_SHA, 'bootstrap_duplicate_interop_setter': False, 'before_after_GPU_resource_gates_present': True, 'RAM_exact_8GiB_boundary_passed': True, 'actual_saved_GPU_metric_checks': 9, 'actual_saved_GPU_confusion_count_checks': 24, 'actual_saved_GPU_prediction_rules_checked': 3, 'refusal_checks': rejected, 'native_one_sample_mismatch_accepted': False, 'Torch_imported': False, 'new_inference': 0, 'accepted424_unchanged': True, 'target_Linux_runtime_verified': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
