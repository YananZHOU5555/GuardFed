"""Original800 formal binding to original v4 strict and unchanged CUDA/saved science.

No CLI/runner. Import uses stdlib only. Root supplies a reviewed live v4 module,
actual dependency/input arguments, formal authorization and fresh resource guard.
"""
from pathlib import Path, PurePosixPath
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time
import types

HERE = Path(__file__).resolve().parent
SCOPE = 'UNIFORM_CUDA_ORIGINAL800_FORMAL_V1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(pin):
    need(set(pin) == {'path', 'sha256'}, 'Exact path/SHA pin required')
    p = Path(pin['path'])
    need(p.is_file() and not p.is_symlink() and sha(p) == pin['sha256'], 'Unbound or changed input')
    return p


def metadata_gate(approval, scope, record, now):
    """Pure metadata gate; positive fixtures are not actual authorization."""
    methods = {'FedAvg', 'FairFed', 'Median', 'FLTrust', 'FairGuard', 'FLTrust+FairGuard', 'FedAA', 'LASA'}
    need(approval.get('status') == 'ROOT_AUTHORIZED_UNIFORM_CUDA_ORIGINAL800_FORMAL_V1'
         and approval.get('execution_authorized') is True, 'No actual formal root approval')
    need(approval.get('decision') == 'B' and approval.get('author_decision_sha256') ==
         'a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce', 'Wrong author decision')
    need(scope.get('status') == 'ROOT_FROZEN_ORIGINAL800_INPUT_OBJECTS_NOT_CUDA_RESULTS'
         and scope.get('selection_basis') == 'all_original900_records_except_GuardFed-AD2+'
         and scope.get('outcome_independent') is True and scope.get('CUDA_results') is None,
         'No accepted-outcome filtering or diagnostic reuse')
    rows = scope['records']
    need(len(rows) == 800 and len({r['id'] for r in rows}) == 800
         and {r['method'] for r in rows} == methods
         and all(sum(r['method'] == method for r in rows) == 100 for method in methods), 'Exact original800 required; Full/FL/LoGo excluded')
    ids = [r['id'] for r in rows]
    need(scope.get('ordered_ids') == approval.get('ordered_input_ids') == ids
         and scope.get('Full_excluded_and_reused_separately') is True, 'Root must bind original ordered800')
    selected = next((r for r in rows if r['id'] == record['id']), None)
    need(selected == record and record['method'] in methods and approval.get('id') == record['id'], 'Foreign or modified original record')
    need((record['terminal_round'], record['original_split'], record['original_n_eval']) == (70, 'valid', 19867), 'Wrong terminal/split')
    contract = record['data_contract']
    need(contract['actual_train_rows'] == 162770 and contract['actual_evaluation_rows'] == 19867
         and contract['evaluation_split'] == 'valid' and contract['train_eval_disjoint'] is True
         and contract['root_client_disjoint'] is True and len(contract['client_sample_counts']) == 20,
         'Wrong original root/train/valid contract')
    need(contract['train_image_ids_sha256'] == '46d42484d5b5f53af8747fcf44ee11a0b051af27f10383d32254faee0311bc99'
         and contract['evaluation_image_ids_sha256'] == '64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf'
         and isinstance(contract['root_image_ids_sha256'], str) and len(contract['root_image_ids_sha256']) == 64,
         'Wrong original image order/root identity')
    need(record['training_torch'] in ('2.11.0+cu128', '2.11.0+cu130')
         and record['config']['device'] in ('cuda', 'cuda:0'), 'Preserve actual training identity')
    record_sha = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    need(approval.get('record_canonical_sha256') == record_sha, 'Wrong complete original record object')
    need(approval.get('native_tolerance') == 1e-12 and approval.get('target_split') == 'valid'
         and approval.get('compute_threads') == approval.get('max_processes') == 1
         and approval.get('CNN_dtype') == 'float32'
         and approval.get('postprocessing') == 'ORIGINAL_FROZEN_CPU_NUMPY', 'Changed science/budget/post-device')
    need(approval.get('device') == 'cuda:0' and isinstance(approval.get('gpu_uuid'), str)
         and approval['gpu_uuid'].startswith('GPU-'), 'Actual GPU UUID required')
    cpus = approval.get('allowed_cpus')
    need(isinstance(cpus, list) and cpus and all(type(x) is int and x >= 0 for x in cpus)
         and len(cpus) == len(set(cpus)), 'Root CPU affinity missing')
    for key in ('training', 'test', 'automatic_retry', 'diagnostic_reuse'):
        need(approval.get(key) is False, 'Forbidden operation: ' + key)
    namespace = approval.get('output_namespace')
    need(isinstance(namespace, str) and namespace.startswith('/workspace/guardfed_checks/uniform_cuda_formal_original800_')
         and '..' not in PurePosixPath(namespace).parts
         and approval.get('output') == str(PurePosixPath(namespace) / 'runs' / record['id']), 'New formal namespace required')
    need(approval.get('receipt_version') == SCOPE and approval.get('old_evidence_preserved') is True
         and type(approval.get('deadline_unix')) in (int, float) and now < approval['deadline_unix'], 'Expired or diagnostic approval')
    return selected

def bind_old800(v4, args, exactcohortpin, actualapprovalpin, resource_gate):
    """Future root API. No model/image access occurs merely by importing this module."""
    need(sys.platform == 'linux' and not sys.flags.optimize, 'Original strict Linux runtime required')
    p = pinned(actualapprovalpin)
    approval = json.loads(p.read_bytes())
    pins = json.loads((HERE / 'SOURCE_PINS.json').read_bytes())
    v3, v2 = v4.v3, v4.v2
    dependencies = approval['runtime_dependencies']
    for key, module in [('v4', v4), ('v3', v3), ('v2', v2)]:
        actual = pinned(dependencies[key])
        need(sha(actual) == pins[key]['sha256'] and Path(module.__file__).resolve() == actual.resolve(), 'Wrong actual dependency/module: ' + key)
    need((v4.BASE / 'v3/replay_v3.py').resolve() == Path(v3.__file__).resolve()
         and (v4.BASE / 'replay.py').resolve() == Path(v2.__file__).resolve()
         and v3.BASE.resolve() == v4.BASE.resolve(), 'Preserve original v4/v3/v2 layout')
    need(args.command != 'inspect', 'Inspect command would bypass original900 semantic gate')
    semantic_path = pinned(dependencies['semantic_inspection'])
    need(sha(semantic_path) == pins['semantic_inspection']['sha256'], 'Original real900 semantic receipt changed')
    # Private global binding retains both original code objects and shared v4.EXTRA.
    extra = types.SimpleNamespace(semantic_inspection=semantic_path,
                                  semantic_inspection_sha256=dependencies['semantic_inspection']['sha256'])
    read_inputs = v3.private(v4.read_inputs, EXTRA=extra)
    runtime_source_pins = v3.private(v4.source_pins, EXTRA=extra)
    scope_path = pinned(exactcohortpin)
    need(approval['exact_cohort'] == exactcohortpin, 'Approval/cohort path and SHA differ')
    scope = json.loads(scope_path.read_bytes())
    author_path = pinned(approval['author_decision'])
    need(sha(author_path) == pins['author_decision']['sha256'], 'Wrong actual author source')
    author = json.loads(author_path.read_bytes())
    need(author['approved']['uniform_CUDA_evaluation'] and author['approved']['corresponding_Full_included']
         and not author['existing_diagnostic_retroactively_adopted'], 'Author scope changed')
    need(sha(__file__) == approval['adapter_sha256'], 'Original800 adapter changed')
    seal = json.loads(pinned(approval['source_seal']).read_bytes())
    for name, item in seal['files'].items():
        need(Path(name).name == name and sha(HERE / name) == item['sha256'], 'Local source seal changed')
    preflight_path = pinned(approval['Linux_preflight'])
    preflight = json.loads(preflight_path.read_bytes())
    need(preflight.get('pass') is True and 0 <= time.time() - preflight.get('observed_unix', 0) <= 90
         and preflight.get('input_scope_sha256') == sha(scope_path)
         and preflight.get('adapter_sha256') == sha(__file__)
         and preflight.get('gpu_uuid') == approval.get('gpu_uuid')
         and preflight.get('allowed_cpus') == approval.get('allowed_cpus'), 'Fresh resource/source identity missing')
    need(callable(resource_gate) and Path(inspect.getsourcefile(resource_gate)).resolve()
         == pinned(approval['resource_gate_source']).resolve(), 'Unreviewed resource gate')
    for field, role in [('inventory', 'baseline900_inventory'), ('storage_map', 'baseline_storage_map'),
                        ('restore_receipt', 'baseline_restore_receipt'), ('restore_acceptance', 'baseline_restore_acceptance')]:
        need(sha(getattr(args, field)) == pins[role]['sha256'], 'Original900 accepted lineage input changed: ' + field)
    # Entire original900 storage validation and original checked_result remain active.
    records, bindings = read_inputs(args)
    original800 = [r for r in records if r['method'] != 'GuardFed-AD2+']
    need(scope['records'] == original800 and scope.get('original_inventory_sha256') == pins['baseline900_inventory']['sha256'],
         'Exact original ordered800 objects required; no outcome/device filtering')
    record = next(r for r in records if r['id'] == approval['id'])
    selected = metadata_gate(approval, scope, record, time.time())
    need(v2.canonical(record) == approval['record_canonical_sha256'], 'Original800 inventory record changed')
    need(v2.torch.get_num_threads() == v2.torch.get_num_interop_threads() == 1
         and sys.modules.get('replay') is v2 and sys.modules.get('replay_v3') is v3, 'Root must bootstrap one original replay/v3 instance')
    need(set(os.sched_getaffinity(0)) == set(approval['allowed_cpus']), 'Wrong actual CPU affinity')
    paths = v3.mapping_paths(record, bindings)
    need(paths == v4.expected_paths(record), 'Original v4 storage targets differ')
    source_paths = runtime_source_pins(args.repo.resolve(), records, args)
    source_paths.update({Path(__file__): sha(__file__), HERE / 'SOURCE_PINS.json': sha(HERE / 'SOURCE_PINS.json'),
                         scope_path: sha(scope_path), author_path: sha(author_path), p: actualapprovalpin['sha256'],
                         preflight_path: sha(preflight_path), pinned(approval['resource_gate_source']): approval['resource_gate_source']['sha256']})
    body_path = pinned(approval['formal_GPU_body'])
    need(sha(body_path) == pins['formal_GPU_body']['sha256'], 'Original formal GPU body changed')
    source_paths.update({body_path: sha(body_path), HERE / 'old800_saved_science.py.txt': sha(HERE / 'old800_saved_science.py.txt')})
    artifact_paths = {paths[k]: record[k]['sha256'] for k in v3.KINDS}
    before_source, before_artifacts = v3.full_hashes(source_paths), v3.full_hashes(artifact_paths)
    original = v2.load('uniform_cuda_original800_original_strict', args.repo / 'scripts/run_revision_ablation.py')
    validate, _ = v4.mapped_functions(record, paths, original)
    core = v2.load('uniform_cuda_original800_core', args.repo / 'scripts/reproduce_paper_tables.py')
    evaluator_path = pinned(dependencies['evaluator'])
    need(evaluator_path.resolve() == (v3.BASE / 'inputs/evaluator.py').resolve()
         and sha(evaluator_path) == pins['evaluator']['sha256'], 'Original frozen postprocessor source/path changed')
    evaluator = v2.load('uniform_cuda_original800_evaluator', evaluator_path)
    cnn = v2.load('uniform_cuda_original800_cnn', args.repo / 'src/celeba_data.py')

    def fresh(snapshot):
        need(time.time() < approval['deadline_unix'], 'Formal execution authorization expired')
        t = v2.torch
        need(t.__version__ == '2.11.0+cu128' and t.cuda.device_count() == 1 and t.cuda.current_device() == 0
             and os.environ.get('CUDA_VISIBLE_DEVICES') == approval['gpu_uuid'], 'Actual CUDA execution identity changed')
        need(t.are_deterministic_algorithms_enabled() and t.backends.cudnn.deterministic and not t.backends.cudnn.benchmark
             and not t.backends.cudnn.allow_tf32 and not t.backends.cuda.matmul.allow_tf32
             and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Original deterministic FP32 contract changed')
        resource_gate(snapshot, approval)

    by_member = {record[k]['member']: paths[k] for k in v3.KINDS}
    def located(repo, relative):
        return by_member[str(relative)] if str(relative) in by_member else v2.inside(repo, relative)
    ns = dict(v2.replay_one.__globals__, inside=located, validate_original=validate, resource_gate=fresh)
    exec(compile(body_path.read_bytes(), str(body_path), 'exec'), ns)
    inference = ns['replay_one']
    output = Path(approval['output'])
    failure = output.with_name(output.name + '.formal_failure.json')

    def replay(wall_seconds=1800):
        need(not output.exists() and not failure.exists() and not output.resolve().is_relative_to(args.repo.resolve()), 'Preserve old output/failure')
        need(1 <= wall_seconds <= 3600, 'Bounded original replay wall time required')
        try:
            fresh(None)
            ids, y, s, _ = v2.metadata(args.repo)
            receipt = inference(core, original, cnn, evaluator, record, args.repo, ids, y, s, output, wall_seconds)
            need(v3.full_hashes(source_paths) == before_source and v3.full_hashes(artifact_paths) == before_artifacts, 'Inputs changed')
            v2.save(output / 'formal_binding.json', {'scope': SCOPE, 'id': record['id'], 'approval_sha256': actualapprovalpin['sha256'],
                    'input_scope_sha256': sha(scope_path), 'receipt_sha256': sha(output / 'receipt.json'),
                    'source_before': before_source, 'source_after': v3.full_hashes(source_paths),
                    'artifact_before': before_artifacts, 'artifact_after': v3.full_hashes(artifact_paths), 'cohort_complete': False})
            return receipt
        except BaseException as exc:
            need(not failure.exists(), 'Preserve first formal failure')
            failure.parent.mkdir(parents=True, exist_ok=True)
            with failure.open('x', encoding='utf-8') as f:
                json.dump({'scope': SCOPE, 'id': record['id'], 'error_type': type(exc).__name__, 'error': str(exc), 'automatic_retry': False}, f)
            raise

    def accept_saved():
        need(not failure.exists(), 'Failed formal evaluation cannot be accepted')
        binding, r = v2.read(output / 'formal_binding.json'), v2.read(output / 'receipt.json')
        need(binding['scope'] == SCOPE and binding['id'] == r['id'] == record['id']
             and binding['approval_sha256'] == actualapprovalpin['sha256'] and binding['receipt_sha256'] == sha(output / 'receipt.json'), 'Mixed formal receipt')
        need(r['status'] == 'NATIVE_VALID_REPLAY_PASS' and r['scope'] == 'FORMAL_ORIGINAL800_CUDA_PER_RECORD_V1'
             and r['runtime']['device'] == 'cuda:0' and r['runtime']['cuda_device_count'] == 1
             and not r['optimizer_created'] and not r['gradients_created'] and not r['test_labels_accessed'], 'Wrong runtime/operation')
        need(r['model_inventory_record_sha256'] == v2.canonical(record) and all(r[k] == record[a]['sha256'] for k, a in
             [('checkpoint_sha256', 'checkpoint'), ('original_result_sha256', 'result'), ('original_job_sha256', 'raw_job')]), 'Mixed terminal model')
        need(binding['source_before'] == binding['source_after'] == v3.full_hashes(source_paths)
             and binding['artifact_before'] == binding['artifact_after'] == v3.full_hashes(artifact_paths), 'Frozen inputs changed')
        original_result = validate(original, record, args.repo)
        ids, y, s, _ = v2.metadata(args.repo)
        cfg = core.ExperimentConfig(**record['config'])
        root_ids, root_y, root_s, root_receipt = v2.rebuild_root(core, cfg, record, ids, y, s)
        need(root_receipt == r['root_reconstruction'] and r['weights_before'] == r['weights_after'], 'Root or weights changed')
        context = dict(v3.__dict__, record=record, path=output, r=r, core=core, evaluator=evaluator, cfg=cfg,
                       root_ids=root_ids, root_y=root_y, root_s=root_s, ids=ids, y=y, s=s, original_result=original_result)
        exec(compile(textwrap.dedent((HERE / 'old800_saved_science.py.txt').read_text()), '<original v3 original800 saved scientific block>', 'exec'), context)
        need(v3.full_hashes(source_paths) == before_source and v3.full_hashes(artifact_paths) == before_artifacts, 'Acceptance inputs changed')
        return {'scope': SCOPE, 'id': record['id'], 'status': 'PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED',
                'views': context['scored'], 'native_comparison': context['comparison'], 'cohort_complete': False,
                'new_training': False, 'test': False, 'binding_sha256': sha(output / 'formal_binding.json')}

    return {'record': record, 'validate_original': validate, 'replay_one': replay, 'accept_saved_predictions': accept_saved}
