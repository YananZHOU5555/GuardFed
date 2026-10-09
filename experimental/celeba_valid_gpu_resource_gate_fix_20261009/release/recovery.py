"""Bounded GPU validation recovery. Every live operation requires external root review."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REMOTE = Path('/workspace/guardfed_checks/celeba_valid_gpu_resource_gate_fix_20261009/release')
PROPOSAL = Path('/workspace/guardfed_checks/celeba_valid_recovery_prepared_20261009')
BASE = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
REPO = Path('/workspace/GuardFed-celeba-expanded')
PROPOSAL_SHA = '125ab344736a0bf306be99fefc71ca701daf7fc0c337f76b3da7dd62e24c5058'
PROPOSAL_SEAL = '586d4443ae6f074e686f872400dd90b91356f0e504d29faa8c8cb0ff3ea8e26a'
V2_SHA = '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'
V3_SHA = 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'
V4_SHA = '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e'
GPU_SHA = '2aafb0d08b2a0c84bbcfc224638dc216329f72d450facf37fd2e50d1cb0f0ad5'
ARCHIVER_SHA = 'c06d145a1c5b97014bc5d6e7a25c53da1020ec298cae9285f0d8d02da1cd4d00'
SCOPE = 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_RESOURCE_GUARD_V2'
FAILED_ID = 'FairGuard_IID_F-Flip_seed91009'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def main_health(service, queue, phase):
    active, failed = queue.get('active'), queue.get('failed')
    fields = service.get('stdout', '').split()
    inputs = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': phase,
              'service': service, 'queue_snapshot': queue, 'active_n': len(active) if isinstance(active, list) else None,
              'failed_n': len(failed) if isinstance(failed, list) else None,
              'required': {'service': 'RUNNING', 'failed_n': 0, 'active_min': 1, 'active_max': 8}}
    if not (service.get('returncode') == 0 and len(fields) > 1 and fields[1] == 'RUNNING'
            and failed == [] and isinstance(active, list) and 1 <= len(active) <= 8):
        error = ValueError('Protected main800 health failed')
        error.resource_guard_inputs = inputs
        raise error
    return inputs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_review(a, package_sha, manifest, command, selected):
    need(a.get('status') == 'ROOT_APPROVED_GPU_VALID_RECOVERY_V1', 'Explicit reviewed root authorization absent')
    need(a.get('implementation_package_sha256') == package_sha and a.get('proposal_manifest_sha256') == PROPOSAL_SHA, 'Root source/proposal binding changed')
    need(a.get('inventory_sha256') == manifest['inventory_sha256'] and a.get('accepted424_collector_sha256') == manifest['accepted424_collector_sha256'], 'Root cohort binding changed')
    need(a.get('native_tolerance') == 1e-12 and a.get('target_split') == 'valid' and a.get('views') == ['native', 'raw', 'shared_calibration'], 'Scientific tolerance/split/views changed')
    need(a.get('CPU') == 105 and a.get('threads') == 1 and a.get('interop_threads') == 1 and a.get('host_GPU_index') == 0 and a.get('max_GPU_workers') == 1, 'Only oneGPU/CPU105/one-thread approved boundary')
    need(a.get('gpu_uuid', '').startswith('GPU-') and a.get('nice') == 10 and a.get('io_class') == 'idle', 'Actual approved GPU/nice/IO identity absent')
    for key in ('test', 'training', 'automatic_retry', 'restart_old872', 'change_accepted424', 'final_freeze'):
        need(a.get(key) is False, 'Prohibited operation: ' + key)
    rows = {r['id']: r for r in manifest['records']}
    allowed = a.get('approved_ids', [])
    need(allowed and len(allowed) == len(set(allowed)) and set(allowed) <= set(rows), 'Unknown/duplicate/empty reviewed IDs')
    need(selected and len(selected) == len(set(selected)) and set(selected) <= set(allowed), 'Unknown/duplicate/unreviewed selected IDs')
    need(not set(selected) & set(manifest['accepted424_ids']), 'Do not replay accepted424')
    if command in ('run-chunk', 'worker'):
        need(a.get('execute_new465') is True and len(selected) <= 11 and all(rows[i]['classification'] == 'UNEXECUTED_465' for i in selected), 'No new execution or automatic rerun of partial/diagnostic IDs')
    elif command == 'import-cpu-partial':
        need(a.get('import_cpu_partial10') is True and set(selected) == {i for i, r in rows.items() if r['classification'] == 'CPU_STRICT_PARTIAL_10_NOT_REGISTERED'}, 'CPU partial import requires reviewed exact10')
    elif command == 'import-gpu-diagnostic':
        need(a.get('import_gpu_diagnostic1') is True and selected == [FAILED_ID], 'Diagnostic import requires reviewed exactone')
    else:
        need(a.get('execute_new465') is True and len(selected) <= 11 and all(rows[i]['classification'] == 'UNEXECUTED_465' for i in selected), 'Strict/backup outside reviewed freshGPU chunk')


def contract(args, selected):
    need(sha(HERE / 'PACKAGE_SHA256.json') == args.package_sha256, 'Externally reviewed package SHA mismatch')
    for name, row in read(HERE / 'PACKAGE_SHA256.json')['members'].items():
        p = HERE / name
        need(p.is_file() and not p.is_symlink() and sha(p) == row['sha256'] and p.stat().st_size == row['bytes'], 'Implementation source drift: ' + name)
    need(sha(PROPOSAL / 'PACKAGE_SHA256.json') == PROPOSAL_SEAL and sha(PROPOSAL / 'manifest.json') == PROPOSAL_SHA, 'Sealed476 preparation changed')
    for name, row in read(PROPOSAL / 'PACKAGE_SHA256.json')['members'].items():
        need(sha(PROPOSAL / name) == row['sha256'], 'Sealed476 member changed: ' + name)
    need(args.review.is_file() and sha(args.review) == args.review_sha256, 'External root review bytes changed')
    a, m = read(args.review), read(PROPOSAL / 'manifest.json')
    validate_review(a, args.package_sha256, m, args.command, selected)
    need(HERE == REMOTE and sys.platform == 'linux' and not sys.flags.optimize, 'Exact isolated Linux path without -O required')
    need(sha('/etc/vast-agents-guide.md') == '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa', 'Guide changed; reread before new reviewed execution')
    parent = Path(a['output_parent'])
    need(parent.is_absolute() and parent.is_relative_to(Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009')) and '..' not in parent.parts, 'Output outside new recovery namespace')
    return a, m


def bind_cpu():
    need(105 in os.sched_getaffinity(0), 'ApprovedCPU105 unavailable')
    for task in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name), {105})
        except ProcessLookupError:
            pass
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 10:
        os.nice(10 - current)
    need(os.getpriority(os.PRIO_PROCESS, 0) == 10, 'Actual CNN nice must equal10')
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    return [105]


def bootstrap(gpu_uuid=None):
    bind_cpu()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid or '', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
    need(sha(BASE / 'replay.py') == V2_SHA and sha(BASE / 'v3/replay_v3.py') == V3_SHA and sha(BASE / 'v4/replay_v4.py') == V4_SHA, 'Original v2/v3/v4 source drift')
    sys.path.insert(0, str(BASE))
    import replay as v2
    need(sha(v2.__file__) == V2_SHA and not v2.torch.cuda.is_initialized(), 'Foreign v2 or CUDA initialized before restoration')
    observed = {'intra_threads': v2.torch.get_num_threads(), 'interop_threads': v2.torch.get_num_interop_threads(), 'cuda_initialized': False, 'import_CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid or '', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    v2.torch.set_num_threads(1)
    need(v2.torch.get_num_interop_threads() == 1, 'Original v2 interop must already be1; never call setter twice')
    bind_cpu()
    v4 = v2.load('gpu_recovery_original_v4', BASE / 'v4/replay_v4.py')
    need(v4.v2 is v4.v3.v2 is sys.modules['replay'] is v2, 'v3/v4 imported another v2')
    need(v2.torch.__version__ == '2.11.0+cu128', 'Use existing isolated cu128 without dependency changes')
    return v4, {'original_import_observation': observed, 'restored_intra_threads': 1, 'interop_threads': 1, 'restored_CUDA_VISIBLE_DEVICES': gpu_uuid or '', 'transient_original_import_intra_env8_disclosed': True, 'CUDA_or_CNN_before_restoration': False}


def scientific_args(m, command):
    return SimpleNamespace(**{k.replace('-', '_'): Path(v) if not k.endswith('sha256') else v for k, v in m['scientific_cli_bindings_unchanged'].items()}, command=command)


def original_inputs(v4, m, command):
    args = scientific_args(m, command)
    v4.EXTRA = SimpleNamespace(semantic_inspection=args.semantic_inspection, semantic_inspection_sha256=args.semantic_inspection_sha256)
    records, bindings = v4.read_inputs(args)
    return args, records, bindings


def source_pins(v4, m, args, records, review):
    pins = v4.source_pins(REPO, records, args)
    pins.update({HERE / n: r['sha256'] for n, r in read(HERE / 'PACKAGE_SHA256.json')['members'].items()})
    pins.update({HERE / 'PACKAGE_SHA256.json': sha(HERE / 'PACKAGE_SHA256.json'), PROPOSAL / 'manifest.json': PROPOSAL_SHA, PROPOSAL / 'PACKAGE_SHA256.json': PROPOSAL_SEAL, review: sha(review)})
    return pins


def resource_preflight(a, allowed_parent_pid=None):
    tasks, used = [], 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in (os.getpid(), allowed_parent_pid):
            continue
        try:
            argv = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'python' not in argv or not any(s in argv for s in ('/workspace/GuardFed-', '/workspace/guardfed_checks/')):
                continue
            affinity = sorted(os.sched_getaffinity(int(p.name)))
            need(not (len(affinity) <= 16 and 105 in affinity), 'Another project process ownsCPU105; no duplicate worker')
            env = dict(s.split('=', 1) for s in (p / 'environ').read_text().split('\0') if '=' in s)
            threads = max([int(env[k]) for k in ('GUARDFED_CPU_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS') if env.get(k, '').isdigit()] or [1])
            used += threads
            tasks.append({'pid': int(p.name), 'declared_upper_threads': threads})
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    need(quota != 'max' and used + 1 <= int(quota) / int(period), 'Actual CPU quota/budget failed')
    command = lambda argv: subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()
    gpu = command(['nvidia-smi', '--id=0', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits']).split(',')
    detail = command(['nvidia-smi', '--id=0', '-q'])
    need(gpu[0].strip() == a['gpu_uuid'] and int(gpu[1]) >= 2048, 'ApprovedGPU0 identity/headroom failed')
    need([x.split(':', 1)[1].strip() for x in detail.splitlines() if 'GPU Recovery Action' in x] == ['None'], 'GPU requires recovery')
    mem, limit = int(Path('/sys/fs/cgroup/memory.current').read_text()), int(Path('/sys/fs/cgroup/memory.max').read_text())
    need(limit - mem >= 8 * 1024**3, 'RAM headroom failed')
    observed = subprocess.run(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'], check=False, capture_output=True, text=True)
    service = {'returncode': observed.returncode, 'stdout': observed.stdout.strip(), 'stderr': observed.stderr.strip()}
    main = service['stdout']
    queue = read(REPO / 'results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json')
    guard_inputs = main_health(service, queue, 'preflight')
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'CPU': 105, 'quota': int(quota) / int(period), 'other_declared_threads': used, 'tasks': tasks, 'gpu_uuid': a['gpu_uuid'], 'gpu_free_MiB': int(gpu[1]), 'RAM_free_bytes': limit - mem, 'main_service': main, 'main_completed_n': len(queue['completed']), 'main_active_n': len(queue['active']), 'main_failed_n': 0, 'main_guard_inputs': guard_inputs}


def GPU_gate(v2, a, snapshot, resource):
    t = v2.torch
    need(t.get_num_threads() == t.get_num_interop_threads() == 1 and t.cuda.device_count() == 1 and t.cuda.current_device() == 0, 'GPU/thread/device identity failed')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == a['gpu_uuid'] and os.getpriority(os.PRIO_PROCESS, 0) == 10, 'UUID/nice identity failed')
    need(all(c == [105] for c in snapshot['thread_cpu_affinities'].values()), 'A worker thread escapedCPU105')
    need(t.are_deterministic_algorithms_enabled() and not t.backends.cudnn.benchmark and t.backends.cudnn.deterministic and not t.backends.cudnn.allow_tf32 and not t.backends.cuda.matmul.allow_tf32 and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Original deterministic FP32 settings changed')
    main_health(snapshot['service'], snapshot['training_queue_snapshot'], 'CNN_before_or_after')
    cg = snapshot['cgroup']; quota, period = cg['cpu.max'].split()
    need(quota != 'max' and resource['other_declared_threads'] + 1 <= int(quota) / int(period), 'Live inference CPU quota changed')
    need(cg['memory.max'] != 'max' and int(cg['memory.max']) - int(cg['memory.current']) >= 8 * 1024**3, 'Live inference RAM headroom failed')
    actual = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=uuid', '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout.strip()
    detail = subprocess.run(['nvidia-smi', '--id=0', '-q'], check=True, capture_output=True, text=True).stdout
    need(actual == a['gpu_uuid'] and [x.split(':', 1)[1].strip() for x in detail.splitlines() if 'GPU Recovery Action' in x] == ['None'], 'Live GPU identity/recovery changed')


def receipt_guard(proof, r, a, run_parent, record):
    need(proof['scope'] == SCOPE and proof['GPU_uuid'] == a['gpu_uuid'] and proof['implementation_source_sha256'] == sha(__file__), 'GPU proof/version changed')
    need(r['scope'] == 'SINGLE_MODEL_GPU_DIAGNOSTIC_NOT_ACCEPTED_COHORT' and r['native_comparison']['tolerance'] == 1e-12 and r['native_comparison']['accepted'], 'Diagnostic receipt/tolerance mismatch')
    need((r['method'], r['distribution'], r['attack'], r['seed']) == (record['method'], record['distribution'], record['attack'], record['seed']) and r['checkpoint_sha256'] == record['checkpoint']['sha256'] and r['original_result_sha256'] == record['result']['sha256'] and r['original_job_sha256'] == record['raw_job']['sha256'] and r['config_canonical_sha256'] == record['config_canonical_sha256'] and r['original_training_torch'] == record['training_torch'], 'Receipt mixes original cell/checkpoint/result/job/config')
    runtime = r['runtime']
    need(runtime['torch'] == '2.11.0+cu128' and runtime['device'] == 'cuda:0' and runtime['cuda_device_count'] == 1 and runtime['torch_threads'] == runtime['interop_threads'] == 1 and runtime['nice'] == 10 and runtime['loader_workers'] == 0 and runtime['original_config_device'] == 'cuda', 'ExplicitGPU runtime changed')
    metadata = proof['metadata_read_receipt']
    need(not metadata['full_loader_executed'] and all(x['requested_entries'] == 182637 and x['test_entries_requested_or_materialized'] == 0 for x in metadata['prefix_reads']), 'Train/valid-only reader violated')
    need(not r['test_inference_performed'] and not r['final_dispatch_created'], 'Test/final dispatch occurred')
    need(Path(proof['bootstrap_receipt']) == run_parent / (proof['id'] + '.bootstrap.json') and Path(proof['resource_receipt']) == run_parent / (proof['id'] + '.resource.json'), 'Foreign bootstrap/resource paths')
    need(sha(proof['bootstrap_receipt']) == proof['bootstrap_receipt_sha256'] and sha(proof['resource_receipt']) == proof['resource_receipt_sha256'], 'Bootstrap/resource proof changed')
    boot, resource = read(proof['bootstrap_receipt']), read(proof['resource_receipt'])
    need(boot['original_import_observation']['cuda_initialized'] is False and boot['restored_CUDA_VISIBLE_DEVICES'] == a['gpu_uuid'] and boot['restored_intra_threads'] == boot['interop_threads'] == 1 and boot['CUDA_or_CNN_before_restoration'] is False, 'Bootstrap execution boundary changed')
    need(resource['CPU'] == 105 and resource['gpu_uuid'] == a['gpu_uuid'] and resource['other_declared_threads'] + 1 <= resource['quota'] and resource['main_failed_n'] == 0, 'Actual execution resource identity changed')


def worker(args):
    a, m = contract(args, [args.id])
    batch = read(args.batch)
    need(args.batch.name == 'batch_inputs.json' and args.batch.parent.name == 'batch' and args.batch.parent.parent.parent == Path(a['output_parent']) and args.batch.parent.resolve() == args.batch.parent, 'Worker batch outside approved real new chunk')
    need(sha(args.batch) == args.batch_sha256 and batch['review_sha256'] == args.review_sha256 and batch['scope'] == SCOPE and args.id in batch['selected_ids'], 'Worker batch/review identity changed')
    need(batch['implementation_package_sha256'] == args.package_sha256 and batch['proposal_manifest_sha256'] == PROPOSAL_SHA and batch['v3_source_sha256'] == sha(__file__) and batch['manager_pid'] == os.getppid(), 'Worker source/actual parent identity changed')
    need(args.output == args.batch.parent / 'runs' / args.id and args.output.parent.is_dir() and args.output.parent.resolve() == args.output.parent and not args.output.exists(), 'Fresh real worker output required')
    need(not args.output.with_name(args.id + '.worker.json').exists(), 'Never overwrite worker attempt')
    v4, boot = bootstrap(a['gpu_uuid']); v2, v3 = v4.v2, v4.v3
    resource = resource_preflight(a, batch['manager_pid'])
    boot_path = args.output.with_name(args.id + '.bootstrap.json'); resource_path = args.output.with_name(args.id + '.resource.json')
    save(boot_path, boot); save(resource_path, resource)
    inputs, records, bindings = original_inputs(v4, m, 'worker')
    record = next(r for r in records if r['id'] == args.id); paths = v3.mapping_paths(record, bindings)
    pins = {paths[k]: record[k]['sha256'] for k in v3.KINDS}; before = v3.full_hashes(pins)
    proof = {'id': args.id, 'scope': SCOPE, 'status': 'FAILED', 'artifact_before': before, 'GPU_uuid': a['gpu_uuid'], 'implementation_source_sha256': sha(__file__), 'allowed_cpus': [105], 'batch_receipt_sha256': args.batch_sha256, 'storage_map_sha256': inputs.storage_map_sha256, 'bootstrap_receipt': str(boot_path), 'bootstrap_receipt_sha256': sha(boot_path), 'resource_receipt': str(resource_path), 'resource_receipt_sha256': sha(resource_path), 'v2_source_sha256': V2_SHA, 'sealed_v3_source_sha256': V3_SHA, 'sealed_v4_source_sha256': V4_SHA, 'gpu_body_sha256': GPU_SHA, 'review_sha256': args.review_sha256}
    try:
        v3.check_source_tokens(batch['source_before'])
        core = v2.load('recovery_core', REPO / 'scripts/reproduce_paper_tables.py'); original = v2.load('recovery_original', REPO / 'scripts/run_revision_ablation.py'); cnn = v2.load('recovery_cnn', REPO / 'src/celeba_data.py'); evaluator = v2.load('recovery_evaluator', BASE / 'inputs/evaluator.py')
        _, mapped = v4.mapped_functions(record, paths, original)
        namespace = dict(mapped.__globals__, resource_gate=lambda snapshot: GPU_gate(v2, a, snapshot, resource))
        need(sha(HERE / 'gpu_replay_body.py') == GPU_SHA, 'Reviewed GPU science body changed')
        exec(compile((HERE / 'gpu_replay_body.py').read_text(), str(HERE / 'gpu_replay_body.py'), 'exec'), namespace)
        ids, y, sensitive, metadata = v2.metadata(REPO)
        signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError('Bounded GPU worker timed out; no retry')))
        r = namespace['replay_one'](core, original, cnn, evaluator, record, REPO, ids, y, sensitive, args.output, 1800)
        v3.check_source_tokens(batch['source_before'])
        resource_preflight(a, batch['manager_pid'])
        proof.update(status='DIAGNOSTIC_NATIVE_MATCH', receipt_sha256=sha(args.output / 'receipt.json'), metadata_read_receipt=metadata, v4_schema_bridge=v4.BRIDGES[args.id])
        receipt_guard(proof, r, a, args.output.parent, record)
    except BaseException as error:
        signal.alarm(0)
        proof.update(status='FAILED', error=repr(error), traceback=traceback.format_exc(), resource_guard_inputs=getattr(error, 'resource_guard_inputs', None))
    finally:
        try:
            proof['artifact_after'] = v3.full_hashes(pins); need(proof['artifact_after'] == before, 'Mapped artifacts changed'); proof['artifacts_unchanged'] = True
        except Exception as error:
            proof.update(status='FAILED', artifacts_unchanged=False, identity_error=repr(error))
        save(args.output.with_name(args.id + '.worker.json'), proof)
    return 0 if proof['status'] == 'DIAGNOSTIC_NATIVE_MATCH' else 1


def run_chunk(args):
    a, m = contract(args, args.ids)
    need(args.output.parent == Path(a['output_parent']) and args.output.parent.is_dir() and args.output.parent.resolve() == args.output.parent and not args.output.exists(), 'Fresh chunk under approved real empty-attempt parent required')
    v4, _ = bootstrap(); v3 = v4.v3
    resource = resource_preflight(a)
    inputs, records, _ = original_inputs(v4, m, 'run')
    pins = source_pins(v4, m, inputs, records, args.review); before = v3.full_hashes(pins)
    args.output.mkdir(); batch_dir = args.output / 'batch'; batch_dir.mkdir(); (batch_dir / 'runs').mkdir()
    import fcntl
    lock = (HERE / 'GPU_CPU105.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(args.output / 'launch_resources.json', resource)
    batch = {'scope': SCOPE, 'selected_ids': args.ids, 'source_before': before, 'v3_source_sha256': sha(__file__), 'v2_source_sha256': V2_SHA, 'storage_map_sha256': inputs.storage_map_sha256, 'restore_acceptance_sha256': inputs.restore_acceptance_sha256, 'workers': 1, 'manager_pid': os.getpid(), 'review_sha256': args.review_sha256, 'implementation_package_sha256': args.package_sha256, 'proposal_manifest_sha256': PROPOSAL_SHA, 'final_protocol_frozen': False}
    batch_path = batch_dir / 'batch_inputs.json'; save(batch_path, batch)
    complete, failures, child = [], [], None; started = time.monotonic()
    try:
        for identity in args.ids:
            command = [sys.executable, str(HERE / 'recovery.py'), 'worker', '--review', str(args.review), '--review-sha256', args.review_sha256, '--package-sha256', args.package_sha256, '--id', identity, '--batch', str(batch_path), '--batch-sha256', sha(batch_path), '--output', str(batch_dir / 'runs' / identity)]
            with (batch_dir / 'runs' / (identity + '.log')).open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                code = child.wait()
            if code:
                failures.append({'id': identity, 'returncode': code}); break
            complete.append(identity)
    except BaseException as error:
        failures.append({'error': repr(error)})
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        try:
            after = v3.full_hashes(pins)
        except Exception as error:
            after = None; failures.append({'source_hash_error': repr(error)})
        if before != after: failures.append({'error': 'Shared source/data changed'})
        save(batch_dir / 'batch_execution.json', {'scope': SCOPE, 'requested_ids': args.ids, 'finished_zero_exit_ids': complete, 'failures': failures, 'source_after': after, 'source_unchanged': before == after, 'wall_seconds': time.monotonic() - started, 'workers': 1, 'cohort_registered': False})
    if failures:
        save(args.output / 'failure_receipt.json', {'status': 'FAILSTOP_PRESERVE_NO_RETRY', 'failures': failures, 'completed_exit_only_not_accepted': complete}); return 1
    return 0


def accept(args):
    batch = read(args.batch / 'batch_inputs.json'); a, m = contract(args, batch['selected_ids'])
    need(args.batch.name == 'batch' and args.batch.parent.parent == Path(a['output_parent']) and args.batch.resolve() == args.batch and args.output == args.batch.parent / 'strict_acceptance.json' and not args.output.exists(), 'Strict output/batch outside approved fresh chunk')
    need(batch['review_sha256'] == args.review_sha256 and batch['implementation_package_sha256'] == args.package_sha256, 'Batch root/source binding changed')
    v4, _ = bootstrap(); inputs, records, _ = original_inputs(v4, m, 'accept')
    inputs.batch, inputs.output = args.batch, args.output
    indexed = {r['id']: r for r in records}
    namespace = dict(v4.v3.accept.__globals__, __file__=__file__, SCOPE=SCOPE, read_inputs=lambda _args: v4.read_inputs(_args), source_pins=lambda _repo, _records, _args: source_pins(v4, m, _args, _records, args.review), mapped_functions=v4.mapped_functions, bind_cpu=lambda _slot: bind_cpu(), gpu_receipt_guard=lambda proof, receipt: receipt_guard(proof, receipt, a, args.batch / 'runs', indexed[proof['id']]))
    exec(compile((HERE / 'strict_body.py').read_text(), str(HERE / 'strict_body.py'), 'exec'), namespace)
    return namespace['accept'](inputs)


def backup(args):
    batch = read(args.stage / 'batch/batch_inputs.json'); a, m = contract(args, batch['selected_ids'])
    bind_cpu()
    need(args.stage.parent == Path(a['output_parent']) and args.stage.resolve() == args.stage, 'Foreign backup stage')
    strict = read(args.stage / 'strict_acceptance.json') if (args.stage / 'strict_acceptance.json').is_file() else None
    if strict:
        need(args.strict_sha256 is not None and sha(args.stage / 'strict_acceptance.json') == args.strict_sha256 and strict['batch_inputs_sha256'] == sha(args.stage / 'batch/batch_inputs.json'), 'Externally bound strict/batch SHA required')
    if not args.failure:
        need(strict and strict['scope'] == SCOPE and strict['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED' and strict['accepted_ids'] == batch['selected_ids'] and not strict['invalid'] and strict['max_abs_native_metric_difference'] <= 1e-12, 'Only complete strict GPU chunk can use success archive name')
    path = BASE / 'v4/remaining872_prepared_v2_20261009/bounded_remaining.py'; need(sha(path) == ARCHIVER_SHA, 'Original archive implementation changed')
    archiver = load('gpu_recovery_original_archiver', path)
    files = {**read(PROPOSAL / 'manifest.json')['scientific_source_pins_unchanged'], str(args.review): args.review_sha256, str(PROPOSAL / 'manifest.json'): PROPOSAL_SHA}
    files.update({str(HERE / n): r['sha256'] for n, r in read(HERE / 'PACKAGE_SHA256.json')['members'].items()})
    for name, expected in files.items(): need(sha(name) == expected, 'Backup source changed: ' + name)
    source_archive = {'sourcefreeze/%03d_%s' % (n, Path(name).name): name for n, name in enumerate(sorted(files)) if not Path(name).is_relative_to(REPO / 'data')}
    accepted = strict['accepted_ids'] if strict else []
    result = archiver.archive_chunk({'source_archive': source_archive}, PROPOSAL / 'manifest.json', args.stage, args.stage / 'batch', args.chunk_index, accepted, failed=args.failure)
    print(json.dumps(result)); return 0


def diagnostic_guard(r, diag, approval, boot, resource, final, a):
    need(diag['id'] == r['id'] == final['id'] == FAILED_ID and diag['status'] == 'DIAGNOSTIC_ONLY_RESULT_SAVED' and diag['error'] is None, 'Original diagnostic wrapper identity/incompleteness')
    need(diag['accepted_for_cohort'] is False and diag['old_CPU_failure_still_invalid'] and not diag['old872_restart_authorized'] and diag['new_training'] == diag['test_inference'] == 0, 'Diagnostic scope changed')
    need(final['accepted_cohort_count_unchanged'] == 424 and final['original_CPU_failure_still_invalid'] and len(final['preserved_failures']) == 2, 'Diagnostic completion/failure boundary changed')
    need(final['GPU_native_comparison'] == diag['native_comparison'] == r['native_comparison'] and r['native_comparison']['tolerance'] == 1e-12 and r['native_comparison']['accepted'], 'Original diagnostic native comparison changed')
    runtime = r['runtime']
    need(runtime['torch'] == '2.11.0+cu128' and runtime['device'] == 'cuda:0' and runtime['cuda_device_count'] == 1 and runtime['torch_threads'] == runtime['interop_threads'] == 1 and runtime['nice'] == 10 and runtime['loader_workers'] == 0 and runtime['original_config_device'] == 'cuda', 'Diagnostic actualGPU/count/thread/priority mismatch')
    need(not r['optimizer_created'] and not r['gradients_created'] and not r['test_labels_accessed'] and not r['test_inference_performed'] and not r['final_dispatch_created'] and r['weights_before'] == r['weights_after'], 'Diagnostic inference-only/weights contract failed')
    metadata = diag['metadata_read']
    need(metadata['train_n'] == 162770 and metadata['valid_n'] == r['valid_n'] == 19867 and r['root_reconstruction']['root_n'] == 16277 and not metadata['full_loader_executed'] and not metadata['test_labels_accessed'] and all(x['requested_entries'] == 182637 and x['test_entries_requested_or_materialized'] == 0 for x in metadata['prefix_reads']), 'Diagnostic root/train/valid-only contract failed')
    need(approval['gpu_uuid'] == resource['gpu_uuid'] == boot['restored_cuda_visible_devices'] == a['gpu_uuid'] and approval['host_gpu_index'] == 0 and approval['allowed_cpus'] == boot['cpus'] == [105] and approval['compute_threads'] == approval['max_processes'] == 1, 'Diagnostic original approvedGPU UUID/CPU identity changed')
    need(boot['observed_import_bootstrap']['cuda_initialized'] is False and boot['restored_intra_threads'] == boot['interop_threads'] == 1 and boot['CUDA_or_CNN_inference_before_restoration'] is False, 'Diagnostic bootstrap was not the successful attempt2 boundary')
    need(resource['no_duplicate_diagnostic'] and resource['allowed_cpu105_free'] and resource['gpu_recovery_action'] == 'None' and resource['nominal_threads_before'] + 1 <= resource['cpu_quota_cores'], 'Diagnostic actual execution resource proof failed')
    need(all(c == [105] for snap in (r['before_resources'], r['after_resources']) for c in snap['thread_cpu_affinities'].values()), 'Diagnostic thread affinity escapedCPU105')


def reviewed_import(args):
    m = read(PROPOSAL / 'manifest.json')
    ids = [r['id'] for r in m['records'] if r['classification'] == 'CPU_STRICT_PARTIAL_10_NOT_REGISTERED'] if args.command == 'import-cpu-partial' else [FAILED_ID]
    a, m = contract(args, ids)
    need(args.output.parent == Path(a['output_parent']) and args.output.parent.resolve() == args.output.parent and not args.output.exists(), 'Fresh explicit import report required')
    v4, _ = bootstrap(); v2, v3 = v4.v2, v4.v3
    inputs, records, bindings = original_inputs(v4, m, 'accept')
    references = {x['id']: x['preserved_evidence_not_cohort_acceptance'] for x in m['records'] if x['id'] in ids}
    provenance = read(HERE / 'IMPORT_DEPENDENCIES.json')
    for identity in ids:
        entry = references[identity]
        proof = HERE / provenance['cpu_offserver_proof' if args.command == 'import-cpu-partial' else 'GPU_offserver_proof']
        need(sha(proof) == entry['proof_sha256'], 'Preserved offserver import proof changed')
        source_dir = BASE / 'v4/remaining872_attempt1/chunk_036/batch/runs' / identity if args.command == 'import-cpu-partial' else Path('/workspace/guardfed_checks/celeba_native_mismatch_diagnostic_execution_20261009/runs') / identity
        for name, original in entry['members'].items(): need(sha(source_dir / name) == original['sha256'] and (source_dir / name).stat().st_size == original['bytes'], 'Reviewed import arrays/receipt identity changed')
    if args.command == 'import-cpu-partial':
        inputs.batch = BASE / 'v4/remaining872_attempt1/chunk_036/batch'; inputs.output = args.output.with_name(args.output.stem + '.original_v4_strict.json')
        code = v3.private(v3.accept, **v4.OVERRIDES, bind_cpu=lambda _slot: bind_cpu())(inputs)
        report = read(inputs.output)
        need(code == 1 and set(report['accepted_ids']) == set(ids) and len(report['invalid']) == 1 and report['invalid'][0]['id'] == FAILED_ID and report['max_abs_native_metric_difference'] == 0, 'Original partial strict subset differs; preserve failure')
    else:
        root = Path('/workspace/guardfed_checks/celeba_native_mismatch_diagnostic_execution_20261009'); diag = read(root / 'runs' / (FAILED_ID + '.diagnostic.json'))
        record = next(r for r in records if r['id'] == FAILED_ID); path = root / 'runs' / FAILED_ID; r = read(path / 'receipt.json')
        reference = next(x for x in m['records'] if x['id'] == FAILED_ID)['preserved_evidence_not_cohort_acceptance']
        need(sha(HERE / provenance['GPU_corrected_offline_acceptance']) == reference['separate_corrected_offline_acceptance_sha256'], 'Corrected independent diagnostic comparison changed')
        snapshots = read(HERE / 'IMPORT_DEPENDENCIES.json')['GPU_snapshot_files']
        for item in snapshots.values(): need(sha(HERE / item['file']) == item['sha256'], 'Sealed diagnostic wrapper/root/source snapshot changed')
        need(sha(root / 'runs' / (FAILED_ID + '.diagnostic.json')) == snapshots['diagnostic']['sha256'], 'Actual preserved diagnostic wrapper changed')
        old_approval = read(HERE / snapshots['approval']['file']); old_resource = read(HERE / snapshots['resource']['file']); old_boot = read(HERE / snapshots['bootstrap']['file']); final = read(HERE / snapshots['final_delivery']['file'])
        need(old_approval['resource_receipt_sha256'] == snapshots['resource']['sha256'] and old_approval['root_authorization_sha256'] == snapshots['root_authorization']['sha256'] and old_approval['execution_source_seal_sha256'] == snapshots['source_seal']['sha256'], 'Original root/resource/attempt2 source authorization chain changed')
        source_seal = read(HERE / snapshots['source_seal']['file'])
        need(source_seal['members']['run_once.py']['sha256'] == snapshots['run_once']['sha256'] == 'c0c6b199a4db9abbdcf5d2b1f6489b50608128f3386fe648ba3f7ef0b9d6974b', 'Original successfulGPU wrapper source changed')
        diagnostic_guard(r, diag, old_approval, old_boot, old_resource, final, a)
        for name, identity in reference['members'].items(): need(sha(path / name) == identity['sha256'], 'Preserved diagnostic member changed')
        need(diag['source_before'] == diag['source_after'] and v3.full_hashes({Path(k): x['sha256'] for k, x in diag['source_before'].items()}) == diag['source_before'], 'Diagnostic original source/data/artifacts changed')
        need(r['model_inventory_record_sha256'] == v2.canonical(record) and r['status'] == 'DIAGNOSTIC_NATIVE_MATCH' and r['runtime']['device'] == 'cuda:0' and r['runtime']['torch_threads'] == r['runtime']['interop_threads'] == 1 and r['native_comparison']['tolerance'] == 1e-12, 'Diagnostic identity/runtime/tolerance changed')
        need(r['checkpoint_sha256'] == record['checkpoint']['sha256'] and r['original_result_sha256'] == record['result']['sha256'] and r['original_job_sha256'] == record['raw_job']['sha256'], 'Diagnostic mixes original artifact identity')
        paths = v3.mapping_paths(record, bindings); proof = {'artifact_before': {str(p): diag['source_before'][str(p)] for p in paths.values()}}; proof['artifact_after'] = proof['artifact_before']
        core = v2.load('import_core', REPO / 'scripts/reproduce_paper_tables.py'); original = v2.load('import_original', REPO / 'scripts/run_revision_ablation.py'); evaluator = v2.load('import_evaluator', BASE / 'inputs/evaluator.py'); ids_data, y, sensitive, metadata = v2.metadata(REPO)
        namespace = dict(v3.accept.__globals__, mapped_functions=v4.mapped_functions)
        exec(compile((HERE / 'saved_science.py').read_text(), str(HERE / 'saved_science.py'), 'exec'), namespace)
        namespace['check_saved'](record, path, r, proof, inputs, bindings, core, original, evaluator, ids_data, y, sensitive)
        report = {'id': FAILED_ID, 'native_comparison': r['native_comparison'], 'metadata_read_receipt': metadata, 'GPU_diagnostic_source_before_after_unchanged': True}
    save(args.output, {'scope': SCOPE, 'status': 'REVIEWED_IMPORT_STRICT_MATCH_PENDING_OFFSERVER_AND_REGISTRATION', 'eligible_ids': ids, 'eligible_n': len(ids), 'review_sha256': args.review_sha256, 'implementation_source_sha256': sha(__file__), 'proposal_sha256': PROPOSAL_SHA, 'original_strict_or_diagnostic_check': report, 'accepted424_unchanged': True, 'original_CPU_failure_still_invalid': True, 'new_CNN_inference': 0, 'cohort_registered': False})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run-chunk', 'worker', 'accept', 'backup', 'import-cpu-partial', 'import-gpu-diagnostic'):
        p = sub.add_parser(name); p.add_argument('--review', type=Path, required=True); p.add_argument('--review-sha256', required=True); p.add_argument('--package-sha256', required=True)
        if name == 'run-chunk': p.add_argument('--ids', nargs='+', required=True)
        if name == 'worker':
            p.add_argument('--id', required=True); p.add_argument('--batch', type=Path, required=True); p.add_argument('--batch-sha256', required=True)
        if name == 'accept': p.add_argument('--batch', type=Path, required=True)
        if name == 'backup':
            p.add_argument('--stage', type=Path, required=True); p.add_argument('--chunk-index', type=int, required=True); p.add_argument('--failure', action='store_true'); p.add_argument('--strict-sha256')
        else: p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command in ('run-chunk', 'worker'):
        def stop(_signal, _frame): raise InterruptedError('Explicit stop; preserve partial without retry')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        return {'run-chunk': run_chunk, 'worker': worker, 'accept': accept, 'backup': backup, 'import-cpu-partial': reviewed_import, 'import-gpu-diagnostic': reviewed_import}[args.command](args)
    except BaseException as error:
        trace = traceback.format_exc()
        target = preserve_outer_failure(args, error, trace)
        print(trace, file=sys.stderr)
        if target is not None: print('Structured failure preserved: ' + str(target), file=sys.stderr)
        return 1


def preserve_outer_failure(args, error, trace):
    """Never create a scientific output when approval is absent or untrusted."""
    try:
        need(args.review.is_file() and sha(args.review) == args.review_sha256, 'No trusted review')
        a = read(args.review)
        need(a.get('status') == 'ROOT_APPROVED_GPU_VALID_RECOVERY_V1' and a.get('implementation_package_sha256') == args.package_sha256 and a.get('proposal_manifest_sha256') == PROPOSAL_SHA, 'No trusted root source/output binding')
        parent = Path(a['output_parent']); need(parent.is_absolute() and parent.is_relative_to(Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009')) and '..' not in parent.parts, 'Foreign approved parent')
        base = args.stage / 'backup' if args.command == 'backup' else args.output
        need(base.is_absolute() and base.parent.is_dir() and base.parent.resolve() == base.parent and (base.parent == parent or base.parent.is_relative_to(parent)), 'Failure outside approved real output namespace')
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target = base.with_name(base.name + '.' + args.command + '.' + stamp + '.failure.json')
        save(target, {'status': 'COMMAND_FAILED_PRESERVED_NO_RETRY', 'command': args.command, 'error_type': type(error).__name__, 'error': str(error), 'traceback': trace, 'resource_guard_inputs': getattr(error, 'resource_guard_inputs', None), 'review_sha256': args.review_sha256, 'implementation_package_sha256': args.package_sha256, 'cohort_registered': False, 'accepted424_unchanged': True})
        return target
    except Exception:
        return None


if __name__ == '__main__':
    raise SystemExit(main())
