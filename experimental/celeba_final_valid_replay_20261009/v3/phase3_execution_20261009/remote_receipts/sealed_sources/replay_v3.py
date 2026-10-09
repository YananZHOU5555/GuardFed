"""Explicit storage-map bindings and a bounded foreground valid-only replay runner.

The sealed v2 scientific inference and original checked_result are reused with
private path globals. No original job/config/output identity is rewritten.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import traceback
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import replay as v2

V2_SHA = '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'
ARTIFACT_ROOT = PurePosixPath('/workspace/guardfed_checks/celeba_validation900_restore_20261009')
REPO = PurePosixPath('/workspace/GuardFed-celeba-expanded')
KINDS = ('checkpoint', 'result', 'raw_job')
SCOPE = 'VALID_ONLY_IMPLEMENTATION_REPLAY_V3'
PROTOCOL_SHA = '8ef15af6adec686fecc3462a038edbc94a78456362292cba01f03a27817aa749'


def private(function, **overrides):
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    result = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def storage_bindings(inventory, mapping, bundle, mapping_sha, bundle_receipt_sha, live_acceptance=None):
    """Validate the externally SHA-bound restore chain and all2700 original identities."""
    records = v2.validate_inventory(inventory)
    indexed = {r['id']: r for r in records}
    v2.require(mapping['inventory_sha256'] == bundle['inventory_sha256'] == v2.INVENTORY_SHA, 'Wrong restore inventory')
    v2.require(mapping['artifact_root'] == bundle['artifact_root'] == str(ARTIFACT_ROOT), 'Untrusted artifact root')
    v2.require(mapping['models'] == 900 and mapping['existing_full_models'] == 100 and mapping['isolated_nonfull_models'] == 800, 'Storage cohort counts differ')
    v2.require(bundle['models'] == 800 and bundle['existing_full_models_reused'] == 100, 'Bundle model counts differ')
    v2.require(bundle['storage_map_sha256'] == mapping_sha, 'Bundle does not bind the trusted storage-map SHA')
    rows = mapping['records']
    v2.require(len(rows) == 2700 and len({(r['id'], r['kind']) for r in rows}) == 2700, 'Duplicate or missing model/kind in storage map')
    v2.require({(r['id'], r['kind']) for r in rows} == {(i, k) for i in indexed for k in KINDS}, 'Storage map omits a declared model/seed/kind')
    v2.require(len({r['target'] for r in rows}) == 2700, 'Different artifacts share a storage target')
    bound, nonfull = {}, set()
    for row in rows:
        r, kind = indexed[row['id']], row['kind']
        reference = r[kind]
        v2.require(row['method'] == r['method'] and all(row[k] == reference[k] for k in ('archive', 'archive_sha256', 'member', 'sha256', 'bytes')), 'Mapped artifact differs from accepted original member')
        target = PurePosixPath(row['target'])
        v2.require(target.is_absolute() and '..' not in target.parts, 'Unsafe storage target path')
        full = r['method'] == 'GuardFed-AD2+'
        v2.require(row['existing_full_reuse'] is full, 'Storage reuse marker contradicts method cohort')
        if full:
            expected = REPO / reference['member'] if kind == 'raw_job' else PurePosixPath(r['original_remote_output']) / ('model.pt' if kind == 'checkpoint' else 'result.json')
            v2.require(target == expected, 'Full historical storage path changed')
        else:
            v2.require(target.is_relative_to(ARTIFACT_ROOT / 'artifact_store'), 'NonFull storage escapes isolated artifact store')
            if kind == 'raw_job':
                expected = ARTIFACT_ROOT / 'artifact_store/archived_jobs' / (r['id'] + '.json')
            else:
                expected = ARTIFACT_ROOT / 'artifact_store/original_outputs' / PurePosixPath(r['original_remote_output']).relative_to('/workspace') / ('model.pt' if kind == 'checkpoint' else 'result.json')
            v2.require(target == expected, 'NonFull storage path differs from explicit original output binding')
            relative = target.relative_to(ARTIFACT_ROOT).as_posix()
            v2.require(bundle['members'].get(relative) == {'sha256': row['sha256'], 'bytes': row['bytes']}, 'Storage target missing from accepted restore bundle')
            nonfull.add(relative)
        bound.setdefault(r['id'], {})[kind] = row
    v2.require(nonfull == set(bundle['members']) and len(nonfull) == 2400, 'Restore bundle has unbound, duplicate or missing members')
    for model_id, artifacts in bound.items():
        v2.require(PurePosixPath(artifacts['checkpoint']['target']).parent == PurePosixPath(artifacts['result']['target']).parent, 'Mapped checkpoint and result are from different model directories')
    if live_acceptance is not None:
        a = live_acceptance
        v2.require(a['status'] == 'EXACT_900_ARTIFACT_STORAGE_VERIFIED' and a['models'] == 900 and a['all_artifact_file_hashes_verified'] == 2700, 'Remote restore has not accepted all2700 files')
        v2.require(a['storage_map_sha256'] == mapping_sha and a['receipt_sha256'] == bundle_receipt_sha and a['bundle_sha256'] == bundle['sha256'], 'Remote acceptance chain differs from trusted map/bundle')
        v2.require(a['inventory_sha256'] == v2.INVENTORY_SHA and a['existing_full_files_verified'] == 300 and a['original_output_files_modified'] == 0, 'Remote restore changed original artifacts or Full support')
    return records, bound


def read_inputs(args, require_live=True):
    v2.require(v2.digest(BASE / 'replay.py') == V2_SHA, 'Sealed measured v2 source changed')
    v2.require(v2.digest(args.inventory) == v2.INVENTORY_SHA, 'Original inventory bytes changed')
    v2.require(v2.digest(args.storage_map) == args.storage_map_sha256, 'Storage-map bytes differ from externally supplied SHA')
    v2.require(v2.digest(args.restore_receipt) == args.restore_receipt_sha256, 'Restore-bundle receipt bytes differ from externally supplied SHA')
    acceptance = None
    if require_live:
        v2.require(args.restore_acceptance is not None and args.restore_acceptance_sha256 is not None, 'Externally SHA-bound remote restore acceptance required')
        v2.require(v2.digest(args.restore_acceptance) == args.restore_acceptance_sha256, 'Remote restore acceptance SHA mismatch')
        acceptance = v2.read(args.restore_acceptance)
    return storage_bindings(v2.read(args.inventory), v2.read(args.storage_map), v2.read(args.restore_receipt),
                            args.storage_map_sha256, args.restore_receipt_sha256, acceptance)


def mapping_paths(record, bindings):
    artifacts = bindings[record['id']]
    return {kind: Path(artifacts[kind]['target']) for kind in KINDS}


def mapped_functions(record, paths, original):
    """Reuse the entire sealed v2 validation/inference bodies; only input lookup is bound."""
    by_member = {record[kind]['member']: paths[kind] for kind in KINDS}
    def located(repo, relative):
        return by_member.get(str(relative), v2.inside(repo, relative))
    def original_checked(job):
        # Historical output remains checked in v2.validate_original above this call.
        # Only checked_result's read location uses a separate runtime dictionary.
        runtime_job = {**job, 'output': str(paths['result'].parent)}
        return original.checked_result(runtime_job)
    proxy = types.SimpleNamespace(checked_result=original_checked)
    validated = private(v2.validate_original, inside=located)
    def validate(_original, actual_record, repo):
        v2.require(actual_record == record, 'Bound record changed before original validation')
        return validated(proxy, actual_record, repo)
    inference = private(v2.replay_one, inside=located, validate_original=validate)
    return validate, inference


def stat_token(path):
    stat = path.stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino, 'device': stat.st_dev,
            'resolved_path': str(path.resolve())}


def source_pins(repo, records, args):
    pins = {Path(__file__).resolve(): v2.digest(__file__), BASE / 'replay.py': V2_SHA,
            BASE / 'inputs/evaluator.py': v2.EVALUATOR_SHA, args.inventory.resolve(): v2.INVENTORY_SHA,
            args.storage_map.resolve(): args.storage_map_sha256, args.restore_receipt.resolve(): args.restore_receipt_sha256,
            args.restore_acceptance.resolve(): args.restore_acceptance_sha256,
            BASE / 'inputs/protocol.json': PROTOCOL_SHA,
            repo / 'scripts/run_revision_ablation.py': v2.RUNNER_SHA}
    expected_adapters = set()
    for r in records:
        expected_adapters.update(r['adapter_source_hashes'].values())
        for rel, sha in r['source_hashes'].items():
            path = v2.inside(repo, rel)
            v2.require(path not in pins or pins[path] == sha, 'Conflicting model source identities')
            pins[path] = sha
    found = {}
    if expected_adapters:
        tree = repo / 'deployment/baseline_adapters_20260928'
        for p in sorted(tree.rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.json'):
                sha = v2.digest(p)
                if sha in expected_adapters and sha not in found:
                    found[sha] = p
        missing = expected_adapters - set(found)
        v2.require(not missing, 'Original pinned adapter sources require exact restoration: ' + ','.join(sorted(missing)))
        pins.update({path: sha for sha, path in found.items()})
    return pins


def full_hashes(pins):
    values = v2.hash_pins(pins)
    return {name: {**identity, 'stat': stat_token(Path(name))} for name, identity in values.items()}


def check_source_tokens(expected):
    for name, record in expected.items():
        v2.require(stat_token(Path(name)) == record['stat'], 'Frozen shared input changed during batch: ' + name)


def bind_cpu(slot, inherited=None):
    inherited = sorted(os.sched_getaffinity(0)) if inherited is None else inherited
    selected = inherited[16 + 8 * slot:24 + 8 * slot]
    v2.require(0 <= slot < 12 and len(selected) == 8, 'Invalid coordinated eight-CPU worker slot')
    for task in Path('/proc/self/task').glob('*'):
        try:
            os.sched_setaffinity(int(task.name), selected)
        except ProcessLookupError:
            pass
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 10:
        os.nice(10 - current_nice)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    return selected


def inspect(args):
    v2.require(sys.platform == 'linux' and v2.torch.__version__ == '2.11.0+cu128', 'Inspect in existing isolated Linux cu128 environment')
    bind_cpu(0)
    records, bindings = read_inputs(args)
    repo = args.repo.resolve()
    pins = source_pins(repo, records, args)
    source = full_hashes(pins)
    artifacts = {}
    for r in records:
        paths = mapping_paths(r, bindings)
        artifacts[r['id']] = full_hashes({paths[k]: r[k]['sha256'] for k in KINDS})
    after = full_hashes(pins)
    v2.require(source == after, 'Shared source/data changed during inspection')
    for values in artifacts.values():
        check_source_tokens(values)
    v2.require(not args.output.exists(), 'Preserve existing inspection receipts')
    v2.save(args.output, {'scope': SCOPE, 'status': 'ALL900_INPUT_IDENTITIES_INSPECTED_NO_INFERENCE',
                          'source_before': source, 'source_after': after, 'artifacts': artifacts, 'models': 900,
                          'storage_map_sha256': args.storage_map_sha256, 'restore_acceptance_sha256': args.restore_acceptance_sha256,
                          'all900_native_valid_replayed': False, 'test_labels_accessed': False, 'final_protocol_frozen': False})
    print(json.dumps({'status': 'ALL900_INPUT_IDENTITIES_INSPECTED_NO_INFERENCE', 'sources': len(source), 'artifact_files': 2700, 'receipt_sha256': v2.digest(args.output)}))


def worker(args):
    records, bindings = read_inputs(args)
    record = next(r for r in records if r['id'] == args.id)
    batch = v2.read(args.batch_receipt)
    v2.require(v2.digest(args.batch_receipt) == args.batch_receipt_sha256 and batch['v3_source_sha256'] == v2.digest(__file__), 'Worker batch/source identity changed')
    v2.require(record['id'] in batch['selected_ids'] and batch['scope'] == SCOPE, 'Worker model outside declared valid batch')
    bind_cpu(args.slot, batch['inherited_allowed_cpus'])
    paths = mapping_paths(record, bindings)
    pins = {paths[k]: record[k]['sha256'] for k in KINDS}
    before = full_hashes(pins)
    output = args.output
    proof = {'id': record['id'], 'scope': SCOPE, 'status': 'FAILED', 'artifact_before': before,
             'historical_output': record['original_remote_output'], 'runtime_read_output': str(paths['result'].parent),
             'storage_map_sha256': args.storage_map_sha256, 'batch_receipt_sha256': args.batch_receipt_sha256,
             'slot': args.slot, 'allowed_cpus': batch['inherited_allowed_cpus'][16 + 8 * args.slot:24 + 8 * args.slot],
             'original_job_bytes_modified': False, 'v2_source_sha256': V2_SHA, 'v3_source_sha256': v2.digest(__file__)}
    try:
        check_source_tokens(batch['source_before'])
        repo = args.repo.resolve()
        evaluator = v2.load('v3_sealed_evaluator', BASE / 'inputs/evaluator.py')
        core = v2.load('v3_frozen_core', repo / 'scripts/reproduce_paper_tables.py')
        original = v2.load('v3_original_checked', repo / 'scripts/run_revision_ablation.py')
        cnn = v2.load('v3_original_cnn', repo / 'src/celeba_data.py')
        ids, y, s, metadata = v2.metadata(repo)
        _, inference = mapped_functions(record, paths, original)
        def timeout(_signum, _frame):
            raise TimeoutError('Bounded valid worker wall-time limit exceeded; no retry')
        signal.signal(signal.SIGALRM, timeout)
        signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(InterruptedError('Batch stopped after another failed worker')))
        r = inference(core, original, cnn, evaluator, record, repo, ids, y, s, output, args.max_wall_seconds)
        check_source_tokens(batch['source_before'])
        proof.update(status='NATIVE_VALID_REPLAY_PASS', receipt_sha256=v2.digest(output / 'receipt.json'),
                     metadata_read_receipt=metadata, native_max_abs_difference=r['native_comparison']['max_abs_difference'])
    except BaseException as exc:
        signal.alarm(0)
        proof.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        try:
            proof['artifact_after'] = full_hashes(pins)
            v2.require(before == proof['artifact_after'], 'Mapped artifact changed during inference')
            proof['artifacts_unchanged'] = True
        except Exception as exc:
            proof.update(status='FAILED', artifacts_unchanged=False, identity_error=str(exc))
        v2.save(output.parent / (record['id'] + '.worker.json'), proof)
    return 0 if proof['status'] == 'NATIVE_VALID_REPLAY_PASS' else 1


def common_args(args):
    values = []
    for key in ('inventory', 'storage_map', 'storage_map_sha256', 'restore_receipt', 'restore_receipt_sha256', 'restore_acceptance', 'restore_acceptance_sha256', 'repo'):
        values += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    return values


def run(args):
    v2.require(sys.platform == 'linux' and v2.torch.__version__ == '2.11.0+cu128' and not sys.flags.optimize, 'Use the unchanged Linux isolated cu128 environment without -O')
    v2.require(1 <= args.workers <= 12 and 1 <= args.max_wall_seconds <= 3600, 'Bounded1..12 workers and <=3600s/model required')
    records, _ = read_inputs(args)
    selected = [r['id'] for r in records] if args.all else args.ids
    v2.require(selected and len(selected) == len(set(selected)) and set(selected) <= {r['id'] for r in records}, 'Unknown/duplicate/empty selected model cohort')
    v2.require(not args.output.exists(), 'Preserve every prior batch and failure; use a new output directory')
    args.output.mkdir(parents=True)
    import fcntl
    lock = (HERE / 'batch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inherited = sorted(os.sched_getaffinity(0))
    v2.require(len(inherited) >= 16 + 8 * args.workers, 'Missing coordinated CPU worker slots')
    # Coordinator shares a worker CPU and performs no image inference.
    for task in Path('/proc/self/task').glob('*'):
        try:
            os.sched_setaffinity(int(task.name), [inherited[16]])
        except ProcessLookupError:
            pass
    v2.torch.set_num_threads(1)
    os.nice(10)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    pins = source_pins(args.repo.resolve(), records, args)
    before = full_hashes(pins)
    batch = {'scope': SCOPE, 'status': 'VALID_INPUT_BATCH_BOUND_NOT_FINAL_FREEZE', 'selected_ids': selected,
             'source_before': before, 'v3_source_sha256': v2.digest(__file__), 'v2_source_sha256': V2_SHA,
             'workers': args.workers, 'inherited_allowed_cpus': inherited, 'storage_map_sha256': args.storage_map_sha256,
             'restore_acceptance_sha256': args.restore_acceptance_sha256, 'final_protocol_frozen': False, 'test_labels_accessed': False}
    batch_path = args.output / 'batch_inputs.json'
    v2.save(batch_path, batch)
    batch_sha = v2.digest(batch_path)
    active, complete, failures = {}, [], []
    started = time.monotonic()
    pending = iter(selected)
    stopped, exhausted = False, False
    try:
        while active or not exhausted:
            for slot in range(args.workers):
                if stopped or exhausted or slot in active:
                    continue
                try:
                    model_id = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                output = args.output / 'runs' / model_id
                output.parent.mkdir(exist_ok=True)
                log = (output.parent / (model_id + '.log')).open('xb')
                command = [sys.executable, str(Path(__file__).resolve()), 'worker', *common_args(args), '--id', model_id,
                           '--slot', str(slot), '--batch-receipt', str(batch_path), '--batch-receipt-sha256', batch_sha,
                           '--output', str(output), '--max-wall-seconds', str(args.max_wall_seconds)]
                # Worker restores its declared slot using the bound original CPU list;
                # no Python preexec function is run after a multithreaded Torch fork.
                p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                active[slot] = (p, model_id, log)
            for slot, (p, model_id, log) in list(active.items()):
                code = p.poll()
                if code is None:
                    continue
                log.close()
                del active[slot]
                if code == 0:
                    complete.append(model_id)
                else:
                    failures.append({'id': model_id, 'returncode': code})
                    stopped, exhausted = True, True
                    for own_p, _, _ in active.values():
                        own_p.terminate()
            if active:
                time.sleep(.25)
    except BaseException:
        stopped = True
        for p, _, _ in active.values():
            p.terminate()
        raise
    finally:
        for p, _, log in active.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            log.close()
        source_after, identity_error = None, None
        try:
            source_after = full_hashes(pins)
            unchanged = before == source_after
        except Exception as exc:
            unchanged, identity_error = False, str(exc)
        v2.save(args.output / 'batch_execution.json', {'scope': SCOPE, 'status': 'EXECUTION_ENDED_REQUIRES_STRICT_ACCEPTANCE',
                    'requested_ids': selected, 'finished_zero_exit_ids': complete, 'failures': failures,
                    'stopped_after_failure': stopped, 'source_after': source_after, 'source_unchanged': unchanged,
                    'identity_error': identity_error,
                    'wall_seconds': time.monotonic() - started, 'workers': args.workers,
                    'all900_native_valid_replayed': False, 'test_labels_accessed': False, 'final_protocol_frozen': False})
    return 1 if failures or stopped or not unchanged else 0


def accept(args):
    v2.require(sys.platform == 'linux', 'Strict valid acceptance uses the original Linux input paths')
    bind_cpu(0)
    records, bindings = read_inputs(args)
    indexed = {r['id']: r for r in records}
    batch = v2.read(args.batch / 'batch_inputs.json')
    execution = v2.read(args.batch / 'batch_execution.json')
    v2.require(batch['scope'] == SCOPE and batch['v3_source_sha256'] == v2.digest(__file__) and execution['source_unchanged'], 'Batch source identity failed')
    v2.require(batch['storage_map_sha256'] == args.storage_map_sha256 and batch['restore_acceptance_sha256'] == args.restore_acceptance_sha256, 'Batch restore binding differs')
    selected = batch['selected_ids']
    v2.require(len(selected) == len(set(selected)) and set(selected) <= set(indexed), 'Duplicate or invalid accepted cohort')
    v2.require(execution['requested_ids'] == selected and len(execution['finished_zero_exit_ids']) == len(set(execution['finished_zero_exit_ids'])), 'Execution model identity drift')
    check_source_tokens(batch['source_before'])
    v2.require(full_hashes(source_pins(args.repo.resolve(), records, args)) == batch['source_before'] == execution['source_after'], 'Batch shared source bytes changed')
    core = v2.load('v3_accept_core', args.repo / 'scripts/reproduce_paper_tables.py')
    original = v2.load('v3_accept_original', args.repo / 'scripts/run_revision_ablation.py')
    evaluator = v2.load('v3_accept_evaluator', BASE / 'inputs/evaluator.py')
    ids, y, s, _ = v2.metadata(args.repo)
    accepted, invalid = [], []
    for model_id in selected:
        try:
            record, path = indexed[model_id], args.batch / 'runs' / model_id
            proof = v2.read(path.parent / (model_id + '.worker.json'))
            r = v2.read(path / 'receipt.json')
            v2.require(proof['status'] == r['status'] == 'NATIVE_VALID_REPLAY_PASS' and proof['artifacts_unchanged'], 'Worker receipt failed or incomplete')
            v2.require(proof['id'] == r['id'] == model_id and r['runtime']['device'] == 'cpu' and r['runtime']['cuda_device_count'] == 0, 'Worker/model/device identity changed')
            v2.require(not r['optimizer_created'] and not r['gradients_created'] and not r['test_labels_accessed'], 'Inference-only label/optimizer contract failed')
            for resources in (r['before_resources'], r['after_resources']):
                v2.require(all(cpus == proof['allowed_cpus'] for cpus in resources['thread_cpu_affinities'].values()), 'Worker escaped its eight-CPU slot')
            v2.require(proof['batch_receipt_sha256'] == v2.digest(args.batch / 'batch_inputs.json') and proof['storage_map_sha256'] == args.storage_map_sha256, 'Worker belongs to another batch/map')
            v2.require(proof['receipt_sha256'] == v2.digest(path / 'receipt.json') and r['model_inventory_record_sha256'] == v2.canonical(record), 'Worker receipt or original record identity changed')
            paths = mapping_paths(record, bindings)
            v2.require(full_hashes({paths[k]: record[k]['sha256'] for k in KINDS}) == proof['artifact_before'] == proof['artifact_after'], 'Accepted mapped artifact changed')
            validate, _ = mapped_functions(record, paths, original)
            original_result = validate(original, record, args.repo)
            cfg = core.ExperimentConfig(**record['config'])
            root_ids, root_y, root_s, root_receipt = v2.rebuild_root(core, cfg, record, ids, y, s)
            v2.require(root_receipt == r['root_reconstruction'] and r['weights_before'] == r['weights_after'], 'Root identity or model weights drift')
            with v2.np.load(path / 'validation_predictions.npz', allow_pickle=False) as z:
                v2.require(v2.digest(path / 'validation_predictions.npz') == r['prediction_arrays_sha256'], 'Prediction array SHA changed')
                v2.require(v2.np.array_equal(z['root_image_ids'], root_ids) and v2.np.array_equal(z['valid_image_ids'], ids[162770:182637]), 'Root/valid sample order drift')
                fits = evaluator.fit_views(core, record['method'], z['root_margins'], root_y, root_s, cfg, v2.VIEWS, evaluator.SHARED_CALIBRATION)
                v2.require(v2.canonical(fits) == v2.canonical(r['fits']), 'Root-only threshold fit changed')
                predictions = evaluator.predict_views(z['valid_margins'], s[162770:], fits)
                v2.require(all(v2.np.array_equal(predictions[v], z['prediction_' + v]) for v in v2.VIEWS), 'Saved predictions contradict frozen margin/tie rules')
                scored = evaluator.evaluate_frozen_predictions(predictions, y[162770:], s[162770:])
                comparison = v2.check_native(scored['native'], original_result['metrics'])
                v2.require(scored == r['views'] and comparison == r['native_comparison'] and comparison['accepted'], 'Recomputed common-checkpoint metrics failed')
            accepted.append(model_id)
        except Exception as exc:
            invalid.append({'id': model_id, 'error_type': type(exc).__name__, 'error': str(exc)})
    complete = len(accepted) == len(selected) and not invalid and not execution['failures']
    report = {'scope': SCOPE, 'status': 'SELECTED_VALID_REPLAY_ACCEPTED' if complete else 'PARTIAL_OR_INVALID_VALID_REPLAY',
              'requested_n': len(selected), 'accepted_n': len(accepted), 'accepted_ids': accepted, 'invalid': invalid,
              'all900_native_valid_replayed': complete and len(accepted) == 900, 'max_abs_native_metric_difference': None if not accepted else max(v2.read(args.batch / 'runs' / i / 'receipt.json')['native_comparison']['max_abs_difference'] for i in accepted),
              'wall_seconds': execution['wall_seconds'], 'models_per_second': len(accepted) / execution['wall_seconds'],
              'workers': execution['workers'], 'test_labels_accessed': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN',
              'inventory_sha256': v2.INVENTORY_SHA, 'valid_image_ids_sha256': v2.VALID_IDS_SHA,
              'calibration_core_sha256': v2.CORE_SHA, 'v2_source_sha256': V2_SHA,
              'storage_map_sha256': args.storage_map_sha256, 'batch_path': str(args.batch.resolve()),
              'batch_inputs_sha256': v2.digest(args.batch / 'batch_inputs.json')}
    v2.require(not args.output.exists(), 'Preserve existing acceptance, including invalid results')
    v2.save(args.output, report)
    print(json.dumps({k: report[k] for k in ('status', 'accepted_n', 'requested_n', 'all900_native_valid_replayed')}))
    return 0 if complete else 1


def collect(args):
    """Join externally reviewed acceptances; useful1/2/4 batches are not replayed again."""
    v2.require(v2.digest(args.inventory) == v2.INVENTORY_SHA, 'Original inventory drift')
    records = v2.validate_inventory(v2.read(args.inventory))
    expected = {r['id'] for r in records}
    accepted, provenances, differences = [], [], []
    for filename, sha in args.acceptance or []:
        path = Path(filename)
        v2.require(v2.digest(path) == sha, 'Acceptance differs from externally reviewed SHA')
        a = v2.read(path)
        v2.require(a['scope'] == SCOPE and a['inventory_sha256'] == v2.INVENTORY_SHA and a['valid_image_ids_sha256'] == v2.VALID_IDS_SHA, 'Acceptance belongs to another cohort/split')
        v2.require(a['calibration_core_sha256'] == v2.CORE_SHA and a['v2_source_sha256'] == V2_SHA and not a['test_labels_accessed'], 'Acceptance scientific source/scope drift')
        v2.require(a['accepted_n'] == len(a['accepted_ids']), 'Accepted sample count differs')
        if a['accepted_ids']:
            v2.require(0 <= a['max_abs_native_metric_difference'] <= v2.TOLERANCE, 'Accepted native mismatch exceeds fixed tolerance')
            differences.append(a['max_abs_native_metric_difference'])
        accepted += a['accepted_ids']
        provenances.append({'path': str(path), 'sha256': sha, 'accepted_n': a['accepted_n']})
    if args.include_sealed_v2:
        # This exact offserver proof was independently reviewed by the parent.
        path = BASE / 'offserver_verification.json'
        sha = '6f0687e67c6a10029e863ff84008cd46090db5f45cad9135630a74df78bb00b1'
        v2.require(v2.digest(path) == sha and v2.digest(BASE / 'FILES_SHA256') == '39fa64bf9762d6975511d154d51cbaf3e7c292010f266b68dd8c98e96f35f402', 'Sealed two-canary proof drift')
        for line in (BASE / 'FILES_SHA256').read_text(encoding='utf-8').splitlines():
            expected_sha, relative = line.split('  ', 1)
            v2.require(v2.digest(BASE / relative) == expected_sha, 'Sealed v2 file changed: ' + relative)
        a = v2.read(path)
        v2.require(a['status'] == 'PASS' and {r['id'] for r in a['canaries']} == set(v2.CANARY_IDS), 'Sealed v2 actual canaries differ')
        accepted += [r['id'] for r in a['canaries']]
        differences += [r['native_max_abs_difference'] for r in a['canaries']]
        provenances.append({'path': str(path), 'sha256': sha, 'accepted_n': 2, 'version': 'sealed v2 actual CPU canaries'})
    v2.require(len(accepted) == len(set(accepted)) and set(accepted) <= expected, 'Duplicate or foreign model IDs across accepted batches')
    v2.require(not args.output.exists(), 'Preserve previous cumulative acceptances')
    report = {'scope': SCOPE, 'status': 'ALL900_NATIVE_VALID_REPLAY_ACCEPTED' if set(accepted) == expected else 'PARTIAL_VALID_REPLAY_ACCEPTED',
              'accepted_n': len(accepted), 'expected_n': 900, 'accepted_ids': sorted(accepted), 'missing_ids': sorted(expected - set(accepted)),
              'all900_native_valid_replayed': set(accepted) == expected, 'inventory_sha256': v2.INVENTORY_SHA,
              'max_abs_native_metric_difference': max(differences) if differences else None, 'accepted_provenance': provenances,
              'test_labels_accessed': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN', 'final_dispatch_created': False}
    v2.save(args.output, report)
    print(json.dumps({k: report[k] for k in ('status', 'accepted_n', 'expected_n', 'all900_native_valid_replayed')}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'run', 'worker', 'accept'):
        p = sub.add_parser(command)
        for name in ('inventory', 'storage-map', 'restore-receipt', 'restore-acceptance', 'repo'):
            p.add_argument('--' + name, type=Path, required=True)
        for name in ('storage-map-sha256', 'restore-receipt-sha256', 'restore-acceptance-sha256'):
            p.add_argument('--' + name, required=True)
        p.add_argument('--output', type=Path, required=True)
        if command == 'run':
            choice = p.add_mutually_exclusive_group(required=True)
            choice.add_argument('--ids', nargs='+')
            choice.add_argument('--all', action='store_true', help='Separate explicit authorization required; never the default')
            p.add_argument('--workers', type=int, default=1)
            p.add_argument('--max-wall-seconds', type=int, default=1800)
        elif command == 'worker':
            p.add_argument('--id', required=True)
            p.add_argument('--slot', type=int, required=True)
            p.add_argument('--batch-receipt', type=Path, required=True)
            p.add_argument('--batch-receipt-sha256', required=True)
            p.add_argument('--max-wall-seconds', type=int, required=True)
        elif command == 'accept':
            p.add_argument('--batch', type=Path, required=True)
    p = sub.add_parser('collect', help='Join externally SHA-reviewed strict acceptances without rerunning their models')
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--acceptance', action='append', nargs=2, metavar=('PATH', 'REVIEWED_SHA256'))
    p.add_argument('--include-sealed-v2', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            inspect(args)
            return 0
        return {'run': run, 'worker': worker, 'accept': accept, 'collect': collect}[args.command](args)
    except BaseException as exc:
        # Preserve pre-inference identity/source errors too; never silently retry.
        failure = args.output / 'launcher_failure.json' if args.output.is_dir() else args.output.with_suffix(args.output.suffix + '.failure.json')
        if not failure.exists():
            v2.save(failure, {'scope': SCOPE, 'status': 'FAILED', 'command': args.command,
                             'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                             'v3_source_sha256': v2.digest(__file__), 'new_image_inference_claimed': False,
                             'test_labels_accessed': False, 'final_protocol_frozen': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
