"""Read-only terminal-model RGB64 validation replay. Never dispatches final evaluation."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import platform
if platform.system() == 'Linux':
    import resource
import signal
import subprocess
import sys
import time
import traceback
import zipfile

# Set before scientific imports. One process; no loader workers or CUDA context.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
for key in ('OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

HERE = Path(__file__).resolve().parent
INVENTORY_SHA = '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd'
CORE_SHA = 'cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
RUNNER_SHA = '4ae9f1e7f53e5fdbe1fe00e16fa5d1852ecf55c9644cf4462f1fe6b73df480a1'
EVALUATOR_SHA = '805eedf1fb08137cd86a543a80c83b9527e5c8937be02f7d2dca83a33b86e04c'
VALID_IDS_SHA = '64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf'
TRAIN_IDS_SHA = '46d42484d5b5f53af8747fcf44ee11a0b051af27f10383d32254faee0311bc99'
METHODS = ['FedAvg', 'Median', 'FLTrust', 'FairFed', 'FairGuard', 'FLTrust+FairGuard', 'GuardFed-AD2+', 'FedAA', 'LASA']
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
VIEWS = ['native', 'raw', 'shared_calibration']
TOLERANCE = 1e-12
CANARY_IDS = ['GuardFed-AD2+_IID_Benign_seed91001', 'GuardFed-AD2+_lr0.0005_drop0.005_Benign_seed91002']
SCIENCE_FILES = {
    'scripts/reproduce_paper_tables.py', 'src/data_loader.py', 'src/celeba_data.py', 'scripts/build_celeba_cache.py',
    'data/celeba/derived/rgb64_v1/manifest.json', 'data/celeba/derived/rgb64_v1/metadata.npz',
    'data/celeba/derived/rgb64_v1/images.npy', 'data/celeba/derived/rgb64_v1/available.npy',
    'data/celeba/list_attr_celeba.txt', 'data/celeba/list_eval_partition.txt',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values).tobytes(order='C')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inside(repo, relative):
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Input path escapes original repository: ' + str(relative))
    # The accepted CelebA path is an existing symlink to the original cache.
    # Its resolved destination is measured with its bytes before and after replay.
    return repo / relative


def validate_inventory(inventory):
    records = inventory['records']
    require(len(records) == 900 and len({r['id'] for r in records}) == 900, '900 unique accepted inventory records required')
    expected = set(itertools.product(METHODS, ('IID', 'non-IID'), ATTACKS, range(91001, 91011)))
    cells = {(r['method'], r['distribution'], r['attack'], r['seed']) for r in records}
    require(cells == expected, 'Inventory has duplicate cells or missing methods/scenes/seeds')
    for r in records:
        c, contract = r['config'], r['data_contract']
        require((r['terminal_round'], r['original_split'], r['original_n_eval']) == (70, 'valid', 19867), 'Non-terminal/non-valid inventory record')
        require(canonical(c) == r['config_canonical_sha256'], 'Inventory configuration identity drift')
        require(c['rounds'] == 70 and c['celeba_evaluation_split'] == 'valid' and not c['celeba_train_limit'] and not c['celeba_eval_limit'], 'Subset or test configuration refused')
        require(c['seed'] == r['seed'] and c['client_alpha'] == r['actual_alpha'] == {'IID': 5000.0, 'non-IID': 5.0}[r['distribution']], 'Seed/alpha mismatch')
        require(not c['synthetic_ratio'] and not c['root_label_noise'] and not c['root_sensitive_noise'] and c['ablation_component'] == 'none', 'Only the original clean root and accepted method are supported')
        require(not c['include_sensitive_feature'] and c['celeba_cache_dir'] == '', 'Unexpected image/data path configuration')
        require(SCIENCE_FILES <= set(r['source_hashes']) and r['source_hashes']['scripts/reproduce_paper_tables.py'] == CORE_SHA, 'Incomplete scientific input lineage')
        require(contract['train_image_ids_sha256'] == TRAIN_IDS_SHA and contract['evaluation_image_ids_sha256'] == VALID_IDS_SHA, 'Official train/valid ID drift')
        require((contract['actual_train_rows'], contract['actual_evaluation_rows'], contract['evaluation_split']) == (162770, 19867, 'valid'), 'Unexpected split support')
        require(contract['train_eval_disjoint'] and contract['root_client_disjoint'], 'Invalid partition disjointness')
    return records


def prepare(args):
    require(digest(args.inventory) == INVENTORY_SHA, 'Only the independently accepted current900 inventory is supported')
    inventory, protocol = read(args.inventory), read(args.protocol)
    validate_inventory(inventory)
    require(protocol['calibration_core_sha256'] == CORE_SHA and protocol['original_split'] == 'valid', 'Prepared calibration source/split drift')
    require(protocol['status'] == 'PREPARED_NOT_FROZEN' and all(x is None for x in protocol['decisions'].values()), 'This preflight does not replace a final frozen protocol')
    require(digest(args.evaluator) == EVALUATOR_SHA, 'Sealed evaluator source drift')
    require(not args.output.exists(), 'Preserve existing input plans; choose an unused output path')
    payload = {
        'schema': 'guardfed_valid_replay_input_plan_v1', 'scope': 'VALID_ONLY_IMPLEMENTATION_PREFLIGHT',
        'final_protocol_status': protocol['status'], 'final_dispatch_created': False, 'test_labels_accessed': False,
        'target_split': 'valid', 'target_n': 19867, 'target_image_ids_sha256': VALID_IDS_SHA,
        'target_order_sha256': VALID_IDS_SHA, 'root': 'original_clean_train_root_by_original_config_and_ID_SHA',
        'model_count': 900, 'inventory_file': args.inventory.name, 'inventory_sha256': digest(args.inventory),
        'prepared_protocol_file': args.protocol.name, 'prepared_protocol_sha256': digest(args.protocol),
        'evaluator_file': args.evaluator.name, 'evaluator_sha256': EVALUATOR_SHA,
        'replay_source_sha256': digest(__file__), 'core_sha256': CORE_SHA, 'checked_result_source_sha256': RUNNER_SHA,
        'native_tolerance': TOLERANCE, 'diagnostic_views': VIEWS,
        'runtime': {'device': 'cpu', 'torch': '2.11.0+cu128', 'max_processes': 1, 'max_allowed_cpus': 8, 'loader_workers': 0,
                    'cpu_affinity': 'sorted inherited allowed CPUs [16:24], as coordinated with other checks'},
        'canary_ids': CANARY_IDS,
        'limits': ['Not a final freeze or dispatch receipt', 'All900 replay remains pending until every record passes',
                   'Original CUDA checkpoint inference is being checked on CPU; no runtime equivalence assumption',
                   'Native metric mismatch stops the sequential run and preserves outputs; no tolerance relaxation or retry'],
    }
    save(args.output, payload)
    print(json.dumps({'plan': str(args.output), 'sha256': digest(args.output), 'scope': payload['scope']}))


def read_prefix(npz, name, total, count):
    """Request only the NPY scalar prefix ending at official valid; never materialize test labels."""
    with zipfile.ZipFile(npz) as archive:
        with archive.open(name + '.npy') as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError('Unsupported NPY metadata version')
            require(shape == (total,) and not fortran and not dtype.hasobject, 'Non-vector or unsafe label metadata')
            require(count < total and count > 0, 'Prefix must exclude test entries')
            payload = member.read(count * dtype.itemsize)
            require(len(payload) == count * dtype.itemsize, 'Truncated train/valid metadata')
            a = np.frombuffer(payload, dtype=dtype).copy()
    require(np.isin(a, (0, 1)).all(), 'Nonbinary train/valid labels or sensitive attribute')
    return a, {'member': name + '.npy', 'shape': list(shape), 'dtype': dtype.str,
               'requested_entries': count, 'requested_payload_bytes': len(payload), 'last_included_position': count - 1,
               'test_entries_requested_or_materialized': 0, 'train_valid_prefix_sha256': array_sha(a)}


def metadata(repo):
    cache = repo / 'data/celeba/derived/rgb64_v1'
    manifest = read(cache / 'manifest.json')
    require(manifest['complete'], 'Incomplete RGB64 cache')
    with np.load(cache / 'metadata.npz', allow_pickle=False) as z:
        ids, split = z['image_id'], z['split']  # explicitly no Smiling/Male here
    require(np.array_equal(ids, np.arange(1, 202600)) and ids.dtype == np.dtype('int64'), 'Unexpected cache ID ordering/type')
    require(np.array_equal(np.flatnonzero(split == 0), np.arange(162770)), 'Train must be the exact official prefix')
    require(np.array_equal(np.flatnonzero(split == 1), np.arange(162770, 182637)), 'Valid must follow train before test')
    require(np.array_equal(np.flatnonzero(split == 2), np.arange(182637, 202599)), 'Unexpected non-label partition structure')
    y, y_receipt = read_prefix(cache / 'metadata.npz', 'Smiling', len(ids), 182637)
    s, s_receipt = read_prefix(cache / 'metadata.npz', 'Male', len(ids), 182637)
    require(array_sha(ids[:162770]) == TRAIN_IDS_SHA and array_sha(ids[162770:182637]) == VALID_IDS_SHA, 'Official sample ID/order SHA mismatch')
    available = np.load(cache / 'available.npy', mmap_mode='r', allow_pickle=False)
    require(available.shape == (202599,) and available.all(), 'Missing RGB images')
    receipt = {'strategy': 'NPY headers then exactly train+valid scalar prefix; no full label-array load',
               'prefix_reads': [y_receipt, s_receipt], 'train_n': 162770, 'valid_n': 19867,
               'test_labels_accessed': False, 'full_loader_executed': False,
               'valid_support': support(s[162770:], y[162770:]),
               'official_metadata': manifest['official_metadata']}
    require(receipt['valid_support'] == {'0|0': 5252, '0|1': 6157, '1|0': 5013, '1|1': 3445}, 'Validation sensitive/label support differs')
    return ids, y, s, receipt


def support(s, y):
    return {f'{g}|{l}': int(((s == g) & (y == l)).sum()) for g in (0, 1) for l in (0, 1)}


def rebuild_root(core, cfg, record, ids, y, s):
    train_df = pd.DataFrame({'image_id': ids[:162770], 'Smiling': y[:162770], 'Male': s[:162770]})
    root, sampling = core.sample_server_dataframe(train_df, 'Smiling', 'Male', cfg)
    clients_df = train_df.drop(root.index).reset_index(drop=True)
    # Existing metadata-only partition path, no client images or local model training.
    clients = core.create_client_data_dict(clients_df, ['image_id'], 'Smiling', 'Male', cfg.num_clients,
                                           record['actual_alpha'], torch.device('cpu'), cfg.seed)
    root, noise = core.apply_root_noise(root.reset_index(drop=True), 'Smiling', 'Male', cfg)
    root_ids = root['image_id'].to_numpy(dtype=np.int64)
    client_ids = np.concatenate([c['X'][:, 0].numpy().astype(np.int64) for c in clients.values()])
    counts = [len(c['y']) for c in clients.values()]
    require(array_sha(root_ids) == record['data_contract']['root_image_ids_sha256'], 'Original root image ID/order SHA mismatch')
    require(counts == record['data_contract']['client_sample_counts'], 'Original metadata client partition counts differ')
    require(len(root_ids) + sum(counts) == 162770 and not np.intersect1d(root_ids, client_ids).size, 'Root/client support or overlap mismatch')
    require(np.array_equal(np.sort(np.concatenate([root_ids, client_ids])), ids[:162770]), 'Root/client union differs from original train IDs')
    require(not np.intersect1d(root_ids, ids[162770:182637]).size, 'Root overlaps validation')
    ry, rs = root['Smiling'].to_numpy(dtype=np.int64), root['Male'].to_numpy(dtype=np.int64)
    return root_ids, ry, rs, {'root_n': len(root_ids), 'root_image_ids_sha256': array_sha(root_ids),
                             'root_support': support(rs, ry), 'client_sample_counts': counts,
                             'train_eval_disjoint': True, 'root_client_disjoint': True,
                             'server_sampling_audit': sampling, 'root_noise_audit': noise}


def adapter_paths(repo, record):
    """Find pinned, original adapter bytes under the existing bounded adapter tree; never execute them."""
    expected = record['adapter_source_hashes']
    if not expected:
        return {}
    root = repo / 'deployment/baseline_adapters_20260928'
    require(root.is_dir(), 'Original adapter sources must be restored before replay')
    names = {'screen_runner': 'run_fedaa_screen.py', 'round_adapter': 'fedaa_round_adapter.py',
             'official_wrapper': 'fedaa_official_adapter.py', 'official_ddpg': 'DDPG.py'}
    found = {}
    for key, sha in expected.items():
        candidates = sorted(root.rglob(names.get(key, key)))
        matching = [p.resolve() for p in candidates if p.is_file() and digest(p) == sha]
        require(matching, 'Missing original pinned adapter bytes: ' + key)
        found[key] = matching[0]
    return found


def input_paths(repo, records, plan_path, plan):
    pinned = {Path(__file__).resolve(): plan['replay_source_sha256'], plan_path: digest(plan_path)}
    for key, sha_key in [('inventory_file', 'inventory_sha256'), ('prepared_protocol_file', 'prepared_protocol_sha256'), ('evaluator_file', 'evaluator_sha256')]:
        path = (plan_path.parent / plan[key]).resolve()
        require(path.is_relative_to(plan_path.parent), 'Plan input path escapes input directory')
        pinned[path] = plan[sha_key]
    pinned[inside(repo, 'scripts/run_revision_ablation.py')] = RUNNER_SHA
    for record in records:
        for rel, sha in record['source_hashes'].items():
            path = inside(repo, rel)
            require(path not in pinned or pinned[path] == sha, 'Conflicting original source identities')
            pinned[path] = sha
        for _, path in adapter_paths(repo, record).items():
            pinned[path] = digest(path)
        for key in ('checkpoint', 'result', 'raw_job'):
            pinned[inside(repo, record[key]['member'])] = record[key]['sha256']
        output = Path(record['original_remote_output']).resolve()
        require(output.is_relative_to(repo) and inside(repo, record['result']['member']).parent == output, 'Original output path drift')
        require(not list(output.glob('failure*.json')), 'Preserved original failure evidence requires review')
    return pinned


def hash_pins(pinned):
    observed = {}
    for path, sha in pinned.items():
        actual = digest(path)
        observed[str(path)] = {'sha256': actual, 'resolved_path': str(path.resolve()), 'bytes': path.stat().st_size}
        require(actual == sha, 'Input SHA mismatch: ' + str(path))
    return observed


def validate_original(original, record, repo):
    job = read(inside(repo, record['raw_job']['member']))
    require(job['config'] == record['config'] and job['source_hashes'] == record['source_hashes'], 'Original job/config/source differs from inventory')
    require((job['id'], job['output'], job['distribution'], job['attack']) ==
            (PurePosixPath(record['raw_job']['member']).stem, record['original_remote_output'], record['distribution'], record['attack']), 'Original job identity drift')
    require(job.get('adapter_hashes', job.get('adapter_source_hashes', {})) == record['adapter_source_hashes'], 'Original adapter job identity drift')
    result = original.checked_result(job)
    require(result is not None, 'Original checked_result did not accept the model')
    require(result['dataset'] == 'celeba' and result['method'] == record['source_method'], 'Original dataset/method mismatch')
    require((result['seed'], result['distribution'], result['attack'], result['alpha'], result['rounds']) ==
            (record['seed'], record['distribution'], record['attack'], record['actual_alpha'], 70), 'Original seed/scenario/horizon mismatch')
    require(result['data_contract']['image_data_contract'] == record['data_contract'], 'Original image contract differs from inventory')
    require(result['metrics'] == result['trajectory_metrics'][-1]['metrics'] == record['prior_validation_metrics'], 'Original native metrics or common checkpoint drift')
    require([r['round'] for r in result['round_summaries']] == list(range(1, 71)), 'Incomplete original terminal diagnostics')
    return result


def check_native(actual, expected):
    differences = {}
    for key in ('accuracy', 'aeod', 'aspd'):
        require(type(actual[key]) in (float, int) and np.isfinite(actual[key]), 'Undefined/nonfinite native metric')
        differences[key] = actual[key] - expected[key]
    return {'tolerance': TOLERANCE, 'expected': expected, 'observed': {k: actual[k] for k in differences},
            'differences': differences, 'max_abs_difference': max(abs(x) for x in differences.values()),
            'accepted': all(abs(x) <= TOLERANCE for x in differences.values())}


def live_snapshot(repo):
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'process_pid': os.getpid(),
              'os_threads': len(list(Path('/proc/self/task').glob('*'))), 'process_status': {},
              'cgroup': {}, 'load_average': list(os.getloadavg())}
    result['thread_cpu_affinities'] = {}
    for task in Path('/proc/self/task').glob('*'):
        try:
            result['thread_cpu_affinities'][task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            pass
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.split(':')[0] in ('VmRSS', 'VmHWM', 'Threads', 'Cpus_allowed_list'):
            key, value = line.split(':', 1)
            result['process_status'][key] = value.strip()
    for name in ('cpu.max', 'cpu.stat', 'memory.current', 'memory.max', 'memory.events'):
        p = Path('/sys/fs/cgroup') / name
        result['cgroup'][name] = p.read_text().strip() if p.exists() else None
    for name, command in [('service', ['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal']),
                          ('gpu', ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu', '--format=csv,noheader'])]:
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
            result[name] = {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
        except Exception as exc:
            result[name] = {'error': str(exc)}
    queue = repo / 'results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json'
    if queue.exists():
        q = read(queue)
        result['training_queue_snapshot'] = q
    return result


def resource_gate(snapshot):
    require(torch.get_num_threads() == 8 and torch.get_num_interop_threads() == 1, 'Compute thread setting drift')
    require('RUNNING' in snapshot['service'].get('stdout', ''), 'Formal training health not observable; do not begin inference')
    cg = snapshot['cgroup']
    quota, period = cg['cpu.max'].split()
    cores = os.cpu_count() if quota == 'max' else int(quota) / int(period)
    require(cores >= 32, 'Insufficient CPU quota headroom for eight low-priority threads')
    require(cg['memory.max'] != 'max' and int(cg['memory.max']) - int(cg['memory.current']) >= 8 * 1024**3, 'Insufficient memory headroom')
    require(torch.cuda.device_count() == 0, 'CUDA must be hidden from the validation process')
    require(len(os.sched_getaffinity(0)) == 8, 'Process must be bound to its eight coordinated CPUs')
    allowed = set(os.sched_getaffinity(0))
    require(all(set(cpus) <= allowed for cpus in snapshot['thread_cpu_affinities'].values()), 'An auxiliary thread escaped the coordinated CPU set')


def replay_one(core, original, cnn, evaluator, record, repo, ids, y, s, output, wall_limit):
    started, usage_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    signal.alarm(wall_limit)
    result = validate_original(original, record, repo)
    cfg = core.ExperimentConfig(**record['config'])  # original config bytes remain unchanged, including device string
    core.set_seed(cfg.seed, deterministic_image=True)
    root_ids, root_y, root_sensitive, root_receipt = rebuild_root(core, cfg, record, ids, y, s)
    model = cnn.CelebACNN(seed=cfg.seed).cpu()
    state = torch.load(inside(repo, record['checkpoint']['member']), map_location='cpu', weights_only=True)
    require(state and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in state.values()), 'Invalid/nonfinite checkpoint tensors')
    model.load_state_dict(state, strict=True)
    model.eval()
    weights_before = evaluator.weights_identity(model)
    image_map = np.load(repo / 'data/celeba/derived/rgb64_v1/images.npy', mmap_mode='r', allow_pickle=False)
    require(image_map.shape == (202599, 3, 64, 64) and image_map.dtype == np.uint8, 'RGB64 image cache shape/type drift')
    root_X = torch.from_numpy(np.array(image_map[root_ids - 1], copy=True))
    valid_X = torch.from_numpy(np.array(image_map[162770:182637], copy=True))
    before = live_snapshot(repo)
    resource_gate(before)
    root_margins, valid_margins, fits, predictions = evaluator.extract_and_predict(
        core, model, {'server_X': root_X, 'server_y': torch.from_numpy(root_y), 'server_sensitive': root_sensitive},
        valid_X, s[162770:], record['method'], cfg, VIEWS, evaluator.SHARED_CALIBRATION)
    # Target labels first enter the scoring interface after fits and predictions are fixed.
    metrics = evaluator.evaluate_frozen_predictions(predictions, y[162770:], s[162770:])
    comparison = check_native(metrics['native'], result['metrics'])
    require(all(p.grad is None for p in model.parameters()), 'Inference created parameter gradients')
    weights_after = evaluator.weights_identity(model)
    require(weights_after == weights_before, 'Inference changed model tensor bytes')
    after = live_snapshot(repo)
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / 'validation_predictions.npz', root_image_ids=root_ids,
                        valid_image_ids=ids[162770:182637], root_margins=root_margins, valid_margins=valid_margins,
                        **{'prediction_' + k: v for k, v in predictions.items()})
    receipt = {
        'scope': 'VALID_ONLY_IMPLEMENTATION_PREFLIGHT', 'status': 'NATIVE_VALID_REPLAY_PASS' if comparison['accepted'] else 'NATIVE_VALID_REPLAY_MISMATCH',
        'id': record['id'], 'method': record['method'], 'distribution': record['distribution'], 'attack': record['attack'], 'seed': record['seed'],
        'model_inventory_record_sha256': canonical(record), 'checkpoint_sha256': record['checkpoint']['sha256'],
        'original_result_sha256': record['result']['sha256'], 'original_job_sha256': record['raw_job']['sha256'],
        'config_canonical_sha256': record['config_canonical_sha256'], 'original_training_torch': record['training_torch'],
        'runtime': {'python': sys.executable, 'torch': torch.__version__, 'device': 'cpu', 'original_config_device': cfg.device,
                    'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(), 'loader_workers': 0,
                    'nice': os.getpriority(os.PRIO_PROCESS, 0), 'cuda_device_count': torch.cuda.device_count(), 'cpu': platform.processor()},
        'root_reconstruction': root_receipt, 'valid_n': len(valid_X), 'valid_image_ids_sha256': array_sha(ids[162770:182637]),
        'fits': fits, 'views': metrics, 'native_comparison': comparison,
        'prediction_arrays_sha256': digest(output / 'validation_predictions.npz'),
        'weights_before': weights_before, 'weights_after': weights_after, 'optimizer_created': False, 'gradients_created': False,
        'zero_margin_count': int(np.sum(valid_margins == 0)),
        'elapsed_seconds': time.monotonic() - started,
        'cpu_user_seconds': usage_end.ru_utime - usage_start.ru_utime, 'cpu_system_seconds': usage_end.ru_stime - usage_start.ru_stime,
        'peak_rss_kib': usage_end.ru_maxrss, 'before_resources': before, 'after_resources': after,
        'test_labels_accessed': False, 'test_inference_performed': False, 'final_dispatch_created': False,
        'claim_limit': 'Two bounded valid-only canaries are implementation evidence, not all900 replay or a new final performance table',
    }
    receipt['effective_cpu_cores'] = (receipt['cpu_user_seconds'] + receipt['cpu_system_seconds']) / receipt['elapsed_seconds']
    save(output / 'receipt.json', receipt)
    signal.alarm(0)
    resource_gate(after)
    require(comparison['accepted'], 'Native metrics exceed fixed tolerance; preserve evidence and stop without retry')
    return receipt


def run(args):
    require(platform.system() == 'Linux' and not sys.flags.optimize, 'Run with assertions enabled in the existing Linux cu128 environment')
    require(not args.output.exists(), 'Preserve existing receipts/failures; output must be a new directory')
    args.output.mkdir(parents=True)
    # A single advisory lock in this owned directory prevents overlapping CLI invocations.
    import fcntl
    lock = (HERE / 'valid-replay.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another validation replay process holds the single-process lock')
    plan_path = args.plan.resolve()
    plan = read(plan_path)
    before = None
    pinned = None
    accepted = []
    receipt = {'scope': 'VALID_ONLY_IMPLEMENTATION_PREFLIGHT', 'status': 'FAILED', 'accepted_ids': accepted,
               'all900_native_valid_replayed': False, 'final_protocol_frozen': False, 'test_labels_accessed': False}
    try:
        require(plan['scope'] == 'VALID_ONLY_IMPLEMENTATION_PREFLIGHT' and plan['target_split'] == 'valid' and plan['target_n'] == 19867
                and plan['target_image_ids_sha256'] == plan['target_order_sha256'] == VALID_IDS_SHA, 'Not a valid-only preflight input plan')
        require(plan['native_tolerance'] == TOLERANCE and plan['replay_source_sha256'] == digest(__file__), 'Tolerance/replay source plan drift')
        require(plan['inventory_sha256'] == INVENTORY_SHA and plan['evaluator_sha256'] == EVALUATOR_SHA, 'Sealed input identity drift')
        require(torch.__version__ == '2.11.0+cu128', 'Use the existing isolated cu128 Python; do not install dependencies')
        inventory = read(plan_path.parent / plan['inventory_file'])
        all_records = validate_inventory(inventory)
        selected = [r['id'] for r in all_records] if args.all else args.ids
        require(selected and len(set(selected)) == len(selected), 'Declare unique IDs or explicitly request --all')
        indexed = {r['id']: r for r in all_records}
        require(set(selected) <= set(indexed), 'Unknown accepted inventory model ID')
        records = [indexed[i] for i in selected]
        receipt['selected_ids'] = selected
        receipt['selected_count'] = len(selected)
        if selected == CANARY_IDS:
            require(all(r['training_torch'] == '2.11.0+cu128' and r['method'] == 'GuardFed-AD2+' and r['attack'] == 'Benign' for r in records), 'Canaries must be original cu128 Full Benign')
        repo = args.repo.resolve()
        require(not args.output.resolve().is_relative_to(repo), 'Replay artifacts must stay outside the original training repository')
        inherited = sorted(os.sched_getaffinity(0))
        require(len(inherited) >= 24, 'At least24 inherited allowed CPUs are required for the coordinated affinity slice')
        # Linux affinity is per thread: bind every existing auxiliary thread too.
        # Future threads inherit their already-bound creator's affinity.
        for task in Path('/proc/self/task').glob('*'):
            try:
                os.sched_setaffinity(int(task.name), inherited[16:24])
            except ProcessLookupError:
                pass
        receipt['coordinated_cpu_affinity'] = {'inherited_allowed': inherited, 'selected': inherited[16:24]}
        os.nice(10)
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True, text=True)
        pinned = input_paths(repo, records, plan_path, plan)
        before = hash_pins(pinned)
        save(args.output / 'input_sha256_before.json', before)
        save(args.output / 'selected_inventory_records.json', records)
        evaluator = load('sealed_valid_evaluator', plan_path.parent / plan['evaluator_file'])
        core = load('frozen_valid_core', repo / 'scripts/reproduce_paper_tables.py')
        original = load('original_valid_checked_result', repo / 'scripts/run_revision_ablation.py')
        cnn = load('original_celeba_cnn', repo / 'src/celeba_data.py')
        initial = live_snapshot(repo)
        resource_gate(initial)
        ids, y, s, metadata_receipt = metadata(repo)
        save(args.output / 'metadata_read_receipt.json', metadata_receipt)
        save(args.output / 'resource_before.json', initial)
        def timeout(_signum, _frame):
            raise TimeoutError('Bounded per-model CPU canary wall-time limit exceeded; no automatic retry')
        signal.signal(signal.SIGALRM, timeout)
        for record in records:
            receipt['current_model_id'] = record['id']
            print(json.dumps({'event': 'start', 'id': record['id'], 'device': 'cpu', 'threads': 8}), flush=True)
            r = replay_one(core, original, cnn, evaluator, record, repo, ids, y, s, args.output / record['id'], args.max_wall_seconds)
            accepted.append(record['id'])
            print(json.dumps({'event': 'accepted', 'id': record['id'], 'seconds': r['elapsed_seconds'],
                              'native_max_abs_difference': r['native_comparison']['max_abs_difference']}), flush=True)
        receipt.update(status='SELECTED_NATIVE_VALID_REPLAY_PASS', selected_ids=selected,
                       selected_count=len(selected), accepted_count=len(accepted),
                       all900_native_valid_replayed=(len(accepted) == 900), final_protocol_status='PREPARED_NOT_FROZEN',
                       resource_after=live_snapshot(repo))
    except Exception as exc:
        signal.alarm(0)
        receipt.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        save(args.output / 'failure.json', receipt)
        print(json.dumps({'event': 'stopped', 'error': str(exc), 'accepted_ids': accepted}), flush=True)
    finally:
        if before is not None:
            try:
                after = hash_pins(pinned)
                save(args.output / 'input_sha256_after.json', after)
                require(before == after, 'Source/model/data changed during validation replay')
                receipt['all_input_sha_unchanged'] = True
                receipt['input_count'] = len(before)
            except Exception as exc:
                receipt.update(status='FAILED', all_input_sha_unchanged=False, identity_error=str(exc))
                save(args.output / 'identity_failure.json', receipt)
        save(args.output / 'run_receipt.json', receipt)
    return 0 if receipt['status'] == 'SELECTED_NATIVE_VALID_REPLAY_PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Bind all900 valid replay inputs without final freeze or dispatch')
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--evaluator', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run', help='Sequential CPU valid-only replay; stops on the first failed model')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    chosen = p.add_mutually_exclusive_group(required=True)
    chosen.add_argument('--ids', nargs='+')
    chosen.add_argument('--all', action='store_true', help='Requires separate authorization; not used for the two canaries')
    p.add_argument('--max-wall-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
        return 0
    require(1 <= args.max_wall_seconds <= 3600, 'Per-model wall limit must be bounded within one hour')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
