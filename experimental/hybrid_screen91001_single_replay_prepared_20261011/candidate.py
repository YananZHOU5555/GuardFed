"""Prepared single accepted Hybrid screen91001 replay. No new formal training."""
from __future__ import annotations
import argparse, ast, copy, datetime, hashlib, importlib.util, json, os, sys, types
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
BASE = '/workspace/guardfed_checks/celeba_hybrid_screen91001_single_20261011'
METHOD = 'CosineFairnessHybrid'
IDS = ['CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91001_screen']
CPUS = None  # Root binds an 8-CPU mask or a same-socket 32-CPU pool; computation stays 8 threads.


def parent_functions():
    assert hashlib.sha256((HERE / 'originals/fl_candidate.py').read_bytes()).hexdigest() == 'd8412c4aa782e767afd174d92b7568f0153afb5215f8f65fbff178d84960d1be'
    text = (HERE / 'originals/fl_candidate.py').read_text(encoding='utf-8')
    names = {'require', 'sha', 'read', 'load', 'package_check', 'runtime_nodes', 'runtime_originals', 'exclusive_cpus'}
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<unchanged original FL candidate helpers>', 'exec'), globals())


parent_functions()
original_runtime_nodes = runtime_nodes
original_exclusive_cpus = exclusive_cpus


def exclusive_cpus():
    original_exclusive_cpus()
    # The original <=8 scan remains; also reject another restricted pool overlapping a 32-CPU pool.
    if len(CPUS) == 32:
        for proc in Path('/proc').glob('[0-9]*'):
            if int(proc.name) == os.getpid():
                continue
            for task in (proc / 'task').glob('*'):
                try:
                    affinity = set(os.sched_getaffinity(int(task.name)))
                    require(not (len(affinity) <= 32 and affinity.intersection(CPUS + list(range(320, 352)))), 'Restricted CPU pool overlap')
                except ProcessLookupError:
                    pass


def runtime_nodes():
    # The original AST identity/path bindings remain; only a report scope string changes.
    nodes = original_runtime_nodes()
    for node in ast.walk(nodes):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('Exactly47 accepted FLGMM checkpoints;'):
            node.value = 'Exactly1 adopted Hybrid screen91001 checkpoint; selection history retained, not new formal training, full100, test, ranking, final-primary or CUDA equivalence'
    gate = next(n for n in nodes.body if n.name == 'resource_gate')
    rebound = 0
    for node in ast.walk(gate):
        if isinstance(node, ast.Compare) and ast.unparse(node) == 'len(os.sched_getaffinity(0)) == 8':
            node.comparators = [ast.parse('len(RESOURCE_CPUS)', mode='eval').body]
            rebound += 1
    require(rebound == 1, 'Only resource affinity cardinality changes; computation threads stay8')
    return ast.fix_missing_locations(nodes)


def validate_manifest(m):
    require(m['scope'] == 'HYBRID_ADOPTED_SCREEN91001_SINGLE_SOURCE_ONLY', 'Wrong scope')
    require(m['exact_ids'] == IDS == [r['id'] for r in m['records']], 'Exact1 screen identity required')
    require(m['canary_id'] == IDS[0] and m['previously_accepted_skip_ids'] == [] and m['phase'] == 'screen' and m['selection_seed'] is True, 'Canary/skip scope drift')
    require(m['native_root_pin']['sha256'] == '26bab727c431679b2913ac76cefe7dc28729c49d660bb3c8141489426bd28e6d', 'Original screen native root changed')
    require(m['threads'] == 8 and m['cpu_affinity'] is None and m['device'] == 'cpu', 'Resource binding is external, FP32 CPU only')
    require(m['views'] == ['native', 'raw', 'shared_calibration'] and m['native_tolerance'] == 1e-12, 'Scientific rules changed')
    require(m['new_three_view_accepted'] == 0 and m['dispatch_authorized'] is False, 'Source is not acceptance')
    for row in m['records']:
        r = row['identity']
        require((row['method'], row['distribution'], row['attack'], row['seed'], row['terminal_round'], row['n_eval']) ==
                (METHOD, 'IID', 'Benign', 91001, 70, 19867), 'Wrong accepted record')
        require(r['id'] == row['id'] and r['method'] == METHOD and r['actual_alpha'] == 5000.0, 'Wrong private identity')
        require(r['config']['rounds'] == 70 and r['config']['device'] == 'cuda' and r['config']['celeba_evaluation_split'] == 'valid', 'Original training provenance changed')
        require(r['config']['celeba_train_limit'] == r['config']['celeba_eval_limit'] == 0, 'Subset refused')
        require(row['runtime_artifacts']['model']['sha256'] == r['checkpoint']['sha256'], 'Wrong checkpoint')
        for pin in row['runtime_artifacts'].values():
            p = PurePosixPath(pin['server_path'])
            require(p.is_absolute() and p.is_relative_to('/workspace') and ':' not in str(p) and '..' not in p.parts, 'Non-Linux runtime path')


def resolve_origin(origin, m, runtime):
    key = str(origin).replace('\\', '/')
    pin = m['path_map'].get(key)
    require(pin is not None, 'Unregistered origin: ' + key)
    if pin['kind'] == 'package':
        p = (HERE / pin['relative']).resolve()
        require(p.is_relative_to(HERE), 'Unsafe source mapping')
        return p
    if not runtime:
        require(sys.platform == 'win32' and key.startswith('F:/YananResearchStorage/GuardFed/'), 'Local metadata only on actual F')
        return Path(key)
    return Path(pin['server_path'])


def bound_bridge(m, *, runtime, torch_module=None, pandas_module=None):
    """Rebind private file locations only; Hybrid identity and science bodies are unmodified."""
    private = load('_hybrid_screen91001_private', HERE / 'originals/hybrid_bridge.py')
    def read_pin(pin, *, decode=True):
        p = resolve_origin(pin['path'], m, runtime)
        require(p.suffix in {'.json', '.py'} and p.stat().st_size <= 2_000_000, 'Compact identity input only')
        raw = p.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == pin['sha256'] and len(raw) == pin['bytes'], 'Private identity pin changed: ' + str(p))
        return json.loads(raw) if decode else raw
    private.read_pin = read_pin
    private.inputs = lambda: read_pin(m['private_inputs_pin'])
    generic = load('_hybrid_screen91001_original_generic', HERE / 'originals/bridge.py')
    reuse = read(HERE / 'originals/SOURCE_REUSE.json')
    def generic_pin(pin):
        if str(pin['path']) == str(generic.HERE / 'SOURCE_REUSE.json'):
            p = HERE / 'originals/SOURCE_REUSE.json'
        else:
            p = resolve_origin(pin['path'], m, runtime)
        require(sha(p) == pin['sha256'], 'Original source pin changed')
        require('bytes' not in pin or p.stat().st_size == pin['bytes'], 'Original source size changed')
        value = read(p)
        if value == reuse:
            for entry in value['function_sources']:
                entry['file']['path'] = str(resolve_origin(entry['file']['path'], m, runtime))
            value['original_core']['path'] = str(resolve_origin(value['original_core']['path'], m, runtime))
        return value
    generic.read_pin = generic_pin
    generic.digest = sha
    private.original_bridge = lambda: generic
    evaluator = private.science_bindings(torch_module=torch_module, pandas_module=pandas_module)
    return private, evaluator


def runtime_record(row):
    record = copy.deepcopy(row['identity'])
    record.update(distribution=row['distribution'], attack=row['attack'], seed=row['seed'],
                  result=record['original_artifact_pins']['metadata']['result'],
                  raw_job=record['original_artifact_pins']['metadata']['job'],
                  training_torch=row['original_training_torch'])
    return record


def authorize(a, m):
    global CPUS
    CPUS = [int(v) for v in a.cpus.split(',')]
    require(CPUS == list(range(64, 96)), 'Fixed64..95 pool required; computation remains8 threads')
    reserved = set(range(11, 19)) | set(range(32, 64)) | set(range(102, 120))
    require(not set(CPUS).intersection(reserved), 'Reserved FL32..63/old11..18/102..119 refused')
    for name in ('authorization', 'preflight', 'source_review'):
        require(sha(getattr(a, name)) == getattr(a, name + '_sha256'), 'Actual external proof SHA changed')
    auth, p, review = read(a.authorization), read(a.preflight), read(a.source_review)
    require(auth['status'] == 'ROOT_AUTHORIZED_HYBRID_SCREEN91001_CANARY_FIRST_VALID', 'No root authorization')
    require(auth['package_sha256'] == a.package_sha256 and auth['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Unbound authorization')
    require(auth['exact_ids'] == IDS and auth['canary_id'] == IDS[0] and auth['cpu_affinity'] == CPUS, 'Authorization identity/resource drift')
    require(auth['device'] == 'cpu' and auth['max_processes'] == 1 and auth['test'] is False, 'Unauthorized runtime')
    require(auth['source_review_sha256'] == a.source_review_sha256 and auth['linux_preflight_sha256'] == a.preflight_sha256, 'Review/preflight binding missing')
    require(review['source_adoptable'] is True and review['package_sha256'] == a.package_sha256, 'Unaccepted source')
    require(p['status'] == 'ROOT_LINUX_HYBRID_SCREEN91001_PREFLIGHT_PASS' and p['cpu_affinity'] == CPUS, 'Wrong actual preflight')
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(p['utc'])).total_seconds()
    require(0 <= age <= 300 and p['no_duplicate_gate'] and p['resources_eligible'], 'Stale/occupied preflight')
    require(set(CPUS) <= set(p['eligible_cpus']) and p['nominal_reserved_cores_including_gate'] <= p['actual_quota_cores'], 'Ineligible/overcommitted CPU')
    require(p['same_socket_cpu_pool'] is True and p['pool_topology_verified'] is True and set(range(320, 352)) <= set(p['eligible_cpus']), 'Pool/SMT topology or eligibility changed')
    require(p['capacity_policy'] == 'sampled_32core_pool_atleast16_pairs_le20pct_cgroup_plus16_lt_quota_v3' and p['sample_monitored_cpu_count'] == 64 and p['quiet_pairs'] >= 16, 'Wrong sampled capacity policy')
    require(p['measured_cgroup_cores_with_gate_and_allowance'] < p['actual_quota_cores'] and p['FL_replay_service_exited'] is True and p['previous_Hybrid8_service_exited'] is True and p['previous_replay_workers_absent'] is True, 'Insufficient measured quota or unverified concurrent FL replay')
    for key in ('source_model_data_hashes_verified', 'selected_producers_quiescent', 'services_healthy', 'gpu_health_verified',
                'cgroup_and_memory_headroom_verified', 'storage_headroom_verified'):
        require(p[key] is True, 'Missing actual runtime fact: ' + key)


def check_inputs(m):
    repo = Path(m['server_repo'])
    for rel, h in m['runtime_repo_hashes'].items():
        require(sha(repo / rel) == h, 'Original source/data changed: ' + rel)
    for row in m['records']:
        for pin in row['runtime_artifacts'].values():
            p = Path(pin['server_path'])
            require(sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Accepted artifact changed: ' + str(p))
        out = Path(row['runtime_output'])
        require(not list(out.glob('failure*.json')) and not (out / 'FAILED.json').exists(), 'Preserved producer failure')
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                require(row['id'] not in (proc / 'cmdline').read_bytes().decode(errors='replace'), 'Selected producer still active')
            except (FileNotFoundError, ProcessLookupError):
                pass


def run(a):
    require(sys.platform == 'linux' and __debug__, 'Linux without -O required')
    package_check(a.package_sha256)
    m = read(HERE / 'MANIFEST.json'); validate_manifest(m); authorize(a, m)
    require(not a.output.exists() and a.output.is_absolute() and a.output.is_relative_to(BASE + '/outputs'), 'Fresh fixed output required; preserve any failure')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before Torch import')
    for k, v in {'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}.items():
        require(os.environ.get(k) == v, 'Thread environment drift')
    import fcntl, signal, traceback
    exclusive_cpus()
    lock = open('/tmp/guardfed_hybrid_screen91001_valid.lock', 'a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np, pandas as pd, torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    require(torch.__version__ == '2.11.0+cu128' and torch.cuda.device_count() == 0, 'Actual FP32 CPU replay runtime changed')
    bridge, ev = bound_bridge(m, runtime=True, torch_module=torch, pandas_module=pd)
    rt = runtime_originals(ev); rt.resource_gate.__globals__['RESOURCE_CPUS'] = CPUS
    repo = Path(m['server_repo']); sys.path.insert(0, str(repo))
    check_inputs(m)
    core = load('_hybrid_screen91001_core', repo / 'scripts/reproduce_paper_tables.py')
    cnn = load('_hybrid_screen91001_cnn', repo / 'src/celeba_data.py')
    rt.resource_gate(rt.live_snapshot(repo))
    ids, y, s, data_receipt = rt.metadata(repo)
    a.output.mkdir(parents=True, exist_ok=False); rt.save(a.output / 'metadata_receipt.json', data_receipt)
    def timed_out(signum, frame):
        raise TimeoutError('Fixed wall limit; preserve output, no retry')
    signal.signal(signal.SIGALRM, timed_out)
    receipts = []
    try:
        for row in m['records']:
            exclusive_cpus(); check_inputs(m)
            actual = bridge.identity_record(row['id'], checkpoint_sha256=row['identity']['checkpoint']['sha256'])
            require(actual == row['identity'], 'Actual Hybrid identity changed')
            record = runtime_record(row)
            rt.replay_one.__globals__['validate_external'] = lambda r: {'metrics': r['prior_validation_metrics']}
            rt.replay_one.__globals__['runtime_checkpoint'] = lambda r: Path(row['runtime_artifacts']['model']['server_path'])
            receipt = rt.replay_one(core, None, cnn, ev, record, repo, ids, y, s, a.output / row['id'], 1800)
            receipts.append(receipt)
            if row['id'] == IDS[0]:
                require(receipt['native_comparison']['accepted'] and receipt['native_comparison']['max_abs_difference'] <= 1e-12, 'Single native canary failed; stop without retry')
                rt.save(a.output / 'CANARY_PASS.json', {'id': IDS[0], 'checkpoint_sha256': receipt['checkpoint_sha256'], 'native_comparison': receipt['native_comparison'], 'root_adopted': False})
        check_inputs(m); package_check(a.package_sha256)
        rt.save(a.output / 'GATE_RESULT.json', {'status': 'HYBRID_SCREEN91001_THREE_VIEW_PASS_NOT_ROOT_ADOPTED', 'receipts': receipts,
                'package_sha256': a.package_sha256, 'manifest_sha256': sha(HERE / 'MANIFEST.json'), 'root_adopted_new': 0, 'full100_complete': False, 'test': False})
    except BaseException:
        rt.save(a.output / 'FAILURE.json', {'status': 'FAILED_PRESERVED_NO_RETRY', 'traceback': traceback.format_exc(), 'completed': len(receipts)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('package-sha256', 'authorization-sha256', 'preflight-sha256', 'source-review-sha256', 'cpus'):
        p.add_argument('--' + key, required=True)
    for key in ('authorization', 'preflight', 'source-review', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    run(p.parse_args())
