"""Exact-three valid interface candidate. No deployment or implicit authorization."""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import types

HERE = Path(__file__).resolve().parent
CPUS = list(range(120, 128))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package_check(expected):
    seal = HERE / 'FILES_SHA256.json'
    require(sha(seal) == expected, 'Candidate seal mismatch')
    for rel, pin in read(seal)['files'].items():
        p = (HERE / rel).resolve()
        require(p.is_relative_to(HERE), 'Unsafe package member')
        require(sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Candidate member changed: ' + rel)


def validate_manifest(m):
    expected = [
        ('FLGMM', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage'),
        ('CosineFairnessHybrid', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage'),
        ('Fed-NGA-gradient', 'FedNGA_eta0.01_non-IID_Benign_seed91001_screen'),
    ]
    require([(r['method'], r['id']) for r in m['records']] == expected, 'Only frozen exact three, in order')
    require(m['scope'] == 'EXACT3_VALID_IMAGE_INTERFACE_GATE_CANDIDATE', 'Wrong scope')
    require(m['views'] == ['native', 'raw', 'shared_calibration'] and m['native_tolerance'] == 1e-12, 'Scientific rule drift')
    require(m['cpu_affinity'] == CPUS and m['threads'] == 8, 'Resource policy drift')
    require(m['new_scientific_acceptances'] == 0 and m['dispatch_authorized'] is False, 'Prepared is not accepted')
    for r in m['records']:
        require(r['terminal_round'] == 70 and r['split'] == 'valid' and r['n_eval'] == 19867, 'Nonterminal/subset/test refused')
        require(r['actual_alpha'] == {'IID': 5000.0, 'non-IID': 5.0}[r['distribution']], 'Actual alpha drift')
        require(r['original_training_device'].startswith('cuda'), 'Never replace original CUDA provenance with CPU')
        for pin in r['runtime_artifacts'].values():
            p = PurePosixPath(pin['server_path'])
            require(p.is_absolute() and p.is_relative_to('/workspace') and '..' not in p.parts and ':' not in str(p), 'Windows or unsafe server path')
        require(r['runtime_artifacts']['model']['sha256'] == r['identity']['checkpoint']['sha256'], 'Checkpoint identity drift')
        c = r['identity']['config']
        require(c['rounds'] == 70 and c['seed'] == r['seed'] and c['client_alpha'] == r['actual_alpha'], 'Configuration drift')
        require(c['celeba_evaluation_split'] == 'valid' and c['celeba_train_limit'] == c['celeba_eval_limit'] == 0, 'Subset/test config refused')


def resolve_origin(origin, m, runtime):
    origin = str(origin).replace('\\', '/')
    pin = m['path_map'].get(origin)
    require(pin is not None, 'Unregistered origin path: ' + origin)
    if pin['kind'] == 'package':
        p = (HERE / pin['relative']).resolve()
        require(p.is_relative_to(HERE), 'Package mapping escapes')
        return p
    require(runtime, 'Server input may not be opened during source-only checks')
    return Path(pin['server_path'])


def bound_bridge(m, *, runtime, torch_module=None, pandas_module=None):
    """Private path registration only; original bridge functions stay byte-exact."""
    bridge = load('_exact3_original_bridge', HERE / 'originals/bridge.py')
    original_reuse = read(HERE / 'originals/SOURCE_REUSE.json')
    virtual = {str(bridge.HERE / 'SOURCE_REUSE.json'): HERE / 'originals/SOURCE_REUSE.json',
               str(bridge.HERE / 'PROOF_PINS.json'): HERE / 'originals/PROOF_PINS.json'}

    def path_for(path):
        return virtual.get(str(path)) or resolve_origin(path, m, runtime)

    def digest(path):
        return sha(path_for(path))

    def read_pin(pin):
        path = path_for(pin['path'])
        require(sha(path) == pin['sha256'], 'Pinned original bytes changed: ' + str(path))
        require('bytes' not in pin or path.stat().st_size == pin['bytes'], 'Pinned original size changed')
        data = read(path)
        if data == original_reuse:
            # Only SOURCE_REUSE file locations change in the private decoded metadata.
            for entry in data['function_sources']:
                entry['file']['path'] = str(resolve_origin(entry['file']['path'], m, runtime))
            data['original_core']['path'] = str(resolve_origin(data['original_core']['path'], m, runtime))
        return data

    bridge.digest, bridge.read_pin = digest, read_pin
    # science_bindings now uses translated private SOURCE_REUSE paths directly.
    bridge_digest = bridge.digest
    translated = {str(resolve_origin(e['file']['path'], m, runtime)) for e in original_reuse['function_sources']}
    translated.add(str(resolve_origin(original_reuse['original_core']['path'], m, runtime)))
    bridge.digest = lambda path: sha(path) if str(path) in translated else bridge_digest(path)
    evaluator = bridge.science_bindings(torch_module=torch_module, pandas_module=pandas_module)
    return bridge, evaluator


def runtime_nodes():
    """Compile-only inspectable binding; no scientific imports or calls."""
    text = (HERE / 'originals/replay.py').read_text(encoding='utf-8')
    tree = ast.parse(text)
    wanted = {'read', 'save', 'canonical', 'read_prefix', 'metadata', 'live_snapshot', 'resource_gate', 'replay_one'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]

    class Bind(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'validate_original':
                return ast.copy_location(ast.Call(func=ast.Name(id='validate_external', ctx=ast.Load()), args=[ast.Name(id='record', ctx=ast.Load())], keywords=[]), node)
            if isinstance(node.func, ast.Name) and node.func.id == 'inside':
                require(ast.unparse(node) == "inside(repo, record['checkpoint']['member'])", 'Unexpected original path call')
                return ast.copy_location(ast.Call(func=ast.Name(id='runtime_checkpoint', ctx=ast.Load()), args=[ast.Name(id='record', ctx=ast.Load())], keywords=[]), node)
            return node

        def visit_Constant(self, node):
            if node.value == 'model_inventory_record_sha256':
                node.value = 'external_identity_record_sha256'
            if isinstance(node.value, str) and node.value.startswith('Two bounded valid-only canaries'):
                node.value = 'Exactly three added-CNN valid interface gates; not full100, method ranking, final primary endpoint, test, or CUDA equivalence'
            return node

    nodes = Bind().visit(ast.Module(body=nodes, type_ignores=[]))
    require(len(nodes.body) == len(wanted), 'Missing original runtime function')
    return ast.fix_missing_locations(nodes)


def runtime_originals(evaluator):
    """Original replay definitions; two identity/path calls and report text rebind."""
    import datetime, platform, resource, signal, subprocess, time, zipfile
    ns = evaluator.rebuild_root.__globals__
    ns.update(dict(datetime=datetime, platform=platform, resource=resource, signal=signal,
                   subprocess=subprocess, time=time, zipfile=zipfile, os=os, sys=sys))
    exec(compile(runtime_nodes(), '<original-replay-private-binding>', 'exec'), ns)
    ns.update(TRAIN_IDS_SHA='46d42484d5b5f53af8747fcf44ee11a0b051af27f10383d32254faee0311bc99',
              VALID_IDS_SHA='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf', VIEWS=evaluator.VIEWS)
    return types.SimpleNamespace(**ns)


def authorize(a, m):
    require(sha(a.authorization) == a.authorization_sha256, 'Authorization SHA mismatch')
    auth = read(a.authorization)
    require(auth['status'] == 'ROOT_AUTHORIZED_EXACT3_VALID_IMAGE_INTERFACE_GATE', 'No actual root authorization')
    require(auth['package_sha256'] == a.package_sha256 and auth['manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Unbound authorization')
    require(auth['exact_ids'] == [r['id'] for r in m['records']] and auth['cpu_affinity'] == CPUS, 'Authorization scope drift')
    require(auth['device'] == 'cpu' and auth['max_processes'] == 1 and auth['test'] is False, 'Unauthorized runtime')
    require(auth['source_review_sha256'] == a.source_review_sha256 and auth['linux_preflight_sha256'] == a.preflight_sha256, 'Missing source/Linux root review')
    require(sha(a.source_review) == a.source_review_sha256, 'Actual source review SHA changed')
    review = read(a.source_review)
    require(review['source_adoptable'] is True and review['package_sha256'] == a.package_sha256, 'Candidate source not root adopted')
    require(sha(a.preflight) == a.preflight_sha256, 'Linux preflight SHA changed')
    p = read(a.preflight)
    require(p['status'] == 'ROOT_LINUX_EXACT3_PREFLIGHT_PASS' and p['cpu_affinity'] == CPUS, 'Linux preflight refused')
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(p['utc'])).total_seconds()
    require(0 <= age <= 300 and p['no_duplicate_gate'] is True and p['all_thread_cpus_free'] is True, 'Stale/occupied preflight')
    require(set(CPUS) <= set(p['eligible_cpus']), 'Requested CPUs outside actual eligible mask')
    require(p['nominal_reserved_cores_including_gate'] <= p['actual_quota_cores'], 'Nominal reservations exceed measured quota')
    for key in ('source_model_data_hashes_verified', 'selected_producers_quiescent', 'services_healthy',
                'gpu_health_verified', 'cgroup_and_memory_headroom_verified', 'storage_headroom_verified'):
        require(p[key] is True, 'Missing actual Linux preflight fact: ' + key)


def exclusive_cpus():
    require(sorted(os.sched_getaffinity(0)) == CPUS and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'CPU/nice drift')
    conflicts = []
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        for task in (proc / 'task').glob('*'):
            try:
                affinity = set(os.sched_getaffinity(int(task.name)))
                if len(affinity) <= 8 and affinity.intersection(CPUS):
                    conflicts.append([int(proc.name), int(task.name), sorted(affinity)])
            except ProcessLookupError:
                pass
    require(not conflicts, 'Restricted-thread CPU overlap: ' + str(conflicts))
    io = __import__('subprocess').run(['ionice', '-p', str(os.getpid())], capture_output=True, text=True, check=True)
    require('idle' in io.stdout, 'Idle I/O priority required')


def run(a):
    require(sys.platform == 'linux' and __debug__, 'Linux without -O required')
    package_check(a.package_sha256)
    m = read(HERE / 'MANIFEST.json')
    validate_manifest(m)
    authorize(a, m)
    require(not a.output.exists() and a.output.is_absolute() and a.output.is_relative_to('/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/outputs'), 'Fresh fixed server output required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before importing Torch')
    for k, v in {'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}.items():
        require(os.environ.get(k) == v, 'Thread environment drift: ' + k)
    import fcntl, signal, traceback
    exclusive_cpus()
    lock = open('/tmp/guardfed_added_cnn_exact3_valid_gate.lock', 'a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np, pandas as pd, torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    require(torch.__version__ == '2.11.0+cu128', 'Actual replay Torch changed')
    bridge, ev = bound_bridge(m, runtime=True, torch_module=torch, pandas_module=pd)
    rt = runtime_originals(ev)
    repo = Path(m['server_repo'])
    sys.path.insert(0, str(repo))
    pins = {repo / rel: h for rel, h in m['runtime_repo_hashes'].items()}
    def check_inputs():
        for p, h in pins.items():
            require(sha(p) == h, 'Runtime source/data SHA changed: ' + str(p))
        for r in m['records']:
            for pin in r['runtime_artifacts'].values():
                p = Path(pin['server_path'])
                require(sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Accepted artifact changed: ' + str(p))
            out = Path(r['runtime_output'])
            require(not list(out.glob('failure*.json')) and not (out / 'FAILED.json').exists(), 'Preserved original failure')
            for proc in Path('/proc').glob('[0-9]*'):
                try:
                    cmd = (proc / 'cmdline').read_bytes().decode(errors='replace')
                    require(r['id'] not in cmd, 'Selected producer still live')
                except (FileNotFoundError, ProcessLookupError):
                    pass
    check_inputs()
    core = load('_exact3_core', repo / 'scripts/reproduce_paper_tables.py')
    cnn = load('_exact3_cnn', repo / 'src/celeba_data.py')
    rt.resource_gate(rt.live_snapshot(repo))
    ids, y, s, data_receipt = rt.metadata(repo)
    a.output.mkdir(parents=True, exist_ok=False)
    rt.save(a.output / 'metadata_receipt.json', data_receipt)
    def timed_out(signum, frame):
        raise TimeoutError('Fixed wall limit; preserve output, no retry')
    signal.signal(signal.SIGALRM, timed_out)
    receipts = []
    try:
        for row in m['records']:
            exclusive_cpus()
            check_inputs()
            actual = bridge.identity_record(row['method'], row['id'])
            require(actual == row['identity'], 'Actual original bridge identity changed')
            record = copy.deepcopy(actual)
            record.update(distribution=row['distribution'], attack=row['attack'], seed=row['seed'],
                          result=actual['original_artifact_pins']['result'], raw_job=actual['original_artifact_pins']['job'],
                          config_canonical_sha256=rt.canonical(actual['config']), training_torch=row['original_training_torch'])
            rt.replay_one.__globals__['validate_external'] = lambda r: {'metrics': r['prior_validation_metrics']}
            rt.replay_one.__globals__['runtime_checkpoint'] = lambda r: Path(row['runtime_artifacts']['model']['server_path'])
            receipts.append(rt.replay_one(core, None, cnn, ev, record, repo, ids, y, s, a.output / row['id'], 1800))
        check_inputs()
        package_check(a.package_sha256)
        rt.save(a.output / 'GATE_RESULT.json', {'status': 'EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED', 'receipts': receipts,
                  'root_adopted_new': 0, 'full100_complete': False, 'test': False, 'package_sha256': a.package_sha256})
    except BaseException:
        rt.save(a.output / 'FAILURE.json', {'status': 'FAILED_PRESERVED_NO_RETRY', 'traceback': traceback.format_exc(), 'completed': len(receipts)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('package-sha256', 'authorization-sha256', 'preflight-sha256', 'source-review-sha256'):
        p.add_argument('--' + key, required=True)
    for key in ('authorization', 'preflight', 'source-review', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    run(p.parse_args())
