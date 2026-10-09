"""Resource classification fix only; reuse the sealed v1 execute function body."""
import argparse
import ast
import datetime
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import traceback
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREPARED = HERE.parent
V1 = PREPARED / 'execution_attachments'
V1_SEAL_SHA = '8415e43e0136fb8234349bf000247d9647ffd0ab78dce2607381d6af4502e1c3'
V1_SOURCE_SHA = '062d27d737cdd48607abeabc83b411cd1f6a8d6573c46663435ea69293cf6e2b'
spec = importlib.util.spec_from_file_location('sealed_mechanism_execute_v1', V1 / 'execute_one.py')
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
require, digest, read = v1.require, v1.digest, v1.read
require(digest(V1 / 'FILES_SHA256.json') == V1_SEAL_SHA and digest(V1 / 'execute_one.py') == V1_SOURCE_SHA, 'Sealed execution v1 changed')
REPO = PurePosixPath('/workspace/GuardFed-celeba-expanded')
CHECKS = PurePosixPath('/workspace/guardfed_checks')
HYBRID = CHECKS / 'celeba_hybrid_realimage_gate_20261009'
GRADIENT = CHECKS / 'celeba_gradient_realimage_gate_20261009'
FLCPU = CHECKS / 'celeba_flgmm_realimage_gate_20261009'
FLGPU = CHECKS / 'celeba_flgmm_gpu_gate_20261009'
REPLAY = CHECKS / 'celeba_final_valid_replay_20261009'
MECHANISM_REPLAY = CHECKS / 'celeba_mechanism_valid_replay_20261009'
READ_ONLY = {'inspect', 'summarize', 'verify', 'approve', 'collect', '--help', '--version', '--source-only'}


def source_checks(approval=None):
    v1.source_checks()  # verifies prepared13 and original execution7 unchanged
    seal = HERE / 'FILES_SHA256.json'
    if approval is not None:
        require(digest(seal) == approval['execution_seal_sha256'], 'Execution v2 seal changed')
    for relative, identity in read(seal)['members'].items():
        require(digest(HERE / relative) == identity['sha256'], 'Execution v2 source/input changed: ' + relative)


def absolute(script, cwd):
    path = PurePosixPath(script)
    return PurePosixPath(posixpath.normpath(str(path if path.is_absolute() else PurePosixPath(cwd) / path)))


def python_entry(argv):
    """Locate the main Python script/-c code, never grep arbitrary later strings."""
    if not argv or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', PurePosixPath(argv[0]).name):
        return None
    i = 1
    while i < len(argv):
        flag = argv[i]
        if flag == '-c':
            return ('code', argv[i + 1], argv[i + 2:]) if i + 1 < len(argv) else None
        if flag in ('-W', '-X'):
            i += 2; continue
        if flag in ('-m', '-'):
            return None
        if flag.startswith('-'):
            i += 1; continue
        return 'script', flag, argv[i + 1:]
    return None


def option(args, name):
    indexes = [i for i, item in enumerate(args) if item == name]
    if len(indexes) != 1 or indexes[0] + 1 == len(args):
        return None
    return args[indexes[0] + 1]


def gradient_code_runs(code, args):
    """Only trusted-cwd actual wrapper/main calls; strings/probes are excluded."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    aliases, imported, calls, action = {}, set(), [], args[0] if args else None
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name] = a.name
                imported.add(a.name)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                               and t.value.id == 'sys' and t.attr == 'argv' for t in n.targets):
            try:
                assigned = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                return False
            if not isinstance(assigned, list) or len(assigned) < 2 or not isinstance(assigned[1], str):
                return False
            action = assigned[1]
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            calls.append((n.func.value.id, n.func.attr))
    if 'shared_cache_wrapper' not in imported or action in READ_ONLY or (action is not None and action != 'run'):
        return False
    return any(name == 'main' and aliases.get(owner, owner) in {'gate', 'shared_cache_wrapper'} for owner, name in calls)


def classify(argv, cwd):
    entry = python_entry(argv)
    if entry is None:
        return None
    kind, value, args = entry
    if kind == 'code':
        if PurePosixPath(cwd) == GRADIENT and gradient_code_runs(value, args):
            return {'role': 'gradient_CPU_gate', 'compute_threads': 8, 'task_identity': str(GRADIENT)}
        return None
    script = absolute(value, cwd)
    if any(a in READ_ONLY for a in args):
        return None
    if script in {MECHANISM_REPLAY / 'execution_attachments/execute_one.py',
                  MECHANISM_REPLAY / 'execution_attachments_v2/execute_one_v2.py'} and args and args[0] == 'run':
        return {'role': 'mechanism_valid_worker', 'compute_threads': 8, 'task_identity': v1.MODEL_ID}
    if script == REPO / 'deployment/celeba_mechanism_20261009/worker.py' and option(args, '--repo') == str(REPO) and option(args, '--job'):
        return {'role': 'formal_GPU_worker', 'compute_threads': 1, 'task_identity': option(args, '--job')}
    if script.is_relative_to(REPLAY) and script.name in {'replay_v3.py', 'replay_v4.py'} and args and args[0] == 'worker' and option(args, '--id'):
        return {'role': 'baseline_valid_worker', 'compute_threads': 8, 'task_identity': option(args, '--id')}
    if script.parent in {HYBRID, FLCPU} and script.name in {'gate.py', 'canary.py'} and (not args or args[0] == 'run' or args[0].startswith('--')):
        return {'role': 'hybrid_CPU_gate' if script.parent == HYBRID else 'FLGMM_CPU_gate', 'compute_threads': 8,
                'task_identity': str(script.parent)}
    if script.parent == GRADIENT and script.name in {'gate.py', 'shared_cache_wrapper.py'} and args and args[0] == 'run':
        return {'role': 'gradient_CPU_gate', 'compute_threads': 8, 'task_identity': str(GRADIENT)}
    if script == FLGPU / 'gpu_worker.py' and option(args, '--repo') == str(REPO) and option(args, '--job') and option(args, '--out'):
        return {'role': 'FLGMM_GPU_canary', 'compute_threads': 1, 'task_identity': option(args, '--job')}
    return None


def reservations(rows):
    tracked, pids, tasks, occupied = [], set(), set(), set()
    for row in rows:
        if row.get('state') == 'Z':
            continue
        classification = classify(row['argv'], row['cwd'])
        if classification is None:
            continue
        pid = row['pid']
        require(pid not in pids, 'Duplicate process snapshot PID')
        task = classification['role'], classification['task_identity']
        require(task not in tasks, 'Duplicate GuardFed compute task: ' + str(task))
        cpus = set(row['cpus'])
        if classification['compute_threads'] == 8:
            require(len(cpus) == 8 and all(type(c) is int and c >= 0 for c in cpus), 'CPU compute worker is not on a complete8-CPU allocation')
            require(not cpus.intersection(v1.CPU_IDS), 'Existing CPU compute worker occupies112..119')
            require(not occupied.intersection(cpus), 'Existing CPU compute allocations overlap')
            occupied.update(cpus)
        pids.add(pid); tasks.add(task)
        tracked.append({**classification, 'pid': pid, 'cpus': sorted(cpus), 'argv': row['argv'], 'cwd': row['cwd']})
    require(sum(r['role'] == 'FLGMM_GPU_canary' for r in tracked) <= 2, 'More than two FLGMM GPU canary workers observed')
    return tracked


def proc_rows():
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [a.decode(errors='replace') for a in (proc / 'cmdline').read_bytes().split(b'\0') if a]
            # Do not persist unrelated processes, environment values or shell text.
            if python_entry(argv) is None:
                continue
            cwd = str((proc / 'cwd').resolve())
            if classify(argv, cwd) is None:
                continue
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            rows.append({'pid': int(proc.name), 'argv': argv, 'cwd': cwd, 'state': stat[0],
                         'cpus': sorted(os.sched_getaffinity(int(proc.name)))})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def resource_snapshot():
    tracked = reservations(proc_rows())
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    require(quota != 'max', 'Use the actual finite CPU quota')
    budget = int(quota) / int(period)
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tracked_compute': tracked,
            'nominal_compute_threads_including_this8': v1.check_budget(tracked, budget), 'actual_quota_cores': budget,
            'formal_service': subprocess.run(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'], capture_output=True, text=True).stdout.strip()}


# The parent already reviewed this execute body. Only its private globals change:
# v2 attachment directory/source seal and corrected resource classification.
namespace = dict(v1.execute.__globals__)
namespace.update(HERE=HERE, PREPARED=PREPARED, __file__=__file__, source_checks=source_checks, resource_snapshot=resource_snapshot)
execute = types.FunctionType(v1.execute.__code__, namespace, v1.execute.__name__, v1.execute.__defaults__, v1.execute.__closure__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'run'])
    parser.add_argument('--dispatch-receipt', type=Path)
    parser.add_argument('--dispatch-receipt-sha256')
    args = parser.parse_args()
    if args.action == 'inspect':
        source_checks()
        print(json.dumps({'status': 'PREPARED_NOT_APPROVED', 'id': v1.MODEL_ID, 'cpus': v1.CPU_IDS, 'inference_started': False}))
    else:
        try:
            require(args.dispatch_receipt is not None and args.dispatch_receipt_sha256 is not None, 'External receipt and SHA required')
            execute(args.dispatch_receipt, args.dispatch_receipt_sha256)
        except BaseException as exc:
            failure = HERE / 'execution_failure.json'
            if not failure.exists():
                with failure.open('x', encoding='utf-8') as stream:
                    json.dump({'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc(),
                               'no_automatic_retry': True, 'all_mechanism_replay_complete': False}, stream, indent=2)
            raise
