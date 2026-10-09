"""Prepared single-mechanism valid replay attachment; approval is external."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PREPARED = HERE.parent
PREPARED_SEAL = '4da76ee750afd7d8824cadc1db7cbc1466c6eeb599c287ca6aa14b17525be0d1'
BRIDGE_SHA = 'ae28a69acbb35413fbe000c8be2a52ff2ed1977100c6e62580543e1c84dc0044'
INVENTORY_SHA = 'be4d1d34b21443572a25e6a189710e8d75b108a8fcbdb58896a71f2ba1d88cc8'
MODEL_ID = 'minus_U_IID_Benign_seed91002'
CPU_IDS = list(range(112, 120))
GUIDE_SHA = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
REPO = Path('/workspace/GuardFed-celeba-expanded')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_budget(tracked, quota_cores):
    nominal = sum(r['compute_threads'] for r in tracked) + 8
    require(nominal <= quota_cores, 'Current nominal GuardFed CPU reservation plus8 exceeds quota; do not overlap phase5 until8 cores release')
    return nominal


def source_checks(approval=None):
    seal = PREPARED / 'FILES_SHA256.json'
    require(digest(seal) == PREPARED_SEAL, 'Prepared bridge seal changed')
    for relative, identity in read(seal)['members'].items():
        require(digest(PREPARED / relative) == identity['sha256'], 'Prepared source/input changed: ' + relative)
    execution_seal = HERE / 'FILES_SHA256.json'
    if approval is not None:
        require(digest(execution_seal) == approval['execution_seal_sha256'], 'Execution attachment seal changed')
    for relative, identity in read(execution_seal)['members'].items():
        require(digest(HERE / relative) == identity['sha256'], 'Execution attachment changed: ' + relative)


def resource_snapshot():
    """Measure known GuardFed compute reservations, not auxiliary thread count."""
    tracked = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            if not argv or 'python' not in Path(argv[0]).name:
                continue
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            command, cwd = ' '.join(argv), str((proc / 'cwd').resolve())
            threads, role = 0, None
            if '/deployment/celeba_mechanism_20261009/worker.py' in command:
                threads, role = 1, 'formal_GPU_worker'
            elif ('replay_v3.py' in command or 'replay_v4.py' in command) and 'worker' in argv:
                threads, role = 8, 'baseline_valid_worker'
            elif any(name in command or name in cwd for name in ('celeba_hybrid_realimage_gate_20261009', 'celeba_gradient_realimage_gate_20261009', 'celeba_flgmm_realimage_gate_20261009')) and ('gate.py' in command or 'shared_cache_wrapper.py' in command):
                threads, role = 8, 'other_CPU_gate'
            if role:
                cpus = sorted(os.sched_getaffinity(int(proc.name)))
                require(role == 'formal_GPU_worker' or not set(cpus).intersection(CPU_IDS), 'An existing CPU compute worker occupies112..119')
                tracked.append({'pid': int(proc.name), 'role': role, 'compute_threads': threads, 'cpus': cpus})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    require(quota != 'max', 'Use the actual finite CPU quota')
    budget = int(quota) / int(period)
    nominal = check_budget(tracked, budget)
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'tracked_compute': tracked,
            'nominal_compute_threads_including_this8': nominal, 'actual_quota_cores': budget,
            'formal_service': subprocess.run(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'], capture_output=True, text=True).stdout.strip()}


def execute(approval_path, approved_sha):
    require(sys.platform == 'linux' and not sys.flags.optimize, 'Linux without -O required')
    require(digest(approval_path) == approved_sha, 'External approval receipt SHA mismatch')
    approval = read(approval_path)
    source_checks(approval)
    require(approval['status'] == 'APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY' and approval['selected_ids'] == [MODEL_ID]
            and approval['allowed_cpus'] == CPU_IDS and approval['execution_source_sha256'] == digest(__file__), 'Only one explicitly approved seed91002/CPU112..119 task')
    require(digest('/etc/vast-agents-guide.md') == GUIDE_SHA, 'Server guide changed; read it before any execution')
    require(set(CPU_IDS) <= os.sched_getaffinity(0), 'Assigned CPU112..119 not available')
    output = Path(approval['output'])
    require(output == HERE / 'runs' / MODEL_ID and not output.exists(), 'Preserve all prior partial outputs')
    # No scientific imports or image access before the external gate/resource check.
    resources = resource_snapshot()
    require('RUNNING' in resources['formal_service'], 'Formal GPU health is not observable')
    os.sched_setaffinity(0, CPU_IDS)
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Use the prepared nice10/idleIO service wrapper')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    spec = importlib.util.spec_from_file_location('approved_mechanism_bridge', PREPARED / 'bridge.py')
    bridge = importlib.util.module_from_spec(spec); sys.modules[spec.name] = bridge; spec.loader.exec_module(bridge)
    require(digest(PREPARED / 'bridge.py') == BRIDGE_SHA, 'Prepared bridge bytes changed')
    def interrupted(signum, frame):
        raise TimeoutError('Bounded replay deadline/termination; preserve and stop, no retry')
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    bridge.save_new(HERE / 'resource_before_approved_run.json', resources)
    runtime = bridge.bind_runtime(PREPARED / 'inventory_actual8_Full100refs.json', INVENTORY_SHA,
                                 approval['dependency_paths'], REPO, MODEL_ID, approval_path, approved_sha)
    runtime['replay_one'](output, wall_seconds=1800)
    accepted = runtime['accept_saved_predictions'](output)
    source_checks(approval)
    bridge.save_new(HERE / 'strict_acceptance.json', accepted)
    bridge.save_new(HERE / 'resource_after_approved_run.json', resource_snapshot())
    print(json.dumps({'status': accepted['status'], 'id': MODEL_ID, 'max_abs_native_metric_difference': accepted['native_comparison']['max_abs_difference'], 'new_Full_inference': 0, 'new_training': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'run'])
    parser.add_argument('--dispatch-receipt', type=Path)
    parser.add_argument('--dispatch-receipt-sha256')
    args = parser.parse_args()
    if args.action == 'inspect':
        source_checks()
        print(json.dumps({'status': 'PREPARED_NOT_APPROVED', 'id': MODEL_ID, 'cpus': CPU_IDS, 'inference_started': False}))
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
