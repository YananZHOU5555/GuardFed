"""SOURCE-ONLY single root-worker example. No queue, discovery or authorization.

Do not call until root freezes the cohort, dependencies, guard and approval.
Import is stdlib-only; main is deliberately not exercised in local source checks.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import types

sys.dont_write_bytecode = True


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load(name, pin):
    path = Path(pin['path'])
    need(set(pin) == {'path', 'sha256'} and path.is_file() and not path.is_symlink()
         and sha(path) == pin['sha256'], 'Unbound or changed module: ' + name)
    need(name not in sys.modules, 'Do not bootstrap a second module instance: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--approval-sha256', required=True)
    parser.add_argument('--wall-seconds', type=int, default=1800)
    a = parser.parse_args()
    need(sys.platform == 'linux' and __debug__ and sha(a.approval) == a.approval_sha256,
         'Actual SHA-bound Linux root approval required')
    approval = json.loads(a.approval.read_bytes())
    need(approval.get('execution_authorized') is True, 'Template is not an authorization')
    def interrupted(_signum, _frame):
        raise TimeoutError('Original800 formal worker interrupted; preserve failure and stop without retry')
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    dependencies = approval['runtime_dependencies']
    # The sole original v2 import sets CUDA hidden and interop=1 itself.
    # Restore the root-bound UUID only afterwards and before any CUDA initialization.
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path.insert(0, str(Path(dependencies['v2']['path']).parent))
    v2 = load('replay', dependencies['v2'])
    need(not v2.torch.cuda.is_initialized() and v2.torch.get_num_interop_threads() == 1,
         'Original single bootstrap required before CUDA initialization')
    os.environ.update(CUDA_VISIBLE_DEVICES=approval['gpu_uuid'], OMP_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    t = v2.torch
    t.set_num_threads(1)
    t.use_deterministic_algorithms(True)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.allow_tf32 = False
    t.backends.cuda.matmul.allow_tf32 = False
    need(len(approval['allowed_cpus']) == 1, 'This example is one CPU/one sequential worker')
    for tid in Path('/proc/self/task').iterdir():
        os.sched_setaffinity(int(tid.name), approval['allowed_cpus'])
    # v3 imports replay; v4 imports replay_v3. Both resolve these same instances.
    v3 = load('replay_v3', dependencies['v3'])
    v4 = load('uniform_original800_v4', dependencies['v4'])
    need(v3.v2 is v2 and v4.v3 is v3 and v4.v2 is v2, 'Mixed original runtime instances')
    adapter = load('uniform_original800_formal', approval['adapter'])
    guard = load('uniform_original800_root_guard', approval['resource_gate_source'])
    inputs = approval['original_inputs']
    args = types.SimpleNamespace(
        command='worker', repo=Path(inputs['repo']), inventory=Path(inputs['inventory']['path']),
        storage_map=Path(inputs['storage_map']['path']), storage_map_sha256=inputs['storage_map']['sha256'],
        restore_receipt=Path(inputs['restore_receipt']['path']), restore_receipt_sha256=inputs['restore_receipt']['sha256'],
        restore_acceptance=Path(inputs['restore_acceptance']['path']), restore_acceptance_sha256=inputs['restore_acceptance']['sha256'])
    bound = adapter.bind_old800(v4, args, approval['exact_cohort'],
                               {'path': str(a.approval), 'sha256': a.approval_sha256}, guard.resource_gate)
    # The original replay fails before saved acceptance if native exceeds 1e-12.
    bound['replay_one'](a.wall_seconds)
    result = bound['accept_saved_predictions']()
    destination = Path(approval['output']) / 'formal_saved_acceptance.json'
    with destination.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'id': result['id'], 'status': result['status'],
                      'acceptance_path': str(destination), 'sha256': sha(destination),
                      'cohort_complete': False}))


if __name__ == '__main__':
    main()
