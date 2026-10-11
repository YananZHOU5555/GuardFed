"""Single FL100 saved-checkpoint CUDA worker; no training, test or retry."""
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
        raise TimeoutError('FL100 formal worker interrupted; preserve failure and stop without retry')
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    dependencies = approval['runtime_dependencies']
    need(approval['allowed_cpus']==[111] and set(os.sched_getaffinity(0))=={111},
         'OS taskset CPU111 must precede all scientific imports')
    for tid in Path('/proc/self/task').iterdir():
        need(os.sched_getaffinity(int(tid.name))=={111},'Pre-import thread escaped CPU111')
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=approval['gpu_uuid'],
                      CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,approval['repo'])
    # One direct bootstrap. FL bound_bridge constructs original function AST;
    # importing replay.py here would reset its CPU8/CUDA-hidden globals.
    import torch
    import pandas
    need(not torch.cuda.is_initialized(),'CUDA initialized before approved bootstrap')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    for tid in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(tid.name),[111])
    candidate=load('uniform_FL100_original_candidate',dependencies['candidate'])
    adapter=load('uniform_FL100_formal',approval['adapter'])
    guard=load('uniform_FL100_root_guard',approval['resource_gate_source'])
    args=types.SimpleNamespace(repo=Path(approval['repo']),torch_module=torch,pandas_module=pandas)
    bound = adapter.bind_fl(candidate, args, approval['exact_cohort'],
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
