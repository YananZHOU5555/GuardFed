"""Fresh actual preflight then one approved, non-retrying supervisor queue."""
from pathlib import Path
import hashlib
import os
import runpy

BASE = Path('/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010')
namespace = runpy.run_path(str(BASE / 'root_operations/remote_preflight_attempt2.py'))
proof = namespace['path']
proof_sha = hashlib.sha256(proof.read_bytes()).hexdigest()
os.environ.update(CUDA_VISIBLE_DEVICES='1', PYTHONDONTWRITEBYTECODE='1', GUARDFED_CPU_THREADS='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
os.execv('/usr/bin/taskset', ['taskset', '-c', '105', '/usr/bin/nice', '-n', '10', '/usr/bin/ionice', '-c', '3',
    '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python', '-B', '-u', str(BASE / 'run_queue.py'),
    '--repo', '/workspace/GuardFed-celeba-expanded', '--out', '/workspace/celeba_gradient_screen64_v2_results_20261010',
    '--resource-preflight', str(proof), '--resource-sha256', proof_sha])
