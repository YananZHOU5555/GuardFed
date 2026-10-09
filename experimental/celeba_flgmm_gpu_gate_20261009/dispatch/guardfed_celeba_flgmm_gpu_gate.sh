#!/bin/bash
set -e
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /workspace/guardfed_checks/celeba_flgmm_gpu_gate_20261009
pty /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -u run_gpu_gate.py --repo /workspace/GuardFed-celeba-expanded --cpu-root /workspace/guardfed_checks/celeba_flgmm_realimage_gate_20261009
