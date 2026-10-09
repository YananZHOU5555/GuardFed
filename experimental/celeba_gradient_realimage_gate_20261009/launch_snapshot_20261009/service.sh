#!/bin/bash
set -e
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" /workspace/guardfed_checks/celeba_gradient_realimage_gate_20261009/gate.log
. "${utils}/environment.sh"
cd /workspace/guardfed_checks/celeba_gradient_realimage_gate_20261009
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
exec /usr/bin/nice -n 10 /usr/bin/ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -u shared_cache_wrapper.py run
