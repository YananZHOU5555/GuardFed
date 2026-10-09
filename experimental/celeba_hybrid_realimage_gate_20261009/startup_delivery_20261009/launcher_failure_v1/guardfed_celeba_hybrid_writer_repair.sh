#!/bin/bash
set -euo pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
stage=/workspace/guardfed_checks/celeba_hybrid_realimage_gate_20261009
cd "$stage/execution_repair_v1"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -u "$stage/execution_repair_v1/repair_execute.py" run --approved "$stage/execution_approved_20261009/APPROVED.json" 2>&1 | tee "$stage/execution_approved_20261009/run.log"
