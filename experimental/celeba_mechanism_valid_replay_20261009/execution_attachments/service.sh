#!/bin/bash
set -e
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" /workspace/guardfed_checks/celeba_mechanism_valid_replay_20261009/execution_attachments/run.log
. "${utils}/environment.sh"
cd /workspace/guardfed_checks/celeba_mechanism_valid_replay_20261009/execution_attachments
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
approved_sha="$(cat dispatch_receipt.APPROVED.sha256)"
exec /usr/bin/nice -n 10 /usr/bin/ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -u execute_one.py run --dispatch-receipt dispatch_receipt.APPROVED.json --dispatch-receipt-sha256 "${approved_sha}"
