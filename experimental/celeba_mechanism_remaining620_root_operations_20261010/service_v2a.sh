#!/bin/bash
set -e
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" /workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010/queue_v2a.log
. "${utils}/environment.sh"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
exec /usr/bin/taskset -c 112-119 /usr/bin/nice -n10 /usr/bin/ionice -c3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -u /workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010/evaluate_remaining.py manage --review /workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010/ROOT_APPROVED.json --review-sha256 55e0c1a5c08fd00a33ff1caaa862b5b1e67c4328559b750a95b6d0a5d1aebb6c
