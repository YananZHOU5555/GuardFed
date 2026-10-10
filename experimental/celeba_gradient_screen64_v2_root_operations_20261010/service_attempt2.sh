#!/bin/bash
set -e
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" /workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010/queue_attempt2.log
. "${utils}/environment.sh"
export PYTHONDONTWRITEBYTECODE=1
exec /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -u /workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010/root_operations/root_launch_attempt2.py
