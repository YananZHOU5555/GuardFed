#!/bin/bash
set -eo pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set -u
stage=/workspace/guardfed_checks/celeba_hybrid_gpu_prepared_20261009
cd "$stage"
read -r approved_sha < "$stage/APPROVED_gate.sha256"
[[ "$approved_sha" =~ ^[0-9a-f]{64}$ ]]
export PYTHONDONTWRITEBYTECODE=1
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -u "$stage/driver.py" run --kind gate --approved "$stage/APPROVED_gate.json" --approved-sha256 "$approved_sha"
