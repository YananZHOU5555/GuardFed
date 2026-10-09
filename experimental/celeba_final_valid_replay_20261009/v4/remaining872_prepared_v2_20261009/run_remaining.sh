#!/bin/bash
set -euo pipefail
exec /workspace/guardfed_envs/celeba-cu128-20261009/bin/python /workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/remaining872_prepared_v2_20261009/bounded_remaining.py run --manifest /workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/remaining872_prepared_v2_20261009/manifest.json --manifest-sha256 ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43
