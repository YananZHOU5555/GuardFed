#!/bin/bash
set -e
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GUARDFED_CPU_THREADS=1
cd /workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2
pty nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -u run_screen.py --repo /workspace/GuardFed-celeba-expanded
