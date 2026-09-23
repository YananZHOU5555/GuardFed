#!/usr/bin/env bash
set -euo pipefail

cd /home/yannan/workspace/GuardFed

run_cfg() {
  local suite="$1"
  local label="$2"
  local seed="$3"
  local budget="$4"
  local acc_drop="$5"

  echo "[$(date --iso-8601=seconds)] start ${label}_seed${seed}"
  .venv/bin/python scripts/reproduce_paper_tables.py \
    --full \
    --datasets adult compas \
    --distributions IID non-IID \
    --methods GuardFed-AD2+ \
    --attacks FedSA \
    --rounds 70 \
    --seed "${seed}" \
    --server-ratio 0.10 \
    --synthetic-ratio 0 \
    --synthetic-method none \
    --server-sampling stratified_sensitive \
    --ad2-plus-mode adaptive \
    --act-fairness-budget "${budget}" \
    --ad2-calibration-budget "${budget}" \
    --ad2-calibration-max-acc-drop "${acc_drop}" \
    --ad2-calibration-quantiles 81 \
    --ad2-calibration-objective acc_floor \
    --experiment-suite "${suite}" \
    --experiment-tag "${label}_seed${seed}"
  echo "[$(date --iso-8601=seconds)] done ${label}_seed${seed}"
}

for seed in 456 789; do
  run_cfg "ad2plus_adaptive_middle_grid" "b007_cal002_q81" "${seed}" 0.07 0.02
  run_cfg "ad2plus_adaptive_middle_grid" "b005_cal002_q81" "${seed}" 0.05 0.02
  run_cfg "ad2plus_adaptive_calibration_grid" "b006_cal003_q81" "${seed}" 0.06 0.03
done

echo "[$(date --iso-8601=seconds)] all AD2+ seed-completion jobs finished"
