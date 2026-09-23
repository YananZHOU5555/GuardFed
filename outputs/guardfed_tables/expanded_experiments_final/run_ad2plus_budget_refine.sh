#!/usr/bin/env bash
set -euo pipefail

cd /home/yannan/workspace/GuardFed

run_cfg() {
  local label="$1"
  local seed="$2"
  local budget="$3"

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
    --ad2-calibration-max-acc-drop 0.02 \
    --ad2-calibration-quantiles 81 \
    --ad2-calibration-objective acc_floor \
    --experiment-suite "ad2plus_adaptive_budget_refine" \
    --experiment-tag "${label}_seed${seed}"
  echo "[$(date --iso-8601=seconds)] done ${label}_seed${seed}"
}

for seed in 123 456 789; do
  run_cfg "b006_cal002_q81" "${seed}" 0.06
  run_cfg "b0065_cal002_q81" "${seed}" 0.065
done

echo "[$(date --iso-8601=seconds)] all AD2+ budget-refine jobs finished"
