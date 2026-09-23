#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

run_cfg() {
  local label="$1"
  local seed="$2"
  local budget="$3"

  echo "[$(date --iso-8601=seconds)] start ${label}_seed${seed}"
  python scripts/reproduce_paper_tables.py \
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
    --ad2-calibration-max-acc-drop 0.0 \
    --ad2-calibration-quantiles 81 \
    --ad2-calibration-objective acc_floor \
    --experiment-suite "ad2plus_adaptive_acc_refine" \
    --experiment-tag "${label}_seed${seed}"
  echo "[$(date --iso-8601=seconds)] done ${label}_seed${seed}"
}

for seed in 123 456 789; do
  run_cfg "b010_accfloor0_q81" "${seed}" 0.10
  run_cfg "b012_accfloor0_q81" "${seed}" 0.12
  run_cfg "b015_accfloor0_q81" "${seed}" 0.15
  run_cfg "b020_accfloor0_q81" "${seed}" 0.20
done

echo "[$(date --iso-8601=seconds)] all AD2+ acc-refine jobs finished"
