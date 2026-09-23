#!/usr/bin/env bash
set -euo pipefail

cd /home/yannan/workspace/GuardFed

ratios=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
seeds=(123 456 789)

for ratio in "${ratios[@]}"; do
  pct=$(.venv/bin/python - <<PY
print(int(round(float("${ratio}") * 100)))
PY
)
  for seed in "${seeds[@]}"; do
    tag="real${pct}_none_seed${seed}"
    echo "[$(date --iso-8601=seconds)] start ${tag}"
    .venv/bin/python scripts/reproduce_paper_tables.py \
      --full \
      --datasets adult compas \
      --distributions IID non-IID \
      --methods GuardFed-AD2+ \
      --attacks Benign FedSA \
      --rounds 70 \
      --seed "${seed}" \
      --server-ratio "${ratio}" \
      --synthetic-ratio 0 \
      --synthetic-method none \
      --server-sampling stratified_sensitive \
      --ad2-plus-mode fixed \
      --experiment-suite synthetic_stratified_clean_cap10_dense_v4 \
      --experiment-tag "${tag}"
    echo "[$(date --iso-8601=seconds)] done ${tag}"
  done
done

echo "[$(date --iso-8601=seconds)] all synthetic stratified clean-cap10 dense jobs finished"
