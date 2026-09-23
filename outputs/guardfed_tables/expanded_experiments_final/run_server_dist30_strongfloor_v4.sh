#!/usr/bin/env bash
set -euo pipefail

cd /home/yannan/workspace/GuardFed

alphas=(0.03 0.04 0.05 0.07 0.10 0.15 0.20 0.30 0.50 0.75 1 1.5 2 3 5 7.5 10 15 20 30 50 75 100 200 500 1000 2000 3000 4000 5000)
seeds=(123 456 789)

for alpha in "${alphas[@]}"; do
  alpha_tag=${alpha//./_}
  for seed in "${seeds[@]}"; do
    tag="alpha_${alpha_tag}_seed${seed}"
    echo "[$(date --iso-8601=seconds)] start ${tag}"
    .venv/bin/python scripts/reproduce_paper_tables.py \
      --full \
      --datasets adult compas \
      --distributions IID non-IID \
      --methods GuardFed-AD2+ \
      --attacks Benign FedSA \
      --rounds 70 \
      --seed "${seed}" \
      --server-ratio 0.10 \
      --synthetic-ratio 0 \
      --synthetic-method none \
      --server-sampling dirichlet_label_preserved_strong_floor \
      --server-alpha "${alpha}" \
      --ad2-plus-mode fixed \
      --experiment-suite expanded_server_dist30_strongfloor_v4 \
      --experiment-tag "${tag}"
    echo "[$(date --iso-8601=seconds)] done ${tag}"
  done
done

echo "[$(date --iso-8601=seconds)] all server-dist30 strongfloor jobs finished"
