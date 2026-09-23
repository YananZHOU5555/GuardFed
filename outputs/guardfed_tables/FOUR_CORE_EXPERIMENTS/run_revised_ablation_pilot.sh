#!/usr/bin/env bash
set -euo pipefail

cd /home/yannan/workspace/GuardFed

run_profile() {
  local tag="$1"
  local attack="$2"
  shift 2

  echo "[$(date --iso-8601=seconds)] start ${tag}_${attack}"
  .venv/bin/python scripts/reproduce_paper_tables.py \
    --full \
    --datasets adult compas \
    --distributions IID non-IID \
    --methods GuardFed-AD2+ \
    --attacks "${attack}" \
    --rounds 20 \
    --seed 123 \
    --server-ratio 0.10 \
    --synthetic-ratio 0 \
    --synthetic-method none \
    --server-sampling stratified_sensitive \
    --ad2-plus-mode fixed \
    --act-fairness-budget 0.08 \
    --act-temperature 0.50 \
    --act-keep-ratio 0.80 \
    --act-fairness-metric aeod_aspd \
    --ad2-score-clip 5 \
    --ad2-norm-mode root \
    --fedsa-gain 2.50 \
    --fedsa-norm-ratio 3.00 \
    --sdfa-foe-mode fedsa \
    --spdfa-foe-mode fedsa \
    --experiment-suite revised_ablation_pilot \
    --experiment-tag "${tag}_${attack}_seed123" \
    "$@"
  echo "[$(date --iso-8601=seconds)] done ${tag}_${attack}"
}

attacks=(FedSA "F Flip")
for attack in "${attacks[@]}"; do
  run_profile full "${attack}" \
    --ad2-utility-weight 2.50 --ad2-centrality-weight 0.30 --ad2-alignment-weight 1.20 \
    --act-risk-weight 0.80 --act-violation-weight 0.25
  run_profile no_utility_U "${attack}" \
    --ad2-utility-weight 0.00 --ad2-centrality-weight 0.30 --ad2-alignment-weight 1.20 \
    --act-risk-weight 0.80 --act-violation-weight 0.25
  run_profile no_fairness_FV "${attack}" \
    --ad2-utility-weight 2.50 --ad2-centrality-weight 0.30 --ad2-alignment-weight 1.20 \
    --act-risk-weight 0.00 --act-violation-weight 0.00
  run_profile no_geometry_CA "${attack}" \
    --ad2-utility-weight 2.50 --ad2-centrality-weight 0.00 --ad2-alignment-weight 0.00 \
    --act-risk-weight 0.80 --act-violation-weight 0.25
  run_profile utility_only_U "${attack}" \
    --ad2-utility-weight 2.50 --ad2-centrality-weight 0.00 --ad2-alignment-weight 0.00 \
    --act-risk-weight 0.00 --act-violation-weight 0.00
  run_profile fairness_only_FV "${attack}" \
    --ad2-utility-weight 0.00 --ad2-centrality-weight 0.00 --ad2-alignment-weight 0.00 \
    --act-risk-weight 0.80 --act-violation-weight 0.25
done

echo "[$(date --iso-8601=seconds)] revised ablation pilot finished"
