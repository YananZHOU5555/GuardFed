#!/usr/bin/env bash
set -euo pipefail

cd /home/yannan/workspace/GuardFed

SEEDS=(123 456 789)
CONFIGS=(
  "gb_real5_none 0.05 0 none"
  "gb_real7_none 0.07 0 none"
  "gb_real10_none 0.10 0 none"
  "gb_real15_none 0.15 0 none"
  "gb_real7_gaussian_synth3 0.07 0.03 gaussian_copula"
  "gb_real5_smote_synth5 0.05 0.05 smote"
)

for seed in "${SEEDS[@]}"; do
  for row in "${CONFIGS[@]}"; do
    set -- $row
    tag="$1_seed${seed}"
    server_ratio="$2"
    synth_ratio="$3"
    synth_method="$4"
    echo "=== synthetic_balanced_ratios_v2 $tag server=$server_ratio synth=$synth_ratio method=$synth_method ===" | tee -a results/paper_tables/expanded_experiments/balanced_synth_v2.log
    .venv/bin/python scripts/reproduce_paper_tables.py \
      --full \
      --rounds 70 \
      --seed "$seed" \
      --datasets adult compas \
      --distributions IID non-IID \
      --methods GuardFed-AD2+ \
      --attacks Benign FedSA \
      --device cuda \
      --server-ratio "$server_ratio" \
      --synthetic-ratio "$synth_ratio" \
      --synthetic-method "$synth_method" \
      --server-sampling group_balanced \
      --ad2-plus-mode fixed \
      --act-fairness-budget 0.12 \
      --act-fairness-metric aeod_aspd \
      --act-risk-weight 0.10 \
      --act-violation-weight 0.02 \
      --act-keep-ratio 1.0 \
      --act-temperature 0.80 \
      --ad2-utility-weight 3.0 \
      --ad2-centrality-weight 0.2 \
      --ad2-alignment-weight 1.5 \
      --ad2-score-clip 5 \
      --ad2-norm-mode root \
      --ad2-calibration-objective original \
      --sdfa-foe-mode fedsa \
      --spdfa-foe-mode fedsa \
      --experiment-suite synthetic_balanced_ratios_v2 \
      --experiment-tag "$tag" \
      >> results/paper_tables/expanded_experiments/balanced_synth_v2.log 2>&1
  done
done
