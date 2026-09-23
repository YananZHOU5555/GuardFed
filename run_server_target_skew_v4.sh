#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

skews=(0.00 0.05 0.10 0.15 0.20 0.30)
targets=("0 0" "0 1" "1 0" "1 1")

for target in "${targets[@]}"; do
  read -r sens label <<<"${target}"
  for skew in "${skews[@]}"; do
    python scripts/reproduce_paper_tables.py \
      --full \
      --rounds 15 \
      --seed 123 \
      --experiment-suite goal_server_target_skew_v4 \
      --experiment-tag "target_s${sens}_y${label}_skew_${skew}_seed123" \
      --server-ratio 0.10 \
      --synthetic-ratio 0 \
      --synthetic-method none \
      --device cuda \
      --methods GuardFed-AD2+ \
      --ad2-plus-mode fixed \
      --act-fairness-budget 0.08 \
      --act-temperature 0.50 \
      --act-keep-ratio 0.80 \
      --act-fairness-metric aeod_aspd \
      --ad2-score-clip 5 \
      --ad2-norm-mode root \
      --sdfa-foe-mode fedsa \
      --spdfa-foe-mode fedsa \
      --fedsa-gain 3.00 \
      --fedsa-norm-ratio 3.50 \
      --datasets adult compas \
      --distributions IID non-IID \
      --attacks Benign FedSA \
      --server-sampling controlled_target_group_skew \
      --server-alpha "${skew}" \
      --server-target-sensitive "${sens}" \
      --server-target-label "${label}" \
      --ad2-utility-weight 2.50 \
      --ad2-centrality-weight 0.30 \
      --ad2-alignment-weight 1.20 \
      --act-risk-weight 0.80 \
      --act-violation-weight 0.25
  done
done

python scripts/run_goal_revision_v2.py --phase summarize
