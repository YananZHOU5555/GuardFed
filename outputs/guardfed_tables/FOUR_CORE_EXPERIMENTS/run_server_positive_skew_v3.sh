#!/usr/bin/env bash
set -euo pipefail

cd /home/yannan/workspace/GuardFed

skews=(0.00 0.05 0.10 0.20 0.35 0.55 0.75 0.90)
for skew in "${skews[@]}"; do
  .venv/bin/python scripts/reproduce_paper_tables.py \
    --full \
    --rounds 20 \
    --seed 123 \
    --experiment-suite goal_server_positive_skew_v3 \
    --experiment-tag "positive_skew_${skew}_seed123" \
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
    --server-sampling controlled_positive_sensitive_skew \
    --server-alpha "${skew}" \
    --ad2-utility-weight 2.50 \
    --ad2-centrality-weight 0.30 \
    --ad2-alignment-weight 1.20 \
    --act-risk-weight 0.80 \
    --act-violation-weight 0.25
done

.venv/bin/python scripts/run_goal_revision_v2.py --phase summarize
