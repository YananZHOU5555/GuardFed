#!/usr/bin/env bash
set -u
while pgrep -f 'scripts/run_attack_strength_method_shard.py --phase main' >/dev/null 2>&1; do
  sleep 60
done
cd /tmp/GuardFed
/tmp/guardfed-venv/bin/python scripts/dedup_attack_results.py
while pgrep -f 'scripts/run_attack_strength_ratio_shard.py --phase ratio' >/dev/null 2>&1; do
  sleep 60
done
/tmp/guardfed-venv/bin/python scripts/dedup_ratio_results.py
/tmp/guardfed-venv/bin/python scripts/run_attack_strength_study.py --phase summarize --rounds 70 --device cuda
/tmp/guardfed-venv/bin/python scripts/build_attack_strength_outputs.py
/tmp/guardfed-venv/bin/python scripts/build_attack_strength_outputs.py
