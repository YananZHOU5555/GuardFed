#!/usr/bin/env bash
set -u
while ! grep -q '"study": "ratio"' /tmp/GuardFed/results/attack_strength/raw_results.jsonl 2>/dev/null; do
  sleep 60
done
cd /tmp/GuardFed
/tmp/guardfed-venv/bin/python scripts/build_attack_strength_outputs.py
