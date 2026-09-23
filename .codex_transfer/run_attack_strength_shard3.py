#!/usr/bin/env python3
"""Run the second disjoint seed shard with isolated JSONL output."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "scripts" / "run_attack_strength_study.py"
spec = importlib.util.spec_from_file_location("attack_strength_runner_shard3", path)
if spec is None or spec.loader is None:
    raise RuntimeError(path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.SEEDS = [3141, 4242, 5050, 6060, 7070]
module.RAW_PATH = module.RESULTS_DIR / "raw_results_shard3.jsonl"
module.FAILURES_PATH = module.RESULTS_DIR / "failures_shard3.jsonl"
raise SystemExit(module.main())
