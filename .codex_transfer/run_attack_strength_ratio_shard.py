#!/usr/bin/env python3
"""Run a disjoint seed shard for the AD2+ ratio study."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "scripts" / "run_attack_strength_study.py"
spec = importlib.util.spec_from_file_location("attack_strength_runner_ratio_shard", path)
if spec is None or spec.loader is None:
    raise RuntimeError(path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
index = int(os.environ.get("RATIO_SHARD_INDEX", "0"))
module.SEEDS = ([123, 789, 2024, 4242, 6060] if index == 0 else [456, 1001, 3141, 5050, 7070])
module.RAW_PATH = module.RESULTS_DIR / f"raw_results_ratio_shard{index}.jsonl"
module.FAILURES_PATH = module.RESULTS_DIR / f"failures_ratio_shard{index}.jsonl"
raise SystemExit(module.main())
