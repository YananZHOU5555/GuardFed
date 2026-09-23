#!/usr/bin/env python3
"""Resume missing main cells for a disjoint subset of methods."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "scripts" / "run_attack_strength_study.py"
spec = importlib.util.spec_from_file_location("attack_strength_runner_method_shard", path)
if spec is None or spec.loader is None:
    raise RuntimeError(path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
index = int(os.environ.get("METHOD_SHARD_INDEX", "0"))
count = int(os.environ.get("METHOD_SHARD_COUNT", "2"))
module.METHODS = list(module.METHODS)[index::count]
module.RAW_PATH = module.RESULTS_DIR / f"raw_results_method_shard{index}.jsonl"
module.FAILURES_PATH = module.RESULTS_DIR / f"failures_method_shard{index}.jsonl"
raise SystemExit(module.main())
