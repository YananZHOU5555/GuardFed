"""Compare model, metrics and complete policy against the frozen 3-round pilot.

Metadata intentionally differs; this is a regression gate, not source identity
acceptance or a replacement for exact-identity screen checkpoint recovery.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def compare(pilot, screen, out):
    compare_file = Path(__file__).resolve().parents[2] / "integration_20260928/compare_pilot_resume.py"
    spec = importlib.util.spec_from_file_location("frozen_pilot_comparator", compare_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = torch.load(pilot, map_location="cpu", weights_only=False)
    b = torch.load(screen, map_location="cpu", weights_only=False)
    assert a["controller"]["rounds"] == b["controller"]["rounds"] == 3
    fields = ["model", "controller", "rng", "trajectory_metrics", "round_summaries", "attack_audit", "warnings"]
    for field in fields:
        mod.equal(a[field], b[field], field)
    report = dict(status="passed", scope="three-round default-parameter regression gate",
                  compared_fields=fields, ignored_fields=["identity"],
                  pilot=str(pilot), new_screen_gate=str(screen),
                  evidence_stage="pipeline_canary_only", formal_table_eligible=False)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--screen", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    compare(args.pilot, args.screen, args.out)
