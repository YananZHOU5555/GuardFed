"""Compare trusted full pilot checkpoints, not serialization bytes or durations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def equal(a, b, path="checkpoint"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            raise AssertionError(path)
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or not np.array_equal(a, b, equal_nan=True):
            raise AssertionError(path)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(path + ".keys")
        for key in a:
            equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        if not isinstance(b, type(a)) or len(a) != len(b):
            raise AssertionError(path + ".length/type")
        for i, (left, right) in enumerate(zip(a, b)):
            equal(left, right, f"{path}[{i}]")
    elif isinstance(a, float) and math.isnan(a):
        if not isinstance(b, float) or not math.isnan(b):
            raise AssertionError(path)
    elif a != b:
        raise AssertionError(path)


def compare(continuous, resumed, out):
    a = torch.load(continuous, map_location="cpu", weights_only=False)
    b = torch.load(resumed, map_location="cpu", weights_only=False)
    if a["controller"]["rounds"] != 3 or b["controller"]["rounds"] != 3:
        raise ValueError("Both branches must complete 3 useful rounds")
    equal(a, b)
    report = dict(status="passed", scope="whole 3-round pilot checkpoint tensor/state equality",
                  evidence_stage="pipeline_canary_only", official_table_eligible=False,
                  continuous=dict(path=str(continuous), sha256=hashlib.sha256(continuous.read_bytes()).hexdigest()),
                  resumed=dict(path=str(resumed), sha256=hashlib.sha256(resumed.read_bytes()).hexdigest()),
                  compared_fields=list(a), final_metrics=a["trajectory_metrics"][-1]["metrics"])
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compare(args.continuous, args.resumed, args.out)
