"""Local validation of job guardrails and comparison sensitivity; no image job."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from compare_pilot_resume import equal
from run_fedaa_pilot import validate_job, json_safe

here = Path(__file__).resolve().parent
jobs = sorted((here / "jobs").glob("*.json"))
assert len(jobs) == 2
for p in jobs:
    job = json.loads(p.read_text(encoding="utf-8"))
    validate_job(job)
    for field, value in [("rounds", 70), ("celeba_evaluation_split", "test"),
                         ("celeba_train_limit", 8192), ("client_alpha", 5000),
                         ("learning_rate", .002), ("root_label_noise", .1)]:
        wrong = copy.deepcopy(job)
        wrong["config"][field] = value
        try:
            validate_job(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(field)
a = dict(model=torch.tensor([1., 2.]), replay=np.array([.1, .2]), audit=[float("nan")])
equal(a, copy.deepcopy(a))
for field in ["model", "replay"]:
    b = copy.deepcopy(a)
    b[field][0] += .01
    try:
        equal(a, b)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"Comparator missed {field} mismatch")
assert json_safe({"warnings": [float("nan")]}) == {"warnings": [None]}
report = dict(status="passed", scope="pilot configuration and recovery comparator checks only",
              real_celeba_started=False, jobs_verified=2, wrong_protocol_variants_rejected=12,
              tensor_replay_mismatches_detected=True)
(here / "pilot_runner_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
