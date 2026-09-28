"""Offline checks of pilot job constraints and aggregation-only routing; no training."""
import copy
import hashlib
import json
from pathlib import Path

import torch

from adapter import lasa_aggregate
from run_pilot import METHOD, aggregation_wrapper, validate_job

HERE = Path(__file__).resolve().parent


def main():
    for attack in ("Benign", "S-DFA"):
        job = json.loads((HERE / f"job_{attack}.json").read_text())
        validate_job(job)
        for name, expected in job["adapter_source_hashes"].items():
            assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
        for key, value in (("rounds", 70), ("celeba_evaluation_split", "test"), ("celeba_train_limit", 100)):
            bad = copy.deepcopy(job)
            bad["config"][key] = value
            try:
                validate_job(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("out-of-scope pilot config accepted")
    calls = []
    sentinel = object()
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    wrapper = aggregation_wrapper(original, {"sparsity": .3, "lambda_n": 1., "lambda_s": 1.})
    updates = [{"w": torch.tensor([[1., -2.], [3., 4.]]) * scale} for scale in (1., 2., 4.)]
    assert wrapper("FedAvg", updates, [1, 2, 3], [.1] * 3, {}, None, bundle="unchanged") is sentinel
    assert calls[-1][1] == {"bundle": "unchanged"}
    actual, info = wrapper(METHOD, updates, [1, 2, 3], [.1] * 3, {}, None, bundle="unchanged")
    expected, _ = lasa_aggregate(updates)
    assert len(calls) == 1
    assert torch.equal(actual["w"], expected["w"])
    assert info["client_counts_for_identity_only"] == [1, 2, 3]
    report = {"status": "PASS", "job_count": 2, "adapter_hashes_match": True,
              "routing": "only LASA-official intercepted; other methods/kwargs passed unchanged",
              "constraints": "3 rounds, full train, valid-only; 70-round/test/subset changes rejected",
              "scope": "offline synthetic aggregation check, no image data loaded or training started"}
    (HERE / "pilot_entry_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
