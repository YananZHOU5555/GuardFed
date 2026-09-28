"""Run offline CPU comparison with pinned official source and boundary checks."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch

from adapter import lasa_aggregate

HERE = Path(__file__).resolve().parent


def official_oracle(updates, sparsity, lambda_n, lambda_s):
    mask_source = (HERE / "upstream/mask_help.py").read_text(encoding="utf-8-sig")
    source = (HERE / "upstream/lasa.py").read_text(encoding="utf-8-sig")
    # Only portability change: remove hard-coded CUDA from new mask creation.
    env = {"oracle_trace": {}}
    exec(compile(mask_source.replace(".cuda()", ""), "official_mask_cpu", "exec"), env)
    source = source.replace("from utils.mask_help import *", "")
    source = source.replace(
        "        key_mean_weight[key] = torch.mean",
        "        oracle_trace[key] = {'selected_clients': sorted(benign_idx), 'norm_selected': sorted(benign_idx1), 'sign_selected': sorted(benign_idx2)}\n        key_mean_weight[key] = torch.mean",
    )
    exec(compile(source, "official_lasa_instrumented", "exec"), env)
    initial = {k: torch.zeros_like(v) for k, v in updates[0].items()}
    args = SimpleNamespace(sparsity=sparsity, lambda_n=lambda_n, lambda_s=lambda_s, num_selected_users=len(updates))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = env["lasa"](copy.deepcopy(updates), initial, args)
    return result, env["oracle_trace"]


def main():
    records = []
    # Distinct signs and magnitudes exercise both filters, global clipping and masks.
    rng = torch.Generator().manual_seed(28371)
    random_updates = [
        {"conv.weight": torch.randn(3, 2, 3, 3, generator=rng) * (i + 1),
         "conv.bias": torch.randn(3, generator=rng),
         "head.weight": torch.randn(2, 3, generator=rng),
         "head.bias": torch.randn(2, generator=rng)}
        for i in range(8)
    ]
    tie_updates = [{"weight": torch.tensor([[1., 1.], [-1., -1.]]) * scale,
                    "bias": torch.tensor([.2, -.3]) * scale} for scale in (1., 2., 3., 4.)]
    identical = [{"weight": torch.tensor([[1., 2.], [-3., 4.]]), "bias": torch.tensor([1., -2.])} for _ in range(4)]
    for name, updates, sparse, ln, ls in [
        ("random_default", random_updates, .3, 1., 1.),
        ("random_no_sparsity", random_updates, 0., 1.5, .8),
        ("random_high_sparsity", random_updates, .8, 2., 2.),
        ("strict_ties", tie_updates, .5, 1., 1.),
        ("identical_zero_std", identical, 0., 1., 1.),
    ]:
        before = copy.deepcopy(updates)
        actual, audit = lasa_aggregate(updates, sparsity=sparse, lambda_n=ln, lambda_s=ls)
        expected, trace = official_oracle(updates, sparse, ln, ls)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for field in ("selected_clients", "norm_selected", "sign_selected"):
                assert audit["layers"][key][field] == trace[key][field], (name, key, field)
        for a, b in zip(updates, before):
            assert all(torch.equal(a[k], b[k]) for k in a), "input mutation"
        json.dumps(audit, allow_nan=False)
        if name == "strict_ties":
            assert all(m["nonzero_after_by_layer"]["weight"] == 0 for m in audit["masks"])
        if name == "identical_zero_std":
            assert all(layer["fallback_all"] for layer in audit["layers"].values())
        records.append({"case": name, "official_output_bitwise_equal": True, "selection_equal": True})

    # Explicit extension relative to upstream: no 0/0 in whole-vector clipping.
    zeros = [{"weight": torch.zeros(2, 2), "bias": torch.zeros(2)} for _ in range(4)]
    actual, audit = lasa_aggregate(zeros)
    assert all(torch.equal(v, torch.zeros_like(v)) for v in actual.values())
    assert audit["zero_norm_clients"] == [0, 1, 2, 3]
    assert all(v["fallback_all"] for v in audit["layers"].values())
    # Use matching dictionaries to isolate zero-vector behaviour.
    mixed = [{k: torch.zeros_like(v) for k, v in random_updates[0].items()}, *random_updates[:3]]
    actual, audit = lasa_aggregate(mixed)
    assert audit["zero_norm_clients"] == [0]
    assert all(torch.isfinite(v).all() for v in actual.values())
    records.append({"case": "zero_norm_explicit_repair", "finite_zero_and_mixed": True})

    for value in (float("nan"), float("inf")):
        bad = copy.deepcopy(random_updates)
        bad[2]["head.bias"][0] = value
        try:
            lasa_aggregate(bad)
        except ValueError as error:
            assert "non-finite" in str(error)
        else:
            raise AssertionError("nonfinite input silently accepted")
    for kwargs in ({"sparsity": 1.}, {"sparsity": .999999}, {"lambda_n": 0.}, {"lambda_s": float("nan")}):
        try:
            lasa_aggregate(random_updates, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid config silently accepted")
    records.append({"case": "nonfinite_and_invalid_config", "fail_fast": True})
    files = ["adapter.py", "check_adapter.py", "upstream/lasa.py", "upstream/mask_help.py", "upstream/main.py"]
    report = {"status": "PASS", "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
              "upstream_commit": "8477367a4e8708cde264f7572805040c650af59f", "cases": records,
              "hashes": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in files},
              "scope": "aggregation only; no image training, GPU determinism or worker integration asserted"}
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cases": len(records), "official_bitwise_cases": 5}))


if __name__ == "__main__":
    main()
