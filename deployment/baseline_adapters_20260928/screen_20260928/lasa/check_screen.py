"""Offline screen coverage, freeze, routing and pilot-equivalence checks; no training."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from prepare_jobs import make_jobs
from worker import METHOD, aggregation_wrapper, validate_job

HERE = Path(__file__).resolve().parent


def main():
    screen, preflight = make_jobs("screen"), make_jobs("preflight")
    assert len(screen) == 32 and len(preflight) == 2
    assert len({j["id"] for j in screen + preflight}) == 34
    assert len({j["tuning_candidate"] for j in screen}) == 8
    for cid in {j["tuning_candidate"] for j in screen}:
        assert {(j["distribution"], j["attack"]) for j in screen if j["tuning_candidate"] == cid} == {
            (d, a) for d in ("IID", "non-IID") for a in ("Benign", "S-DFA")}
    for job in screen + preflight:
        validate_job(job)
        assert job["config"]["rounds"] == (70 if job["phase"] == "screen" else 3)
        stored = json.loads((HERE / "jobs" / job["phase"] / (job["id"] + ".json")).read_text())
        assert stored == job
        for filename, expected in job["adapter_source_hashes"].items():
            assert hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == expected
    for field, value in (("rounds", 3), ("celeba_evaluation_split", "test"), ("root_label_noise", .1),
                         ("learning_rate", .003), ("client_alpha", .1)):
        bad = copy.deepcopy(screen[0])
        bad["config"][field] = value
        try:
            validate_job(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unfrozen config accepted: " + field)
    bad = copy.deepcopy(screen[0])
    bad["adapter"]["lambda_s"] = 2.
    try:
        validate_job(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("asymmetric lambda not in grid accepted")

    pilot_dir = HERE.parents[1] / "lasa_20260928"
    pilot_adapter_hash = hashlib.sha256((pilot_dir / "adapter.py").read_bytes()).hexdigest()
    assert pilot_adapter_hash == hashlib.sha256((HERE / "adapter.py").read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("lasa_original_pilot_for_check", pilot_dir / "run_pilot.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    for job in preflight:
        prior = json.loads((pilot_dir / ("job_" + job["attack"] + ".json")).read_text())
        for key in prior["config"]:
            if key not in {"experiment_suite", "experiment_tag"}:
                assert prior["config"][key] == job["config"][key], key
        assert prior["source_hashes"] == job["source_hashes"]
    sentinel = object()
    def original(*args, **kwargs):
        return sentinel
    parameters = preflight[0]["adapter"]
    new_route, old_route = aggregation_wrapper(original, parameters), pilot.aggregation_wrapper(original, parameters)
    updates = [{"w": torch.tensor([[1., -2.], [3., 4.]]) * scale} for scale in (1., 2., 4.)]
    new_output, new_diag = new_route(METHOD, updates, [1, 2, 3], [.1] * 3, {}, None)
    old_output, old_diag = old_route(METHOD, updates, [1, 2, 3], [.1] * 3, {}, None)
    assert torch.equal(new_output["w"], old_output["w"])
    assert new_diag == old_diag
    assert new_route("FedAvg", updates, [1, 2, 3], [.1] * 3, {}, None) is sentinel
    report = dict(status="PASS", screen_jobs=32, candidates=8, preflight_jobs=2,
                  adapter_byte_identical_to_passed_pilot=True, preflight_training_config_same=True,
                  synthetic_aggregation_output_and_diagnostics_equal=True,
                  all_generated_job_hashes_match=True, scope="offline only; no image training or server activity")
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
