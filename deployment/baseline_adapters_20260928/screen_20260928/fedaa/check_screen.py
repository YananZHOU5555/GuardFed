"""CPU component regression and exact candidate-grid checks, not CelebA training."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import torch

from fedaa_round_adapter import FedAARounds
from run_fedaa_screen import digest, validate_job

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parents[1]
torch.set_num_threads(1)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = load("frozen_round_adapter", BASELINES / "integration_20260928/fedaa_round_adapter.py")
compare = load("frozen_compare", BASELINES / "integration_20260928/compare_pilot_resume.py")
official = BASELINES / "fedaa/official"
model = torch.nn.Linear(2, 2)
frozen = old.FedAARounds(model, official, 91001, .5)
new = FedAARounds(model, official, 91001, .5)
for rnd in range(3):
    states = [{k: v.detach().clone() + (cid + 1) * .01 + rnd * .003
               for k, v in model.state_dict().items()} for cid in range(20)]
    a, da = frozen.aggregate(states, list(range(20)))
    b, db = new.aggregate(states, list(range(20)))
    compare.equal(a, b)
    compare.equal(da, db)
    frozen.observe_root(.6)
    new.observe_root(.6)
compare.equal(frozen.state_dict(), new.state_dict())
for lr, keep in itertools.product([.001, .01], [10, 16]):
    candidate = FedAARounds(model, official, 91001, .5, keep=keep, actor_lr=lr, critic_lr=lr)
    _, diag = candidate.aggregate(states, list(range(20)))
    candidate.observe_root(.6)
    assert len(diag["selected_client_ids"]) == keep
    state = candidate.state_dict()
    assert state["policy"]["config"]["actor_lr"] == state["policy"]["config"]["critic_lr"] == lr
    for opt in ["actor_optimizer", "critic_optimizer"]:
        assert state["policy"][opt]["param_groups"][0]["lr"] == lr
    resumed = FedAARounds(model, official, 91001, .5, keep=keep, actor_lr=lr, critic_lr=lr)
    resumed.load_state_dict(state)
    compare.equal(candidate.state_dict(), resumed.state_dict())
manifest = json.loads((HERE / "screen_jobs/manifest.json").read_text(encoding="utf-8"))
assert manifest["new_run_count"] == 32 and manifest["candidate_count"] == 8
seen = set()
for item in manifest["jobs"]:
    p = HERE / "screen_jobs" / item["path"]
    assert digest(p) == item["sha256"]
    job = json.loads(p.read_text(encoding="utf-8"))
    validate_job(job)
    key = (job["policy_config"]["actor_lr"], job["aggre_num"], job["config"]["learning_rate"], job["distribution"], job["attack"])
    assert key not in seen
    seen.add(key)
    wrong = copy.deepcopy(job)
    wrong["config"]["celeba_evaluation_split"] = "test"
    try:
        validate_job(wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("Test split accepted")
assert seen == set(itertools.product([.001,.01],[10,16],[.0005,.001],["IID","non-IID"],["Benign","S-DFA"]))
report = dict(status="passed", device="cpu", torch=torch.__version__,
              default_controller_three_rounds_equal_to_frozen_pilot=True,
              policy_lr_keep_combinations_checked=4, screen_jobs_verified=32,
              candidates_verified=8, grid_complete=True,
              actual_celeba_run=False, GPU_regression_gate_pending=True)
(HERE / "local_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
