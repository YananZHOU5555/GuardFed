"""Synthetic-parameter timing/resume check and fabricated RGB64 CNN plumbing check.

These are implementation checks. No real CelebA data or formal results are used.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import random

import numpy as np
import torch

from fedaa_round_adapter import FedAARounds, train_and_aggregate_round

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent / "fedaa" / "official"
torch.set_num_threads(1)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def check_controller():
    model = torch.nn.Linear(2, 2)
    rounds = FedAARounds(model, OFFICIAL, 7101, .5, participants=4, keep=2)
    initial_actor = copy.deepcopy(rounds.policy.agent.actor.state_dict())
    for rnd in range(3):
        states = [{k: v.detach() + (i + 1) * .02 + rnd * .001 for k, v in model.state_dict().items()}
                  for i in range(4)]
        before_rng = torch.get_rng_state().clone()
        result, info = rounds.aggregate(states, list(range(4)))
        assert torch.equal(before_rng, torch.get_rng_state())
        expected = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        for w, cid in zip(info["action"], info["selected_client_ids"]):
            for k in expected:
                expected[k] += states[cid][k] * w
        same(result, expected)
        assert info["policy_updates"] == rnd + 1
        rounds.observe_root(.6 + rnd * .02)
        model.load_state_dict(result)
    assert any(not torch.equal(initial_actor[k], v) for k, v in rounds.policy.agent.actor.state_dict().items())
    clone = FedAARounds(model, OFFICIAL, 7101, .5, participants=4, keep=2)
    clone.load_state_dict(rounds.state_dict())
    for rnd in range(3):
        states = [{k: v.detach() + i * .003 for k, v in model.state_dict().items()} for i in range(4)]
        a, da = rounds.aggregate(states, list(range(4)))
        b, db = clone.aggregate(states, list(range(4)))
        same(a, b)
        same(da, db)
        rounds.observe_root(.7)
        clone.observe_root(.7)
    same(rounds.state_dict(), clone.state_dict())
    try:
        rounds.aggregate(states, [0, 1, 3, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("Unfrozen client ordering accepted")


def check_cnn_path():
    sys.path.insert(0, str(HERE / "source_snapshot" / "scripts"))
    import reproduce_paper_tables as core
    from src.celeba_data import CelebACNN
    cfg = core.ExperimentConfig(rounds=3, num_clients=4, num_malicious=0,
                                local_epochs=1, batch_size=4, learning_rate=.0005,
                                include_sensitive_feature=False, synthetic_ratio=0.,
                                celeba_evaluation_split="valid", device="cpu")
    torch.manual_seed(270928)
    device = torch.device("cpu")
    X = torch.randint(0, 256, (32, 3, 64, 64), dtype=torch.uint8)
    y = torch.tensor([0, 0, 1, 1] * 8)
    sensitive = np.array([0, 1, 0, 1] * 8)
    clients, audits = [], []
    for cid in range(4):
        span = slice(cid * 4, cid * 4 + 4)
        raw = dict(X=X[span], y=y[span], sensitive=sensitive[span], sensitive_feature_index=None)
        client, audit = core.client_runtime_data(raw, cid, "Benign", [],
                {(s, label): 1. for s in [0, 1] for label in [0, 1]}, device,
                cfg.fflip_mode, cfg.foe_mode, cfg.sdfa_foe_mode, cfg.spdfa_foe_mode)
        clients.append(client)
        audits.append(audit)
    bundle = dict(dataset="celeba", num_features=3, server_X=X[16:], server_y=y[16:],
                  server_sensitive=sensitive[16:], root_synthetic_rows=0)
    model = CelebACNN(cfg.seed)
    initial_root = core.evaluate_model(model, X[16:], y[16:], sensitive[16:], cfg.batch_size)["accuracy"]
    rounds = FedAARounds(model, OFFICIAL, 7102, initial_root, participants=4, keep=2)
    diagnostics = []
    for rnd in range(3):
        diagnostics.append(train_and_aggregate_round(core, model, bundle, clients, audits, cfg, rounds, device))
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        if rnd == 0:
            checkpoint = io.BytesIO()
            torch.save(dict(model=model.state_dict(), controller=rounds.state_dict(),
                            audits=audits, torch_rng=torch.get_rng_state(),
                            numpy_rng=np.random.get_state(), python_rng=random.getstate()), checkpoint)
    assert [d["policy_updates"] for d in diagnostics] == [1, 2, 3]
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=False)
    resumed_model = CelebACNN(cfg.seed)
    resumed_model.load_state_dict(saved["model"])
    resumed = FedAARounds(resumed_model, OFFICIAL, 7102, initial_root, participants=4, keep=2)
    resumed.load_state_dict(saved["controller"])
    torch.set_rng_state(saved["torch_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    for rnd in [1, 2]:
        got = train_and_aggregate_round(core, resumed_model, bundle, clients, saved["audits"], cfg, resumed, device)
        same(got, diagnostics[rnd])
    same(model.state_dict(), resumed_model.state_dict())
    same(rounds.state_dict(), resumed.state_dict())
    return diagnostics


if __name__ == "__main__":
    check_controller()
    cnn_diagnostics = check_cnn_path()
    report = dict(status="passed", torch=torch.__version__, device="cpu",
                  scope="synthetic parameters and fabricated RGB64; not real CelebA",
                  actual_celeba_verified=False, GPU_verified=False,
                  actor_changed=True, controller_resume_three_rounds_bitwise_equal=True,
                  full_model_weighted_aggregation_exact=True,
                  cnn_real_core_local_train_attack_root_path_rounds=3,
                  synthetic_cnn_serialized_checkpoint_resume_two_rounds_bitwise_equal=True,
                  cnn_diagnostics=cnn_diagnostics,
                  source_hashes={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [HERE / "fedaa_round_adapter.py", Path(__file__),
                                           HERE / "source_snapshot/scripts/reproduce_paper_tables.py",
                                           HERE / "source_snapshot/src/celeba_data.py"]})
    (HERE / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
