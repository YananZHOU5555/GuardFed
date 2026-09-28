"""CPU upstream parity, learned parameters, RNG isolation and resume check."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fedaa_official_adapter import OfficialFedAA

torch.set_num_threads(1)
root = Path(__file__).parent / "official"
seed = 41001
wrapper = OfficialFedAA(root, 3, seed, buffer_size=32)
spec = importlib.util.spec_from_file_location("upstream_direct", root / "DDPG" / "DDPG.py")
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
torch.manual_seed(seed)
np.random.seed(seed)
direct = upstream.DDPG(SimpleNamespace(device="cpu", **wrapper.config))
replay = upstream.ExperienceReplayBuffer(3, 3, 32)
replay.device = torch.device("cpu")
initial = copy.deepcopy(wrapper.agent.actor.state_dict())
state = np.array([.3, .6, 1.], np.float32)
next_state = np.array([.4, .7, 1.], np.float32)
action = direct.select_action(state).reshape(-1)
replay.add(state, action, next_state, .8, False)
direct.update_parameters(replay, 16)
outside_torch = torch.get_rng_state().clone()
outside_numpy = np.random.get_state()
wrapped_action = wrapper.choose(state)
assert np.array_equal(action, wrapped_action)
wrapper.learn(state, wrapped_action, next_state, .8)
assert torch.equal(outside_torch, torch.get_rng_state())
assert np.array_equal(outside_numpy[1], np.random.get_state()[1])
for name in ("actor", "critic", "actor_target", "critic_target"):
    for key, tensor in getattr(direct, name).state_dict().items():
        assert torch.equal(tensor, getattr(wrapper.agent, name).state_dict()[key]), (name, key)
assert any(not torch.equal(initial[k], v) for k, v in wrapper.agent.actor.state_dict().items())

saved = wrapper.state_dict()
resumed = OfficialFedAA(root, 3, seed, buffer_size=32)
resumed.load_state_dict(saved)
for i in range(4):
    left, right = wrapper.choose(next_state), resumed.choose(next_state)
    assert np.array_equal(left, right)
    wrapper.learn(next_state, left, state, .8 + .01 * i)
    resumed.learn(next_state, right, state, .8 + .01 * i)
for key, value in wrapper.agent.actor.state_dict().items():
    assert torch.equal(value, resumed.agent.actor.state_dict()[key]), key

vectors = torch.tensor([[0., 1.], [1., 2.], [1., 3.], [15., 13.]])
selected_state, selected_ids = wrapper.state_and_clients(vectors)
assert set(selected_ids) == {0, 1, 2} and np.isfinite(selected_state).all()
zero_state, _ = wrapper.state_and_clients(torch.zeros(4, 2))
assert np.array_equal(zero_state, np.zeros(3))
report = {"status": "passed", "torch": torch.__version__, "device": "cpu",
          "upstream_action_and_one_update_bitwise_equal": True,
          "actor_actually_updated": True, "global_rng_unchanged": True,
          "resume_four_updates_bitwise_equal": True,
          "state_selection_outlier_and_zero_distance": True,
          "end_to_end_celeba_verified": False}
(Path(__file__).parent / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
