"""Load pinned upstream FedAA DDPG; never silently substitute a heuristic.

Owns the policy/replay RNG on CPU. Caller owns federated model and transition
timing; see ADAPTER_REPORT.md. This module does not access any dataset.
"""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

OFFICIAL_COMMIT = "1fba884934cbec1b3506812d9d612e6a181b1d79"
DDPG_SHA256 = "e9bdbccdaf3aba7e1eece10370cbf7cedfe66104a691bc24a9d4e1300cd12d68"


class OfficialFedAA:
    def __init__(self, official_dir, aggre_num, seed, *, actor_lr=.01,
                 critic_lr=.01, actor_decay=1e-5, critic_decay=1e-5,
                 buffer_size=100000):
        path = Path(official_dir) / "DDPG" / "DDPG.py"
        if hashlib.sha256(path.read_bytes()).hexdigest() != DDPG_SHA256:
            raise ValueError("FedAA DDPG source differs from pinned upstream")
        if aggre_num < 1 or buffer_size < 1:
            raise ValueError("aggre_num and buffer_size must be positive")
        spec = importlib.util.spec_from_file_location("guardfed_upstream_ddpg", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.k = int(aggre_num)
        self.config = dict(aggre_num=self.k, actor_lr=actor_lr, critic_lr=critic_lr,
                           actor_decay=actor_decay, critic_decay=critic_decay,
                           buffer_size=int(buffer_size), seed=int(seed))
        self.torch_rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self.numpy_rng = np.random.RandomState(seed).get_state()
        with self._rng():
            self.agent = module.DDPG(SimpleNamespace(device="cpu", **self.config))
            self.replay = module.ExperienceReplayBuffer(self.k, self.k, buffer_size)
        # Upstream auto-chooses CUDA for replay irrespective of agent.device.
        self.replay.device = torch.device("cpu")
        self.transitions = 0

    @contextmanager
    def _rng(self):
        outer_torch, outer_numpy = torch.get_rng_state(), np.random.get_state()
        torch.set_rng_state(self.torch_rng)
        np.random.set_state(self.numpy_rng)
        try:
            yield
        finally:
            self.torch_rng, self.numpy_rng = torch.get_rng_state(), np.random.get_state()
            torch.set_rng_state(outer_torch)
            np.random.set_state(outer_numpy)

    def state_and_clients(self, vectors):
        """Upstream sum of pairwise distances, k smallest, max normalization.

        Pass full local parameter vectors in a fixed client order. Selection
        among participating clients must already be fixed by the FL protocol.
        Zero-distance degeneracy is explicitly handled instead of upstream NaN.
        """
        vectors = torch.as_tensor(vectors).detach().to(device="cpu", dtype=torch.float32)
        if vectors.ndim != 2 or len(vectors) < self.k or not torch.isfinite(vectors).all():
            raise ValueError("invalid finite client parameter matrix")
        # Preserve upstream torch.norm pairwise operation; avoid cdist's GEMM path.
        distances = torch.stack([torch.stack([torch.norm(a - b) for b in vectors])
                                 for a in vectors])
        sums = distances.sum(dim=1)
        _, indices = torch.topk(sums, k=self.k, largest=False)
        selected = sums[indices]
        maximum = selected.max()
        state = selected / maximum if maximum > 0 else torch.zeros_like(selected)
        return state.numpy().copy(), indices.tolist()

    def choose(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (self.k,) or not np.isfinite(state).all():
            raise ValueError("invalid FedAA state")
        with self._rng(), torch.no_grad():
            action = self.agent.select_action(state).reshape(-1).copy()
        if not np.isfinite(action).all() or (action < 0).any() or not np.isclose(action.sum(), 1):
            raise RuntimeError("upstream actor emitted invalid aggregation weights")
        return action

    def learn(self, state, action, next_state, root_accuracy, done=False):
        """One upstream replay update (batch 16, sampling with replacement).

        Reward must be accuracy on the train-derived clean root set, never
        official validation/test data. next_state is the NEXT local model cohort.
        """
        arrays = [np.asarray(v, dtype=np.float32).reshape(-1) for v in (state, action, next_state)]
        if any(v.shape != (self.k,) or not np.isfinite(v).all() for v in arrays):
            raise ValueError("invalid transition shape/value")
        if not np.isfinite(root_accuracy) or not 0 <= root_accuracy <= 1:
            raise ValueError("root_accuracy must lie in [0,1]")
        if (arrays[1] < 0).any() or not np.isclose(arrays[1].sum(), 1):
            raise ValueError("action must be nonnegative simplex weights")
        with self._rng():
            self.replay.add(*arrays, float(root_accuracy), bool(done))
            self.agent.update_parameters(self.replay, 16)
        self.transitions += 1

    def state_dict(self):
        state = {"config": self.config, "source_sha256": DDPG_SHA256,
                 "transitions": self.transitions, "torch_rng": self.torch_rng,
                 "numpy_rng": self.numpy_rng}
        for name in ("actor", "actor_target", "critic", "critic_target",
                     "actor_optimizer", "critic_optimizer"):
            state[name] = getattr(self.agent, name).state_dict()
        state["replay"] = {"ptr": self.replay.ptr, "size": self.replay.size}
        for name in ("state", "action", "next_state", "reward", "not_done"):
            state["replay"][name] = getattr(self.replay, name)[:self.replay.size].copy()
        return copy.deepcopy(state)

    def load_state_dict(self, state):
        if state["config"] != self.config or state["source_sha256"] != DDPG_SHA256:
            raise ValueError("FedAA checkpoint protocol/source mismatch")
        for name in ("actor", "actor_target", "critic", "critic_target",
                     "actor_optimizer", "critic_optimizer"):
            getattr(self.agent, name).load_state_dict(state[name])
        self.replay.ptr, self.replay.size = state["replay"]["ptr"], state["replay"]["size"]
        for name in ("state", "action", "next_state", "reward", "not_done"):
            getattr(self.replay, name)[:self.replay.size] = state["replay"][name]
        self.torch_rng = state["torch_rng"].clone()
        self.numpy_rng = copy.deepcopy(state["numpy_rng"])
        self.transitions = state["transitions"]
