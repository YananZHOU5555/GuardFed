"""Stateful caller for pinned official FedAA; component, not a formal runner.

70-useful-update adaptation: initialize the pending (ones, actor(ones), root
accuracy) transition; complete it from the first local cohort before acting.
After the last aggregation preserve the pending transition without inventing a
next cohort. Policy is CPU; CNN remains on its original device.
"""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fedaa"))
from fedaa_official_adapter import OfficialFedAA


class FedAARounds:
    def __init__(self, model, official_dir, seed, initial_root_accuracy, *,
                 participants=20, keep=10):
        self.names = tuple(name for name, _ in model.named_parameters())
        if set(self.names) != set(model.state_dict()):
            raise ValueError("Buffer handling is not defined; current CelebACNN has none")
        if not 1 <= keep <= participants:
            raise ValueError("Invalid retention count")
        self.layout = {k: (tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}
        self.participants = participants
        self.policy = OfficialFedAA(official_dir, keep, seed)
        initial_state = np.ones(keep, dtype=np.float32)
        self.pending = [initial_state, self.policy.choose(initial_state), None]
        self.rounds = 0
        self.selected_ids = []
        self.observe_root(initial_root_accuracy)

    def aggregate(self, local_states, client_ids):
        if self.pending[2] is None:
            raise RuntimeError("Record clean-root reward before the next local cohort")
        if self.pending[2] == 1.0:
            raise RuntimeError("Upstream terminal reward reached; do not silently change stopping rule")
        if len(local_states) != self.participants or list(client_ids) != list(range(self.participants)):
            raise ValueError("Expected all participating clients in frozen ascending ID order")
        for state in local_states:
            layout = {k: (tuple(v.shape), str(v.dtype)) for k, v in state.items()}
            if layout != self.layout:
                raise ValueError("Client model parameter layout changed")
        vectors = torch.stack([torch.cat([s[k].detach().cpu().reshape(-1) for k in self.names])
                               for s in local_states])
        state, selected = self.policy.state_and_clients(vectors)
        self.policy.learn(*self.pending[:2], state, self.pending[2])
        action = self.policy.choose(state)
        # Upstream alpha=0: average FULL models with actor weights (no renormalization).
        result = {k: torch.zeros_like(local_states[0][k]) for k in self.names}
        for weight, index in zip(action, selected):
            for name in self.names:
                result[name] += local_states[index][name] * float(weight)
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite full-model aggregation")
        self.rounds += 1
        self.selected_ids = [client_ids[i] for i in selected]
        self.pending = [state.copy(), action.copy(), None]
        return result, {"implementation": "official_DDPG_useful_round_adaptation_v1",
                        "round": self.rounds, "selected_client_ids": self.selected_ids,
                        "state": state.tolist(), "action": action.tolist(),
                        "policy_updates": self.policy.transitions,
                        "policy_device": "cpu", "reward_split": "train_clean_root"}

    def observe_root(self, accuracy):
        if self.pending[2] is not None:
            raise RuntimeError("Root reward already recorded")
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Root accuracy must be finite in [0,1]")
        self.pending[2] = float(accuracy)

    def state_dict(self):
        return copy.deepcopy(dict(schema=1, participants=self.participants, names=self.names,
                                  layout=self.layout, rounds=self.rounds,
                                  selected_ids=self.selected_ids, pending=self.pending,
                                  policy=self.policy.state_dict()))

    def load_state_dict(self, saved):
        if (saved["schema"], saved["participants"], saved["names"], saved["layout"]) != (
                1, self.participants, self.names, self.layout):
            raise ValueError("FedAA model/protocol checkpoint mismatch")
        if saved["rounds"] != saved["policy"]["transitions"]:
            raise ValueError("Round and policy update count mismatch")
        self.policy.load_state_dict(saved["policy"])
        self.rounds, self.selected_ids, self.pending = copy.deepcopy(
            (saved["rounds"], saved["selected_ids"], saved["pending"]))


def train_and_aggregate_round(core, model, bundle, clients, audits, config, controller, device):
    """Use existing CNN local training, attacks and root evaluation without editing core.

    Caller must provide a train-derived clean root (never official valid/test).
    Save model + controller + external RNG at this function's completed boundary.
    This does not create jobs, choose configs, evaluate validation, or write results.
    """
    if config.root_label_noise or config.root_sensitive_noise or bundle.get("root_synthetic_rows", 0):
        raise ValueError("FedAA reward requires the clean train-derived root")
    global_state = core.clone_state(model)
    server_update = core.train_server_update(model, bundle, config)
    local_states = []
    # Preserve the current core's local-training / attack / root-evaluation order.
    for client in clients:
        local = core.train_local_model(model, client, config)
        update, audits[client["cid"]] = core.apply_foe_if_needed(
            local, global_state, client, audits[client["cid"]], server_update, config)
        received = {k: global_state[k] + update[k] for k in global_state}
        core.evaluate_state_on_server(received, bundle["num_features"], bundle, config, device)
        local_states.append(received)
    state, diagnostics = controller.aggregate(local_states, [c["cid"] for c in clients])
    model.load_state_dict(state)
    metrics = core.evaluate_model(model, bundle["server_X"], bundle["server_y"],
                                  bundle["server_sensitive"], config.batch_size)
    controller.observe_root(metrics["accuracy"])
    diagnostics["root_accuracy"] = metrics["accuracy"]
    return diagnostics
