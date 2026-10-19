"""Stateful extraction of the FLGMM author implementation, not the old lite branch.

Upstream: HantaoZhu/FLGMM a064c82a4bc460e168ce09e0654924a39d258bcc.
Derived under MIT; see sources/flgmm_license.txt for the copyright and license.
Copyright (c) 2025 tomzhu11 (upstream).
The author code selects the largest GMM component, while the paper introduction
describes the smaller-mean component. This adapter deliberately preserves code
behavior and must be labelled an author-code aggregation adaptation.
"""
import copy
import math

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


SOURCE_COMMIT = "a064c82a4bc460e168ce09e0654924a39d258bcc"
SOURCE_SHA256 = "a3f8ff07cffcf451d7aae83a3dd08949495da598fb2e688f8278d057ca6b0aff"


def largest_component(values):
    """Exactly the upstream maximum-count rule; ties use sorted label order."""
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("GMM requires at least two finite distance observations")
    if np.ptp(data) == 0:
        # Same selected observations as upstream, without its convergence warning.
        return data.copy(), (float(data[0]), float(data[0]))
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(data.reshape(-1, 1))
    labels = gmm.predict(data.reshape(-1, 1))
    labels_unique, counts = np.unique(labels, return_counts=True)
    component = data[labels == labels_unique[np.argmax(counts)]]
    return component, (float(component.min()), float(component.max()))


def equal_average(models):
    """Preserve the upstream sequential tensor addition and equal FedAvg order."""
    result = copy.deepcopy(models[0])
    for key in result:
        weighted_sum = torch.zeros_like(result[key])
        for model in models:
            weighted_sum += model[key].to(weighted_sum.dtype)
        result[key] = weighted_sum / len(models)
    return result


class FLGMMAdapter:
    """Full participation with stable identities; no server data is consumed.

    ``warmup_rounds`` retains the author's zero-based ``ccepochs`` semantics:
    rounds 0..Tg-1 use per-round GMM; round Tg fits UCL; later rounds monitor.
    ``step`` receives full local model state dictionaries after attack handling.
    It returns (aggregated model or None, diagnostics); None means keep the
    existing global model, matching the author's empty-participant behavior.
    """

    def __init__(self, client_ids, warmup_rounds=20, control_width=3.0):
        self.client_ids = tuple(client_ids)
        if len(self.client_ids) < 2 or len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("FLGMM requires at least two distinct client identities")
        if not isinstance(warmup_rounds, int) or warmup_rounds < 1:
            raise ValueError("warmup_rounds must be a positive integer")
        if not math.isfinite(control_width) or control_width <= 0:
            raise ValueError("control_width must be positive and finite")
        self.warmup_rounds = warmup_rounds
        self.control_width = float(control_width)
        self.round_index = 0
        self.history = [[] for _ in self.client_ids]
        self.ucl = None

    def state_dict(self):
        return copy.deepcopy({"version": 1, "source_commit": SOURCE_COMMIT,
                              "client_ids": list(self.client_ids),
                              "warmup_rounds": self.warmup_rounds,
                              "control_width": self.control_width,
                              "round_index": self.round_index,
                              "history": self.history, "ucl": self.ucl})

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("version", "source_commit", "client_ids", "warmup_rounds", "control_width"):
            if state.get(key) != expected[key]:
                raise ValueError("FLGMM restore identity mismatch: " + key)
        n = state.get("round_index")
        history = state.get("history")
        if not isinstance(n, int) or n < 0 or not isinstance(history, list):
            raise ValueError("Invalid FLGMM round/history")
        if len(history) != len(self.client_ids) or any(len(row) != n for row in history):
            raise ValueError("FLGMM history must cover every completed round and client")
        if not all(math.isfinite(value) for row in history for value in row):
            raise ValueError("Nonfinite FLGMM history")
        ucl = state.get("ucl")
        if (n > self.warmup_rounds) != (ucl is not None):
            raise ValueError("FLGMM threshold does not match warmup stage")
        if ucl is not None and not math.isfinite(ucl):
            raise ValueError("Invalid FLGMM threshold")
        self.round_index, self.history, self.ucl = n, copy.deepcopy(history), ucl

    def step(self, local_models, client_ids=None):
        if client_ids is not None and tuple(client_ids) != self.client_ids:
            raise ValueError("FLGMM full-participation client identity/order changed")
        if len(local_models) != len(self.client_ids) or not local_models or not local_models[0]:
            raise ValueError("Missing FLGMM local models")
        keys = tuple(local_models[0])
        for model in local_models:
            if tuple(model) != keys:
                raise ValueError("Local model state keys/order differ")
            for key in keys:
                reference, value = local_models[0][key], model[key]
                if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
                    raise ValueError("Local model tensor identity differs")
                if not value.is_floating_point() or not torch.isfinite(value).all():
                    raise ValueError("FLGMM expects finite floating model tensors")
        center = equal_average(local_models)
        distances = []
        for model in local_models:
            distance = 0
            for key in keys:
                distance += torch.pow(model[key] - center[key], 2).sum()
            distances.append(torch.sqrt(distance).item())
        component, bounds = largest_component(distances)
        mean, std = float(np.mean(component)), float(np.std(component))
        # Explicit finite extension: upstream divides by zero on identical models.
        safe_std = std if std > 0 else np.finfo(np.float64).eps
        standardized = [(value - mean) / safe_std for value in distances]
        next_history = [row + [value] for row, value in zip(self.history, standardized)]
        iters = self.round_index
        threshold = self.ucl
        if iters < self.warmup_rounds:
            selected = [i for i, value in enumerate(distances) if value in component]
            stage = "per_round_gmm"
        elif iters == self.warmup_rounds:
            all_history = np.asarray(next_history, dtype=np.float64).reshape(-1)
            first_component, first_bounds = largest_component(all_history)
            # Upstream computes a second fit but uses first_bounds in filtering.
            # Preserve this observable behavior; do not silently substitute bounds_2.
            largest_component(first_component)
            normal = all_history[all_history <= first_bounds[1]]
            threshold = float(np.mean(normal) + self.control_width * np.std(normal))
            means = np.mean(np.asarray(next_history), axis=1)
            selected = [i for i, value in enumerate(means) if not value > threshold]
            stage = "fit_control_limit"
        else:
            # Upstream excludes z >= UCL; retain strict inequality on accepted z.
            selected = [i for i, value in enumerate(standardized) if value < threshold]
            stage = "monitor"
        aggregate = equal_average([local_models[i] for i in selected]) if selected else None
        self.history, self.ucl, self.round_index = next_history, threshold, iters + 1
        return aggregate, {"round_index": iters, "stage": stage,
                           "selected_indices": selected,
                           "selected_client_ids": [self.client_ids[i] for i in selected],
                           "distances": distances, "largest_component_bounds": list(bounds),
                           "component_mean": mean, "component_std": std,
                           "zero_component_std_extension": std == 0,
                           "standardized_distances": standardized, "ucl": threshold,
                           "no_update": not selected,
                           "selection_rule": "author_code_largest_component"}
