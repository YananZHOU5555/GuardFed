"""Explicit group-A adapters; no old baseline labels or results are overwritten.

Fed-NGA: NeurIPS 2025 Eq. 9, actual same-point empirical gradients.
Huber: AAAI 2024 Eq. 5--6, sample-weighted fixed-threshold vector objective.
CosineFairnessHybrid: project's legacy GuardFed branch, equal selected mean.
LoGoFair: execute the separately pinned official DP optimizer via its adapter.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASELINES = Path(__file__).resolve().parents[2]


def _load(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_layout(model):
    """Stable, trainable parameter order; buffers never become gradients."""
    return [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters()
            if p.requires_grad]


def empirical_gradient(model, client, batch_size, *, use_reweighting=False):
    """Full local gradient at the unchanged global model, with no optimizer step.

    Unweighted objective: mean per-example CE. Optional reweighting explicitly
    changes the objective to sum(weight * CE) / sum(weight), once per client;
    it is never an unweighted average of unequal minibatch means. Current CelebA
    CNN has neither BN nor dropout. Other model families must freeze their mode.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,
                          torch.nn.modules.dropout._DropoutNd)) for m in model.modules()):
        raise ValueError("Freeze a BN/dropout policy before using this gradient path")
    x, y = client["X"], client["y"]
    n = len(y)
    if n < 1 or len(x) != n or client["n"] != n:
        raise ValueError("Nonempty complete client samples and exact count required")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable model parameters")
    device = parameters[0].device
    if any(p.device != device or not p.is_floating_point() for p in parameters):
        raise ValueError("Parameters must share a device and be floating point")
    weight = client["weights"] if use_reweighting else torch.ones(n)
    if len(weight) != n or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Finite nonnegative per-example weights required")
    denominator = float(weight.double().sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Positive finite total sample weight required")
    total = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    try:
        for first in range(0, n, batch_size):
            last = min(n, first + batch_size)
            losses = F.cross_entropy(model(x[first:last]), y[first:last].to(device), reduction="none")
            loss_sum = (losses * weight[first:last].to(device, dtype=losses.dtype)).sum()
            values = torch.autograd.grad(loss_sum, parameters, allow_unused=False)
            total += torch.cat([value.detach().reshape(-1).double() for value in values])
    finally:
        model.train(was_training)
    total = (total / denominator).to(parameters[0].dtype)
    if not torch.isfinite(total).all():
        raise FloatingPointError("Nonfinite empirical client gradient")
    return total


@torch.no_grad()
def apply_parameter_step(model, step):
    """Apply an already signed vector step; validate before any mutation."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    if step.ndim != 1 or step.numel() != sum(p.numel() for p in parameters):
        raise ValueError("Step does not match trainable parameter layout")
    if not torch.isfinite(step).all():
        raise ValueError("Nonfinite parameter step")
    next_values, offset = [], 0
    for p in parameters:
        candidate = p + step[offset:offset + p.numel()].reshape_as(p).to(p)
        if not torch.isfinite(candidate).all():
            raise FloatingPointError("Parameter step would overflow; model unchanged")
        next_values.append(candidate)
        offset += p.numel()
    for p, value in zip(parameters, next_values):
        p.copy_(value)


def fednga_step(gradients, counts, server_eta):
    # Reuse the audited Eq. 9 component, without magnitude restoration.
    source = _load(BASELINES / "fednga/fednga.py", "guardfed_pinned_fednga")
    return source.normalized_gradient_step(gradients, counts, server_eta)


@torch.no_grad()
def huber_center(gradients, counts, thresholds, *, max_iter=1000, tolerance=1e-9):
    """Solve sum_i n_i Huber_Ti(||center-gradient_i||), keeping Ti fixed.

    A bounded failure returns converged=False; callers must preserve failure and
    must not use the center as an accepted completed scientific result. Double
    precision applies only to this deterministic aggregation solver.
    """
    if gradients.ndim != 2 or min(gradients.shape) == 0 or not gradients.is_floating_point():
        raise ValueError("Floating gradients [clients, parameters] required")
    if not torch.isfinite(gradients).all():
        raise ValueError("Nonfinite client gradients")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    x = gradients.double()
    n = torch.as_tensor(counts, dtype=torch.float64, device=x.device)
    t = torch.as_tensor(thresholds, dtype=torch.float64, device=x.device)
    if n.shape != (len(x),) or t.shape != (len(x),):
        raise ValueError("One count and fixed threshold per client required")
    if not torch.isfinite(n).all() or (n <= 0).any() or not torch.isfinite(n.sum()):
        raise ValueError("Positive finite sample counts required")
    if not torch.isfinite(t).all() or (t <= 0).any():
        raise ValueError("Positive finite Huber thresholds required")
    n = n / n.sum()  # Positive common scaling does not change the minimizer.

    def quantities(center):
        residual = center[None, :] - x
        distance = torch.linalg.vector_norm(residual, dim=1)
        factor = torch.minimum(torch.ones_like(t), t / torch.where(distance > 0, distance, t))
        weights = n * factor
        objective = (n * torch.where(distance <= t, .5 * distance.square(),
                                     t * distance - .5 * t.square())).sum()
        stationarity = torch.linalg.vector_norm((weights[:, None] * residual).sum(dim=0))
        return weights, objective, stationarity

    center = (n[:, None] * x).sum(dim=0)
    trace, converged = [], False
    weights, objective, stationarity = quantities(center)
    trace.append(float(objective))
    for iteration in range(max_iter):
        if not torch.isfinite(objective) or not torch.isfinite(stationarity):
            raise FloatingPointError("Huber objective or stationarity overflow")
        if float(stationarity) <= tolerance:
            converged = True
            break
        center = (weights[:, None] * x).sum(dim=0) / weights.sum()
        weights, objective, stationarity = quantities(center)
        if float(objective) > trace[-1] + 1e-10 * max(1., abs(trace[-1])):
            raise FloatingPointError("Fixed Huber objective increased")
        trace.append(float(objective))
    if float(stationarity) <= tolerance:
        converged = True
    info = {"converged": converged, "iterations": len(trace) - 1,
            "stationarity_l2": float(stationarity), "objective": float(objective),
            "objective_normalization": "divide by total sample count",
            "objective_trace": trace, "fixed_thresholds": t.cpu().tolist(),
            "normalized_final_weights": (weights / weights.sum()).cpu().tolist(),
            "method_core": "sample-weighted fixed-Ti vector Huber gradient objective"}
    return center.to(gradients.dtype), info


def huber_thresholds(counts, t0, m):
    """AAAI threshold family Ti=T0+M/sqrt(ni); choose T0/M before the solve."""
    n = torch.as_tensor(counts, dtype=torch.float64)
    if n.ndim != 1 or len(n) == 0 or not torch.isfinite(n).all() or (n <= 0).any():
        raise ValueError("Positive finite client counts required")
    if any(not math.isfinite(v) or v < 0 for v in (t0, m)) or (t0 == 0 and m == 0):
        raise ValueError("T0/M must be finite nonnegative and not both zero")
    return t0 + m / n.sqrt()


@torch.no_grad()
def cosine_fairness_hybrid(updates, fairness, server_update, *, fairness_lambda=20., threshold=.2):
    """Legacy custom control; ReLU cosine * exp(-lambda*AEOD), equal mean.

    AEOD input is a proportion in [0,1], measured on each attacked local model
    against the clean root. Undefined fairness is conservatively assigned 1,
    as in legacy GuardFed. Undefined zero-vector cosine is assigned 0.
    """
    if not updates or len(updates) != len(fairness):
        raise ValueError("One root fairness value per nonempty update required")
    if not math.isfinite(fairness_lambda) or fairness_lambda < 0 or not math.isfinite(threshold):
        raise ValueError("Invalid hybrid parameters")
    keys = list(server_update)
    reference = torch.cat([server_update[k].reshape(-1) for k in keys])
    if not reference.is_floating_point() or not torch.isfinite(reference).all():
        raise ValueError("Finite floating server update required")
    trusts = []
    for update, fair in zip(updates, fairness):
        if set(update) != set(keys) or any(update[k].shape != server_update[k].shape for k in keys):
            raise ValueError("Client/server update layout mismatch")
        vector = torch.cat([update[k].reshape(-1) for k in keys])
        if not torch.isfinite(vector).all():
            raise ValueError("Nonfinite client update")
        fair = float(fair)
        if math.isnan(fair):
            fair = 1.
        if not math.isfinite(fair) or not 0 <= fair <= 1:
            raise ValueError("AEOD must use proportion units in [0,1]")
        cosine = float(F.cosine_similarity(reference[None], vector[None]).item())
        trusts.append(max(0., cosine) * math.exp(-fairness_lambda * fair))
    selected = [i for i, trust in enumerate(trusts) if trust > threshold]
    if not selected:
        selected = [int(np.argmax(trusts))]
    # Same sequential float arithmetic and equal weights as the old branch.
    out = {k: torch.zeros_like(server_update[k]) for k in keys}
    for i in selected:
        for k in keys:
            out[k] = out[k] + updates[i][k] * (1. / len(selected))
    return out, {"selected_clients": selected, "trust_scores": trusts,
                 "aggregation_weighting": "equal_after_threshold",
                 "method_core": "CosineFairnessHybrid (legacy GuardFed control)"}


def fit_logofair(calibration_probability, calibration_y, calibration_sensitive,
                calibration_client_id, *, source_dir=None, **settings):
    """Call genuine local/global optimizer; never invent evaluation client IDs.

    The caller must supply a frozen stable client mapping for calibration AND
    evaluation, using clean train-root data for calibration, not ranking labels.
    Missing netcal is a dependency failure, never a silent calibration=False.
    """
    official_dir = Path(source_dir) if source_dir else BASELINES / "logofair"
    adapter = _load(official_dir / "adapter.py", "guardfed_official_logofair_dp")
    post = adapter.OfficialLoGoFairDP(
        source=official_dir / "upstream/fedlearn/models/FedFairPostClient.py", **settings)
    post.fit(calibration_probability, calibration_y, calibration_sensitive, calibration_client_id)
    post.integration_provenance = {
        "upstream_commit": adapter.COMMIT,
        "upstream_client_lf_sha256": adapter.SOURCE_SHA256,
        "adapter_sha256": hashlib.sha256((official_dir / "adapter.py").read_bytes()).hexdigest(),
        "calibration_source": "caller must record clean train-root identities",
        "group_priors": "calibration-only corrected denominator",
        "metric": "DP only; EO upstream issues unresolved",
    }
    return post
