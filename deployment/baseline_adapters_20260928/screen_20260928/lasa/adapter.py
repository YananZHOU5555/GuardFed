"""Pure LASA aggregation adapter; preserves the pinned official selection rules.

Upstream: JiiahaoXU/LASA@8477367a4e8708cde264f7572805040c650af59f.
This returns an additive model delta, without modifying any input model/update.
See AUDIT.md for explicit numerical edge policies and remaining integration gates.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _modified_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    median, std = np.median(values), np.std(values)
    # Preserve upstream NaN => false threshold comparisons => empty filter set.
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.abs((values - median) / std)
    return scores, float(median), float(std)


def _json_numbers(values) -> list[float | None]:
    return [float(v) if math.isfinite(float(v)) else None for v in values]


@torch.no_grad()
def lasa_aggregate(
    updates: list[dict[str, torch.Tensor]],
    *,
    sparsity: float = 0.3,
    lambda_n: float = 1.0,
    lambda_s: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Aggregate finite floating parameter deltas in stable dictionary order.

    sparsity is the fraction dropped, not the retained fraction. Matrices and
    Conv2d weights share a top-k threshold per client; 1D biases are not masked.
    Counts/root data/malicious-client identities are not inputs to official LASA.
    """
    if not updates or not updates[0]:
        raise ValueError("at least one nonempty client update is required")
    if not math.isfinite(sparsity) or not 0 <= sparsity < 1:
        raise ValueError("sparsity must be finite in [0, 1)")
    if any(not math.isfinite(v) or v <= 0 for v in (lambda_n, lambda_s)):
        raise ValueError("lambda_n and lambda_s must be finite and positive")
    keys = list(updates[0])
    first = updates[0][keys[0]]
    if first.dtype not in (torch.float32, torch.float64):
        raise ValueError("only float32/float64 parameter updates are supported")
    for update in updates:
        if list(update) != keys:
            raise ValueError("all clients must use the same parameter ordering")
        for key, value in update.items():
            if "num_batches_tracked" in key:
                raise ValueError("pass parameter deltas only, not integer model buffers")
            if value.dtype != first.dtype or value.device != first.device:
                raise ValueError("all parameters must share one dtype and device")
            if value.shape != updates[0][key].shape or value.numel() == 0:
                raise ValueError("parameter shapes must match and be nonempty")
            if not torch.isfinite(value).all():
                raise ValueError("non-finite client update: " + key)

    vectors = torch.stack([torch.cat([u[k].reshape(-1) for k in keys]) for u in updates])
    norms = torch.norm(vectors, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise ValueError("whole-update norm overflow")
    # Median values only, on CPU: compatible with strict CUDA determinism mode.
    clip = float(norms.cpu().median(dim=0).values.item())
    # Explicit extension: zero-norm input remains zero instead of producing 0/0.
    clipped_vectors = (vectors / torch.where(norms > 0, norms, torch.ones_like(norms))) * norms.clamp(0, clip)
    sparse = []
    mask_records = []
    masked_keys = [k for k in keys if updates[0][k].ndim in (2, 4)]
    if sparsity > 0 and not masked_keys:
        raise ValueError("positive sparsity requires a 2D or 4D weight tensor")
    for row in clipped_vectors:
        client, offset = {}, 0
        for key in keys:
            size = updates[0][key].numel()
            client[key] = row[offset:offset + size].reshape_as(updates[0][key]).clone()
            offset += size
        threshold, requested_keep = None, None
        if sparsity > 0 and masked_keys:
            scores = torch.cat([client[k].abs().reshape(-1) for k in masked_keys])
            requested_keep = int(scores.numel() * (1 - sparsity))
            if requested_keep == 0:
                raise ValueError("sparsity leaves zero top-k positions; reduce sparsity")
            threshold = float(torch.topk(scores, requested_keep, sorted=True).values[-1])
            for key in masked_keys:
                # Strict > and dtype promotion reproduce official float masks.
                client[key] = client[key] * (client[key].abs() > threshold).float()
        sparse.append(client)
        mask_records.append({
            "threshold": threshold,
            "requested_keep": requested_keep,
            "nonzero_after_by_layer": {k: int(torch.count_nonzero(client[k])) for k in keys},
        })

    result, layer_records = {}, {}
    for key in keys:
        values = torch.stack([u[key] for u in sparse])
        layer_norms = torch.norm(values.reshape(len(updates), -1).float(), dim=1).cpu().numpy()
        if not np.isfinite(layer_norms).all():
            raise ValueError("layer norm overflow: " + key)
        norm_z, norm_median, norm_std = _modified_z(layer_norms)
        norm_selected = np.flatnonzero(norm_z < lambda_n).tolist()
        signs = []
        for client in sparse:
            sign = torch.sign(client[key])
            signs.append(float(0.5 * (1 + sign.sum() / sign.abs().sum()) * (1 - sparsity)))
        sign_z, sign_median, sign_std = _modified_z(np.asarray(signs))
        # Upstream constructs a tensor before applying the sign threshold.
        sign_selected = np.flatnonzero(torch.tensor(list(sign_z)).numpy() < lambda_s).tolist()
        selected = sorted(set(norm_selected).intersection(sign_selected))
        fallback = not selected
        if fallback:
            selected = list(range(len(updates)))
        # Upstream clipped dictionaries alias the masked dictionaries: sparse output.
        result[key] = torch.mean(values[selected], dim=0)
        layer_records[key] = {
            "norms": _json_numbers(layer_norms),
            "norm_median": norm_median, "norm_std": norm_std,
            "norm_z": _json_numbers(norm_z), "norm_selected": norm_selected,
            "sign_stat": _json_numbers(signs),
            "sign_median": _json_numbers([sign_median])[0],
            "sign_std": _json_numbers([sign_std])[0],
            "sign_z": _json_numbers(sign_z), "sign_selected": sign_selected,
            "selected_clients": selected, "fallback_all": fallback,
        }
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise ValueError("non-finite LASA output")
    return result, {
        "method": "LASA-pinned-official-adapter",
        "upstream_commit": "8477367a4e8708cde264f7572805040c650af59f",
        "sparsity": sparsity, "lambda_n": lambda_n, "lambda_s": lambda_s,
        "client_norms": _json_numbers(norms.flatten().cpu().tolist()),
        "norm_clip": clip,
        "zero_norm_clients": torch.where(norms.flatten() == 0)[0].cpu().tolist(),
        "masked_keys": masked_keys, "masks": mask_records, "layers": layer_records,
        "edge_policy": "zero_norm_to_zero; degenerate_filter_rejects_all_then_official_fallback; reject_nonfinite",
    }
