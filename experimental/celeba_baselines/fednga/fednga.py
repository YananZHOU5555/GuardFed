"""Fed-NGA Eq. (9) aggregation component, not a complete training method.

Source: https://arxiv.org/html/2408.09539v1 (Eq. 9 / Algorithm 1).
Input rows must be uploaded gradient vectors, not Adam model differences.
"""

import math

import torch


@torch.no_grad()
def normalized_gradient_step(
    gradients: torch.Tensor, counts: list[int], server_eta: float
) -> torch.Tensor:
    """Return the additive parameter step -eta * sum_i alpha_i * unit(g_i).

    Rows share the same parameter ordering and global-model evaluation point.
    alpha_i = counts[i] / sum(counts). Exact zero gradients contribute zero;
    their counts remain in the denominator (explicit extension of Eq. 9).
    No merged-direction renormalization or client-norm magnitude restoration.
    Inputs are not mutated; output is a fresh vector in the input dtype/device.
    """
    if gradients.ndim != 2 or min(gradients.shape) == 0:
        raise ValueError("gradients must have nonempty shape [clients, parameters]")
    if not gradients.is_floating_point() or not torch.isfinite(gradients).all():
        raise ValueError("gradients must be finite real floating-point values")
    if len(counts) != gradients.shape[0]:
        raise ValueError("one sample count is required per client")
    if not math.isfinite(server_eta) or server_eta < 0:
        raise ValueError("server_eta must be finite and nonnegative")
    weights = torch.as_tensor(counts, device=gradients.device, dtype=torch.float64)
    if weights.ndim != 1 or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample counts must be finite and nonnegative")
    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("total sample count must be finite and positive")
    vectors = gradients.to(torch.float64)
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise ValueError("gradient norm overflow")
    unit = vectors / torch.where(norms > 0, norms, torch.ones_like(norms))
    step = -server_eta * ((weights / total)[:, None] * unit).sum(dim=0)
    step = step.to(gradients.dtype)
    if not torch.isfinite(step).all():
        raise ValueError("parameter step overflow")
    return step


def _cpu_check() -> None:
    g = torch.tensor([[3., 4.], [0., 2.]], dtype=torch.float64)
    original = g.clone()
    step = normalized_gradient_step(g, [1, 3], 2.)
    torch.testing.assert_close(step, torch.tensor([-.3, -1.9], dtype=g.dtype))
    assert torch.equal(g, original)
    # Magnitudes disappear; direction cancellation must survive aggregation.
    torch.testing.assert_close(step, normalized_gradient_step(g * torch.tensor([[7.], [0.2]]), [1, 3], 2.))
    opposed = torch.tensor([[5., 0.], [-1., 0.]])
    assert torch.equal(normalized_gradient_step(opposed, [1, 1], 3.), torch.zeros(2))
    partial_zero = torch.tensor([[0., 0.], [0., 5.]])
    torch.testing.assert_close(normalized_gradient_step(partial_zero, [3, 1], 2.), torch.tensor([0., -.5]))
    assert torch.equal(normalized_gradient_step(torch.zeros(2, 3), [1, 2], 4.), torch.zeros(3))
    # Tiny nonzero gradients still have unit direction; no epsilon clipping.
    torch.testing.assert_close(normalized_gradient_step(torch.tensor([[1e-30, 0.]]), [1], .25), torch.tensor([-.25, 0.]))
    torch.testing.assert_close(normalized_gradient_step(g, [0, 3], 2.), torch.tensor([0., -2.], dtype=g.dtype))
    for bad_g, counts, eta in [(g, [0, 0], 1.), (g, [-1, 3], 1.), (g, [1], 1.), (g, [1, 3], -1.), (g * float('nan'), [1, 3], 1.)]:
        try:
            normalized_gradient_step(bad_g, counts, eta)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was accepted")
    print("PASS: CPU analytical weighting/sign, scaling, cancellation, zero/tiny gradients, input validation, no mutation")


if __name__ == "__main__":
    _cpu_check()
