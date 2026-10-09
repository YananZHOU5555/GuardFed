"""Independent formula/legacy regression gates; no scientific result training."""
import ast
import copy
import hashlib
import json
import math
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

from adapters import (BASELINES, _load, apply_parameter_step, cosine_fairness_hybrid,
                      empirical_gradient, fednga_step, fit_logofair,
                      huber_center, huber_thresholds, parameter_layout)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
checks = []


def must_reject(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError("Invalid input accepted")


def check_gradients():
    torch.manual_seed(7109)
    model = torch.nn.Linear(3, 2).double()
    x = torch.tensor([[1., 0., 2.], [0., 1., 3.], [2., 2., 1.],
                      [1., -1., 2.], [0., 3., 1.]], dtype=torch.float64)
    y = torch.tensor([0, 1, 1, 0, 1])
    client = {"X": x, "y": y, "n": len(y), "weights": torch.tensor([1., 2., 4., .5, 3.])}
    saved = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 17
    before_grad = [p.grad.clone() for p in model.parameters()]
    with torch.no_grad():
        errors = model(x).softmax(dim=1) - F.one_hot(y, 2)
        analytic = torch.cat([(errors.T @ x / len(y)).reshape(-1), errors.mean(0)])
    g = empirical_gradient(model, client, 2)
    torch.testing.assert_close(g, analytic, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(g, empirical_gradient(model, client, 5), atol=1e-14, rtol=1e-14)
    weights = client["weights"].double()
    with torch.no_grad():
        err = errors * weights[:, None]
        weighted = torch.cat([(err.T @ x / weights.sum()).reshape(-1), err.sum(0) / weights.sum()])
    torch.testing.assert_close(empirical_gradient(model, client, 2, use_reweighting=True),
                               weighted, atol=1e-14, rtol=1e-14)
    assert model.training and all(torch.equal(saved[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(p.grad, old) for p, old in zip(model.parameters(), before_grad))
    step = fednga_step(torch.stack([g, -g]), [1, 1], .1)
    assert torch.equal(step, torch.zeros_like(step))
    old = torch.cat([p.detach().flatten() for p in model.parameters()])
    applied = fednga_step(torch.stack([g, g * 3]), [1, 3], .2)
    apply_parameter_step(model, applied)
    torch.testing.assert_close(torch.cat([p.detach().flatten() for p in model.parameters()]), old + applied)
    must_reject(lambda: apply_parameter_step(model, applied * float("nan")))
    overflow_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in overflow_model.parameters():
            p.fill_(1e308)
    unchanged = copy.deepcopy(overflow_model.state_dict())
    must_reject(lambda: apply_parameter_step(overflow_model, torch.full_like(applied, 1e308)), FloatingPointError)
    assert all(torch.equal(unchanged[k], v) for k, v in overflow_model.state_dict().items())
    assert parameter_layout(model) == [("weight", (2, 3), 6), ("bias", (2,), 2)]
    checks.append("same-global empirical gradient equals analytical linear CE; unequal minibatch and sample weights; input/grads unchanged; signed Eq9 application")

    image = _load(ROOT / "tmp/revision-publish-20260928/src/celeba_data.py", "gate_celeba")
    cnn = image.CelebACNN(seed=7109)
    pixels = torch.randint(0, 256, (5, 3, 64, 64), dtype=torch.uint8)
    client = {"X": pixels, "y": torch.tensor([0, 1, 1, 0, 1]), "n": 5,
              "weights": torch.tensor([1., 2., 3., 4., 5.])}
    saved = copy.deepcopy(cnn.state_dict())
    for weighted in [False, True]:
        torch.testing.assert_close(empirical_gradient(cnn, client, 2, use_reweighting=weighted),
                                   empirical_gradient(cnn, client, 5, use_reweighting=weighted),
                                   rtol=2e-5, atol=1e-7)
    assert all(torch.equal(saved[k], v) for k, v in cnn.state_dict().items())
    checks.append("existing uint8 RGB64 CelebACNN full-versus-minibatch gradient equivalence and unchanged global checkpoint")


def check_fednga():
    gradients = torch.tensor([[3., 4.], [0., 2.]], dtype=torch.float64)
    torch.testing.assert_close(fednga_step(gradients, [1, 3], 2.), torch.tensor([-.3, -1.9], dtype=torch.float64))
    torch.testing.assert_close(fednga_step(gradients * torch.tensor([[4.], [7.]]), [1, 3], 2.),
                               fednga_step(gradients, [1, 3], 2.))
    torch.testing.assert_close(fednga_step(torch.tensor([[0., 0.], [0., 5.]]), [3, 1], 2.), torch.tensor([0., -.5]))
    checks.append("Fed-NGA final NeurIPS Eq9 hand calculation, per-client scaling invariance, exact zero extension with unchanged denominator")


def check_huber():
    gradients = torch.tensor([[0., 0.], [.1, -.2], [.2, .1], [.4, .3], [10., -7.]], dtype=torch.float64)
    counts = [1, 3, 2, 5, 1]
    thresholds = huber_thresholds(counts, .2, .4)
    center, info = huber_center(gradients, counts, thresholds)
    assert info["converged"] and info["stationarity_l2"] <= 1e-9
    x, n, t = gradients.numpy(), np.array(counts) / sum(counts), thresholds.numpy()

    def reference(c):
        d = np.linalg.norm(c[None] - x, axis=1)
        objective = np.sum(n * np.where(d <= t, .5 * d * d, t * d - .5 * t * t))
        factor = np.minimum(1., t / np.where(d > 0, d, t))
        derivative = np.sum((n * factor)[:, None] * (c[None] - x), axis=0)
        return objective, derivative

    solved = minimize(reference, np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-10})
    np.testing.assert_allclose(center.numpy(), solved.x, rtol=0, atol=2e-8)
    assert abs(info["objective"] - solved.fun) < 1e-12
    assert all(a >= b - 1e-12 for a, b in zip(info["objective_trace"], info["objective_trace"][1:]))
    mean, mean_info = huber_center(gradients, counts, [100.] * len(counts))
    torch.testing.assert_close(mean, (gradients * torch.tensor(counts)[:, None]).sum(0) / sum(counts))
    assert mean_info["converged"]
    same, same_info = huber_center(torch.ones(3, 2), [1, 2, 3], [.1, .2, .3])
    assert torch.equal(same, torch.ones(2)) and same_info["iterations"] == 0
    _, unfinished = huber_center(gradients, counts, thresholds, max_iter=1, tolerance=1e-16)
    assert not unfinished["converged"]
    must_reject(lambda: huber_center(gradients, [0, 3, 2, 5, 1], thresholds))
    must_reject(lambda: huber_center(gradients, counts, [0.] * len(counts)))
    checks.append("fixed-Ti sample-weighted Huber solver versus independent SciPy BFGS objective/gradient; decreasing objective; quadratic and zero-residual limits; incomplete solve rejected by convergence flag")


def check_hybrid():
    source = ROOT / "tmp/revision-publish-20260928/scripts/reproduce_paper_tables.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in {"vectorize", "cosine_between", "weighted_average", "aggregate_round"}]
    assert len(functions) == 4
    env = {"torch": torch, "F": F, "np": np, "math": math}
    exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + functions,
                            type_ignores=[]), str(source), "exec"), env)
    rng = torch.Generator().manual_seed(7109)
    for case in range(60):
        updates = [{"w": torch.randn(2, 3, generator=rng), "b": torch.randn(2, generator=rng)} for _ in range(6)]
        server = {"w": torch.randn(2, 3, generator=rng), "b": torch.randn(2, generator=rng)}
        fairness = [.01, .03, .1, .2, .4, float("nan")]
        lam, threshold = [0., 5., 20.][case % 3], [0., .2, 1.][case % 3]
        if case % 5 == 0:
            updates[0] = {k: torch.zeros_like(v) for k, v in server.items()}
        config = SimpleNamespace(guardfed_fairness_lambda=lam, trust_threshold=threshold)
        expected, diag = env["aggregate_round"]("GuardFed", updates, [1] * 6, fairness, server, config)
        actual, report = cosine_fairness_hybrid(updates, fairness, server, fairness_lambda=lam, threshold=threshold)
        assert all(torch.equal(expected[k], actual[k]) for k in expected)
        assert diag["selected_clients"] == report["selected_clients"]
        assert diag["trust_scores"] == report["trust_scores"]
    updates = [{"w": torch.tensor([1., 0.])}, {"w": torch.tensor([2., 0.])}]
    out, diag = cosine_fairness_hybrid(updates, [0., 0.], updates[0], fairness_lambda=0., threshold=1.)
    assert diag["selected_clients"] == [0] and torch.equal(out["w"], updates[0]["w"])
    out, diag = cosine_fairness_hybrid(updates, [0., .2], updates[0], fairness_lambda=1., threshold=0.)
    assert diag["selected_clients"] == [0, 1] and torch.equal(out["w"], torch.tensor([1.5, 0.]))
    must_reject(lambda: cosine_fairness_hybrid(updates, [0., 4.], updates[0]))
    checks.append("60 exact AST regressions against legacy GuardFed branch; strict threshold equality, first-argmax fallback, NaN fairness, zero vector, equal-versus-trust weighting, proportion units")


def check_logofair():
    rng = np.random.default_rng(7109)
    cid = np.repeat(np.arange(20), 80)
    group = np.tile(np.repeat([0, 1], 40), 20)
    p = rng.uniform(.1, .9, len(cid)).astype(np.float32)
    y = (rng.random(len(p)) < p).astype(int)
    settings = dict(post_rounds=3, local_steps=3, global_steps=3, calibration=False)
    first = fit_logofair(p, y, group, cid, **settings)
    replay = fit_logofair(p, y, group, cid, **settings)
    assert np.array_equal(first.predict(p, group, cid), replay.predict(p, group, cid))
    assert len({tuple(t.tolist()) for t in first.thresholds.values()}) > 1
    changed = fit_logofair(p, y, group, cid, **dict(settings, local_delta=.4))
    assert any(not torch.equal(first.thresholds[c], changed.thresholds[c]) for c in first.clients)
    global_changed = fit_logofair(p, y, group, cid, **dict(settings, global_delta=.4))
    assert first.global_lambda != global_changed.global_lambda
    must_reject(lambda: first.predict([.5], [1], [999]))
    checks.append("LoGoFair official pinned DP optimizer deterministic replay, per-client thresholds and local/global constraint activation; unknown evaluation client rejected")
    # Full default calibration remains a dependency-specific separate gate.
    return first.integration_provenance


if __name__ == "__main__":
    check_gradients()
    check_fednga()
    check_huber()
    check_hybrid()
    upstream = check_logofair()
    report = {"status": "PASS", "evidence": "CPU synthetic analytical/regression gates only; no CelebA measured scientific result",
              "python": platform.python_version(), "torch": torch.__version__, "scipy": scipy.__version__,
              "checks": checks, "logofair_provenance": upstream,
              "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [HERE / "adapters.py", HERE / "check_adapters.py"]}}
    (HERE / "gate_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
