"""Component gates against AST-extracted author code; does not train CelebA."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from flgmm_adapter import FLGMMAdapter, SOURCE_SHA256, largest_component


BASE = Path(__file__).resolve().parent


class NoPlot:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def author_oracle(warmup_rounds, control_width, nclients):
    source = BASE / "sources/flgmm_pinned.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
    tree = ast.parse(source.read_text(encoding="utf8"))
    required = {"decompose_normal_distributions", "euclidean_distance", "plot_control_chart"}
    functions = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in required]
    fedavg = ast.parse((BASE / "sources/flgmm_fedavg.py").read_text(encoding="utf8"))
    functions += [x for x in fedavg.body if isinstance(x, ast.FunctionDef) and x.name == "FedAvg_0"]
    branches = [x for x in ast.walk(tree) if isinstance(x, ast.If)
                and isinstance(x.test, ast.Compare)
                and ast.unparse(x.test) == "args.method == 'flgmm'"]
    assert len(branches) == 1
    env = {"np": np, "torch": torch, "GaussianMixture": GaussianMixture,
           "copy": copy, "os": os, "plt": NoPlot(), "sns": NoPlot(),
           "args": SimpleNamespace(method="flgmm", num_users=nclients,
                                   ccepochs=warmup_rounds, L=control_width),
           "save_dir": ".", "print": lambda *args: None,
           "calculate_accuracy": lambda *args: (0.5, 0.5),
           "test_img": lambda *args: (0.5, 1.0), "dataset_test": None,
           "distances_matrix": [[] for _ in range(nclients)],
           "normal_dis": [[] for _ in range(nclients)],
           "excluded_clients": [], "noisy_clients": [], "noisy_this_round": [],
           "r": [], "p": [], "f1": [], "o": 1,
           "global_acctotal": [], "global_losses": []}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(source), "exec"), env)
    body = compile(ast.fix_missing_locations(ast.Module(body=branches[0].body,
                                                       type_ignores=[])), str(source), "exec")

    def step(models, iters):
        captured = []
        env.update({"w_locals": models, "iters": iters, "normal_id": [],
                    "excluded": [], "distances_matrix_this_round": [],
                    "loss_locals": [], "net_glob": SimpleNamespace(load_state_dict=captured.append)})
        exec(body, env)
        selected = ([i for i in range(nclients) if i in env["normal_id"]]
                    if iters < warmup_rounds else
                    [i for i in range(nclients) if i not in env["excluded_clients"]])
        return (captured[0] if captured else None, selected, env["distances_matrix"],
                env.get("UCL"))

    return step


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(134)
    ids = tuple(range(20))
    adapter = FLGMMAdapter(ids, warmup_rounds=2, control_width=3)
    reference = author_oracle(2, 3, len(ids))
    gates = []
    selections = []
    for round_index in range(7):
        models = []
        for client in ids:
            values = rng.normal(0, 0.10, size=5)
            if client >= 16:
                values += 3 + round_index * 0.2
            # A previously benign client becomes Byzantine after the UCL exists.
            if round_index >= 4 and client == 0:
                values += 6
            models.append({"weight": torch.tensor(values[:4].reshape(2, 2), dtype=torch.float32),
                           "bias": torch.tensor(values[4:], dtype=torch.float32)})
        before = copy.deepcopy(models)
        actual, diagnostic = adapter.step(models, ids)
        expected, selected, history, ucl = reference(models, round_index)
        assert diagnostic["selected_indices"] == selected
        assert adapter.history == history and adapter.ucl == ucl
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert all(torch.equal(actual[key], expected[key]) for key in actual)
        assert all(torch.equal(x[key], y[key]) for x, y in zip(models, before) for key in x)
        selections.append(selected)
        gates.append({"gate": "upstream_round_" + str(round_index), "passed": True,
                      "stage": diagnostic["stage"], "selected": selected,
                      "bitwise_aggregate": True, "exact_history_and_ucl": True})
        if round_index in (0, 2, 3):
            restored = FLGMMAdapter(ids, 2, 3)
            restored.load_state_dict(json.loads(json.dumps(adapter.state_dict())))
            adapter = restored
    assert 0 in selections[3] and 0 not in selections[4]
    gates.append({"gate": "newly_malicious_client_affects_actual_aggregate", "passed": True})
    gates.append({"gate": "json_state_restore_crosses_warmup", "passed": True})

    # Independent behavioral distinction: largest cluster can have the larger mean.
    large_cluster = np.array([4.0, 4.1, 4.2, 4.3, 4.4, 0.01, 0.02])
    chosen, _ = largest_component(large_cluster)
    assert len(chosen) == 5 and np.mean(chosen) > 4
    gates.append({"gate": "upstream_largest_is_not_paper_smallest_mean", "passed": True})

    # Degenerate case is an explicitly disclosed extension, not upstream equivalence.
    equal = FLGMMAdapter(ids, 1, 3)
    models = [{"weight": torch.zeros(3)} for _ in ids]
    for i in range(3):
        result, diagnostic = equal.step(models)
        assert diagnostic["zero_component_std_extension"]
        assert all(np.isfinite(row).all() for row in equal.history)
        assert result is None or torch.isfinite(result["weight"]).all()
    assert result is None  # z == UCL == 0 is excluded by author's strict boundary.
    gates.append({"gate": "identical_models_finite_extension_and_empty_skip", "passed": True})

    frozen = copy.deepcopy(adapter.state_dict())
    bad = copy.deepcopy(models)
    bad[0]["weight"][0] = float("nan")
    before_rejection = copy.deepcopy(equal.state_dict())
    for function in (lambda: equal.step(bad), lambda: equal.step(models, tuple(reversed(ids))),
                     lambda: FLGMMAdapter(ids, 2, 2).load_state_dict(frozen)):
        try:
            function()
            raise AssertionError("Invalid input/restore was accepted")
        except ValueError:
            pass
    assert equal.state_dict() == before_rejection
    gates.append({"gate": "nonfinite_identity_and_recipe_rejected", "passed": True})
    output = {"status": "PASS", "kind": "component_only_not_real_image_training",
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "source_sha256": SOURCE_SHA256,
              "adapter_sha256": hashlib.sha256((BASE / "flgmm_adapter.py").read_bytes()).hexdigest(),
              "gates": gates, "limitations": ["No real CelebA or GPU gate in this task",
              "Author code largest-component semantics differ from paper introduction",
              "Degenerate zero-std extension is not an exact upstream execution"]}
    (BASE / "component_acceptance.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
