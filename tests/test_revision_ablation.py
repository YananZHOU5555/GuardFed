import importlib.util
import subprocess
import sys
import types
import unittest
import tempfile
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reproduce_paper_tables as core


def legacy_module():
    source = subprocess.check_output(
        ["git", "show", "ec419b7:scripts/reproduce_paper_tables.py"], cwd=ROOT, text=True
    )
    module = types.ModuleType("revision_legacy_reference")
    module.__file__ = str(ROOT / "scripts" / "reproduce_paper_tables.py")
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class RevisionAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.legacy = legacy_module()

    def setUp(self):
        self.updates = [{"w": torch.tensor(v, dtype=torch.float64)} for v in [
            [1.0, 0.2, 0.1], [0.2, 0.8, 0.5], [-0.4, 1.1, 0.3],
            [1.4, -0.2, 0.8], [0.1, 0.3, 1.7], [2.2, 0.4, -0.3],
        ]]
        self.root = {"w": torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64)}
        self.details = [{"accuracy": a, "aeod": f, "aspd": f * 0.6} for a, f in
                        zip([0.6, 0.7, 0.82, 0.63, 0.77, 0.52], [0.01, 0.08, 0.14, 0.21, 0.3, 0.4])]
        self.fairness = [d["aeod"] for d in self.details]

    def call(self, module, config, adaptive=False):
        args = (self.updates, self.fairness, self.root, config)
        if not adaptive:
            return module.guardfed_act_aggregate(*args, fairness_details=self.details)
        def root_metrics(state, *_):
            return {"accuracy": 0.7 + float(torch.sigmoid(state["w"].sum())) * 0.1,
                    "aeod": 0.1, "aspd": 0.2}
        with patch.object(module, "evaluate_state_on_server", side_effect=root_metrics):
            return module.guardfed_ad2plus_adaptive_aggregate(
                *args, fairness_details=self.details,
                global_state={"w": torch.zeros(3, dtype=torch.float64)},
                bundle={"num_features": 3}, device=torch.device("cpu"))

    def test_none_is_bitwise_legacy_for_fixed_and_adaptive(self):
        for adaptive in (False, True):
            for norm_mode in ("adaptive", "root"):
                opts = dict(num_malicious=2, ad2_norm_mode=norm_mode)
                expected, before = self.call(self.legacy, self.legacy.ExperimentConfig(**opts), adaptive)
                actual, after = self.call(core, core.ExperimentConfig(**opts), adaptive)
                self.assertTrue(torch.equal(expected["w"], actual["w"]))
                for key in ("trust_scores", "selected_clients", "norm_clip_scales"):
                    self.assertEqual(before[key], after[key])
                self.assertAlmostEqual(sum(after["client_weights"]), 1.0)
                if adaptive:
                    self.assertEqual(before["ad2_plus_selected_candidate"], after["ad2_plus_selected_candidate"])
                    self.assertEqual(before["ad2_plus_candidates"], after["ad2_plus_candidates"])

    def test_every_adaptive_candidate_respects_each_mask(self):
        original = core.guardfed_act_aggregate
        for component in ("U", "C", "A", "F", "V", "N"):
            seen = []
            def checked(*args, **kwargs):
                update, info = original(*args, **kwargs)
                self.assertEqual(info["ablation_component"], component)
                if component == "N":
                    self.assertEqual(info["norm_clip_scales"], [1.0] * len(info["selected_clients"]))
                    manual = core.weighted_average(self.updates, info["client_weights"])
                    torch.testing.assert_close(update["w"], manual["w"], rtol=1e-14, atol=1e-14)
                else:
                    self.assertTrue(all(t[component] == 0 for t in info["component_contributions"]))
                    full_config = core.clone_config_with(args[3], {"ablation_component": "none"})
                    _, full_info = original(*args[:3], full_config, **kwargs)
                    self.assertTrue(any(t[component] != 0 for t in full_info["component_contributions"]))
                    for actual, full in zip(info["component_contributions"], full_info["component_contributions"]):
                        for other in set("UCAFV") - {component}:
                            self.assertEqual(actual[other], full[other])
                seen.append(info)
                return update, info
            with patch.object(core, "guardfed_act_aggregate", side_effect=checked):
                self.call(core, core.ExperimentConfig(num_malicious=2, ablation_component=component), adaptive=True)
            self.assertEqual(len(seen), 10)

    def test_diagnostics_alpha_callback_checkpoint_preserve_trajectory_and_rng(self):
        x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        sensitive = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        bundle = {
            "num_features": 3, "feature_includes_label": False,
            "feature_includes_sensitive": False, "label_col": "income",
            "sensitive_col": "sex", "train_rows": 40, "test_rows": 8,
            "server_X": x, "server_y": y, "server_sensitive": sensitive,
            "X_test": x, "y_test": y, "test_sensitive": sensitive,
            "rw_weights": {(s, label): 1.0 for s in (0, 1) for label in (0, 1)},
            "clients": {i: {"X": x + i * 0.01, "y": y,
                            "sensitive": sensitive, "sensitive_feature_index": None}
                        for i in range(4)},
        }
        options = dict(num_clients=4, num_malicious=1, rounds=3, device="cpu",
                       ad2_calibration_quantiles=7, batch_size=8)
        args = ("adult", "IID", "GuardFed-AD2+", "Benign")
        with patch.object(self.legacy, "load_bundle", return_value=copy.deepcopy(bundle)):
            expected = self.legacy.run_experiment(
                *args, self.legacy.ExperimentConfig(**options), "test", torch.device("cpu"))
        expected_rng = torch.get_rng_state().clone()
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            with patch.object(core, "load_bundle", return_value=copy.deepcopy(bundle)) as loader:
                actual = core.run_experiment(
                    *args, core.ExperimentConfig(**options, client_alpha=0.5, full_round_diagnostics=True),
                    "test", torch.device("cpu"), progress_callback=observed.append,
                    checkpoint_path=checkpoint)
            self.assertEqual(loader.call_args.args[1], 0.5)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertTrue(checkpoint.is_file())
            state = torch.load(checkpoint, weights_only=True)
            self.assertIn("linear1.weight", state)
        self.assertEqual(expected["trajectory_metrics"], actual["trajectory_metrics"])
        self.assertEqual(expected["metrics"], actual["metrics"])
        self.assertEqual(len(actual["round_summaries"]), 3)
        self.assertEqual(observed, actual["trajectory_metrics"])
        for summary in actual["round_summaries"]:
            self.assertEqual(summary["client_ids"], [0, 1, 2, 3])
            self.assertEqual(summary["malicious_mask"], [True, False, False, False])
            self.assertAlmostEqual(sum(summary["aggregate"]["client_weights"]), 1.0)

    def test_run_ids_and_config_validation(self):
        args = ("full", "adult", "IID", "GuardFed-AD2+", "S-DFA")
        self.assertEqual(core.make_run_id(*args, core.ExperimentConfig()),
                         self.legacy.make_run_id(*args, self.legacy.ExperimentConfig()))
        ids = {core.make_run_id(*args, core.ExperimentConfig(ablation_component=c))
               for c in ("none", "U", "C", "A", "F", "V", "N")}
        self.assertEqual(len(ids), 7)
        for alpha in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                core.ExperimentConfig(client_alpha=alpha)
        with self.assertRaises(ValueError):
            core.ExperimentConfig(ablation_component="typo")


if __name__ == "__main__":
    unittest.main()
