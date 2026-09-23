import copy
import sys
import types
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reproduce_paper_tables as core


class FakeLoader:
    def __init__(self, **kwargs):
        self.sensitive_column = "sex"
        self.train_df = pd.DataFrame({"x": np.arange(80, dtype=float),
                                     "sex": np.tile([0, 0, 1, 1], 20),
                                     "income": np.tile([0, 1, 0, 1], 20)})
        self.test_df = self.train_df.iloc[:8].copy()
        self.test_df["x"] += 1000


class RootNoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = subprocess.check_output(
            ["git", "show", "9e62b78:scripts/reproduce_paper_tables.py"], cwd=ROOT, text=True)
        cls.legacy = types.ModuleType("root_noise_legacy_reference")
        cls.legacy.__file__ = str(ROOT / "scripts" / "reproduce_paper_tables.py")
        sys.modules[cls.legacy.__name__] = cls.legacy
        exec(compile(source, cls.legacy.__file__, "exec"), cls.legacy.__dict__)

    def load(self, module, **config):
        with patch.object(module, "DatasetLoader", FakeLoader):
            return module.load_bundle("adult", 5.0, module.ExperimentConfig(
                num_clients=4, server_ratio=0.25, **config), torch.device("cpu"))

    def assert_untouched(self, left, right):
        self.assertEqual(left["rw_weights"], right["rw_weights"])
        for key in ("X_test", "y_test"):
            self.assertTrue(torch.equal(left[key], right[key]))
        np.testing.assert_array_equal(left["test_sensitive"], right["test_sensitive"])
        pd.testing.assert_frame_equal(left["loader"].train_df, right["loader"].train_df)
        for cid in left["clients"]:
            for key in ("X", "y"):
                self.assertTrue(torch.equal(left["clients"][cid][key], right["clients"][cid][key]))
            np.testing.assert_array_equal(left["clients"][cid]["sensitive"], right["clients"][cid]["sensitive"])

    def test_zero_noise_matches_frozen_bundle_and_global_rng(self):
        before = self.load(self.legacy)
        before_numpy = np.random.get_state()
        before_torch = torch.get_rng_state()
        after = self.load(core)
        self.assert_untouched(before, after)
        for key in ("server_X", "server_y"):
            self.assertTrue(torch.equal(before[key], after[key]))
        np.testing.assert_array_equal(before["server_sensitive"], after["server_sensitive"])
        self.assertEqual(before["server_sampling_audit"], after["server_sampling_audit"])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        current_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], current_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], current_numpy[1])
        self.assertEqual(before_numpy[2:], current_numpy[2:])
        args = ("full", "adult", "IID", "GuardFed-AD2+", "S-DFA")
        self.assertEqual(self.legacy.make_run_id(*args, self.legacy.ExperimentConfig()),
                         core.make_run_id(*args, core.ExperimentConfig()))

    def test_nested_masks_exact_counts_and_no_input_mutation(self):
        frame = FakeLoader().train_df
        pristine = frame.copy(deep=True)
        for field, column in (("root_label_noise", "income"), ("root_sensitive_noise", "sex")):
            masks = []
            for rate in (0.0, 0.1, 0.2, 0.4, 1.0):
                config = core.ExperimentConfig(**{field: rate})
                noisy, audit = core.apply_root_noise(frame, "income", "sex", config)
                again, again_audit = core.apply_root_noise(frame, "income", "sex", config)
                pd.testing.assert_frame_equal(noisy, again)
                self.assertEqual(audit, again_audit)
                mask = (noisy[column] != frame[column]).to_numpy()
                self.assertEqual(int(mask.sum()), int(rate * len(frame)))
                if masks:
                    self.assertTrue(np.all(~masks[-1] | mask))
                masks.append(mask)
                other = "sex" if column == "income" else "income"
                pd.testing.assert_series_equal(frame[other], noisy[other])
            pd.testing.assert_frame_equal(frame, pristine)

    def test_corruption_only_changes_root_not_clients_test_or_global_reweighting(self):
        clean = self.load(core)
        for noise in ({"root_label_noise": 0.4}, {"root_sensitive_noise": 0.4}):
            noisy = self.load(core, **noise)
            self.assert_untouched(clean, noisy)
            self.assertTrue(torch.equal(clean["server_X"], noisy["server_X"]))
            self.assertEqual(clean["server_sampling_audit"], noisy["server_sampling_audit"])
            audit = noisy["root_noise_audit"]
            self.assertEqual(audit["clean_group_label_counts"], clean["root_noise_audit"]["clean_group_label_counts"])
            self.assertEqual(audit["clean_root_sha256"], clean["root_noise_audit"]["clean_root_sha256"])
            if "root_label_noise" in noise:
                self.assertEqual(int((clean["server_y"] != noisy["server_y"]).sum()), 8)
                np.testing.assert_array_equal(clean["server_sensitive"], noisy["server_sensitive"])
            else:
                self.assertTrue(torch.equal(clean["server_y"], noisy["server_y"]))
                self.assertEqual(int((clean["server_sensitive"] != noisy["server_sensitive"]).sum()), 8)

    def test_noise_config_rejects_invalid_rates_and_ambiguous_synthetic_mix(self):
        for field in ("root_label_noise", "root_sensitive_noise"):
            for value in (-0.1, 1.1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    core.ExperimentConfig(**{field: value})
        with self.assertRaises(ValueError):
            core.ExperimentConfig(root_label_noise=0.1, synthetic_ratio=0.1)


if __name__ == "__main__":
    unittest.main()
