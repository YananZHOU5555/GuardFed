import contextlib
import io
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import DatasetLoader


class CompasTrainOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data/compas/compas-scores-two-years.csv")
        source = subprocess.check_output(
            ["git", "show", "9e62b78:src/data_loader.py"], cwd=ROOT, text=True
        )
        old = types.ModuleType("compas_legacy_reference")
        old.__file__ = str(ROOT / "src/data_loader.py")
        exec(compile(source, old.__file__, "exec"), old.__dict__)
        cls.old_loader = old.DatasetLoader

    def load(self, loader_class=DatasetLoader, raw=None, **kwargs):
        frame = self.raw if raw is None else raw
        with patch("pandas.read_csv", side_effect=lambda *a, **k: frame.copy(deep=True)):
            with contextlib.redirect_stdout(io.StringIO()):
                return loader_class(dataset_name="compas", **kwargs)

    def test_default_and_explicit_legacy_match_frozen_implementation(self):
        for seed in (123, 456):
            expected = self.load(self.old_loader, seed=seed)
            for options in ({}, {"preprocessing_version": "legacy"}):
                actual = self.load(seed=seed, **options)
                for name in ("train_df", "test_df", "X_train", "X_test"):
                    pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
                for name in ("y_train", "y_test"):
                    pd.testing.assert_series_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
                for name in ("sex_train", "sex_test"):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
                for name in ("mean_", "var_", "scale_"):
                    np.testing.assert_array_equal(getattr(actual.scaler, name), getattr(expected.scaler, name))
                self.assertEqual(actual.get_info(), expected.get_info())

    def test_heldout_changes_cannot_change_training_scaler_or_rows(self):
        seed = 123
        clean = self.load(seed=seed, preprocessing_version="train_only")
        legacy = self.load(seed=seed)
        np.testing.assert_array_equal(clean.X_train.index, legacy.X_train.index)
        np.testing.assert_array_equal(clean.X_test.index, legacy.X_test.index)
        changed = self.raw.copy(deep=True)
        changed.loc[clean.X_test.index, clean.numerical_columns] += 100000
        perturbed = self.load(raw=changed, seed=seed, preprocessing_version="train_only")
        for name in ("mean_", "var_", "scale_"):
            np.testing.assert_array_equal(getattr(clean.scaler, name), getattr(perturbed.scaler, name))
        pd.testing.assert_frame_equal(clean.X_train, perturbed.X_train, check_exact=True)
        self.assertFalse(clean.X_test.equals(perturbed.X_test))
        self.assertEqual(int(clean.scaler.n_samples_seen_), len(clean.X_train))
        np.testing.assert_allclose(clean.scaler.mean_, self.raw.loc[clean.X_train.index, clean.numerical_columns].mean().values)
        self.assertEqual(clean.get_info(), legacy.get_info())
        leaked = self.load(raw=changed, seed=seed)
        self.assertFalse(np.array_equal(legacy.scaler.mean_, leaked.scaler.mean_))

    def test_unknown_preprocessing_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preprocessing_version"):
            self.load(preprocessing_version="typo")


if __name__ == "__main__":
    unittest.main()
