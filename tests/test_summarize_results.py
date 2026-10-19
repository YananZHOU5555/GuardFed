import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_results import summarize_records


def record(seed, accuracy=0.6, aeod=0.1, aspd=0.2, learning_rate=0.001):
    metrics = {"accuracy": accuracy, "aeod": aeod, "aspd": aspd}
    return {
        "dataset": "adult", "distribution": "IID", "method": "GuardFed-AD2+",
        "attack": "Benign", "seed": seed, "rounds": 2,
        "config": {"seed": seed, "rounds": 2, "learning_rate": learning_rate,
                   "experiment_tag": f"seed{seed}", "experiment_suite": f"suite{seed}"},
        "metrics": dict(metrics),
        "trajectory_metrics": [{"round": 1, "metrics": dict(metrics)},
                               {"round": 2, "metrics": dict(metrics)}],
        "last10_metrics": [{"round": 1, "metrics": {"accuracy": 1.0, "aeod": 0.0, "aspd": 0.0}}],
    }


class TerminalSummaryTest(unittest.TestCase):
    manifest = {"seeds": [3, 1, 2], "rounds": 2}

    def summarize(self, rows):
        return summarize_records(rows, self.manifest)

    def test_terminal_metrics_and_hand_calculated_sample_sd(self):
        rows = [record(1, 0.2), record(2, 0.4), record(3, 0.6)]
        result = self.summarize(rows)
        accuracy = next(row for row in result if row["metric"] == "accuracy")
        self.assertEqual((accuracy["n_total"], accuracy["n_defined"]), (3, 3))
        self.assertAlmostEqual(accuracy["mean"], 0.4)
        self.assertAlmostEqual(accuracy["sample_std"], 0.2)
        changed = copy.deepcopy(rows)
        for row in changed:
            row["last10_metrics"] = "This field must not be read"
            row["trajectory_metrics"][0]["metrics"] = {"accuracy": 1.0, "aeod": 0.0, "aspd": 0.0}
        self.assertEqual(result, self.summarize(changed))

    def test_missing_extra_and_duplicate_seeds_are_rejected(self):
        for rows in ([record(1), record(2)],
                     [record(1), record(2), record(3), record(4)],
                     [record(1), record(2), record(3), record(1)]):
            with self.subTest(seeds=[row["seed"] for row in rows]):
                with self.assertRaises(ValueError):
                    self.summarize(rows)

    def test_configurations_are_separate_and_each_requires_all_seeds(self):
        rows = [record(seed, learning_rate=rate) for rate in (0.001, 0.002) for seed in (1, 2, 3)]
        result = self.summarize(rows)
        self.assertEqual(len(result), 6)
        self.assertEqual(len({row["config_sha256"] for row in result}), 2)
        self.assertEqual({row["n_total"] for row in result}, {3})
        with self.assertRaises(ValueError):
            self.summarize([record(1), record(2), record(3, learning_rate=0.002)])

    def test_invalid_rounds_and_trajectory_are_rejected(self):
        mutations = (
            lambda row: row.update(rounds=1),
            lambda row: row["config"].update(rounds=1),
            lambda row: row["trajectory_metrics"].pop(),
            lambda row: row["trajectory_metrics"][0].update(round=2),
            lambda row: row["trajectory_metrics"][1].update(round=True),
        )
        for mutate in mutations:
            rows = [record(1), record(2), record(3)]
            mutate(rows[0])
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValueError):
                    self.summarize(rows)

    def test_terminal_mismatch_and_invalid_metrics_are_rejected(self):
        for value in (0.6 + 1e-9, None, True, "0.6"):
            rows = [record(1), record(2), record(3)]
            rows[0]["trajectory_metrics"][-1]["metrics"]["accuracy"] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.summarize(rows)
        rows = [record(1), record(2), record(3)]
        rows[0]["trajectory_metrics"][-1]["metrics"]["accuracy"] += 5e-13
        self.summarize(rows)

    def test_seed_identity_and_manifest_validation(self):
        rows = [record(1), record(2), record(3)]
        rows[0]["config"]["seed"] = 2
        with self.assertRaises(ValueError):
            self.summarize(rows)
        for manifest in ({"seeds": [1, 1], "rounds": 2},
                         {"seeds": [True], "rounds": 2},
                         {"seeds": [], "rounds": 2},
                         {"seeds": [1], "rounds": 0},
                         {"seeds": [1], "rounds": True}):
            with self.subTest(manifest=manifest):
                with self.assertRaises(ValueError):
                    summarize_records([record(1)], manifest)
        with self.assertRaises(ValueError):
            self.summarize([])

    def test_nonfinite_metrics_have_explicit_denominators(self):
        rows = [record(1, aeod=math.nan, aspd=math.nan),
                record(2, aeod=0.2, aspd=math.nan),
                record(3, aeod=0.4, aspd=math.inf)]
        result = {row["metric"]: row for row in self.summarize(rows)}
        self.assertEqual((result["aeod"]["n_total"], result["aeod"]["n_defined"]), (3, 2))
        self.assertAlmostEqual(result["aeod"]["mean"], 0.3)
        self.assertAlmostEqual(result["aeod"]["sample_std"], math.sqrt(0.02))
        self.assertEqual(result["aspd"]["n_defined"], 0)
        self.assertIsNone(result["aspd"]["mean"])
        self.assertIsNone(result["aspd"]["sample_std"])
        single = summarize_records([record(1)], {"seeds": [1], "rounds": 2})
        self.assertTrue(all(row["sample_std"] is None for row in single))

    def test_cli_validates_before_creation_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, manifest, output = base / "raw.jsonl", base / "manifest.json", base / "summary.csv"
            manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/summarize_results.py"),
                       "--input", str(source), "--manifest", str(manifest), "--output", str(output)]
            source.write_text(json.dumps(record(1)) + "\n", encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2, rejected.stderr)
            self.assertFalse(output.exists())
            source.write_text("\n".join(json.dumps(record(seed)) for seed in (1, 2, 3)) + "\n", encoding="utf-8")
            original = source.read_bytes()
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            csv_bytes = output.read_bytes()
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2, repeated.stderr)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), csv_bytes)


if __name__ == "__main__":
    unittest.main()
