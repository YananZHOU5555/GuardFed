from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_experiment as runner


class ReleaseRunnerTest(unittest.TestCase):
    def job(self):
        return {'dataset':'compas', 'distribution':'IID', 'method':'FedAvg', 'attack':'Benign',
                'config':{'rounds':2, 'seed':123, 'device':'cpu'}}

    def test_existing_output_and_wrong_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary)/'existing'
            existing.mkdir()
            marker = existing/'evidence.txt'
            marker.write_text('preserve')
            with self.assertRaises(FileExistsError):
                runner.run_job(self.job(), existing)
            self.assertEqual(marker.read_text(), 'preserve')
            job = self.job()
            job['source_hashes'] = {'src/data_loader.py':'0'*64}
            with self.assertRaises(ValueError):
                runner.run_job(job, Path(temporary)/'bad-identity')
            self.assertFalse((Path(temporary)/'bad-identity').exists())

    def test_failure_preserves_partial_artifacts(self):
        def fail(*args, checkpoint_path, **kwargs):
            Path(checkpoint_path).write_bytes(b'partial')
            raise RuntimeError('example worker failure')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'failure'
            with patch.object(runner.core, 'run_experiment', side_effect=fail):
                with self.assertRaises(RuntimeError):
                    runner.run_job(self.job(), output)
            self.assertEqual((output/'model.pt').read_bytes(), b'partial')
            self.assertTrue((output/'failure.json').is_file())
            self.assertFalse((output/'result.json').exists())

    def test_saved_terminal_record_is_identical_in_json_and_jsonl(self):
        def complete(dataset, distribution, method, attack, config, mode, device,
                     *, progress_callback, checkpoint_path):
            metrics = {'accuracy':0.7, 'aeod':0.1, 'aspd':0.2}
            trajectory = [{'round':r, 'metrics':dict(metrics)} for r in [1,2]]
            Path(checkpoint_path).write_bytes(b'example checkpoint')
            progress_callback(trajectory[-1])
            return {'dataset':dataset, 'distribution':distribution, 'method':method,
                    'attack':attack, 'seed':config.seed, 'rounds':config.rounds,
                    'config':asdict(config), 'metrics':metrics, 'trajectory_metrics':trajectory}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'complete'
            with patch.object(runner.core,'run_experiment',side_effect=complete):
                runner.run_job(self.job(), output)
            saved = json.loads((output/'result.json').read_text())
            self.assertEqual(saved, json.loads((output/'raw_results.jsonl').read_text()))
            self.assertEqual(saved['metrics'], saved['trajectory_metrics'][-1]['metrics'])


if __name__ == '__main__':
    unittest.main()
