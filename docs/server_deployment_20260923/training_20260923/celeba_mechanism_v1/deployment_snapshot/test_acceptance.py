"""Reject corrupt evidence and unsafe skips using a small synthetic result fixture."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch

from worker import checked, load
from adapter import verify_ledger


class Evidence(unittest.TestCase):
    def test_reuse_requires_intact_model_config_counts_ledger_and_failure_clearance(self):
        core = Path(__file__).resolve().parents[1] / 'revision-publish-20260928'
        original = load('mechanism_acceptance_test_original', core / 'scripts/run_revision_ablation.py')
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            job = {'id': 'synthetic_only', 'output': str(out), 'variant': 'Full',
                   'distribution': 'IID', 'attack': 'Benign', 'source_hashes': {}, 'adapter_hashes': {},
                   'config': {'rounds': 1, 'seed': 1, 'client_alpha': 5000., 'celeba_evaluation_split': 'valid'}}
            job_path = out / 'job.json'; original.write_json(job_path, job)
            torch.save({'weight': torch.tensor([1.])}, out / 'model.pt')
            metrics = {'accuracy': .5, 'aeod': .1, 'aspd': .2}
            result = {'config': job['config'], 'seed': 1, 'distribution': 'IID', 'attack': 'Benign', 'alpha': 5000.,
                      'revision_job': dict(job, checkpoint_sha256=original.digest(out / 'model.pt')),
                      'trajectory_metrics': [{'round': 1, 'metrics': metrics}], 'round_summaries': [{}], 'metrics': metrics,
                      'data_contract': {'image_data_contract': {'evaluation_split': 'valid', 'actual_train_rows': 162770,
                          'actual_evaluation_rows': 19867, 'train_eval_disjoint': True, 'root_client_disjoint': True}}}
            original.write_json(out / 'result.json', result)
            original.write_json(out / 'candidate_mask_audit.json', [])
            audit = dict(verify_ledger([], 'Full', 1), job_sha256=original.digest(job_path),
                         result_sha256=original.digest(out / 'result.json'),
                         checkpoint_sha256=result['revision_job']['checkpoint_sha256'],
                         candidate_audit_sha256=original.digest(out / 'candidate_mask_audit.json'), adapter_hashes={})
            original.write_json(out / 'mechanism_acceptance.json', audit)
            self.assertIsNotNone(checked(original, job, job_path))
            for target, mutate in (
                ('result.json', lambda r: r['data_contract']['image_data_contract'].update(actual_evaluation_rows=1)),
                ('result.json', lambda r: r['config'].update(seed=2)),
                ('mechanism_acceptance.json', lambda r: r.update(checkpoint_sha256='wrong')),
                ('candidate_mask_audit.json', lambda r: r.append({'mask_verified': True})),
            ):
                with self.subTest(target=target):
                    path = out / target; saved = path.read_bytes(); value = json.loads(saved); mutate(value)
                    original.write_json(path, value)
                    with self.assertRaises(AssertionError):
                        checked(original, job, job_path)
                    path.write_bytes(saved)
            original.write_json(out / 'failure.json', {'historical_attempt': 'must preserve and review'})
            with self.assertRaises(AssertionError):
                checked(original, job, job_path)
            (out / 'failure.json').unlink()
            torch.save({'weight': torch.tensor([float('nan')])}, out / 'model.pt')
            with self.assertRaises(AssertionError):
                checked(original, job, job_path)


if __name__ == '__main__':
    unittest.main()
