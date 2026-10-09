"""CPU mechanism gates, not image-training or paper-performance evidence."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from adapter import mechanism, verify_ledger

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'tmp/revision-publish-20260928/scripts/reproduce_paper_tables.py'
spec = importlib.util.spec_from_file_location('mechanism_frozen_core_test', SOURCE)
core = importlib.util.module_from_spec(spec); sys.modules[spec.name] = core; spec.loader.exec_module(core)
torch.set_num_threads(1)


class Controls(unittest.TestCase):
    def setUp(self):
        self.updates = [{'w': torch.tensor(v, dtype=torch.float64)} for v in
                        [[1, .2, .1], [.2, .8, .5], [-.4, 1.1, .3],
                         [1.4, -.2, .8], [.1, .3, 1.7], [-22, -.4, -.3]]]
        self.root = {'w': torch.tensor([.1, .2, .1], dtype=torch.float64)}
        self.details = [{'accuracy': a, 'aeod': f, 'aspd': .6 * f} for a, f in
                        zip([.6, .7, .82, .63, .77, .52], [.01, .08, .14, .21, .3, .4])]
        self.fairness = [d['aeod'] for d in self.details]

    def call(self, cfg):
        return core.guardfed_ad2plus_adaptive_aggregate(
            self.updates, self.fairness, self.root, cfg, fairness_details=self.details,
            global_state={'w': torch.zeros(3, dtype=torch.float64)},
            bundle={'num_features': 3}, device=torch.device('cpu'))

    def root_metric(self, state, *_):
        # Include RNG consumption so the untouched-Full gate checks it too.
        torch.rand(1)
        return {'accuracy': .7 + float(torch.sigmoid(state['w'].sum())) * .1,
                'aeod': .1, 'aspd': .2}

    def test_full_preserves_output_diagnostics_rng_and_function_identity(self):
        fn, selector = core.guardfed_act_aggregate, core.guardfed_ad2plus_adaptive_aggregate
        cfg = core.ExperimentConfig(num_clients=6, num_malicious=2)
        torch.manual_seed(31)
        with patch.object(core, 'evaluate_state_on_server', side_effect=self.root_metric):
            expected, before = self.call(cfg)
        state = torch.get_rng_state().clone()
        torch.manual_seed(31)
        with mechanism(core, 'Full') as ledger, patch.object(core, 'evaluate_state_on_server', side_effect=self.root_metric):
            actual, after = self.call(cfg)
        self.assertTrue(torch.equal(expected['w'], actual['w']))
        self.assertEqual(before, after)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIs(fn, core.guardfed_act_aggregate); self.assertIs(selector, core.guardfed_ad2plus_adaptive_aggregate)
        self.assertTrue(verify_ledger(ledger, 'Full', 1)['pass'])

    def test_each_mask_is_applied_to_all_ten_candidates(self):
        for component in 'UCAFVN':
            variant = 'minus_' + component
            cfg = core.ExperimentConfig(num_clients=6, num_malicious=2, ablation_component=component)
            with mechanism(core, variant) as ledger, patch.object(core, 'evaluate_state_on_server', side_effect=self.root_metric):
                _, info = self.call(cfg)
            self.assertEqual(verify_ledger(ledger, variant, 1)['candidate_calls_verified'], 10)
            if component == 'N':
                self.assertEqual(info['norm_clip_scales'], [1.] * len(info['selected_clients']))
            else:
                self.assertTrue(all(t[component] == 0 for t in info['component_contributions']))

    def test_no_hard_screen_admits_every_client_in_every_candidate(self):
        cfg = core.ExperimentConfig(num_clients=6, num_malicious=2)
        with patch.object(core, 'evaluate_state_on_server', side_effect=self.root_metric):
            _, original = self.call(cfg)
        self.assertLess(len(original['selected_clients']), 6)
        with mechanism(core, 'no_hard_screen') as ledger, patch.object(core, 'evaluate_state_on_server', side_effect=self.root_metric):
            _, info = self.call(cfg)
        self.assertEqual(info['selected_clients'], list(range(6)))
        self.assertTrue(all(x['selected_count'] == 6 for x in ledger))
        self.assertEqual(verify_ledger(ledger, 'no_hard_screen', 1)['candidate_calls_verified'], 10)

    def test_fixed_balanced_has_one_aggregate_and_no_candidate_evaluation(self):
        cfg = core.ExperimentConfig(num_clients=6, num_malicious=2)
        selected = next(dict(x) for x in core.ad2plus_candidate_overrides(cfg) if x['candidate_name'] == 'balanced')
        selected.pop('candidate_name')
        reference, _ = core.guardfed_act_aggregate(self.updates, self.fairness, self.root,
                                                   core.clone_config_with(cfg, selected), fairness_details=self.details)
        with mechanism(core, 'fixed_balanced') as ledger, patch.object(core, 'evaluate_state_on_server', side_effect=AssertionError('candidate evaluation remained')):
            update, info = self.call(cfg)
        self.assertTrue(torch.equal(reference['w'], update['w']))
        self.assertEqual(info['fixed_candidate'], 'balanced')
        self.assertEqual(verify_ledger(ledger, 'fixed_balanced', 1)['candidate_calls_verified'], 1)

    def test_drift_and_incomplete_ledger_are_rejected_and_patches_restore(self):
        original = core.guardfed_act_aggregate
        with self.assertRaises(AssertionError):
            with mechanism(core, 'minus_U'):
                self.call(core.ExperimentConfig(num_clients=6, num_malicious=2))
        self.assertIs(original, core.guardfed_act_aggregate)
        with self.assertRaises(AssertionError): verify_ledger([], 'minus_U', 70)
        with self.assertRaises(ValueError):
            with mechanism(core, 'unknown'): pass


if __name__ == '__main__':
    unittest.main()
