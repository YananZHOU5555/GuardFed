"""Local identity/rejection checks only: no Torch, image, margin or inference."""
from __future__ import annotations
import ast
import argparse
import copy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import types
sys.dont_write_bytecode = True
import bridge as b

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def main(output_dir=HERE):
    inventory = b.read(HERE / 'inventory_actual60_Full100refs.json')
    baseline = b.read(WORKSPACE / 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
    pins = inventory['input_pins']
    before = {k: b.digest(WORKSPACE / k) for k in pins}
    b.require(before == pins, 'Sealed inputs changed before checks')
    checks, rejections = [], []
    b.require(len(b.validate_inventory(inventory, baseline)) == 60, 'Actual accepted60 did not pass')
    checks.append('actual60_accepted_terminals_and100_reference_only_Full')

    def reject(name, mutate):
        value = copy.deepcopy(inventory)
        mutate(value)
        try:
            b.validate_inventory(value, baseline)
        except (ValueError, KeyError) as error:
            rejections.append({'case': name, 'error': str(error)})
        else:
            raise AssertionError('Must reject: ' + name)

    def config_change(value, key, new_value):
        r = value['records'][0]
        r['config'][key] = new_value
        r['config_canonical_sha256'] = b.canonical(r['config'])

    reject('missing_actual_terminal', lambda x: x['records'].pop())
    reject('duplicate_actual_terminal', lambda x: x['records'].__setitem__(1, copy.deepcopy(x['records'][0])))
    reject('duplicate_Full_reference', lambda x: x['full_references'].__setitem__(1, copy.deepcopy(x['full_references'][0])))
    reject('replay_Full_again', lambda x: x['full_references'][0].__setitem__('replay_required_here', True))
    reject('mixed_Full_checkpoint', lambda x: x['full_references'][0].__setitem__('checkpoint_sha256', x['full_references'][1]['checkpoint_sha256']))
    reject('mixed_new_checkpoint', lambda x: x['records'][0]['checkpoint'].__setitem__('sha256', x['records'][1]['checkpoint']['sha256']))
    reject('mixed_manifest_id', lambda x: x['records'][0]['manifest_entry'].__setitem__('id', x['records'][1]['id']))
    reject('mixed_raw_job', lambda x: x['records'][0]['raw_job'].__setitem__('sha256', x['records'][1]['raw_job']['sha256']))
    reject('wrong_paired_root', lambda x: x['records'][0]['data_contract'].__setitem__('root_image_ids_sha256', '0' * 64))
    reject('wrong_valid_order', lambda x: x['records'][0]['data_contract'].__setitem__('evaluation_image_ids_sha256', '0' * 64))
    reject('changed_nonintervention_recipe', lambda x: config_change(x, 'learning_rate', 0.001))
    reject('silently_remove_ablation', lambda x: config_change(x, 'ablation_component', 'none'))
    reject('changed_seed', lambda x: config_change(x, 'seed', 91009))
    reject('test_split_config', lambda x: config_change(x, 'celeba_evaluation_split', 'test'))
    reject('missing_real_support', lambda x: x['records'][0]['data_contract']['client_sample_counts'].__setitem__(0, 1))
    reject('wrong_native_recipe', lambda x: config_change(x, 'ad2_calibration_enabled', False))
    reject('foreign_pending_ID', lambda x: x['pending_new_ids_no_checkpoint'].__setitem__(0, 'invented_pending_seed'))
    reject('pending_as_new_checkpoint', lambda x: x['records'][0].__setitem__('id', x['pending_new_ids_no_checkpoint'][0]))
    reject('relaxed_native_tolerance', lambda x: x.__setitem__('native_tolerance', 1e-6))
    reject('fake_inference_claim', lambda x: x.__setitem__('new_image_inference_performed', True))
    reject('remove_prior_closed_replay', lambda x: x['closed_replay_ids'].pop())
    reject('reinclude_closed_ID_in_selected37', lambda x: x['selected_replay_ids'].__setitem__(0, b.CLOSED_IDS[0]))
    reject('unavailable_variant', lambda x: x['records'][0].__setitem__('variant', 'no_candidate'))
    checks.append('corrupt_or_mixed_inventory_boundaries_refused')

    # Execute the actual sealed check_native body without importing scientific
    # libraries. Only np.isfinite is required and is bound to math.isfinite.
    replay_path = WORKSPACE / 'tmp/celeba_final_valid_replay_20261009/replay.py'
    b.require(b.digest(replay_path) == b.V2_SHA, 'Native function source changed')
    tree = ast.parse(replay_path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_native')
    namespace = {'require': b.require, 'np': types.SimpleNamespace(isfinite=math.isfinite), 'TOLERANCE': b.TOLERANCE}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(replay_path), 'exec'), namespace)
    expected = {'accuracy': 0.8, 'aeod': 0.04, 'aspd': 0.05}  # scalar function fixtures, never experiment results
    near = {**expected, 'accuracy': expected['accuracy'] + 1e-13}
    far = {**expected, 'accuracy': expected['accuracy'] + 1e-11}
    b.require(namespace['check_native'](near, expected)['accepted'] and not namespace['check_native'](far, expected)['accepted'], 'Native tolerance boundary changed')
    checks.append('actual_sealed_native_function_accepts1e-13_refuses1e-11')

    approval = {'status': 'APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY', 'scope': b.SCOPE,
                'inventory_sha256': 'fixture_inventory', 'bridge_sha256': 'fixture_bridge',
                'selected_ids': b.REPLAY_IDS, 'device': 'cpu', 'compute_threads': 8,
                'max_processes': 1, 'allowed_cpus': list(range(112, 120)), 'target_split': 'valid',
                'final_test_dispatch': False, 'native_tolerance': b.TOLERANCE}
    b.require_approval(approval, 'fixture_inventory', b.REPLAY_IDS[0], 'fixture_bridge')
    for name, mutate in [
        ('no_approval', lambda x: x.__setitem__('status', 'PREPARED_NOT_FROZEN')),
        ('unallocated_CPUs', lambda x: x.__setitem__('allowed_cpus', [])),
        ('GPU_approval', lambda x: x.__setitem__('device', 'cuda')),
        ('finaltest_approval', lambda x: x.__setitem__('final_test_dispatch', True)),
        ('already_replayed_approval', lambda x: x['selected_ids'].__setitem__(0, b.CLOSED_IDS[0])),
        ('pending_ID_approval', lambda x: x['selected_ids'].__setitem__(0, inventory['pending_new_ids_no_checkpoint'][0])),
        ('changed_CPU_allocation', lambda x: x.__setitem__('allowed_cpus', list(range(8)))),
        ('reordered_IDs', lambda x: x['selected_ids'].reverse()),
        ('foreign_Full_approval', lambda x: x['selected_ids'].append(inventory['full_references'][0]['id'])),
    ]:
        changed = copy.deepcopy(approval); mutate(changed)
        try:
            b.require_approval(changed, 'fixture_inventory', b.REPLAY_IDS[0], 'fixture_bridge')
        except ValueError as error:
            rejections.append({'case': name, 'error': str(error)})
        else:
            raise AssertionError('Must reject: ' + name)
    checks.append('prepared_no_CPU_no_GPU_no_Full_no_test_dispatch_boundaries_refused')

    # Full references are joins to externally approved real batch receipts; this
    # synthetic receipt exercises identity only and is deleted with its tmpdir.
    fixture = {'scope': 'VALID_ONLY_IMPLEMENTATION_REPLAY_V3', 'status': 'SELECTED_VALID_REPLAY_ACCEPTED',
               'inventory_sha256': b.BASELINE_INVENTORY_SHA, 'valid_image_ids_sha256': b.VALID_IDS_SHA,
               'calibration_core_sha256': b.CORE_SHA, 'v2_source_sha256': b.V2_SHA,
               'accepted_ids': [inventory['full_references'][0]['id']], 'accepted_n': 1,
               'max_abs_native_metric_difference': 0.0, 'test_labels_accessed': False, 'invalid': []}
    with tempfile.TemporaryDirectory(prefix='identity_fixtures_', dir=HERE) as temporary:
        path = Path(temporary) / 'NOT_REAL_REPLAY_RECEIPT.json'
        b.save_new(path, fixture)
        joined = b.reference_baseline_full(inventory, baseline, path, b.digest(path))
        b.require(joined['accepted_full_reference_count'] == 1 and joined['new_full_inference'] == joined['full_weights_repacked'] == 0, 'Reference join ran/copies Full')
        for name, changed in [
            ('duplicate_baseline_accepted_ID', {**fixture, 'accepted_ids': fixture['accepted_ids'] * 2, 'accepted_n': 2}),
            ('baseline_native_mismatch', {**fixture, 'max_abs_native_metric_difference': 1e-6}),
            ('baseline_other_calibration_source', {**fixture, 'calibration_core_sha256': '0' * 64}),
            ('aggregate_label_without_batch_proof', {**fixture, 'status': 'ALL900_NATIVE_VALID_REPLAY_ACCEPTED'}),
        ]:
            p = Path(temporary) / (name + '.json'); b.save_new(p, changed)
            try:
                b.reference_baseline_full(inventory, baseline, p, b.digest(p))
            except ValueError as error:
                rejections.append({'case': name, 'error': str(error)})
            else:
                raise AssertionError('Must reject: ' + name)
    checks.append('Full_existing_batch_reference_join_and_duplicate_source_tolerance_refusals')

    reused = {
        'tmp/celeba_final_valid_replay_20261009/replay.py': ['replay_one', 'check_native', 'metadata', 'rebuild_root'],
        'tmp/celeba_final_valid_replay_20261009/v3/replay_v3.py': ['private', 'full_hashes', 'check_source_tokens'],
        'tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py': ['fit_views', 'thresholds_from_root', 'predict_views', 'evaluate_frozen_predictions', 'extract_and_predict'],
        'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py': ['accept_new', 'terminal_checks', 'partition_identity', 'verify_archive'],
    }
    functions = []
    for relative, names in reused.items():
        source = (WORKSPACE / relative).read_text(encoding='utf-8')
        functions_ast = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        for name in names:
            node = functions_ast[name]
            functions.append({'source': relative, 'source_sha256': b.digest(WORKSPACE / relative), 'function': name,
                              'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
                              'source_segment_sha256': hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest(),
                              'body_edited': False})
    b.require({k: b.digest(WORKSPACE / k) for k in pins} == before, 'Source/archive/identity inputs changed during local checks')
    b.require('torch' not in sys.modules and 'numpy' not in sys.modules, 'Local preparation must not import scientific runtime')
    b.require(b.read(HERE / 'reuse_function_hashes.json') == {'unchanged_functions': functions, 'no_source_patch': True}, 'Original16 scientific function proof changed')
    report = {'status': 'PASS_LOCAL_IDENTITIES_AND_REJECTION_ONLY', 'checks': checks, 'rejections': rejections,
              'rejection_count': len(rejections), 'actual_terminal_records': 60, 'Full_reference_count': 100,
              'pending_new_without_checkpoint': 740, 'image_inference_performed': False, 'new_training_performed': False,
              'torch_or_numpy_imported': False, 'runtime_bridge_executed': False,
              'sources_and_archives_unchanged': True, 'inventory_sha256': b.digest(HERE / 'inventory_actual60_Full100refs.json'),
              'bridge_source_sha256': b.digest(b.__file__), 'selfcheck_source_sha256': b.digest(__file__)}
    b.save_new(output_dir / 'selfcheck.json', report)
    print(b.canonical(report), len(rejections), 'rejections; no inference/training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    main(parser.parse_args().output_dir)
