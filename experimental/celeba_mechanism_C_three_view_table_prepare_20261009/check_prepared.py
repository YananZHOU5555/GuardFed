"""No scientific statistics, prediction-array load, fit, inference or future proof."""
import ast
import copy
import sys
sys.dont_write_bytecode = True
import build as b


def main():
    basis = b.read(b.H / 'INPUTS.json')
    for name, pin in basis['files'].items(): b.need(b.sha(b.R / name) == pin['sha256'], 'Actual pinned input changed ' + name)
    old = b.read(b.C1 / 'inventory_actual101_Full100refs.json')
    current = b.read(b.C11 / 'inventory_actual112_Full100refs.json')
    controls = b.scope(old, current)
    join, science, proof = b.source_functions(basis)
    b.need(len(controls) == 12 and len(current['selected_replay_ids']) == 11, 'Wrong native scope')
    # Extracting exact source functions must not run their readers or scientific body.
    b.need(callable(join) and proof['only_two_variant_literals_rebound'], 'Minimal variant rebind missing')
    for filename in ['build.py', 'panels.py', 'verify_numeric.py', 'check_prepared.py']:
        ast.parse((b.H / filename).read_text(encoding='utf-8'), filename=filename)
    refusals = []
    def changed_prior(j): j['records'][0]['checkpoint']['sha256'] = '0' * 64
    def wrong_split(j): next(r for r in j['records'] if r['id'] in b.EXPECTED)['original_split'] = 'test'
    cases = [('missing_C_record', lambda j: j['records'].pop()), ('duplicate_C_record', lambda j: j['records'].append(copy.deepcopy(j['records'][-1]))),
        ('changed_prior101_identity', changed_prior), ('unmatched_selected11', lambda j: j['selected_replay_ids'].pop()),
        ('tolerance_changed', lambda j: j.__setitem__('native_tolerance', 1e-6)), ('wrong_split', wrong_split),
        ('Full_reference_drift', lambda j: j['full_references'][0].__setitem__('checkpoint_sha256', '0' * 64))]
    for name, mutate in cases:
        bad = copy.deepcopy(current); mutate(bad)
        try: b.scope(old, bad)
        except ValueError: refusals.append(name)
        else: raise AssertionError('Must refuse ' + name)
    try: b.adoption_gate(b.H / 'NOT_A_ROOT_ADOPTION_REVIEW.json', '0' * 64)
    except ValueError: refusals.append('unreviewed_or_wrong_namespace_adoption')
    else: raise AssertionError('Unreviewed adoption must be refused')
    b.need('torch' not in sys.modules and 'numpy' not in sys.modules, 'Unexpected scientific runtime import')
    b.need(not (b.H / 'snapshot').exists(), 'No result may be generated before actual adoption')
    result = dict(status='PREPARED_ONLY_NO_RESULT_NO_NEW_C11_ADOPTION_BOUND', actual_native_C_records=12, already_adopted_C_views=1,
        awaited_new_adopted_C_views=11, expected_complete_scene='IID Benign', expected_mean_table_models=20, expected_preserved_models=24,
        expected_partial_pairs_excluded=2, expected_display_cells=81, expected_mean_sd_scalars=162,
        source_function_reuse=proof, pinned_actual_inputs=len(basis['files']), refusals=refusals,
        scientific_statistics_computed=False, prediction_arrays_loaded=False, new_CNN=False, new_Full_inference=False, test=False, new_acceptance_registration=False)
    b.write(b.H / 'PREPARED_CHECKS.json', result)
    print(__import__('json').dumps(result))


if __name__ == '__main__': main()
