"""Stdlib-only scope/tamper checks. No scientific modules, arrays or fit."""
from pathlib import Path
import argparse, ast, copy, hashlib, importlib.util, json, math, types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('New compact check output required')
    path = HERE / 'audit_saved.py'
    text = path.read_text(encoding='utf-8')
    compile(text, str(path), 'exec')
    spec = importlib.util.spec_from_file_location('_no_fit_audit_source', path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    pins = audit.read(HERE / 'AUDIT_TEST_PINS.json')
    for rel, pin in pins['files'].items():
        audit.check_pin(ROOT / rel, pin)
    evaluator_text = (ROOT / pins['evaluator']).read_text(encoding='utf-8')
    functions = [n for n in ast.parse(evaluator_text).body if isinstance(n, ast.FunctionDef) and n.name in ('canonical_sha', '_seal_fit')]
    ns = {'json': json, 'hashlib': hashlib}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<original pure JSON fit seal>', 'exec'), ns)
    evaluator = types.SimpleNamespace(_seal_fit=ns['_seal_fit'])
    saved = audit.read(ROOT / pins['example_diagnostic'])['saved_fits']
    typed = audit.saved_fits_for_original_predict(saved, evaluator)
    assert set(typed['shared_calibration']['thresholds']) == {0, 1}
    assert typed['shared_calibration']['thresholds'][1] < 0  # Legitimate negative thresholds remain allowed.
    refusals = []
    def refuse(name, value):
        try:
            audit.saved_fits_for_original_predict(value, evaluator)
        except (ValueError, TypeError):
            refusals.append(name)
        else:
            raise AssertionError('Tampered fit accepted: ' + name)
    def reseal(fit):
        return evaluator._seal_fit({k: v for k, v in fit.items() if k != 'fit_sha256'})
    bad = copy.deepcopy(saved); bad['native']['fit_sha256'] = '0' * 64; refuse('bad_fit_SHA', bad)
    bad = copy.deepcopy(saved); bad['native']['unknown'] = 1; bad['native'] = reseal(bad['native']); refuse('unknown_field_even_with_new_valid_hash', bad)
    bad = copy.deepcopy(saved); bad['unknown_view'] = bad.pop('raw'); refuse('unknown_view', bad)
    bad = copy.deepcopy(saved); bad['shared_calibration']['fit_diagnostics']['server_adaptive_lambda'] = -1.0
    bad['shared_calibration'] = reseal(bad['shared_calibration']); refuse('negative_lambda', bad)
    bad = copy.deepcopy(saved); bad['shared_calibration']['thresholds']['0'] = float('nan'); refuse('NaN_threshold', bad)
    bad = copy.deepcopy(saved); bad['shared_calibration']['thresholds'][0] = bad['shared_calibration']['thresholds']['0']; refuse('mixed_key_collision', bad)
    original = (ROOT / pins['saved_science']).read_text(encoding='utf-8')
    function = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'check_saved')
    original_block = next(n for n in function.body if isinstance(n, ast.With))
    block = audit.saved_output_block(original_block)
    retained = [0, 1, 4, 5, 6, 7, 8]
    assert [ast.dump(n, include_attributes=False) for n in block.body] == [ast.dump(original_block.body[i], include_attributes=False) for i in retained]
    compile(ast.Module(body=[block], type_ignores=[]), '<original seven output statements>', 'exec')
    views = ['native', 'raw', 'shared_calibration']
    fixture = {'v2': types.SimpleNamespace(require=audit.need, np=types.SimpleNamespace(array_equal=lambda a, b: a == b), VIEWS=views),
               'predictions': {v: [0, 1] for v in views}, 'z': {'prediction_' + v: [0, 1] for v in views}}
    predicate = compile(ast.Module(body=[original_block.body[5]], type_ignores=[]), '<original prediction exactness guard>', 'exec')
    exec(predicate, fixture)
    fixture['z']['prediction_shared_calibration'] = [1, 1]
    try: exec(predicate, fixture)
    except ValueError: refusals.append('saved_prediction_tamper')
    else: raise AssertionError('Saved prediction tamper accepted')
    fixture.update(scored={'native': {'accuracy': 1.0}}, comparison={'accepted': True}, r={'views': {'native': {'accuracy': 1.0}}, 'native_comparison': {'accepted': True}})
    predicate = compile(ast.Module(body=[original_block.body[8]], type_ignores=[]), '<original metric exactness guard>', 'exec')
    exec(predicate, fixture)
    fixture['r']['views']['native']['accuracy'] = 0.0
    try: exec(predicate, fixture)
    except ValueError: refusals.append('saved_metric_tamper')
    else: raise AssertionError('Saved metric tamper accepted')
    try: audit.check_pin(path, {'sha256': '0' * 64, 'bytes': path.stat().st_size})
    except ValueError: refusals.append('source_SHA_change')
    else: raise AssertionError('Wrong source pin accepted')
    forbidden = {'fit_views', 'thresholds_from_root', 'fit_group_thresholds', 'model_margins', 'extract_and_predict'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in ast.walk(ast.parse(text)))
    result = {'status': 'SOURCE_AND_STDLIB_TAMPER_CHECK_PASS_NOT_SCIENTIFIC_ACCEPTANCE',
              'exact_original_output_statements': 7, 'original_statement_indexes': retained,
              'original_seal_fit_functions_used': True, 'legal_negative_threshold_preserved': True,
              'refusals': refusals, 'fit_calls': 0, 'CNN_calls': 0, 'SSH': False, 'actual_arrays_read': False,
              'source_sha256': audit.sha(path), 'scientific_acceptances': 0}
    audit.save(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
