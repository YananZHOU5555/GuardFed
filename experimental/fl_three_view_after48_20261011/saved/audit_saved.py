"""Exact13 saved-output audit after accepted48. Zero refit; Windows exact-refit failure remains."""
from pathlib import Path
import argparse, ast, copy, datetime, hashlib, importlib.util, json, math, sys, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ACTUAL = ROOT / 'tmp/fl_three_view_after48_20261011/actual/saved001'
HISTORICAL = ROOT / 'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001'
PREPARED = HERE
OUTPUT = ACTUAL / 'SAVED_OUTPUTS_AUDIT_NO_REFIT.json'
STARTED = ACTUAL / 'SAVED_OUTPUTS_AUDIT_NO_REFIT.started.json'
FAILED = ACTUAL / 'SAVED_OUTPUTS_AUDIT_NO_REFIT.failure.json'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_pin(path, pin):
    need(sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes'], 'Changed pinned source/evidence: ' + str(path))


def saved_fits_for_original_predict(saved, evaluator):
    need(set(saved) == {'native', 'raw', 'shared_calibration'}, 'Require all three saved views')
    fits = copy.deepcopy(saved)
    for view, fit in fits.items():
        common = {'view', 'method', 'rule', 'thresholds', 'fit_data', 'fit_sha256'}
        need(set(fit) == common | ({'fit_diagnostics'} if view == 'shared_calibration' else set()), 'Unknown/missing fit payload field')
        need(fit['view'] == view and fit['method'] == 'FLGMM', 'Saved fit method/view identity differs')
        payload = {k: v for k, v in fit.items() if k != 'fit_sha256'}
        need(evaluator._seal_fit(payload) == fit, 'Saved fit payload SHA differs')
        if view == 'shared_calibration':
            need(set(fit['thresholds']) == {'0', '1'}, 'Require exact JSON group keys')
            need(all(type(v) in (int, float) and math.isfinite(v) for v in fit['thresholds'].values()), 'Nonfinite or invalid threshold')
            diagnostic = fit['fit_diagnostics']
            expected = {'server_calibration_metrics', 'server_calibration_score', 'server_calibration_risk',
                        'server_base_calibration_metrics', 'server_base_calibration_risk', 'server_adaptive_lambda',
                        'server_calibration_budget', 'server_calibration_acc_floor',
                        'server_calibration_max_acc_drop', 'server_calibration_objective'}
            need(set(diagnostic) == expected, 'Unknown/missing original calibration diagnostic field')
            for key in ('server_calibration_metrics', 'server_base_calibration_metrics'):
                need(set(diagnostic[key]) == {'accuracy', 'aeod', 'aspd'} and
                     all(type(v) in (int, float) and math.isfinite(v) for v in diagnostic[key].values()), 'Invalid root calibration metrics')
            for key in expected - {'server_calibration_metrics', 'server_base_calibration_metrics', 'server_calibration_objective'}:
                value = diagnostic[key]
                need(type(value) in (int, float) and math.isfinite(value), 'Nonfinite calibration diagnostic')
                if key != 'server_calibration_score':
                    need(value >= 0, 'Negative nonnegative calibration diagnostic')
            need(diagnostic['server_calibration_objective'] == 'acc_floor', 'Wrong frozen shared objective')
            fit['thresholds'] = {int(k): v for k, v in fit['thresholds'].items()}
        typed_payload = {k: v for k, v in fit.items() if k != 'fit_sha256'}
        need(evaluator._seal_fit(typed_payload) == fit, 'JSON group-key restoration changed fit SHA')
    return fits


def saved_output_block(original_block):
    """Keep seven original statements; exclude only refit and refit-equality statements."""
    need(len(original_block.body) == 9, 'Original saved-array statement layout changed')
    need(ast.unparse(original_block.body[2]).startswith('fits = evaluator.fit_views('), 'Original refit statement changed')
    need('Root-only threshold fit changed' in ast.unparse(original_block.body[3]), 'Original refit equality changed')
    block = copy.deepcopy(original_block)
    block.body = [copy.deepcopy(original_block.body[i]) for i in (0, 1, 4, 5, 6, 7, 8)]
    forbidden = {'fit_views', 'thresholds_from_root', 'fit_group_thresholds', 'model_margins', 'extract_and_predict'}
    need(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in ast.walk(block)), 'Saved-output path contains fitting or CNN')
    return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-seal-sha256', required=True)
    parser.add_argument('--allow-saved-output-audit', action='store_true', required=True)
    args = parser.parse_args()
    need(sys.platform == 'win32' and __debug__, 'Windows without -O only')
    need(sha(HERE / 'FILES_SHA256.json') == args.source_seal_sha256, 'Wrong audit source seal')
    for rel, pin in read(HERE / 'FILES_SHA256.json')['files'].items():
        path = (HERE / rel).resolve()
        need(path.is_relative_to(HERE), 'Unsafe source member')
        check_pin(path, pin)
    pins = read(HERE / 'SOURCE_PINS.json')
    for rel, pin in pins['files'].items():
        check_pin(ROOT / rel, pin)
    need(not any(p.exists() for p in (OUTPUT, STARTED, FAILED)), 'Preserve original audit attempt; no retry')
    need(read(HISTORICAL / 'OFFSERVER_ARRAY_REFIT_CHECK.failure.json')['completed'] == 4, 'Original failed refit must remain preserved')
    save(STARTED, {'scope': 'SAVED_OUTPUTS_AUDIT_NO_REFIT', 'source_seal_sha256': args.source_seal_sha256, 'fit_calls': 0})
    rows = []
    try:
        sys.path.insert(0, str(PREPARED))
        verifier = load('_fl47_saved_output_setup', PREPARED / 'verify_arrays.py')
        diagnostic = load('_fl47_setup_extractor', ROOT / pins['setup_extractor'])
        prefix = diagnostic.setup_nodes((PREPARED / 'verify_arrays.py').read_text(encoding='utf-8'))
        need(hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False).encode()).hexdigest() == pins['setup_AST_sha256'], 'Original setup AST changed')
        # Legacy parser marker is consumed by setup only. No original fit loop is executed.
        sys.argv = [str(PREPARED / 'verify_arrays.py'), '--transport-proof', str(ACTUAL / 'TRANSPORT_VERIFICATION.json'),
                    '--transport-proof-sha256', pins['transport_sha256'], '--linux-proof-sha256', pins['linux_sha256'],
                    '--metadata-npz', pins['metadata_path'], '--output', str(OUTPUT), '--allow-original-cached-root-refit']
        ns = dict(verifier.__dict__)
        exec(compile(ast.Module(body=prefix, type_ignores=[]), '<unchanged original metadata/source/member setup>', 'exec'), ns)
        expected_ids = pins['exact_ids']
        need(len(expected_ids) == len(set(expected_ids)) == 13, 'Require exact13')
        need([r['id'] for r in ns['manifest']['records']] == [r['id'] for r in ns['gate']['receipts']] == expected_ids, 'Frozen scope differs')
        linux = read(ACTUAL / 'LINUX_SAVED_CHECK.json')
        need([r['id'] for r in linux['records']] == expected_ids, 'Linux whole proof scope differs')
        block = saved_output_block(ns['block'])
        for row, receipt, linux_row in zip(ns['manifest']['records'], ns['gate']['receipts'], linux['records']):
            run = ns['extract'] / 'bundle' / row['id']
            need(read(run / 'receipt.json') == receipt, 'Saved receipt differs from gate')
            need(sha(run / 'receipt.json') == linux_row['receipt_sha256'] and sha(run / 'validation_predictions.npz') == linux_row['array_sha256'], 'Linux whole proof/member identity differs')
            ctx = dict(ns['b'].__dict__, row=row, run=run, bridge=ns['bridge'], evaluator=ns['ev'], core=ns['core'],
                       ids=ns['ids'], y=ns['y'], sensitive=ns['sensitive'], root_authorized_cached_refit=True)
            exec(compile(ast.Module(body=ns['preamble'], type_ignores=[]), '<unchanged original identity guards; no fit body>', 'exec'), ctx)
            original_result = ctx['validate_external'](None, ctx['record'], ROOT)
            cfg = ns['core'].ExperimentConfig(**ctx['record']['config'])
            root_ids, root_y, root_s, root_receipt = ns['ev'].rebuild_root(ns['core'], cfg, ctx['record'], ns['ids'], ns['y'], ns['sensitive'])
            need(receipt['weights_before'] == receipt['weights_after'], 'Saved model weight identity drift')
            need(receipt['checkpoint_sha256'] == linux_row['checkpoint_sha256'], 'Whole Linux checkpoint differs')
            fits = saved_fits_for_original_predict(receipt['fits'], ns['ev'])
            scope = dict(ctx, path=run, r=receipt, root_ids=root_ids, root_y=root_y, root_s=root_s,
                         s=ns['sensitive'], cfg=cfg, original_result=original_result, fits=fits)
            exec(compile(ast.Module(body=[block], type_ignores=[]), '<seven original saved-output assertions/statements; zero refit>', 'exec'), scope)
            after = ns['b'].measure(ctx['paths'], ctx['record'])
            need(ctx['before'] == ctx['second'] == after, 'Accepted local artifacts changed')
            root_difference = ns['diff_ns']['diff'](json.loads(json.dumps(root_receipt, allow_nan=False)), receipt['root_reconstruction'])
            rows.append({'id': row['id'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                         'receipt_sha256': sha(run / 'receipt.json'), 'array_sha256': sha(run / 'validation_predictions.npz'),
                         'native_comparison': scope['comparison'], 'saved_fit_payload_hashes_exact': True,
                         'saved_threshold_predictions_metrics_counts_exact': True, 'root_valid_ID_partition_exact': True,
                         'root_receipt_differences': root_difference, 'local_full_root_receipt_exact': not root_difference,
                         'saved_fit_hashes': {v: f['fit_sha256'] for v, f in receipt['fits'].items()},
                         'fit_calls': 0, 'artifact_observations': {'before': ctx['before'], 'second_before': ctx['second'], 'after': after}})
        for rel, pin in pins['files'].items():
            check_pin(ROOT / rel, pin)
        for rel, pin in ns['transport']['members'].items():
            check_pin(ns['extract'] / rel, pin)
        report = {'status': 'SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'records': rows, 'exact_ids': expected_ids, 'source_seal_sha256': args.source_seal_sha256,
                  'linux_whole_proof_sha256': pins['linux_sha256'], 'transport_proof_sha256': pins['transport_sha256'],
                  'original_Windows_refit_failure_sha256': pins['original_failure_sha256'],
                  'original_Windows_refit_remains_failed': True, 'Windows_whole_check_pass': False,
                  'Windows_exact_recalibration_claimed': False, 'saved_fit_parameters_used_without_modification': True,
                  'threshold_group_key_restoration_only': True, 'seven_original_saved_output_statements_unchanged': True,
                  'original_predict_score_native_functions_unchanged': True, 'native_tolerance': 1e-12,
                  'fit_calls': 0, 'new_CNN': 0, 'new_training': 0, 'test': False, 'root_adopted': False,
                  'metric_values_checked': 117, 'integer_base_counts_checked': 312, 'prediction_rules_checked': 39,
                  'runtime': {'python': sys.version, 'numpy': ns['np'].__version__, 'pandas': ns['pd'].__version__, 'torch': ns['torch'].__version__, 'device': 'cpu'},
                  'scope': 'Independent saved-output consistency only. Original Linux whole check supplies root-refit evidence. Windows exact refit remains failed; no acceptance or cross-platform recalibration equivalence is inferred.'}
        save(OUTPUT, report)
        print(json.dumps({'status': report['status'], 'records': len(rows), 'fit_calls': 0, 'report_sha256': sha(OUTPUT)}))
    except BaseException:
        save(FAILED, {'status': 'SAVED_OUTPUTS_AUDIT_NO_REFIT_FAILED_PRESERVED', 'completed': len(rows), 'traceback': traceback.format_exc(), 'fit_calls': 0, 'root_adopted': False})
        raise


if __name__ == '__main__':
    main()
