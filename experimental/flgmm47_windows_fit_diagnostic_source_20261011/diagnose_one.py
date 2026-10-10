"""One explicit cached-root diagnostic; never acceptance, CNN or a 47-record retry."""
from pathlib import Path
import argparse, ast, datetime, hashlib, importlib.util, json, sys, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACTUAL = ROOT / 'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001'
PREPARED = ROOT / 'tmp/celeba_flgmm_closed47_saved_acceptance_preparation_20261011/attempt_v2'
RID = 'FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91006_fullcoverage'
OUTPUT = ACTUAL / 'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC.json'
STARTED = ACTUAL / 'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC.started.json'
FAILED = ACTUAL / 'OFFSERVER_SINGLE_RECORD_DIAGNOSTIC.failure.json'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def setup_nodes(text):
    main = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    result = []
    for n in main.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rows' for t in n.targets):
            return result
        result.append(n)
    raise ValueError('Original setup boundary missing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-seal-sha256', required=True)
    parser.add_argument('--allow-single-record-cached-root-diagnostic', action='store_true', required=True)
    args = parser.parse_args()
    need(__debug__ and sys.platform == 'win32', 'Windows diagnostic without -O only')
    need(sha(HERE / 'FILES_SHA256.json') == args.source_seal_sha256, 'Diagnostic source seal differs')
    for rel, pin in read(HERE / 'FILES_SHA256.json')['files'].items():
        p = (HERE / rel).resolve()
        need(p.is_relative_to(HERE) and sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Diagnostic source member differs')
    pins = read(HERE / 'SOURCE_PINS.json')
    for rel, h in pins['files'].items():
        need(sha(ROOT / rel) == h, 'Source/proof changed: ' + rel)
    failure = read(ACTUAL / 'OFFSERVER_ARRAY_REFIT_CHECK.failure.json')
    need(failure['completed'] == 4 and 'Root-only threshold fit changed' in failure['traceback'], 'Wrong preserved failure')
    need(not any(p.exists() for p in (OUTPUT, STARTED, FAILED)), 'One diagnostic attempt only; preserve existing evidence')
    save(STARTED, {'id': RID, 'scope': 'SINGLE_RECORD_DIAGNOSTIC_NOT_ACCEPTANCE', 'source_seal_sha256': args.source_seal_sha256})
    fits_called = 0
    try:
        sys.path.insert(0, str(PREPARED))
        spec = importlib.util.spec_from_file_location('_fl47_diagnostic_original_setup', PREPARED / 'verify_arrays.py')
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        text = (PREPARED / 'verify_arrays.py').read_text(encoding='utf-8')
        ns = dict(verifier.__dict__)
        sys.argv = [str(PREPARED / 'verify_arrays.py'), '--transport-proof', str(ACTUAL / 'TRANSPORT_VERIFICATION.json'),
                    '--transport-proof-sha256', pins['transport_sha256'], '--linux-proof-sha256', pins['linux_sha256'],
                    '--metadata-npz', pins['metadata_path'], '--output', str(OUTPUT), '--allow-original-cached-root-refit']
        exec(compile(ast.Module(body=setup_nodes(text), type_ignores=[]), '<unchanged sealed verifier setup>', 'exec'), ns)
        row = ns['manifest']['records'][4]
        need(row['id'] == RID == ns['gate']['receipts'][4]['id'], 'Only failed fifth record allowed')
        run = ns['extract'] / 'bundle' / RID
        receipt = ns['read'](run / 'receipt.json')
        need(receipt == ns['gate']['receipts'][4], 'Gate/receipt differs')
        need(sha(run / 'receipt.json') == pins['receipt_sha256'] and sha(run / 'validation_predictions.npz') == pins['array_sha256'], 'Fixed record artifact differs')
        ctx = dict(ns['b'].__dict__, row=row, run=run, bridge=ns['bridge'], evaluator=ns['ev'], core=ns['core'],
                   ids=ns['ids'], y=ns['y'], sensitive=ns['sensitive'], root_authorized_cached_refit=True)
        exec(compile(ast.Module(body=ns['preamble'], type_ignores=[]), '<unchanged original identity guards>', 'exec'), ctx)
        original_result = ctx['validate_external'](None, ctx['record'], ROOT)
        core, ev, np = ns['core'], ns['ev'], ns['np']
        cfg = core.ExperimentConfig(**ctx['record']['config'])
        root_ids, root_y, root_s, root_receipt = ev.rebuild_root(core, cfg, ctx['record'], ns['ids'], ns['y'], ns['sensitive'])
        with np.load(run / 'validation_predictions.npz', allow_pickle=False) as z:
            need(np.array_equal(z['root_image_ids'], root_ids) and np.array_equal(z['valid_image_ids'], ns['ids'][162770:182637]), 'Original root/valid IDs differ')
            fits_called += 1
            fits = ev.fit_views(core, ctx['record']['method'], z['root_margins'], root_y, root_s, cfg, ev.VIEWS, ev.SHARED_CALIBRATION)
            predictions = ev.predict_views(z['valid_margins'], ns['sensitive'][162770:], fits)
            mismatches = {v: int(np.count_nonzero(predictions[v] != z['prediction_' + v])) for v in ev.VIEWS}
            scored = ev.evaluate_frozen_predictions(predictions, ns['y'][162770:], ns['sensitive'][162770:])
        differences = ns['diff_ns']['diff']
        after = ns['b'].measure(ctx['paths'], ctx['record'])
        need(ctx['before'] == ctx['second'] == after, 'Original accepted artifacts changed')
        for rel, h in pins['files'].items():
            need(sha(ROOT / rel) == h, 'Source/proof changed during diagnostic: ' + rel)
        need(sha(run / 'validation_predictions.npz') == pins['array_sha256'], 'Saved array changed')
        report = {'status': 'SINGLE_RECORD_WINDOWS_SAVED_FIT_DIAGNOSTIC_NOT_ACCEPTANCE', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'id': RID, 'checkpoint_sha256': receipt['checkpoint_sha256'], 'array_sha256': pins['array_sha256'], 'receipt_sha256': pins['receipt_sha256'],
                  'original_failure_sha256': pins['failure_sha256'], 'source_seal_sha256': args.source_seal_sha256,
                  'actual_fits': fits, 'saved_fits': receipt['fits'], 'fit_field_differences': differences(fits, receipt['fits']),
                  'actual_root_receipt': root_receipt, 'saved_root_receipt': receipt['root_reconstruction'],
                  'root_receipt_differences': differences(root_receipt, receipt['root_reconstruction']),
                  'prediction_mismatch_counts': mismatches, 'actual_views': scored, 'saved_views': receipt['views'],
                  'metric_and_count_differences': differences(scored, receipt['views']),
                  'native_comparison_diagnostic_only': ev.check_native(scored['native'], original_result['metrics']),
                  'runtime': {'python': sys.version, 'numpy': np.__version__, 'pandas': ns['pd'].__version__, 'torch': ns['torch'].__version__, 'device': 'cpu'},
                  'fit_views_calls': fits_called, 'diagnostic_records': 1, 'scientific_acceptances': 0, 'root_adopted': False,
                  'Windows_whole_check_pass': None, 'original_tolerance': 1e-12, 'test': False, 'new_CNN': 0, 'new_training': 0,
                  'labels': 'Original train+valid prefix only; 182637 rows', 'original_failure_preserved': True}
        save(OUTPUT, report)
        print(json.dumps({'status': report['status'], 'id': RID, 'report_sha256': sha(OUTPUT), 'fit_differences': len(report['fit_field_differences']), 'prediction_mismatches': mismatches}))
    except BaseException:
        save(FAILED, {'status': 'SINGLE_RECORD_DIAGNOSTIC_FAILED_PRESERVED_NO_RETRY', 'id': RID, 'fit_views_calls': fits_called, 'traceback': traceback.format_exc(), 'root_adopted': False})
        raise


if __name__ == '__main__':
    main()
