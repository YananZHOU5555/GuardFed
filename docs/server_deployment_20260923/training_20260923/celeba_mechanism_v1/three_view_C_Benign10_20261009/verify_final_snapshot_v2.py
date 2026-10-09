"""Finite final-source/display checks; reuse the existing independent arithmetic."""
import collections
import datetime
import sys
sys.dont_write_bytecode = True
import build as b


def main():
    prepared = b.read(b.H / 'PREPARED_FILES_SHA256.json')
    b.need(b.sha(b.H / 'PREPARED_FILES_SHA256.json') == '66acce6116398fb80724241e093aa3cc2c2f842c4cfbb7eb9a506363f7bbf71c', 'Prepared seal drift')
    for member in prepared['members']: b.need(b.sha(b.H / member['path']) == member['sha256'], 'Prepared source changed')
    basis = b.read(b.H / 'INPUTS.json')
    for name, pin in basis['files'].items(): b.need(b.sha(b.R / name) == pin['sha256'], 'Actual source changed')
    root = b.C11 / 'execution_candidate/backups/incremental_20261009T193419Z/ROOT_ADOPTION_REVIEW.json'
    proof = b.adoption_gate(root, '8d064687ad7841e1050120a12ada9e10458fea5bb6a4d9da5aeed77584437fc5')
    paths = [root, root.parent / 'backup_receipt.json', root.parent / 'OFFSERVER_VERIFICATION.json', root.parent / 'incremental_valid_three_views.tar.gz']
    pins = {p.relative_to(b.R).as_posix(): dict(sha256=b.sha(p), bytes=p.stat().st_size) for p in paths}
    for path, key in zip(paths[1:], ['backup_receipt_sha256', 'offserver_verification_sha256', 'archive_sha256']): b.need(b.sha(path) == proof[key], 'Actual new closure drift')
    b.write(b.H / 'FINAL_INPUTS.json', dict(status='ACTUAL_C1_C11_ROOT_ADOPTED_INPUTS', prepared_inputs_sha256=b.sha(b.H / 'INPUTS.json'), prepared_input_pins=basis['files'], actual_new_closure_files=pins, actual_new_adoption_sha256=b.sha(root), prior_C1_adoption_sha256=proof['prior101_root_adoption_sha256'], native_tolerance=1e-12))
    snapshot = b.H / 'snapshot'; records = b.read(snapshot / 'records.json')['records']; tables = b.read(snapshot / 'tables.json')
    checker = b.module('existing_independent_numeric_checker_final', b.H / 'verify_numeric.py')
    checks = checker.verify(records, tables['panels'])
    original_checks = b.read(snapshot / 'verification.json')
    b.need(all(checks[key] == original_checks[key] for key in checks), 'Existing independent check drift')
    native = b.read(b.R / basis['native_C_tables']); native_checks = 0; maximum = 0.
    for panel in native['panels']:
        actual = next(p for p in tables['panels'] if p['view'] == 'native' and p['label'] == panel['label'])
        for left, right in zip(panel['rows'], actual['rows']):
            b.need(left['variant'] == right['variant'] and left['seeds'] == right['seeds'], 'Native panel identity drift')
            for metric in ['accuracy_pct', 'aeod', 'aspd']:
                for stat in ['mean', 'sample_sd_ddof1']:
                    error = abs(left[metric][stat] - right[metric][stat]); b.need(error <= 1e-12, 'Native statistic mismatch'); maximum = max(maximum, error); native_checks += 1
    lines = [line for line in (snapshot / 'TABLES.md').read_text(encoding='utf-8').splitlines() if line.startswith('| ') and ' ± ' in line]
    flat = [row for panel in tables['panels'] for row in panel['rows']]
    b.need(len(lines) == len(flat) == 27, 'Missing displayed row'); display = 0
    for line, row in zip(lines, flat):
        cells = line.strip('| ').split(' | ')
        for index, metric in enumerate(['accuracy_pct', 'aeod', 'aspd'], 2):
            precision = 3 if metric == 'accuracy_pct' else 5
            b.need(cells[index] == f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}", 'Display mismatch'); display += 1
    b.need(display == 81 and native_checks == 54, 'Incomplete cell/native check')
    shown = [r for r in records if r['attack'] == 'Benign']
    b.need(len(shown) == 20 and len(records) == 24, 'Complete/partial scope changed')
    identical = sum(r['views']['native'] == r['views']['shared_calibration'] for r in records)
    b.need(identical == 24, 'Unexpected native/shared difference')
    checks.update(status='C_SINGLE_SCENE_THREE_VIEW_FINAL_OFFLINE_CHECKS_PASS', checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), prepared_seal11_unchanged=True, prepared_source_pins25_unchanged=True, actual_new_adoption_sha256=b.sha(root), display_mean_sd_cells=display, original_native_table_scalar_matches=native_checks, max_native_scalar_difference=maximum, preserved_records=24, displayed_records=20, native_shared_metrics_and_counts_identical_preserved_records=24, native_shared_metrics_and_counts_identical_display_records=20, replay_devices={v:dict(collections.Counter(r['replay_runtime']['device'] for r in shown if r['variant']==v)) for v in ['Full','minus_C']}, training_torch={v:dict(collections.Counter(r['training_torch'] for r in shown if r['variant']==v)) for v in ['Full','minus_C']}, new_CNN=0, new_Full_inference=0, threshold_refits=0, test=False, primary_endpoint='PENDING_AUTHOR', root_table_adoption='PENDING_INDEPENDENT_REVIEW')
    b.write(b.H / 'FINAL_CHECKS.json', checks)
    print(b.json.dumps(checks))


if __name__ == '__main__': main()
