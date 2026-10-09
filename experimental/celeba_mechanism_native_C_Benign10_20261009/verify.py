"""Independent fsum/sample-SD check of the actual twenty native records."""
import copy
import math
import sys
import build

sys.dont_write_bytecode = True
HERE, ROOT = build.HERE, build.ROOT


def main():
    basis = build.read(HERE / 'INPUTS.json')
    for name, pin in basis['files'].items():
        assert build.sha(ROOT / name) == pin['sha256'], name
    inspection = build.read(ROOT / basis['inspection'])
    all_rows, selected = build.validate_scope(inspection)
    records = build.read(HERE / 'records.json')['records']
    assert records == selected and len(records) == 20
    indexed = {(r['variant'], r['seed']): r for r in records}
    tables = build.read(HERE / 'tables.json')
    prior = build.read(ROOT / basis['prior_tables'])
    markdown_rows = [line for line in (HERE / 'TABLES.md').read_text(encoding='utf-8').splitlines() if line.startswith('| Full |') or line.startswith('| minus_C')]
    assert len(tables['panels']) == 3 and len(markdown_rows) == 9
    checks = 0; maximum = 0.; display_checks = 0; old_full_checks = 0
    for panel_index, (panel, (_, seeds)) in enumerate(zip(tables['panels'], build.PANELS)):
        assert panel['seeds'] == seeds and len(panel['rows']) == 3
        assert [r['variant'] for r in panel['rows']] == ['Full', 'minus_C', 'minus_C minus Full']
        for row_index, row in enumerate(panel['rows']):
            assert row['n'] == row['expected_n'] == len(seeds) and row['complete'] and row['seeds'] == seeds
            assert row['distribution'] == 'IID' and row['attack'] == 'Benign'
            display = []
            for metric in build.METRICS:
                values = [indexed['minus_C', s][metric] - indexed['Full', s][metric] if row['variant'] == 'minus_C minus Full' else indexed[row['variant'], s][metric] for s in seeds]
                mean = math.fsum(values) / len(values)
                sd = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
                for name, actual in [('mean', mean), ('sample_sd_ddof1', sd)]:
                    saved = row[metric][name]
                    assert abs(saved - actual) < 1e-12 and f'{actual:.10f}' == f'{saved:.10f}'
                    maximum = max(maximum, abs(saved - actual)); checks += 1
                precision = 3 if metric == 'accuracy_pct' else 5
                formatted = f'{mean:.{precision}f} ± {sd:.{precision}f}'
                assert formatted == f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}"
                display.append(formatted); display_checks += 1
            assert markdown_rows[panel_index * 3 + row_index] == '| ' + row['variant'] + ' | ' + str(len(seeds)) + ' | ' + ' | '.join(display) + ' |'
        old_full = next(r for r in prior['panels'][panel_index]['rows'] if (r['distribution'], r['attack'], r['variant']) == ('IID', 'Benign', 'Full'))
        assert panel['rows'][0] == old_full; old_full_checks += 6
    assert checks == 54 and display_checks == 27 and old_full_checks == 18
    paired = build.read(HERE / 'paired_differences.json')
    bindings = build.read(HERE / 'identity_bindings.json')['bindings']
    assert len(paired['records']) == len(paired['checkpoint_pairs']) == len(bindings) == 10
    paired_checks = 0
    for row, pair, binding in zip(sorted(paired['records'], key=lambda r: r['seed']), paired['checkpoint_pairs'], bindings):
        seed = row['seed']; full, control = indexed['Full', seed], indexed['minus_C', seed]
        assert row['variant'] == 'minus_C' and row['distribution'] == 'IID' and row['attack'] == 'Benign'
        for metric in build.METRICS:
            assert row[metric] == control[metric] - full[metric]; paired_checks += 1
        assert pair['id'] == binding['id'] == control['id'] and pair['Full_id'] == binding['Full_id'] == full['id']
        assert pair['checkpoint_sha256'] == binding['checkpoint_sha256'] == control['checkpoint_sha256']
        assert pair['Full_checkpoint_sha256'] == binding['Full_checkpoint']['sha256'] == full['checkpoint_sha256']
        assert binding['Full_training_torch'] == binding['C_training_torch'] == '2.11.0+cu128'
        assert binding['data_contract']['image_data_contract']['evaluation_split'] == 'valid'
        assert binding['acceptance']['pass'] and binding['acceptance']['checkpoint_sha256'] == control['checkpoint_sha256']
    coverage = build.read(HERE / 'coverage.json')
    assert coverage['original_record_count'] == 212 and coverage['minus_C_count'] == 12 and coverage['minus_U_count'] == 100
    assert len(coverage['accepted_new112_ids']) == 112 and len(coverage['Full100_ids']) == 100
    assert coverage['panel_paired_counts'] == [10, 9, 6]
    assert sorted(len(r['seeds']) for r in coverage['minus_C_scenes']) == [0] * 8 + [2, 10]
    refusals = []
    def remove_c(j): j['records'].pop(next(n for n, r in enumerate(j['records']) if r['variant'] == 'minus_C' and r['attack'] == 'Benign'))
    def wrong_seed(j): next(r for r in j['records'] if r['variant'] == 'minus_C')['seed'] = 92001
    def bad_checkpoint(j): next(r for r in j['records'] if r['variant'] == 'minus_C')['checkpoint_sha256'] = ''
    mutations = [('missing_C_seed', remove_c), ('duplicate_record', lambda j: j['records'].append(copy.deepcopy(j['records'][0]))),
        ('foreign_seed', wrong_seed), ('erase_other_C_partial', lambda j: j.__setitem__('records', [r for r in j['records'] if not (r['variant'] == 'minus_C' and r['attack'] == 'F Flip')])),
        ('wrong_evidence_source', lambda j: j.__setitem__('source_script_sha256', '0' * 64)), ('missing_checkpoint_identity', bad_checkpoint)]
    for name, mutate in mutations:
        bad = copy.deepcopy(inspection); mutate(bad)
        try: build.validate_scope(bad)
        except AssertionError: refusals.append(name)
        else: raise AssertionError('Must refuse ' + name)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result = dict(status='NATIVE_C_BENIGN10_TABLE_IDENTITIES_AND_INDEPENDENT_STATISTICS_PASS', original_records=212, displayed_records=20, paired_seeds=10,
        panels=3, table_rows=9, display_mean_sd_cells=display_checks, scalar_mean_sd_checks=checks, maximum_abs_difference=maximum, per_seed_delta_checks=paired_checks,
        checkpoint_pairs_verified=10, prior_Full_statistic_scalars_exact=old_full_checks, prior204_records_unchanged=True,
        input_source_files_before_after_sha_unchanged=len(basis['files']), refusals=refusals, new_CNN=0, new_training=0, test=False,
        other_C_scenes_complete=False, significance_test=False, primary_endpoint='PENDING_AUTHOR', whole_mechanism900_complete=False)
    build.save('verification.json', result)
    print(__import__('json').dumps(result))


if __name__ == '__main__': main()
