"""Render only completed paired scenes from an accepted mechanism snapshot."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check_table_precision(actual, saved, differences):
    # Linux/Windows statistics.stdev can differ at the final binary ULP.
    # Check ten decimal places, beyond every displayed table precision.
    assert type(actual) is type(saved)
    if isinstance(actual, dict):
        assert actual.keys() == saved.keys()
        for key in actual:
            check_table_precision(actual[key], saved[key], differences)
    elif isinstance(actual, list):
        assert len(actual) == len(saved)
        for left, right in zip(actual, saved):
            check_table_precision(left, right, differences)
    elif isinstance(actual, float):
        assert f'{actual:.10f}' == f'{saved:.10f}'
        differences.append(abs(actual - saved))
    else:
        assert actual == saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inspection', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = Path(__file__).parent / 'celeba_mechanism_evidence_20261009/evidence_v4.py'
    assert sha(source) == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
    assert sha(args.inspection) == args.inspection.with_suffix('.sha256').read_text().strip()
    inspection = read(args.inspection)
    assert not inspection['invalid']
    spec = importlib.util.spec_from_file_location('original_mechanism_evidence', source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    rows = inspection['records']
    summary = original.summarize(rows)
    saved = read(args.inspection.with_name('statistics.json'))
    roundoff = []
    check_table_precision(summary['per_scene'], saved['per_scene'], roundoff)
    check_table_precision(summary['paired_per_scene'], saved['paired_per_scene'], roundoff)
    complete = [row for row in summary['paired_per_scene'] if row['complete']]
    assert complete, 'No complete ten-seed paired scene yet'
    assert not args.output.exists(), 'Keep earlier snapshots immutable'
    args.output.mkdir(parents=True)
    panels = []
    text = ['# CelebA mechanism ablation: interim accepted scenes', '',
            'Validation-only terminal-round results; this is an interim extraction, not the completed 900-record mechanism comparison.', '',
            'Each displayed scene contains all ten declared paired seeds (91001–91010). '
            'The two additional panels apply the same rule to every method: omit selection seed 91001, '
            'or retain seeds 91005–91010. These previously observed validation seeds are not an untouched confirmation set.', '']
    for label, seeds in [('All 10 seeds', list(range(91001, 91011))),
                         ('Exclude selection seed: 9 seeds', list(range(91002, 91011))),
                         ('Seeds 91005–91010: 6 seeds', list(range(91005, 91011)))]:
        text += ['## ' + label, '', '| Distribution | Scenario | Variant | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |',
                 '|---|---|---|---:|---:|---:|---:|']
        panel = {'label': label, 'seeds': seeds, 'rows': []}
        for scene in complete:
            dist, attack, variant = (scene[key] for key in ('distribution', 'attack', 'variant'))
            selected = [row for row in rows if row['distribution'] == dist and row['attack'] == attack
                        and row['variant'] in ('Full', variant) and row['seed'] in seeds]
            by_variant = {}
            for name in ('Full', variant):
                records = [row for row in selected if row['variant'] == name]
                assert {row['seed'] for row in records} == set(seeds)
                by_variant[name] = {row['seed']: row for row in records}
                stats = original.statistic(records, len(seeds))
                panel['rows'].append(dict(distribution=dist, attack=attack, variant=name, **stats))
                values = [f"{stats[m]['mean']:.3f} ± {stats[m]['sample_sd_ddof1']:.3f}"
                          if m == 'accuracy_pct' else f"{stats[m]['mean']:.5f} ± {stats[m]['sample_sd_ddof1']:.5f}"
                          for m in original.METRICS]
                text.append(f'| {dist} | {attack} | {name} | {len(seeds)} | ' + ' | '.join(values) + ' |')
            differences = [dict(seed=seed, **{metric: by_variant[variant][seed][metric] - by_variant['Full'][seed][metric]
                                           for metric in original.METRICS}) for seed in seeds]
            delta = original.statistic(differences, len(seeds))
            panel['rows'].append(dict(distribution=dist, attack=attack, variant=variant + ' minus Full', **delta))
        text += ['', 'Values are mean ± sample SD (ddof=1). Paired differences are retained in the accompanying JSON; '
                 'ACC differences use percentage points. No hypothesis test or superiority claim is made.', '']
        panels.append(panel)
    text += ['AEOD is the absolute TPR gap, not full equalized odds. '
             'Full reuses historical terminal checkpoints; controls were trained in the current cu128 environment. '
             'Driver differences remain a limitation even where the PyTorch build matches. '
             'Reported native metrics include each procedure’s original calibration; these tables do not isolate aggregation from calibration.', '',
             'Completed paired scenes are included by coverage, regardless of which variant wins. '
             'Incomplete scenes and other variants remain pending. Lower disparity after a deletion is retained; '
             'these interim results do not establish that every component is indispensable.', '',
             'Source inspection SHA256: `' + sha(args.inspection) + '`.',
             'Original evidence tool SHA256: `' + sha(source) + '`.', '']
    (args.output / 'TABLES.md').write_text('\n'.join(text), encoding='utf-8')
    result = dict(status='INTERIM_COMPLETE_SCENES_ONLY_NO_NEW_INFERENCE', inspection_sha256=sha(args.inspection),
                  evidence_source_sha256=sha(source), manifest_sha256=inspection['manifest_sha256'],
                  accepted_new_ids=inspection['accepted_new_ids'], reused_full_ids=inspection['accepted_reused_ids'],
                  complete_paired_scenes=len(complete), panels=panels, new_training=0, new_inference=0, test_used=False,
                  cross_platform_summary_check=dict(decimal_places=10, max_binary_roundoff=max(roundoff, default=0),
                                                   original_native_acceptance_tolerance_unchanged=True))
    (args.output / 'tables.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(complete_paired_scenes=len(complete), panel_count=len(panels), output=str(args.output))))


if __name__ == '__main__':
    main()
