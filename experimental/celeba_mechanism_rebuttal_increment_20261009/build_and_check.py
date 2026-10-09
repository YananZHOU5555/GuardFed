"""Local prose/data extraction only. No metric fitting, CNN, labels, or network."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = REPO / 'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009'
TRAIN = REPO / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1'
THREE = TRAIN / 'three_view_interim_20261009T145900Z'
NATIVE = TRAIN / 'interim_tables_20261009T144558Z'
GATE = 'DO_NOT_SUBMIT_BEFORE_FULL_COHORT'
PINS = {
    'ROOT_REVIEW.json': 'd69255d2059ff6a7449b036ff1ef34225dcd2546c1a027699ba6bd8b35dea55c',
    'tables.json': '7aa7defb7703a83b2cf7af4c75049e01dfe4c673a27420ba5e53de3b9cb415fa',
    'records.json': 'e08c8eb19e7aa11ec0605b06c7f9984c9462e081053eb02038be0fb0d32fc09f',
    'TABLES.md': '5a5dc0550dba8a95741b9a9f5a23cd280c0ef4ca177811049391e31e6fa2c9eb',
}
METRICS = ('accuracy_pct', 'aeod', 'aspd')
OUTPUTS = ('candidate_replies.en.md', 'candidate_replies.zh.md', 'MANUSCRIPT_INSERT.md',
           'evidence_excerpt.md', 'numeric_references.json', 'comment_alignment.json',
           'source_map.json', 'verification.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(name, value):
    target = HERE / name
    if target.exists():
        raise FileExistsError(f'No overwrite: {target}')
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    target.write_text(text, encoding='utf-8', newline='\n')


def link(path):
    return path.as_posix()


def main():
    for name in OUTPUTS:
        if (HERE / name).exists():
            raise FileExistsError(f'Use the existing frozen output, not a rebuild: {name}')
    for name, digest in PINS.items():
        if sha(THREE / name) != digest:
            raise ValueError(f'Accepted input drift: {name}')
    root, tables, stored = read(THREE / 'ROOT_REVIEW.json'), read(THREE / 'tables.json'), read(THREE / 'records.json')
    records, panels = stored['records'], tables['panels']
    comments = read(OLD / 'comment_source_map.json')
    if comments['comment_count'] != 24 or root['status'] != 'ROOT_SIX_SCENE_THREE_VIEW_PAIRED_SAVED_RECEIPTS_AND_STATISTICS_PASS':
        raise ValueError('Wrong original comments or unaccepted scientific snapshot')
    if (len(records), tables['complete_scenes'], tables['paired_checkpoints']) != (120, 6, 60):
        raise ValueError('Wrong interim denominator')
    if root['native_tolerance'] != 1e-12 or not root['all_original_identity_and_native_tolerance_checks_reused']:
        raise ValueError('Original strict/native acceptance is not pinned')
    for variant, expected in [('Full', {'cpu': 5, 'cuda:0': 55}), ('minus_U', {'cpu': 60})]:
        if dict(Counter(r['replay_runtime']['device'] for r in records if r['variant'] == variant)) != expected:
            raise ValueError('Displayed replay-device disclosure drift')
    if dict(Counter(r['training_torch'] for r in records if r['variant'] == 'Full')) != {'2.11.0+cu128': 59, '2.11.0+cu130': 1}:
        raise ValueError('Full training-runtime disclosure drift')
    if dict(Counter(r['training_torch'] for r in records if r['variant'] == 'minus_U')) != {'2.11.0+cu128': 60}:
        raise ValueError('Control training-runtime disclosure drift')
    mixed = [r for r in records if r['training_torch'] == '2.11.0+cu130']
    if [(r['distribution'], r['attack'], r['seed']) for r in mixed] != [('non-IID', 'Benign', 91001)]:
        raise ValueError('cu130 identity disclosure drift')
    if [x['seeds'] for x in tables['seed_panels']] != [list(range(91001, 91011)), list(range(91002, 91011)), list(range(91005, 91011))]:
        raise ValueError('Seed-panel drift')
    for row in records:
        if row['views']['native'] != row['views']['shared_calibration']:
            raise ValueError('Native/shared equality is not supported by this record')
        if set(row['views']) != {'native', 'raw', 'shared_calibration'} or not row['checkpoint_sha256']:
            raise ValueError('Missing same-checkpoint three-view provenance')
        if row['data_contract']['actual_evaluation_rows'] != 19867:
            raise ValueError('Wrong evaluation split')
    all10 = [(i, p) for i, p in enumerate(panels) if p['seeds'] == list(range(91001, 91011))]
    if len(all10) != 3 or any(len(p['rows']) != 18 for _, p in all10):
        raise ValueError('Incomplete main table')
    native = next(p for _, p in all10 if p['view'] == 'native')
    deltas = [r for r in native['rows'] if r['variant'] == 'minus_U minus Full']
    acc_deltas = [r['accuracy_pct']['mean'] for r in deltas]
    if f'{-max(acc_deltas):.3f}' != '0.309' or f'{-min(acc_deltas):.3f}' != '1.384':
        raise ValueError('Requested ACC range is not the accepted n10 range')
    directions = {}
    for _, panel in all10:
        ds = [r for r in panel['rows'] if r['variant'] == 'minus_U minus Full']
        directions[panel['view']] = {m: sum(r[m]['mean'] < 0 for r in ds) for m in METRICS}
    if directions != {'native': {'accuracy_pct': 6, 'aeod': 1, 'aspd': 6}, 'raw': {'accuracy_pct': 6, 'aeod': 5, 'aspd': 6}, 'shared_calibration': {'accuracy_pct': 6, 'aeod': 1, 'aspd': 6}}:
        raise ValueError('Prose direction count is not supported')
    native7 = read(NATIVE / 'tables.json')
    native_overlap_rows = 0
    for panel in (p for p in panels if p['view'] == 'native'):
        oldpanel = next(p for p in native7['panels'] if p['seeds'] == panel['seeds'])
        index = {(r['distribution'], r['attack'], r['variant']): r for r in oldpanel['rows']}
        for row in panel['rows']:
            if row != index[(row['distribution'], row['attack'], row['variant'])]:
                raise ValueError('Overlapping native table differs from accepted three-view cells')
            native_overlap_rows += 1
    old_response = (OLD / 'rebuttal_20261009.md').read_text(encoding='utf-8-sig')
    if '280 records including Full controls' not in old_response or 'all 12 deletion conditions have higher mean accuracy than Full' not in old_response:
        raise ValueError('Original tabular/negative evidence is not the expected version')
    refs = []
    excerpt = ['# Accepted three-view evidence excerpt', '', f'**{GATE} — interim author-review material.**', '',
        'This is a direct extraction of existing summary cells, with no recalculation of scientific metrics or statistics. Full-precision numbers and source JSON pointers are in [numeric_references.json](numeric_references.json).', '',
        'Six scenes; 60 minus_U checkpoints and 60 matched historical Full checkpoints. n=10 shared seeds 91001–91010 in every row. Values are mean ± sample SD (ddof=1); Δ is minus_U−Full. ACC is %, ΔACC is percentage points, and both gaps are on [0,1]. AEOD is the absolute TPR gap, not full equalized odds.', '',
        f'The complete equal-rule n=10/9/6 panels remain in the [accepted table]({link(THREE / "TABLES.md")}). The nine-seed panel excludes 91001; the six-seed panel retains 91005–91010. Neither is an untouched confirmation set. Directions discussed in the replies refer only to the n=10 panel.', '']
    for pi, panel in all10:
        excerpt += [f'## {panel["view"]}', '', '| Distribution | Scenario | Procedure / paired difference | n | ACC (%) / ΔACC (pp) | AEOD / ΔAEOD | ASPD / ΔASPD |', '|---|---|---|---:|---:|---:|---:|']
        for ri, row in enumerate(panel['rows']):
            if not row['complete'] or row['n'] != 10:
                raise ValueError('Incomplete row cannot appear in the main excerpt')
            cells = []
            for metric, precision in zip(METRICS, (3, 5, 5)):
                value = row[metric]
                token = f'{value["mean"]:.{precision}f} ± {value["sample_sd_ddof1"]:.{precision}f}'
                cells.append(token)
                refs.append({'view': panel['view'], 'distribution': row['distribution'], 'attack': row['attack'], 'variant': row['variant'], 'metric': metric, 'n': 10, 'mean': value['mean'], 'sample_sd_ddof1': value['sample_sd_ddof1'], 'rendered': token, 'source_json_pointer': f'/panels/{pi}/rows/{ri}/{metric}'})
            excerpt.append(f'| {row["distribution"]} | {row["attack"]} | {row["variant"]} | 10 | ' + ' | '.join(cells) + ' |')
        excerpt.append('')
    excerpt += ['Native and shared-calibration saved metrics and group confusion-count dictionaries are identical for all 120 displayed checkpoints. This is not a newly verified prediction-vector equality claim, and gives no independent calibration-gain evidence.', '',
        'Displayed Full inference uses 5 CPU and 55 GPU checkpoints; all 60 minus_U checkpoints use CPU. Historical Full training comprises 59 cu128 and one cu130 checkpoint (non-IID Benign, seed91001); minus_U training uses cu128 on the current driver595 environment. Driver equality was not established. This is not a uniform-device final fairness comparison.', '',
        f'The [separate native-only seven-scene table]({link(NATIVE / "TABLES.md")}) also has non-IID F Flip. That seventh scene is not inserted into these three-view tables or their claims.', '']
    values = {'ACC_MIN': f'{-max(acc_deltas):.3f}', 'ACC_MAX': f'{-min(acc_deltas):.3f}',
              'EVIDENCE_TABLE': link(HERE / 'evidence_excerpt.md'), 'SOURCE_TABLE': link(THREE / 'TABLES.md'),
              'NATIVE_TABLE': link(NATIVE / 'TABLES.md')}
    benign = {r['variant']: r for r in native['rows'] if (r['distribution'], r['attack']) == ('IID', 'Benign')}
    for variant, prefix in [('Full', 'FULL'), ('minus_U', 'DELETE')]:
        for metric, precision, key in zip(METRICS, (3, 5, 5), ('ACC', 'AEOD', 'ASPD')):
            values[f'{prefix}_{key}'] = f'{benign[variant][metric]["mean"]:.{precision}f}'
    template_checks = {}
    for template, output in [('replies.en.template.md', 'candidate_replies.en.md'), ('replies.zh.template.md', 'candidate_replies.zh.md'), ('manuscript.template.md', 'MANUSCRIPT_INSERT.md')]:
        text = (HERE / template).read_text(encoding='utf-8')
        keys = re.findall(r'\{\{([A-Z_]+)\}\}', text)
        for key in keys:
            text = text.replace('{{' + key + '}}', values[key])
        if '{{' in text or GATE not in text:
            raise ValueError('Unresolved number/link or missing author-review gate')
        write(output, text)
        template_checks[output] = keys
    write('evidence_excerpt.md', '\n'.join(excerpt))
    rendered_rows = [line.split('|')[1:-1] for line in (HERE / 'evidence_excerpt.md').read_text(encoding='utf-8').splitlines() if line.startswith('| IID |') or line.startswith('| non-IID |')]
    if len(rendered_rows) != 54:
        raise ValueError('Rendered main-table row count drift')
    for i, row in enumerate(rendered_rows):
        if [cell.strip() for cell in row[4:]] != [r['rendered'] for r in refs[3*i:3*i+3]]:
            raise ValueError('Rendered number differs from its source cell')
    write('numeric_references.json', {'source_path': link(THREE / 'tables.json'), 'source_sha256': sha(THREE / 'tables.json'), 'main_panel_cells': refs, 'prose_tokens': values, 'direction_checks_n10_only': directions, 'new_scientific_metrics': False, 'new_mean_SD': False})
    rebuttal_lines = (OLD / 'rebuttal_20261009.md').read_text(encoding='utf-8-sig').splitlines()
    locations = {'AE': (35, 45, 'Append after response paragraph at line43; retain original positioning/theory text.'), 'R3.2': (245, 255, 'Replace only line255 final response paragraph; retain lines251–253 and original comment.'), 'R3.7': (295, 303, 'Append after line303; retain all original COMPAS negative results and calibration reversal.'), 'P2': (347, 356, 'Replace only pending-row line352; retain the pending status and other P1/P3–P6 rows.')}
    alignment = {}
    for key, (start, end, action) in locations.items():
        comment = comments['comments'].get(key)
        alignment[key] = {'source_path': link(OLD / 'rebuttal_20261009.md'), 'source_sha256': sha(OLD / 'rebuttal_20261009.md'), 'start_line': start, 'end_line': end, 'original_excerpt': '\n'.join(rebuttal_lines[start-1:end]), 'original_comment': comment, 'comment_origin': 'verbatim original reviewer/AE comment' if comment else 'internal pending-register item; not an original reviewer comment', 'integration_action': action}
    alignment['manuscript'] = {'source_path': link(OLD / 'manuscript_insertions_20261009.md'), 'source_sha256': sha(OLD / 'manuscript_insertions_20261009.md'), 'insert_after_line': 139, 'action': 'Append image-specific interim paragraph after the existing tabular ablation paragraph; do not replace COMPAS evidence.'}
    write('comment_alignment.json', {'original_comment_count': 24, 'original_comment_source_sha256': comments['source_sha256'], 'targets': alignment, 'canonical_edits_performed': False})
    files = [(OLD / n, 'original sealed rebuttal/comment/insertion material') for n in ['rebuttal_20261009.md', 'comment_source_map.json', 'reviewer_comments_verbatim.md', 'manuscript_insertions_20261009.md']]
    files += [(THREE / n, 'root-accepted six-scene same-checkpoint three-view snapshot') for n in ['ROOT_REVIEW.json', 'TABLES.md', 'tables.json', 'records.json', 'paired_per_seed.json', 'INPUTS_SHA256.json', 'NUMERIC_CHECKS.json']]
    files += [(NATIVE / n, 'separate seven-scene native-only scope cross-check; seventh scene excluded here') for n in ['TABLES.md', 'tables.json']]
    files += [(TRAIN / 'PROTOCOL.md', 'original intervention definition only; its prepared status is not current execution evidence')]
    write('source_map.json', {'status': 'READ_ONLY_FIXED_INPUTS_FOR_INTERIM_PROSE', 'files': [{'path': link(f), 'sha256': sha(f), 'bytes': f.stat().st_size, 'role': role} for f, role in files], 'upstream_302_path_SHA_chain': link(THREE / 'INPUTS_SHA256.json'), 'root_review_sha256': PINS['ROOT_REVIEW.json'], 'new_input_models_or_archives_copied': False})
    links = []
    for name in ['candidate_replies.en.md', 'candidate_replies.zh.md', 'MANUSCRIPT_INSERT.md', 'evidence_excerpt.md']:
        text = (HERE / name).read_text(encoding='utf-8')
        for value in re.findall(r'\]\(([^)]+)\)', text):
            dest = Path(value) if re.match(r'^[A-Z]:/', value) else HERE / value
            if not dest.is_file():
                raise ValueError(f'Broken local evidence link: {value}')
            links.append({'document': name, 'target': value})
    write('verification.json', {'status': 'PASS_LOCAL_PROSE_NUMBERS_SCOPE_AND_LINKS', 'accepted_root_proof_SHA': True, 'accepted_tables_records_SHA': True, 'original_comments_n': 24, 'actual_same_checkpoint_three_view_records': 120, 'native_shared_exact_saved_view_dictionaries': 120, 'main_table_mean_SD_cells_copied_without_recomputation': len(refs), 'main_table_rendered_numbers': 2 * len(refs), 'native_seven_scene_overlap_exact_summary_rows': native_overlap_rows, 'source_panels_seed_rules_checked': 3, 'actual_device_torch_counts_and_cu130_ID_checked': True, 'n10_direction_checks': directions, 'native_ACC_loss_range_pp': [values['ACC_MIN'], values['ACC_MAX']], 'template_substitutions': template_checks, 'local_links_checked': links, 'preserved_original_tabular_matrix_n': 280, 'COMPAS_original_text_preserved_by_add_only_R3_7': True, 'seventh_native_scene_excluded_from_three_view_claims': True, 'DO_NOT_SUBMIT_BEFORE_FULL_COHORT': True, 'new_CNN': False, 'new_training': False, 'new_scientific_metric_or_statistic': False, 'network': False, 'test_labels_read': False, 'old_package_STATE_RUNNING_Git_modified': False})
    print(json.dumps({'status': 'PASS', 'main_table_metric_cells': len(refs), 'records': len(records), 'output': link(HERE)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
