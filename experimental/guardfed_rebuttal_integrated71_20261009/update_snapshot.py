"""Minimal prose snapshot update. Copies accepted cells; computes no statistics."""
import difflib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
REPO = HERE.parents[1]
PRIOR = REPO / 'tmp/guardfed_rebuttal_integrated_20261009'
TRAIN = REPO / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1'
SNAP = TRAIN / 'three_view_interim71_20261009'
PINS = {'ROOT_REVIEW.json': 'c616d97c9eb6ad0112773925f563791ad9d9154261f7258acadc9c331a754281',
        'TABLES.md': 'a91e9fbeb32b54103069f07b51d1e771b1a21bc16e1d7f367774add4df973ac3',
        'tables.json': '8e0ba25c50e91513e72e6d6e1e546158888141b0c7c0ab0f952ab6ef415f11a4',
        'records.json': 'b8d6f912579a322dbb651680cc0e7acfaf2a621699e6b7f4cc05661a9c6ff2d4'}


def write(name, value):
    target = HERE / name
    if target.exists():
        raise FileExistsError(f'No overwrite: {target}')
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    target.write_text(value, encoding='utf-8', newline='\n')


def main():
    # The original integrator supplies only text/hash/link helpers; its main() is not run.
    import hashlib
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    load = lambda f: json.loads(f.read_text(encoding='utf-8-sig'))
    if sha(PRIOR / 'FILES_SHA256.json') != 'ee8147a759bdccf548094b49b52a13570ae142df3ca5abdda6ae8e07fe321881':
        raise ValueError('Prior six-scene writing seal drift')
    for member in load(PRIOR / 'FILES_SHA256.json')['members']:
        if sha(PRIOR / member['path']) != member['sha256']:
            raise ValueError('Prior writing member drift')
    for name, digest in PINS.items():
        if sha(SNAP / name) != digest:
            raise ValueError(f'Accepted seven-scene input drift: {name}')
    spec = importlib.util.spec_from_file_location('prior_integrator', PRIOR / 'integrate.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    proof, tables, saved = [load(SNAP / n) for n in ['ROOT_REVIEW.json', 'tables.json', 'records.json']]
    if (proof['complete_paired_scenes'], proof['paired_checkpoints'], proof['preserved_pairs'], len(saved['records'])) != (7, 70, 71, 142):
        raise ValueError('Wrong complete/preserved scope')
    old_table = load(TRAIN / 'three_view_interim_20261009T145900Z/tables.json')
    old_records = load(TRAIN / 'three_view_interim_20261009T145900Z/records.json')['records']
    new_index = {r['id']: r for r in saved['records']}
    if any(new_index[r['id']] != r for r in old_records):
        raise ValueError('Original six-scene records changed')
    overlap = 0
    for old in old_table['panels']:
        panel = next(p for p in tables['panels'] if (p['view'], p['seeds']) == (old['view'], old['seeds']))
        index = {(r['distribution'], r['attack'], r['variant']): r for r in panel['rows']}
        for row in old['rows']:
            if index[(row['distribution'], row['attack'], row['variant'])] != row:
                raise ValueError('Original six-scene summary changed')
            overlap += 1
    all10 = [(i, p) for i, p in enumerate(tables['panels']) if p['seeds'] == list(range(91001, 91011))]
    if len(all10) != 3 or any(len(p['rows']) != 21 for _, p in all10):
        raise ValueError('Incomplete seven-scene main panels')
    refs, dirs = [], {}
    for pi, panel in all10:
        for ri, row in enumerate(panel['rows']):
            if row['n'] != 10 or not row['complete']:
                raise ValueError('Incomplete scene included in means')
            for metric in ('accuracy_pct', 'aeod', 'aspd'):
                refs.append({'view': panel['view'], 'distribution': row['distribution'], 'attack': row['attack'], 'variant': row['variant'], 'metric': metric,
                             'source_json_pointer': f'/panels/{pi}/rows/{ri}/{metric}', **row[metric]})
        ds = [r for r in panel['rows'] if r['variant'] == 'minus_U minus Full']
        dirs[panel['view']] = {m: sum(r[m]['mean'] < 0 for r in ds) for m in ('accuracy_pct', 'aeod', 'aspd')}
    if dirs != {'native': {'accuracy_pct': 7, 'aeod': 1, 'aspd': 7}, 'raw': {'accuracy_pct': 7, 'aeod': 5, 'aspd': 7}, 'shared_calibration': {'accuracy_pct': 7, 'aeod': 1, 'aspd': 7}}:
        raise ValueError('Seven-scene directions not supported')
    native = next(p for _, p in all10 if p['view'] == 'native')
    delta = next(r for r in native['rows'] if (r['distribution'], r['attack'], r['variant']) == ('non-IID', 'F Flip', 'minus_U minus Full'))
    extra = (f'For the added non-IID F Flip scene, the mean paired differences (minus_U−Full) are {delta["accuracy_pct"]["mean"]:.6f} percentage points in ACC, '
             f'{delta["aeod"]["mean"]:+.9f} in AEOD and {delta["aspd"]["mean"]:.9f} in ASPD. ')
    complete = {(r['distribution'], r['attack']) for r in native['rows']}
    displayed = [r for r in saved['records'] if (r['distribution'], r['attack']) in complete]
    if len(displayed) != 140 or any(r['views']['native'] != r['views']['shared_calibration'] for r in displayed):
        raise ValueError('Displayed native/shared140 equality not supported')
    excluded = [r for r in saved['records'] if (r['distribution'], r['attack']) not in complete]
    if sorted((r['variant'], r['distribution'], r['attack'], r['seed']) for r in excluded) != [('Full', 'non-IID', 'FedSA', 91001), ('minus_U', 'non-IID', 'FedSA', 91001)]:
        raise ValueError('Incomplete-pair exclusion changed')
    if tables['replay_devices'] != {'Full': {'cpu': 5, 'cuda:0': 65}, 'minus_U': {'cpu': 70}} or tables['training_torch'] != {'Full': {'2.11.0+cu128': 69, '2.11.0+cu130': 1}, 'minus_U': {'2.11.0+cu128': 70}}:
        raise ValueError('Runtime/device counts drift')
    note = ('**DO_NOT_SUBMIT_BEFORE_FULL_COHORT. Integrated author-review copy; fixed intermediate evidence snapshot.** '
            f'This copy uses only the closed seven-scene snapshot `three_view_interim71_20261009`, accepted at {proof["checked_utc"]} '
            f'in the [independent root review]({(SNAP / "ROOT_REVIEW.json").as_posix()}). The [accepted three-view table]({(SNAP / "TABLES.md").as_posix()}) '
            'contains 70 complete matched seed pairs (140 displayed checkpoints), with equal-rule 10/9/6-seed panels. '
            'A total of 71 pairs/142 original receipts are preserved; the additional non-IID FedSA seed91001 pair is retained individually and excluded from mean/SD panels because its scene lacks ten seeds. '
            'The original draft\'s evidence cut-off and completed-cohort descriptions retain their historical meaning. '
            'This version supersedes the six-scene writing copy without changing its evidence. Later results are not incorporated without explicit acceptance and a new revision. '
            'P2 and the native/shared main endpoint remain pending. Completion of a 900-model validation replay would not itself complete the mechanism cohort or a final test evaluation.')
    old_path = (TRAIN / 'three_view_interim_20261009T145900Z/TABLES.md').as_posix()
    def update(paragraph):
        if paragraph.startswith('**DO_NOT_SUBMIT_BEFORE_FULL_COHORT.'):
            return note
        target = any(paragraph.startswith(x) for x in ['The new image mechanism', 'The 280-record tabular', 'The intermediate image comparison', '**Interim image comparability.**', '**Image mechanism evidence at', '**Comparison boundary.**']) or paragraph.startswith('| Item |') and '| P2 — CelebA mechanisms;' in paragraph
        if not target:
            return paragraph
        text = paragraph.replace(old_path, (SNAP / 'TABLES.md').as_posix())
        replacements = {'six complete scenes': 'seven complete scenes', 'six scenes': 'seven scenes', 'all six': 'all seven', 'five scenes': 'six scenes', 'higher in five and': 'higher in six and',
                        'all five IID scenarios and non-IID Benign': 'all five IID scenarios plus non-IID Benign and F Flip',
                        'the five IID scenarios and non-IID Benign': 'the five IID scenarios plus non-IID Benign and F Flip',
                        'and non-IID Benign.': 'and non-IID Benign and F Flip.',
                        'five CPU and 55 GPU': 'five CPU and 65 GPU', '59 cu128': '69 cu128'}
        for a, b in replacements.items():
            text = text.replace(a, b)
        text = text.replace("The same checkpoints' raw outputs instead show lower AEOD after deletion in six scenes.", "The same checkpoints' raw outputs instead show lower AEOD after deletion in five scenes.")
        text = text.replace('Raw AEOD is lower in six scenes and higher for IID Sp-DFA.', 'Raw AEOD is lower in five scenes and higher for IID Sp-DFA and non-IID F Flip.')
        text = text.replace('The raw outputs instead have lower AEOD after deletion in six scenes.', 'The raw outputs instead have lower AEOD after deletion in five scenes, and higher AEOD for IID Sp-DFA and non-IID F Flip.')
        text = re.sub(r'\b60\b', '70', text)
        text = re.sub(r'\b120\b', '140', text)
        if paragraph.startswith('The intermediate image comparison'):
            text = text.replace('Thus retaining U', extra + 'Thus retaining U')
        if paragraph.startswith('**Interim image comparability.**'):
            text = text.split(' The [separate seven-scene native table]', 1)[0]
            text += ' All seven complete scenes are included by coverage; the additional non-IID FedSA seed91001 pair is preserved individually and excluded from the mean/SD panels.'
        return text
    names = ['rebuttal_integrated_20261009.md', 'manuscript_insertions_integrated_20261009.md']
    outputs, unchanged, changes = {}, {}, {}
    for name in names:
        old = helper.read(PRIOR / name)
        chunks = re.split(r'(\n\s*\n)', old)
        ledger, edits = [], []
        for i in range(0, len(chunks), 2):
            before, after = chunks[i], update(chunks[i])
            chunks[i] = after
            ledger.append({'paragraph': i//2+1, 'old_sha256': helper.sha_bytes(before.encode()), 'new_sha256': helper.sha_bytes(after.encode()), 'unchanged': before == after})
            if before != after:
                edits.append({'paragraph': i//2+1, 'starts_with': before[:100]})
        result = ''.join(chunks)
        pattern = r'\*\*Original comment \(verbatim\)\.\*\*\n\n(.*?)\n\n\*\*Response\.\*\*'
        old_comments, new_comments = re.findall(pattern, old, re.S), re.findall(pattern, result, re.S)
        if old_comments != new_comments or ('rebuttal' in name and len(new_comments) != 24):
            raise ValueError('Original comments changed')
        if 'rebuttal' in name:
            for item in ('P1', 'P3', 'P4', 'P5', 'P6'):
                line = next(line for line in old.splitlines() if line.startswith('| ' + item + ' —'))
                if line not in result:
                    raise ValueError('Original pending row changed')
            if '| P2 — CelebA mechanisms; still pending |' not in result:
                raise ValueError('P2 lost its pending status')
        for sentence in ['In the new COMPAS train-only protocol, all 12 deletion conditions have higher mean ACC than Full', 'On COMPAS non-IID, deleting F gives raw AEOD 0.250984 versus Full 0.239597'] if 'rebuttal' in name else ['In the new COMPAS train-only protocol, all 12 individual-deletion conditions have higher mean accuracy than Full']:
            if sentence not in result:
                raise ValueError('Original COMPAS negative evidence changed')
        outputs[name], unchanged[name], changes[name] = result, ledger, edits
    links = helper.check_links(outputs)
    for text in outputs.values():
        if 'in all six' in text or 'higher in five and lower only for IID Sp-DFA' in text or 'all 120 displayed' in text or old_path in text:
            raise ValueError('Stale six-scene claim/link remained')
    diff = ''.join(''.join(difflib.unified_diff(helper.read(PRIOR / n).splitlines(True), outputs[n].splitlines(True), fromfile='six_scene/' + n, tofile='seven_scene/' + n)) for n in names)
    for name, text in outputs.items():
        write(name, text)
    write('NUMERIC_REFERENCES.json', {'source_path': (SNAP / 'tables.json').as_posix(), 'source_sha256': PINS['tables.json'], 'n10_cells': refs, 'direction_checks_n10': dirs, 'added_nonIID_FFlip_delta_exact': {m: delta[m] for m in ('accuracy_pct', 'aeod', 'aspd')}, 'added_delta_prose': extra, 'new_statistics': False})
    write('UNCHANGED_PARAGRAPHS.json', unchanged)
    write('COMMENT_CONSISTENCY.json', {'status': '24_ORIGINAL_QUOTE_BLOCKS_VERBATIM_UNCHANGED_FROM_SEALED_FULL_COPY', 'original_comment_chain': (PRIOR / 'comment_consistency.json').as_posix(), 'source_sha256': sha(PRIOR / 'comment_consistency.json'), 'comments': load(PRIOR / 'comment_consistency.json')['comments']})
    write('LINK_CHECKS.json', {'status': 'LOCAL_FILES_EXIST_EXTERNAL_URL_SYNTAX_ONLY_NO_NETWORK', 'links': links})
    write('UPDATE_DIFF.patch', diff)
    inputs = [PRIOR / 'FILES_SHA256.json', PRIOR / 'integrate.py', PRIOR / 'comment_consistency.json'] + [PRIOR / n for n in names] + [SNAP / n for n in ['ROOT_REVIEW.json', 'TABLES.md', 'tables.json', 'records.json', 'INPUTS_SHA256.json', 'incomplete_pairs.json']]
    write('SOURCE_MAP.json', {'fixed_snapshot': SNAP.as_posix(), 'root_review_sha256': PINS['ROOT_REVIEW.json'], 'prior_six_scene_seal_sha256': sha(PRIOR / 'FILES_SHA256.json'), 'sources': [{'path': f.as_posix(), 'sha256': sha(f)} for f in inputs], 'explicit_paragraph_updates': changes, 'old_sources_and_evidence_modified': False})
    write('verification.json', {'status': 'PASS_MINIMAL_SEVEN_SCENE_WRITING_UPDATE', 'comments_verbatim': 24, 'preserved_receipts': 142, 'displayed_checkpoints': 140, 'complete_pairs': 70, 'complete_scenes': 7, 'incomplete_nonIID_FedSA91001_pair_excluded_from_means': True, 'native_shared_displayed_record_dicts_equal': 140, 'original_six_scene_record_dicts_exact': 120, 'original_six_scene_summary_rows_exact': overlap, 'n10_metric_cells_referenced': len(refs), 'root_reported_mean_SD_checks': 1134, 'new_statistics_computed': False, 'P1_to_P6_and_tabular280_COMPAS_negative_results_preserved': True, 'links': len(links), 'external_access_not_network_verified': True, 'submission_gate': 'DO_NOT_SUBMIT_BEFORE_FULL_COHORT', 'primary_endpoint_selected': False, 'final_test': False, 'network_CNN_training_Git_canonical_changes': False})
    print(json.dumps({'status': 'PASS', 'n10_metric_cells': len(refs), 'old_summary_rows_unchanged': overlap, 'links': len(links)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
