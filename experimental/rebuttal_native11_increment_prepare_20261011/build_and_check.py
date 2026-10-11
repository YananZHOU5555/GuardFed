"""Local text-only native11 increment; no statistics, inference or adoption."""
import difflib
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
PARENT = ROOT / 'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL90_Hybrid20_statistics_clarified_20261011.md'
BASE = ROOT / 'outputs/guardfed_tables/celeba_eleven_method_native_20261011'
NATIVE = ROOT / 'tmp/fl_native_final10_prepared_20261011/ROOT_ADOPTION_REVIEW.json'
OUTPUT = HERE / 'rebuttal_working_F80_FL90_Hybrid20_native11_candidate_20261011.md'
PARENT_SHA = 'c1d5978604256747ab1da0305936fc41fc3e0a27cb13e2919772128dd40c5a43'
ROOT_SHA = '69e9c479c061a1aa428157b6f1d121aa5dd72a48e0e6d1d16dd21ac296391d02'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    (HERE / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def main():
    assert not OUTPUT.exists(), 'Do not overwrite an existing candidate.'
    raw = PARENT.read_bytes()
    assert sha(raw) == PARENT_SHA
    root_raw = (BASE / 'ROOT_REVIEW.json').read_bytes()
    assert sha(root_raw) == ROOT_SHA
    adopted = json.loads(root_raw)
    assert (adopted['methods'], adopted['records'], adopted['seed_counts']) == (11, 1100, [10, 9, 6])
    for name, expected in adopted['files_sha256'].items():
        assert sha((BASE / name).read_bytes()) == expected, name
    native = json.loads(NATIVE.read_bytes())
    assert sha(NATIVE.read_bytes()) == adopted['native96_root']['sha256']
    assert (native['accepted_total'], native['reused_separately'], native['native_coverage_complete']) == (96, 4, True)
    aggregates = json.loads((BASE / 'seed_first_aggregates.json').read_bytes())
    scenes = json.loads((BASE / 'summary_statistics.json').read_bytes())
    table_url = 'E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md'
    old_table_url = 'E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_20261010/TABLES.md'
    text = raw.decode('utf-8')
    edits = []

    def replace(label, before, after):
        nonlocal text
        assert before != after and text.count(before) == 1, label
        text = text.replace(before, after, 1)
        edits.append({'label': label, 'before': before, 'after': after})

    replace('Opening: separate completed native100 from three-view90',
            'FLGMM native and three-view acceptance now cover the same 86 new coverage checkpoints plus four reused screen checkpoints.',
            'The separate native table now contains 1,100 records across eleven methods, including complete FLGMM native coverage of 96 new checkpoints plus four explicitly reused screen checkpoints. FLGMM three-view acceptance remains 90 checkpoints across nine complete scenes.')
    replace('AE: native11 addition preserves original900 three-view denominator',
            'Its three prediction views expose utility–disparity trade-offs and show that native disparity advantages cannot be attributed wholly to aggregation.',
            'Its three prediction views expose utility–disparity trade-offs and show that native disparity advantages cannot be attributed wholly to aggregation. A separate completed native-output table now contains 1,100 records across eleven methods, including LoGoFair and FLGMM; it does not enlarge the nine-method three-view cohort.')
    replace('R1.3: complete FLnative100 while three-view Sp-DFA remains outside',
            'The separately accepted FLGMM extension adds nine complete ten-seed scenes (R3.8); non-IID Sp-DFA remains incomplete. It is not a complete additional 100-record method and does not change the denominator of the nine-method summaries below.',
            'The separate native-output table now contains 1,100 records across eleven methods, including all 100 FLGMM checkpoints (96 new coverage runs and four explicitly reused screen runs). The FLGMM three-view extension still covers nine complete ten-seed scenes (R3.8); non-IID Sp-DFA remains outside that accepted three-view scope. Neither extension changes the denominator of the nine-method three-view summaries below.')
    comparison = ('In the completed eleven-method native table, the balanced ten-seed means are ACC 88.42%, AEOD 0.0097 and ASPD 0.0610 for GuardFed-AD2+, versus 89.35%, 0.0439 and 0.1119 for FLGMM. Each seed is averaged over all ten distribution–scenario cells before summarizing across seeds. GuardFed has lower mean AEOD and ASPD but lower mean accuracy than FLGMM in each of those ten cells. These descriptive directions do not establish significance or uniform superiority: FLTrust retains higher balanced mean accuracy than GuardFed, and LoGoFair retains lower balanced mean ASPD. The [eleven-method native table](' + table_url + ') reports all methods and the predefined 10/9/6-seed panels separately from the nine-method raw/native/shared comparison.')
    old = 'Thus the comparison shows lower disparity with lower accuracy, rather than universal superiority.'
    replace('R1.3: saved native11 balanced means and ten-scene trade-off', old, old + '\n\n' + comparison)
    replace('R2 benchmark: native1000 to1100 only',
            'A separate native-method table adds LoGoFair, yielding 1,000 records across ten methods; it does not expand the nine-method common-calibration comparison to ten methods.',
            'A separate native-method table adds LoGoFair and complete FLGMM coverage, yielding 1,100 records across eleven methods; it does not expand the nine-method common-calibration comparison to eleven methods.')
    replace('R2 uncertainty: current native11 reporting scope',
            'the separate ten-method native table uses the same reporting principle.',
            'the separate eleven-method native table uses the same reporting principle.')
    replace('R3.8: native11 and six unfinished native methods',
            'The completed CelebA comparison contains ten native methods across two distributions, five scenarios and ten model seeds (1,000 records). Nine of these methods also have a separate 900-record raw/native/shared comparison. These scopes are kept distinct. FairFed, FairGuard, their combination, FedAA-DDPG and LASA retain project-adaptation labels. Seven target methods still lack complete accepted coverage; a simplified inspired branch is not presented as the original algorithm.',
            'The completed CelebA native comparison contains eleven methods across two distributions, five scenarios and ten model seeds (1,100 records), including complete FLGMM native coverage. Nine of these methods also have a separate 900-record raw/native/shared comparison. These scopes are kept distinct. FairFed, FairGuard, their combination, FedAA-DDPG and LASA retain project-adaptation labels. Six target methods still lack complete accepted native coverage; a simplified inspired branch is not presented as the original algorithm.')
    replace('R3.8: native100 versus saved three-view90',
            'The [FLGMM extension](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_flgmm_nine_scenes90_20261011/TABLES.md) contains 86 newly accepted coverage checkpoints and four explicitly reused validation-screen checkpoints. All 90 have accepted saved three-view evidence and comprise nine complete ten-seed scenes:',
            'FLGMM native coverage is complete: 96 newly accepted coverage checkpoints plus four explicitly reused validation-screen checkpoints give 100 native records. Its [three-view extension](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_flgmm_nine_scenes90_20261011/TABLES.md) remains limited to 86 coverage checkpoints and the same four reused screen checkpoints. These 90 have accepted saved three-view evidence and comprise nine complete ten-seed scenes:')
    replace('R3.8: incomplete applies to three-view scope only',
            'Only non-IID Sp-DFA remains incomplete.',
            'Only non-IID Sp-DFA remains outside the accepted ten-seed three-view scope; its native results are included in the completed native table.')
    replace('R3.8: LoGo evidence links current native11 table',
            'The [native table](' + old_table_url + ') preserves the resulting large seed spread and unfavorable comparisons.',
            'The [eleven-method native table](' + table_url + ') preserves the resulting large seed spread, lower balanced mean ASPD than GuardFed and unfavorable utility comparisons.')
    replace('P1: six unfinished native methods; FL three-view gap remains',
            'Complete accepted coverage for the seven remaining target methods. FLGMM native and three-view acceptance now cover 86 new coverage checkpoints plus four reused screen checkpoints: 90 records in nine complete scenes, including non-IID S-DFA. Only non-IID Sp-DFA remains outside its complete accepted ten-seed three-view coverage. FLGMM and the cosine/fairness hybrid therefore still lack complete accepted fixed-recipe coverage;',
            'Complete accepted native coverage for the six remaining target methods. FLGMM native coverage is complete at 96 new coverage checkpoints plus four explicitly reused screen checkpoints (100 records). Its accepted three-view coverage remains 90 records in nine complete scenes, including non-IID S-DFA; non-IID Sp-DFA still lacks accepted ten-seed three-view coverage. The cosine/fairness hybrid still lacks complete accepted fixed-recipe coverage;')
    replace('Supporting material: current native11 table; historical native10 PDF',
            '- [Ten-method native IID/non-IID tables](' + old_table_url + '), with the [three-page paper-table PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf).',
            '- [Eleven-method native IID/non-IID tables](' + table_url + ') and [adopted 1,100-record scope](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_eleven_method_native_20261011/ROOT_REVIEW.json). The [ten-method, three-page PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf) remains a historical rendering.')

    candidate = text.encode('utf-8')
    inverse = text
    for edit in reversed(edits):
        assert inverse.count(edit['after']) == 1, edit['label']
        inverse = inverse.replace(edit['after'], edit['before'], 1)
    assert inverse.encode('utf-8') == raw
    old_lines, new_lines = raw.splitlines(keepends=True), candidate.splitlines(keepends=True)
    comments = lambda lines: [l for l in lines if l.startswith(b'### ') or l.strip() == b'## Associate Editor']
    quotes = lambda lines: [l for l in lines if l.startswith(b'>')]
    assert comments(old_lines) == comments(new_lines) and len(comments(old_lines)) == 24
    assert quotes(old_lines) == quotes(new_lines) and len(quotes(old_lines)) == 38
    old_text = raw.decode('utf-8')
    section = lambda t, a, b: t.split(a, 1)[1].split(b, 1)[0]
    assert section(old_text, '### R3.9 —', '### R3.10 —') == section(text, '### R3.9 —', '### R3.10 —')
    for a, b in [('### R3.2 —', '### R3.3 —'), ('### R3.7 —', '### R3.8 —')]:
        assert section(old_text, a, b) == section(text, a, b)
    # Every unedited paragraph is retained, including prior statistical-scope corrections.
    unchanged = [p for p in old_text.split('\n\n') if not any(e['before'] in p for e in edits)]
    assert all(p in text for p in unchanged)
    links = re.findall(r'\]\(([^)]+)\)', text)
    local = [url for url in links if url.startswith('E:/')]
    assert all(pathlib.Path(url.split('#', 1)[0]).is_file() for url in local)
    assert [x for x in links if x.startswith('https:')] == [x for x in re.findall(r'\]\(([^)]+)\)', old_text) if x.startswith('https:')]

    numeric = []
    for method in ['GuardFed-AD2+', 'FLGMM']:
        i, row = next((i, r) for i, r in enumerate(aggregates['ten']) if r['method'] == method and r['scope'] == 'balanced_all10')
        assert row['n'] == 10 and all(s['n_scenes'] == 10 for s in row['seed_first'])
        for metric in ['accuracy', 'aeod', 'aspd']:
            value = row[metric]['mean']
            display = f'{value * 100:.2f}%' if metric == 'accuracy' else f'{value:.4f}'
            assert display in comparison
            numeric.append({'method': method, 'metric': metric, 'path': str((BASE / 'seed_first_aggregates.json').relative_to(ROOT)).replace('\\', '/'), 'pointer': f'/ten/{i}/{metric}/mean', 'actual_value': value, 'display': display, 'transform': '100*x; .2f; percent' if metric == 'accuracy' else '.4f'})
    directions = []
    for fi, fl in enumerate(scenes['ten']):
        if fl['method'] != 'FLGMM':
            continue
        gi, gf = next((i, r) for i, r in enumerate(scenes['ten']) if r['method'] == 'GuardFed-AD2+' and (r['distribution'], r['attack']) == (fl['distribution'], fl['attack']))
        assert gf['seeds'] == fl['seeds'] == list(range(91001, 91011))
        for metric in ['accuracy', 'aeod', 'aspd']:
            assert gf[metric]['mean'] < fl[metric]['mean']
            directions.append({'distribution': fl['distribution'], 'attack': fl['attack'], 'metric': metric, 'GF_pointer': f'/ten/{gi}/{metric}/mean', 'FL_pointer': f'/ten/{fi}/{metric}/mean', 'GF_saved_mean': gf[metric]['mean'], 'FL_saved_mean': fl[metric]['mean'], 'GF_lower': True})
    assert len(directions) == 30
    by_method = {r['method']: (i, r) for i, r in enumerate(aggregates['ten']) if r['scope'] == 'balanced_all10'}
    tradeoffs = []
    for baseline, metric, relation in [('FLTrust', 'accuracy', 'higher'), ('LoGoFair-DP', 'aspd', 'lower')]:
        bi, br = by_method[baseline]
        gi, gr = by_method['GuardFed-AD2+']
        assert (br[metric]['mean'] > gr[metric]['mean']) if relation == 'higher' else (br[metric]['mean'] < gr[metric]['mean'])
        tradeoffs.append({'baseline': baseline, 'metric': metric, 'relation_to_GuardFed': relation, 'baseline_pointer': f'/ten/{bi}/{metric}/mean', 'GF_pointer': f'/ten/{gi}/{metric}/mean', 'baseline_saved_mean': br[metric]['mean'], 'GF_saved_mean': gr[metric]['mean']})
    pins = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p.read_bytes()) for p in [PARENT, BASE / 'ROOT_REVIEW.json', BASE / 'seed_first_aggregates.json', BASE / 'summary_statistics.json', BASE / 'SOURCE_BINDINGS.json', BASE / 'TABLES.md', NATIVE]}
    OUTPUT.write_bytes(candidate)
    save('REPLACEMENTS.json', edits)
    save('SOURCE_PINS.json', pins)
    save('SOURCE_POINTERS.json', {'displayed_means': numeric, 'saved_scene_mean_directions': directions, 'retained_tradeoffs': tradeoffs, 'scope': {'native_root': {'/methods': 11, '/records': 1100, '/seed_counts': [10, 9, 6]}, 'FL_native_root': {'/accepted_total': 96, '/reused_separately': 4, '/native_coverage_complete': True}, 'FL_three_view90': 'Unchanged accepted scope and local table link inherited from pinned parent; no new three-view adoption.'}, 'no_statistics_recomputed': True})
    (HERE / 'REVERSIBLE_DIFF.patch').write_bytes(''.join(difflib.unified_diff(old_text.splitlines(keepends=True), text.splitlines(keepends=True), fromfile=str(PARENT.relative_to(ROOT)).replace('\\', '/'), tofile=str(OUTPUT.relative_to(ROOT)).replace('\\', '/'))).encode('utf-8'))
    results = {'status': 'PASS_TEXT_ONLY_AUTHOR_REVIEW_CANDIDATE_NO_ADOPTION', 'parent_sha256': PARENT_SHA, 'native11_root_sha256': ROOT_SHA, 'candidate_sha256': sha(candidate), 'replacement_spans': len(edits), 'full_raw_inverse_exact': True, 'comment_titles_exact_and_ordered': 24, 'reviewer_quote_lines_exact_and_ordered': 38, 'all_unedited_parent_paragraphs_preserved': len(unchanged), 'R3_2_and_R3_7_entire_bytes_preserved': True, 'historical_R3_9_entire_bytes_preserved': True, 'local_link_occurrences_exist': len(local), 'external_links_unchanged_not_requested_again': True, 'new_saved_mean_display_pointers': len(numeric), 'saved_mean_direction_comparisons': len(directions), 'retained_baseline_tradeoffs': 2, 'native_records': 1100, 'native_methods': 11, 'FL_native_new': 96, 'FL_screen_reuse': 4, 'FL_three_view_unchanged': 90, 'remaining_native_methods': 6, 'F80_mechanism380_Hybrid20_gradient64_NGAaccepted0_unchanged': True, 'new_statistics_or_inference_or_fit': 0, 'candidate_root_adoption': False, 'final_test': False}
    save('CHECK_RESULTS.json', results)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
