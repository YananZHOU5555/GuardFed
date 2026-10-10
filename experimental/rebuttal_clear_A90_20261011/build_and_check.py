"""Minimal reversible A90 editorial update; reuse original quote/link checks, no science."""
from pathlib import Path
import ast, difflib, hashlib, json, re

H = Path(__file__).resolve().parent
R = H.parents[1]
OLD = R/'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_20261011/rebuttal_clear_20261011.md'
OUT = H/'rebuttal_clear_A90_20261011.md'
ORIGINAL = R/'tmp/rebuttal_clean_reader_20261011/assemble.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def inputs():
    for name, pin in read(H/'SOURCE_PINS.json').items():
        p = R/name
        assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
    b = read(H/'DETAIL_BINDING.json')  # Must be actual root-adopted detailed A90, never a placeholder.
    assert b['root_adopted'] is True
    for key in ['root_proof', 'rebuttal', 'insertions']:
        assert sha(R/b[key]) == b[key+'_sha256'], key
    proof = read(R/b['root_proof'])
    assert proof['status'] == b['root_status']
    facts = read(H/'FACT_BINDINGS.json')
    folder = R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011'
    root = read(folder/'ROOT_VERIFICATION.json')
    assert root['root_adoption'] and sha(folder/'ROOT_VERIFICATION.json') == facts['root_A90_sha256']
    assert (root['preserved_records'],root['paired_models'],root['complete_scenes']) == (180,90,9)
    assert sha(R/root['source_acceptance_path']) == root['source_acceptance_sha256']
    assert read(R/root['source_acceptance_path'])['cumulative_accepted'] == 295
    assert sha(folder/'tables.json') == facts['tables_sha256'] == root['files_sha256']['tables.json']
    tables = read(folder/'tables.json')
    for view, saved in facts['exact_rows'].items():
        panel = next(p for p in tables['panels'] if p['view'] == view and p['seeds'] == list(range(91001,91011)))
        row = next(x for x in panel['rows'] if (x['distribution'],x['attack'],x['variant']) == ('non-IID','S-DFA','minus_A minus Full'))
        assert row == saved and row['n'] == 10
    assert facts['exact_rows']['native'] == facts['exact_rows']['shared_calibration']
    # These are display conversions of stored means; no new statistic calculation.
    expected = {'native':['-0.036','-0.00434','-0.00145'], 'raw':['+0.014','-0.01156','-0.00835']}
    for view, tokens in expected.items():
        row = facts['exact_rows'][view]
        rendered = [format(row['accuracy_pct']['mean'], '.3f' if view == 'native' else '+.3f'), format(row['aeod']['mean'], '.5f'), format(row['aspd']['mean'], '.5f')]
        assert rendered == tokens
    return b


def edited(binding):
    original = OLD.read_text(encoding='utf8')
    edits = read(H/'EDITS.json')
    base = 'E:/OneDrive/文档/GuardFed/'
    for key, oldname in [('rebuttal','rebuttal_integrated_20261011.md'),('insertions','manuscript_insertions_integrated_20261011.md')]:
        old = base+'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011/'+oldname
        edits.append(['Accepted detailed A90 '+key, old, base+binding[key], 1])
    draft = original
    for role, before, after, count in edits:
        assert draft.count(before) == count, role
        draft = draft.replace(before, after)
    inverse = draft
    for role, before, after, count in reversed(edits):
        assert inverse.count(after) == count, role
        inverse = inverse.replace(after, before)
    assert inverse == original
    return original, draft, edits


def check():
    binding = inputs()
    original, expected, edits = edited(binding)
    draft = OUT.read_text(encoding='utf8'); assert draft == expected
    original_quotes = [line for line in original.splitlines() if line.startswith('>')]
    # Execute the original assembler's exact final quotation/count/local-link gates.
    source = ORIGINAL.read_text(encoding='utf8'); tree = ast.parse(source)
    retained = []
    for node in tree.body:
        token = ast.get_source_segment(source,node)
        if token.startswith("assert [line for line in draft.splitlines()") or token.startswith("assert draft.count('**Original comment") or token.startswith('local_links=re.findall') or token.startswith('assert all(Path(link).is_file()'):
            retained.append(node)
    assert len(retained) == 4
    ns = dict(draft=draft,original_quotes=original_quotes,re=re,Path=Path)
    exec(compile(ast.Module(body=retained,type_ignores=[]),str(ORIGINAL)+' [original editorial gates]','exec'),ns)
    headings = lambda text: re.findall(r'^(?:## Associate Editor|## Reviewer [123]|### [^\n]+)$',text,re.M)
    assert headings(draft) == headings(original)
    assert draft.count('**Response.**') == original.count('**Response.**') == 24
    for line in original.splitlines():
        if re.match(r'\| P[13456] ',line): assert line in draft
    assert re.findall(r'https://[^\s)]+',original) == re.findall(r'https://[^\s)]+',draft)
    assert 'No best Full seed is compared with deletion means.' in draft
    assert 'identity projection on CNN parameters' in draft and '20 fixed image-ID virtual cohorts' in draft
    assert 'Proposed manuscript changes have not yet been applied' in draft
    assert 'No final evaluation or submission-ready completion is claimed.' in draft
    return dict(status='CLEAR_A90_EDITORIAL_CHECK_PASS',source_sha256=sha(OLD),draft_sha256=sha(OUT),source_words=len(original.split()),draft_words=len(draft.split()),original_comments=24,response_sections=24,quotation_lines_exact_and_ordered=True,headings_exact_and_ordered=True,forward_and_inverse_text_exact=True,edit_groups=len(edits),local_link_occurrences_checked=len(ns['local_links']),original_editorial_gate_AST_reused=4,source_assembler_sha256=sha(ORIGINAL),A90_table_root_sha256=read(H/'FACT_BINDINGS.json')['root_A90_sha256'],detailed_A90_binding=binding,science_recomputed=False,final_test=False,submitted_manuscript_applied=False,whole_rebuttal_complete=False)


if __name__ == '__main__':
    assert not OUT.exists() and not (H/'SELF_CHECK.json').exists()
    binding = inputs(); original, draft, edits = edited(binding)
    OUT.write_text(draft,encoding='utf8',newline='\n')
    report = check()
    (H/'SELF_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (H/'REVERSIBLE_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),draft.splitlines(True),fromfile='accepted_clear_A80',tofile='candidate_clear_A90')),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))
