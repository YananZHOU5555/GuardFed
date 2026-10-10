"""Minimal reversible A100 editorial update; reuse original quote/link checks, no science."""
from pathlib import Path
import ast, difflib, hashlib, json, re

H = Path(__file__).resolve().parent
R = H.parents[1]
OLD = R/'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A90_20261011/rebuttal_clear_20261011.md'
OUT = H/'rebuttal_clear_A100_20261011.md'
ORIGINAL = R/'tmp/rebuttal_clean_reader_20261011/assemble.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def inputs():
    for name, pin in read(H/'SOURCE_PINS.json').items():
        p = R/name
        assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
    assert (H/'DETAIL_BINDING.json').is_file() and (H/'FACT_BINDINGS.json').is_file(), 'Actual A100 table and detailed-root bindings absent; construction forbidden'
    b = read(H/'DETAIL_BINDING.json'); facts = read(H/'FACT_BINDINGS.json')
    assert b['root_adopted'] is True and facts['root_adopted'] is True
    for key in ['root_proof', 'rebuttal', 'insertions']:
        assert isinstance(b[key+'_sha256'], str) and len(b[key+'_sha256']) == 64
        assert sha(R/b[key]) == b[key+'_sha256'], key
    proof = read(R/b['root_proof'])
    assert proof['status'] == b['root_status'] and 'A100' in proof['status'] and 'ADOPTED' in proof['status']
    assert proof['A100_table_root_sha256'] == facts['root_A100_sha256']
    assert proof['documents_sha256'][Path(b['rebuttal']).name] == b['rebuttal_sha256']
    assert proof['documents_sha256'][Path(b['insertions']).name] == b['insertions_sha256']
    folder = R/facts['table_directory']
    root = read(folder/'ROOT_VERIFICATION.json')
    assert root['root_adoption'] and sha(folder/'ROOT_VERIFICATION.json') == facts['root_A100_sha256']
    assert (root['preserved_records'],root['paired_models'],root['complete_scenes']) == (200,100,10)
    assert sha(R/root['source_acceptance_path']) == root['source_acceptance_sha256']
    assert read(R/root['source_acceptance_path'])['cumulative_accepted'] == 300
    for filename, key in [('tables.json','tables_sha256'),('CROSS_SCENE_ADDITIONAL.json','cross_scene_sha256'),('TABLES.md','table_reader_sha256')]:
        assert sha(folder/filename) == facts[key] == root['files_sha256'][filename]
    tables = read(folder/'tables.json'); cross = read(folder/'CROSS_SCENE_ADDITIONAL.json')
    wanted = {(scope,view,tuple(seeds)) for scope in ('per_scene','balanced_ten_scene_panels') for view in ('native','raw','shared_calibration') for seeds in (range(91001,91011),range(91002,91011),range(91005,91011))}
    assert len(facts['exact_rows']) == 18 and {(x['scope'],x['view'],tuple(x['seeds'])) for x in facts['exact_rows']} == wanted
    for saved in facts['exact_rows']:
        panels = tables['panels'] if saved['scope']=='per_scene' else cross[saved['scope']]
        panel = next(p for p in panels if p['view']==saved['view'] and p['seeds']==saved['seeds'])
        row = next(x for x in panel['rows'] if (x['distribution'],x['attack'],x['variant'])==tuple(saved['selector']))
        assert row == saved['row'] and row['n']==len(saved['seeds'])
    sp = next(x['row'] for x in facts['exact_rows'] if x['scope']=='per_scene' and x['view']=='native' and len(x['seeds'])==10)
    assert format(sp['accuracy_pct']['mean'],'.3f')=='-0.158'
    assert facts['paired_delta_definition']=='minus_A minus Full'
    return b


def edited(binding):
    original = OLD.read_text(encoding='utf8')
    edits = read(H/'EDITS.json')
    base = 'E:/OneDrive/文档/GuardFed/'
    for key, oldname in [('rebuttal','rebuttal_integrated_20261011.md'),('insertions','manuscript_insertions_integrated_20261011.md')]:
        old = base+'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A90_reader_20261011/'+oldname
        edits.append(['Accepted detailed A100 '+key, old, base+binding[key], 1])
    facts = read(H/'FACT_BINDINGS.json')
    edits.append(['Accepted A100 table pointers',base+'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011/TABLES.md',base+facts['table_directory']+'/TABLES.md',2])
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
    return dict(status='CLEAR_A100_EDITORIAL_CHECK_PASS',source_sha256=sha(OLD),draft_sha256=sha(OUT),source_words=len(original.split()),draft_words=len(draft.split()),original_comments=24,response_sections=24,quotation_lines_exact_and_ordered=True,headings_exact_and_ordered=True,forward_and_inverse_text_exact=True,edit_groups=len(edits),local_link_occurrences_checked=len(ns['local_links']),original_editorial_gate_AST_reused=4,source_assembler_sha256=sha(ORIGINAL),A100_table_root_sha256=read(H/'FACT_BINDINGS.json')['root_A100_sha256'],detailed_A100_binding=binding,science_recomputed=False,final_test=False,submitted_manuscript_applied=False,whole_rebuttal_complete=False)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        print(json.dumps(check(),ensure_ascii=False,indent=2))
    else:
        assert not OUT.exists() and not (H/'BUILD_RECEIPT.json').exists()
        binding=inputs();original,draft,edits=edited(binding)
        OUT.write_text(draft,encoding='utf8',newline='\n')
        (H/'REVERSIBLE_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),draft.splitlines(True),fromfile='accepted_clear_A90',tofile='candidate_clear_A100')),encoding='utf8')
        receipt=dict(status='CLEAR_A100_MINIMAL_DRAFT_BUILT_ROOT_EDITORIAL_CHECK_PENDING',source_sha256=sha(OLD),draft_sha256=sha(OUT),edit_groups=len(edits),editorial_checker_executed=False,science_recomputed=False,final_test=False)
        (H/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(receipt,ensure_ascii=False))
