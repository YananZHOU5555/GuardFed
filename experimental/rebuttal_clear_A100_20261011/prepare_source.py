"""Prepare reversible clear-A100 editorial source only; do not construct a draft."""
from pathlib import Path
import ast, hashlib, json
H=Path(__file__).resolve().parent;R=H.parents[1]
O=R/'tmp/rebuttal_clear_A90_20261011'
old=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A90_20261011/rebuttal_clear_20261011.md'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,value):
    with (H/name).open('x',encoding='utf8',newline='\n') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
assert sha(old)=='c2041b79aa3ceedc3fe62f0c76b18ffa7656d7e9fb9b994ef7571a4344d89e46'
source=(O/'build_and_check.py').read_text('utf8')
source=source.replace('Minimal reversible A90 editorial update','Minimal reversible A100 editorial update')
source=source.replace("rebuttal_clear_20261011/rebuttal_clear_20261011.md'","rebuttal_clear_A90_20261011/rebuttal_clear_20261011.md'")
source=source.replace("OUT = H/'rebuttal_clear_A90_20261011.md'","OUT = H/'rebuttal_clear_A100_20261011.md'")
start=source.index('def inputs():');end=source.index('\n\ndef edited(binding):')
inputs='''def inputs():
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
'''
source=source[:start]+inputs+source[end:]
source=source.replace('rebuttal_integrated_A80_reader_20261011/','rebuttal_integrated_A90_reader_20261011/')
source=source.replace('Accepted detailed A90 ','Accepted detailed A100 ')
source=source.replace("    draft = original\n", "    facts = read(H/'FACT_BINDINGS.json')\n    edits.append(['Accepted A100 table pointers',base+'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011/TABLES.md',base+facts['table_directory']+'/TABLES.md',2])\n    draft = original\n")
source=source.replace('CLEAR_A90_EDITORIAL_CHECK_PASS','CLEAR_A100_EDITORIAL_CHECK_PASS')
source=source.replace("A90_table_root_sha256=read(H/'FACT_BINDINGS.json')['root_A90_sha256'],detailed_A90_binding=binding","A100_table_root_sha256=read(H/'FACT_BINDINGS.json')['root_A100_sha256'],detailed_A100_binding=binding")
source=source[:source.index("if __name__ == '__main__':")]+'''if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        print(json.dumps(check(),ensure_ascii=False,indent=2))
    else:
        assert not OUT.exists() and not (H/'BUILD_RECEIPT.json').exists()
        binding=inputs();original,draft,edits=edited(binding)
        OUT.write_text(draft,encoding='utf8',newline='\\n')
        (H/'REVERSIBLE_DIFF.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),draft.splitlines(True),fromfile='accepted_clear_A90',tofile='candidate_clear_A100')),encoding='utf8')
        receipt=dict(status='CLEAR_A100_MINIMAL_DRAFT_BUILT_ROOT_EDITORIAL_CHECK_PENDING',source_sha256=sha(OLD),draft_sha256=sha(OUT),edit_groups=len(edits),editorial_checker_executed=False,science_recomputed=False,final_test=False)
        (H/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
        print(json.dumps(receipt,ensure_ascii=False))
'''
compile(source,str(H/'build_and_check.py'),'exec')
with (H/'build_and_check.py').open('x',encoding='utf8',newline='\n') as f:f.write(source)
edits=[
['AE complete scope','nine complete A-deletion scenes','ten complete A-deletion scenes',1],
['AE remaining scope','The remaining A scenes, other image controls, broader method comparability, synthetic provenance, and frozen final evaluation remain unresolved.','The other five image controls, broader method comparability, synthetic provenance, and frozen final evaluation remain unresolved.',1],
['R3.2 scope','A currently covers nine cells with 90 pairs.','A now also covers all ten cells with 100 pairs.',1],
['R3.2 completed final scene','Non-IID A S-DFA now has all ten seeds. Sp-DFA has five accepted seeds and is excluded from the complete-scene table; its remaining seeds and the other five image controls are unfinished. This response is not an all-component completion claim.','Non-IID A Sp-DFA now has all ten seeds. The other five image controls remain unfinished; this response is not an all-component completion claim.',1],
['R3.7 final scene and aggregate tradeoffs','Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes.','The completed non-IID Sp-DFA scene also shows panel-dependent trade-offs. With native/shared calibration, Full has higher mean ACC (the ten-seed deletion difference is −0.158 percentage points) and lower AEOD in the ten- and nine-seed panels, but higher ASPD. All three directions reverse in the fixed six-seed panel. Raw ten-/nine-seed means favor Full on all three metrics, whereas the six-seed panel retains a disparity trade-off.\n\nThe balanced ten-scene, seed-first summary likewise favors Full on all three raw ten-seed means, while calibrated results trade lower ACC and higher AEOD after deletion against lower ASPD. Subset directions vary; native/shared outputs coincide here and are not independent confirmations. These descriptive summaries retain every fixed panel and do not select a primary endpoint.\n\nCorrelated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes.',1],
['R3.9 current increment','That historical snapshot includes the earlier A80 response update. The current draft adds A90 evidence; neither establishes completion of the remaining benchmark or final evaluation.','That historical snapshot includes the earlier A80 response update. The current draft adds complete A100 evidence while retaining the A90 findings; neither establishes completion of the remaining benchmark or final evaluation.',1],
['P2 accepted cutoff','At this draft\'s accepted cutoff, 295 new models have native and three-view evidence. U100 and C100 each cover ten cells; A90 covers nine. The five accepted non-IID A Sp-DFA seeds are partial and excluded from complete-scene statistics. Finish that scene and the other five controls, then produce complete matched-seed tables.','At this draft\'s accepted cutoff, 300 new models have native and three-view evidence. U100, C100 and A100 each cover all ten cells. Finish the other five controls, then produce their complete matched-seed tables.',1],
['Supporting A table label','nine-scene A-deletion table','complete ten-scene A-deletion table',1]]
base=old.read_text('utf8')
for role,before,after,count in edits:assert base.count(before)==count,role
put('EDITS.json',edits)
paths=[old,old.parent/'ROOT_REVIEW.json',O/'build_and_check.py',R/'tmp/rebuttal_clean_reader_20261011/assemble.py']
put('SOURCE_PINS.json',{p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths})
put('DETAIL_BINDING.template.json',dict(root_adopted=False,root_proof=None,root_proof_sha256=None,rebuttal=None,rebuttal_sha256=None,insertions=None,insertions_sha256=None,root_status=None))
c=R/'tmp/celeba_mechanism_A100_candidate_20261011';t=json.loads((c/'tables.json').read_bytes());cross=json.loads((c/'CROSS_SCENE_ADDITIONAL.json').read_bytes())
rows=[]
for scope,panels in [('per_scene',t['panels']),('balanced_ten_scene_panels',cross['balanced_ten_scene_panels'])]:
    for panel in panels:
        row=next(x for x in panel['rows'] if x['variant']=='minus_A minus Full' and (scope!='per_scene' or (x['distribution'],x['attack'])==('non-IID','Sp-DFA')))
        rows.append(dict(scope=scope,view=panel['view'],seeds=panel['seeds'],selector=[row['distribution'],row['attack'],row['variant']],row=row))
put('FACT_BINDINGS.template.json',dict(root_adopted=False,table_directory=None,root_A100_sha256=None,tables_sha256=None,cross_scene_sha256=None,table_reader_sha256=None,paired_delta_definition='minus_A minus Full',exact_rows=rows,observed_unadopted_candidate_tables_sha256=sha(c/'tables.json'),observed_unadopted_candidate_cross_sha256=sha(c/'CROSS_SCENE_ADDITIONAL.json'),new_statistical_calculation=False))
# Compilation and static correspondence only: never invoke inputs/edited/check.
oldtree=ast.parse((O/'build_and_check.py').read_text('utf8'));newtree=ast.parse(source)
oldcheck=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='check')
newcheck=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='check')
assert len(oldcheck.body)==len(newcheck.body)
assert all(ast.dump(a)==ast.dump(b) for a,b in zip(oldcheck.body[:-1],newcheck.body[:-1]))
put('SOURCE_PREPARATION.json',dict(status='CLEAR_A100_SOURCE_PREPARED_NO_DRAFT_NO_CHECKER',source_compiled=True,prior_check_invariants_AST_exact_excluding_return_metadata=True,original_quote_link_gate_AST_count=4,edit_groups_prepared=8,dynamic_link_groups=3,table_root_bound=False,detailed_root_bound=False,future_hashes_null=True,checker_executed=False,draft_generated=False,science_recomputed=False,SSH=0,fit=0,test=False))
print(json.dumps(dict(status='SOURCE_PREPARED_ONLY',source_sha256=sha(H/'build_and_check.py'),draft_generated=False,checker_executed=False)))
