"""No-Git, no-network metadata and unchanged transport checks only."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
H=Path(__file__).resolve().parent;R=H.parents[1];O=H.with_name('publication_increment34_prepared_v2_20261010')
spec=importlib.util.spec_from_file_location('increment35_metadata_only',H/'publish_increment35.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def no_git(*a,**k):raise AssertionError('Git must not run during source checks')
m.git=no_git
read=lambda p:json.loads(Path(p).read_bytes())
scope=read(R/m.C3/'SCOPE.json');science=m.sha(R/m.C3/'FILES_SHA256.json');execution=m.sha(R/m.C3/'execution_candidate/EXECUTION_SOURCE_SHA256.json')
# Synthetic closure fixture in memory only; not an actual root approval/adoption.
fixture=dict(status='ROOT_C_AFTER47_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',prior_three_view_models=147,accepted_new=3,cumulative_three_view_models=150,accepted_new_ids=m.IDS,original147_unchanged=True,all_native_differences_zero=True,server_strict_bound_in_saved_receipts=True,negative_results_preserved=True,new_training=0,new_Full_inference=0,new_CNN_inference_for_root_review=0,test_inference=False,source_scope_complete=True,science_seal_sha256=science,execution_seal_sha256=execution,prior147_root_adoption_sha256='64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea')
m.closure_guard(fixture,scope,science,execution)
refused=[]
for k,v in [('status','PENDING'),('prior_three_view_models',140),('accepted_new',2),('cumulative_three_view_models',147),('accepted_new_ids',m.IDS[:-1]),('original147_unchanged',False),('all_native_differences_zero',False),('negative_results_preserved',False),('new_training',1),('test_inference',True),('source_scope_complete',False),('science_seal_sha256','0'*64),('execution_seal_sha256','0'*64),('prior147_root_adoption_sha256','0'*64)]:
 b=copy.deepcopy(fixture);b[k]=v
 try:m.closure_guard(b,scope,science,execution)
 except (AssertionError,KeyError):refused.append(k)
 else:raise AssertionError('Accepted drift: '+k)
bad=copy.deepcopy(scope);bad['excluded_prior_ids'].append(m.IDS[0])
try:m.closure_guard(fixture,bad,science,execution)
except AssertionError:refused.append('prior_selected_overlap')
else:raise AssertionError('Accepted prior/selected overlap')
def funcs(p):
 s=Path(p).read_text(encoding='utf-8-sig');return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
assert funcs(O/'publish_increment34.py')['verify_blobs']==funcs(H/'publish_increment35.py')['verify_blobs']
source=(H/'publish_increment35.py').read_text();verifier=(H/'verify_increment35.py').read_text();ast.parse(source);ast.parse(verifier)
assert "git('add', '-f', '--', *names[start:start + 25])" in source and "git('add', '--renormalize', '--'" in source
assert "' -text" not in source and "'\"' + n + '\" -text\\n'" in source
assert source.index('sources, mapping, proof, health = plan(args)') < source.index('    output.mkdir()')
assert '100_000_000' in source and "name not in tracked, 'Old archive must not be republished" in source
assert "assert args.execute_stage" in source and 'ROOT_CLOSED_INCREMENT35_INPUTS' in source
assert 'publication_increment34_prepared_20261010' not in source and 'prepare_publication34_v2_root' not in source
assert '[900, 150, 150, 9, 19]' in verifier and "git('ls-remote', '--heads', 'origin', BRANCH)" in verifier
template=read(H/'ROOT_CLOSED_INPUTS_TEMPLATE.json');assert template['counts'] is None and template['C50_table'] is None and all(p['sha256'] is None for p in template['closure_pins'].values())
report=dict(status='SOURCE_ONLY_METADATA_AND_TRANSPORT_CHECKS_PASS_NOT_STAGED',positive_in_memory_exact3_fixture=True,actual_C3_adoption_inspected=False,refusals=refused,refusal_count=len(refused),index_blob_verifier_source_exact=True,publisher_AST=True,commit_verifier_AST=True,force_add_exact_allowlist_retained=True,text_attributes_and_blob_SHA_retained=True,closure_checks_before_first_output_write=True,old_archive_same_path_republish_refused=True,optional_C50_no_actual_inputs=True,actual_Git_calls=0,SSH=False,network=False,CNN=False,stage=False,commit=False,push=False)
with (H/'SELF_CHECK.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
