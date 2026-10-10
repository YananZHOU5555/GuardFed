"""Small no-Git fixtures for C6 closure, unchanged Hybrid and archive exclusion."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,re
H=Path(__file__).resolve().parent;R=H.parents[1];O=H.with_name('publication_increment35_prepared_20261010')
spec=importlib.util.spec_from_file_location('increment36_source_fixture_only',H/'publish_increment36.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def no_git(*a,**k):raise AssertionError('No Git during source checks')
m.git=no_git
read=lambda p:json.loads(Path(p).read_bytes())
scope=read(R/m.C6/'SCOPE.json');science=m.sha(R/m.C6/'FILES_SHA256.json');execution=m.sha(R/m.C6/'execution_candidate/EXECUTION_SOURCE_SHA256.json')
fixture=dict(status='ROOT_C_AFTER50_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',prior_three_view_models=150,accepted_new=6,cumulative_three_view_models=156,accepted_new_ids=m.IDS,original150_unchanged=True,all_native_differences_zero=True,server_strict_bound_in_saved_receipts=True,negative_results_preserved=True,new_training=0,new_Full_inference=0,new_CNN_inference_for_root_review=0,test_inference=False,source_scope_complete=True,science_seal_sha256=science,execution_seal_sha256=execution,prior150_root_adoption_sha256='3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8')
m.closure_guard(fixture,scope,science,execution);refused=[]
def reject(label,fn):
 try:fn()
 except (AssertionError,KeyError):refused.append(label);return
 raise AssertionError('Accepted drift: '+label)
for k,v in [('status','PENDING'),('prior_three_view_models',147),('accepted_new',5),('cumulative_three_view_models',150),('accepted_new_ids',m.IDS[:-1]),('original150_unchanged',False),('all_native_differences_zero',False),('negative_results_preserved',False),('new_training',1),('test_inference',True),('source_scope_complete',False),('science_seal_sha256','0'*64),('execution_seal_sha256','0'*64),('prior150_root_adoption_sha256','0'*64)]:
 b=copy.deepcopy(fixture);b[k]=v;reject('C6_'+k,lambda b=b:m.closure_guard(b,scope,science,execution))
bad=copy.deepcopy(scope);bad['excluded_prior_ids'].append(m.IDS[0]);reject('prior_selected_overlap',lambda:m.closure_guard(fixture,bad,science,execution))
prepared=read(H/'PREPARED_INPUTS.json');prior=read(R/m.TRAIN/'publication_closed_increment35_20261010.json');hy=R/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010/ROOT_ADOPTION_REVIEW.json';hp=read(hy);rel='experimental/'+hy.relative_to(R/'tmp').as_posix();h=m.sha(hy)
m.hybrid_unchanged_guard(hp,h,prior,rel,prepared['unchanged_Hybrid_root_sha256'])
reject('changed_Hybrid_SHA',lambda:m.hybrid_unchanged_guard(hp,'0'*64,prior,rel,h))
bad=copy.deepcopy(prior);bad['copied_sha256'][rel]='0'*64;reject('Hybrid_not_bound_publication35',lambda:m.hybrid_unchanged_guard(hp,h,bad,rel,h))
for k,v in [('accepted_total',20),('new_inference',1),('selection_performed',True),('scientific_changes',True),('final_test',True)]:
 b=copy.deepcopy(hp);b[k]=v;reject('Hybrid_'+k,lambda b=b:m.hybrid_unchanged_guard(b,h,prior,rel,h))
source=(H/'publish_increment36.py').read_text('utf8');tree=ast.parse(source);plan=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan');add=next(n for n in plan.body if isinstance(n,ast.FunctionDef) and n.name=='add')
ns=dict(ROOT=R.resolve(),EXCLUDED=m.EXCLUDED,sha=m.sha,re=re,Path=Path,hy_adoption=hy.resolve(),tracked={'experimental/source_fixture_old.tar.gz'},sources={},mapping={})
exec(compile(ast.Module(body=[add],type_ignores=[]),'original_publisher_add_fixture','exec'),ns)
p=H/'PREPARED_INPUTS.json';reject('already_published_archive_path',lambda:ns['add'](p,'experimental/source_fixture_old.tar.gz'))
reject('old_Hybrid_seal_tree',lambda:ns['add'](hy.parent/'DELIVERY_FILES_SHA256.json'))
reject('old_Hybrid_LATEST_not_republished',lambda:ns['add'](hy.parent.parent/'LATEST_BACKUP.json'))
ns['add'](p,'experimental/source_fixture_new.tar.gz');ns['add'](hy)
assert set(ns['mapping'])=={'experimental/source_fixture_new.tar.gz',rel}
# Run the original final archive-uniqueness expression on the in-memory mapping.
archive_nodes=[n for n in plan.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='archives' for t in n.targets)]
archive_assert=next(n for n in plan.body if isinstance(n,ast.Assert) and 'len(archives)' in ast.unparse(n.test))
compiled=compile(ast.Module(body=archive_nodes+[archive_assert],type_ignores=[]),'publisher_original_archive_guard','exec')
exec(compiled,ns);ns['mapping']['experimental/source_fixture_duplicate.tar.gz']=m.sha(p);reject('duplicate_new_archive_content',lambda:exec(compiled,ns))
def funcs(p):
 s=Path(p).read_text('utf8');return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
assert funcs(O/'publish_increment35.py')['verify_blobs']==funcs(H/'publish_increment36.py')['verify_blobs']
marker='    try:\n        for name, source in sources.items():';assert (O/'publish_increment35.py').read_text('utf8').split(marker,1)[1]==source.split(marker,1)[1]
assert "git('add', '-f', '--', *names[start:start + 25])" in source and "git('add', '--renormalize', '--'" in source and "'\"' + n + '\" -text\\n'" in source
assert source.index('sources, mapping, proof, health = plan(args)')<source.index('    output.mkdir()')
assert "closed.get('C50_table') is None" in source and "seal(hy_adoption.parent" not in source and '100_000_000' in source
v=(H/'verify_increment36.py').read_text('utf8');ast.parse(v);assert '[900, 156, 156, 11, 19]' in v
t=read(H/'ROOT_CLOSED_INPUTS_TEMPLATE.json');assert t['counts'] is None and t['C50_table'] is None and all(x['sha256'] is None and x['path'] is None for x in t['closure_pins'].values())
for path,pin in prepared['ready_sha256'].items():assert m.sha(R/path)==pin
result=dict(status='SOURCE_ONLY_EXACT6_UNCHANGED_HYBRID_ARCHIVE_AND_TRANSPORT_CHECKS_PASS',positive_in_memory_exact6_fixture=True,actual_C6_adoption_inspected=False,unchanged_Hybrid_actual_publication35_SHA_bound=True,old_Hybrid_archive_or_seal_tree_omitted=True,C50_table_must_remain_null=True,refusals=refused,refusal_count=len(refused),index_mutation_and_failure_block_byte_exact=True,force_add_text_renormalize_blob_SHA_retained=True,checks_before_first_output_write=True,source_AST=True,actual_Git_calls=0,SSH=False,network=False,CNN=False,stage=False,commit=False,push=False)
parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
if args.report:
 assert args.report.resolve().is_relative_to(H.resolve())
 with args.report.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
