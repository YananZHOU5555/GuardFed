"""No-Git metadata/manifest fixtures; no actual future C60 results are consumed."""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,re,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
spec=importlib.util.spec_from_file_location('prepared37_only',H/'publish_increment37.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def no_git(*a,**k):raise RuntimeError('Git forbidden in source-only checks')
p.git=no_git;p.base.git=no_git
read=lambda q:json.loads(Path(q).read_bytes())
d=read(H/'PREPARED_INPUTS.json');template=read(H/'ROOT_CLOSED_INPUTS_TEMPLATE.json')
for row in d['ready_manifest']:assert p.sha(R/row['source'])==row['sha256'] and (R/row['source']).stat().st_size==row['bytes']
for name,digest in d['reference_pins'].items():assert p.sha(R/name)==digest
assert template['counts'] is None and all(v['path'] is v['sha256'] is None for v in template['closure_pins'].values())
assert d['actual_C4_adoption_sha256'] is d['actual_C60_root_sha256'] is None
scope=read(R/p.C4/'SCOPE.json');s=p.sha(R/p.C4/'FILES_SHA256.json');e=p.sha(R/p.C4/'execution_candidate/EXECUTION_SOURCE_SHA256.json')
fixture=dict(status='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',prior_three_view_models=156,accepted_new=4,cumulative_three_view_models=160,accepted_new_ids=p.IDS,original156_unchanged=True,all_native_differences_zero=True,server_strict_bound_in_saved_receipts=True,negative_results_preserved=True,new_training=0,new_Full_inference=0,new_CNN_inference_for_root_review=0,test_inference=False,source_scope_complete=True,science_seal_sha256=s,execution_seal_sha256=e,prior156_root_adoption_sha256='a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080')
p.closure_guard(fixture,scope,s,e);refused=[]
def reject(name,call):
 try:call()
 except (AssertionError,KeyError,TypeError):refused.append(name);return
 raise AssertionError('Unexpected acceptance '+name)
for key,value in [('status','PREPARED'),('prior_three_view_models',150),('accepted_new',3),('cumulative_three_view_models',156),('accepted_new_ids',p.IDS[:-1]),('original156_unchanged',False),('all_native_differences_zero',False),('negative_results_preserved',False),('new_training',1),('new_Full_inference',1),('test_inference',True),('source_scope_complete',False),('science_seal_sha256','0'*64),('prior156_root_adoption_sha256','0'*64)]:
 x=copy.deepcopy(fixture);x[key]=value;reject('C4_'+key,lambda x=x:p.closure_guard(x,scope,s,e))
reject('prepared_template_not_closed',lambda:p.closed_guard(template))
closed=copy.deepcopy(template);closed.update(status='ROOT_CLOSED_INCREMENT37_INPUTS',counts=p.COUNTS)
for v in closed['closure_pins'].values():v.update(path='IN_MEMORY_FIXTURE_ONLY',sha256='0'*64)
p.closed_guard(closed)
x=copy.deepcopy(closed);x['closure_pins']['C60_root']['sha256']=None;reject('missing_actual_C60_ROOT_SHA',lambda:p.closed_guard(x))
x=copy.deepcopy(closed);x['counts']['three_view']=156;reject('views_not_closed160',lambda:p.closed_guard(x))
# Only metadata cardinalities are synthetic. No records, metrics or means are fabricated.
t=dict(unique_records=120,paired_models=60,complete_scenes=6,nonIID_complete_scenes=['Benign'],full_nonIID_coverage=False,primary_endpoint_selected=False,final_test=False,new_inference=0)
v=dict(mean_sd_scalars=972,display_mean_sd_cells=486,receipt_metrics_from_group_counts=1080,base_confusion_counts_structurally_checked=2880,old100_records_bytes_exact=True,old810_statistics_exact=True,old405_cells_exact=True,old162_IID_seed_first_bytes_exact=True)
tr=dict(status='ROOT_C60_SIX_SCENE_THREE_VIEW_TABLES_ADOPTED');binding=dict(actual_C4_binding=dict(adoption_sha256='FIXTURE_ONLY'),new_inference=0)
p.table_guard(tr,t,v,binding,'FIXTURE_ONLY')
x=dict(tr,status='PREPARED');reject('C60_prepared_not_adopted',lambda:p.table_guard(x,t,v,binding,'FIXTURE_ONLY'))
x=dict(t,complete_scenes=5);reject('C60_missing_sixth_scene',lambda:p.table_guard(tr,x,v,binding,'FIXTURE_ONLY'))
x=dict(v,old162_IID_seed_first_bytes_exact=False);reject('old_IID_aggregate_drift',lambda:p.table_guard(tr,t,x,binding,'FIXTURE_ONLY'))
x=dict(binding,actual_C4_binding=dict(adoption_sha256='wrong'));reject('C60_not_bound_to_C4',lambda:p.table_guard(tr,t,v,x,'FIXTURE_ONLY'))
source=(H/'publish_increment37.py').read_text('utf-8');tree=ast.parse(source)
plan=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan')
add=next(n for n in plan.body if isinstance(n,ast.FunctionDef) and n.name=='add')
ns=dict(Path=Path,ROOT=R.resolve(),EXCLUDED=p.EXCLUDED,sha=p.sha,re=re,prepared=d,tracked={'experimental/old_fixture.tar.gz'},sources={},mapping={})
exec(compile(ast.Module(body=[add],type_ignores=[]),'original37_add_guard_fixture','exec'),ns)
meta=H/'PREPARED_INPUTS.json';reject('old_archive_path',lambda:ns['add'](meta,'experimental/old_fixture.tar.gz'))
reject('old_Hybrid_not_repacked',lambda:ns['add'](R/d['unchanged_Hybrid_root_path']))
reject('old_C50_reply_not_repacked',lambda:ns['add'](R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010/rebuttal_integrated_20261009.md'))
reject('third_party_poster_not_redistributed',lambda:ns['add'](R/'tmp/celeba_baselines/source_recovery_novel_20261010/FEDWA_AAMAS2022_OFFICIAL_POSTER.pdf'))
ns['add'](meta,'experimental/new_fixture.tar.gz')
recovery=[row for row in d['ready_manifest'] if 'source_recovery_novel_20261010' in row['source']]
assert len(recovery)==8 and {Path(row['source']).name for row in recovery}==set(d['source_recovery_published_subset'])
assert all(not set(Path(row['source']).parts)&p.EXCLUDED for row in d['ready_manifest'])
assert not any('celeba_hybrid_screen_execution_20261009' in row['source'] or 'rebuttal_integrated_C50_20261010' in row['source'] for row in d['ready_manifest'])
assert "verify_blobs=base.verify_blobs" in source and "git('add','-f','--'" in source and "git('add','--renormalize','--'" in source and '-text\\n' in source
assert source.index('sources,mapping,proof,health=plan(args)')<source.index('    output.mkdir()')
for path in [H/'publish_increment37.py',H/'verify_increment37.py']:ast.parse(path.read_text('utf-8'))
verifier=(H/'verify_increment37.py').read_text('utf-8');assert '[900, 160, 160, 12, 19]' in verifier and '277be091be761249372c1da17c97d2bf83ef62ef' in verifier
result=dict(status='SOURCE_ONLY_MANIFEST_AND_CLOSURE_GUARDS_PASS_NO_STAGE',ready_manifest_files=len(d['ready_manifest']),ready_bytes=sum(row['bytes'] for row in d['ready_manifest']),positive_in_memory_metadata_fixtures=3,refusals=refused,refusal_count=len(refused),third_party_published_body_count=0,recovery_published_metadata_or_report_files=8,original18member_recovery_seal_local_only=True,old_Hybrid_and_C50_bundle_omitted=True,actual_C60_ROOT_read=False,future_closure_template_null=True,Git_calls=0,stage=False,commit=False,push=False,SSH=False,CNN=False,new_statistics=0)
parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
if args.report:
 assert args.report.resolve().parent==H.resolve() and not args.report.exists()
 args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
