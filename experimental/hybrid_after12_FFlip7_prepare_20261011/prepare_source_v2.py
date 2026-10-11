"""Metadata-only exact7 source derivation; no execution or scientific imports."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/hybrid_native_after9_20261011'
pending={}
raw=lambda p:pending[Path(p).name] if Path(p).parent==H and Path(p).name in pending else Path(p).read_bytes()
sha=lambda p:hashlib.sha256(raw(p)).hexdigest()
read=lambda p:json.loads(raw(p))
def put(n,v):
 raw=v if isinstance(v,bytes) else (v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf8')
 assert not (H/n).exists() and n not in pending,n
 pending[n]=raw
priorpath=R/'tmp/celeba_hybrid_native12_root_adoption_20261011/ROOT_ADOPTION.json'
assert sha(priorpath)=='181e4219f967811a25446f682d94f4db511a5d7917b0650aabff380eba4c4ab1'
prior=read(priorpath);prior_ids=prior['accepted_ids']
prefix='CosineFairness_lam20.0_tau0.1_lr0.001_'
assert prior_ids==[prefix+f'IID_Benign_seed{s}_fullcoverage' for s in range(91002,91011)]+[prefix+f'IID_F Flip_seed{s}_fullcoverage' for s in range(91001,91004)]
assert prior['cumulative_accepted']==12 and prior['reused_separate']==4
assert sha(P/'OFFSERVER_VERIFICATION.json')==prior['offserver_sha256']=='fd774ec8884d7db041c8db96f3d81b20f72047f67d1c3133f80fda276f88e0e9'
assert read(P/'OFFSERVER_VERIFICATION.json')['accepted_ids']==prior_ids
assert sha(P/'FILES_SHA256.json')=='bb5a7f444f5326459bd41958c8c0e72795cdbe67ad425398d55ddf385ed87c15'
assert sha(P/'adopt_root.py')=='6a1cfb2a6a13340a680ae444cae3dd17240bad2c93d2e0343c2a27df3595173a'
manifestpath=R/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010/bound_metadata/stage/manifest.json'
assert sha(manifestpath)=='b0a014dd45dd27a2c38698144e49577e17ffe4b38fcdbee3dedb50f2e16e6ae9'
manifest=read(manifestpath);ids=[prefix+f'IID_F Flip_seed{s}_fullcoverage' for s in range(91004,91011)]
entries=[x for x in manifest['jobs'] if x['id'] in ids];assert [x['id'] for x in entries]==ids and len(manifest['jobs'])==96 and len(manifest['reused_jobs'])==4
assert not set(ids)&set(prior_ids) and not set(ids)&{x['id'] for x in manifest['reused_jobs']}
for x in entries:assert sha(manifestpath.parent/x['job'])==x['job_sha256']
observedpath=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261011T004147Z.raw.json'
assert sha(observedpath)=='58909812013a8f0aa2c9a1e2c0ecbca3782f0e7ab86d54fd5125b31c0a351464'
observed=read(observedpath);live=observed['Hybrid96'];observed_ids=[x['id'] for x in live['terminal_records']]
assert len(observed_ids)==22 and set(prior_ids+ids)<=set(observed_ids) and not live['failure_files']
assert not live['source']['changed_members'] and not observed['source_data_binding']['changed_paths']
excluded=[x for x in observed_ids if x not in prior_ids+ids]
assert excluded==[prefix+f'IID_FedSA_seed{s}_fullcoverage' for s in range(91001,91004)]
put('PRIOR_ROOT_ADOPTION.json',priorpath.read_bytes());put('PRIOR_OFFSERVER.json',(P/'OFFSERVER_VERIFICATION.json').read_bytes())
copy_names=['collect_once.py','verify_saved.py','restore_verify.py','storage.py','GUIDE.md','ROOT_BOUND_ADOPTION.json','ROOT_SEVEN_CANARY_CLOSURE.json','ROOT_COVERAGE_STARTUP.json']
parentseal=read(P/'FILES_SHA256.json')
for n in copy_names:
 assert sha(P/n)==parentseal['files'][n]['sha256'];put(n,(P/n).read_bytes())
remote='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/native_after12_FFlip7_20261011'
bulk='F:/YananResearchStorage/GuardFed/hybrid_after12_FFlip7_prepare_20261011'
put('DELTA_SCOPE.json',dict(status='SOURCE_ONLY_FIXED_EXACT7_NOT_EXECUTED',ids=ids,accepted_before=12,prior_offserver_path='PRIOR_OFFSERVER.json',prior_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),remote_output=remote+'/output',local_bulk_root=bulk,no_later_completions=True))
put('OBSERVED_SCOPE.json',dict(status='HISTORICAL_OBSERVATION_ONLY_NOT_EXECUTION_BINDING',source_path=str(observedpath.relative_to(R)),source_sha256=sha(observedpath),utc=live['checked_utc'],terminal_count=22,prior_accepted=12,selected_ids=ids,excluded_other_completed_ids=excluded,current_new_accepted=0))
put('FIXED_MANIFEST.json',dict(manifest_path=str(manifestpath.relative_to(R)),manifest_sha256=sha(manifestpath),package_sha256=prior['package_sha256'],full_scope_sha256=sha(manifestpath.parent/'full_scope.json'),selected_entries=entries,prior_ids=prior_ids,reused_jobs_separate=4))
diffs=[];maps={}
def derive(n,changes):
 old=(P/n).read_bytes();s=old.decode('utf8');spans=[]
 for a,b,role in changes:
  assert a in s,(n,a);count=s.count(a);s=s.replace(a,b);spans.append(dict(old=a,new=b,count=count,role=role))
 inv=s
 for x in reversed(spans):
  assert inv.count(x['new'])==x['count'],(n,x['role'],inv.count(x['new']),x['count']);inv=inv.replace(x['new'],x['old'])
 assert inv.encode('utf8')==old
 new=s.encode('utf8');compile(new,str(H/n),'exec');put(n,new);maps[n]=spans
 diffs.append(''.join(difflib.unified_diff(old.decode('utf8').splitlines(True),s.splitlines(True),fromfile='parent_after9/'+n,tofile='after12_FFlip7/'+n)))
namespace=('hybrid_native_after9_20261011',H.name,'namespace')
bind_check=""" assert sha(H/'EXECUTION_BINDINGS.json')==sys.argv[3]
 binding=read(H/'EXECUTION_BINDINGS.json')
 assert binding['execution_authorized'] is True and binding['source_seal_sha256']==sys.argv[2]
 assert binding['allowed_ids']==read(H/'DELTA_SCOPE.json')['ids']
 assert sha(binding['snapshot_path'])==binding['snapshot_sha256']
 assert sha(binding['source_review_path'])==binding['source_review_sha256']
 assert read(binding['source_review_path'])['source_adoptable'] is True
 frozen=read(H/'FROZEN_OBSERVATION.json');assert sha(H/'FROZEN_OBSERVATION.json')==binding['frozen_observation_sha256']
 assert frozen['selected_ids']==binding['allowed_ids'] and frozen['prior_ids']==read(H/'PRIOR_ROOT_ADOPTION.json')['accepted_ids']
 assert frozen['sha256']==binding['snapshot_sha256'] and frozen['source_path']==binding['snapshot_path']
 assert not list(H.glob('EXECUTION_FAILURE_*.json'))
 refusal={'collect':'COLLECT_COMMAND.json','download':'DOWNLOAD.json','verify':'RESTORE_VERIFY_COMMAND.json'}
 assert sys.argv[1] in refusal and not (H/refusal[sys.argv[1]]).exists()
"""
derive('execute_transport.py',[namespace,('/native_after9_20261011','/native_after12_FFlip7_20261011','remote namespace'),('1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775',sha(priorpath),'actual parent root'),("['cumulative_accepted']==9","['cumulative_accepted']==12",'parent count'),("['accepted_before']==9","['accepted_before']==12",'parent count'),(repr(read(P/'DELTA_SCOPE.json')['ids']),repr(ids),'exact7 ids'),('assert len(sys.argv)==3','assert len(sys.argv)==4','required actual execution binding SHA'),(" assert sha(H/'FILES_SHA256.json')==sys.argv[2]"," assert sha(H/'FILES_SHA256.json')==sys.argv[2]\n"+bind_check.rstrip('\n'),'late actual source-review authorization/snapshot binding and local failstop')])
derive('finalize_saved.py',[namespace,('native_after9_20261011/collect_once.py','native_after12_FFlip7_20261011/collect_once.py','release namespace'),('==(12,9,3)','==(19,12,7)','counts'),("off['accepted_ids'][:9]","off['accepted_ids'][:12]",'parent prefix'),('len(tensor_rows)==3','len(tensor_rows)==7','new count'),('models=3,tensors=24','models=7,tensors=56','tensor metadata'),('RECORDS3.json','RECORDS7.json','record namespace'),('EXACT3','EXACT7','status metadata'),('partial3 only;not a complete10seed scene or full100.','exact7 increment completes IID F Flip with prior3 only after root adoption;not full100.','scope limit'),('only three seeds','only this fixed seven-record increment; no scene statistics are generated','scope limit'),('accepted_before=9,accepted_new=3,accepted_total=12','accepted_before=12,accepted_new=7,accepted_total=19','counts'),('celeba_hybrid_native9_root_adoption_20261011','celeba_hybrid_native12_root_adoption_20261011','parent namespace'),('saved_models=3,saved_tensors=24','saved_models=7,saved_tensors=56','tensor metadata'),('old9','old12','parent label'),('Old9','Old12','parent label'),('new_accepted_pending=3,total_if_root_adopted=12','new_accepted_pending=7,total_if_root_adopted=19','counts')])
# Root adopter: original loops retained; future result literals become explicit actual-pins fields.
adopt=(P/'adopt_root.py').read_text('utf8')
old_delivery="[('DELIVERY_FILES_SHA256.json','52e8105e5654dac4cb0957848ee1906e004a4964c8cbb53ef63748e2014298e9',67),('FILES_SHA256.json','bb5a7f444f5326459bd41958c8c0e72795cdbe67ad425398d55ddf385ed87c15',22)]"
old_outer="{'OUTER_COLLECT_COMMAND.json': '8cba634bde0874b6217eaff2d2b35b425e9cf670183a2daac779b2ac7bcb58f3', 'OUTER_DOWNLOAD_COMMAND.json': '9f981b81e64cf6fba406c4a624308f7cb87d3c668d04a9d299af01d4a6c69f97', 'OUTER_VERIFY_COMMAND.json': 'b64fc0108a2d41d903ac729104ac286739b135bba183f3db246b3e0f076a1c44', 'OUTER_FINALIZE_COMMAND.json': 'be53c0bf0a3c04a34e7b96770245c70a59432df9d07c788a3e59e6a2dd86befb'}"
derive('adopt_root.py',[namespace,("PRIOR=ROOT/'tmp/celeba_hybrid_native9_root_adoption_20261011/ROOT_ADOPTION.json'","PRIOR=ROOT/'tmp/celeba_hybrid_native12_root_adoption_20261011/ROOT_ADOPTION.json'",'actual parent namespace'),("DEST=ROOT/'tmp/celeba_hybrid_native12_root_adoption_20261011'","DEST=ROOT/'tmp/celeba_hybrid_native19_FFlip10_root_adoption_20261011'",'future output namespace'),('exact3','exact7','status metadata'),('EXACT3','EXACT7','status metadata'),('parent9->12','parent12->19','scope metadata'),('range(91001,91004)','range(91004,91011)','exact7 seed range'),("prior_ids=[f'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed{s}_fullcoverage' for s in range(91002,91011)]","prior_ids="+repr(prior_ids),'actual parent ids'),('1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775',sha(priorpath),'actual parent root'),('ROOT_HYBRID_EXACT8_ORIGINAL_STRICT_OFFSERVER_CHAIN_ADOPTED','ROOT_HYBRID_EXACT3_ORIGINAL_STRICT_OFFSERVER_CHAIN_ADOPTED','actual parent status'),('==(1,8,9)','==(9,3,12)','actual parent counts'),('assert not sys.flags.optimize and not DEST.exists()',"import argparse\nparser=argparse.ArgumentParser();parser.add_argument('--actual-pins-sha256',required=True);args=parser.parse_args()\nassert sha(SRC/'ROOT_ACTUAL_PINS.json')==args.actual_pins_sha256\nactual_pins=read(SRC/'ROOT_ACTUAL_PINS.json')\nassert actual_pins['root_adoption_authorized'] is True\nassert not sys.flags.optimize and not DEST.exists()",'required actual result pins'),(old_delivery,"[('DELIVERY_FILES_SHA256.json',actual_pins['delivery_seal_sha256'],actual_pins['delivery_members']),('FILES_SHA256.json',actual_pins['source_seal_sha256'],actual_pins['source_members'])]",'late actual seals'),("=='f952dfeec71f4d849a8e3f51e41baac7fb863674d94f732110bed535d78b1a86'","==actual_pins['handoff_sha256']",'late handoff'),('RECORDS3.json','RECORDS7.json','record namespace'),('==(9,3,12,96,4,100)','==(12,7,19,96,4,100)','scope counts'),("len(set(h['accepted_ids']))==12","len(set(h['accepted_ids']))==19",'cumulative count'),('d8fb8e8e6772b829012062ef8c1b160fae517a056bf8218df61e8de83655558e',prior['offserver_sha256'],'actual previous offserver'),("scope['accepted_before']==9","scope['accepted_before']==12",'parent count'),("=='290bfb0d456d10c83d206f341ce85674971a413f8c264dbfdb003389d9a89cb8'","==actual_pins['archive_sha256']",'late archive'),('==(12,9,3,215)',"==(19,12,7,actual_pins['archive_members'])",'actual archive count'),("receipt['content_members']==214 and receipt['archive_members']==215","receipt['content_members']+1==receipt['archive_members']==actual_pins['archive_members']",'actual dynamic inventory count'),('==215 and all(m.isfile()',"==actual_pins['archive_members'] and all(m.isfile()",'actual safe member count'),("=='027631e7d89f1c4bae86ae987bdd07f1a681baa880b7edd261112440f2a00adf'","==actual_pins['root_source_review_sha256']",'late source review'),(old_outer,"actual_pins['outer_command_sha256']",'late outer actual command hashes'),('old9','old12','parent labels'),("off['accepted_ids'][:9]","off['accepted_ids'][:12]",'parent prefix'),('==(3,24)','==(7,56)','tensor counts'),('new_accepted=3,prior_accepted=9,cumulative_accepted=12','new_accepted=7,prior_accepted=12,cumulative_accepted=19','root counts'),('delivery_members=67',"delivery_members=actual_pins['delivery_members']",'actual member count'),('archive_members=215',"archive_members=actual_pins['archive_members']",'actual archive count'),('saved_tensor_count=24','saved_tensor_count=56','tensor count'),('Exact3 is adopted by this root execution;prior9/reuse4 remain separate and unrepacked;no ten-seed F Flip table is adopted here.','Exact7 is adopted by this root execution;prior12/reuse4 remain separate and unrepacked;IID F Flip10 complete without a scene table;no full100 claim.','scope limitation'),('new_accepted=3,cumulative_accepted=12','new_accepted=7,cumulative_accepted=19','print counts')])
put('SOURCE_DIFF.patch',''.join(diffs));put('SOURCE_ADAPTATIONS.json',maps)
put('INPUT_PINS.json',dict(prior_root_path=str(priorpath.relative_to(R)),prior_root_sha256=sha(priorpath),prior_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),parent_source_seal_sha256=sha(P/'FILES_SHA256.json'),parent_tools={n:sha(P/n) for n in copy_names+['execute_transport.py','finalize_saved.py','adopt_root.py']},package_sha256=prior['package_sha256'],guide_sha256=sha(H/'GUIDE.md'),manifest_path=str(manifestpath.relative_to(R)),manifest_sha256=sha(manifestpath),new_ids=ids,prior_ids=prior_ids,prior_objects_byte_exact=True,reused_separate=4,canaries_separate=7,CPU_candidate=108,future_actual_snapshot_sha256=None,future_authorization_sha256=None,execution_authorized=False))
put('EXECUTION_INPUTS.template.json',dict(status='TEMPLATE_NOT_EXECUTION_AUTHORIZATION',execution_authorized=False,source_seal_sha256=None,source_review_path=None,source_review_sha256=None,snapshot_path=None,snapshot_sha256=None,frozen_observation_sha256=None,allowed_ids=ids))
put('ROOT_ACTUAL_PINS.template.json',dict(status='TEMPLATE_ONLY_NOT_ACCEPTANCE',source_seal_sha256=None,source_members=None,root_source_review_sha256=None,handoff_sha256=None,delivery_seal_sha256=None,delivery_members=None,archive_sha256=None,archive_members=None,outer_command_sha256={f'OUTER_{s}_COMMAND.json':None for s in ['COLLECT','DOWNLOAD','VERIFY','FINALIZE']},adopter_sha256=None,root_adoption_authorized=False))
def block(raw,a,b):return raw[raw.index(a):raw.index(b,raw.index(a))]
for n in ['collect_once.py','verify_saved.py','restore_verify.py','storage.py']:assert raw(H/n)==(P/n).read_bytes()
a=block((P/'finalize_saved.py').read_bytes(),b"for row in off['local']['records']:",b'assert len(tensor_rows)')
b=block(raw(H/'finalize_saved.py'),b"for row in off['local']['records']:",b'assert len(tensor_rows)')
assert a==b and ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False)
a=block((P/'adopt_root.py').read_bytes(),b'for entry,row,tensor,record in zip(',b"assert h['server_runtime']")
b=block(raw(H/'adopt_root.py'),b'for entry,row,tensor,record in zip(',b"assert h['server_runtime']")
assert a==b and ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False)
compiled=[]
for n,v in sorted(dict({q.name:q.read_bytes() for q in H.glob('*.py')},**pending).items()):
 if n.endswith('.py'):compile(v,str(H/n),'exec');compiled.append(n)
put('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_COMPILE_INVERSE_AND_ORIGINAL_SCIENTIFIC_LOOP_REUSE_PASS_NOT_EXECUTED',compiled=compiled,raw_exact_original_tools={n:sha(H/n) for n in ['collect_once.py','verify_saved.py','restore_verify.py','storage.py']},full_inverse_raw_exact=['execute_transport.py','finalize_saved.py','adopt_root.py'],finalizer_tensor_per_record_raw_AST_exact=True,adopter_full_per_record_raw_AST_exact=True,prior12_objects_and_order_preserved_by_byte_exact_copies=True,exact7_manifest_order=True,FedSA_three_completed_excluded=True,original_resource_F_storage_guards_preserved=True,actual_execution_binding_required=True,future_actual_result_pins_unfilled=True,archive_count_dynamic=True,SSH=0,model_array_reads=0,F_write=0,scientific_execution=0,root_adopted=False))
print(json.dumps(dict(status='SOURCE_ONLY_EXACT7_PREPARED_NOT_EXECUTED',ids=ids,excluded=excluded,prior=12,proposed_total=19)))

for n,v in pending.items():
 with (H/n).open('xb') as f:f.write(v)
