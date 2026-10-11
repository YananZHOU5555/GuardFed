"""Prepare only parameter pins for original generic NGA incremental tools."""
from pathlib import Path
import json,hashlib,copy,importlib.util
R=Path.cwd();H=Path(__file__).resolve().parent;P=R/'tmp/NGA96_incremental_acceptance_prepare_20261011';B=R/'tmp/NGA96_IID_Benign9_after1_prepare_20261011'
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
rootp=R/'tmp/NGA96_IID_Benign9_root_adopter_prepare_20261011/ROOT_ADOPTION.json';root=read(rootp);offp=R/root['offserver']['path'];off=read(offp)
assert root['root_adopted'] and not root['test'] and root['accepted_total']==off['accepted_total']==9 and root['accepted_ids']==off['accepted_job_ids']
assert sha(offp)==root['offserver']['sha256'];contract=read(P/'SCOPE.json');pkg=R/contract['source_local'];manifest=read(pkg/'jobs/manifest.json');assert sha(pkg/'jobs/manifest.json')==contract['expected_manifest_sha256']
selected=[]
for j in manifest['new_jobs']:
 if '_IID_F Flip_' not in j['id']:continue
 jp=pkg/'jobs'/j['job'];assert sha(jp)==j['sha256'];job=read(jp);cfg=job['config']
 assert job['id']==j['id'] and job['distribution']=='IID' and job['attack']=='F Flip' and job['method']=='Fed-NGA-gradient' and cfg['rounds']==70 and cfg['client_alpha']==5000 and cfg['celeba_evaluation_split']=='valid'
 selected.append(dict(id=j['id'],job_path=jp.relative_to(R).as_posix(),job_sha256=sha(jp),seed=cfg['seed'],client_alpha=cfg['client_alpha'],attack=job['attack'],rounds=cfg['rounds'],evaluation_split=cfg['celeba_evaluation_split']))
ids=[j['id'] for j in selected];assert len(ids)==10 and [j['seed'] for j in selected]==list(range(91001,91011));assert not set(ids)&set(root['accepted_ids'])
assert [i for i in contract['new_ids'] if i in ids]==ids
files={}
for n in ['run_once.py','collect_delta.py','pinned_collect_one.py','verify_delta.py']:
 assert (P/n).read_bytes()==(B/n).read_bytes();compile((P/n).read_bytes(),n,'exec');files[n]={'path':(B/n).relative_to(R).as_posix(),'sha256':sha(B/n)}
# Reuse the existing generic scope, not the previous stage's exact8-after1 wrapper.
for n in ['scope.py','SCOPE.json','DEPENDENCIES.json']:files[n]={'path':(P/n).relative_to(R).as_posix(),'sha256':sha(P/n)}
proposal=dict(status='PARAMETERS_ONLY_NOT_EXECUTABLE_OR_AUTHORIZED',stage=H.name,exact_new_ids=ids,new_jobs=selected,accepted_prior_ids=root['accepted_ids'],accepted_prior_count=9,conditional_total_after_strict_offserver_and_root=19,reused_screen4_separate=True,source_files_to_materialize_byte_exact=files,previous_proof_files={'PREVIOUS_ROOT_ADOPTION.json':{'path':rootp.relative_to(R).as_posix(),'sha256':sha(rootp)},'PREVIOUS_OFFSERVER_ACCEPTANCE.json':{'path':offp.relative_to(R).as_posix(),'sha256':sha(offp)}},local_runtime_directory=H.relative_to(R).as_posix(),server_namespace='/workspace/guardfed_checks/'+H.name,F_namespace='F:/YananResearchStorage/GuardFed/'+H.name+'/batch',no_core_code_changes=True,scientific_execution=False,actual_terminal_observation=None,actual_review=None,archive=None,new_offserver_acceptance=None,new_root_adoption=None)
put('ROOT_PARAMETERS_PROPOSAL.json',proposal)
a=read(P/'AUTHORIZED_SNAPSHOT.json');a.update(status='UNBOUND_TEMPLATE_NOT_EXECUTION_AUTHORITY',authorized_ids=ids,observed_terminal_ids=None,observed_failure_paths=None,accepted_prior_ids=root['accepted_ids'],accepted_prior_count=9,snapshot_path=None,snapshot_sha256=None,prior_root_path=rootp.relative_to(R).as_posix(),prior_root_sha256=sha(rootp),prior_offserver_path=offp.relative_to(R).as_posix(),prior_offserver_sha256=sha(offp));put('AUTHORIZED_SNAPSHOT.template.json',a)
put('ROOT_REVIEW.template.json',dict(status='UNBOUND_TEMPLATE_NOT_EXECUTION_AUTHORITY',root_adoption_performed=False,source_files_sha256=None,authorization_sha256=None,exact_new_ids=ids,parameter_proposal_sha256=sha(H/'ROOT_PARAMETERS_PROPOSAL.json')))
# Pure stdlib metadata validation with synthetic observation and mocked reads of byte-identical prior-copy destinations. No files/arrays loaded from F.
sp=importlib.util.spec_from_file_location('_generic_NGA_scope_source_only',P/'scope.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
expected={str(H/'PREVIOUS_ROOT_ADOPTION.json'):rootp,str(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json'):offp}
m.sha=lambda p:sha(expected[str(p)]);m.read=lambda p:read(expected[str(p)])
s=copy.deepcopy(a);s.update(status='ROOT_BOUND_EXACT_NGA96_DELTA_NOT_ACCEPTED',observed_terminal_ids=ids,observed_failure_paths=[],snapshot_path='synthetic-metadata-fixture-only',snapshot_sha256='1'*64)
prior,previous=m.validate(s,contract,H);assert prior['accepted_total']==9;checks=['original four collector/verify/transport/close files byte exact to successful prior','existing generic scope supports new10 after actual9 without code changes','fixed jobs selected from original manifest actual config; IID alpha5000 F Flip 70 valid seeds91001..10','prior actual9 root/offserver pins and ordered identities match','synthetic metadata scope positive passes; not a terminal observation']
for label,fn in [('prior overlap',lambda q:q['authorized_ids'].append(root['accepted_ids'][0])),('not observed',lambda q:q.update(observed_terminal_ids=ids[:-1])),('failure present',lambda q:q.update(observed_failure_paths=['synthetic_failure'])),('wrong prior root',lambda q:q.update(prior_root_sha256='0'*64)),('reordered delta',lambda q:q['authorized_ids'].reverse()),('wrong denominator',lambda q:q.update(accepted_prior_count=10))]:
 q=copy.deepcopy(s);fn(q)
 try:m.validate(q,contract,H)
 except ValueError:checks.append('reject '+label)
 else:raise AssertionError(label)
put('SOURCE_PARAMETER_CHECK.json',dict(status='PASS_PARAMETERS_AND_PURE_METADATA_ONLY',checks=checks,scientific_imports=0,model_reads=0,SSH=0,F_access=0,newly_accepted=0,actual_observation=None))
print(json.dumps({'ids':ids,'prior_root_sha256':sha(rootp),'proposal_sha256':sha(H/'ROOT_PARAMETERS_PROPOSAL.json')}))
