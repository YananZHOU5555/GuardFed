"""Bounded stdlib-only source/metadata checks; never runs collector or science."""
from pathlib import Path
import ast, difflib, hashlib, json

H=Path(__file__).resolve().parent; R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
pins=read(H/'SOURCE_PARENT_PINS.json'); parent=R/pins['parent_directory']
for name,digest in pins['files'].items():assert sha(R/name)==digest,name
assert sha(parent/'PREPARED_FILES_SHA256.json')==pins['parent_prepared_seal_sha256']
scope=read(H/'FROZEN_SCOPE.json'); ids=scope['ids']
prior=read(H/'PREVIOUS_ROOT_ADOPTION.json')['accepted_job_ids']
latest=read(H/'PREVIOUS_LATEST.json')
assert len(prior)==len(set(prior))==86 and len(ids)==len(set(ids))==10 and not set(prior)&set(ids)
assert scope['prior_accepted_job_ids']==prior
assert [j['seed'] for j in scope['jobs']]==list(range(91001,91011))
assert all((j['distribution'],j['attack'])==('non-IID','Sp-DFA') for j in scope['jobs'])
assert sha(H/'PREVIOUS_LATEST.json')==pins['previous_latest_sha256']
assert sha(H/'PREVIOUS_ROOT_ADOPTION.json')==latest['root_adoption_sha256']==pins['previous_root_sha256']
assert latest['next_collector_previous_sha256']==pins['previous_offserver_sha256']
manifest=read(R/scope['original_manifest_path'])
assert sha(R/scope['original_manifest_path'])==scope['original_manifest_sha256']
assert len(manifest['jobs'])==96 and len(manifest['reused_jobs'])==4
assert set(prior)|set(ids)=={j['id'] for j in manifest['jobs']}
pending=read(H/'PENDING_BINDING.json')
assert pending['snapshot_path'] is pending['snapshot_sha256'] is pending['terminal_count'] is None
assert pending['actual_archive_sha256'] is pending['actual_offserver_sha256'] is pending['actual_root_adoption_sha256'] is None
assert not pending['execution_authorized'] and not pending['scientific_execution_performed']
assert not (H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json').exists()
assert not any((H/n).exists() for n in ('AUTHORIZED_SNAPSHOT.json','SERVER_COLLECTOR_RECEIPT.json','ROOT_READY_HANDOFF.json','ROOT_ADOPTION_REVIEW.json'))

def function(raw,name):
    return next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name)
def raw_node(raw,node):return b''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
def same_node(a,b,an,bn):
    assert raw_node(a,an)==raw_node(b,bn)
    assert ast.dump(an,include_attributes=False)==ast.dump(bn,include_attributes=False)
    return hashlib.sha256(raw_node(a,an)).hexdigest()

compiled=[]
for n in ('run_once.py','collect_delta.py','preflight.py','close_observation.py','root_adopt.py','prepare_source.py','check_source.py'):
    raw=(H/n).read_bytes();assert b'\r' not in raw
    compile(raw,str(H/n),'exec');compiled.append(n)
inverse=read(H/'SOURCE_INVERSE.json')
for n,delta in inverse.items():
    assert ''.join(difflib.restore(delta,1)).encode()==(parent/n).read_bytes()
    assert ''.join(difflib.restore(delta,2)).encode()==(H/n).read_bytes()
    assert sha(H/n)==pins['source_changes'][n]['source_sha256']

new=(H/'collect_delta.py').read_bytes()
old=(R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py').read_bytes()
assert new[new.index(b'    before=repo_identity'):]==old[old.index(b'    before=repo_identity'):]
def loop(raw,target):
    return next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id==target)
science_hash=same_node(new,old,loop(new,'identity'),loop(old,'identity'))
ad=(H/'root_adopt.py').read_bytes();oldad=(parent/'root_adopt.py').read_bytes()
record_hash=same_node(ad,oldad,loop(ad,'record'),loop(oldad,'record'))
member_hash=same_node(ad,oldad,loop(ad,'m'),loop(oldad,'m'))
transport={}
a=(H/'run_once.py').read_bytes();b=(parent/'run_once.py').read_bytes()
for n in ('source_check','storage','run','checked_source','replace_exact','save'):
    transport[n]=same_node(a,b,function(a,n),function(b,n))
thread_gates={}
for n in ('preflight.py','close_observation.py'):
    a,b=(H/n).read_bytes(),(parent/n).read_bytes()
    thread_gates[n]=same_node(a,b,loop(a,'t'),loop(b,'t'))
pre=(H/'preflight.py').read_bytes();close=(H/'close_observation.py').read_bytes()
for n in ('service_contract','service_gate'):same_node(pre,close,function(pre,n),function(close,n))

# Execute only isolated pure metadata assertions, without any filesystem/import helpers.
ns={};contract=function(pre,'service_contract')
exec(compile(ast.Module(body=[contract],type_ignores=[]),'<service metadata contract>','exec'),ns)
expected=[j['id'] for j in manifest['jobs']]
def gate(**changes):
    values=dict(name='guardfed_celeba_flgmm_fullcoverage',rc=3,state='EXITED',exitstatus=0,worker_count=0,queue=dict(completed_new=96,pending=0,active=[],failed=False),expected_ids=expected,completed_ids=expected)
    values.update(changes);return ns['service_contract'](**values)
assert gate().startswith('NORMAL_EXITED')
assert gate(rc=0,state='RUNNING',exitstatus=None,worker_count=1).startswith('ORIGINAL_RUNNING')
assert gate(rc=0,state='RUNNING',exitstatus=None,worker_count=2).startswith('ORIGINAL_RUNNING')
refused=[]
cases=[('wrong_service',dict(name='other_service')),('exited_rc0',dict(rc=0)),('nonzero_exitstatus',dict(exitstatus=1)),('exited_worker',dict(worker_count=1)),('missing_terminal',dict(completed_ids=expected[:-1])),('duplicate_terminal',dict(completed_ids=expected[:-1]+expected[:1])),('pending',dict(queue=dict(completed_new=96,pending=1,active=[],failed=False))),('active',dict(queue=dict(completed_new=96,pending=0,active=[dict(id=ids[0])],failed=False))),('failed',dict(queue=dict(completed_new=96,pending=0,active=[],failed=True))),('completed_count95',dict(queue=dict(completed_new=95,pending=0,active=[],failed=False))),('running_without_worker',dict(rc=0,state='RUNNING',worker_count=0)),('running_three_workers',dict(rc=0,state='RUNNING',worker_count=3)),('running_rc3',dict(rc=3,state='RUNNING',worker_count=1)),('fatal',dict(state='FATAL'))]
for name,values in cases:
    try:gate(**values)
    except AssertionError:refused.append(name)
    else:raise AssertionError('Not refused: '+name)

# Retained frozen-ID selection asserts only; no accepted()/Torch/model invocation.
selection=[]
for node in function(new,'execute').body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('wanted','authorized_ids') for t in node.targets):selection.append(node)
    if isinstance(node,ast.Assert) and (ast.unparse(node.test).startswith(('wanted ==','len(wanted)','not set(wanted) &')) or ast.unparse(node.test) in ("not queue_snapshot['failed']","not queue['failed']")):selection.append(node)
selection_code=compile(ast.Module(body=selection,type_ignores=[]),'<retained selection assertions>','exec')
def select(terminal,previous=prior,active=(),failed=False):
    values=dict(snapshot={'flgmm':{'rows':[dict(id=i,terminal_acceptance=True,progress={'round':70}) for i in terminal]}},previous={'accepted_job_ids':previous},queue_snapshot={'failed':failed},queue={'active':[{'id':i} for i in active],'failed':failed})
    exec(selection_code,values);return values['wanted']
assert select(prior+ids)==ids and select(prior+ids+['FUTURE_UNAUTHORIZED'])==ids
selection_refused=[]
for name,values in [('missing',dict(terminal=prior+ids[:-1])),('duplicate',dict(terminal=prior+ids+[ids[0]])),('order',dict(terminal=prior+list(reversed(ids)))),('overlap',dict(terminal=prior+ids,previous=prior+[ids[0]])),('active',dict(terminal=prior+ids,active=[ids[0]])),('failed',dict(terminal=prior+ids,failed=True))]:
    try:select(**values)
    except AssertionError:selection_refused.append(name)
    else:raise AssertionError('Selection not refused: '+name)

run=(H/'run_once.py').read_bytes().decode()
assert "a['prior_count']==86 and a['terminal_count']==96" in run
assert "review['authorization_sha256']==sha(H/'AUTHORIZED_SNAPSHOT.json')" in run
assert "not q['queue_failed'] and not q['failure_paths']" in run
tensor=R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/check_saved_tensors.py'
assert sha(tensor)=='877ab0396a9814fd8c7cb41f1fce03ebdf9b84a04b410edeb5b8663e882e3b8d'
compile(tensor.read_text('utf8').replace("proof['accepted_new']==2 and proof['accepted_total']==16","proof['accepted_new']==10 and proof['accepted_total']==96").replace("H/'batch/","S/'"),'<original tensor checker future count/F metadata only>','exec')
assert sha(R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/verify_delta_offserver.py')==pins['original_verifier_sha256']
result=dict(status='SOURCE_ONLY_COMPILE_RAW_INVERSE_AND_METADATA_GUARDS_PASS_NOT_SCIENTIFIC_ACCEPTANCE',exact_selected_ids=ids,prior_native_new=86,proposed_native_new_if_root_adopts=96,screen_reused_separately=4,new_native_accepted=0,new_three_view_accepted=0,compiled=compiled,full_five_source_raw_inverse_exact=True,strict_and_archive_tail_raw_bytes_exact=True,per_ID_scientific_loop_raw_bytes_and_AST_exact=science_hash,root_per_record_loop_raw_bytes_and_AST_exact=record_hash,root_member_loop_raw_bytes_and_AST_exact=member_hash,transport_functions_raw_bytes_and_AST_exact=transport,CPU110_all_thread_gate_raw_bytes_and_AST_exact=thread_gates,service_gate_identical_in_preflight_and_close=True,service_contract_metadata_positive_cases=3,service_contract_refusals=refused,fixed_selection_metadata_positive=True,future_unauthorized_terminal_not_selected=True,fixed_selection_refusals=selection_refused,original_offserver_verifier_sha256=pins['original_verifier_sha256'],actual_snapshot_and_previous_F_copy_pending=True,actual_supervisor_rpc_and_CPU110_and_F_guards_not_run=True,collector_verify_close_finalize_adopt_executed=False,Torch_or_models_imported_or_loaded=False,scientific_execution=False)
with (H/'SOURCE_CHECK.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],source_check_sha256=sha(H/'SOURCE_CHECK.json'))))
