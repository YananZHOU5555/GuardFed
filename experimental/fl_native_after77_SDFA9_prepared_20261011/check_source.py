"""One bounded stdlib source/metadata check; never dispatches or runs strict science."""
from pathlib import Path
import ast, hashlib, json

H=Path(__file__).resolve().parent; R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,obj):
    with (H/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n')
def function(text,name):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
def raw_node(raw,node):
    return b''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])

pins=read(H/'SOURCE_PARENT_PINS.json')
for name,digest in pins['files'].items(): assert sha(R/name)==digest,name
scope=read(H/'FROZEN_SCOPE.json'); ids=scope['ids']
prior=read(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')['accepted_job_ids']
latest=read(H/'PREVIOUS_LATEST.json')
assert len(prior)==len(set(prior))==77 and len(ids)==9 and not set(prior)&set(ids)
assert [r['seed'] for r in scope['jobs']]==list(range(91002,91011))
assert all(r['attack']=='S-DFA' and r['distribution']=='non-IID' for r in scope['jobs'])
assert sha(H/'PREVIOUS_LATEST.json')==pins['previous_latest_sha256']
assert sha(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')==latest['next_collector_previous_sha256']==pins['previous_offserver_sha256']
assert read(R/latest['root_adoption_path'])['accepted_job_ids']==prior
assert not (H/'AUTHORIZED_SNAPSHOT.json').exists()
pending=read(H/'PENDING_BINDING.json')
assert pending['snapshot_path'] is pending['snapshot_sha256'] is pending['terminal_count'] is None
assert not pending['execution_authorized'] and not pending['scientific_execution_performed']

reb={'parent_directory':pins['parent_directory']}
for n in ('run_once.py','collect_delta.py','preflight.py','close_observation.py','root_adopt.py','prepare_source.py','check_source.py'):
    compile((H/n).read_bytes(),str(H/n),'exec')
assert sha(R/reb['parent_directory']/'PREPARED_FILES_SHA256.json')==pins['parent_prepared_seal_sha256']
newraw=(H/'collect_delta.py').read_bytes(); oldraw=(R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py').read_bytes()
new,old=newraw.decode('utf-8'),oldraw.decode('utf-8')
assert newraw[newraw.index(b'    before=repo_identity'):]==oldraw[oldraw.index(b'    before=repo_identity'):]
newfn,oldfn=function(new,'execute'),function(old,'execute')
def identity_loop(fn):
    return next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='identity')
nl,ol=identity_loop(newfn),identity_loop(oldfn)
assert raw_node(newraw,nl)==raw_node(oldraw,ol)
assert ast.dump(nl,include_attributes=False)==ast.dump(ol,include_attributes=False)
preserved={}
parent=R/reb['parent_directory']
for name in ('source_check','storage','run','checked_source','replace_exact','save'):
    a=(H/'run_once.py').read_bytes(); b=(parent/'run_once.py').read_bytes()
    an,bn=function(a.decode(),name),function(b.decode(),name)
    assert raw_node(a,an)==raw_node(b,bn)
    assert ast.dump(an,include_attributes=False)==ast.dump(bn,include_attributes=False)
    preserved[name]=hashlib.sha256(raw_node(a,an)).hexdigest()
ad=(H/'root_adopt.py').read_bytes(); oldad=(parent/'root_adopt.py').read_bytes()
def record_loop(raw):
    return next(n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='record')
al,bl=record_loop(ad),record_loop(oldad)
assert raw_node(ad,al)==raw_node(oldad,bl) and ast.dump(al,include_attributes=False)==ast.dump(bl,include_attributes=False)

# Original exact-ID guards only, on synthetic metadata; no accepted()/Torch/model calls.
selection=[]
for node in newfn.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('wanted','authorized_ids') for t in node.targets): selection.append(node)
    if isinstance(node,ast.Assert) and (ast.unparse(node.test).startswith('wanted ==') or ast.unparse(node.test).startswith('len(wanted)') or ast.unparse(node.test).startswith('not set(wanted) &') or ast.unparse(node.test) in ("not queue_snapshot['failed']","not queue['failed']")): selection.append(node)
code=compile(ast.Module(body=selection,type_ignores=[]),'<retained exact-ID guards>','exec')
def fixture(terminal,previous=prior,active=(),failed=False):
    ns={'snapshot':{'flgmm':{'rows':[dict(id=i,terminal_acceptance=True,progress={'round':70}) for i in terminal]}},'previous':{'accepted_job_ids':previous},'queue_snapshot':{'failed':failed},'queue':{'active':[{'id':i} for i in active],'failed':failed}}
    exec(code,ns); return ns['wanted']
assert fixture(prior+ids)==ids and fixture(prior+ids+['FUTURE_UNAUTHORIZED_TERMINAL'])==ids
refused=[]
for name,kw in [('missing_one',dict(terminal=prior+ids[:-1])),('duplicate',dict(terminal=prior+ids+[ids[0]])),('out_of_order',dict(terminal=prior+list(reversed(ids)))),('prior_overlap',dict(terminal=prior+ids,previous=prior+[ids[0]])),('active',dict(terminal=prior+ids,active=[ids[0]])),('queue_failed',dict(terminal=prior+ids,failed=True))]:
    try: fixture(**kw)
    except AssertionError: refused.append(name)
    else: raise AssertionError('Guard did not refuse '+name)
run=(H/'run_once.py').read_text('utf-8')
assert "review['authorization_sha256']==sha(H/'AUTHORIZED_SNAPSHOT.json')" in run
assert "set(IDS)<=set(q['terminal_ids'])" in run and "not q['queue_failed'] and not q['failure_paths']" in run
assert 'taskset -c 110' in run and 'CPU110_free' in run and 'CPU110_released' in run
assert "H/'ACTUAL_README.md'" in run
tensorpath=R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/check_saved_tensors.py'
assert sha(tensorpath)=='877ab0396a9814fd8c7cb41f1fce03ebdf9b84a04b410edeb5b8663e882e3b8d'
tensor=tensorpath.read_text('utf8');oldline="proof['accepted_new']==2 and proof['accepted_total']==16"
assert tensor.count(oldline)==1
adapted=tensor.replace(oldline,"proof['accepted_new']==9 and proof['accepted_total']==86").replace("H/'batch/","S/'")
compile(adapted,'<future original tensor checker with metadata counts/path only>','exec')
assert "{77+len(ids)}" in run and "{67+len(ids)}" not in run
assert "h['archive_members'] == receipt['archived_member_count'] == off['archived_member_count']" in ad.decode()
assert 'archive_members_verified=len(listed)' in ad.decode() and '== 107' not in ad.decode()
assert scope['screen91001_excluded_from_new_native']['id'].endswith('_seed91001_screen')
assert all('_seed91001_' not in i for i in ids)
assert not any((H/n).exists() for n in ['ROOT_READY_HANDOFF.json','ROOT_EXECUTION_REVIEW.json','SERVER_COLLECTOR_RECEIPT.json','ROOT_ADOPTION_REVIEW.json'])
save('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_COMPILE_RAW_REUSE_AND_METADATA_GUARDS_PASS_NOT_SCIENTIFIC_ACCEPTANCE',exact_selected_ids=ids,prior_native_new=77,proposed_native_new_if_root_adopts=86,screen_reused_separately=4,new_native_accepted=0,new_three_view_accepted=0,compiled=['run_once.py','collect_delta.py','preflight.py','close_observation.py','root_adopt.py','check_source.py'],scope_diff_sha256=sha(H/'SOURCE_DIFF.patch'),scientific_per_ID_loop_raw_bytes_and_AST_exact=True,strict_archive_tail_raw_bytes_exact=True,root_adopter_per_record_loop_raw_bytes_and_AST_exact=True,original_resource_and_transport_functions_raw_bytes_and_AST_exact=preserved,original_offserver_checker_sha256=pins['files']['tmp/celeba_flgmm_fullcoverage_incremental_20261009/verify_delta_offserver.py'],fixture_positive_exact9=True,fixture_extra_future_terminal_excluded=True,fixture_refusals=refused,actual_authorized_snapshot_absent=True,actual_snapshot_bound=False,fresh_execution_CPU110_gate_performed=False,fresh_execution_F_storage_gate_performed=False,collect_download_verify_close_finalize_adopt_executed=False,models_arrays_loaded=False,scientific_execution_performed=False))
print(json.dumps(dict(status='SOURCE_PREPARED_NOT_EXECUTED',source_check_sha256=sha(H/'SOURCE_CHECK.json'),new_accepted=0)))
