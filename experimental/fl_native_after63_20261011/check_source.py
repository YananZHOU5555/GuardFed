"""Stdlib source/metadata fixtures only; fresh resource gates retained in execution entry."""
from pathlib import Path
import ast, datetime, hashlib, json, runpy, subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,obj):
    with (H/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def fn(text,name):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)

for rel,digest in read(H/'SOURCE_PARENT_PINS.json')['files'].items():assert sha(R/rel)==digest
for name in ('run_once.py','collect_delta.py','preflight.py','close_observation.py','root_adopt.py'):
    compile((H/name).read_text('utf-8'),str(H/name),'exec')
ns=runpy.run_path(str(H/'run_once.py'),run_name='safe_source_metadata_only')
ns['source_check']()
new=(H/'collect_delta.py').read_text('utf-8')
old=(R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py').read_text('utf-8')
assert new[new.index('    before=repo_identity'):]==old[old.index('    before=repo_identity'):]
new_fn,old_fn=fn(new,'execute'),fn(old,'execute')
new_loop=next(n for n in new_fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='identity')
old_loop=next(n for n in old_fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='identity')
assert ast.get_source_segment(new,new_loop)==ast.get_source_segment(old,old_loop)
assert ast.dump(new_loop,include_attributes=False)==ast.dump(old_loop,include_attributes=False)
auth=read(H/'AUTHORIZED_SNAPSHOT.json'); ids=auth['authorized_ids']; prior=read(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')['accepted_job_ids']
assert len(prior)==63 and not set(ids)&set(prior)
assert sha(R/auth['snapshot_path'])==auth['snapshot_sha256']
snapshot=read(R/auth['snapshot_path']); latest=read(H/'PREVIOUS_LATEST.json')
assert sha(R/auth['prior_root_path'])==auth['prior_root_sha256']==latest['root_adoption_sha256']
assert read(R/auth['prior_root_path'])['accepted_job_ids']==prior
assert sha(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')==auth['prior_offserver_sha256']==latest['next_collector_previous_sha256']
assert len(snapshot['FLGMM']['terminal_ids'])==67 and set(prior)<=set(snapshot['FLGMM']['terminal_ids'])
assert [i for i in snapshot['FLGMM']['terminal_ids'] if i not in prior]==ids and len(ids)==4
assert not snapshot['FLGMM']['queue_failed'] and not snapshot['FLGMM']['failure_paths']

# Execute the original retained selection/refusal statements with metadata only.
selection=[]
for node in new_fn.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('wanted','authorized_ids') for t in node.targets):selection.append(node)
    if isinstance(node,ast.Assert) and (ast.unparse(node.test).startswith('wanted ==') or
       ast.unparse(node.test).startswith('len(wanted)') or ast.unparse(node.test).startswith("not set(wanted) &") or
       ast.unparse(node.test) in ("not queue_snapshot['failed']","not queue['failed']")):selection.append(node)
code=compile(ast.Module(body=selection,type_ignores=[]),'<original exact-ID exclusion guards>','exec')
def fixture(terminal_ids,previous=None,active=None,failed=False):
    space={'snapshot':{'flgmm':{'rows':[{'id':i,'terminal_acceptance':True,'progress':{'round':70}} for i in terminal_ids]}},
           'previous':{'accepted_job_ids':prior if previous is None else previous},'queue_snapshot':{'failed':failed},
           'queue':{'active':[] if active is None else [{'id':i} for i in active],'failed':failed}}
    exec(code,space)
    return space['wanted']
assert fixture(prior+ids)==ids
assert fixture(prior+ids+['FUTURE_UNAUTHORIZED_TERMINAL'])==ids
refused=[]
for name,kw in [('missing_one',{'terminal_ids':prior+ids[:-1]}),('duplicate',{'terminal_ids':prior+ids+[ids[0]]}),
                ('out_of_order',{'terminal_ids':prior+list(reversed(ids))}),
                ('prior_overlap',{'terminal_ids':prior+ids,'previous':prior+[ids[0]]}),
                ('active',{'terminal_ids':prior+ids,'active':[ids[0]]}),
                ('queue_failed',{'terminal_ids':prior+ids,'failed':True})]:
    try:fixture(**kw)
    except AssertionError:refused.append(name)
    else:raise AssertionError('Guard did not refuse '+name)
# Original source SHA guard, no fake acceptance receipt and no strict checker execution.
try:ns['checked_source'](H/'collect_delta.py','0'*64)
except AssertionError:refused.append('wrong_source_SHA')
else:raise AssertionError('Wrong SHA accepted')
run=(H/'run_once.py').read_text('utf-8')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run_path' for n in ast.walk(ast.parse(run)))
assert 'def parent(' not in run and 'def reused(' not in run
assert 'fit_views(' not in run and 'predict_views(' not in run
save('SOURCE_CHECK.json',dict(status='SOURCE_METADATA_GUARDS_PASS_EXECUTION_RESOURCE_GATES_NOT_RUN',
    exact_selected_ids=ids,prior_count=63,proposed_total=67,scientific_per_ID_loop_bytes_and_AST_exact=True,
    strict_archive_tail_bytes_exact=True,original_offserver_checker_sha256=sha(ns['B']/'verify_delta_offserver.py'),
    fixture_refusals=refused,extra_future_terminal_ignored=True,old63_excluded=True,negative_results_not_filtered=True,
    no_runtime_parent_wrapper_chain=True,fresh_F_gate_performed=False,fresh_CPU109_gate_performed=False,
    collect_verify_finalize_executed=False,no_models_loaded=True,no_fit_CNN_predict_training=True,
    gate_limit='No SSH or resource observation in source preparation; collect performs original fresh guide/all-thread CPU109 preflight and F check, verify performs fresh capacity checks.'))
files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.iterdir()) if p.is_file()}
save('PREPARED_FILES_SHA256.json',dict(status='EXACT4_SOURCE_PREPARED_ROOT_REVIEW_REQUIRED',files=files))
print(json.dumps({'path':H.relative_to(R).as_posix(),'seal_sha256':sha(H/'PREPARED_FILES_SHA256.json'),'source_check_sha256':sha(H/'SOURCE_CHECK.json'),'refusals':refused,'CPU109_measured':False,'scientific_acceptances':0}))
