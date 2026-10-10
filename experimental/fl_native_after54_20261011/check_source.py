"""Stdlib source/metadata fixtures, fresh storage and read-only CPU ownership gate."""
from pathlib import Path
import ast, datetime, hashlib, json, runpy, subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,obj):
    with (H/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def fn(text,name):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)

for rel,digest in read(H/'SOURCE_PARENT_PINS.json')['files'].items():assert sha(R/rel)==digest
for name in ('run_once.py','collect_delta.py','preflight.py','close_observation.py'):
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
assert len(prior)==54 and not set(ids)&set(prior)
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
storage=ns['storage'](0);save('PREPARED_F_VOLUME.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**storage))
collect=fn(run,'collect')
owner=ast.literal_eval(next(n.value for n in collect.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='owner' for t in n.targets)))
owner="from pathlib import Path\nimport hashlib\nassert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\n"+owner
command=ns['SSH']+['python -B -'];start=datetime.datetime.now(datetime.timezone.utc).isoformat()
cp=subprocess.run(command,input=owner.encode(),capture_output=True,timeout=35)
for name,data in [('PREPARED_OWNER.stdout',cp.stdout),('PREPARED_OWNER.stderr',cp.stderr)]:
    with (H/name).open('xb') as f:f.write(data)
save('PREPARED_OWNER_COMMAND.json',dict(command=command,start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     exit_code=cp.returncode,source_sha256=hashlib.sha256(owner.encode()).hexdigest(),read_only=True))
cp.check_returncode();owner_result=json.loads(cp.stdout)
assert owner_result['CPU110_free'] and not owner_result['busy']
assert owner_result['parent_collector_sha256']==read(H/'SOURCE_REUSE.json')['parent_collector_sha256']
save('PREPARED_OWNER.json',owner_result)
save('SOURCE_CHECK.json',dict(status='SOURCE_METADATA_GUARDS_AND_PREPARATION_GATES_PASS_NOT_EXECUTED',
    exact_selected_ids=ids,prior_count=54,proposed_total=57,scientific_per_ID_loop_bytes_and_AST_exact=True,
    strict_archive_tail_bytes_exact=True,original_offserver_checker_sha256=sha(ns['B']/'verify_delta_offserver.py'),
    fixture_refusals=refused,extra_future_terminal_ignored=True,old54_excluded=True,negative_results_not_filtered=True,
    no_runtime_parent_wrapper_chain=True,fresh_F_gate_pass=True,fresh_CPU110_gate_pass=True,
    collect_verify_finalize_executed=False,no_models_loaded=True,no_fit_CNN_predict_training=True,
    gate_limit='Preparation-time observation only; collect repeats actual guide/all-thread CPU110 preflight and F check, verify repeats fresh capacity checks.'))
files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.iterdir()) if p.is_file()}
save('PREPARED_FILES_SHA256.json',dict(status='EXACT3_SOURCE_PREPARED_ROOT_REVIEW_REQUIRED',files=files))
print(json.dumps({'path':H.relative_to(R).as_posix(),'seal_sha256':sha(H/'PREPARED_FILES_SHA256.json'),'source_check_sha256':sha(H/'SOURCE_CHECK.json'),'refusals':refused,'CPU110_free':True,'scientific_acceptances':0}))
