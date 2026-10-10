"""Prepare exact10 transport from accepted after56 transport; no remote action."""
from pathlib import Path
import ast, copy, difflib, hashlib, importlib.util, json, re, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'tmp/celeba_mechanism_C_after56_root_operations_20261010'
NEW = ROOT/'tmp/celeba_mechanism_C_after60_root_operations_20261010'
PRIOR = ROOT/'tmp/celeba_mechanism_valid_C_after56_20261010'
BASE = ROOT/'tmp/celeba_mechanism_valid_C_after60_20261010'
REVIEW = ROOT/'tmp/celeba_mechanism_C_after60_source_review_20261010/ROOT_INDEPENDENT_REVIEW.json'
REVIEW_SHA = 'ac039e1aeb616eb2a37e1188d36609746a717df1a51d7b28e6a49dc3e34b73da'
SCIENCE = 'ed8ecc84781b799e205c9e139dce8e753cb5545139b7e48ef23d5f8d07a50a77'
EXECUTION = '12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896'
PACKAGE = '5d35280b387c7ff621d94c00fb5983284edf06f272dba4f5a01dc78fb297f824'
RECEIPT = '08aaf7a04ca819b32f1adcf102079a7953f9e928e00bd48ff5c0370e1f17958e'
INVENTORY = 'c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef'
PRIOR_ADOPTION = PRIOR/'execution_candidate/backups/incremental_20261010T005530Z/ROOT_ADOPTION_REVIEW.json'
PRIOR_SHA = '21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e'
expected = [f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
def save(name,value):
    with (NEW/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2);f.write('\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def fn(source,name):
    return ast.get_source_segment(source,next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))

assert not sys.flags.optimize and sha(REVIEW)==REVIEW_SHA and sha(PRIOR_ADOPTION)==PRIOR_SHA
assert read(PRIOR_ADOPTION)['cumulative_three_view_models']==160
for path,pin in [(BASE/'FILES_SHA256.json',SCIENCE),(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json',EXECUTION),(BASE/'PACKAGE_SHA256.json',PACKAGE),(BASE/'PACKAGE_RECEIPT.json',RECEIPT),(BASE/'inventory_actual170_Full100refs.json',INVENTORY)]: assert sha(path)==pin
for folder,seal in [(BASE,'FILES_SHA256.json'),(BASE/'execution_candidate','EXECUTION_SOURCE_SHA256.json'),(BASE,'PACKAGE_SHA256.json')]:
    for row in read(folder/seal)['members']:
        p=folder/row['path'];assert p.is_file() and sha(p)==row['sha256'] and p.stat().st_size==row['size']
scope=read(BASE/'SCOPE.json');handoff=read(BASE/'HANDOFF.json')
assert scope['selected_ids']==expected and len(scope['excluded_prior_ids'])==len(set(scope['excluded_prior_ids']))==160
assert handoff['native_accepted_snapshot']==170 and handoff['new_three_view_accepted']==0
changes={
 'celeba_mechanism_valid_C_after56_20261010':'celeba_mechanism_valid_C_after60_20261010',
 'celeba_mechanism_valid_C_after50_20261010':'celeba_mechanism_valid_C_after56_20261010',
 'C_AFTER56':'C_AFTER60','C_after56':'C_after60','C_AFTER50':'C_AFTER56','C_after50':'C_after56',
 'EXACT4':'EXACT10','exact4':'exact10','old156_records_exact':'old160_records_exact',
 'original156':'original160','prior156':'prior160',
 sha(PRIOR/'FILES_SHA256.json'):SCIENCE,
 sha(PRIOR/'execution_candidate/EXECUTION_SOURCE_SHA256.json'):EXECUTION,
 sha(PRIOR/'PACKAGE_RECEIPT.json'):RECEIPT,
 sha(PRIOR/'PACKAGE_SHA256.json'):PACKAGE,
 sha(PRIOR/'inventory_actual160_Full100refs.json'):INVENTORY,
 'inventory_actual160_Full100refs.json':'inventory_actual170_Full100refs.json',
 repr(read(PRIOR/'SCOPE.json')['selected_ids']):repr(expected),
 "review['native_accepted_snapshot'] == 160":"review['native_accepted_snapshot'] == 170",
 "review['excluded_prior_three_view_ids'] == 156":"review['excluded_prior_three_view_ids'] == 160",
 "len(scope['selected_ids']) == 4":"len(scope['selected_ids']) == 10",
 "len(scope['excluded_prior_ids']) == 156":"len(scope['excluded_prior_ids']) == 160",
 "len(data['completed']) == 4":"len(data['completed']) == 10",
 "len(live['completed'])==4":"len(live['completed'])==10",
 'len(ids)==len(set(ids))==len(expected)==4':'len(ids)==len(set(ids))==len(expected)==10',
 'len(expected)==len(set(expected))==4':'len(expected)==len(set(expected))==10',
 'len(complete)==len(set(complete))==4':'len(complete)==len(set(complete))==10',
 "len(proof['records'])==4":"len(proof['records'])==10",
 "read(prior)['cumulative_three_view_models']==156":"read(prior)['cumulative_three_view_models']==160",
 "len(scope['excluded_prior_ids'])==len(set(scope['excluded_prior_ids']))==156":"len(scope['excluded_prior_ids'])==len(set(scope['excluded_prior_ids']))==160",
 '(36,96,12)':'(90,240,30)',
 'prior_three_view_models=156,accepted_new=4,cumulative_three_view_models=160':'prior_three_view_models=160,accepted_new=10,cumulative_three_view_models=170',
 'incremental_20261010T002635Z':'incremental_20261010T005530Z',
 'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080':PRIOR_SHA,
}
old_seal={r['path']:r['sha256'] for r in read(OLD/'TRANSPORT_REBIND.json')['members']}
before={name:(OLD/name).read_text('utf-8') for name in ('deploy.py','observe.py','backup.py','adopt.py')}
for name in before:assert sha(OLD/name)==old_seal[name]
assert all(any(k in text for text in before.values()) for k in changes)
pattern='|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True))
after={name:re.sub(pattern,lambda m:changes[m.group()],text) for name,text in before.items()}
for name,text in after.items():ast.parse(text);assert text!=before[name]
assert fn(before['adopt.py'],'expected_archive_names')==fn(after['adopt.py'],'expected_archive_names')
assert NEW.is_dir() and {p.name for p in NEW.iterdir()}=={'prepare_operations.py'}
for name,text in after.items():(NEW/name).write_text(text,encoding='utf-8',newline='\n')
(NEW/'SOURCE_DIFF.patch').write_text(''.join(''.join(difflib.unified_diff(before[n].splitlines(True),after[n].splitlines(True),fromfile='after56/'+n,tofile='after60/'+n)) for n in before),encoding='utf-8')
checks=0
for name in ('backup.py','adopt.py'):
    mod=module('after60_'+name[:-3],NEW/name)
    good=dict(service='guardfed_celeba_mechanism_valid_C_after60 EXITED',processes=[],batch_failure=None,batch_complete={'completed':10},completed=[{'id':i} for i in expected])
    mod.check_terminal(good,expected);checks+=1
    for field,value in [('service','RUNNING'),('processes',[1]),('batch_failure',{'failure':True}),('batch_complete',None),('completed',[{'id':expected[0]}]*10),('completed',good['completed'][:-1]),('completed',[{'id':i} for i in expected[:-1]+['wrong_id']])]:
        case=dict(good);case[field]=value
        try:mod.check_terminal(case,expected)
        except AssertionError:checks+=1
        else:raise AssertionError('Terminal refusal failed')
assert checks==16
adopt=module('after60_adopt_layout',NEW/'adopt.py')
names=adopt.expected_archive_names(expected,read(BASE/'FILES_SHA256.json'),read(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json'))
assert len(names)==103 and len([n for n in names if n.startswith(('runs/','logs/','approvals/','runtime/completed_'))])==70
deploy=module('after60_deploy_gate',NEW/'deploy.py')
deploy.validate_source_review(REVIEW,REVIEW_SHA)
review=read(REVIEW);refusals=[]
for key,value in [('native_accepted_snapshot',169),('excluded_prior_three_view_ids',159),('old160_records_exact',False),('positive_approval_exact10',False),('package_sha256','0'*64),('exact_selected_ids',expected[:-1]),('actual_worker_pre_science_bind_ids',list(reversed(expected)))]:
    fixture=copy.deepcopy(review);fixture[key]=value;deploy.read=lambda p,x=fixture:x
    try:deploy.validate_source_review(REVIEW,REVIEW_SHA)
    except AssertionError:refusals.append(key)
    else:raise AssertionError('Review gate refused no drift: '+key)
deploy.read=read
try:deploy.validate_source_review(REVIEW,'0'*64)
except AssertionError:refusals.append('external_review_sha256')
else:raise AssertionError('Wrong external SHA accepted')
# Check the transport-produced authority/draft against the real sealed batch API, in memory only.
batch=module('after60_batch_contract',BASE/'execution_candidate/batch.py')
bound_scope=batch.bind_scope(scope,read(BASE/'execution_candidate/RUNTIME_BINDINGS.json'))
authority=read(BASE/'execution_candidate/ROOT_REVIEW_TEMPLATE.json')
authority.update(status='ROOT_REVIEW_PASS_BOUNDED_C_AFTER60_VALID_REPLAY',execution_authorized_within_existing_user_request=True,execution_seal_sha256=EXECUTION)
authority_bytes=(json.dumps(authority,indent=2)+'\n').encode();fixture_root_sha=hashlib.sha256(authority_bytes).hexdigest()
draft=read(BASE/'execution_candidate/APPROVED_TEMPLATE.json')
draft.update(status='APPROVED_C_AFTER60_MECHANISM_VALID_REPLAY_ONLY',root_approval_sha256=fixture_root_sha,execution_seal_sha256=EXECUTION)
real_read=batch.read;real_digest=batch.digest
batch.read=lambda p:authority if Path(p).name=='ROOT_APPROVED.json' else real_read(p)
batch.digest=lambda p:fixture_root_sha if Path(p).name=='ROOT_APPROVED.json' else real_digest(p)
batch.check_approval(draft,bound_scope,sha(BASE/'SCOPE.json'),EXECUTION)
assert draft['compute_threads']==8 and draft['allowed_cpus']==list(range(112,120)) and draft['max_processes']==1
assert all(x in after['deploy.py'] for x in ["('sglang','STOPPED')","('guardfed_celeba_mechanism_valid_C_after56','EXITED')","'112-119'","'--draft-sha256'"])
assert "'--draft-sha256',required=True" in (BASE/'execution_candidate/install_once.py').read_text('utf-8')
for body in after.values():assert 'utf-4' not in body and 'original156' not in body and 'exact4' not in body and 'EXACT4' not in body
assert not any((BASE/'execution_candidate'/n).exists() for n in ['ROOT_APPROVED.json','EXECUTION_DRAFT.json','deployment_receipt.json','ROOT_DEPLOYMENT_FAILURE.json'])
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
members=[dict(path=n,sha256=sha(NEW/n),accepted_transport_source_sha256=sha(OLD/n)) for n in before]
save('TRANSPORT_REBIND.json',dict(source_only=True,SSH=False,original_science_changed=False,exact10=True,prior160_not_replayed=True,native_snapshot=170,science_seal_sha256=SCIENCE,execution_seal_sha256=EXECUTION,package_sha256=PACKAGE,package_receipt_sha256=RECEIPT,inventory_sha256=INVENTORY,actual_source_review_sha256=REVIEW_SHA,prior160_adoption_sha256=PRIOR_SHA,terminal_positive_and_refusal_checks=checks,archive_member_layout_fixture=103,fixture_metrics=90,fixture_counts=240,fixture_rules=30,source_review_gate_positive=True,source_review_gate_refusals=refusals,preflight_contract_connections_pass=True,authority_draft_fixture_in_memory_only=True,authority_written=False,resource_contract='CPU112-119/8threads/nice10/idleIO/CUDA hidden; original actual Linux preflight remains mandatory',actual_deployment=False,new_three_view_accepted=0,members=members))
(NEW/'README.md').write_text('Source-only exact10 root transport. No SSH, authority, deployment or scientific acceptance was executed. Root must independently review SOURCE_DIFF.patch and actual source review, then invoke deploy.py with --execution-seal, --review and --review-sha256. Original observe.py --progress, backup.py and adopt.py CLI remain unchanged. CPU112-119/8 threads/nice10/idleIO/hidden CUDA and prior after56 EXITED are mandatory. Archive103/metrics90/counts240/rules30 are local layout/count fixtures, not actual outputs. The original expected_archive_names function is byte-exact.\n',encoding='utf-8')
save('FILES_SHA256.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(NEW.iterdir()) if p.is_file()},preparer_sha256=sha(Path(__file__))))
print(json.dumps(dict(status='PREPARED_SOURCE_ONLY_NO_DISPATCH',terminal_checks=checks,archive_fixture_members=len(names),source_review_gate_refusals=len(refusals),preflight_contract_pass=True,transport_rebind_sha256=sha(NEW/'TRANSPORT_REBIND.json'),seal_sha256=sha(NEW/'FILES_SHA256.json'),members=members)))
