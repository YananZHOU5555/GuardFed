"""Local metadata-only preparation; no imports of scientific or execution modules."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'tmp/hybrid_native_after1_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
def copy(p,n):
 with (H/n).open('xb') as f:f.write(p.read_bytes())
def write(n,s):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:f.write(s)
P=R/'tmp/celeba_hybrid_native9_root_adoption_20261011/ROOT_ADOPTION.json'
PS='1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775'
assert sha(P)==PS
parent=read(P);assert parent['status']=='ROOT_HYBRID_EXACT8_ORIGINAL_STRICT_OFFSERVER_CHAIN_ADOPTED' and parent['cumulative_accepted']==9
assert sha(O/'FILES_SHA256.json')==parent['source_seal_sha256']=='32c45838444f673ccecc7424f5f541a3e23e308c2e0a917e2129fd319b2634c3'
for n,pin in read(O/'FILES_SHA256.json')['files'].items():assert sha(O/n)==pin['sha256'] and (O/n).stat().st_size==pin['bytes']
assert sha(O/'OFFSERVER_VERIFICATION.json')==parent['offserver_sha256']=='d8fb8e8e6772b829012062ef8c1b160fae517a056bf8218df61e8de83655558e'
assert read(O/'OFFSERVER_VERIFICATION.json')['accepted_ids']==parent['accepted_ids']
L=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261010T180131Z.raw.json'
assert sha(L)=='59f180458fd5fcf06dd2065a80b14150aa3ffc490928eb5bd989963ea7410884'
live=read(L)['Hybrid96'];rows=live['terminal_records'];prior=parent['accepted_ids']
IDS=[x['id'] for x in rows if x['id'] not in prior]
assert IDS==[f'CosineFairness_lam20.0_tau0.1_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in range(91001,91004)]
assert len(rows)==len({x['id'] for x in rows})==12 and set(prior).issubset({x['id'] for x in rows})
selected=[x for x in rows if x['id'] in IDS];assert all(x['acceptance']['status']=='PASS' for x in selected)
copy(P,'PRIOR_ROOT_ADOPTION.json');copy(O/'OFFSERVER_VERIFICATION.json','PRIOR_OFFSERVER.json')
names=['collect_once.py','restore_verify.py','verify_saved.py','storage.py','ROOT_BOUND_ADOPTION.json','ROOT_SEVEN_CANARY_CLOSURE.json','ROOT_COVERAGE_STARTUP.json','GUIDE.md']
for n in names:copy(O/n,n)
remote='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/native_after9_20261011'
bulk='F:/YananResearchStorage/GuardFed/hybrid_native_after9_20261011'
put('DELTA_SCOPE.json',dict(status='SOURCE_ONLY_FIXED_EXACT3_NOT_EXECUTED',ids=IDS,accepted_before=9,prior_offserver_path='PRIOR_OFFSERVER.json',prior_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),remote_output=remote+'/output',local_bulk_root=bulk,no_later_completions=True))
put('FROZEN_OBSERVATION.json',dict(source_path=L.relative_to(R).as_posix(),sha256=sha(L),observation_utc=live['checked_utc'],terminal_count=12,selected_records=selected,selected_ids=IDS,prior_ids=prior,not_new_acceptance=True))
s=(O/'execute_transport.py').read_text('utf8')
s=s.replace('hybrid_native_after1_20261011','hybrid_native_after9_20261011').replace('/native_after1_20261011','/native_after9_20261011')
s=s.replace('04d62c367609d1c6d079ff538f45a575bff368944d4833f47fd1cf28159c5fa4',PS)
s=s.replace("['cumulative_accepted']==1","['cumulative_accepted']==9").replace("['accepted_before']==1","['accepted_before']==9")
old=read(O/'DELTA_SCOPE.json')['ids'];assert repr(old) in s;s=s.replace(repr(old),repr(IDS))
s=s.replace("read(H/'PRIOR_ROOT_ADOPTION.json')['accepted_new_ids']","read(H/'PRIOR_ROOT_ADOPTION.json')['accepted_ids']")
write('execute_transport.py',s)
put('INPUT_PINS.json',dict(prior_root_path=P.relative_to(R).as_posix(),prior_root_sha256=PS,prior_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),original_source_seal_sha256=sha(O/'FILES_SHA256.json'),original_tool_hashes={n:sha(O/n) for n in ['collect_once.py','verify_saved.py','storage.py','restore_verify.py','execute_transport.py','finalize_delivery.py']},package_sha256=parent['package_sha256'],guide_sha256=sha(H/'GUIDE.md'),new_ids=IDS,prior_ids=prior,prior_offserver_records_unchanged=True,reused_separate=4,canaries_separate=7,CPU_candidate=108))
diff=''.join(difflib.unified_diff((O/'execute_transport.py').read_text('utf8').splitlines(True),s.splitlines(True),fromfile='after1/execute_transport.py',tofile='after9/execute_transport.py'))
write('SOURCE_DIFF.patch',diff)
compiled=[]
for p in H.glob('*.py'):compile(p.read_text('utf8'),str(p),'exec');compiled.append(p.name)
exact={n:sha(H/n) for n in ['collect_once.py','restore_verify.py','verify_saved.py','storage.py'] if (H/n).read_bytes()==(O/n).read_bytes()};assert len(exact)==4
science=ast.parse((H/'verify_saved.py').read_text('utf8'));science_pins={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in science.body if isinstance(n,ast.FunctionDef)}
put('SOURCE_CHECK.json',dict(status='PASS_SOURCE_ONLY_NOT_SCIENTIFIC_ACCEPTANCE',compiled=sorted(compiled),byte_exact_original_tools=exact,verify_saved_function_AST_sha256=science_pins,original_collector_fresh_guide_source_CPU_affinity_quota_RAM_guards_unchanged=True,original_F_label_health_reserve_guards_unchanged=True,transport_diff_scope=['fresh E/F/remote namespace','actual prior9 root SHA/count','exact3 frozen IDs','prior accepted_ids rather than only last-batch accepted_new_ids'],old9_objects_preserved_by_exact_parent_bytes=True,expected_after_if_success=12,SSH=0,CNN=0,fit=0,training=0,root_adopted=False))
print(json.dumps(dict(status='SOURCE_PREPARED_NOT_EXECUTED',new_ids=IDS,compiled=compiled,exact_tools=exact),indent=2))
