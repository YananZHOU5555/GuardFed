"""Review the exact accepted C11 complement and derive one-shot root operations."""
from pathlib import Path, PurePosixPath
import ast, copy, datetime, hashlib, importlib.util, json, sys, tarfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'tmp/celeba_mechanism_valid_C1_gate_20261009'
BASE=ROOT/'tmp/celeba_mechanism_valid_C_after1_20261009'
EX=BASE/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
functions=lambda p:{n.name:ast.get_source_segment(p.read_text(encoding='utf8'),n) for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)}
assert sha(BASE/'PACKAGE_RECEIPT.json')=='ee5a2e6c5a3ec30cc8ae35352ce5d9161fc441e0aaa67cda461804bba3f730cd'
package=read(BASE/'PACKAGE_RECEIPT.json')
assert sha(BASE/'prepared_source.tar.gz')==package['source_archive_sha256']=='4d5050c4d52e40227d2308f0755678d79f059ea561bf2168a5ae2cfafc9b43f2'
with tarfile.open(BASE/'prepared_source.tar.gz') as bundle:
 assert len(bundle.getnames())==len(set(bundle.getnames()))==29
 assert set(bundle.getnames())=={BASE.name+'/'+name for name in package['members']}
 for name,pin in package['members'].items():
  item=bundle.getmember(BASE.name+'/'+name);rel=PurePosixPath(item.name)
  assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
  payload=bundle.extractfile(item).read()
  assert len(payload)==pin['bytes'] and hashlib.sha256(payload).hexdigest()==pin['sha256']==sha(BASE/name)
for name,pin in read(BASE/'INPUT_PINS.json').items():
 assert sha(ROOT/name)==pin['sha256'] and (ROOT/name).stat().st_size==pin['bytes']
prior=OLD/'execution_candidate/backups/incremental_20261009T185829Z/ROOT_ADOPTION_REVIEW.json'
assert sha(prior)=='d045665b066dafc25f9970adfdffef9c9a8a388575ec87b9b54d5dcabfa65cab' and read(prior)['cumulative_three_view_models']==101
scope=read(BASE/'SCOPE.json');old_inv=read(OLD/'inventory_actual101_Full100refs.json');inv=read(BASE/'inventory_actual112_Full100refs.json')
selected=[f'minus_C_IID_Benign_seed{s}' for s in range(91002,91011)]+[f'minus_C_IID_F Flip_seed{s}' for s in (91001,91002)]
assert scope['selected_ids']==selected and len(scope['excluded_prior_ids'])==101
assert set(scope['excluded_prior_ids'])=={r['id'] for r in old_inv['records']}
indexed={r['id']:r for r in inv['records']}
assert len(indexed)==112 and all(indexed[r['id']]==r for r in old_inv['records']) and inv['full_references']==old_inv['full_references']
assert set(indexed)-set(scope['excluded_prior_ids'])==set(selected)
archives={}
for identity in selected:
 row=indexed[identity]
 assert row['variant']=='minus_C' and row['config']['ablation_component']=='C' and row['terminal_round']==70
 assert row['original_split']=='valid' and row['original_n_eval']==19867
 for kind in ('checkpoint','result','raw_job'):
  binding=row[kind];archive=ROOT/binding['archive']
  if archive not in archives:archives[archive]=sha(archive)
  assert archives[archive]==binding['archive_sha256']
  with tarfile.open(archive) as bundle:
   payload=bundle.extractfile(binding['member']).read()
   assert len(payload)==binding['bytes'] and hashlib.sha256(payload).hexdigest()==binding['sha256']
before,after=functions(OLD/'bridge.py'),functions(BASE/'bridge.py')
unchanged=[n for n in before if n not in ('validate_inventory','require_approval')]
assert len(unchanged)==11 and set(before)==set(after) and all(before[n]==after[n] for n in unchanged)
assert after['require_approval']==before['require_approval'].replace('len(ids) == len(set(ids)) == 1','len(ids) == len(set(ids)) == 11')
assert sha(EX/'resource_extra.py')==sha(OLD/'execution_candidate/resource_extra.py')
spec=importlib.util.spec_from_file_location('root_C_after1',BASE/'bridge.py');bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
bridge.validate_inventory(inv,read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json'))
inventory_sha,bridge_sha=sha(BASE/'inventory_actual112_Full100refs.json'),sha(BASE/'bridge.py')
approval=dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY',scope=bridge.SCOPE,inventory_sha256=inventory_sha,bridge_sha256=bridge_sha,
 selected_ids=selected,device='cpu',compute_threads=8,max_processes=1,allowed_cpus=list(range(112,120)),target_split='valid',final_test_dispatch=False,native_tolerance=1e-12)
for identity in selected:bridge.require_approval(approval,inventory_sha,identity,bridge_sha)
refused=[]
for name,key,value in [('zero','selected_ids',[]),('missing','selected_ids',selected[:-1]),('extra','selected_ids',selected+[scope['excluded_prior_ids'][0]]),
 ('duplicate','selected_ids',selected*2),('source','bridge_sha256','0'*64),('inventory','inventory_sha256','0'*64),('test','target_split','test'),
 ('tolerance','native_tolerance',1e-6),('final','final_test_dispatch',True),('threads','compute_threads',9),('gpu','device','cuda')]:
 bad=copy.deepcopy(approval);bad[key]=value
 try:bridge.require_approval(bad,inventory_sha,selected[0],bridge_sha)
 except ValueError:refused.append(name)
 else:raise AssertionError(name)
for name,identity in [('priorC1','minus_C_IID_Benign_seed91001'),('priorU','minus_U_IID_Benign_seed91001'),('Full','Full_IID_Benign_seed91001'),('futureC','minus_C_IID_F Flip_seed91003')]:
 try:bridge.require_approval(approval,inventory_sha,identity,bridge_sha)
 except ValueError:refused.append(name)
 else:raise AssertionError(name)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
SCIENCE=sha(BASE/'FILES_SHA256.json');EXECUTION=sha(EX/'EXECUTION_SOURCE_SHA256.json');PACKAGE=sha(BASE/'PACKAGE_RECEIPT.json')
deploy=(ROOT/'tmp/deploy_mechanism_C1_gate_root_20261009.py').read_text(encoding='utf8')
for a,b in [('celeba_mechanism_valid_C1_gate_20261009',BASE.name),('C1_gate','C_after1'),('C1_GATE','C_AFTER1'),
 ('1fac2f4e1cad1068cbf8421c885c16b4ffe0f52066f1889192c8b7f7b1b82e42',SCIENCE),
 ('7a580b765cbc6fab1aa1ab592d830ee6e063e27fba86fd6449b42a11faf14fa1',EXECUTION),
 (sha(OLD/'PACKAGE_RECEIPT.json'),PACKAGE)]:
 assert a in deploy,a
 deploy=deploy.replace(a,b)
for a,b in [('excluded100_ids','excluded101_ids'),("len(scope['selected_ids']) == 1","len(scope['selected_ids']) == 11"),('exact1','exact11'),
 ("('guardfed_celeba_mechanism_valid_after92','EXITED')","('guardfed_celeba_mechanism_valid_C1_gate','EXITED')")]:
 assert a in deploy,a
 deploy=deploy.replace(a,b)
observer=(ROOT/'tmp/observe_mechanism_C1_gate_root_20261009.py').read_text(encoding='utf8')
for a,b in [('celeba_mechanism_valid_C1_gate_20261009',BASE.name),('C1_gate','C_after1'),('C1_GATE','C_AFTER1'),('exact1','exact11'),
 ("len(scope['selected_ids']) == 1","len(scope['selected_ids']) == 11"),("len(scope['excluded_prior_ids']) == 100","len(scope['excluded_prior_ids']) == 101"),
 ("len(data['completed']) == 1","len(data['completed']) == 11"),('original100_not_rerun','original101_not_rerun')]:
 assert a in observer,a
 observer=observer.replace(a,b)
helper=ROOT/'tmp/deploy_mechanism_C_after1_root_20261009.py';observer_path=ROOT/'tmp/observe_mechanism_C_after1_root_20261009.py'
for target,code in ((helper,deploy),(observer_path,observer)):
 ast.parse(code)
 with target.open('x',encoding='utf8',newline='\n') as f:f.write(code)
review=BASE/'root_independent_review';review.mkdir(exist_ok=False)
proof=dict(status='ROOT_READY_C_AFTER1_SOURCE_REVIEW_PASS_NOT_DISPATCHED',reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 science_seal_sha256=SCIENCE,execution_seal_sha256=EXECUTION,package_receipt_sha256=PACKAGE,scientific_functions_unchanged=True,
 unchanged_bridge_functions=unchanged,positive_valid11_approval_checks=11,invalid_approval_refusals=refused,selected_ids=selected,
 excluded101_ids=scope['excluded_prior_ids'],original101_records_exact=True,Full100_references_exact=True,
 prior_success_root_adoption_sha256=sha(prior),source_archive_members_verified=29,C11_native_archive_33_artifacts_verified=True,
 helper_sha256=sha(helper),observer_sha256=sha(observer_path),short_argv_stdin_from_first_call=True,CNN_executed=False,
 dispatch_performed=False,Linux_resources_measured=False,C1_GATE_already_closed=True)
with (review/'ROOT_READY_REVIEW.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],review_sha256=sha(review/'ROOT_READY_REVIEW.json'),execution=EXECUTION,positive_checks=11,refusals=len(refused))))
