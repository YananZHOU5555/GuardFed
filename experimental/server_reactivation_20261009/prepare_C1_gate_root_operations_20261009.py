"""Independently review the actual C1 gate and derive one-shot ROOT operations."""
from pathlib import Path, PurePosixPath
import ast, copy, datetime, hashlib, importlib.util, json, sys, tarfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'tmp/celeba_mechanism_valid_incremental_after92_20261009'
BASE=ROOT/'tmp/celeba_mechanism_valid_C1_gate_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
functions=lambda p:{n.name:ast.get_source_segment(p.read_text(encoding='utf8'),n) for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)}

assert sha(BASE/'PACKAGE_RECEIPT.json')=='17d2eb8f11e28df3bb1a8287d1d0227103f12adf37d3b84c3f9fe3bfb5d86c30'
package=read(BASE/'PACKAGE_RECEIPT.json');assert not package['actual_remote_execution']
assert sha(BASE/'prepared_source.tar.gz')==package['source_archive_sha256']=='dcaf415c838303baff116c8aa8f5fb3e51d68ca39164758e21bce76308099df9'
with tarfile.open(BASE/'prepared_source.tar.gz') as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==package['archive_members']==31
    assert set(bundle.getnames())=={BASE.name+'/'+name for name in package['members']}
    for name,pin in package['members'].items():
        item=bundle.getmember(BASE.name+'/'+name);rel=PurePosixPath(item.name)
        assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        payload=bundle.extractfile(item).read()
        assert len(payload)==pin['bytes'] and hashlib.sha256(payload).hexdigest()==pin['sha256']==sha(BASE/name)
for name,pin in read(BASE/'INPUT_PINS.json').items():
    assert sha(ROOT/name)==pin['sha256'] and (ROOT/name).stat().st_size==pin['bytes']
prior=OLD/'execution_candidate/backups/incremental_20261009T183102Z/ROOT_ADOPTION_REVIEW.json'
assert sha(prior)=='9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee'
assert read(prior)['cumulative_three_view_models']==100
scope=read(BASE/'SCOPE.json');old_inv=read(OLD/'inventory_actual100_Full100refs.json');inv=read(BASE/'inventory_actual101_Full100refs.json')
assert scope['selected_ids']==['minus_C_IID_Benign_seed91001']
assert len(scope['excluded_prior_ids'])==100 and set(scope['excluded_prior_ids'])=={r['id'] for r in old_inv['records']}
indexed={r['id']:r for r in inv['records']}
assert len(indexed)==101 and all(indexed[r['id']]==r for r in old_inv['records']) and inv['full_references']==old_inv['full_references']
assert set(indexed)-set(scope['excluded_prior_ids'])==set(scope['selected_ids'])
c1=indexed[scope['selected_ids'][0]]
assert c1['variant']=='minus_C' and c1['config']['ablation_component']=='C' and c1['terminal_round']==70
assert c1['original_split']=='valid' and c1['original_n_eval']==19867
for kind in ('checkpoint','result','raw_job'):
    binding=c1[kind];archive=ROOT/binding['archive'];assert sha(archive)==binding['archive_sha256']
    with tarfile.open(archive) as bundle:
        payload=bundle.extractfile(binding['member']).read()
        assert len(payload)==binding['bytes'] and hashlib.sha256(payload).hexdigest()==binding['sha256']
before,after=functions(OLD/'bridge.py'),functions(BASE/'bridge.py')
unchanged=[n for n in before if n not in ('validate_inventory','require_approval')]
assert len(unchanged)==10 and all(before[n]==after[n] for n in unchanged)
assert set(after)-set(before)=={'validate_variant_metadata'}
planned=functions(ROOT/'tmp/celeba_mechanism_remaining_variants_source_plan_20261009/variant_metadata_draft.py')
assert after['validate_variant_metadata']==planned['validate_variant_metadata']
assert after['require_approval']==before['require_approval'].replace('len(ids) == len(set(ids)) == 8','len(ids) == len(set(ids)) == 1')
assert sha(BASE/'execution_candidate/resource_extra.py')==sha(OLD/'execution_candidate/resource_extra.py')
spec=importlib.util.spec_from_file_location('root_C1_gate',BASE/'bridge.py');bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
bridge.validate_inventory(inv,read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json'))
inventory_sha,bridge_sha=sha(BASE/'inventory_actual101_Full100refs.json'),sha(BASE/'bridge.py')
approval=dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY',scope=bridge.SCOPE,inventory_sha256=inventory_sha,
    bridge_sha256=bridge_sha,selected_ids=scope['selected_ids'],device='cpu',compute_threads=8,max_processes=1,
    allowed_cpus=list(range(112,120)),target_split='valid',final_test_dispatch=False,native_tolerance=1e-12)
bridge.require_approval(approval,inventory_sha,scope['selected_ids'][0],bridge_sha)
refused=[]
for name,key,value in [('zero','selected_ids',[]),('two','selected_ids',scope['selected_ids']+[scope['excluded_prior_ids'][0]]),
        ('duplicate','selected_ids',scope['selected_ids']*2),('source','bridge_sha256','0'*64),('inventory','inventory_sha256','0'*64),
        ('test','target_split','test'),('tolerance','native_tolerance',1e-6),('final','final_test_dispatch',True),
        ('threads','compute_threads',9),('gpu','device','cuda')]:
    bad=copy.deepcopy(approval);bad[key]=value
    try:bridge.require_approval(bad,inventory_sha,scope['selected_ids'][0],bridge_sha)
    except ValueError:refused.append(name)
    else:raise AssertionError('Invalid approval accepted: '+name)
for name,rid in [('priorU100',scope['excluded_prior_ids'][0]),('Full','Full_IID_Benign_seed91001'),('otherC','minus_C_IID_Benign_seed91002')]:
    try:bridge.require_approval(approval,inventory_sha,rid,bridge_sha)
    except ValueError:refused.append(name)
    else:raise AssertionError('Invalid ID accepted: '+name)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
SCIENCE=sha(BASE/'FILES_SHA256.json');EXECUTION=sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json');PACKAGE=sha(BASE/'PACKAGE_RECEIPT.json')
deploy=(ROOT/'tmp/deploy_mechanism_after92_root_20261009.py').read_text(encoding='utf8')
for a,b in [('celeba_mechanism_valid_incremental_after92_20261009',BASE.name),('after92','C1_gate'),('AFTER92','C1_GATE'),
        ('832e02a7ab0bc58dd22373c1793c39fc7926a4e961d4fba0c750964eb7b7a94a',SCIENCE),
        ('68b11d80698d1073250736ad73695a79b29ed26d0bbc133441876a56f6f5c5bf',EXECUTION),
        ('a3983358664a0556677e920fcea9e2b8c3556de67f11d1d304f67559255e6a94',PACKAGE),
        ('excluded92_ids','excluded100_ids'),('== 8','== 1'),('exact8','exact1'),
        ("('guardfed_celeba_mechanism_valid_after82_v2','EXITED')","('guardfed_celeba_mechanism_valid_after92','EXITED')")]:
    assert a in deploy,a
    deploy=deploy.replace(a,b)
observer=(ROOT/'tmp/observe_mechanism_after92_root_20261009.py').read_text(encoding='utf8')
for a,b in [('celeba_mechanism_valid_incremental_after92_20261009',BASE.name),('after92','C1_gate'),('AFTER92','C1_GATE'),
        ('exact8','exact1'),('== 8','== 1'),('== 92','== 100'),('original92_not_rerun','original100_not_rerun')]:
    assert a in observer,a
    observer=observer.replace(a,b)
helper=ROOT/'tmp/deploy_mechanism_C1_gate_root_20261009.py';observer_path=ROOT/'tmp/observe_mechanism_C1_gate_root_20261009.py'
for target,code in [(helper,deploy),(observer_path,observer)]:
    ast.parse(code)
    with target.open('x',encoding='utf8',newline='\n') as stream:stream.write(code)
review=BASE/'root_independent_review';review.mkdir(exist_ok=False)
proof=dict(status='ROOT_READY_C1_GATE_SOURCE_REVIEW_PASS_NOT_DISPATCHED',reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    science_seal_sha256=SCIENCE,execution_seal_sha256=EXECUTION,package_receipt_sha256=PACKAGE,
    scientific_functions_unchanged=True,unchanged_bridge_functions=unchanged,positive_valid1_approval_checks=1,invalid_approval_refusals=refused,
    selected_ids=scope['selected_ids'],excluded100_ids=scope['excluded_prior_ids'],original100_records_exact=True,Full100_references_exact=True,
    prior_success_root_adoption_sha256=sha(prior),source_archive_members_verified=31,C1_native_archive_three_artifacts_verified=True,
    helper_sha256=sha(helper),observer_sha256=sha(observer_path),short_argv_stdin_from_first_call=True,CNN_executed=False,
    dispatch_performed=False,Linux_resources_measured=False,new_variant_gate_only=True,other_C3_excluded=True)
with (review/'ROOT_READY_REVIEW.json').open('x',encoding='utf8',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],science=SCIENCE,execution=EXECUTION,package=PACKAGE,
    review_sha256=sha(review/'ROOT_READY_REVIEW.json'),positive_checks=1,refusals=len(refused))))
