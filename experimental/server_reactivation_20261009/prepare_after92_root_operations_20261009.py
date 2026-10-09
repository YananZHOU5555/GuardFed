"""Review exact eight accepted models; derive one-shot ROOT operations only."""
from pathlib import Path, PurePosixPath
import ast, copy, datetime, hashlib, importlib.util, json, sys, tarfile
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009'
BASE = ROOT/'tmp/celeba_mechanism_valid_incremental_after92_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

def functions(path):
    text = path.read_text(encoding='utf8')
    return {n.name: ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}

def write_new(path,text):
    ast.parse(text)
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(text)

assert sha(BASE/'PACKAGE_RECEIPT.json') == 'a3983358664a0556677e920fcea9e2b8c3556de67f11d1d304f67559255e6a94'
package = read(BASE/'PACKAGE_RECEIPT.json')
assert not package['actual_remote_execution']
for name,pin in package['members'].items():
    assert sha(BASE/name) == pin['sha256'] and (BASE/name).stat().st_size == pin['bytes']
assert sha(BASE/'prepared_source.tar.gz') == package['source_archive_sha256']
with tarfile.open(BASE/'prepared_source.tar.gz') as tar:
    assert len(tar.getnames()) == len(set(tar.getnames())) == package['archive_members'] == 30
    assert set(tar.getnames()) == {BASE.name+'/'+name for name in package['members']}
    for item in tar.getmembers():
        rel = PurePosixPath(item.name)
        assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
    for name,pin in package['members'].items():
        raw = tar.extractfile(BASE.name+'/'+name).read()
        assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256']
for name,pin in read(BASE/'INPUT_PINS.json').items():
    assert sha(ROOT/name) == pin['sha256'] and (ROOT/name).stat().st_size == pin['bytes']
prior = OLD/'execution_candidate/backups/incremental_20261009T174922Z/ROOT_ADOPTION_REVIEW.json'
assert sha(prior) == 'b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683'
assert read(prior)['cumulative_three_view_models'] == 92
scope = read(BASE/'SCOPE.json'); old_inv = read(OLD/'inventory_actual92_Full100refs.json')
inv = read(BASE/'inventory_actual100_Full100refs.json')
assert scope['selected_ids'] == [f'minus_U_non-IID_Sp-DFA_seed{s}' for s in range(91003,91011)]
assert set(scope['excluded_prior_ids']) == {r['id'] for r in old_inv['records']} and len(scope['excluded_prior_ids']) == 92
indexed = {r['id']:r for r in inv['records']}
assert len(indexed) == 100 and all(indexed[r['id']] == r for r in old_inv['records'])
assert inv['full_references'] == old_inv['full_references']
assert set(indexed)-set(scope['excluded_prior_ids']) == set(scope['selected_ids'])
before,after = functions(OLD/'bridge.py'), functions(BASE/'bridge.py')
unchanged = [n for n in before if n not in ('validate_inventory','require_approval')]
assert len(unchanged) == 10 and all(before[n] == after[n] for n in unchanged)
expected = before['validate_inventory']
for a,b in [('== 92','== 100'),('exactly92','exactly100'),('accepted92','accepted100'),('prior82/selected10','prior92/selected8'),('== 708','== 700')]:
    expected = expected.replace(a,b)
assert after['validate_inventory'] == expected
assert after['require_approval'] == before['require_approval'].replace('== 10','== 8')
assert sha(BASE/'execution_candidate/resource_extra.py') == sha(OLD/'execution_candidate/resource_extra.py')
spec = importlib.util.spec_from_file_location('root_after92_gate',BASE/'bridge.py')
bridge = importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
inventory_sha,bridge_sha = sha(BASE/'inventory_actual100_Full100refs.json'),sha(BASE/'bridge.py')
approval = dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY',scope=bridge.SCOPE,
    inventory_sha256=inventory_sha,bridge_sha256=bridge_sha,selected_ids=scope['selected_ids'],device='cpu',
    compute_threads=8,max_processes=1,allowed_cpus=list(range(112,120)),target_split='valid',
    final_test_dispatch=False,native_tolerance=1e-12)
for rid in scope['selected_ids']:bridge.require_approval(approval,inventory_sha,rid,bridge_sha)
refused=[]
for name,key,value in [('seven','selected_ids',scope['selected_ids'][:-1]),
    ('nine','selected_ids',scope['selected_ids']+[scope['excluded_prior_ids'][0]]),
    ('duplicate','selected_ids',scope['selected_ids'][:-1]+[scope['selected_ids'][0]]),
    ('source','bridge_sha256','0'*64),('inventory','inventory_sha256','0'*64),('test','target_split','test'),
    ('tolerance','native_tolerance',1e-6),('final','final_test_dispatch',True),('threads','compute_threads',9),('gpu','device','cuda')]:
    bad=copy.deepcopy(approval);bad[key]=value
    try:bridge.require_approval(bad,inventory_sha,scope['selected_ids'][0],bridge_sha)
    except ValueError:refused.append(name)
    else:raise AssertionError('Accepted invalid approval: '+name)
for name,rid in [('prior92',scope['excluded_prior_ids'][0]),('Full','Full_IID_Benign_seed91001'),('foreign','foreign')]:
    try:bridge.require_approval(approval,inventory_sha,rid,bridge_sha)
    except ValueError:refused.append(name)
    else:raise AssertionError('Accepted invalid ID: '+name)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
SCIENCE=sha(BASE/'FILES_SHA256.json');EXECUTION=sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json')
PACKAGE=sha(BASE/'PACKAGE_RECEIPT.json')
text=(ROOT/'tmp/deploy_mechanism_after82_v2_root_20261009.py').read_text(encoding='utf8')
text=text.replace('after82_v2','after92').replace('AFTER82_V2','AFTER92')
for a,b in [('b95801ac039bb39276e79012393792adfde290b92091ff45c0a4323bd4fdd8f0',SCIENCE),
    ('94d99842b334346ae8a6715b84f7fddea35574a77fbf51f9460b25301c6ccae6',EXECUTION),
    ('4f2232933fcf474548dba28724afb8dbb8b39c12ae42169c91e1440f51934d28',PACKAGE),
    ('PACKAGE_SHA256.json','PACKAGE_RECEIPT.json'),('package_seal_sha256','package_receipt_sha256'),
    ('excluded82_ids','excluded92_ids'),('== 10','== 8'),('exact10','exact8'),
    ("('guardfed_celeba_mechanism_valid_after82','EXITED')","('guardfed_celeba_mechanism_valid_after82_v2','EXITED')")]:
    assert a in text;text=text.replace(a,b)
helper=ROOT/'tmp/deploy_mechanism_after92_root_20261009.py';write_new(helper,text)
observer=(ROOT/'tmp/observe_mechanism_after82_v2_root_20261009.py').read_text(encoding='utf8')
observer=observer.replace('after82_v2','after92').replace('AFTER82_V2','AFTER92').replace('exact10','exact8')
observer=observer.replace('== 10','== 8').replace('== 82','== 92').replace('original82_not_rerun','original92_not_rerun')
observer_path=ROOT/'tmp/observe_mechanism_after92_root_20261009.py';write_new(observer_path,observer)
review=BASE/'root_independent_review';review.mkdir(exist_ok=False)
proof=dict(status='ROOT_READY_AFTER92_SOURCE_REVIEW_PASS_NOT_DISPATCHED',reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    science_seal_sha256=SCIENCE,execution_seal_sha256=EXECUTION,package_receipt_sha256=PACKAGE,
    scientific_functions_unchanged=True,unchanged_bridge_functions=unchanged,positive_valid8_approval_checks=8,
    invalid_approval_refusals=refused,selected_ids=scope['selected_ids'],excluded92_ids=scope['excluded_prior_ids'],
    original92_records_exact=True,Full100_references_exact=True,prior_success_root_adoption_sha256=sha(prior),
    source_archive_members_verified=30,helper_sha256=sha(helper),observer_sha256=sha(observer_path),
    short_argv_stdin_from_first_call=True,CNN_executed=False,dispatch_performed=False,Linux_resources_measured=False)
with (review/'ROOT_READY_REVIEW.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],science=SCIENCE,execution=EXECUTION,package=PACKAGE,
    review_sha256=sha(review/'ROOT_READY_REVIEW.json'),positive_checks=8,refusals=len(refused))))
