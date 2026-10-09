"""Independent positive approval gate and minimal derivative ROOT operations."""
from pathlib import Path
import ast, copy, datetime, hashlib, importlib.util, json, sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'tmp/celeba_mechanism_valid_incremental_after82_20261009'
BASE = ROOT / 'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

def functions(path):
    text = path.read_text(encoding='utf8')
    return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}

def write_new(path, text):
    ast.parse(text)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)

assert sha(OLD/'execution_candidate/ROOT_FAILURE_REVIEW.json') == '947e809b1db1585ea41622532730c243e86aae4b5e29212d5d6984a5ba2af169'
assert read(OLD/'execution_candidate/ROOT_FAILURE_REVIEW.json')['accepted_new'] == 0
assert sha(BASE/'PACKAGE_SHA256.json') == '4f2232933fcf474548dba28724afb8dbb8b39c12ae42169c91e1440f51934d28'
package = read(BASE/'PACKAGE_SHA256.json')
for name, pin in package['artifacts'].items():
    path = BASE/name
    assert sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes']
SCIENCE = sha(BASE/'FILES_SHA256.json')
EXECUTION = sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json')
PACKAGE = sha(BASE/'PACKAGE_SHA256.json')
old_scope, scope = read(OLD/'SCOPE.json'), read(BASE/'SCOPE.json')
assert scope['selected_ids'] == old_scope['selected_ids'] and len(scope['selected_ids']) == 10
assert scope['excluded_prior_ids'] == old_scope['excluded_prior_ids'] and len(scope['excluded_prior_ids']) == 82
old_inv = read(OLD/'inventory_actual92_Full100refs.json')
inv = read(BASE/'inventory_actual92_Full100refs.json')
assert old_inv['records'] == inv['records'] and old_inv['full_references'] == inv['full_references']
old_functions, new_functions = functions(OLD/'bridge.py'), functions(BASE/'bridge.py')
unchanged = [n for n in old_functions if n not in ('require_approval','validate_inventory')]
assert len(unchanged) == 10 and all(old_functions[n] == new_functions[n] for n in unchanged)
assert old_functions['validate_inventory'].replace('selected11 boundary changed','selected10 boundary changed') == new_functions['validate_inventory']
assert old_functions['require_approval'].replace('== 11', '== 10') == new_functions['require_approval']
assert sha(OLD/'execution_candidate/resource_extra.py') == sha(BASE/'execution_candidate/resource_extra.py')
spec = importlib.util.spec_from_file_location('root_after82_v2_bridge_gate', BASE/'bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
inventory_sha, bridge_sha = sha(BASE/'inventory_actual92_Full100refs.json'), sha(BASE/'bridge.py')
approval = dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY', scope=bridge.SCOPE,
    inventory_sha256=inventory_sha, bridge_sha256=bridge_sha, selected_ids=scope['selected_ids'],
    device='cpu', compute_threads=8, max_processes=1, allowed_cpus=list(range(112,120)),
    target_split='valid', final_test_dispatch=False, native_tolerance=1e-12)
for identity in scope['selected_ids']:
    bridge.require_approval(approval, inventory_sha, identity, bridge_sha)
refusals = []
mutations = [('nine', 'selected_ids', scope['selected_ids'][:-1]),
    ('eleven', 'selected_ids', scope['selected_ids']+[scope['excluded_prior_ids'][0]]),
    ('duplicate', 'selected_ids', scope['selected_ids'][:-1]+[scope['selected_ids'][0]]),
    ('source', 'bridge_sha256', '0'*64), ('inventory', 'inventory_sha256', '0'*64),
    ('test', 'target_split', 'test'), ('tolerance', 'native_tolerance', 1e-6),
    ('final', 'final_test_dispatch', True), ('threads', 'compute_threads', 9), ('gpu', 'device', 'cuda')]
for name, key, value in mutations:
    changed = copy.deepcopy(approval); changed[key] = value
    try:
        bridge.require_approval(changed, inventory_sha, scope['selected_ids'][0], bridge_sha)
    except ValueError:
        refusals.append(name)
    else:
        raise AssertionError('Accepted invalid approval: '+name)
for name, identity in [('prior82',scope['excluded_prior_ids'][0]),('Full','Full_IID_Benign_seed91001'),('foreign','foreign')]:
    try:
        bridge.require_approval(approval, inventory_sha, identity, bridge_sha)
    except ValueError:
        refusals.append(name)
    else:
        raise AssertionError('Accepted invalid identity: '+name)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
old_helper = ROOT/'tmp/deploy_mechanism_after82_root_20261009.py'
assert sha(old_helper) == '8ed8fa91d946a5afbc33125cb83f3bcedba7fa614cf362ce9d187fb44967864f'
text = old_helper.read_text(encoding='utf8').replace('after82','after82_v2').replace('AFTER82','AFTER82_V2')
for before, after in [
    ('b95e4377343794137914a8f9836bbf4f29a3bb784461c00a9d4a08c7eaed1ea4', SCIENCE),
    ('cc0215d8c716be8adf0bb5888166bace31b8b176e7e0b19b3fcf09a0d972a539', EXECUTION),
    ('5dc4cb3f5c6d824e886b7f570b0b8c948407a6fbfddbca650a4530de66d636a7', PACKAGE),
    ("('guardfed_celeba_mechanism_valid_after71','EXITED')", "('guardfed_celeba_mechanism_valid_after82','EXITED')")]:
    assert before in text
    text = text.replace(before,after)
helper = ROOT/'tmp/deploy_mechanism_after82_v2_root_20261009.py'
write_new(helper,text)
observer = (ROOT/'tmp/observe_mechanism_after82_root_20261009.py').read_text(encoding='utf8')
write_new(ROOT/'tmp/observe_mechanism_after82_v2_root_20261009.py',observer.replace('after82','after82_v2').replace('AFTER82','AFTER82_V2'))
review_dir = BASE/'root_independent_review';review_dir.mkdir(exist_ok=False)
proof = dict(status='ROOT_READY_AFTER82_V2_SOURCE_REVIEW_PASS_NOT_DISPATCHED',
    reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    science_seal_sha256=SCIENCE, execution_seal_sha256=EXECUTION, package_seal_sha256=PACKAGE,
    scientific_functions_unchanged=True, unchanged_bridge_functions=unchanged,
    approval_guard_only_cardinality_11_to_10=True, inventory_diagnostic_label_only_change=True,
    positive_valid10_approval_checks=10, invalid_approval_refusals=refusals,
    selected_ids=scope['selected_ids'], excluded82_ids=scope['excluded_prior_ids'],
    original92_records_exact=True, Full100_references_exact=True,
    failed_attempt_review_sha256=sha(OLD/'execution_candidate/ROOT_FAILURE_REVIEW.json'),
    original_helper_sha256=sha(old_helper), helper_sha256=sha(helper),
    helper_path=helper.relative_to(ROOT).as_posix(), observer_sha256=sha(ROOT/'tmp/observe_mechanism_after82_v2_root_20261009.py'),
    short_argv_stdin_from_first_call=True, CNN_executed=False, dispatch_performed=False, Linux_resources_measured=False)
with (review_dir/'ROOT_READY_REVIEW.json').open('x',encoding='utf8',newline='\n') as stream:
    json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],science=SCIENCE,execution=EXECUTION,package=PACKAGE,
    review_sha256=sha(review_dir/'ROOT_READY_REVIEW.json'),positive_checks=10,refusals=len(refusals))))
