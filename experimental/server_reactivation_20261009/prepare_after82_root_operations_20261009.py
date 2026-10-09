"""Review fixed source/data identities and prepare the existing one-shot ROOT flow."""
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_mechanism_valid_incremental_after82_20261009'
OLD = ROOT / 'tmp/celeba_mechanism_valid_incremental_after71_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
SCIENCE = 'b95e4377343794137914a8f9836bbf4f29a3bb784461c00a9d4a08c7eaed1ea4'
EXECUTION = 'cc0215d8c716be8adf0bb5888166bace31b8b176e7e0b19b3fcf09a0d972a539'
PACKAGE = '5dc4cb3f5c6d824e886b7f570b0b8c948407a6fbfddbca650a4530de66d636a7'
for p, digest in [(BASE/'FILES_SHA256.json', SCIENCE),
        (BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json', EXECUTION),
        (BASE/'PACKAGE_SHA256.json', PACKAGE),
        (BASE/'HANDOFF.json', '775eb5f0c2bad953874ab448e4fc0d79b6684a269d0eb0a245c60b267865aad1')]:
    assert sha(p) == digest
for name, pin in read(BASE/'PACKAGE_SHA256.json')['artifacts'].items():
    assert sha(BASE/name) == pin['sha256'] and (BASE/name).stat().st_size == pin['bytes']
for name, pin in read(BASE/'INPUT_PINS.json').items():
    assert sha(ROOT/name) == pin['sha256'] and (ROOT/name).stat().st_size == pin['bytes']
scope = read(BASE/'SCOPE.json')
inspection = read(ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/mechanism_inspection_v4_root_delta_20261009T171247Z/inspection.json')
prior = read(OLD/'inventory_actual82_Full100refs.json')
expected = sorted(set(inspection['accepted_new_ids']) - {r['id'] for r in prior['records']})
assert len(expected) == 10 and expected == sorted(scope['selected_ids'])
assert set(scope['excluded_prior_ids']) == {r['id'] for r in prior['records']} and len(scope['excluded_prior_ids']) == 82

def functions(path):
    text = path.read_text(encoding='utf8')
    return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}

old_functions, new_functions = functions(OLD/'bridge.py'), functions(BASE/'bridge.py')
scientific = [name for name in old_functions if name != 'validate_inventory']
assert len(scientific) == 11 and all(old_functions[name] == new_functions[name] for name in scientific)
changed = old_functions['validate_inventory']
for before, after in [('== 82','== 92'),('exactly82','exactly92'),('accepted82','accepted92'),('prior71','prior82'),('== 718','== 708')]:
    changed = changed.replace(before, after)
assert changed == new_functions['validate_inventory']
assert sha(OLD/'execution_candidate/resource_extra.py') == sha(BASE/'execution_candidate/resource_extra.py')
spec = importlib.util.spec_from_file_location('root_after82_bridge_review', BASE/'bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inventory = read(BASE/'inventory_actual92_Full100refs.json')
baseline = read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
assert len(module.validate_inventory(inventory, baseline)) == 92
assert {r['id']:r for r in prior['records']} == {r['id']:r for r in inventory['records'] if r['id'] in scope['excluded_prior_ids']}
assert prior['full_references'] == inventory['full_references']
assert 'torch' not in sys.modules and 'numpy' not in sys.modules

old_helper = ROOT/'tmp/deploy_mechanism_after71_root_20261009.py'
assert sha(old_helper) == 'bff690f818c5a0a58e9410930b3c4a1e118ab13ac51b5f8b54a56bd95ac911bd'
text = old_helper.read_text(encoding='utf8')
replacements = [
    ('after71','after82'),('AFTER71','AFTER82'),('excluded71_ids','excluded82_ids'),
    ('exact11','exact10'),('== 11','== 10'),
    ('d05a0b81620d1791858a9f71405252443e593600f390359175ff961dde0e2bec',SCIENCE),
    ('0fe232aab75a870b7840fd6ef0bba85b3d12c541ca49a0c0976f3f8a7095352f',EXECUTION),
    ('64224d899d591b07de068b48c59c477032e0f898c869abb2a34070b89d2e5b6b',PACKAGE),
    ("('guardfed_celeba_mechanism_valid_next11','EXITED')","('guardfed_celeba_mechanism_valid_after71','EXITED')"),
    ("ssh + ['python -B -c ' + shlex.quote(preflight)], capture_output=True", "ssh + ['python -B -'], input=preflight.encode(), capture_output=True"),
    ("ssh + ['python -B -c ' + shlex.quote(code)], capture_output=True", "ssh + ['python -B -'], input=code.encode(), capture_output=True")]
for before, after in replacements:
    assert before in text, before
    text = text.replace(before, after)
assert 'shlex.quote(' not in text and "input=preflight.encode()" in text and "input=code.encode()" in text
helper = ROOT/'tmp/deploy_mechanism_after82_root_20261009.py'
assert not helper.exists()
ast.parse(text)
helper.write_text(text, encoding='utf8', newline='\n')
local = read(BASE/'LOCAL_REBIND_CHECK.json')
assert local['refusal_count'] == 10 and local['selected_count'] == 10 and local['excluded_count'] == 82
assert local['bridge_scientific_functions_exact'] == 11 and local['resource_extra_byte_exact']
assert sum(row['functions'] for row in local['execution_functions'].values()) == 26
review_dir = BASE/'root_independent_review'
assert not review_dir.exists()
review_dir.mkdir()
proof = dict(status='ROOT_READY_AFTER82_SOURCE_REVIEW_PASS_NOT_DISPATCHED',
    reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    science_seal_sha256=SCIENCE, execution_seal_sha256=EXECUTION, package_seal_sha256=PACKAGE,
    scientific_functions_unchanged=True, bridge_pure_source_functions_exact=11,
    validate_inventory_only_snapshot_rebind=True, selected_ids=scope['selected_ids'],
    excluded82_ids=scope['excluded_prior_ids'], original82_records_exact=True, Full100_references_exact=True,
    prepared_handoff_sha256=sha(BASE/'HANDOFF.json'), local_refusal_review_sha256=sha(BASE/'LOCAL_REBIND_CHECK.json'),
    original_helper_sha256=sha(old_helper), helper_sha256=sha(helper),
    helper_path=helper.relative_to(ROOT).as_posix(), short_argv_stdin_from_first_call=True,
    CNN_executed=False, dispatch_performed=False, Linux_resources_measured=False)
with (review_dir/'ROOT_READY_REVIEW.json').open('x', encoding='utf8', newline='\n') as stream:
    json.dump(proof, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status=proof['status'], review_sha256=sha(review_dir/'ROOT_READY_REVIEW.json'),
    helper_sha256=sha(helper), selected=10, new_inference=0)))
