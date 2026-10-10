"""Rebind the existing transport to the sealed actual three-checkpoint package."""
from pathlib import Path
import ast, hashlib, importlib.util, json, re

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'tmp/celeba_mechanism_C_after40_root_operations_20261010'
NEW = ROOT/'tmp/celeba_mechanism_C_after47_root_operations_20261010'
BASE = ROOT/'tmp/celeba_mechanism_valid_C_after47_20261010'
PRIOR = ROOT/'tmp/celeba_mechanism_valid_C_after40_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
expected = [f'minus_C_IID_Sp-DFA_seed{s}' for s in (91008, 91009, 91010)]
assert read(BASE/'SCOPE.json')['selected_ids'] == expected
assert len(read(BASE/'SCOPE.json')['excluded_prior_ids']) == 147
assert sha(BASE/'FILES_SHA256.json') == 'aa5073948bd1c0854b3a4d760ee58b892909f702c31950e5ee80f0cf83b2efc0'
assert sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json') == 'b647c5a6759e709ab4c4fdd9d7ba74361901f98973d67f69176569b1a14c8ef5'
assert sha(BASE/'inventory_actual150_Full100refs.json') == 'fc5f9b31aef29b1f3c43537301c4eacac40cd4d789114a3238f41e5c6d58cab4'
for folder, seal in [(BASE, 'FILES_SHA256.json'), (BASE/'execution_candidate', 'EXECUTION_SOURCE_SHA256.json')]:
    for row in read(folder/seal)['members']:
        assert sha(folder/row['path']) == row['sha256'] and (folder/row['path']).stat().st_size == row['size']
changes = {
    'celeba_mechanism_valid_C_after40_20261010': 'celeba_mechanism_valid_C_after47_20261010',
    'C_AFTER40': 'C_AFTER47', 'C_after40': 'C_after47', 'EXACT7': 'EXACT3', 'exact7': 'exact3',
    sha(PRIOR/'FILES_SHA256.json'): sha(BASE/'FILES_SHA256.json'),
    sha(PRIOR/'execution_candidate/EXECUTION_SOURCE_SHA256.json'): sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),
    sha(PRIOR/'PACKAGE_RECEIPT.json'): sha(BASE/'PACKAGE_RECEIPT.json'),
    sha(PRIOR/'PACKAGE_SHA256.json'): sha(BASE/'PACKAGE_SHA256.json'),
    sha(PRIOR/'inventory_actual147_Full100refs.json'): sha(BASE/'inventory_actual150_Full100refs.json'),
    'inventory_actual147_Full100refs.json': 'inventory_actual150_Full100refs.json',
    "review['native_accepted_snapshot'] == 147": "review['native_accepted_snapshot'] == 150",
    'old140_records_exact': 'old147_records_exact', 'original140': 'original147', 'prior140': 'prior147',
    repr(read(PRIOR/'SCOPE.json')['selected_ids']): repr(expected),
    '== 140': '== 147', '==140': '==147', '== 7': '== 3', '==7': '==3',
    '(63,168,21)': '(27,72,9)',
    'prior_three_view_models=140,accepted_new=7,cumulative_three_view_models=147': 'prior_three_view_models=147,accepted_new=3,cumulative_three_view_models=150',
    'C_after36_20261010/execution_candidate/backups/incremental_20261009T225335Z': 'C_after40_20261010/execution_candidate/backups/incremental_20261009T233028Z',
    'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a': '64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea',
}
old_seal = {row['path']: row['sha256'] for row in read(OLD/'TRANSPORT_REBIND.json')['members']}
pattern = '|'.join(re.escape(k) for k in sorted(changes, key=len, reverse=True))
NEW.mkdir(exist_ok=False)
rows = []
for name in ('deploy.py', 'observe.py', 'backup.py', 'adopt.py'):
    assert sha(OLD/name) == old_seal[name]
    before = (OLD/name).read_text(encoding='utf8')
    after = re.sub(pattern, lambda m: changes[m.group()], before)
    if name == 'adopt.py':
        after = after.replace("len(names)==3*len(expected)", "len(names)==7*len(expected)")
    if name == 'deploy.py':
        after = after.replace("('guardfed_celeba_mechanism_valid_C_after36','EXITED')", "('guardfed_celeba_mechanism_valid_C_after40','EXITED')").replace('PRIOR_C_AFTER36_EXITED', 'PRIOR_C_AFTER40_EXITED')
    ast.parse(after)
    assert after != before and 'utf-7' not in after and 'EXACT7' not in after and 'exact7' not in after
    with (NEW/name).open('x', encoding='utf8', newline='\n') as stream:
        stream.write(after)
    rows.append({'path': name, 'sha256': sha(NEW/name), 'accepted_transport_source_sha256': sha(OLD/name)})
checks = 0
for name in ('backup.py', 'adopt.py'):
    spec = importlib.util.spec_from_file_location('after47_'+name[:-3], NEW/name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = dict(service='guardfed_celeba_mechanism_valid_C_after47 EXITED', processes=[], batch_failure=None, batch_complete={'completed': 3}, completed=[{'id': i} for i in expected])
    mod.check_terminal(good, expected)
    checks += 1
    for field, value in [('service', 'RUNNING'), ('processes', [1]), ('batch_failure', {'failure': True}), ('batch_complete', None), ('completed', [{'id': expected[0]}]*3), ('completed', good['completed'][:2]), ('completed', [{'id': i} for i in expected[:2]+['wrong_id']])]:
        case = dict(good); case[field] = value
        try:
            mod.check_terminal(case, expected)
        except AssertionError:
            checks += 1
        else:
            raise AssertionError('Terminal refusal failed')
assert checks == 16
with (NEW/'TRANSPORT_REBIND.json').open('x', encoding='utf8') as stream:
    json.dump(dict(source_only=True, SSH=False, original_science_changed=False, exact3=True, prior147_not_replayed=True, terminal_positive_and_refusal_checks=checks, members=rows), stream, indent=2)
    stream.write('\n')
print(json.dumps({'transport_files': rows, 'terminal_positive_and_refusal_checks': checks}))
