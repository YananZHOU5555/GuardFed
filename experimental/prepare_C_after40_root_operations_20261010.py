"""Rebind the accepted transport to the sealed seven-checkpoint package; no SSH here."""
from pathlib import Path
import ast, hashlib, importlib.util, json, re

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'tmp/celeba_mechanism_C_after36_root_operations_20261010'
NEW = ROOT/'tmp/celeba_mechanism_C_after40_root_operations_20261010'
BASE = ROOT/'tmp/celeba_mechanism_valid_C_after40_20261010'
PRIOR = ROOT/'tmp/celeba_mechanism_valid_C_after36_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
expected = [f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001, 91008)]
assert read(BASE/'SCOPE.json')['selected_ids'] == expected
assert len(read(BASE/'SCOPE.json')['excluded_prior_ids']) == 140
assert sha(BASE/'FILES_SHA256.json') == '22da16b734f2c6041d494e30c200d586ff79626ac0a5ec6b438bea405c7dc48b'
assert sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json') == '1ac84b16f00c30ed3effd230f539824bb426540e66ec4981a53fdee676c50ec8'
assert sha(BASE/'inventory_actual147_Full100refs.json') == 'c65a6e8a88ef1e97a687bbf259840582c3d1f4c26e8e09deb604e94a4cc5b10f'
for folder, seal in [(BASE, 'FILES_SHA256.json'), (BASE/'execution_candidate', 'EXECUTION_SOURCE_SHA256.json')]:
    for row in read(folder/seal)['members']:
        assert sha(folder/row['path']) == row['sha256'] and (folder/row['path']).stat().st_size == row['size']
changes = {
    'celeba_mechanism_valid_C_after36_20261010': 'celeba_mechanism_valid_C_after40_20261010',
    'C_AFTER36': 'C_AFTER40', 'C_after36': 'C_after40', 'EXACT4': 'EXACT7', 'exact4': 'exact7',
    sha(PRIOR/'FILES_SHA256.json'): sha(BASE/'FILES_SHA256.json'),
    sha(PRIOR/'execution_candidate/EXECUTION_SOURCE_SHA256.json'): sha(BASE/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),
    sha(PRIOR/'PACKAGE_RECEIPT.json'): sha(BASE/'PACKAGE_RECEIPT.json'),
    sha(PRIOR/'PACKAGE_SHA256.json'): sha(BASE/'PACKAGE_SHA256.json'),
    sha(PRIOR/'inventory_actual140_Full100refs.json'): sha(BASE/'inventory_actual147_Full100refs.json'),
    'inventory_actual140_Full100refs.json': 'inventory_actual147_Full100refs.json',
    "review['native_accepted_snapshot'] == 140": "review['native_accepted_snapshot'] == 147",
    'old136_records_exact': 'old140_records_exact', 'original136': 'original140', 'prior136': 'prior140',
    repr(read(PRIOR/'SCOPE.json')['selected_ids']): repr(expected),
    '== 136': '== 140', '==136': '==140', '== 4': '== 7', '==4': '==7',
    '(36,96,12)': '(63,168,21)',
    'prior_three_view_models=136,accepted_new=4,cumulative_three_view_models=140': 'prior_three_view_models=140,accepted_new=7,cumulative_three_view_models=147',
    'C_after28_20261009/execution_candidate/backups/incremental_20261009T221150Z': 'C_after36_20261010/execution_candidate/backups/incremental_20261009T225335Z',
    'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0': 'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a',
}
old_seal = {row['path']: row['sha256'] for row in read(OLD/'TRANSPORT_REBIND.json')['members']}
pattern = '|'.join(re.escape(k) for k in sorted(changes, key=len, reverse=True))
NEW.mkdir(exist_ok=False)
rows = []
for name in ('deploy.py', 'observe.py', 'backup.py', 'adopt.py'):
    assert sha(OLD/name) == old_seal[name]
    before = (OLD/name).read_text(encoding='utf8')
    after = re.sub(pattern, lambda m: changes[m.group()], before)
    if name == 'deploy.py':
        after = after.replace("('guardfed_celeba_mechanism_valid_C_after28','EXITED')", "('guardfed_celeba_mechanism_valid_C_after36','EXITED')").replace('PRIOR_C_AFTER28_EXITED', 'PRIOR_C_AFTER36_EXITED')
    ast.parse(after)
    assert after != before and 'utf-7' not in after and 'EXACT4' not in after and 'exact4' not in after
    with (NEW/name).open('x', encoding='utf8', newline='\n') as stream:
        stream.write(after)
    rows.append({'path': name, 'sha256': sha(NEW/name), 'accepted_transport_source_sha256': sha(OLD/name)})
checks = 0
for name in ('backup.py', 'adopt.py'):
    spec = importlib.util.spec_from_file_location('after40_'+name[:-3], NEW/name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = dict(service='guardfed_celeba_mechanism_valid_C_after40 EXITED', processes=[], batch_failure=None, batch_complete={'completed': 7}, completed=[{'id': i} for i in expected])
    mod.check_terminal(good, expected)
    checks += 1
    for field, value in [('service', 'RUNNING'), ('processes', [1]), ('batch_failure', {'failure': True}), ('batch_complete', None), ('completed', [{'id': expected[0]}]*7), ('completed', good['completed'][:6]), ('completed', [{'id': i} for i in expected[:6]+['wrong_id']])]:
        case = dict(good)
        case[field] = value
        try:
            mod.check_terminal(case, expected)
        except AssertionError:
            checks += 1
        else:
            raise AssertionError('Terminal refusal failed')
assert checks == 16
with (NEW/'TRANSPORT_REBIND.json').open('x', encoding='utf8') as stream:
    json.dump(dict(source_only=True, SSH=False, original_science_changed=False, exact7=True, prior140_not_replayed=True, terminal_positive_and_refusal_checks=checks, members=rows), stream, indent=2)
    stream.write('\n')
print(json.dumps({'transport_files': rows, 'terminal_positive_and_refusal_checks': checks}))
