"""Local source-only rebind of the preserved after82 failed preparation. No CNN/SSH."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.with_name('celeba_mechanism_valid_incremental_after82_20261009')
OLD_NAME = OLD.name
NEW_NAME = HERE.name
OLD_SCOPE = 'MECHANISM_TERMINAL_VALID_REPLAY_INCREMENTAL_AFTER82'
NEW_SCOPE = OLD_SCOPE + '_V2'
OLD_SERVICE = 'guardfed_celeba_mechanism_valid_after82'
NEW_SERVICE = OLD_SERVICE + '_v2'
PARENT_PACKAGE = '5dc4cb3f5c6d824e886b7f570b0b8c948407a6fbfddbca650a4530de66d636a7'
FAILURE_SHA = 'b63558fb33e5dd7c576ff3a45f76e334e4d40ab5ca3e884618aec6a862852415'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(rel, value):
    path = HERE / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False) + '\n')


def rebind(text):
    return text.replace(OLD_NAME, NEW_NAME).replace(OLD_SCOPE, NEW_SCOPE).replace(OLD_SERVICE, NEW_SERVICE).replace('AFTER82_VALID', 'AFTER82_V2_VALID').replace('APPROVED_AFTER82_MECHANISM', 'APPROVED_AFTER82_V2_MECHANISM').replace('PREPARED_AFTER82 ', 'PREPARED_AFTER82_V2 ')


def seal(paths, **metadata):
    return dict(metadata, members=[dict(path=p, sha256=sha(HERE / p), size=(HERE / p).stat().st_size) for p in paths])


def main():
    assert sha(OLD / 'PACKAGE_SHA256.json') == PARENT_PACKAGE
    package = read(OLD / 'PACKAGE_SHA256.json')
    for rel, row in package['artifacts'].items():
        assert sha(OLD / rel) == row['sha256'] and (OLD / rel).stat().st_size == row['bytes'], rel
    failure_path = OLD / 'execution_candidate/ROOT_FAILURE_DIAGNOSIS_1.RAW.json'
    assert sha(failure_path) == FAILURE_SHA
    failure = read(failure_path)
    assert failure['runs'] == []
    failed_batch = json.loads(failure['files']['batch_failure.json']['text'])
    assert failed_batch['accepted_ids'] == [] and failed_batch['automatic_retry'] is False
    assert "len(ids) == len(set(ids)) == 11" in failure['files']['logs/minus_U_non-IID_FedSA_seed91010.log']['text']

    for rel in ['inventory_actual92_Full100refs.json', 'SCOPE.json', 'APPROVAL_TEMPLATE.json']:
        write(rel, rebind((OLD / rel).read_text(encoding='utf-8-sig')))
    for rel in ['SELECTED_10.txt', 'FULL100_ACTUAL_SOURCE_BINDINGS.json', 'NATIVE92_MINUS_THREEVIEW82.json']:
        (HERE / rel).write_bytes((OLD / rel).read_bytes())
    source = (OLD / 'bridge.py').read_text(encoding='utf-8-sig')
    assert source.count('len(ids) == len(set(ids)) == 11') == 1
    bridge = rebind(source).replace('len(ids) == len(set(ids)) == 11', 'len(ids) == len(set(ids)) == 10')
    bridge = bridge.replace('Excluded-prior82/selected11 boundary changed', 'Excluded-prior82/selected10 boundary changed')
    ast.parse(bridge)
    write('bridge.py', bridge)
    scope = read(HERE / 'SCOPE.json')
    scope.update(inventory_sha256=sha(HERE / 'inventory_actual92_Full100refs.json'), bridge_sha256=sha(HERE / 'bridge.py'))
    # Rewrite only our newly created file before its first seal.
    (HERE / 'SCOPE.json').write_text(json.dumps(scope, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    approval = read(HERE / 'APPROVAL_TEMPLATE.json')
    approval.update(inventory_sha256=scope['inventory_sha256'], bridge_sha256=scope['bridge_sha256'])
    (HERE / 'APPROVAL_TEMPLATE.json').write_text(json.dumps(approval, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    lineage = dict(status='ENGINEERING_CARDINALITY_FIX_PREPARED_ONLY', parent_package_sha256=PARENT_PACKAGE,
        parent_science_seal_sha256=sha(OLD / 'FILES_SHA256.json'), parent_execution_seal_sha256=sha(OLD / 'execution_candidate/EXECUTION_SOURCE_SHA256.json'),
        preserved_failure_path=failure_path.relative_to(HERE.parents[1]).as_posix(), preserved_failure_sha256=FAILURE_SHA,
        actual_failed_id='minus_U_non-IID_FedSA_seed91010', parent_accepted=0, parent_runs=[],
        cause='require_approval retained eleven for the exact-ten scope; rejected before scientific imports/CNN',
        fix='approval cardinality11 to10; fresh versioned scope/namespace/service; source pins rebound',
        unchanged_native_actual=92, unchanged_three_view_closed=82, selected_ids=scope['selected_ids'],
        original_eleven_science_functions_claim_corrected='require_approval is an engineering boundary; its cardinality must change',
        scientific_bind_runtime_replay_and_accept_source_unchanged=True,
        prior_after71_transport_failure_preconditions_not_reused=True)
    write('SOURCE_REUSE.json', lineage)
    pins = read(OLD / 'INPUT_PINS.json')
    pins['after82_v1_preserved_failure_lineage'] = lineage
    write('INPUT_PINS.json', pins)
    science_paths = [r['path'] for r in read(OLD / 'FILES_SHA256.json')['members']]
    write('FILES_SHA256.json', seal(science_paths, status='PREPARED_NOT_APPROVED', scope=NEW_SCOPE,
        excludes_this_seal_itself=True, CNN=False, dispatch=False))

    execution = HERE / 'execution_candidate'
    execution.mkdir(exist_ok=False)
    old_execution = OLD / 'execution_candidate'
    old_seal = read(old_execution / 'EXECUTION_SOURCE_SHA256.json')
    replacements = {
        sha(OLD / 'bridge.py'): sha(HERE / 'bridge.py'),
        sha(OLD / 'inventory_actual92_Full100refs.json'): sha(HERE / 'inventory_actual92_Full100refs.json'),
        sha(OLD / 'SCOPE.json'): sha(HERE / 'SCOPE.json'),
        sha(OLD / 'FILES_SHA256.json'): sha(HERE / 'FILES_SHA256.json'),
    }
    binding = rebind((old_execution / 'RUNTIME_BINDINGS.json').read_text(encoding='utf-8-sig'))
    write('execution_candidate/RUNTIME_BINDINGS.json', binding)
    replacements[sha(old_execution / 'RUNTIME_BINDINGS.json')] = sha(execution / 'RUNTIME_BINDINGS.json')
    for row in old_seal['members']:
        rel = row['path']
        if rel == 'RUNTIME_BINDINGS.json':
            continue
        text = rebind((old_execution / rel).read_text(encoding='utf-8-sig'))
        for old_sha, new_sha in replacements.items():
            text = text.replace(old_sha, new_sha)
        if rel == 'install_once.py':
            point = '\ndef budget_snapshot(snapshot):'
            guard = '''\ndef prior_failed_service_stopped():
    service = 'guardfed_celeba_mechanism_valid_after82'
    original_batch = '/workspace/guardfed_checks/celeba_mechanism_valid_incremental_after82_20261009/execution_candidate/batch.py'
    result = subprocess.run(['supervisorctl', 'status', service], capture_output=True, text=True)
    words = result.stdout.split()
    batch.require(len(words) >= 2 and words[0] == service and words[1] == 'EXITED',
        'Preserved original after82 service must be EXITED: ' + result.stdout + result.stderr)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            argv = [s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\\0') if s]
            batch.require(original_batch not in argv, 'Original after82 worker remains: ' + proc.name)
        except (FileNotFoundError, ProcessLookupError): continue
    return dict(service=service, status=result.stdout.strip(), supervisor_returncode=result.returncode, original_batch_processes=0,
        preserved_failure_sha256='b63558fb33e5dd7c576ff3a45f76e334e4d40ab5ca3e884618aec6a862852415')

'''
            assert text.count(point) == 1
            text = text.replace(point, guard + point)
            text = text.replace('    scope=batch.identities()', '    previous_failed_service=prior_failed_service_stopped()\n    scope=batch.identities()')
            text = text.replace('resources_before=before,resources_after=after,', 'previous_failed_service=previous_failed_service,resources_before=before,resources_after=after,')
        if rel.endswith('.py'):
            ast.parse(text)
        write('execution_candidate/' + rel, text)
    execution_paths = ['execution_candidate/' + r['path'] for r in old_seal['members']]
    new_execution_seal = seal(execution_paths, status='PREPARED_ONLY_REQUIRES_EXTERNAL_ROOT_EXECUTION_APPROVAL',
        parent_science_seal_sha256=sha(HERE / 'FILES_SHA256.json'), original_after82_execution_seal_sha256=sha(old_execution / 'EXECUTION_SOURCE_SHA256.json'))
    for row in new_execution_seal['members']:
        row['path'] = row['path'].removeprefix('execution_candidate/')
    write('execution_candidate/EXECUTION_SOURCE_SHA256.json', new_execution_seal)
    changed = ['bridge.py'] + execution_paths
    diff = ''.join(''.join(difflib.unified_diff((OLD / rel).read_text(encoding='utf-8-sig').splitlines(True),
        (HERE / rel).read_text(encoding='utf-8-sig').splitlines(True), fromfile=OLD_NAME+'/'+rel, tofile=NEW_NAME+'/'+rel)) for rel in changed)
    write('MINIMAL_SOURCE_DIFF.patch', diff)
    print(json.dumps(dict(status='LOCAL_SOURCE_REBIND_ONLY', selected=len(scope['selected_ids']),
        science_seal_sha256=sha(HERE/'FILES_SHA256.json'), execution_seal_sha256=sha(execution/'EXECUTION_SOURCE_SHA256.json'),
        old_failure_preserved=FAILURE_SHA, CNN=False, SSH=False, dispatch=False)))


if __name__ == '__main__':
    main()
