"""No-CNN checks using one actual accepted terminal and finite fail-stop fixtures."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import types
from unittest.mock import patch

sys.dont_write_bytecode = True
import bridge_adapter as adapter
import evaluate_remaining as queue

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = HERE / 'SELF_CHECK.json'
ACTUAL = ROOT / ('docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/'
                 'mechanism_science_backups_20261009/root_delta_20261010T034047Z')


def main():
    adapter.require(not REPORT.exists(), 'Keep the original check report; do not overwrite')
    plan = adapter.read(HERE / 'PLAN.json')
    bindings = adapter.read(HERE / 'SOURCE_BINDINGS.json')
    for relative, pin in bindings['input_pins'].items():
        path = ROOT / relative
        adapter.require(path.stat().st_size == pin['bytes'] and adapter.digest(path) == pin['sha256'], 'Input drift: ' + relative)
    parent = adapter.load('selfcheck_original_bridge', ROOT / bindings['dependency_local_paths']['parent_bridge'], bindings['parent_bridge_sha256'])
    baseline = adapter.read(ROOT / bindings['dependency_local_paths']['baseline_inventory'])
    prior = adapter.read(ROOT / 'tmp/celeba_mechanism_valid_C_after70_20261010/inventory_actual180_Full100refs.json')
    parent.validate_inventory(prior, baseline)
    manifest = adapter.read(ROOT / bindings['dependency_local_paths']['manifest'])
    adapter.require(plan['remaining620_ids'] == [e['id'] for e in manifest['jobs'] if e['id'] not in set(plan['excluded180_ids'])], 'Exact manifest-order complement differs')
    adapter.require(all(e['checkpoint_sha256'] is None for e in plan['entries']), 'Future checkpoint SHA invented')
    root = adapter.read(ACTUAL / 'ROOT_DELTA_VERIFICATION.json')
    adapter.require(root['archive_sha256'] == 'dae91858953378f79cd260ef8117b09120909022677ac6feabb9158399df3ed4', 'Actual fixture archive identity changed')
    archive = Path(root['archive_local_path'])
    adapter.require(archive.drive.upper() == 'F:' and adapter.digest(archive) == root['archive_sha256'], 'Fixture archive SHA/storage drift')
    identity = 'minus_C_non-IID_S-DFA_seed91001'
    entry = next(e for e in manifest['jobs'] if e['id'] == identity)
    inspection = adapter.read(ACTUAL / 'inspection/inspection.json')
    adapter.require(adapter.digest(ACTUAL / 'inspection/inspection.json') == root['inspection_sha256'], 'Actual fixture strict inspection changed')
    row = next(r for r in inspection['records'] if r['id'] == identity)
    with tarfile.open(archive, 'r:gz') as tar:
        members = json.load(tar.extractfile('backup_inventory.json'))['members']
        def checked_json(member):
            data = tar.extractfile(member).read()
            adapter.require(len(data) == members[member]['bytes'] and hashlib.sha256(data).hexdigest() == members[member]['sha256'], 'Actual JSON member drift')
            return json.loads(data)
        job = checked_json('jobs/' + identity + '.json')
        result = checked_json('runs/' + identity + '/result.json')
    full = next(r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+' and parent.cell(r) == (job['distribution'], job['attack'], job['config']['seed']))
    refs = {kind: dict(archive=str(archive), archive_sha256=root['archive_sha256'], member=member, **members[member]) for kind, member in
            [('checkpoint', 'runs/' + identity + '/model.pt'), ('result', 'runs/' + identity + '/result.json'), ('raw_job', 'jobs/' + identity + '.json')]}
    ns = dict(model_id=identity, job=job, entry=entry, result=result, row=row, control=full, refs=refs, b=parent, copy=copy)
    exec(compile((HERE / 'record_constructor.py').read_text(encoding='utf-8'), '<original constructor fixture>', 'exec'), ns)
    record = ns['record']; native_sha = adapter.canonical(row)
    inv = adapter.inventory(plan, record, native_sha)
    projected = adapter.project(parent, plan, identity, native_sha)
    projected.validate_inventory(inv, baseline)
    approval = dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY', scope=adapter.SCOPE,
        inventory_sha256='TEST_ONLY_IN_MEMORY', bridge_sha256=bindings['parent_bridge_sha256'], selected_ids=[identity],
        device='cpu', compute_threads=8, max_processes=1, allowed_cpus=adapter.CPUS,
        target_split='valid', final_test_dispatch=False, native_tolerance=1e-12)
    projected.require_approval(approval, 'TEST_ONLY_IN_MEMORY', identity, bindings['parent_bridge_sha256'])
    refusals = {}
    def refuse(label, call):
        try:
            call()
        except (ValueError, KeyError) as exc:
            refusals[label] = str(exc)
        else:
            raise AssertionError('Expected refusal: ' + label)
    for label, change in [('tolerance', {'native_tolerance': 1e-9}), ('test_split', {'target_split': 'test'}),
                          ('two_CNN_workers', {'max_processes': 2}), ('wrong_source', {'bridge_sha256': '0' * 64}),
                          ('duplicate_ID', {'selected_ids': [identity, identity]}), ('old180_ID', {'selected_ids': [plan['excluded180_ids'][0]]})]:
        refuse(label, lambda change=change: projected.require_approval(dict(approval, **change), 'TEST_ONLY_IN_MEMORY', identity, bindings['parent_bridge_sha256']))
    for label, field, value in [('checkpoint_drift', 'checkpoint', dict(record['checkpoint'], sha256='0' * 64)),
                                ('root_ID_drift', 'data_contract', dict(record['data_contract'], root_image_ids_sha256='0' * 64)),
                                ('partial_round69', 'terminal_round', 69), ('foreign_Full', 'variant', 'Full'),
                                ('wrong_rawjob', 'raw_job', dict(record['raw_job'], sha256='0' * 64))]:
        bad = copy.deepcopy(inv); bad['records'][0][field] = value
        refuse(label, lambda bad=bad: projected.validate_inventory(bad, baseline))
    source = Path(parent.__file__).read_text(encoding='utf-8')
    functions = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    for name, sha in bindings['unchanged_function_source_sha256'].items():
        adapter.require(hashlib.sha256(functions[name].encode()).hexdigest() == sha and getattr(projected, name).__code__ is getattr(parent, name).__code__, 'Original scientific/helper function changed: ' + name)
    snapshot = dict(completed=[identity], active=[], failed=[])
    adapter.require(queue.disposition(entry, snapshot, None, True, []) == 'READY_FOR_ORIGINAL_STRICT', 'Closed producer not ready')
    adapter.require(queue.disposition(entry, snapshot, {'pid': 1}, True, []) == 'WAIT_PRODUCER_EXIT', 'Live producer not deferred')
    pending = dict(completed=[], active=[], failed=[])
    adapter.require(queue.disposition(entry, pending, None, False, []) == 'WAIT_QUEUED_PRODUCER', 'Pending producer not deferred')
    refuse('orphan_partial', lambda: queue.disposition(entry, pending, None, True, []))
    refuse('preserved_training_failure', lambda: queue.disposition(entry, snapshot, None, True, ['failure.json']))
    fixtures = HERE / 'no_CNN_fixtures'; fixtures.mkdir()
    atom = fixtures / 'binding.json'; queue.save_new(atom, {'checkpoint_sha256': record['checkpoint']['sha256']})
    original_atom = atom.read_bytes()
    try:
        queue.save_new(atom, {'checkpoint_sha256': '0' * 64})
    except FileExistsError:
        adapter.require(atom.read_bytes() == original_atom, 'Atomic binding overwritten')
    else:
        raise AssertionError('Duplicate atomic binding allowed')
    # Exercise the real coordinator control flow, with no subprocess or CNN.
    failstop = []
    fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)
    for label, child_effect in [('child_nonzero', subprocess.CalledProcessError(7, ['TEST_ONLY_CHILD'])), ('exit0_missing_closure', None)]:
        attempt = fixtures / label; calls = []
        def child(*args, **kwargs):
            calls.append(args[0])
            if child_effect is not None:
                raise child_effect
            return types.SimpleNamespace(returncode=0)
        with patch.dict(sys.modules, fcntl=fake_fcntl), patch.object(queue, 'HERE', fixtures), patch.object(queue, 'RUNTIME', attempt), \
             patch.object(queue, 'policy'), patch.object(queue, 'dependencies', return_value={'evidence_v4': 'TEST_ONLY'}), \
             patch.object(queue, 'load', return_value=types.SimpleNamespace(live_worker=lambda *a: None)), \
             patch.object(queue, 'training_snapshot', return_value=(snapshot, {'TEST_ONLY': True})), patch.object(queue.subprocess, 'run', side_effect=child):
            small = dict(plan, entries=[entry, plan['entries'][1]])
            try:
                queue.manage(small, {'deadline_unix': queue.time.time() + 60}, Path('TEST_ONLY'), 'TEST_ONLY')
            except (subprocess.CalledProcessError, FileNotFoundError):
                failure = adapter.read(attempt / 'QUEUE_FAILURE.json')
                adapter.require(len(calls) == 1 and failure['remote_closed_ids'] == [] and failure['accepted_offserver'] == 0, 'Coordinator retried/advanced after failure')
                failstop.append(label)
            else:
                raise AssertionError('Coordinator accepted incomplete child')
    adapter.require('torch' not in sys.modules, 'No-CNN check unexpectedly imported Torch')
    queue.save_new(REPORT, dict(status='NO_CNN_METADATA_AND_FINITE_FAILSTOP_PASS', input_pin_checks=len(bindings['input_pins']),
        exact_remaining620=True, excluded180_metadata_regression=True, original_Full100_references_exact=True,
        actual_terminal_fixture=identity, fixture_archive_sha256=root['archive_sha256'], fixture_inspection_sha256=root['inspection_sha256'],
        future620_checkpoint_SHAs_still_null=True, positive_original_inventory_and_single_ID_approval=True,
        unchanged_parent_functions=list(bindings['unchanged_function_source_sha256']), original_scientific_function_code_exact=True,
        metadata_and_scheduler_refusals=refusals, immutable_atomic_binding_refusal=True, real_coordinator_failstop_fixtures=failstop,
        fresh_children_attempted_per_failure=1, CNN=0, Torch_imported=False, local_large_files_written=0,
        accepted_offserver=0, root_execution_approval_created=False, fixture_outputs_are_small_TEST_ONLY_JSON_and_logs=True))
    print(json.dumps({'status': 'PASS', 'metadata_refusals': len(refusals), 'real_failstop_fixtures': len(failstop), 'CNN': 0}))


if __name__ == '__main__':
    main()
