"""Local lifecycle/resource refusals with synthetic authority; no Linux/CNN dispatch."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types

sys.dont_write_bytecode = True
import batch
import install_once

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OLD = REPO / 'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009'


def function(path, name, substitutions=()):
    text = path.read_text(encoding='utf-8-sig')
    for a, b in substitutions:
        text = text.replace(a, b)
    tree = ast.parse(text)
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def main():
    scope = batch.identities(False)
    assert len(batch.SELECTED) == len(set(batch.SELECTED)) == 37
    assert not set(batch.SELECTED) & set(scope['already_closed_replay_ids'])
    assert len(scope['already_closed_replay_ids']) == 23
    unchanged = []
    for name in ('require', 'digest', 'read', 'save_new', 'load', 'approved', 'runtime_policy', 'fresh_output'):
        assert function(HERE/'batch.py', name) == function(OLD/'batch.py', name)
        unchanged.append('batch.'+name)
    normalizations = [('inventory_actual23_', 'inventory_actual60_'), ('closed8 ID', 'closed23 ID'),
        ('INCREMENTAL_V2', 'INCREMENTAL_NEXT37'), ('FIFTEEN_VALID', 'NEXT37_VALID')]
    for name in ('worker', 'manage'):
        assert function(HERE/'batch.py', name) == function(OLD/'batch.py', name, normalizations)
        unchanged.append('batch.'+name+' (scope metadata only)')
    assert (HERE/'resource_extra.py').read_bytes() == (OLD/'resource_extra.py').read_bytes()
    assert function(HERE/'verify_saved_increment.py', 'verify') == function(OLD/'verify_saved_increment.py', 'verify',
        [('288d2afb260f7eb77bcccba82e7edf6dbfe0519cd5bcc42fb546496236eeadbc', batch.INVENTORY_SHA), ('exact15-scope', 'exact37-scope')])
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'))
    refusals = []

    def reject(name, action):
        try:
            action()
        except (ValueError, FileExistsError):
            refusals.append(name)
        else:
            raise AssertionError('Unexpected acceptance: '+name)

    original_here = batch.HERE
    fixture = Path(tempfile.mkdtemp(prefix='_selfcheck_', dir=HERE))
    assert fixture.resolve().parent == HERE.resolve()
    try:
        batch.HERE = fixture
        seal = 'f'*64
        root = batch.read(HERE/'ROOT_REVIEW_TEMPLATE.json')
        root.update(status='ROOT_REVIEW_PASS_BOUNDED_NEXT37_VALID_REPLAY',
            execution_authorized_within_existing_user_request=True, execution_seal_sha256=seal)
        batch.save_new(fixture/'ROOT_APPROVED.json', root)
        a = batch.read(HERE/'APPROVED_TEMPLATE.json')
        reject('unfilled_approval_template', lambda: batch.check_approval(a, scope, batch.digest(batch.PREPARED/'SCOPE.json'), seal))
        a.update(status='APPROVED_NEXT37_MECHANISM_VALID_REPLAY_ONLY', root_approval_sha256=batch.digest(fixture/'ROOT_APPROVED.json'), execution_seal_sha256=seal)
        check = lambda value: batch.check_approval(value, scope, batch.digest(batch.PREPARED/'SCOPE.json'), seal)
        check(a)
        for key, value in [('selected_ids', batch.SELECTED[:-1]), ('selected_ids', batch.SELECTED[:-1]+[batch.SELECTED[0]]),
            ('selected_ids', [scope['already_closed_replay_ids'][0]]+batch.SELECTED[1:]), ('allowed_cpus', list(range(8))),
            ('compute_threads', 1), ('max_processes', 2), ('outputs', scope['outputs']), ('dependency_paths', {}),
            ('automatic_retry_authorized', True), ('final_test_dispatch', True), ('native_tolerance', 1e-9),
            ('new_full_inference', 1), ('target_split', 'test'), ('root_approval_sha256', '0'*64),
            ('inventory_sha256', '0'*64), ('bridge_sha256', '0'*64), ('execution_seal_sha256', '0'*64)]:
            bad = copy.deepcopy(a); bad[key] = value
            reject(key+'_'+str(len(refusals)), lambda bad=bad: check(bad))
        root['execution_seal_sha256']='0'*64
        (fixture/'ROOT_APPROVED.json').write_text(json.dumps(root))
        bad=copy.deepcopy(a);bad['root_approval_sha256']=batch.digest(fixture/'ROOT_APPROVED.json')
        reject('root_review_wrong_execution_seal', lambda: check(bad))
        root['execution_seal_sha256']=seal
        (fixture/'ROOT_APPROVED.json').write_text(json.dumps(root,indent=2)+'\n')
        (fixture/'ROOT_APPROVED.json').write_bytes((fixture/'ROOT_APPROVED.json').read_bytes()+b' ')
        reject('root_review_byte_drift', lambda: check(a))
        batch.save_new(fixture/'EXECUTION_DRAFT.json',a)
        installer_original=(install_once.HERE,install_once.os,batch.REMOTE)
        try:
            install_once.HERE=fixture;batch.REMOTE=fixture
            install_once.os=types.SimpleNamespace(PRIO_PROCESS=0,getpriority=lambda *args:10)
            reject('external_draft_SHA_mismatch',lambda: install_once.main('0'*64))
            assert not (fixture/'APPROVED.json').exists()
        finally:
            install_once.HERE,install_once.os,batch.REMOTE=installer_original
        fresh = fixture/'fresh'
        batch.fresh_output(fresh)
        fresh.mkdir()
        reject('existing_output', lambda: batch.fresh_output(fresh))
        failed = fixture/'failed'
        failed.with_name(failed.name+'.bridge_failure.json').write_text('{}')
        reject('preserved_bridge_failure', lambda: batch.fresh_output(failed))

        good = install_once.budget_snapshot(dict(total_effective_nominal=119.88, actual_quota_cores=122.88))
        assert good['conservative_total_including_this8'] == 122.88 and good['additional_three_are_reservation_not_measured_usage']
        reject('extra_three_exceeds_quota', lambda: install_once.budget_snapshot(dict(total_effective_nominal=120, actual_quota_cores=122.88)))
        for value in (float('inf'), float('nan'), -1, 'max', True):
            reject('invalid_quota_'+repr(value), lambda value=value: install_once.budget_snapshot(dict(total_effective_nominal=8, actual_quota_cores=value)))
        install_once.assert_cpu_available([dict(pid=1,cpus=list(range(112,120))),dict(pid=2,cpus=[105]),dict(pid=3,cpus=list(range(512)))], 1)
        reject('restricted_CPU_owner', lambda: install_once.assert_cpu_available([dict(pid=2,cpus=list(range(112,120)))], 1))
        reject('restricted_CPU_overlap', lambda: install_once.assert_cpu_available([dict(pid=2,cpus=[111,112])], 1))

        # Execute the real outer failure path with only the subprocess call replaced.
        flow = fixture/'flow'; flow.mkdir()
        batch.HERE = flow
        flow_a = copy.deepcopy(a); flow_a['outputs'] = {i:str(flow/'runs'/i) for i in batch.SELECTED}
        originals = (batch.approved, batch.runtime_policy, batch.subprocess.run)
        modules = {name:sys.modules.get(name) for name in ('fcntl','resource_extra')}
        calls = []
        try:
            batch.approved = lambda *args: (scope, flow_a)
            batch.runtime_policy = lambda: None
            sys.modules['fcntl'] = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            sys.modules['resource_extra'] = types.SimpleNamespace(resource_snapshot=lambda: {'synthetic_no_CNN':True})
            def fail_child(command, **kwargs):
                calls.append(command)
                assert (flow/'runs').is_dir() and (flow/'approvals').is_dir()
                raise subprocess.CalledProcessError(7, command)
            batch.subprocess.run = fail_child
            try:
                batch.manage(flow/'synthetic_parent.json', 'e'*64)
            except subprocess.CalledProcessError:
                pass
            else:
                raise AssertionError('Child failure did not failstop')
            failure = batch.read(flow/'batch_failure.json')
            assert len(calls)==1 and not failure['automatic_retry'] and failure['accepted_ids']==[]
            assert not (flow/'batch_complete.json').exists() and len(list((flow/'approvals').glob('*.json')))==1
            reject('failed_batch_blocks_resume', lambda: batch.manage(flow/'synthetic_parent.json', 'e'*64))
            # Close synthetic lock streams before Windows fixture cleanup.
            import gc
            gc.collect()
        finally:
            batch.approved, batch.runtime_policy, batch.subprocess.run = originals
            for name, value in modules.items():
                if value is None:sys.modules.pop(name, None)
                else:sys.modules[name]=value
    finally:
        batch.HERE = original_here
        assert fixture.resolve().parent == HERE.resolve()
        shutil.rmtree(fixture)
    assert not any(name in sys.modules for name in ('torch','numpy','replay'))
    report = dict(status='PASS_NEXT37_EXECUTION_NO_CNN', selected_n=37, excluded_closed_n=23,
        rejections=refusals, rejection_count=len(refusals), source16_seal_sha256=batch.OLD_SEAL_SHA,
        unchanged_functions=unchanged, resource_extra_original_bytes=True, saved_array_science_unchanged=True,
        first_child_failure_calls=1, failure_preserved=True, automatic_retry=False,
        extra_three_reservation_not_measurement=True, fresh_Linux_proc_scan_executed=False,
        supervisor_installed=False, real_subprocesses_started=0, CNN_inference=False, registration=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__=='__main__':
    main()
