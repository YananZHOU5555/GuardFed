"""Authorized fifteen-terminal valid replay outer lifecycle; sealed bridge science reused."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
PREPARED = HERE / 'sealed_source'
REMOTE = Path('/workspace/guardfed_checks/celeba_mechanism_valid_incremental_v2_execution_20261009')
REPO = Path('/workspace/GuardFed-celeba-expanded')
CPUS = list(range(112, 120))
ROOT_APPROVAL_SHA = '701f95342f1da3a7e0ba6247b85320a7f5043ccf92475c1c713105866636f8de'
BRIDGE_SHA = '2950199b48e1b10a131ee8dafb992e44e002915b475976fa4525a63f25cc35b3'
INVENTORY_SHA = '288d2afb260f7eb77bcccba82e7edf6dbfe0519cd5bcc42fb546496236eeadbc'
OLD_SEAL_SHA = '70d0d920c4c5351c42efc9968fe3c38eed431d208b94bc8af486ba49d869a42d'
SELECTED = ['minus_U_IID_Benign_seed91009', 'minus_U_IID_Benign_seed91010', 'minus_U_IID_F Flip_seed91001', 'minus_U_IID_F Flip_seed91002', 'minus_U_IID_F Flip_seed91003', 'minus_U_IID_F Flip_seed91004', 'minus_U_IID_F Flip_seed91005', 'minus_U_IID_F Flip_seed91006', 'minus_U_IID_F Flip_seed91007', 'minus_U_IID_F Flip_seed91008', 'minus_U_IID_F Flip_seed91009', 'minus_U_IID_F Flip_seed91010', 'minus_U_IID_FedSA_seed91001', 'minus_U_IID_FedSA_seed91002', 'minus_U_IID_FedSA_seed91003']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(name, path, expected):
    require(digest(path) == expected, 'Sealed source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def identities(check_new_seal=True):
    require(digest(PREPARED / 'FILES_SHA256.json') == OLD_SEAL_SHA, 'Prepared19 seal changed')
    for row in read(PREPARED / 'FILES_SHA256.json')['members']:
        p=PREPARED / row['path']
        require(digest(p)==row['sha256'] and p.stat().st_size==row['size'], 'Prepared source/member changed')
    if check_new_seal:
        for row in read(HERE / 'EXECUTION_SOURCE_SHA256.json')['members']:
            p=HERE / row['path']
            require(digest(p)==row['sha256'] and p.stat().st_size==row['size'], 'Execution source changed')
    scope=read(PREPARED / 'SCOPE.json')
    require(scope['selected_ids']==SELECTED and scope['inventory_sha256']==INVENTORY_SHA
        and scope['bridge_sha256']==BRIDGE_SHA, 'Only approved15 frozen identities')
    return scope


def check_approval(approval, scope, scope_sha, seal_sha):
    require(digest(HERE / 'ROOT_APPROVED.json') == ROOT_APPROVAL_SHA, 'Root approval changed')
    root=read(HERE / 'ROOT_APPROVED.json')
    require(root['status']=='ROOT_REVIEW_PASS_BOUNDED_FIFTEEN_VALID_REPLAY' and root['execution_authorized_within_existing_user_request'] is True
        and root['source_seal_sha256']==OLD_SEAL_SHA and root['scope_sha256']==digest(PREPARED/'SCOPE.json')
        and root['inventory_sha256']==INVENTORY_SHA and root['bridge_sha256']==BRIDGE_SHA and root['selected_ids']==SELECTED,
        'Exact reviewed15 root authority required')
    require(approval.get('status')=='APPROVED_FIFTEEN_MECHANISM_VALID_REPLAY_ONLY'
        and approval.get('root_approval_sha256')==ROOT_APPROVAL_SHA and approval.get('execution_seal_sha256')==seal_sha,
        'Execution authority/seal changed')
    require(approval.get('scope_sha256')==scope_sha and approval.get('selected_ids')==SELECTED
        and approval.get('allowed_cpus')==CPUS and approval.get('compute_threads')==8 and approval.get('max_processes')==1,
        'Only15 IDs, one8-thread worker')
    require(approval.get('outputs')=={i:(REMOTE/'runs'/i).as_posix() for i in SELECTED}
        and approval.get('original_scope_outputs')==scope['outputs'] and approval.get('dependency_paths')==scope['dependency_paths'],
        'Exact new owned output mapping/dependencies required')
    require(approval.get('inventory_sha256')==INVENTORY_SHA and approval.get('bridge_sha256')==BRIDGE_SHA
        and approval.get('target_split')=='valid' and approval.get('native_tolerance')==1e-12
        and approval.get('final_test_dispatch') is False and approval.get('new_full_inference')==0
        and approval.get('automatic_retry_authorized') is False, 'Scientific scope changed')


def approved(path, expected_sha):
    scope = identities()
    require(path is not None and expected_sha is not None and Path(path).is_file(), 'External approval absent; preparation only')
    require(digest(path) == expected_sha, 'External approval changed')
    approval = read(path)
    check_approval(approval, scope, digest(PREPARED / 'SCOPE.json'), digest(HERE / 'EXECUTION_SOURCE_SHA256.json'))
    return scope, approval


def runtime_policy():
    require(sys.platform == 'linux' and not sys.flags.optimize and HERE == REMOTE, 'Exact isolated Linux directory without -O required')
    require(digest('/etc/vast-agents-guide.md') == '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa',
            'Guide changed; read current instance guide before dispatch')
    require(set(CPUS) <= os.sched_getaffinity(0) and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Approved CPUs and nice10 required')
    os.sched_setaffinity(0, CPUS)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def fresh_output(output):
    output = Path(output)
    require(not output.exists() and not output.with_name(output.name + '.bridge_failure.json').exists(),
            'Existing partial/failure preserved; no retry')


def worker(parent_path, parent_sha, identity, only_strict=False):
    scope, parent = approved(parent_path, parent_sha)
    require(identity in SELECTED, 'Foreign/pending/Full or any closed8 ID refused')
    runtime_policy()
    child_path = HERE / 'approvals' / (identity + '.json')
    child = read(child_path)
    require(child['parent_batch_approval_sha256'] == parent_sha and child['selected_ids'] == SELECTED and child['selected_id'] == identity
            and child['output'] == parent['outputs'][identity] and child['dependency_paths'] == parent['dependency_paths'],
            'Per-ID delegated approval mismatch')
    output = Path(child['output'])
    if not only_strict:
        fresh_output(output)
    require(output.parent == HERE / 'runs' and not output.parent.is_symlink()
            and output.parent.resolve() == HERE.resolve() / 'runs', 'Output parent cannot escape exclusive batch directory')
    import fcntl
    lock = (HERE / 'compute_worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bridge = load('sealed_original_mechanism_bridge', PREPARED / 'bridge.py', BRIDGE_SHA)
    runtime = bridge.bind_runtime(PREPARED / 'inventory_actual23_Full100refs.json', INVENTORY_SHA,
        child['dependency_paths'], REPO, identity, child_path, digest(child_path))
    if not only_strict:
        runtime['replay_one'](output, wall_seconds=1800)
    acceptance = runtime['accept_saved_predictions'](output)
    require(acceptance['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED'
            and acceptance['native_comparison']['accepted'], 'Original strict bridge did not accept')
    if not only_strict:
        save_new(output / 'strict_acceptance.json', acceptance)
    else:
        require(read(output / 'strict_acceptance.json') == acceptance, 'Strict result changed')
    print(json.dumps(dict(status=acceptance['status'], id=identity,
        native_difference=acceptance['native_comparison']['max_abs_difference'], new_training=0, new_Full_inference=0)), flush=True)


def manage(parent_path, parent_sha):
    scope, parent = approved(parent_path, parent_sha); runtime_policy()
    from resource_extra import resource_snapshot
    before = resource_snapshot()
    require(not (HERE / 'batch_resource_before.json').exists() and not (HERE / 'batch_failure.json').exists()
            and not (HERE / 'runs').exists() and not (HERE / 'approvals').exists(), 'Prior batch/runtime evidence blocks implicit retry')
    import fcntl
    lock = (HERE / 'batch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save_new(HERE / 'batch_resource_before.json', before)
    for name in ('runs', 'approvals', 'logs'):
        (HERE / name).mkdir(exist_ok=False)
    completed = []
    try:
        for identity in SELECTED:
            child = dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY', scope='MECHANISM_TERMINAL_VALID_REPLAY_INCREMENTAL_V2',
                inventory_sha256=INVENTORY_SHA, bridge_sha256=BRIDGE_SHA, selected_ids=SELECTED, selected_id=identity, device='cpu',
                compute_threads=8, max_processes=1, allowed_cpus=CPUS, target_split='valid', native_tolerance=1e-12,
                final_test_dispatch=False, output=parent['outputs'][identity], dependency_paths=parent['dependency_paths'],
                parent_batch_approval_sha256=parent_sha, source_outer_sha256=digest(__file__))
            save_new(HERE / 'approvals' / (identity + '.json'), child)
            with (HERE / 'logs' / (identity + '.log')).open('x') as log:
                subprocess.run([sys.executable, '-u', str(HERE / 'batch.py'), 'worker', '--approved', str(parent_path),
                    '--approved-sha256', parent_sha, '--id', identity], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1900)
            output = Path(parent['outputs'][identity]); strict = read(output / 'strict_acceptance.json')
            require(strict['id'] == identity and strict['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED', 'Foreign/incomplete child strict result')
            completed.append(dict(id=identity, checkpoint_sha256=strict['checkpoint_sha256'], views=strict['views'],
                strict_acceptance_sha256=digest(output / 'strict_acceptance.json'),
                native_difference=strict['native_comparison']['max_abs_difference']))
            save_new(HERE / ('completed_' + identity + '.json'), completed[-1])
        save_new(HERE / 'batch_complete.json', dict(status='FIFTEEN_VALID_REPLAYS_STRICT_ACCEPTED_BACKUP_PENDING', records=completed,
            new_training=0, new_Full_inference=0, final_protocol_status='PREPARED_NOT_FROZEN',
            excluded_already_accepted=read(PREPARED/'SCOPE.json')['already_closed_replay_ids'], resources_after=resource_snapshot()))
    except BaseException as error:
        save_new(HERE / 'batch_failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            accepted_ids=[row['id'] for row in completed], automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect','manage','worker','strict'])
    parser.add_argument('--approved', type=Path); parser.add_argument('--approved-sha256'); parser.add_argument('--id')
    args = parser.parse_args()
    if args.action == 'inspect':
        identities(); print('AUTHORIZED_FIFTEEN accepted-terminal valid replays; no inference or training')
    elif args.action == 'manage':
        manage(args.approved, args.approved_sha256)
    else:
        worker(args.approved, args.approved_sha256, args.id, only_strict=args.action == 'strict')
