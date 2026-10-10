"""One explicit remaining620 export/offserver verification; never runs CNN or adopts IDs."""
import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE_REMOTE = Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010')
TRANSPORT_REMOTE = Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f: f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def own_seal(expected):
    require(sha(HERE / 'FILES_SHA256.json') == expected, 'External transport source seal changed')
    for name, pin in read(HERE / 'FILES_SHA256.json')['files'].items():
        require(sha(HERE / name) == pin['sha256'] and (HERE / name).stat().st_size == pin['bytes'], 'Transport source drift: ' + name)


def load(name, path, expected):
    require(sha(path) == expected, 'Dependency SHA drift: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def source(path, expected):
    pins = read(HERE / 'INPUTS.json')
    require(expected == pins['remaining620_v2_seal_sha256'] and sha(path / 'FILES_SHA256.json') == expected, 'Wrong frozen620 source release')
    if 'bridge_adapter' in sys.modules:
        require(Path(sys.modules['bridge_adapter'].__file__).resolve() == (path / 'bridge_adapter.py').resolve(), 'Foreign metadata module cached')
    sys.path.insert(0, str(path))
    q = load('remaining620_transport_source', path / 'evaluate_remaining.py', read(path / 'FILES_SHA256.json')['files']['evaluate_remaining.py']['sha256'])
    return q, q.source_identity(), pins


def select_delta(plan, requested, prior):
    require(requested and len(requested) == len(set(requested)) and len(prior) == len(set(prior)), 'Empty/duplicate export IDs')
    allowed = plan['remaining620_ids']
    require(set(requested) <= set(allowed) and set(prior) <= set(allowed) and not set(requested) & set(prior), 'Old180/Full/foreign/already transported ID')
    require(requested == [i for i in allowed if i in set(requested)], 'Export must preserve original manifest order')
    return [i for i in allowed if i in set(prior) | set(requested)]


def previous(root, path, expected, plan):
    latest_path = root / 'TRANSPORT_LATEST.json'
    if not latest_path.exists():
        require(path is None and expected is None and not any(p.is_dir() for p in root.iterdir()), 'Missing first-export chain or preserved partial/failure')
        return [], None
    latest = read(latest_path)
    require(path is not None and path.resolve() == Path(latest['receipt']).resolve() and expected == latest['receipt_sha256'] == sha(path), 'Explicit previous transport receipt differs from actual latest')
    seen, visited, current, current_sha = set(), set(), path, expected
    while current:
        require(current.resolve().is_relative_to(root.resolve()) and current.resolve() not in visited, 'Unsafe/cyclic receipt chain')
        visited.add(current.resolve())
        require(sha(current) == current_sha, 'Prior receipt chain drift')
        r = read(current)
        require(r['source_seal_sha256'] == read(HERE / 'INPUTS.json')['remaining620_v2_seal_sha256'] and r['accepted_offserver'] == 0, 'Wrong prior source/adoption status')
        select_delta(plan, r['accepted_new_ids'], list(seen))
        seen.update(r['accepted_new_ids'])
        require(not (current.parent / 'EXPORT_FAILURE.json').exists(), 'Prior transport failure requires review')
        current_sha = r['previous_backup_receipt_sha256']; current = Path(r['previous_receipt']) if current_sha else None
    require(set(latest['all_transported_ids']) == seen and len(latest['all_transported_ids']) == len(seen), 'Prior cumulative IDs differ from chain')
    known = {p.resolve() for p in root.glob('*/backup_receipt.json')}
    require(known == visited and all((d / 'backup_receipt.json').resolve() in known and not (d / 'EXPORT_FAILURE.json').exists() for d in root.iterdir() if d.is_dir()), 'Orphan/failed export forbids automatic recovery')
    return latest['all_transported_ids'], latest


def closed(q, plan, runtime, identity, review_sha, baseline, parent):
    task, out = runtime / 'tasks' / identity, runtime / 'runs' / identity
    require({p.name for p in out.iterdir()} == {'receipt.json', 'bridge_receipt.json', 'validation_predictions.npz', 'strict_acceptance.json'}, 'Partial/unexpected output')
    remote, binding, inv, approval = (read(task / n) for n in ('REMOTE_COMPLETE.json', 'binding.json', 'inventory.json', 'APPROVED.json'))
    strict, receipt, bridge = (read(out / n) for n in ('strict_acceptance.json', 'receipt.json', 'bridge_receipt.json'))
    require(all(v['id'] == identity for v in (remote, binding, strict, receipt, bridge)), 'Mixed ID')
    require(remote['status'] == 'REMOTE_STRICT_CLOSED_PENDING_OFFSERVER' and remote['accepted_offserver'] == 0, 'Not remote closed; never accept offserver here')
    require(binding['status'] == 'IMMUTABLE_TERMINAL_ONCE_BOUND' and binding['plan_sha256'] == sha(q.HERE / 'PLAN.json'), 'Wrong immutable terminal/plan')
    require(remote['parent_review_sha256'] == binding['parent_review_sha256'] == approval['parent_queue_review_sha256'] == review_sha, 'Root approval lineage changed')
    for name, field in [('binding.json', 'binding_sha256'), ('inventory.json', 'inventory_sha256')]:
        require(sha(task / name) == remote[field], 'Task binding/inventory SHA drift')
    record = binding['record']; native = binding['native_acceptance']
    require(native['status'] == 'ORIGINAL_V4_TERMINAL_STRICT_PASS_REMOTE_ONLY' and native['id'] == identity
            and q.canonical(native) == binding['native_acceptance_canonical_sha256']
            and native['source_sha256'] == plan['dependency_sha256']['evidence_v4'], 'Original per-ID strict70 proof changed')
    require(inv['records'] == [record] and record['accepted_v4_row'] == native['row'], 'Bound/native/inventory records differ')
    projected = q.project(parent, plan, identity, binding['native_acceptance_canonical_sha256'])
    projected.validate_inventory(inv, baseline)
    require(approval['binding_sha256'] == remote['binding_sha256'] and approval['inventory_sha256'] == remote['inventory_sha256'], 'Delegated approval changed')
    projected.require_approval(approval, remote['inventory_sha256'], identity, plan['dependency_sha256']['parent_bridge'])
    require(approval['output'] == str(out) and bridge['scope'] == strict['scope'] == q.SCOPE, 'Namespace/scope changed')
    require(record['checkpoint']['sha256'] == binding['checkpoint_sha256'] == remote['checkpoint_sha256'] == strict['checkpoint_sha256'] == receipt['checkpoint_sha256'], 'Mixed checkpoint')
    require(strict['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['native_comparison']['accepted']
            and strict['native_comparison']['tolerance'] == 1e-12 and strict['native_comparison']['max_abs_difference'] <= 1e-12
            and strict['native_comparison'] == receipt['native_comparison'] and strict['views'] == receipt['views'], 'Strict/native/views changed')
    require(remote['strict_sha256'] == sha(out / 'strict_acceptance.json') and remote['prediction_arrays_sha256'] == receipt['prediction_arrays_sha256'] == sha(out / 'validation_predictions.npz'), 'Strict/arrays SHA drift')
    require(sha(out / 'receipt.json') == bridge['scientific_body_receipt_sha256'] and sha(out / 'bridge_receipt.json') == strict['bridge_receipt_sha256']
            and bridge['inventory_sha256'] == strict['inventory_sha256'] == remote['inventory_sha256'] and bridge['approval_sha256'] == sha(task / 'APPROVED.json'), 'Saved receipt chain changed')
    require(bridge['source_before'] == bridge['source_after'] and bridge['artifact_before'] == bridge['artifact_after']
            and receipt['weights_before'] == receipt['weights_after'] and not receipt['optimizer_created'] and not receipt['gradients_created'], 'Inputs/weights changed')
    require(bridge['inventory_record_sha256'] == receipt['model_inventory_record_sha256'] == q.canonical(record)
            and bridge['bridge_source_sha256'] == plan['dependency_sha256']['parent_bridge']
            and bridge['v2_source_sha256'] == plan['dependency_sha256']['v2'] and bridge['v4_source_sha256'] == plan['dependency_sha256']['evidence_v4'], 'Original scientific source/record drift')
    require(all(receipt[k] == record[k] for k in ('method', 'distribution', 'attack', 'seed', 'config_canonical_sha256'))
            and receipt['original_result_sha256'] == record['result']['sha256'] and receipt['original_job_sha256'] == record['raw_job']['sha256']
            and bridge['paired_full_reference'] == strict['paired_full_reference'] == record['paired_full'], 'Recipe/seed/original native/Full identity changed')
    expected_queue = {n: pin['sha256'] for n, pin in read(q.HERE / 'FILES_SHA256.json')['files'].items()}
    require(remote['queue_source_before'] == approval['queue_source_before'] == expected_queue, 'Queue version/source identity drift')
    require(all(sha(p) == expected for p, expected in record['accepted_v4_row']['files'].items()), 'Actual model/result/rawjob/audit/log changed after strict closure')
    require(receipt['valid_n'] == 19867 and receipt['root_reconstruction']['root_n'] == 16277, 'Wrong root/valid count')
    return record, {**{f'runs/{identity}/{p.name}': p for p in out.iterdir()},
                    **{f'runtime/{identity}/{n}': task / n for n in ('binding.json', 'inventory.json', 'APPROVED.json', 'REMOTE_COMPLETE.json', 'worker.log')}}


def writer_body(path, expected):
    require(sha(path) == expected, 'Original archive writer changed')
    text = path.read_text(encoding='utf-8'); nodes = ast.parse(text).body
    index = next(i for i, n in enumerate(nodes) if isinstance(n, ast.With) and ast.unparse(n.items[0].context_expr) == "tarfile.open(archive, 'w:gz')")
    require(isinstance(nodes[index + 1], ast.For), 'Original post-archive hash guard absent')
    return '\n'.join(ast.get_source_segment(text, n) for n in nodes[index:index + 2])


def saved_verifier_body(path, expected):
    require(sha(path) == expected, 'Original saved-array verifier changed')
    text = path.read_text(encoding='utf-8'); node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    body = ast.get_source_segment(text, node); anchor = "assert sha(inventory)=='b4ffdf5f3dc549e397f82db49290cdceb1ee024478cdc4e58365f5c537d144a9'"
    require(body.count(anchor) == 1, 'Original verifier inventory-pin anchor changed')
    body = body.replace(anchor, 'assert sha(inventory)==EXPECTED_SNAPSHOT_SHA', 1)
    claim = 'This exact10-scope subset of validation terminal replays; not final test or complete900 mechanism evidence.'
    require(body.count(claim) == 1, 'Original metadata claim anchor changed')
    return body.replace(claim, 'This explicit remaining620 transport subset of validation terminal replays; not final test or complete900 mechanism evidence.', 1)


def export(args, q, plan, pins):
    require(sys.platform == 'linux' and HERE == TRANSPORT_REMOTE and args.source == SOURCE_REMOTE and args.runtime == q.RUNTIME, 'Exact new Linux/source/runtime namespace required')
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE') and os.getpriority(os.PRIO_PROCESS, 0) == 10, 'Assertions/nice10 required')
    os.sched_setaffinity(0, [110])
    require('idle' in subprocess.check_output(['ionice', '-p', str(os.getpid())], text=True).lower(), 'Idle I/O required')
    require(sha(args.review) == args.review_sha256, 'External actual queue review changed')
    review = read(args.review)
    require(review['status'] == 'ROOT_APPROVED_REMAINING620_MECHANISM_VALID_EVALUATION' and review['execution_authorized'] is True
            and review['source_seal_sha256'] == args.source_seal and review['plan_sha256'] == sha(args.source / 'PLAN.json')
            and review['selected_ids'] == plan['remaining620_ids'] and review['excluded_ids'] == plan['excluded180_ids'], 'Wrong queue approval/scope')
    root = HERE / 'exports'; root.mkdir(exist_ok=True)
    import fcntl
    lock = (root / 'transport.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior, latest = previous(root, args.previous, args.previous_sha256, plan)
    cumulative = select_delta(plan, args.ids, prior)
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            argv = [x.decode(errors='replace') for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            if str(args.source / 'evaluate_remaining.py') in argv and 'worker' in argv and '--id' in argv: active.append(argv[argv.index('--id') + 1])
        except (FileNotFoundError, ProcessLookupError): pass
    require(not set(active) & set(args.ids), 'Evaluation child still alive; no transport of open output')
    require(args.tag and all(c.isalnum() or c in '_-' for c in args.tag), 'Unsafe export tag')
    dest = root / args.tag; dest.mkdir(exist_ok=False)
    try:
        d = plan['remote_dependencies']; baseline = read(d['baseline_inventory'])
        parent = q.load('transport_original_bridge', d['parent_bridge'], plan['dependency_sha256']['parent_bridge'])
        files, records = {}, []
        for identity in args.ids:
            record, selected = closed(q, plan, args.runtime, identity, args.review_sha256, baseline, parent)
            records.append(record); files.update(selected)
        snapshot = dict(status='TRANSPORT_SNAPSHOT_NOT_SCIENTIFIC_ADOPTION', scope=q.SCOPE, records=records,
            selected_replay_ids=args.ids, excluded_prior_replay_ids=plan['excluded180_ids'], previously_transported_ids=prior,
            source_seal_sha256=args.source_seal, plan_sha256=sha(args.source / 'PLAN.json'), Full_inference=0, accepted_offserver=0)
        save(dest / 'snapshot_inventory.json', snapshot); files['snapshot_inventory.json'] = dest / 'snapshot_inventory.json'
        if latest is None:
            for name in read(args.source / 'FILES_SHA256.json')['files']: files['source/queue/' + name] = args.source / name
            files['source/queue/FILES_SHA256.json'] = args.source / 'FILES_SHA256.json'
            for name in read(HERE / 'FILES_SHA256.json')['files']: files['source/transport/' + name] = HERE / name
            files['source/transport/FILES_SHA256.json'] = HERE / 'FILES_SHA256.json'
            files['source/ROOT_APPROVED.json'] = args.review
            preflight = args.source / 'ROOT_LINUX_PREFLIGHT.json'
            require(sha(preflight) == review['Linux_preflight_sha256'], 'Original root Linux preflight changed')
            files['source/ROOT_LINUX_PREFLIGHT.json'] = preflight
            for key in ('v2', 'v3', 'evaluator', 'parent_bridge', 'evidence_v4'):
                require(sha(d[key]) == plan['dependency_sha256'][key], 'Original scientific dependency changed')
                files['source/dependencies/' + key + '.py'] = Path(d[key])
            for key in ('writer', 'saved_verifier'):
                pin = pins['dependencies'][key]; require(sha(pin['remote']) == pin['sha256'], 'Compatibility tool drift')
                files['source/dependencies/' + key + '.py'] = Path(pin['remote'])
        require(all(p.is_file() and not p.is_symlink() and p.name != 'model.pt' for p in files.values()), 'Missing/symlink/old-model archive input')
        files = {n: dict(path=p, sha256=sha(p), bytes=p.stat().st_size) for n, p in files.items()}
        inventory = dict(schema='remaining620_replay_transport_v1', accepted_new_ids=args.ids, all_transported_ids=cumulative,
            previous_backup=latest, records=records, source_seal_sha256=args.source_seal, snapshot_sha256=sha(dest / 'snapshot_inventory.json'),
            models_repacked=0, new_training=0, new_test_inference=0, accepted_offserver=0,
            members={n: {k: r[k] for k in ('sha256', 'bytes')} for n, r in files.items()})
        payload = (json.dumps(inventory, indent=2, allow_nan=False) + '\n').encode(); archive = dest / 'incremental_valid_three_views.tar.gz'
        pin = pins['dependencies']['writer']; body = writer_body(Path(pin['remote']), pin['sha256'])
        exec(compile(body, '<original sealed archive writer>', 'exec'), dict(tarfile=tarfile, io=io, payload=payload, archive=archive, files=files, batch=SimpleNamespace(require=require, digest=sha)))
        q.source_identity(); own_seal(args.transport_seal)
        require(sha(args.review) == args.review_sha256 and all(sha(p) == expected for r in records for p, expected in r['accepted_v4_row']['files'].items()), 'Review/model/native artifacts changed during transport')
        receipt = dict(status='REMOTE_TRANSPORT_VERIFIED_PENDING_OFFSERVER', archive_sha256=sha(archive), inventory_sha256=hashlib.sha256(payload).hexdigest(),
            accepted_new_ids=args.ids, all_transported_ids=cumulative, members=len(files) + 1, source_host=socket.gethostname(), archive=str(archive),
            source_seal_sha256=args.source_seal, snapshot_sha256=sha(dest / 'snapshot_inventory.json'), parent_review_sha256=args.review_sha256,
            previous_receipt=str(args.previous) if args.previous else None, previous_backup_receipt_sha256=args.previous_sha256,
            accepted_offserver=0, models_repacked=0, original_writer_sha256=pin['sha256'], transport_source_seal_sha256=args.transport_seal)
        pin = pins['dependencies']['archive_verifier']; verifier = load('transport_original_archive_verifier', Path(pin['remote']), pin['sha256'])
        verifier.verify_archive(archive, receipt)
        save(dest / 'backup_receipt.json', receipt)
        new_latest = root / 'TRANSPORT_LATEST.next.json'
        save(new_latest, dict(receipt=str(dest / 'backup_receipt.json'), receipt_sha256=sha(dest / 'backup_receipt.json'), all_transported_ids=cumulative, accepted_offserver=0))
        os.replace(new_latest, root / 'TRANSPORT_LATEST.json')
        print(json.dumps(receipt))
    except BaseException as exc:
        save(dest / 'EXPORT_FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), requested_ids=args.ids, accepted_offserver=0, automatic_retry=False))
        raise


def F_guard(path, needed):
    require(sys.platform == 'win32' and Path(path).resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()), 'Local bulk must stay on F:/YananResearchStorage/GuardFed')
    volume = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command', 'Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,SizeRemaining | ConvertTo-Json -Compress'], text=True))
    require(volume['DriveLetter'] == 'F' and volume['FileSystemLabel'] == 'Yanan 2TB' and volume['SizeRemaining'] >= needed + 512 * 1024 ** 2, 'Fresh F-volume identity/headroom failed')
    return volume


def verify(args, q, plan, pins):
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE') and sha(args.receipt) == args.receipt_sha256, 'Assertions and external receipt SHA required')
    F_guard(args.archive, 0); require(not args.out.exists(), 'Never overwrite an offserver attempt')
    root = HERE.parents[1]; receipt = read(args.receipt)
    require(receipt['source_seal_sha256'] == args.source_seal and receipt['transport_source_seal_sha256'] == args.transport_seal and receipt['accepted_offserver'] == 0, 'Wrong release or adoption claim')
    select_delta(plan, receipt['accepted_new_ids'], [])
    pin = pins['dependencies']['archive_verifier']; verifier = load('offserver_original_archive_verifier', root / pin['local_relative'], pin['sha256'])
    proof = verifier.verify_archive(args.archive, receipt)
    require(proof['different_host_observed'] and proof['members_verified'] == receipt['members'], 'Offserver host/member count not proved')
    with tarfile.open(args.archive, 'r:gz') as tar:
        inv = json.load(tar.extractfile('backup_inventory.json'))
        require(inv['accepted_offserver'] == inv['models_repacked'] == inv['new_training'] == inv['new_test_inference'] == 0, 'Archive scope changed')
        require(inv['source_seal_sha256'] == args.source_seal and inv['all_transported_ids'] == receipt['all_transported_ids'], 'Archive/release/chain changed')
        prior = inv['previous_backup']
        if prior:
            require(receipt['previous_receipt'] == prior['receipt'], 'Archive/receipt previous-pointer drift')
            require(args.previous is not None and args.previous_sha256 == prior['receipt_sha256'] == receipt['previous_backup_receipt_sha256'] == sha(args.previous), 'Actual previous receipt SHA required')
            previous_receipt = read(args.previous)
            require(previous_receipt['source_seal_sha256'] == args.source_seal and previous_receipt['accepted_offserver'] == 0
                    and previous_receipt['all_transported_ids'] == prior['all_transported_ids'], 'Previous source/ID chain drift')
        else:
            require(args.previous is None and args.previous_sha256 is None and receipt['previous_backup_receipt_sha256'] is None, 'Unexpected previous receipt')
        require(select_delta(plan, inv['accepted_new_ids'], prior['all_transported_ids'] if prior else []) == receipt['all_transported_ids'], 'Cumulative transport IDs changed')
        volume = F_guard(args.out, sum(r['bytes'] for r in inv['members'].values()))
        args.out.mkdir(parents=True, exist_ok=False); extract = args.out / 'verified_extract'; extract.mkdir()
        for member in tar:
            verifier.safe_member(member.name); require(member.isfile(), 'Nonregular archive member')
            target = extract / member.name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as out, tar.extractfile(member) as inp:
                for block in iter(lambda: inp.read(4 * 1024 * 1024), b''): out.write(block)
    snapshot = extract / 'snapshot_inventory.json'
    require(sha(snapshot) == receipt['snapshot_sha256'] == inv['snapshot_sha256'], 'Snapshot SHA changed')
    pin = pins['dependencies']['saved_verifier']; original = load('original_saved_array_verifier', root / pin['local_relative'], pin['sha256'])
    body = saved_verifier_body(Path(original.__file__), pin['sha256'])
    namespace = dict(original.__dict__, EXPECTED_SNAPSHOT_SHA=sha(snapshot)); exec(compile(body, '<metadata inventory SHA projection only>', 'exec'), namespace)
    saved = namespace['verify'](extract / 'runs', snapshot, args.cache)
    save(args.out / 'OFFSERVER_TRANSPORT_VERIFICATION.json', dict(status='ARCHIVE_MEMBERS_AND_ORIGINAL_SAVED_ARRAYS_VERIFIED_NOT_ADOPTED',
        accepted_offserver=0, actual_transport_archive_sha256=sha(args.archive), receipt_sha256=args.receipt_sha256, snapshot_sha256=sha(snapshot),
        archive_member_verification=proof, original_saved_array_verification=saved, source_seal_sha256=args.source_seal,
        original_saved_verifier_sha256=pin['sha256'], metadata_projection='Inventory literal SHA and descriptive subset label only; metric/count/prediction rules unchanged.',
        fresh_F_volume=volume, original180_excluded=True, Full_inference=0, test_inference=0, root_adoption_pending=True,
        native_model_restore_chain_join='Root must bind each native checkpoint to its accepted native archive chain; models are not repacked here.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['export', 'verify'])
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--source-seal', required=True); parser.add_argument('--transport-seal', required=True)
    parser.add_argument('--runtime', type=Path); parser.add_argument('--ids', nargs='+'); parser.add_argument('--tag')
    parser.add_argument('--review', type=Path); parser.add_argument('--review-sha256'); parser.add_argument('--previous', type=Path); parser.add_argument('--previous-sha256')
    parser.add_argument('--archive', type=Path); parser.add_argument('--receipt', type=Path); parser.add_argument('--receipt-sha256'); parser.add_argument('--cache', type=Path); parser.add_argument('--out', type=Path)
    args = parser.parse_args(); own_seal(args.transport_seal); q, plan, pins = source(args.source, args.source_seal)
    if args.action == 'export': export(args, q, plan, pins)
    else: verify(args, q, plan, pins)
