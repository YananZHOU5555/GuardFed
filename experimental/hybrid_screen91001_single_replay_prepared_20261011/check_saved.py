"""Later execution only: original Linux whole check or Windows saved-output zero-fit audit."""
from __future__ import annotations
import argparse, ast, copy, datetime, hashlib, json, os, subprocess, sys, types, zipfile
from pathlib import Path
import candidate as c

SAVED_SHA = 'd512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def selected_originals():
    source = c.HERE / 'originals/saved_science.py'
    c.require(c.sha(source) == SAVED_SHA, 'Original whole checker changed')
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'check_saved')
    audit = (c.HERE / 'originals/saved_output_audit.py').read_text(encoding='utf-8')
    wanted = {'need', 'saved_fits_for_original_predict', 'saved_output_block'}
    helpers = [n for n in ast.parse(audit).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    c.require(len(helpers) == 3, 'Missing original saved-output helper')
    # Sole metadata adaptation: the saved fit must identify Hybrid, never FLGMM.
    for n in ast.walk(helpers[1]):
        if isinstance(n, ast.Constant) and n.value == 'FLGMM':
            n.value = c.METHOD
    import math
    ns = dict(copy=copy, ast=ast, math=math)
    exec(compile(ast.Module(body=helpers, type_ignores=[]), '<original saved-output helpers; method identity rebound>', 'exec'), ns)
    diff_source = (c.HERE / 'originals/receipt_diff.py').read_text(encoding='utf-8')
    diff = next(n for n in ast.parse(diff_source).body if isinstance(n, ast.FunctionDef) and n.name == 'diff')
    exec(compile(ast.Module(body=[diff], type_ignores=[]), '<original receipt-difference reporter>', 'exec'), ns)
    return node, ns


def f_volume():
    value = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command',
        'Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'], text=True))
    c.require(value['FileSystemLabel'] == 'Yanan 2TB' and value['HealthStatus'] == 'Healthy' and value['SizeRemaining'] > 1024**3, 'Healthy F capacity required')
    return value


def contained_f(path):
    p = Path(path).resolve()
    c.require(p.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()), 'Saved arrays/output must stay on F')
    return p


def gate_check(gate, a):
    c.require(gate['status'] == 'HYBRID_SCREEN91001_THREE_VIEW_PASS_NOT_ROOT_ADOPTED' and gate['package_sha256'] == a.package_sha256, 'Actual completed exact8 replay required')
    c.require([r['id'] for r in gate['receipts']] == c.IDS and gate['root_adopted_new'] == 0 and gate['test'] is False, 'Wrong saved scope')


def linux_fresh(a):
    c.require(a.cpu == 110 and os.sched_getaffinity(0) == {110} and os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Whole checker CPU110/nice10 only after fresh freedom')
    c.require(c.sha(a.post_replay_preflight) == a.post_replay_preflight_sha256, 'Actual post-replay preflight changed')
    pre = c.read(a.post_replay_preflight)
    c.require(pre['fl_evaluation_service_exited'] is True and pre['cpu110_all_thread_free'] is True and pre['hybrid_replay_service_exited'] is True and pre['previous_Hybrid8_service_exited'] is True and pre['previous_replay_workers_absent'] is True, 'FL/Hybrid EXITED and CPU110 free proof required')
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(pre['utc'])).total_seconds()
    c.require(0 <= age <= 300, 'Stale post-replay preflight')
    state = subprocess.run(['supervisorctl', 'status', 'guardfed_hybrid_screen91001_valid'], capture_output=True, text=True)
    c.require(state.returncode == 3 and state.stdout.split()[:2] == ['guardfed_hybrid_screen91001_valid', 'EXITED'], 'Hybrid replay service must be EXITED rc3')
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            c.require(not any(Path(t).name == 'candidate.py' and 'hybrid_screen91001_single' in t for t in args), 'Live replay worker')
        except (FileNotFoundError, ProcessLookupError):
            pass
        for task in (proc / 'task').glob('*'):
            try:
                aff = set(os.sched_getaffinity(int(task.name)))
                c.require(not (len(aff) <= 32 and 110 in aff), 'Restricted-thread CPU110 overlap')
            except ProcessLookupError:
                pass
    io = subprocess.run(['ionice', '-p', str(os.getpid())], capture_output=True, text=True, check=True)
    c.require('idle' in io.stdout, 'Idle I/O required')


def transport_check(a):
    c.require(c.sha(a.transport_proof) == a.transport_proof_sha256, 'Actual F transport proof changed')
    proof = c.read(a.transport_proof)
    c.require(proof['status'] == 'HYBRID_SCREEN91001_F_SAVED_MEMBERS_SHA_PASS' and proof['package_sha256'] == a.package_sha256, 'Unverified exact8 F transport')
    c.require(proof['linux_proof_sha256'] == a.linux_proof_sha256 and proof['gate_result_sha256'] == a.gate_result_sha256, 'Transport/whole/gate link changed')
    extract = contained_f(proof['verified_extract'])
    expected = {'bundle/GATE_RESULT.json', 'bundle/metadata_receipt.json', 'LINUX_SAVED_CHECK.json'} | {'bundle/' + rid + '/' + n for rid in c.IDS for n in ('receipt.json', 'validation_predictions.npz')}
    c.require(set(proof['members']) == expected, 'Exact5 saved members required; no models or unrelated arrays')
    for rel, pin in proof['members'].items():
        p = (extract / rel).resolve()
        c.require(p.is_relative_to(extract) and c.sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Saved member bytes changed')
    linux = c.read(extract / 'LINUX_SAVED_CHECK.json')
    c.require(c.sha(extract / 'LINUX_SAVED_CHECK.json') == a.linux_proof_sha256 and linux['status'] == 'HYBRID_SCREEN91001_LINUX_ORIGINAL_WHOLE_SAVED_PASS_NOT_ADOPTED', 'Original whole proof missing')
    c.require(linux['package_sha256'] == a.package_sha256 and linux['original_check_saved_sha256'] == SAVED_SHA and linux['gate_result_sha256'] == a.gate_result_sha256, 'Wrong whole source/gate')
    c.require([r['id'] for r in linux['records']] == c.IDS and linux['cached_root_refits'] == 1, 'Whole exact1 incomplete')
    c.require(all(r['root_receipt_exact'] and r['cached_root_fit_exact'] and r['saved_predictions_metrics_counts_exact'] for r in linux['records']), 'Whole check has not passed')
    c.require(a.gate_dir.resolve() == extract / 'bundle', 'Use exact verified saved bundle')
    return linux


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('linux-whole', 'windows-saved-output'), required=True)
    for key in ('package-sha256', 'gate-result-sha256'):
        p.add_argument('--' + key, required=True)
    for key in ('gate-dir', 'output', 'metadata-npz', 'transport-proof', 'post-replay-preflight'):
        p.add_argument('--' + key, type=Path)
    for key in ('transport-proof-sha256', 'linux-proof-sha256', 'post-replay-preflight-sha256'):
        p.add_argument('--' + key)
    p.add_argument('--cpu', type=int)
    p.add_argument('--allow-original-cached-root-refit', action='store_true')
    p.add_argument('--allow-saved-output-zero-fit', action='store_true')
    a = p.parse_args()
    c.require(__debug__ and a.output is not None and a.gate_dir is not None, 'Explicit unused output and saved gate required')
    started, failed = a.output.with_suffix('.started.json'), a.output.with_suffix('.failure.json')
    c.require(not any(x.exists() for x in (a.output, started, failed)), 'Preserve prior output/start/failure; no retry')
    c.require(not a.output.resolve().is_relative_to(c.HERE), 'Prepared source is immutable')
    c.package_check(a.package_sha256); m = c.read(c.HERE / 'MANIFEST.json'); c.validate_manifest(m)
    linux_mode = a.mode == 'linux-whole'
    if linux_mode:
        c.require(sys.platform == 'linux' and a.allow_original_cached_root_refit and not a.allow_saved_output_zero_fit, 'Explicit original Linux whole role only')
        c.require(a.post_replay_preflight is not None and a.post_replay_preflight_sha256 is not None, 'Actual post-replay resource proof required')
        c.require(a.gate_dir.is_absolute() and a.gate_dir.is_relative_to(c.BASE + '/outputs') and a.output.is_relative_to(c.BASE), 'Independent fixed Linux namespace')
        linux_fresh(a); linux = None
    else:
        c.require(sys.platform == 'win32' and a.allow_saved_output_zero_fit and not a.allow_original_cached_root_refit, 'Explicit Windows zero-fit role only')
        c.require(all(v is not None for v in (a.transport_proof, a.transport_proof_sha256, a.linux_proof_sha256, a.metadata_npz)), 'Actual F transport, Linux proof and original metadata required')
        f_volume(); contained_f(a.output); linux = transport_check(a); contained_f(a.metadata_npz)
        c.require(c.sha(a.metadata_npz) == '161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091', 'Original metadata bytes changed')
    c.require(c.sha(a.gate_dir / 'GATE_RESULT.json') == a.gate_result_sha256, 'Actual gate SHA changed')
    c.require(not any(a.gate_dir.rglob('FAILURE.json')), 'Preserved replay failure')
    gate = c.read(a.gate_dir / 'GATE_RESULT.json'); gate_check(gate, a)
    c.require(c.read(a.gate_dir / 'CANARY_PASS.json')['id'] == c.IDS[0], 'First native canary proof absent') if linux_mode else None
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    save(started, dict(mode=a.mode, package_sha256=a.package_sha256, gate_result_sha256=a.gate_result_sha256, root_adopted=False))
    rows = []
    try:
        import numpy as np, pandas as pd, torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        c.require(torch.cuda.device_count() == 0, 'CPU only')
        bridge, ev = c.bound_bridge(m, runtime=linux_mode, torch_module=torch, pandas_module=pd)
        rt = c.runtime_originals(ev); repo = Path(m['server_repo'])
        if linux_mode:
            sys.path.insert(0, str(repo)); core = c.load('_hybrid_saved_core', repo / 'scripts/reproduce_paper_tables.py')
            ids, y, s, _metadata = rt.metadata(repo)
            c.check_inputs(m)
        else:
            dependency = c.HERE.parents[1] / 'tmp/revision-publish-20260928'
            c.require(c.sha(dependency / 'src/data_loader.py') == '41723cc5dd57d7d7c4878bb93989426569279d608c0357cf0b6429c8c4a2a091', 'Original core dependency changed')
            sys.path.insert(0, str(dependency)); core = c.load('_hybrid_saved_core', c.HERE / 'originals/core.py')
            ns = dict(ev.rebuild_root.__globals__, zipfile=zipfile)
            prefix = next(n for n in ast.parse((c.HERE / 'originals/replay.py').read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'read_prefix')
            exec(compile(ast.Module(body=[prefix], type_ignores=[]), '<unchanged original label-prefix reader>', 'exec'), ns)
            with np.load(a.metadata_npz, allow_pickle=False) as z:
                ids, split = z['image_id'], z['split']
            c.require(np.array_equal(ids, np.arange(1, 202600)) and np.array_equal(np.flatnonzero(split == 1), np.arange(162770, 182637)), 'Original valid ordering changed')
            y, _ = ns['read_prefix'](a.metadata_npz, 'Smiling', 202599, 182637)
            s, _ = ns['read_prefix'](a.metadata_npz, 'Male', 202599, 182637)
        node, audit = selected_originals()
        block = audit['saved_output_block'](next(n for n in node.body if isinstance(n, ast.With)))
        for index, (row, receipt) in enumerate(zip(m['records'], gate['receipts'])):
            run = a.gate_dir / row['id']
            c.require(c.read(run / 'receipt.json') == receipt, 'Saved/gate receipt changed')
            actual = bridge.identity_record(row['id'], checkpoint_sha256=row['identity']['checkpoint']['sha256'])
            c.require(actual == row['identity'], 'Hybrid native identity changed')
            record = c.runtime_record(row)
            c.require(rt.canonical(record) == receipt['external_identity_record_sha256'] and receipt['checkpoint_sha256'] == record['checkpoint']['sha256'], 'Different replay identity/checkpoint')
            expected_header = dict(id=row['id'], method=c.METHOD, distribution='IID', attack='Benign', seed=row['seed'],
                config_canonical_sha256=record['config_canonical_sha256'], original_result_sha256=record['result']['sha256'],
                original_job_sha256=record['raw_job']['sha256'], original_training_torch=record['training_torch'],
                valid_n=19867, valid_image_ids_sha256=record['data_contract']['evaluation_image_ids_sha256'],
                status='NATIVE_VALID_REPLAY_PASS', scope='VALID_ONLY_IMPLEMENTATION_PREFLIGHT',
                test_labels_accessed=False, test_inference_performed=False, final_dispatch_created=False,
                optimizer_created=False, gradients_created=False)
            c.require(all(receipt[k] == v for k, v in expected_header.items()), 'Saved receipt identity/scope/header drift')
            c.require(receipt['runtime']['device'] == 'cpu' and receipt['runtime']['original_config_device'] == 'cuda' and all(v['dtype'] == '<f4' for v in receipt['weights_before'].values()), 'FP32 CPU/historical CUDA provenance changed')
            paths = {k: Path(row['runtime_artifacts'][j]['server_path']) if linux_mode else Path(row['runtime_artifacts'][j]['path']) for k, j in [('checkpoint', 'model'), ('result', 'result'), ('raw_job', 'job')]}
            def measure(_pins=None):
                result = {}
                for kind, path in paths.items():
                    h = c.sha(path); c.require(h == record[kind]['sha256'], 'Accepted artifact changed')
                    result[str(path)] = dict(sha256=h, bytes=path.stat().st_size)
                return result
            before = measure(); proof = dict(artifact_before=before, artifact_after=measure())
            def validate(_original, rec, _repo):
                c.require(rec == record and bridge.identity_record(row['id']) == row['identity'], 'Wrong Hybrid identity')
                return c.read(paths['result'])
            v2 = types.SimpleNamespace(np=np, require=c.require, rebuild_root=ev.rebuild_root, digest=c.sha, canonical=rt.canonical, VIEWS=ev.VIEWS, check_native=ev.check_native)
            if linux_mode:
                context = dict(v2=v2, mapping_paths=lambda rec, bindings: paths, full_hashes=measure,
                    mapped_functions=lambda rec, locations, original: (validate, None), KINDS=('checkpoint', 'result', 'raw_job'))
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<unchanged original whole check_saved>', 'exec'), context)
                comparison = context['check_saved'](record, run, receipt, proof, types.SimpleNamespace(repo=repo), None, core, None, ev, ids, y, s)
                result = dict(root_receipt_exact=True, cached_root_fit_exact=True, saved_predictions_metrics_counts_exact=True)
            else:
                previous = linux['records'][index]
                c.require(previous['receipt_sha256'] == c.sha(run / 'receipt.json') and previous['array_sha256'] == c.sha(run / 'validation_predictions.npz') and previous['checkpoint_sha256'] == record['checkpoint']['sha256'], 'Linux whole/member mismatch')
                cfg = core.ExperimentConfig(**record['config'])
                root_ids, root_y, root_s, root_receipt = ev.rebuild_root(core, cfg, record, ids, y, s)
                c.require(receipt['weights_before'] == receipt['weights_after'], 'Saved weight identity changed')
                fits = audit['saved_fits_for_original_predict'](receipt['fits'], ev)
                context = dict(v2=v2, evaluator=ev, path=run, r=receipt, root_ids=root_ids, root_y=root_y, root_s=root_s,
                    s=s, y=y, ids=ids, cfg=cfg, record=record, original_result=validate(None, record, repo), fits=fits)
                exec(compile(ast.Module(body=[block], type_ignores=[]), '<seven original saved-output statements, zero fit>', 'exec'), context)
                comparison = context['comparison']
                differences = audit['diff'](json.loads(json.dumps(root_receipt, allow_nan=False)), receipt['root_reconstruction'])
                result = dict(saved_fit_payload_hashes_exact=True, saved_predictions_metrics_counts_exact=True,
                    root_valid_ID_partition_exact=True, root_receipt_differences=differences, local_full_root_receipt_exact=not differences, fit_calls=0)
            c.require(measure() == before, 'Accepted artifacts changed during saved check')
            rows.append(dict(id=row['id'], checkpoint_sha256=receipt['checkpoint_sha256'], receipt_sha256=c.sha(run / 'receipt.json'),
                array_sha256=c.sha(run / 'validation_predictions.npz'), native_comparison=comparison, **result))
        c.package_check(a.package_sha256)
        if linux_mode:
            c.check_inputs(m)
        else:
            transport_check(a)
        report = dict(status='HYBRID_SCREEN91001_LINUX_ORIGINAL_WHOLE_SAVED_PASS_NOT_ADOPTED' if linux_mode else 'HYBRID_SCREEN91001_WINDOWS_SAVED_OUTPUT_ZERO_FIT_PASS_NOT_ADOPTED',
            records=rows, exact_ids=c.IDS, original_check_saved_sha256=SAVED_SHA, package_sha256=a.package_sha256,
            gate_result_sha256=a.gate_result_sha256, cached_root_refits=1 if linux_mode else 0, fit_calls=1 if linux_mode else 0,
            new_CNN=0, new_training=0, test=False, native_tolerance=1e-12, root_adopted=False,
            Windows_whole_check_pass=None, Windows_exact_recalibration_claimed=False,
            prior_FL_Windows_whole_and_refit_failures_not_relabelled=True,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(a.output, report); print(json.dumps(dict(status=report['status'], records=1, fit_calls=report['fit_calls'], report_sha256=c.sha(a.output))))
    except BaseException:
        import traceback
        save(failed, dict(status='FAILED_PRESERVED_NO_RETRY', mode=a.mode, completed=len(rows), traceback=traceback.format_exc(), root_adopted=False))
        raise


if __name__ == '__main__':
    main()
