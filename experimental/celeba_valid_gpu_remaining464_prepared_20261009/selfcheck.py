"""Use real identity inputs and synthetic files only; no Torch, subprocess or CNN."""
from pathlib import Path
from types import SimpleNamespace
import ast
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('remaining464_checked', HERE / 'remaining.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
m = q.read(HERE / 'manifest.json'); root = HERE.parent.parent
proposal = q.read(root / 'tmp/celeba_valid_recovery_prepared_20261009/manifest.json')
prior = q.read(HERE / 'prior/cumulative_425_accepted.json'); proof = q.read(HERE / 'prior/ROOT_OFFSERVER_VERIFICATION.json')
a = q.read(HERE / 'ROOT_REVIEW_TEMPLATE.json')
a.update(status='ROOT_APPROVED_GPU_VALID_RECOVERY_V1', execute_remaining464=True, execute_new465=True, queue_package_sha256='f' * 64, approved_ids=m['ids'], gpu_uuid=prior['new_GPU_uuid'])
q.validate(m, proposal, prior, proof, a, 'f' * 64)
rejected = []
def refuse(name, action):
    try: action()
    except (ValueError, KeyError): rejected.append(name)
    else: raise AssertionError('Unexpected pass: ' + name)
refuse('PREPARED_is_not_authorization', lambda: q.validate(m, proposal, prior, proof, q.read(HERE / 'ROOT_REVIEW_TEMPLATE.json'), 'f' * 64))
for name, key, value in [('tolerance_drift', 'native_tolerance', 1e-8), ('CPU_partial_import_permission', 'import_cpu_partial10', True), ('diagnostic_import_permission', 'import_gpu_diagnostic1', True), ('restart_failed872', 'restart_old872', True), ('two_GPU_workers', 'max_GPU_workers', 2), ('old425_binding_drift', 'accepted425_collector_sha256', '0' * 64)]:
    bad = copy.deepcopy(a); bad[key] = value
    refuse(name, lambda bad=bad: q.validate(m, proposal, prior, proof, bad, 'f' * 64))
for name, mutate in [('duplicate_ID', lambda x: x['ids'].__setitem__(1, x['ids'][0])), ('omit_ID', lambda x: x['ids'].pop()), ('changed_order', lambda x: x['ids'].reverse()), ('replay_first1', lambda x: x['ids'].__setitem__(0, prior['new_GPU_id'])), ('oversize_chunk', lambda x: x['chunks'][0]['ids'].append(x['chunks'][1]['ids'][0]))]:
    bad = copy.deepcopy(m); mutate(bad)
    refuse(name, lambda bad=bad: q.validate(bad, proposal, prior, proof, a, 'f' * 64))
node = ast.parse((HERE / 'remaining.py').read_text())
q.need(not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(x.name.startswith(('torch', 'numpy')) for x in n.names) for n in ast.walk(node)), 'Heavy scientific import added')
calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
q.need(sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'nice' for n in calls) == 1, 'Repeated nice increments introduced')
popen = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen')
q.need(any(k.arg == 'preexec_fn' and isinstance(k.value, ast.Lambda) and ast.unparse(k.value.body) == 'os.sched_setaffinity(0, {105})' for k in popen.keywords), 'CPU106 affinity inherited by child')
scenarios = []
with tempfile.TemporaryDirectory(prefix='flow_', dir=HERE) as temporary:
    folder = Path(temporary).resolve(); q.need(folder.is_relative_to(HERE), 'Fixture cleanup outside ownership')
    for fail in (None, 'run-chunk', 'accept', 'backup', 'mixed_strict', 'wrong_archive'):
        out = folder / (fail or 'success'); review = folder / ((fail or 'success') + '.review.json'); q.save(review, a)
        args = SimpleNamespace(review=review, review_sha256=q.sha(review), package_sha256='f' * 64)
        small = dict(m, chunks=m['chunks'][:2]); approval = dict(a, output_parent=str(out)); called = []
        def runner(command, log):
            step = command[2]; called.append(step)
            q.need(command[:2] == [sys.executable, str(q.RECOVERY / 'recovery.py')] and '--package-sha256' in command and command[command.index('--package-sha256') + 1] == q.IMPL_SHA, 'Child package/auth not original')
            if step == fail: return 1
            if step == 'run-chunk':
                stage = Path(command[command.index('--output') + 1]); stage.mkdir(); (stage / 'batch').mkdir()
            else: stage = Path(command[command.index('--batch') + 1]).parent if step == 'accept' else Path(command[command.index('--stage') + 1])
            chunk = small['chunks'][int(stage.name.split('_')[1])]
            if step == 'accept': q.save(stage / 'strict_acceptance.json', {'scope': 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1', 'status': 'SELECTED_VALID_REPLAY_ACCEPTED', 'accepted_ids': chunk['ids'][::-1] if fail == 'mixed_strict' else chunk['ids'], 'invalid': [], 'max_abs_native_metric_difference': 0})
            if step == 'backup':
                q.need(command[command.index('--strict-sha256') + 1] == q.sha(stage / 'strict_acceptance.json'), 'StrictSHA not passed to original backup')
                archive = stage / 'chunk_evidence.tar.gz'; archive.write_bytes(b'only synthetic flow fixture')
                q.save(stage / 'remote_archive_inventory.json', {'status': 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER', 'accepted_ids': chunk['ids'], 'chunk_index': chunk['index'], 'old_model_files_archived': 0, 'offserver_verified': False, 'archive': archive.name, 'sha256': '0' * 64 if fail == 'wrong_archive' else q.sha(archive)})
            return 0
        try:
            with contextlib.redirect_stdout(io.StringIO()): q.execute(small, approval, args, runner)
        except ValueError: q.need(fail is not None and (out / 'queue_failure.json').is_file(), 'Failure not preserved')
        else: q.need(fail is None, 'Failure unexpectedly continued')
        if fail:
            expected = {'run-chunk': ['run-chunk'], 'accept': ['run-chunk', 'accept'], 'backup': ['run-chunk', 'accept', 'backup'], 'mixed_strict': ['run-chunk', 'accept'], 'wrong_archive': ['run-chunk', 'accept', 'backup']}[fail]
            q.need(called == expected and not (out / 'chunk_001').exists() and not (out / 'queue_exit.json').exists(), 'Failure advanced/retried')
        else:
            exit_receipt = q.read(out / 'queue_exit.json'); q.need(called == ['run-chunk', 'accept', 'backup'] * 2 and exit_receipt['remote_closed_n'] == 22 and exit_receipt['accepted_new_n'] == 0 and not exit_receipt['cohort_registered'], 'Successful remote closure counted as acceptance')
        scenarios.append({'case': fail or 'two_chunks_closed_REMOTE_only', 'commands': called, 'new_accepted': 0})
q.need('torch' not in sys.modules, 'Torch imported')
print(json.dumps({'status': 'PASS_LOCAL_EXACT464_NO_CNN_NO_DISPATCH', 'exact_ordered_ids': 464, 'chunks': 43, 'chunk_sizes': {'11': 42, '2': 1}, 'prior_accepted': 425, 'prior_first1_excluded': True, 'pending_import11_excluded': True, 'refusal_checks': rejected, 'flow_scenarios': scenarios, 'CPU106_parent_CPU105_child_AST_verified': True, 'nice_set_once_no_child_prefix': True, 'new_CNN_inference': 0, 'cohort_changed': False, 'Linux_spawn_runtime_verified': False}, indent=2))
