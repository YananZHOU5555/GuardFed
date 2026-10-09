"""Real440 identity plus bounded synthetic flow; no subprocess, Torch, CNN or server."""
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
spec = importlib.util.spec_from_file_location('remaining440_checked', HERE / 'remaining.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
m = q.read(HERE / 'manifest.json'); root = HERE.parent.parent
inventory = q.read(root / 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
proposal = q.read(root / 'tmp/celeba_valid_recovery_prepared_20261009/manifest.json')
prior, parent, proof, guard = [q.read(HERE / 'prior' / n) for n in ('cumulative_460_accepted.json', 'parent464_manifest.json', 'ROOT_OFFSERVER_IMPORT_VERIFICATION.json', 'GPU_RESOURCE_GUARD_V2_ROOT_REVIEW.json')]
a = q.read(HERE / 'ROOT_REVIEW_TEMPLATE.json')
a.update(status='ROOT_APPROVED_GPU_VALID_RECOVERY_V1', execute_remaining440=True, execute_new465=True, queue_package_sha256='f' * 64, approved_ids=m['ids'], gpu_uuid=proof['results'][0]['GPU_uuid'])
def validate(mm=m, aa=a, pp=prior): return q.validate(mm, inventory, proposal, pp, parent, proof, guard, aa, 'f' * 64)
validate()
rejected = []
def refuse(name, action):
    try: action()
    except (ValueError, KeyError): rejected.append(name)
    else: raise AssertionError('Unexpected pass: ' + name)
refuse('PREPARED_is_not_authorization', lambda: validate(aa=q.read(HERE / 'ROOT_REVIEW_TEMPLATE.json')))
for name, key, value in [('tolerance_drift', 'native_tolerance', 1e-8), ('CPU_partial_import', 'import_cpu_partial10', True), ('diagnostic_import', 'import_gpu_diagnostic1', True), ('restart_old464', 'restart_old464', True), ('two_GPU_workers', 'max_GPU_workers', 2), ('old460_binding_drift', 'accepted460_collector_sha256', '0' * 64), ('V1_runtime', 'implementation_package_sha256', '6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56'), ('V1_scope', 'recovery_scope', 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1'), ('parent_source_drift', 'parent464_package_sha256', '0' * 64)]:
    bad = copy.deepcopy(a); bad[key] = value
    refuse(name, lambda bad=bad: validate(aa=bad))
for name, mutate in [('duplicate_ID', lambda x: x['ids'].__setitem__(1, x['ids'][0])), ('omit_ID', lambda x: x['ids'].pop()), ('changed_order', lambda x: x['ids'].reverse()), ('replay_accepted_partial2', lambda x: x['ids'].__setitem__(0, prior['added_ids'][0])), ('oversize_chunk', lambda x: x['chunks'][0]['ids'].append(x['chunks'][1]['ids'][0]))]:
    bad = copy.deepcopy(m); mutate(bad)
    refuse(name, lambda bad=bad: validate(mm=bad))
bad_prior = copy.deepcopy(prior); bad_prior['accepted_ids'][1] = bad_prior['accepted_ids'][0]
refuse('duplicate_prior460', lambda: validate(pp=bad_prior))
node = ast.parse((HERE / 'remaining.py').read_text())
q.need(not any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(x.name.startswith(('torch', 'numpy')) for x in n.names) for n in ast.walk(node)), 'Heavy scientific import added')
calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
q.need(sum(isinstance(n.func, ast.Attribute) and n.func.attr == 'nice' for n in calls) == 1, 'Repeated nice increments introduced')
popen = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen')
q.need(any(k.arg == 'preexec_fn' and isinstance(k.value, ast.Lambda) and ast.unparse(k.value.body) == 'os.sched_setaffinity(0, {105})' for k in popen.keywords), 'CPU106 inherited by child')
scenarios = []
with tempfile.TemporaryDirectory(prefix='flow_', dir=HERE) as temporary:
    folder = Path(temporary).resolve(); q.need(folder.is_relative_to(HERE), 'Fixture cleanup outside ownership')
    for fail in (None, 'run-chunk', 'accept', 'backup', 'mixed_strict', 'wrong_archive', 'old_scope', 'native_mismatch'):
        out = folder / (fail or 'success'); review = folder / ((fail or 'success') + '.review.json'); q.save(review, a)
        args = SimpleNamespace(review=review, review_sha256=q.sha(review), package_sha256='f' * 64)
        small = dict(m, chunks=m['chunks'][:2]); approval = dict(a, output_parent=str(out)); called = []
        def runner(command, log):
            step = command[2]; called.append(step)
            q.need(command[:2] == [sys.executable, str(q.RECOVERY / 'recovery.py')] and command[command.index('--package-sha256') + 1] == q.IMPL_SHA, 'Child package/source not V2')
            if step == fail: return 1
            if step == 'run-chunk':
                stage = Path(command[command.index('--output') + 1]); stage.mkdir(); (stage / 'batch').mkdir()
            else: stage = Path(command[command.index('--batch') + 1]).parent if step == 'accept' else Path(command[command.index('--stage') + 1])
            chunk = small['chunks'][int(stage.name.split('_')[1])]
            if step == 'accept': q.save(stage / 'strict_acceptance.json', {'scope': 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1' if fail == 'old_scope' else q.SCOPE, 'status': 'SELECTED_VALID_REPLAY_ACCEPTED', 'accepted_ids': chunk['ids'][::-1] if fail == 'mixed_strict' else chunk['ids'], 'invalid': [], 'max_abs_native_metric_difference': 1.01e-12 if fail == 'native_mismatch' else 0})
            if step == 'backup':
                q.need(command[command.index('--strict-sha256') + 1] == q.sha(stage / 'strict_acceptance.json'), 'StrictSHA not passed to original backup')
                binding = q.read(stage / 'queue_binding.json'); q.need(binding['prior460_sha256'] == m['accepted460_collector_sha256'] and binding['implementation_package_sha256'] == q.IMPL_SHA and binding['recovery_scope'] == q.SCOPE, 'Archive source/prior binding lost')
                archive = stage / 'chunk_evidence.tar.gz'; archive.write_bytes(b'only synthetic flow fixture')
                q.save(stage / 'remote_archive_inventory.json', {'status': 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER', 'accepted_ids': chunk['ids'], 'chunk_index': chunk['index'], 'old_model_files_archived': 0, 'offserver_verified': False, 'archive': archive.name, 'sha256': '0' * 64 if fail == 'wrong_archive' else q.sha(archive)})
            return 0
        try:
            with contextlib.redirect_stdout(io.StringIO()): q.execute(small, approval, args, runner)
        except ValueError: q.need(fail is not None and (out / 'queue_failure.json').is_file(), 'Failure not preserved')
        else: q.need(fail is None, 'Failure unexpectedly continued')
        if fail:
            expected = ['run-chunk'] if fail == 'run-chunk' else ['run-chunk', 'accept'] if fail in ('accept', 'mixed_strict', 'old_scope', 'native_mismatch') else ['run-chunk', 'accept', 'backup']
            q.need(called == expected and not (out / 'chunk_001').exists() and not (out / 'queue_exit.json').exists(), 'Failure advanced/retried')
        else:
            receipt0 = out / 'chunk_000.REMOTE_PENDING_OFFSERVER.json'; receipt1 = q.read(out / 'chunk_001.REMOTE_PENDING_OFFSERVER.json')
            q.need(receipt1['previous_receipt_sha256'] == q.sha(receipt0) and receipt1['implementation_package_sha256'] == q.IMPL_SHA and receipt1['recovery_scope'] == q.SCOPE, 'V2 receipt chain changed')
            exit_receipt = q.read(out / 'queue_exit.json'); q.need(called == ['run-chunk', 'accept', 'backup'] * 2 and exit_receipt['remote_closed_n'] == 22 and exit_receipt['accepted_new_n'] == 0 and not exit_receipt['cohort_registered'], 'Remote closure counted as acceptance')
            refuse('existing_output_no_resume', lambda: q.execute(small, approval, args, runner))
        scenarios.append({'case': fail or 'two_chunks_closed_REMOTE_only', 'commands': called, 'new_accepted': 0})
q.need('torch' not in sys.modules, 'Torch imported')
print(json.dumps({'status': 'PASS_LOCAL_EXACT440_NO_CNN_NO_DISPATCH', 'exact_inventory900_minus460_in_parent464_order': True, 'exact_ordered_ids': 440, 'chunks': 40, 'chunk_size': 11, 'prior_accepted': 460, 'accepted_partial2_excluded': True, 'refusal_checks': rejected, 'flow_scenarios': scenarios, 'V2_source_scope_and_remote_receipt_chain_checked': True, 'CPU106_parent_CPU105_child_AST_verified': True, 'nice_set_once_no_child_prefix': True, 'new_CNN_inference': 0, 'new_server_access': 0, 'cohort_changed': False, 'Linux_spawn_runtime_verified': False}, indent=2))
