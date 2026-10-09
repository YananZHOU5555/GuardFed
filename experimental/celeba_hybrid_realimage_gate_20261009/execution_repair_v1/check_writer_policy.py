"""Local component fault tests only; no Torch, images, inference or training."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

import repair_execute as execution
from writer_policy import sanitize_result, check_sidecar

HERE = Path(__file__).resolve().parent
checks = []


def rejected(name, action):
    try:
        action()
    except (ValueError, AssertionError, KeyError, TypeError):
        checks.append(dict(name=name, rejected=True))
    else:
        raise AssertionError('Expected refusal: ' + name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


job_path = HERE.parent / 'jobs/non-IID_S-DFA_hybrid_seed91001_cpu_gate3.json'
job = json.loads(job_path.read_text(encoding='utf-8-sig')); expected_sha = sha(job_path)
audits = [dict(client_id=i, is_malicious=i < 4, samples=100 + i, attack_types=['fflip', 'foe'] if i < 4 else [],
    fflip_mode='all_unprivileged', fflip_overwrite_ratio=1.0, fflip_requires_full_flip=False,
    label_changed_count=0, fflip_label_corr_after=float('nan') if i < 4 else .2) for i in range(20)]
value = dict(config=copy.deepcopy(job['config']), method=job['method'], attack=job['attack'], attack_audit=audits,
             metrics=dict(accuracy=.5, aeod=.1, aspd=.2), finite_nested=[{'a': [1, None, False, 'text', -.2]}])
before = copy.deepcopy(value)
clean, rows = sanitize_result(value, job, expected_sha, expected_sha)
assert len(rows) == 4 and all(clean['attack_audit'][i]['fflip_label_corr_after'] is None for i in range(4))
assert clean['metrics'] == before['metrics'] and clean['finite_nested'] == before['finite_nested']
assert all(math.isnan(value['attack_audit'][i]['fflip_label_corr_after']) for i in range(4))
assert clean['config'] == value['config'] and all(clean['attack_audit'][i] == value['attack_audit'][i] for i in range(4, 20))
checks.append(dict(name='exact_four_undefined_paths_null_sidecar_finite_structure_and_input_preserved', pass_=True))
sidecar = dict(status='EXPLICIT_UNDEFINED_DIAGNOSTIC_ONLY', job_sha256=expected_sha,
    original_gate_sha256=execution.GATE_SHA, original_core_sha256=execution.CORE_SHA, undefined_values=rows)
check_sidecar(clean, sidecar, job, expected_sha, execution.GATE_SHA, execution.CORE_SHA)
checks.append(dict(name='sidecar_nan_bits_reason_and_frozen_cause_roundtrip', pass_=True))
for name, path, bad in [
    ('main_metric_nan', ('metrics', 'accuracy'), float('nan')),
    ('main_metric_inf', ('metrics', 'aeod'), float('inf')),
    ('unknown_diagnostic_nan', ('extra_diagnostic',), float('nan')),
    ('weight_nan', ('weights',), [float('nan')]),
    ('control_inf', ('control',), float('-inf')),
    ('other_client_same_field_nan', ('attack_audit', 4, 'fflip_label_corr_after'), float('nan')),
    ('allowed_path_infinity', ('attack_audit', 0, 'fflip_label_corr_after'), float('inf')),
    ('allowed_path_preexisting_null', ('attack_audit', 0, 'fflip_label_corr_after'), None),
    ('allowed_path_finite_zero', ('attack_audit', 0, 'fflip_label_corr_after'), 0.),
    ('wrong_zero_variance_cause', ('attack_audit', 0, 'fflip_mode'), 'invert')]:
    fault = copy.deepcopy(value); target = fault
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = bad
    rejected(name, lambda fault=fault: sanitize_result(fault, job, expected_sha, expected_sha))
rejected('changed_job_hash', lambda: sanitize_result(value, job, '0' * 64, expected_sha))
wrong_job = copy.deepcopy(job); wrong_job['config']['rounds'] = 70
rejected('formal_rounds_unauthorized', lambda: sanitize_result(value, wrong_job, expected_sha, expected_sha))
for name, change in [
    ('sidecar_unknown_path', lambda x: x['undefined_values'][0]['path'].__setitem__(0, 'metrics')),
    ('sidecar_finite_bits', lambda x: x['undefined_values'][0].__setitem__('original_ieee754_binary64_big_endian_hex', '0000000000000000')),
    ('sidecar_zero_semantics', lambda x: x['undefined_values'][0].__setitem__('serialized_value', 0)),
    ('sidecar_wrong_reason', lambda x: x['undefined_values'][0].__setitem__('reason', 'correlation is zero')),
    ('sidecar_duplicate', lambda x: x['undefined_values'].__setitem__(1, copy.deepcopy(x['undefined_values'][0])))]:
    faulty = copy.deepcopy(sidecar); change(faulty)
    rejected(name, lambda faulty=faulty: check_sidecar(clean, faulty, job, expected_sha, execution.GATE_SHA, execution.CORE_SHA))
gate = execution.load_gate(); original = gate.read(HERE.parent / 'scope.json')
scope = dict(new_jobs=original['jobs'][2:])
runner, checker, comparison = execution.functions(gate, scope, original)
assert runner.__code__ is gate.run_one.__code__ and comparison.__code__ is gate.compare.__code__
assert checker.__closure__ and runner.__globals__['checked'] is checker
checks.append(dict(name='scientific_training_and_pair_comparison_original_codeobjects_reused', pass_=True))
rejected('absent_external_approval_before_runtime_creation', lambda: execution.validate_approval(gate, scope, None))
with tempfile.TemporaryDirectory(prefix='hybrid_writer_component_') as directory:
    old_runtime = execution.RUNTIME; execution.RUNTIME = Path(directory)
    try:
        entry = scope['new_jobs'][0]
        job_target = execution.RUNTIME / entry['job']; job_target.parent.mkdir(parents=True)
        job_target.write_bytes(job_path.read_bytes())
        runner, _, _ = execution.functions(gate, scope, original)
        writer = runner.__globals__['write']
        # Runtime scope is an identity-only input here, never a dispatch approval.
        (HERE / 'REPAIR_SCOPE.json').is_file() or (_ for _ in ()).throw(AssertionError('Prepare scope first'))
        out = execution.RUNTIME / entry['output']
        writer(out / 'result.json', value)
        serialized = json.loads((out / 'result.json').read_text())
        assert serialized == clean and (out / 'undefined_diagnostics.json').is_file()
        rejected('unknown_non_result_writer_route_still_strict', lambda: writer(out / 'unknown.json', {'x': float('nan')}))
        wrong_metric = copy.deepcopy(value); wrong_metric['metrics']['accuracy'] = float('nan')
        rejected('actual_result_writer_refuses_main_metric', lambda: writer(out / 'result.json', wrong_metric))
    finally:
        execution.RUNTIME = old_runtime
checks.append(dict(name='actual_writer_exact_route_and_fault_boundaries', pass_=True))
report = dict(status='PASS_COMPONENT_ONLY_NOT_REAL_TRAINING', checks=checks, check_count=len(checks),
    images_read=0, inference_runs=0, training_runs=0, formal_scientific_records=0,
    original_scientific_gate_sha256=execution.GATE_SHA)
target = HERE / 'writer_checks.json'
assert not target.exists(), 'Never replace accepted component evidence'
target.write_bytes((json.dumps(report, indent=2, allow_nan=False) + '\n').encode())
print(json.dumps(dict(status=report['status'], checks=len(checks))))
