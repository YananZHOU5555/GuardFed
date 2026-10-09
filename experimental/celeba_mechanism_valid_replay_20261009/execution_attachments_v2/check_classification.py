"""Actual /proc snapshot and bounded classifier rejection checks, no inference."""
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import execute_one_v2 as m

HERE = Path(__file__).resolve().parent
PY = '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'


def main():
    real = m.read(HERE / 'real_proc_snapshot.json')
    tracked = m.reservations(real['processes'])
    roles = [r['role'] for r in tracked]
    assert roles.count('hybrid_CPU_gate') == roles.count('gradient_CPU_gate') == 1
    assert roles.count('formal_GPU_worker') == 8
    assert roles.count('FLGMM_CPU_gate') == 0  # observed complete; no PID assumption
    excluded = [r for r in real['processes'] if m.classify(r['argv'], r['cwd']) is None]
    cases = []
    def classify(name, argv, cwd, expected):
        result = m.classify(argv, cwd)
        actual = result['role'] if result else None
        assert actual == expected, (name, actual, expected)
        cases.append({'name': name, 'role': actual, 'PASS': True})
    classify('Hybrid_relative_canary', [PY, '-B', 'canary.py'], str(m.HYBRID), 'hybrid_CPU_gate')
    classify('FLCPU_actual_canary_entry', [PY, '-u', str(m.FLCPU / 'canary.py')], '/root', 'FLGMM_CPU_gate')
    classify('gradient_actual_script', [PY, '-u', 'shared_cache_wrapper.py', 'run'], str(m.GRADIENT), 'gradient_CPU_gate')
    code = "import sys; import shared_cache_wrapper as w; w.bind_shared_paths(); import gate; sys.argv=['gate.py','run','--dispatch-receipt','dispatch_receipt.APPROVED.json']; gate.main()"
    classify('gradient_dash_c_run', [PY, '-c', code], str(m.GRADIENT), 'gradient_CPU_gate')
    classify('gradient_dash_c_main_default', [PY, '-c', 'import shared_cache_wrapper; import gate; gate.main()'], str(m.GRADIENT), 'gradient_CPU_gate')
    classify('gradient_dash_c_summary_helper', [PY, '-c', code.replace("'run'", "'summarize'")], str(m.GRADIENT), None)
    classify('gradient_dash_c_readonly_probe', [PY, '-c', 'import shared_cache_wrapper; print("gate.main()")'], str(m.GRADIENT), None)
    classify('gradient_dash_c_wrong_cwd', [PY, '-c', code], '/root', None)
    classify('log_tee_is_not_compute', ['python3', '/usr/local/bin/log-tee', str(m.GRADIENT / 'gate.log')], str(m.GRADIENT), None)
    classify('verify_canary_is_not_compute', [PY, str(m.FLCPU / 'verify_canary.py')], str(m.FLCPU), None)
    classify('gradient_script_summary_helper', [PY, 'gate.py', 'summarize'], str(m.GRADIENT), None)
    classify('gradient_script_approve_helper', [PY, 'shared_cache_wrapper.py', 'approve'], str(m.GRADIENT), None)
    classify('foreign_similar_basename', [PY, '/tmp/gpu_worker.py', '--repo', str(m.REPO), '--job', 'a', '--out', 'b'], '/tmp', None)
    gpu = [PY, '-u', str(m.FLGPU / 'gpu_worker.py'), '--repo', str(m.REPO), '--job', str(m.FLGPU / 'jobs/a.json'), '--out', str(m.FLGPU / 'runs/a')]
    classify('FLGPU_actual_worker_interface', gpu, str(m.FLGPU), 'FLGMM_GPU_canary')
    classify('FLGPU_coordinator_not_compute', [PY, str(m.FLGPU / 'run_gpu_gate.py'), '--repo', str(m.REPO)], str(m.FLGPU), None)
    classify('FLGPU_missing_job', gpu[:5], str(m.FLGPU), None)
    classify('replay_v4_worker', [PY, str(m.REPLAY / 'v4/replay_v4.py'), 'worker', '--id', 'model1'], '/root', 'baseline_valid_worker')
    classify('replay_v4_run_coordinator', [PY, str(m.REPLAY / 'v4/replay_v4.py'), 'run', '--id', 'model1'], '/root', None)
    rejected = []
    cpu_rows = [r for r in real['processes'] if m.classify(r['argv'], r['cwd']) and m.classify(r['argv'], r['cwd'])['compute_threads'] == 8]
    duplicate_task = copy.deepcopy(cpu_rows); duplicate_task.append({**duplicate_task[0], 'pid': 999999})
    overlap = copy.deepcopy(cpu_rows); overlap[1]['cpus'] = overlap[0]['cpus']
    own_slot = copy.deepcopy(cpu_rows); own_slot[0]['cpus'] = list(range(112, 120))
    partial_slot = copy.deepcopy(cpu_rows); partial_slot[0]['cpus'] = [8]
    gpu_rows = [{'pid': 990000 + i, 'argv': [a.replace('/a.', f'/{i}.').replace('/runs/a', f'/runs/{i}') for a in gpu], 'cwd': str(m.FLGPU), 'state': 'R', 'cpus': list(range(512))} for i in range(3)]
    for name, rows in [('duplicate_PID', cpu_rows + [copy.deepcopy(cpu_rows[0])]), ('duplicate_task_new_PID', duplicate_task),
                       ('CPU8_overlap', overlap), ('CPU112_119_occupied', own_slot), ('CPU_missing_affinity', partial_slot), ('FLGPU_more_than2', gpu_rows)]:
        try:
            m.reservations(rows)
        except ValueError as error:
            rejected.append({'name': name, 'error': str(error)})
        else:
            raise AssertionError('Must reject: ' + name)
    assert m.v1.check_budget([{'compute_threads': 114}], 122.87999) == 122
    try:
        m.v1.check_budget([{'compute_threads': 122}], 122.87999)
    except ValueError:
        rejected.append({'name': 'phase5_with_unreleased8_exceeds_quota', 'error': '122+8=130>122.87999'})
    else:
        raise AssertionError('Overbudget must refuse')
    report = {'status': 'PASS_CLASSIFIER_ONLY_NO_DISPATCH', 'snapshot_at_unix': real['at_unix'],
              'actual_snapshot_sha256': m.digest(HERE / 'real_proc_snapshot.json'), 'actual_classified': tracked,
              'actual_excluded_helpers': excluded, 'actual_CPUgate_count': sum(r['compute_threads'] == 8 for r in tracked),
              'actual_no_CPU_overlap': True, 'actual_existing_nominal': sum(r['compute_threads'] for r in tracked),
              'classification_cases': cases, 'rejected': rejected, 'FLGPU_snapshot_count': roles.count('FLGMM_GPU_canary'),
              'FLGPU_fixture_interface_checked_not_actual_launch': True,
              'v1_execute_code_object_reused_unchanged': m.execute.__code__ is m.v1.execute.__code__,
              'new_inference': False, 'new_training': False, 'supervisor_started': False, 'torch_imported': 'torch' in sys.modules}
    path = HERE / 'classification_checks.json'
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(len(cases), 'classifier cases;', len(rejected), 'rejections;', roles)


if __name__ == '__main__':
    main()
