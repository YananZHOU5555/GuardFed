"""Summarize verified actual v4 phase receipts; never infer or accept a model."""
import collections
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
phase = int(sys.argv[1])
assert phase in (4, 5)
stage = BASE / 'v4' / ('phase' + str(phase) + '_attempt1_execution_20261009')

def read(n):
    return json.loads((stage / n).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

proof, diag = read('offserver_verification.json'), read('resource_diagnostic.json')
assert proof['status'] == 'PASS' and proof['accepted_n'] == {4: 8, 5: 11}[phase]
assert diag['status'] == 'FINITE_BOUND_ALLOCATED_WORKERS_SAMPLE_COMPLETE'
profiles, wchans = [], collections.Counter()
for pid, delta in diag['delta'].items():
    assert not delta.get('incomplete')
    a, b = diag['before']['workers'][pid], diag['after']['workers'][pid]
    for record in (a, b):
        for t in record['threads_detail'].values():
            wchans[str(t['wchan'])] += 1
    profiles.append({'id': delta['id'], 'effective_cpu_cores': delta['effective_cpu_cores'], 'cpu_seconds': delta['cpu_seconds'], 'minor_fault_delta': delta['minor_fault_delta'], 'major_fault_delta': delta['major_fault_delta'], 'rss_kib_before_after': [int(x['rss'].split()[0]) for x in (a, b)], 'os_threads_before_after': [int(x['threads']) for x in (a, b)], 'io_delta': delta['io_delta']})
total = sum(p['effective_cpu_cores'] for p in profiles)
report = {'phase': phase, 'accepted_n': proof['accepted_n'], 'workers': proof['workers'], 'batch_wall_seconds': proof['batch_wall_seconds'], 'batch_accepted_models_per_second': proof['batch_accepted_models_per_second'], 'native_max_abs_difference': max(p['native_max_abs_difference'] for p in proof['models']), 'independent_metric_checks': proof['independent_metric_checks'], 'independent_confusion_count_checks': proof['independent_confusion_count_checks'], 'models': proof['models'], 'finite_profile': {'sample_seconds': diag['sample_seconds'], 'total_effective_worker_cores': total, 'per_worker': profiles, 'thread_wchan_endpoint_observations': dict(wchans), 'cgroup_cpu_stat_delta': diag['cgroup_cpu_stat_delta'], 'scope': 'Eight-CPU slot-bound, allocated inference tensors, finite15 second sample; process CPU includes runtime overhead', 'interpretation': 'Active computation and futex waits coexist with many minor faults; no measured disk I/O or quota throttling in this window. This does not isolate the specific operator, allocator or memory-bandwidth cause.'}, 'shared_input_identities_unchanged': proof['shared_input_identities_unchanged'], 'model_result_job_hashes_unchanged': proof['model_result_job_hashes_unchanged'], 'training_round_growth': proof['training_round_growth'], 'training_completed_during_after': proof['training_completed_during_after'], 'training_failed_during_after': proof['training_failed_during_after'], 'gpu_during_after': proof['gpu_during_after'], 'verification_inputs_sha256': {n: sha(stage / n) for n in ('offserver_verification.json', 'strict_acceptance.json', 'remote_archive_inventory.json', 'resource_diagnostic.json')}, 'test_inference_performed': False, 'all900_native_valid_replayed': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN', 'claim_limit': 'Distinct representative useful checkpoints across concurrency levels, not same-model controlled speedup or an optimum for all900. Observed GPU advancement does not quantify zero slowdown.'}
target = stage / 'measurement.json'
assert not target.exists()
target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'phase': phase, 'accepted_n': report['accepted_n'], 'native_max_abs_difference': report['native_max_abs_difference'], 'batch_wall_seconds': report['batch_wall_seconds'], 'batch_accepted_models_per_second': report['batch_accepted_models_per_second'], 'profile_total_cores': total, 'measurement_sha256': sha(target)}, indent=2))
