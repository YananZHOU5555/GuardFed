"""One bounded read-only /proc sample; no scientific imports or inference."""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
OUTPUT = BASE / 'v4/remaining872_execution_20261009/bounded_queue_cpu_diagnostic_10s.json'
HZ = os.sysconf('SC_CLK_TCK')
os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[16]])
os.nice(10)
subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
assert not OUTPUT.exists()


def role(argv):
    text = ' '.join(argv).lower()
    if len(argv) > 2 and argv[1] == str(BASE / 'v4/replay_v4.py'):
        return 'cnn_replay_worker' if argv[2] == 'worker' else 'replay_manager_or_acceptance'
    if 'bounded_remaining.py' in text:
        return 'outer_bookkeeping'
    if 'celeba_mechanism' in text and 'worker' in text:
        return 'formal_mechanism_training_command_match'
    for token in ('flgmm', 'hybrid', 'gradient'):
        if token in text:
            return token + '_command_match'
    return 'other'


def snap():
    r = {'at': time.monotonic(), 'processes': {}, 'cpu_stat': {}}
    for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines():
        key, value = line.split()
        r['cpu_stat'][key] = int(value)
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            text = (directory / 'stat').read_text()
            stat = text[text.rfind(')') + 2:].split()
            argv = [x.decode(errors='replace') for x in (directory / 'cmdline').read_bytes().split(b'\0') if x]
            item = {'pid': int(directory.name), 'start': int(stat[19]), 'cpu_ticks': int(stat[11]) + int(stat[12]), 'role': role(argv), 'rss_kib': int(stat[21]) * os.sysconf('SC_PAGE_SIZE') // 1024, 'minor_faults': int(stat[7]), 'major_faults': int(stat[9]), 'threads': int(stat[17])}
            if item['role'] == 'cnn_replay_worker':
                item.update(id=argv[argv.index('--id') + 1], slot=int(argv[argv.index('--slot') + 1]), nice=os.getpriority(os.PRIO_PROCESS, item['pid']), io={}, threads_detail={})
                for line in (directory / 'io').read_text().splitlines():
                    key, value = line.split(':')
                    item['io'][key] = int(value)
                for task in (directory / 'task').glob('*'):
                    item['threads_detail'][task.name] = {'affinity': sorted(os.sched_getaffinity(int(task.name))), 'wchan': (task / 'wchan').read_text().strip(), 'state': next(x for x in (task / 'status').read_text().splitlines() if x.startswith('State:'))}
            r['processes'][directory.name] = item
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return r


started = time.monotonic()
while True:
    before = snap()
    workers = [p for p in before['processes'].values() if p['role'] == 'cnn_replay_worker']
    ready = len(workers) == 11 and all(p['rss_kib'] >= 2 * 1024 * 1024 and p['nice'] == 10 and all(len(t['affinity']) == 8 for t in p['threads_detail'].values()) for p in workers)
    if ready or time.monotonic() - started >= 90:
        break
    time.sleep(2)
assert ready, 'No allocated eleven-worker steady window within bounded90s; preserve tasks without retry'
time.sleep(10)
after = snap()
seconds = after['at'] - before['at']
rows, totals, lost = [], collections.defaultdict(float), []
for pid, a in before['processes'].items():
    b = after['processes'].get(pid)
    if b is None or b['start'] != a['start']:
        if a['role'] == 'cnn_replay_worker':
            lost.append(pid)
        continue
    cores = (b['cpu_ticks'] - a['cpu_ticks']) / HZ / seconds
    totals[a['role']] += cores
    if a['role'] == 'cnn_replay_worker':
        wchan = collections.Counter(t['wchan'] for p in (a, b) for t in p['threads_detail'].values())
        rows.append({'id': a['id'], 'pid': int(pid), 'slot': a['slot'], 'effective_cpu_cores': cores, 'minor_fault_delta': b['minor_faults'] - a['minor_faults'], 'major_fault_delta': b['major_faults'] - a['major_faults'], 'io_delta': {k: b['io'][k] - a['io'][k] for k in a['io']}, 'rss_kib_before_after': [a['rss_kib'], b['rss_kib']], 'thread_n_before_after': [a['threads'], b['threads']], 'thread_affinities_before_after': [sorted({tuple(t['affinity']) for t in p['threads_detail'].values()}) for p in (a, b)], 'thread_wchan_endpoint_observations': dict(wchan)})
cpu_delta = {k: after['cpu_stat'][k] - before['cpu_stat'][k] for k in before['cpu_stat']}
global_cores = cpu_delta['usage_usec'] / 1e6 / seconds
quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
closed = []
for path in sorted((BASE / 'v4/remaining872_attempt1').glob('chunk_*/strict_acceptance.json')):
    a = json.loads(path.read_text())
    if a['status'] == 'SELECTED_VALID_REPLAY_ACCEPTED':
        closed.append({'chunk': path.parent.name, 'accepted_n': a['accepted_n'], 'batch_wall_seconds': a['wall_seconds'], 'batch_models_per_second': a['models_per_second']})
report = {'status': 'STEADY10S_PROC_DIAGNOSTIC_COMPLETE' if not lost and len(rows) == 11 else 'WINDOW_INCOMPLETE_PRESERVE_NO_RETRY', 'wait_until_allocated_seconds': before['at'] - started, 'sample_seconds': seconds, 'global_effective_cpu_cores': global_cores, 'quota_cores': None if quota == 'max' else int(quota) / int(period), 'cpu_max': quota + ' ' + period, 'cnn_effective_cpu_cores': totals['cnn_replay_worker'], 'noncnn_cgroup_cpu_cores_by_subtraction': global_cores - totals['cnn_replay_worker'], 'stable_visible_process_cpu_by_command_role': dict(totals), 'stable_process_accounting_residual_cores': global_cores - sum(totals.values()), 'lost_worker_pids': lost, 'workers': rows, 'cpu_stat_delta': cpu_delta, 'closed_actual_batches': closed, 'gpu': subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scope': 'Read-only10s /proc sample of allocated actual workers; no operator/bandwidth profiler, new inference, thread/concurrency change, or artificial load. Endpoint wait states are observations, not wait-duration attribution. Stable-process accounting excludes exited/new PIDs and has snapshot skew.'}
with OUTPUT.open('x') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print(json.dumps({'status': report['status'], 'cnn_cores': report['cnn_effective_cpu_cores'], 'other_cores': report['noncnn_cgroup_cpu_cores_by_subtraction'], 'global_cores': global_cores, 'quota': report['quota_cores'], 'workers': len(rows), 'cpu_throttled_usec': cpu_delta.get('throttled_usec', 0), 'worker_read_bytes': sum(r['io_delta']['read_bytes'] for r in rows), 'worker_major_faults': sum(r['major_fault_delta'] for r in rows), 'receipt_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}))
