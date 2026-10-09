"""One finite15-second read-only worker/proc/cgroup sample; no inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def fields(path):
    return {a: b.strip() for line in path.read_text().splitlines() if ':' in line for a, b in [line.split(':', 1)]}


def stat(path):
    a = path.read_text().rsplit(') ', 1)[1].split()
    return {'state': a[0], 'minor_faults': int(a[7]), 'major_faults': int(a[9]),
            'user_ticks': int(a[11]), 'system_ticks': int(a[12]), 'start_ticks': int(a[19])}


def optional(path):
    try:
        return path.read_text().strip()
    except (OSError, ProcessLookupError) as exc:
        return {'unavailable': type(exc).__name__}


def workers(phase):
    found = {}
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd = p.read_bytes().replace(b'\0', b' ').decode()
            if '/v3/replay_v3.py worker ' in cmd and 'phase' + str(phase) + '_useful' in cmd:
                found[int(p.parent.name)] = cmd
        except (OSError, ProcessLookupError):
            pass
    return found


def snapshot(pids):
    r = {'monotonic': time.monotonic(), 'unix': time.time(), 'workers': {}}
    cg = Path('/sys/fs/cgroup')
    r['cgroup_cpu_stat'] = {a: int(b) for line in (cg / 'cpu.stat').read_text().splitlines() for a, b in [line.split()]}
    r['cgroup_memory_current'] = optional(cg / 'memory.current')
    r['pressure'] = {k: optional(cg / (k + '.pressure')) for k in ('cpu', 'io', 'memory')}
    for pid in pids:
        p = Path('/proc') / str(pid)
        try:
            status = fields(p / 'status')
            record = {'stat': stat(p / 'stat'), 'rss': status.get('VmRSS'), 'peak_rss': status.get('VmHWM'),
                      'threads': status.get('Threads'), 'io': fields(p / 'io'), 'schedstat': optional(p / 'schedstat'),
                      'threads_detail': {}}
            for t in (p / 'task').glob('*'):
                try:
                    record['threads_detail'][t.name] = {**stat(t / 'stat'), 'wchan': optional(t / 'wchan'),
                                                       'schedstat': optional(t / 'schedstat'),
                                                       'affinity': sorted(os.sched_getaffinity(int(t.name)))}
                except (OSError, ProcessLookupError):
                    pass
            r['workers'][str(pid)] = record
        except (OSError, ProcessLookupError) as exc:
            r['workers'][str(pid)] = {'exited_or_unavailable': type(exc).__name__}
    return r


def main(phase):
    assert phase in (4, 5)
    expected = {4: 8, 5: 11}[phase]
    cpus = sorted(os.sched_getaffinity(0))[16:24]
    assert len(cpus) == 8
    os.sched_setaffinity(0, cpus)
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    deadline = time.monotonic() + 30
    while True:
        pids = workers(phase)
        if len(pids) == expected or time.monotonic() >= deadline:
            break
        time.sleep(1)
    assert len(pids) == expected, 'Finite diagnostic could not observe all declared workers; no retries'
    first = snapshot(pids)
    time.sleep(15)
    last = snapshot(pids)
    elapsed, hz = last['monotonic'] - first['monotonic'], os.sysconf('SC_CLK_TCK')
    delta = {}
    for pid in first['workers']:
        a, b = first['workers'][pid], last['workers'][pid]
        if 'stat' not in a or 'stat' not in b:
            delta[pid] = {'incomplete': True}
            continue
        assert a['stat']['start_ticks'] == b['stat']['start_ticks'], 'PID reused during sample'
        threads = {}
        for tid in set(a['threads_detail']) & set(b['threads_detail']):
            x, y = a['threads_detail'][tid], b['threads_detail'][tid]
            threads[tid] = {'cpu_seconds': ((y['user_ticks'] + y['system_ticks']) - (x['user_ticks'] + x['system_ticks'])) / hz,
                            'state_before_after': [x['state'], y['state']], 'wchan_before_after': [x['wchan'], y['wchan']]}
        cpu = ((b['stat']['user_ticks'] + b['stat']['system_ticks']) - (a['stat']['user_ticks'] + a['stat']['system_ticks'])) / hz
        delta[pid] = {'cpu_seconds': cpu, 'effective_cpu_cores': cpu / elapsed,
                      'minor_fault_delta': b['stat']['minor_faults'] - a['stat']['minor_faults'],
                      'major_fault_delta': b['stat']['major_faults'] - a['stat']['major_faults'],
                      'io_delta': {k: int(b['io'][k]) - int(a['io'][k]) for k in a['io']}, 'threads': threads}
    report = {'scope': 'ONE_FINITE_READ_ONLY_PROC_SAMPLE', 'phase': phase, 'sample_seconds': elapsed,
              'workers': pids, 'before': first, 'after': last, 'delta': delta,
              'cgroup_cpu_stat_delta': {k: last['cgroup_cpu_stat'][k] - first['cgroup_cpu_stat'][k] for k in first['cgroup_cpu_stat']},
              'probe_cpus': cpus, 'probe_nice': os.getpriority(os.PRIO_PROCESS, 0),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'claim_limit': 'Cgroup counters include every process. Wchan is a point observation, not Python stack attribution. No model recipe/thread change, restart or repeated inference.'}
    output = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v3') / ('phase' + str(phase) + '_execution_20261009/resource_diagnostic.json')
    assert not output.exists()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'phase': phase, 'sample_seconds': elapsed, 'workers': {pid: {k: v[k] for k in ('effective_cpu_cores', 'minor_fault_delta', 'major_fault_delta', 'io_delta')} for pid, v in delta.items() if not v.get('incomplete')}, 'cgroup_cpu_stat_delta': report['cgroup_cpu_stat_delta']}, indent=2))


if __name__ == '__main__':
    main(int(sys.argv[1]))
