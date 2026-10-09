"""One finite15s /proc sample after v4 worker slot binding and tensor allocation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009')
ORIGINAL = BASE / 'v3/throughput_execution_20261009/sample_worker_resources.py'
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == '7cbbc5b69827c368226636fb0e1dca5b854b4fd3f9d7fe7e5ece9136be6ea198'
spec = importlib.util.spec_from_file_location('original_readonly_sampler', ORIGINAL)
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)

def locate(batch, selected):
    found = {}
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            a = [v.decode() for v in p.read_bytes().split(b'\0') if v]
            if len(a) < 3 or a[1:3] != [str(BASE / 'v4/replay_v4.py'), 'worker']:
                continue
            i = a[a.index('--id') + 1]
            output = Path(a[a.index('--output') + 1])
            if i in selected and output == batch / 'runs' / i:
                found[int(p.parent.name)] = {'id': i, 'slot': int(a[a.index('--slot') + 1]), 'argv': a}
        except (OSError, ProcessLookupError, ValueError):
            pass
    return found

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.batch.is_relative_to(BASE / 'v4') and args.output.is_relative_to(BASE / 'v4') and not args.output.exists()
    cpus = sorted(os.sched_getaffinity(0))[16:24]
    assert len(cpus) == 8
    os.sched_setaffinity(0, cpus)
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
    started = time.monotonic()
    deadline = started + 120
    first, observed, contract = None, {}, None
    while time.monotonic() < deadline:
        receipt = args.batch / 'batch_inputs.json'
        if not receipt.exists():
            time.sleep(.5)
            continue
        contract = json.loads(receipt.read_text())
        assert contract['scope'] == 'VALID_ONLY_IMPLEMENTATION_REPLAY_V4' and contract['workers'] in (8, 11)
        observed = locate(args.batch, contract['selected_ids'])
        snapshot = sampler.snapshot(observed)
        ready = len(observed) == contract['workers'] and len({x['id'] for x in observed.values()}) == contract['workers']
        for pid, value in snapshot['workers'].items():
            slot = observed[int(pid)]['slot']
            expected = contract['inherited_allowed_cpus'][16 + 8 * slot:24 + 8 * slot]
            ready = ready and 'stat' in value and int(value.get('rss', '0 kB').split()[0]) >= 2 * 1024**2
            ready = ready and bool(value.get('threads_detail')) and all(t['affinity'] == expected for t in value.get('threads_detail', {}).values())
        if ready:
            first = snapshot
            break
        if (args.batch / 'batch_execution.json').exists():
            break
        time.sleep(.5)
    if first is None:
        report = {'status': 'NO_ALL_BOUND_ALLOCATED_WORKER_WINDOW_OBSERVED', 'wait_seconds': time.monotonic() - started, 'observed_workers': observed, 'last_snapshot': snapshot if contract else None, 'no_inference_or_recipe_changes': True}
    else:
        time.sleep(15)
        last = sampler.snapshot(observed)
        elapsed, hz = last['monotonic'] - first['monotonic'], os.sysconf('SC_CLK_TCK')
        delta = {}
        for pid, a in first['workers'].items():
            b = last['workers'][pid]
            if 'stat' not in a or 'stat' not in b:
                delta[pid] = {'incomplete': True}
                continue
            assert a['stat']['start_ticks'] == b['stat']['start_ticks']
            cpu = ((b['stat']['user_ticks'] + b['stat']['system_ticks']) - (a['stat']['user_ticks'] + a['stat']['system_ticks'])) / hz
            tids = {}
            for tid in set(a['threads_detail']) & set(b['threads_detail']):
                x, y = a['threads_detail'][tid], b['threads_detail'][tid]
                tids[tid] = {'cpu_seconds': ((y['user_ticks'] + y['system_ticks']) - (x['user_ticks'] + x['system_ticks'])) / hz, 'state_before_after': [x['state'], y['state']], 'wchan_before_after': [x['wchan'], y['wchan']]}
            delta[pid] = {'id': observed[int(pid)]['id'], 'cpu_seconds': cpu, 'effective_cpu_cores': cpu / elapsed, 'minor_fault_delta': b['stat']['minor_faults'] - a['stat']['minor_faults'], 'major_fault_delta': b['stat']['major_faults'] - a['stat']['major_faults'], 'io_delta': {k: int(b['io'][k]) - int(a['io'][k]) for k in a['io']}, 'threads': tids}
        report = {'status': 'FINITE_BOUND_ALLOCATED_WORKERS_SAMPLE_COMPLETE', 'scope': 'ONE_FINITE_READ_ONLY_PROC_SAMPLE', 'sample_seconds': elapsed, 'wait_for_bound_allocated_workers_seconds': first['monotonic'] - started, 'batch_inputs_sha256': hashlib.sha256((args.batch / 'batch_inputs.json').read_bytes()).hexdigest(), 'observed_workers': observed, 'before': first, 'after': last, 'delta': delta, 'cgroup_cpu_stat_delta': {k: last['cgroup_cpu_stat'][k] - first['cgroup_cpu_stat'][k] for k in first['cgroup_cpu_stat']}, 'probe_cpus': cpus, 'probe_nice': os.getpriority(os.PRIO_PROCESS, 0), 'claim_limit': 'Allocated/slot-bound inference window; wchan does not identify a Python stack. Cgroup includes every process. No thread/recipe changes or repeated models.'}
    report.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), snapshot_helper_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), test_inference=0)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k in ('status', 'sample_seconds', 'wait_for_bound_allocated_workers_seconds', 'delta', 'cgroup_cpu_stat_delta')}, indent=2))

if __name__ == '__main__':
    main()
