"""Linux stdlib spawn/resource canary; no scientific imports, labels or images."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import bounded_remaining as outer

here = Path(__file__).resolve().parent
assert sys.platform == 'linux' and os.getpriority(os.PRIO_PROCESS, 0) == 0
allowed = sorted(os.sched_getaffinity(0))
assert len(allowed) >= 104
os.sched_setaffinity(0, [allowed[16]])
subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True, capture_output=True)
code = '''import os,json,subprocess,pathlib
allowed=sorted(os.sched_getaffinity(0))
before=os.getpriority(os.PRIO_PROCESS,0)
os.nice(10)
cpus=allowed[16:24]
os.sched_setaffinity(0,cpus)
print(json.dumps({'nice_before_original_run_adjustment':before,'nice_after_original_run_adjustment':os.getpriority(os.PRIO_PROCESS,0),'inherited_allowed_cpu_n':len(allowed),'child_bound_cpus':sorted(os.sched_getaffinity(0)),'ionice':subprocess.run(['ionice','-p',str(os.getpid())],capture_output=True,text=True).stdout.strip(),'child_pid':os.getpid()}))
'''
log = here / 'preflight_spawn_child.log'
assert not log.exists()
with log.open('xb') as f:
    result = outer.invoke([sys.executable, '-c', code], f, allowed)
child = json.loads(log.read_text())
assert result == 0 and child['nice_before_original_run_adjustment'] == 0 and child['nice_after_original_run_adjustment'] == 10
assert child['child_bound_cpus'] == allowed[16:24] and child['ionice'].startswith('idle')
assert os.getpriority(os.PRIO_PROCESS, 0) == 0 and sorted(os.sched_getaffinity(0)) == [allowed[16]]
report = {'status': 'LINUX_REAL_SPAWN_NICE_AND_AFFINITY_PASS_NO_INFERENCE', 'utc_unix': time.time(), 'outer_source_sha256': hashlib.sha256((here / 'bounded_remaining.py').read_bytes()).hexdigest(), 'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'parent_nice_before_after': [0, 0], 'parent_final_allowed_cpus': sorted(os.sched_getaffinity(0)), 'child': child, 'priority_elevation_attempted': False, 'capability_or_system_changes': False, 'scientific_imports': False, 'new_image_inference': 0, 'semantic_label_arrays_loaded': False, 'child_process_exited': True}
outer.save(here / 'preflight_spawn.json', report)
print(json.dumps(report, indent=2))
