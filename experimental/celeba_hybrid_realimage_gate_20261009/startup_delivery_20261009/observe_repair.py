"""One bounded read-only progress snapshot; does not import scientific modules."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent / 'execution_repair_v1'
runtime = REPAIR / 'runtime_overlay'
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
processes = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if len(argv) < 4 or argv[1:4] != ['-u', str(REPAIR / 'repair_execute.py'), 'run']:
            continue
        env = dict(x.split('=', 1) for x in (proc / 'environ').read_text().split('\0') if '=' in x)
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        processes.append(dict(pid=int(proc.name), argv=argv, cpus=sorted(os.sched_getaffinity(int(proc.name))),
            task_cpus={p.name: sorted(os.sched_getaffinity(int(p.name))) for p in (proc / 'task').iterdir()},
            nice=os.getpriority(os.PRIO_PROCESS, int(proc.name)), cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
            process_state=stat[0], user_seconds=int(stat[11])/os.sysconf('SC_CLK_TCK'),system_seconds=int(stat[12])/os.sysconf('SC_CLK_TCK'),
            thread_environment={key:env.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
jobs = []
for entry in read(REPAIR / 'REPAIR_SCOPE.json')['new_jobs']:
    out = runtime / entry['output']
    jobs.append(dict(id=entry['id'], progress=read(out / 'progress.json') if (out / 'progress.json').exists() else None,
        diagnostic_rounds=[x['round'] for x in read(out / 'diagnostics.json')] if (out / 'diagnostics.json').exists() else [],
        acceptance_receipt_exists=(out / 'acceptance.json').exists(), failure=read(out / 'failure.json') if (out / 'failure.json').exists() else None))
formal = Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
queue = read(formal / 'formal_queue_progress.json')
active = []
for entry in queue['active']:
    progress = read(formal / 'runs' / entry['id'] / 'progress.json')
    active.append(dict(id=entry['id'], pid=entry['pid'], round=progress['round'], updated_unix=progress['updated_unix']))
now = datetime.datetime.now(datetime.timezone.utc)
report = dict(utc=now.isoformat(), service=subprocess.run(['supervisorctl','status','guardfed_celeba_hybrid_writer_repair_v2'], capture_output=True,text=True).stdout.strip(),
    actual_python_processes=processes, new_CANARY_jobs=jobs, new_scientific_table_records=0,
    original_failed_stage_status='TERMINAL_FAILURE_UNCHANGED', reused_IID_CANARY=2, repeated_IID_inference_or_training=0,
    original9_seal_sha256=digest(REPAIR / 'FILES_SHA256.json'),
    original_approval_sha256=digest(HERE / 'APPROVED.json'),
    new_repair_summary=read(runtime / 'repair_summary.json') if (runtime / 'repair_summary.json').exists() else None,
    new_repair_failure=read(runtime / 'repair_failure.json') if (runtime / 'repair_failure.json').exists() else None,
    formal_completed=len(queue['completed']), formal_failed=queue['failed'], formal_active=active,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True).strip())
target = HERE / ('live_repair_' + now.strftime('%Y%m%dT%H%M%SZ') + '.json')
assert not target.exists()
target.write_bytes((json.dumps(report, indent=2,allow_nan=False) + '\n').encode())
print(json.dumps(dict(snapshot=str(target), service=report['service'], processes=[{k:r[k] for k in ('pid','cpus','nice','cuda_visible_devices')} for r in processes],
    jobs=[dict(id=r['id'], round=r['progress']['round'] if r['progress'] else None, diagnostic_rounds=r['diagnostic_rounds'], failure=r['failure']) for r in jobs],
    formal_completed=report['formal_completed'], formal_failed=report['formal_failed'], formal_rounds=[r['round'] for r in active])), flush=True)
