"""Read-only bounded snapshot of the two already authorized validation screens."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

H = Path('/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009')
F = Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def probe(base, entries, output_key, parent=None):
    rows = []
    for e in entries:
        out = base/e[output_key] if output_key else base/parent/e['id']
        progress = read(out/'progress.json') if (out/'progress.json').exists() else None
        rows.append(dict(id=e['id'], output=str(out), progress=progress,
                         progress_mtime=(out/'progress.json').stat().st_mtime if progress else None,
                         terminal_acceptance=(out/'acceptance.json').exists(),
                         result_present=(out/'result.json').exists(),
                         failure_files=[str(p) for p in out.glob('*failure*.json')]))
    return rows
workers=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        argv=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        if not argv or 'python' not in Path(argv[0]).name or int(proc.name)==os.getpid(): continue
        cpus=sorted(os.sched_getaffinity(int(proc.name)))
        cwd=str((proc/'cwd').resolve())
        interesting=any('guardfed' in s.lower() or 'celeba' in s.lower() for s in argv) or len(cpus)<=16
        if interesting:
            env=dict(s.split('=',1) for s in (proc/'environ').read_text().split('\0') if '=' in s)
            workers.append(dict(pid=int(proc.name), argv=argv, cwd=cwd, nice=os.getpriority(os.PRIO_PROCESS,int(proc.name)),
                affinity=cpus if len(cpus)<=16 else dict(count=len(cpus),min=min(cpus),max=max(cpus)),
                threads={k:env.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','GUARDFED_CPU_THREADS','CUDA_VISIBLE_DEVICES')}))
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
hscope=read(H/'screen_scope.json'); fmanifest=read(F/'jobs/manifest.json')
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), host=os.uname().nodename,
    guide_sha256=sha(Path('/etc/vast-agents-guide.md')),
    services=subprocess.run(['supervisorctl','status'],capture_output=True,text=True).stdout,
    hybrid=dict(source_seal_sha256=sha(H/'FILES_SHA256.json'),scope_sha256=sha(H/'screen_scope.json'),
                protocol_sha256=sha(H/hscope['runtime_protocol']),
                terminal_flags=[p.name for p in H.glob('screen_*json') if p.name in {'screen_failure.json','screen_complete.json'}],
                rows=probe(H,hscope['jobs'],'output')),
    flgmm=dict(source_seal_sha256=sha(F/'PACKAGE_SHA256.json'),protocol_sha256=sha(F/'source/protocol.json'),
               manifest_sha256=sha(F/'jobs/manifest.json'),queue=read(F/'queue_progress.json'),
               terminal_flags=[p.name for p in F.glob('*FAILURE*.json')],
               rows=probe(F,fmanifest['jobs'],None,'runs')),
    workers=workers, cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),
    memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,
    gpu_recovery=subprocess.run(['nvidia-smi','--query-gpu=index,gpu_recovery_action','--format=csv,noheader'],capture_output=True,text=True).stdout,
    disk_free_bytes=os.statvfs('/workspace').f_bavail*os.statvfs('/workspace').f_frsize)
print(json.dumps(result,ensure_ascii=False))
