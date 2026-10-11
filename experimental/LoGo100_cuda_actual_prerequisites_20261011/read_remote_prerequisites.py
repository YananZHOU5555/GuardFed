"""Read-only stdlib environment/resource inventory. No arrays, models or package imports."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')==GUIDE
envroots=[Path('/venv/main')]
parent=Path('/workspace/guardfed_envs')
if parent.is_dir():envroots+=sorted(p for p in parent.iterdir() if p.is_dir())
interpreters=[str(p/'bin/python') for p in envroots if (p/'bin/python').exists()]
interpreters+=['/usr/bin/python3','/usr/bin/python3.10']
query="import sys,json,importlib.metadata as m; out={'python':sys.version,'executable':sys.executable,'packages':{}}\nfor n in ['torch','numpy','netcal','scikit-learn','scipy','pandas','pip']:\n try: out['packages'][n]=m.version(n)\n except m.PackageNotFoundError:out['packages'][n]=None\nprint(json.dumps(out))"
envs=[]
for exe in dict.fromkeys(interpreters):
    if not Path(exe).exists():continue
    proc=subprocess.run([exe,'-B','-c',query],capture_output=True,text=True,timeout=20,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    row={'requested_executable':exe,'resolved_executable':str(Path(exe).resolve()),'executable_sha256':sha(Path(exe).resolve()),'rc':proc.returncode,'stderr':proc.stderr[-1200:]}
    if proc.returncode==0:row['metadata']=json.loads(proc.stdout)
    envs.append(row)
siblingtext=Path('/sys/devices/system/cpu/cpu113/topology/thread_siblings_list').read_text().strip();siblings=set()
for part in siblingtext.split(','):
    r=[int(i) for i in part.split('-')];siblings.update(range(r[0],r[-1]+1))
projects=[];conflicts=[];unknown=[];nominal=0;reserved=set();pidcount=taskcount=0
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:
        state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
        if state in ('Z','X'):continue
        argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        if not argv:continue
        pidcount+=1;python=Path(argv[0]).name.startswith('python')
        cwd=str((p/'cwd').resolve()) if python else ''
        project=python and (cwd.startswith(('/workspace/GuardFed-','/workspace/guardfed_checks/')) or any('/workspace/GuardFed-' in v or '/workspace/guardfed_checks/' in v for v in argv[1:]))
        declared=0
        if project:
            environment=dict(v.split('=',1) for v in (p/'environ').read_text().split('\0') if '=' in v)
            declared=max([int(environment[k]) for k in ['GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'] if environment.get(k,'').isdigit()]or[1]);nominal+=declared
        affinities=set()
        for t in (p/'task').iterdir():
            aff=set(os.sched_getaffinity(int(t.name)));taskcount+=1;affinities.add(tuple(sorted(aff)))
            if len(aff)<=16:
                if project:reserved.update(aff)
                if siblings & aff:conflicts.append({'pid':int(p.name),'tid':int(t.name),'affinity':sorted(aff),'argv':argv,'cwd':cwd})
        if project:projects.append({'pid':int(p.name),'argv':argv,'cwd':cwd,'declared_threads':declared,'affinities':[list(v) for v in sorted(affinities)]})
    except (FileNotFoundError,ProcessLookupError):continue
    except PermissionError as exc:
        try:
            state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (FileNotFoundError,ProcessLookupError):continue
        if state in ('Z','X'):continue
        unknown.append({'pid':int(p.name),'error':repr(exc),'fresh_state':state})
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();cores=None if quota=='max' else int(quota)/int(period)
memorymax=Path('/sys/fs/cgroup/memory.max').read_text().strip();memoryused=int(Path('/sys/fs/cgroup/memory.current').read_text())
services=subprocess.run(['supervisorctl','status'],capture_output=True,text=True,timeout=20)
repo=Path('/workspace/GuardFed-celeba-expanded/scripts/reproduce_paper_tables.py')
out={'utc':datetime.now(timezone.utc).isoformat(),'guide_sha256':GUIDE,'environments':envs,'uv':shutil.which('uv'),'python310_on_path':shutil.which('python3.10'),
 'cpu113':{'siblings':sorted(siblings),'scanned_pids':pidcount,'scanned_tasks':taskcount,'conflicting_restricted_tasks':conflicts,'unreadable_live_processes':unknown,'nominal_project_threads':nominal,'restricted_project_cpu_union':sorted(reserved),'quota_cores':cores,'candidate_nominal_budget_fits':cores is not None and max(nominal,len(reserved))+1<=cores,'candidate_available_observation_only':not conflicts and not unknown and cores is not None and max(nominal,len(reserved))+1<=cores,'project_processes':projects},
 'memory':{'max':memorymax,'current':memoryused},'workspace_disk_free':shutil.disk_usage('/workspace').free,'services':{'rc':services.returncode,'stdout':services.stdout,'stderr':services.stderr},
 'historical_core_current':{'path':str(repo),'exists':repo.exists(),'sha256':sha(repo) if repo.is_file() else None},'package_imports_executed':False,'scientific_execution':False,'permission_to_start':False}
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
