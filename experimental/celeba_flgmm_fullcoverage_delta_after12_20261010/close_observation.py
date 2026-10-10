from pathlib import Path
import datetime,hashlib,json,os,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
busy=[];collectors=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any('celeba_flgmm_fullcoverage_delta_after12_20261010/collect_delta.py' in x for x in argv):collectors.append(dict(pid=int(p.name),argv=argv))
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 107 in aff:busy.append(dict(pid=int(p.name),tid=int(t.name),affinity=sorted(aff)))
 except (FileNotFoundError,ProcessLookupError):continue
assert not collectors and not busy
stage=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage');q=json.loads((stage/'queue_progress.json').read_bytes())
service=subprocess.check_output(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],text=True);assert service.split()[1]=='RUNNING' and not list(stage.glob('*FAILURE*.json'))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),collector_processes=collectors,restricted_CPU107_threads=busy,CPU107_released=True,service=service.strip(),queue=q,package_sha256=sha(stage/'PACKAGE_SHA256.json'),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),disk_free_bytes=__import__('shutil').disk_usage('/workspace').free,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True),new_acceptance_performed=False)))
