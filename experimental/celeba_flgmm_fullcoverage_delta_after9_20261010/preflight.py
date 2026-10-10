from pathlib import Path
import hashlib,json,os,subprocess,sys,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
stage=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage')
assert sha(stage/'PACKAGE_SHA256.json')=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
sys.path.insert(0,str(stage));from screen_common import local_identity,repo_identity
protocol,manifest=local_identity();identity=repo_identity(Path('/workspace/GuardFed-celeba-expanded'),protocol)
service=subprocess.run(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],capture_output=True,text=True);assert service.stdout.split()[1]=='RUNNING'
threads=[];workers=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  argv=[a for a in (proc/'cmdline').read_bytes().decode(errors='replace').split('\0') if a]
  if not argv:continue
  for t in (proc/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16:assert 106 not in aff,('CPU106 conflict',proc.name,t.name);threads.append(dict(pid=int(proc.name),tid=int(t.name),affinity=sorted(aff)))
  if str(stage/'run_one.py') in argv:workers.append(dict(pid=int(proc.name),argv=argv))
 except (FileNotFoundError,ProcessLookupError):continue
assert not list(stage.glob('QUEUE_FAILURE*.json')) and 1<=len(workers)<=2
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),guide_sha256=sha('/etc/vast-agents-guide.md'),package_sha256=sha(stage/'PACKAGE_SHA256.json'),source_data_verified=identity,scope_jobs=len(manifest['jobs']),reuse_references=len(manifest['reused_jobs']),service=service.stdout,actual_workers=workers,restricted_threads=threads,CPU106_free=True,terminal_collection_not_observed_here=True)))
