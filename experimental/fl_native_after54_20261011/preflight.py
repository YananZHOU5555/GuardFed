from pathlib import Path
import hashlib,json,os,subprocess,sys,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert os.sched_getaffinity(0)=={110} and os.getpriority(os.PRIO_PROCESS,0)==10
assert 'idle' in subprocess.check_output(['ionice','-p',str(os.getpid())],text=True)
stage=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage')
assert sha(stage/'PACKAGE_SHA256.json')=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
sys.path.insert(0,str(stage));from screen_common import local_identity,repo_identity
protocol,manifest=local_identity();identity=repo_identity(Path('/workspace/GuardFed-celeba-expanded'),protocol)
service=subprocess.run(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage'],capture_output=True,text=True);assert service.stdout.split()[1]=='RUNNING'
workers=[];restricted=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  argv=[a for a in (proc/'cmdline').read_bytes().decode(errors='replace').split('\0') if a]
  if not argv:continue
  for t in (proc/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16:assert 110 not in aff,('CPU110 conflict',proc.name,t.name)
  aff=os.sched_getaffinity(int(proc.name))
  if len(aff)<=16:restricted.append(dict(pid=int(proc.name),affinity=sorted(aff),executable=Path(argv[0]).name))
  if str(stage/'run_one.py') in argv:workers.append(dict(pid=int(proc.name),argv=argv))
 except (FileNotFoundError,ProcessLookupError):continue
assert not list(stage.glob('*FAILURE*.json')) and 1<=len(workers)<=2
q=read(stage/'queue_progress.json');active={r['id'] for r in q['active']};ids=['FLGMM_Tg20_L2.0_lr0.001_non-IID_Benign_seed91008_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_Benign_seed91009_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_Benign_seed91010_fullcoverage'];rows=[]
for identity_id in ids:
 item=next(r for r in manifest['jobs'] if r['id']==identity_id);out=stage/'runs'/identity_id
 assert identity_id not in active and all((out/n).is_file() for n in ('result.json','acceptance.json','screen_identity.json','model.pt','job.json'))
 progress=read(out/'progress.json');assert progress['round']==70 and not Path('/proc',str(progress['pid'])).exists()
 assert sha(out/'job.json')==item['job_sha256']
 rows.append(dict(id=identity_id,round=progress['round'],original_pid_absent=True,job_sha256=item['job_sha256'],model_sha256=sha(out/'model.pt'),result_sha256=sha(out/'result.json')))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),guide_sha256=sha('/etc/vast-agents-guide.md'),package_sha256=sha(stage/'PACKAGE_SHA256.json'),source_data_verified=identity,scope_jobs=len(manifest['jobs']),reuse_references=len(manifest['reused_jobs']),service=service.stdout,actual_workers=workers,restricted_processes=restricted,CPU110_free=True,collector=dict(pid=os.getpid(),CPU=110,threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),IO='idle'),authorized_terminal_rows=rows,queue=q,cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),disk_free_bytes=__import__('shutil').disk_usage('/workspace').free,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True),scientific_acceptance_performed=False)))
