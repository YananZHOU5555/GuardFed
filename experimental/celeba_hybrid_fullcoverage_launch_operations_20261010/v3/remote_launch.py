"""One externally approved96-new Hybrid validation launch after real seven-canary closure."""
from pathlib import Path
from collections import deque
import datetime,hashlib,json,os,re,shutil,subprocess,sys,time,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
HERE=Path(__file__).resolve().parent
STAGE=Path('/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage')
REPO=Path('/workspace/GuardFed-celeba-expanded');MAIN=REPO/'results/revision_20261009/celeba_mechanism_v1'
SERVICE='guardfed_celeba_hybrid_fullcoverage'
PYTHON='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
from main_health import main_health
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def command(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=30);return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
def state(result):
 fields=result['stdout'].split();return fields[1] if len(fields)>1 else None

def resource(baseline):
 assert os.sched_getaffinity(0)=={107} and os.getpriority(os.PRIO_PROCESS,0)>=10,'Outer helper must use CPU107/nice10'
 assert 104 in set(os.sched_getaffinity(1)),'Worker CPU104 outside container affinity'
 raw={};tasks=[];nominal=0;actual_main=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   argv=[x for x in (proc/'cmdline').read_bytes().decode(errors='replace').split('\0') if x]
   if not argv:continue
   stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
   if stat[0]=='Z':continue
   env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
   affinity=sorted(os.sched_getaffinity(int(proc.name)));per_thread={}
   for t in (proc/'task').iterdir():
    try:cpus=sorted(os.sched_getaffinity(int(t.name)))
    except ProcessLookupError:continue
    per_thread[t.name]=cpus
    assert not (len(cpus)<=16 and set(cpus)&{104}),('Restricted CPU104 owner conflict',proc.name,t.name)
   assert not any(str(STAGE) in x for x in argv),('Existing stage producer',proc.name)
   upper=max([int(env[k]) for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS') if env.get(k,'').isdigit()] or [1])
   if 'python' in Path(argv[0]).name:nominal+=upper
   tasks.append(dict(pid=int(proc.name),argv=argv,affinity=affinity,thread_affinities=per_thread,declared_upper_threads=upper))
   if str(REPO/'deployment/celeba_mechanism_20261009/worker.py') in argv and '--job' in argv:
    job=Path(argv[argv.index('--job')+1]);assert job.parent==MAIN/'jobs'
    actual_main.append(dict(pid=int(proc.name),id=job.stem,argv=argv))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 queue=read(MAIN/'formal_queue_progress.json');service=command(['supervisorctl','status','guardfed_celeba_mechanism_formal'])
 guard=main_health(service,queue,'hybrid96_validation_launch')
 assert 1<=len(actual_main)<=8,'Protected main actual worker count outside1..8'
 failures=[str(p) for p in (MAIN/'runs').glob('*/failure*.json')];assert not failures
 errors=[];progress=[];pattern=re.compile(r'Traceback|RuntimeError|CUDA error|out of memory|OutOfMemory|\bnan\b|\binf\b|fatal',re.I)
 for item in queue['active']:
  path=MAIN/'runs'/item['id']/'progress.json';row=read(path) if path.exists() else {}
  log=MAIN/'logs'/(item['id']+'.log')
  progress.append(dict(id=item['id'],pid=item['pid'],progress=row,log_exists=log.exists(),log_mtime=log.stat().st_mtime if log.exists() else None))
  if log.exists():
   with log.open(errors='replace') as stream:errors.extend(dict(id=item['id'],line=x.strip()[:1000]) for x in deque(stream,maxlen=100) if pattern.search(x))
 assert not errors
 previous={r['id']:r.get('progress',{}) for r in baseline['active']}
 changes=[dict(id=r['id'],before=previous[r['id']].get('round'),after=r['progress'].get('round')) for r in progress if r['id'] in previous]
 grown=len(queue['completed'])>baseline['queue_completed'] or any(type(x['before']) is int and type(x['after']) is int and x['after']>x['before'] for x in changes)
 assert len(queue['completed'])>=baseline['queue_completed'] and grown,'No measured main round/completion growth versus approved snapshot'
 baseline_time=datetime.datetime.fromisoformat(baseline['checked_utc']).timestamp();assert 0<=time.time()-baseline_time<=3600
 old=command(['supervisorctl','status','guardfed_celeba_hybrid_screen32']);sg=command(['supervisorctl','status','sglang'])
 canary=command(['supervisorctl','status','guardfed_celeba_hybrid_fullcoverage_canary'])
 assert state(old)=='EXITED' and state(canary)=='EXITED' and state(sg)=='STOPPED'
 raw={n:Path('/sys/fs/cgroup',n).read_text().strip() for n in ('cpu.max','cpu.stat','memory.current','memory.max','memory.events')}
 sample_started=time.monotonic();time.sleep(0.2);cpu_stat_after=Path('/sys/fs/cgroup/cpu.stat').read_text().strip()
 cpu_delta=int(dict(x.split() for x in cpu_stat_after.splitlines())['usage_usec'])-int(dict(x.split() for x in raw['cpu.stat'].splitlines())['usage_usec'])
 measured_cpu_cores=cpu_delta/1e6/(time.monotonic()-sample_started)
 quota,period=raw['cpu.max'].split();assert quota!='max';cores=int(quota)/int(period)
 # One sequential canary child + coordinator, with one extra launch helper reserve.
 planned=nominal+3;assert planned<=cores
 assert raw['memory.max']!='max';free=int(raw['memory.max'])-int(raw['memory.current']);assert free>=8*1024**3
 memory_events=dict(line.split() for line in raw['memory.events'].splitlines());assert int(memory_events.get('oom_kill','0'))==0
 gpu=command(['nvidia-smi','--query-gpu=index,uuid,memory.free,temperature.gpu','--format=csv,noheader,nounits']);assert gpu['returncode']==0
 values=[[x.strip() for x in line.split(',')] for line in gpu['stdout'].splitlines()];assert len(values)==2 and [int(x[0]) for x in values]==[0,1]
 gpu_free=[int(x[2]) for x in values];assert gpu_free[0]>=4096
 assert values[0][1]==read(STAGE/'BINDINGS.json')['gpu_uuid'],'Bound GPU0 identity changed'
 details=command(['nvidia-smi','-q']);assert details['returncode']==0
 recovery=[x.split(':',1)[1].strip() for x in details['stdout'].splitlines() if 'GPU Recovery Action' in x];assert recovery==['None','None']
 disk=shutil.disk_usage(STAGE).free;assert disk>=5*1024**3
 return dict(observed_unix=time.time(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protected_main_max_workers=8,protected_main_healthy=True,
  planned_total_cpu_threads=planned,cpu_quota_cores=cores,free_memory_bytes=free,gpu_recovery_actions=recovery,gpu_free_memory_mib=gpu_free[0],gpu_recovery_action=recovery[0],
  original_screen_exited=True,no_duplicate_workers=True,no_restricted_CPU_overlap=True,
  raw_cgroup=raw,cpu_stat_after=cpu_stat_after,measured_cpu_cores=measured_cpu_cores,all_process_thread_owners=tasks,main_guard=guard,main_actual_workers=actual_main,main_progress=progress,main_round_changes=changes,
  main_growth_observed=grown,main_failure_files=failures,main_recent_log_errors=errors,baseline_snapshot_sha256=sha(HERE/'BASELINE.json'),
  old_hybrid=old,completed_canary_service=canary,sglang=sg,gpu_query=gpu,gpu_detail=details,disk_free_bytes=disk,reserved_affinity=[104],helper_affinity=[107],computational_threads_per_child=1,
  notes='Completed count only supports live health; no scientific acceptance. Auxiliary library threads are not counted as full cores; declared upper thread budgets recorded.')


def check_closure(approval,gate,offserver,closure):
 assert approval['status']=='ROOT_AUTHORIZED_HYBRID96_VALID_ONLY' and approval['scope']=='96_new_70round_valid_only'
 assert approval['service']==SERVICE and approval['cpus']==[104] and approval['cpu_threads']==1
 assert (approval['planned_new'],approval['reused'],approval['planned_total'])==(96,4,100) and approval['final_test'] is False
 assert len(approval['implementation_source_seal_sha256'])==64 and set(approval['implementation_source_seal_sha256'])<=set('0123456789abcdef')
 assert gate['status']=='SEVEN_HYBRID_CANARIES_STRICT_PASS_BACKUP_PENDING' and gate['package_sha256']==approval['package_sha256']
 assert len(gate['accepted_ids'])==len(set(gate['accepted_ids']))==7 and len(gate['pairs'])==2 and gate['scientific70records']==0 and gate['test'] is False
 assert closure['status']=='ROOT_SEVEN_HYBRID_CANARIES_OFFSERVER_ADOPTED' and closure['package_sha256']==approval['package_sha256']
 assert closure['gate_sha256']==approval['gate_acceptance_sha256']
 assert offserver['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON'
 local=offserver['local'];assert local['package_sha256']==approval['package_sha256'] and local['gate_sha256']==closure['gate_sha256']
 assert (local['accepted_new'],local['same_horizon_pairs'],local['total_runs'],local['rounds'],local['formal_table_samples'])==(7,2,7,3,0)
 assert local['CNN_calls']==0 and local['local_CUDA_initialized'] is False

def unlocked_canary_file(path):
 import fcntl
 assert path.is_file() and not path.is_symlink(),'Original canary lock file required'
 with path.open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  fcntl.flock(lock,fcntl.LOCK_UN)
 # The historical lock file remains unchanged; its existence is not a running owner.

def main():
 assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
 transfer=read(HERE/'TRANSFER.json')
 for name,pin in transfer['members'].items():assert sha(HERE/name)==pin['sha256']
 approval=read(HERE/'APPROVAL.json')
 assert sha(HERE/'FILES_SHA256.json')==approval['helper_seal_sha256']
 assert sha(STAGE/'PACKAGE_SHA256.json')==approval['package_sha256']
 for key,name in [('gate_acceptance_sha256','GATE_ACCEPTANCE.json'),('gate_offserver_sha256','GATE_OFFSERVER.json'),('root_closure_sha256','ROOT_CLOSURE.json'),('baseline_snapshot_sha256','BASELINE.json')]:assert sha(HERE/name)==approval[key]
 gate=read(HERE/'GATE_ACCEPTANCE.json');offserver=read(HERE/'GATE_OFFSERVER.json');closure=read(HERE/'ROOT_CLOSURE.json')
 check_closure(approval,gate,offserver,closure)
 assert sha(STAGE/'GATE_ACCEPTANCE.json')==approval['gate_acceptance_sha256']
 previous=(STAGE/'EXECUTION_AUTHORIZATION.json').read_bytes()
 assert hashlib.sha256(previous).hexdigest()==approval['canary_authorization_sha256']
 prior=json.loads(previous);assert prior['status']=='AUTHORIZED' and prior['scope']=='seven_same_horizon_3round_canaries' and prior['package_sha256']==approval['package_sha256']
 assert prior['max_workers']==prior['cpu_threads_per_worker']==1 and prior['allowed_cpus']==[104] and prior['gpu_index']==0 and prior['automatic_retry'] is False and prior['final_test'] is False
 for n in ('runs','logs','queue_progress.json','summary.json','EXECUTION_AUTHORIZATION.fullcoverage.pending.json'):assert not (STAGE/n).exists(),('Prior fullcoverage output preserved',n)
 assert not list(STAGE.glob('QUEUE_FAILURE*.json')) and not (STAGE/'GATE_FAILURE.json').exists() and not list((STAGE/'gate_runs').rglob('failure*.json'))
 assert not list(HERE.glob('FAILURE*.json')) and not (HERE/'START.json').exists()
 unlocked_canary_file(STAGE/'coordinator.lock')
 sys.path.insert(0,str(STAGE))
 from common import local_identity,repo_identity
 protocol,manifest=local_identity();assert len(manifest['preflight_jobs'])==7 and len(manifest['jobs'])==96 and len(manifest['reused_jobs'])==4
 assert sha(STAGE/'PREPARED_SOURCE_SEAL.json')==approval['implementation_source_seal_sha256']
 repo_identity(REPO,protocol);source_data=read(STAGE/'full_scope.json')['protected_source_hashes']
 assert gate['accepted_ids']==[r['id'] for r in manifest['preflight_jobs']]
 actual={p.relative_to(STAGE).as_posix():sha(p) for parent in ('gate_runs','gate_logs') for p in (STAGE/parent).rglob('*') if p.is_file()}
 assert actual==gate['artifact_hashes'],'Actual final seven-canary files/logs changed'
 from run_fullcoverage import original_runner
 reused=original_runner()['reused_records'](manifest);assert len(reused)==4
 assert not (Path('/etc/supervisor/conf.d')/(SERVICE+'.conf')).exists()
 services=command(['supervisorctl','status',SERVICE]);assert 'no such process' in services['stdout'].lower() or 'no such process' in services['stderr'].lower()
 resource_receipt=resource(read(HERE/'BASELINE.json'));resource_receipt['source_data_verified']=source_data
 save(HERE/'RESOURCE.json',resource_receipt)
 authorization=dict(status='AUTHORIZED',scope='96_new_70round_valid_only',package_sha256=approval['package_sha256'],
  resource_review_utc=resource_receipt['utc'],no_duplicate_workers_verified=True,max_workers=1,cpu_threads_per_worker=1,allowed_cpus=[104],gpu_index=0,automatic_retry=False,final_test=False,
  resource_receipt_path=str(HERE/'RESOURCE.json'),resource_receipt_sha256=sha(HERE/'RESOURCE.json'),root_approval_sha256=sha(HERE/'APPROVAL.json'),
  gate_acceptance_sha256=closure['gate_sha256'],gate_root_closure_path=str(HERE/'ROOT_CLOSURE.json'),gate_root_closure_sha256=sha(HERE/'ROOT_CLOSURE.json'),planned_new=96,reused=4,planned_total=100)
 with (HERE/'PREVIOUS_CANARY_AUTHORIZATION.json').open('xb') as f:f.write(previous)
 save(HERE/'NEW_EXECUTION_AUTHORIZATION.json',authorization)
 temporary=STAGE/'EXECUTION_AUTHORIZATION.fullcoverage.pending.json'
 with temporary.open('xb') as f:f.write((HERE/'NEW_EXECUTION_AUTHORIZATION.json').read_bytes())
 assert sha(STAGE/'EXECUTION_AUTHORIZATION.json')==approval['canary_authorization_sha256']
 os.replace(temporary,STAGE/'EXECUTION_AUTHORIZATION.json')
 config=f'''[program:{SERVICE}]
command=/usr/bin/taskset -c 104 /usr/bin/ionice -c 3 /usr/bin/nice -n 10 {PYTHON} -B {STAGE}/run_fullcoverage.py run --repo {REPO}
directory={STAGE}
autostart=false
autorestart=false
startretries=0
startsecs=1
stopasgroup=true
killasgroup=true
environment=CUDA_VISIBLE_DEVICES="0",GUARDFED_CPU_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONDONTWRITEBYTECODE="1"
stdout_logfile={HERE}/fullcoverage.stdout.log
stderr_logfile={HERE}/fullcoverage.stderr.log
'''
 path=Path('/etc/supervisor/conf.d')/(SERVICE+'.conf')
 with path.open('x') as f:f.write(config)
 (HERE/'SUPERVISOR.conf').write_text(config)
 receipts=[]
 for index,argv in enumerate((['supervisorctl','reread'],['supervisorctl','update',SERVICE])):
  result=command(argv);receipts.append(dict(command=argv,**result));save(HERE/f'SUPERVISOR_COMMAND_{index}.json',receipts[-1]);assert result['returncode']==0,result
 assert time.time()-resource_receipt['observed_unix']<=120
 result=command(['supervisorctl','start',SERVICE]);receipts.append(dict(command=['supervisorctl','start',SERVICE],**result))
 save(HERE/'START_COMMANDS.json',receipts);assert result['returncode']==0,result
 save(HERE/'START.json',dict(status='START_COMMAND_SUCCEEDED_NOT_SCIENTIFIC_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  service=command(['supervisorctl','status',SERVICE]),package_sha256=approval['package_sha256'],authorization_sha256=sha(STAGE/'EXECUTION_AUTHORIZATION.json'),resource_sha256=sha(HERE/'RESOURCE.json'),
  only_service_changed=SERVICE,planned_new=96,reused=4,planned_total=100,fullcoverage_start_command_succeeded=True,scientific_acceptance=False))
 print(json.dumps(read(HERE/'START.json')))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  save(HERE/('FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,
   resource_guard_inputs=getattr(e,'resource_guard_inputs',None),
   timeout_stdout=str(e.stdout) if isinstance(e,subprocess.TimeoutExpired) else None,
   timeout_stderr=str(e.stderr) if isinstance(e,subprocess.TimeoutExpired) else None));raise
