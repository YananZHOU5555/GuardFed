from pathlib import Path
import json,hashlib,subprocess,datetime,sys,os
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def invoke(cmd,stem,**kw):
 r=subprocess.run(cmd,capture_output=True,**kw)
 for n,b in [('stdout',r.stdout),('stderr',r.stderr)]:
  with (H/(stem+'.'+n)).open('xb') as f:f.write(b)
 save(stem+'_COMMAND.json',dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(H/(stem+'.stdout')),stderr_sha256=sha(H/(stem+'.stderr'))))
 return r
seal='40fa2578e9247d3bc746262915d4f54da470edcb980e364aa5a749babf94fd9e'
assert sha(P/'FILES_SHA256.json')==seal
for name,pin in json.loads((P/'FILES_SHA256.json').read_bytes())['files'].items():assert sha(P/name)==pin['sha256']
review=R/'tmp/LoGo100_cuda100_saved_closure_independent_review_20261011/SOURCE_REVIEW.json'
rv=json.loads(review.read_bytes());assert rv['source_adoptable'] and rv['source_seal_sha256']==seal and not rv['blocking_source']
s=json.loads((H/'observe002/stdout').read_bytes());scope=json.loads((P/'SCOPE.json').read_bytes())
assert s['service']['returncode'] in (0,3) and s['service']['stdout'].split()[:2]==['guardfed_uniform_cuda_LoGo100_remaining_valid','EXITED']
assert s['complete']['record']['ids']==scope['remaining_ids'] and s['complete']['record']['server_saved_accepted'] and not s['processes'] and not s['unreadable_pids'] and s['failure'] is None
assert len(s['tasks'])==98 and all(t['accepted'] and t['worker_command']['record']['exit_code']==0 for t in s['tasks'].values())
base='/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001'
code="IDS="+repr(scope['ids'])+"\nCANARIES="+repr(scope['canary_ids'])+"\n"+r'''from pathlib import Path
import json,hashlib,datetime,subprocess,os
B=Path('/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def pin(p):return dict(path=str(p),sha256=sha(p))
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(B/'ROOT_QUEUE_AUTH.json')=='9713c11187351d93f9f5869963926bc0bc5f04159f315f946807c7fa040cfad9'
assert sha(B/'DEPLOYMENT_SEAL.json')=='ca5eabc8d84559a934a0c822840af1a2dab3dee89e96c1bb0b80b5dd5e7154c1'
auth=read(B/'ROOT_QUEUE_AUTH.json');assert auth['ordered_ids']==IDS and auth['canary_ids']==CANARIES
for rel,p in read(B/'DEPLOYMENT_SEAL.json')['files'].items():assert sha(B/rel)==p['sha256']
assert sha(auth['exact_cohort']['path'])==auth['exact_cohort']['sha256']=='52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'
for name in ['FedAvg_CUDA_root','FedAvg_CUDA_index']:assert sha(auth[name]['path'])==auth[name]['sha256']
statuses={};complete={}
for phase,service,ids in [('canary','guardfed_uniform_cuda_LoGo100_canary_valid',CANARIES),('remaining','guardfed_uniform_cuda_LoGo100_remaining_valid',[i for i in IDS if i not in CANARIES])]:
 r=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True,timeout=20)
 assert r.returncode in (0,3) and r.stdout.split()[:2]==[service,'EXITED']
 assert not (B/phase/'QUEUE_FAILURE.json').exists()
 c=read(B/phase/'COMPLETE.json');p=read(B/phase/'progress.json');assert c['ids']==ids and c['cohort']==auth['exact_cohort'] and c['server_saved_accepted'] and p['accepted_ids']==ids and p['planned']==len(ids) and p['failed']==0
 statuses[phase]={'rc':r.returncode,'stdout':r.stdout};complete[phase]=pin(B/phase/'COMPLETE.json')
tasks={};workers={}
for i in IDS:
 t=B/'tasks'/i;o=B/'runs'/i
 assert not o.with_name(o.name+'.formal_failure.json').exists() and not list(o.glob('failure*.json'))
 a=read(t/'ACCEPTED.json');assert a['post_id']==i and a['root_adopted'] is False
 for role in ['approval','binding','acceptance']:assert sha(a[role]['path'])==a[role]['sha256']
 w=read(t/'worker_command.json');assert w['exit_code']==0
 tasks[i]=pin(t/'ACCEPTED.json');workers[i]=pin(t/'worker_command.json')
for proc in Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  for task in (proc/'task').iterdir():
   try:aff=os.sched_getaffinity(int(task.name))
   except ProcessLookupError:continue
   assert not (len(aff)<=16 and 110 in aff),'CPU110 occupied'
  argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
  assert str(B/'worker.py') not in argv,'LoGo producer present'
 except (FileNotFoundError,ProcessLookupError):continue
gate=read(B/'ROOT_CANARY_ACCEPTANCE.json');assert gate['root_accepted'] and gate['ids']==CANARIES and gate['cohort']==auth['exact_cohort']
assert all(gate['records'][i]['task_acceptance']==tasks[i] for i in CANARIES)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base=str(B),services=statuses,complete=complete,root_canary_gate=pin(B/'ROOT_CANARY_ACCEPTANCE.json'),root_queue_auth=pin(B/'ROOT_QUEUE_AUTH.json'),deployment_seal=pin(B/'DEPLOYMENT_SEAL.json'),task_accepted=tasks,worker_commands_exit0=workers,CPU110_unoccupied=True,source_auth_root_checked=True,accepted_count=len(tasks),no_producer_or_failure=True)))
'''
compile(code,'<actual100 JSON-only gates>','exec')
with (H/'BIND_REMOTE_EXECUTED.py').open('x',encoding='utf8') as f:f.write(code)
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -B -']
r=invoke(cmd,'BIND',input=code.encode(),timeout=90);assert r.returncode==0,r.stderr.decode(errors='replace')
b=json.loads(r.stdout);save('ACTUAL100_BINDING.json',b);assert b['accepted_count']==100 and b['CPU110_unoccupied']
assert all(b['task_accepted'][i]['sha256']==s['tasks'][i]['accepted']['sha256'] for i in scope['remaining_ids'])
prior=json.loads((R/'tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json').read_bytes())
assert all(b['task_accepted'][i]==prior['records'][i]['task_acceptance'] for i in scope['canary_ids'])
x=json.loads((P/'ROOT_INPUTS.template.json').read_bytes())
x.update(status='ROOT_AUTHORIZED_ACTUAL_LOGO100_DELTA98_TRANSPORT_NOT_ADOPTED',root_authorized_transport=True,server_base=base,service='guardfed_uniform_cuda_LoGo100_canary_valid',remaining_service='guardfed_uniform_cuda_LoGo100_remaining_valid',source_review={'path':str(review),'sha256':sha(review)},root_queue_auth=b['root_queue_auth'],deployment_seal=b['deployment_seal'],task_accepted=b['task_accepted'],canary_complete=b['complete']['canary'],remaining_complete=b['complete']['remaining'],root_canary_gate=b['root_canary_gate'])
save('ACTUAL_TRANSPORT_INPUTS.json',x);inp=H/'ACTUAL_TRANSPORT_INPUTS.json'
sys.path.insert(0,str(R/'tmp'));from guardfed_local_storage import check_bulk_storage
save('TRANSPORT_F_VOLUME.json',check_bulk_storage(250*1024**2))
cmd=[sys.executable,str(P/'transport.py'),'--inputs',str(inp),'--inputs-sha256',sha(inp),'--source-seal-sha256',seal]
save('TRANSPORT_START.json',dict(command=cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs_sha256=sha(inp)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
r=invoke(cmd,'TRANSPORT',cwd=R,env=env,timeout=450)
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
