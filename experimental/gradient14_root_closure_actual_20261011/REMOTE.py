
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J=lambda p:json.loads(Path(p).read_bytes())
stage=Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011')
repo=Path('/workspace/GuardFed-celeba-expanded');package=repo/'tmp/celeba_gradient_fullcoverage_gates_prepare_20261010'
python='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
assert H('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert H(stage/'scope.json')=='22eba673a701d569b35736875245373d15696bca6a1ee9421165a5ed0e7232ff'
for p,h in [('adapter.py','248d6fdce737c142304ca0d9f8955ff734f01edb3bcfe1a9b984ab8fbbb81ba1'),('metadata.py','71616f5fba61d9fee704f3d57bfe220e7c4cdfbdecd95fc7abde6d6037ea50ef'),('compare_saved.py','a5e8bd339e418aea4c46834c022cdb504004e3e2067b897f0a35ea6a645e56fb')]:assert H(package/p)==h
scope=J(stage/'scope.json');assert len(scope['jobs'])==14
assert not (stage/'GATE_COMPARISON.json').exists() and not list(stage.glob('failure*.json'))
for row in scope['jobs']:
 out=stage/row['output'];a=J(out/'acceptance.json');i=J(out/'item_receipt.json')
 assert a['status']=='PASS' and a['job_sha256']==row['job_sha256'] and a['scope_sha256']==H(stage/'scope.json')
 assert i['status']=='CPU3_GATE_ORIGINAL_STRICT_AND_EXTERNAL_IDENTITY_PASS' and i['acceptance_sha256']==H(out/'acceptance.json') and i['rng_sha256']==H(out/'rng_after.json')
 assert not list(out.glob('failure*.json'))
owners=[];producers=[];broad=0
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) in (os.getpid(),os.getppid()):continue
 try:
  if (proc/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  if str(stage) in argv and any(x.endswith('/adapter.py') or x.endswith('/compare_saved.py') for x in argv):producers.append({'pid':int(proc.name),'argv':argv})
  for task in (proc/'task').iterdir():
   mask=os.sched_getaffinity(int(task.name))
   if 110 in mask and len(mask)<=32:owners.append({'pid':int(proc.name),'tid':int(task.name),'cpus':sorted(mask)})
   elif 110 in mask:broad+=1
 except (FileNotFoundError,ProcessLookupError):pass
assert not owners and not producers,{'owners':owners,'producers':producers}
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota!='max' and int(quota)/int(period)>=32
assert int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())>=8*1024**3
out=stage/'runtime/root_compare_actual001';out.mkdir()
pre={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_sha256':H(stage/'scope.json'),'original_items':14,'cpu':110,'threads':1,'restricted_owners':owners,'producers':producers,'broad_threads_containing_cpu':broad,'host_exclusive_claim':False,'guide_sha256':H('/etc/vast-agents-guide.md'),'test':False,'new_training':0}
(out/'PREFLIGHT.json').write_text(json.dumps(pre,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
argv=['taskset','-c','110','nice','-n','10','ionice','-c','3',python,'-B',str(package/'compare_saved.py'),'--stage',str(stage),'--repo',str(repo)]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();r=subprocess.run(argv,env=env,capture_output=True)
(out/'stdout.json').write_bytes(r.stdout);(out/'stderr.txt').write_bytes(r.stderr)
(out/'COMMAND.json').write_text(json.dumps({'argv':argv,'environment':{k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'automatic_retry':False},indent=2)+'\n')
assert r.returncode==0,r.stderr.decode(errors='replace')[-3000:]
comparison=J(stage/'GATE_COMPARISON.json');assert comparison['status']=='EXACT14_CPU3_GATE_SAVED_COMPARISON_PASS' and comparison['checked_jobs']==14 and len(comparison['comparisons'])==8
assert comparison['scientific_table_records']==0 and comparison['coverage_authorized'] is False
files={};runtime_sources={};commands={}
for p in sorted((stage/'runtime').rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink();rel=p.relative_to(stage).as_posix();pin={'sha256':H(p),'bytes':p.stat().st_size};files[rel]=pin
 if p.suffix=='.py' and 'local_evidence' not in p.relative_to(stage/'runtime').parts:runtime_sources[rel]=pin
 if p.name.endswith('_COMMAND.json') or p==out/'COMMAND.json':commands[rel]=J(p)
flbase=Path('/workspace/guardfed_checks/fl_SDFA9_three_view_source_prepare_20261011');g=flbase/'outputs/attempt001/GATE_RESULT.json'
sv=subprocess.run(['supervisorctl','status','guardfed_flgmm_SDFA9_valid'],capture_output=True,text=True)
fl={'service_exit':sv.returncode,'service':sv.stdout.strip(),'gate_exists':g.is_file(),'failure':(flbase/'outputs/attempt001/FAILURE.json').exists()}
if g.is_file():
 v=J(g);fl.update(gate_sha256=H(g),status=v['status'],receipts=len(v['receipts']))
print(json.dumps({'status':'ROOT_ACTUAL_ORIGINAL14_COMPARISON_AND_RUNTIME_INVENTORY','comparison_sha256':H(stage/'GATE_COMPARISON.json'),'comparison':comparison,'runtime_files':files,'actual_runtime_source_inventory':runtime_sources,'commands':commands,'preflight':pre,'FL9':fl}))
