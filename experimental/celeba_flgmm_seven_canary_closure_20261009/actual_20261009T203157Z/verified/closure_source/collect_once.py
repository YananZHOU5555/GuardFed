"""Future CPU106-only quiescent seven-canary acceptance/archive; no service mutations."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tarfile,time,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
from verify_saved import sha,verify,PACKAGE
HERE=Path(__file__).resolve().parent
STAGE=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage')
REPO=Path('/workspace/GuardFed-celeba-expanded')
SERVICE='guardfed_celeba_flgmm_fullcoverage_canary'
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source-seal-sha256',required=True);a=p.parse_args()
 assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
 assert sha(HERE/'FILES_SHA256.json')==a.source_seal_sha256
 seal=json.loads((HERE/'FILES_SHA256.json').read_bytes())
 for n,pin in seal['files'].items():assert sha(HERE/n)==pin['sha256']
 expected=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/seven_canary_closure/output')
 assert a.out==expected and a.out.parent.resolve()==a.out.parent and not a.out.exists() and not a.out.is_symlink()
 a.out.mkdir()
 try:
  assert os.sched_getaffinity(0)=={106} and os.getpriority(os.PRIO_PROCESS,0)>=10
  assert all(os.environ.get(n)=='1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES')==''
  owners=[];nominal=1
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
   try:
    argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
    if not argv[0]:continue
    assert not any(str(STAGE) in arg for arg in argv),'Stage producer still present'
    for task in (proc/'task').iterdir():
     try:aff=os.sched_getaffinity(int(task.name))
     except ProcessLookupError:continue
     assert not (len(aff)<=16 and 106 in aff),('CPU106 occupied',proc.name,task.name)
    if 'python' in Path(argv[0]).name:
     env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
     threads=max([int(env[k]) for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS') if env.get(k,'').isdigit()] or [1]);nominal+=threads
     owners.append({'pid':int(proc.name),'argv':argv,'declared_threads':threads})
   except (ProcessLookupError,FileNotFoundError):continue
  r=subprocess.run(['supervisorctl','status',SERVICE],capture_output=True,text=True,timeout=30)
  assert r.stdout.split()[1]=='EXITED',r.stdout
  raw={n:Path('/sys/fs/cgroup',n).read_text() for n in ('cpu.max','memory.current','memory.max','memory.events')}
  quota,period=raw['cpu.max'].split();assert quota!='max' and nominal<=int(quota)/int(period)
  assert raw['memory.max'].strip()!='max' and int(raw['memory.max'])-int(raw['memory.current'])>=8*1024**3
  save(a.out/'PREFLIGHT.json',{'time_unix':time.time(),'service':{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},'python_owners':owners,'nominal_cpu_threads':nominal,'raw_cgroup':raw,'no_producer':True})
  assert sha(STAGE/'PACKAGE_SHA256.json')==PACKAGE
  sys.path.insert(0,str(STAGE));import screen_common as sc
  protocol,manifest=sc.local_identity();before=sc.repo_identity(REPO,protocol)
  result=verify(STAGE);assert sc.repo_identity(REPO,protocol)==before;sc.local_identity()
  result['source_data_before']=before;result['source_data_after']=before
  save(a.out/'REMOTE_STRICT.json',result)
  paths={('stage/'+name):STAGE/name for name in json.loads((STAGE/'PACKAGE_SHA256.json').read_bytes())['files']}
  paths['stage/PACKAGE_SHA256.json']=STAGE/'PACKAGE_SHA256.json'
  for name in ('GATE_ACCEPTANCE.json','EXECUTION_AUTHORIZATION.json'):paths['stage/'+name]=STAGE/name
  for path in (STAGE/'preflight').rglob('*'):
   if path.is_file():paths['stage/'+path.relative_to(STAGE).as_posix()]=path
  auth=sc.read(STAGE/'EXECUTION_AUTHORIZATION.json');resource=Path(auth['resource_receipt_path'])
  assert sha(resource)==auth['resource_receipt_sha256']
  paths['evidence/RESOURCE.json']=resource
  for name in ('APPROVAL.json','START.json','START_COMMANDS.json','SUPERVISOR.conf'):
   paths['evidence/'+name]=resource.parent/name
  assert sha(paths['evidence/APPROVAL.json'])==auth['root_approval_sha256']
  for name in ('PREFLIGHT.json','REMOTE_STRICT.json'):paths['evidence/'+name]=a.out/name
  for name in seal['files']:paths['closure_source/'+name]=HERE/name
  paths['closure_source/FILES_SHA256.json']=HERE/'FILES_SHA256.json'
  inventory={n:{'sha256':sha(path),'bytes':path.stat().st_size} for n,path in sorted(paths.items())}
  save(a.out/'MEMBERS.json',{'files':inventory,'package_sha256':PACKAGE,'scope':'seven_same_horizon_3round_canaries'})
  archive=a.out/'seven_canaries.tar.gz'
  with tarfile.open(archive,'x:gz',dereference=True) as tar:
   for n,path in sorted(paths.items()):
    assert sha(path)==inventory[n]['sha256'];tar.add(path,arcname=n,recursive=False);assert sha(path)==inventory[n]['sha256']
   tar.add(a.out/'MEMBERS.json',arcname='MEMBERS.json',recursive=False)
  save(a.out/'BACKUP_RECEIPT.json',{'status':'REMOTE_STRICT_BACKUP_READY_NOT_OFFSERVER','archive_sha256':sha(archive),'inventory_sha256':sha(a.out/'MEMBERS.json'),'content_members':len(paths),'archive_members':len(paths)+1,'package_sha256':PACKAGE,'gate_sha256':result['gate_sha256'],'source_seal_sha256':a.source_seal_sha256,'scope':'seven_same_horizon_3round_canaries','formal_table_samples':0,'CNN_calls':0})
  print(json.dumps(sc.read(a.out/'BACKUP_RECEIPT.json')))
 except BaseException as e:
  save(a.out/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
