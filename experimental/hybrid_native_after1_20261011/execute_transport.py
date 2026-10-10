from pathlib import Path
import json,hashlib,subprocess,sys,base64,datetime,traceback,importlib.util
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
F=Path('F:/YananResearchStorage/GuardFed/hybrid_native_after1_20261011')
REMOTE='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/native_after1_20261011'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def run(n,cmd,code=None,timeout=600):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=code.encode() if code else None,capture_output=True,timeout=timeout)
 (H/(n+'.stdout')).write_bytes(p.stdout);(H/(n+'.stderr')).write_bytes(p.stderr);save(n+'_COMMAND.json',dict(command=cmd,returncode=p.returncode,start=start,finish=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdin_sha256=hashlib.sha256(code.encode()).hexdigest() if code else None));p.check_returncode();return p.stdout

def check():
 assert sha(H/'PRIOR_ROOT_ADOPTION.json')=='04d62c367609d1c6d079ff538f45a575bff368944d4833f47fd1cf28159c5fa4'
 assert read(H/'PRIOR_ROOT_ADOPTION.json')['cumulative_accepted']==1
 assert len(sys.argv)==3 and len(sys.argv[2])==64 and all(c in '0123456789abcdef' for c in sys.argv[2])
 assert sha(H/'FILES_SHA256.json')==sys.argv[2]
 for n,v in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/n)==v['sha256']
 assert read(H/'DELTA_SCOPE.json')['ids']==['CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91003_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91004_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91005_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91006_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91007_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91008_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91009_fullcoverage', 'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91010_fullcoverage']
 assert read(H/'DELTA_SCOPE.json')['accepted_before']==1
 assert read(H/'PRIOR_OFFSERVER.json')['accepted_ids']==read(H/'PRIOR_ROOT_ADOPTION.json')['accepted_new_ids']
def collect():
 names=list(read(H/'FILES_SHA256.json')['files'])+['FILES_SHA256.json']
 payload={n:base64.b64encode((H/n).read_bytes()).decode() for n in names};pins={n:sha(H/n) for n in names}
 code='''from pathlib import Path
import os,subprocess,hashlib,json,base64,sys,datetime
G=Path('/etc/vast-agents-guide.md').read_bytes();assert hashlib.sha256(G).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
busy=[];workers=[]
for p in Path('/proc').glob('[0-9]*'):
 if int(p.name)==os.getpid():continue
 try:
  a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if any(x.endswith('deployment/celeba_mechanism_20261009/worker.py') for x in a):workers.append({'pid':int(p.name),'argv':a})
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 108 in aff:busy.append({'pid':p.name,'tid':t.name,'cpus':sorted(aff),'argv':a})
 except (FileNotFoundError,ProcessLookupError):continue
main=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_formal'],capture_output=True,text=True)
q=json.loads(Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json').read_bytes())
pre=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),guide_sha256=hashlib.sha256(G).hexdigest(),main_service=main.stdout,main_workers=workers,main_queue=q,CPU108_owners=busy)
print(json.dumps({'preflight':pre}),flush=True)
assert not busy and main.returncode==0 and 'RUNNING' in main.stdout and 1<=len(workers)<=8 and not q['failed']
b=Path(%r);assert not b.exists() and not b.is_symlink();b.mkdir()
payload=%r;pins=%r
for n,value in payload.items():
 raw=base64.b64decode(value);assert hashlib.sha256(raw).hexdigest()==pins[n]
 with (b/n).open('xb') as f:f.write(raw)
(b/'ACTUAL_PRELAUNCH.json').write_text(json.dumps(pre,indent=2)+'\\n')
cmd=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','taskset','-c','108','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(b/'collect_once.py'),'--out',str(b/'output'),'--source-seal-sha256',pins['FILES_SHA256.json']]
p=subprocess.run(cmd,capture_output=True);(b/'COLLECT.stdout').write_bytes(p.stdout);(b/'COLLECT.stderr').write_bytes(p.stderr);sys.stdout.buffer.write(p.stdout);sys.stderr.buffer.write(p.stderr);sys.exit(p.returncode)
'''%(REMOTE,payload,pins)
 output=run('COLLECT',SSH+['python -B -'],code)
 values=[json.loads(x) for x in output.splitlines() if x.strip()];save('ACTUAL_PRELAUNCH.json',values[0]['preflight']);save('SERVER_RECEIPT.json',values[-1]);print(json.dumps(values[-1]))
def download():
 names=['delta.tar.gz','BACKUP_RECEIPT.json','MEMBERS.json','REMOTE_STRICT.json']
 code="from pathlib import Path\nimport hashlib,json\nb=Path(%r)\nprint(json.dumps({n:{'sha256':hashlib.sha256((b/n).read_bytes()).hexdigest(),'bytes':(b/n).stat().st_size} for n in %r}))\n"%(REMOTE+'/output',names)
 pins=json.loads(run('REMOTE_PINS',SSH+['python -B -'],code,60))
 spec=importlib.util.spec_from_file_location('storage',ROOT/'tmp/guardfed_local_storage.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);save('F_VOLUME_DOWNLOAD.json',m.check_bulk_storage(sum(x['bytes'] for x in pins.values())))
 assert not F.exists() and not F.is_symlink();F.mkdir(parents=True)
 run('SCP',['scp','-q','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15']+['root@89.22.197.55:'+REMOTE+'/output/'+n for n in names]+[str(F)],timeout=240)
 for n,p in pins.items():assert sha(F/n)==p['sha256'] and (F/n).stat().st_size==p['bytes']
 save('DOWNLOAD.json',dict(F_directory=F.as_posix(),files=pins,all_bulk_F=True))
def verify():
 run('RESTORE_VERIFY',[sys.executable,'-B',str(H/'restore_verify.py'),'--archive',str(F/'delta.tar.gz'),'--receipt',str(F/'BACKUP_RECEIPT.json'),'--receipt-sha256',read(H/'DOWNLOAD.json')['files']['BACKUP_RECEIPT.json']['sha256'],'--out',str(F/'verified')],timeout=240)
 print(sha(F/'verified/OFFSERVER_VERIFICATION.json'))
if __name__=='__main__':
 try:check();globals()[sys.argv[1]]()
 except BaseException as e:save('EXECUTION_FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
