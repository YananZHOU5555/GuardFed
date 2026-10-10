"""Root-reviewed exact snapshot delta transport; no shared chain writes/retries."""
from pathlib import Path
import argparse, base64, datetime, hashlib, importlib.util, json, subprocess, sys, traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
F=Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
REMOTE='/workspace/guardfed_checks/'+H.name
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def run(n,command,code=None,timeout=600):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=subprocess.run(command,input=code.encode() if code else None,capture_output=True,timeout=timeout)
 for suffix,b in [('stdout',p.stdout),('stderr',p.stderr)]:
  with (H/(n+'.'+suffix)).open('xb') as f:f.write(b)
 save(n+'_COMMAND.json',dict(command=command,start=start,finish=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=p.returncode,source_sha256=hashlib.sha256(code.encode()).hexdigest() if code else None))
 p.check_returncode();return p.stdout
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def guard_review(args):
 assert sha(args.review)==args.review_sha256,'Actual root review SHA required'
 review=read(args.review);auth=read(H/'AUTHORIZED_SNAPSHOT.json')
 assert review['status']=='PASS_GRADIENT64_EXACT4_DELTA_SOURCE_READY' and review['source_adoptable'] is True
 assert review['source_files_sha256']==sha(H/'SOURCE_FILES_SHA256.json') and review['authorization_sha256']==sha(H/'AUTHORIZED_SNAPSHOT.json')
 assert review['exact_new_ids']==auth['authorized_ids'] and len(review['exact_new_ids'])==4
 for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():assert sha(H/name)==pin['sha256']
 assert sha(R/auth['prior_root_path'])==auth['prior_root_sha256'] and sha(R/auth['prior_offserver_path'])==auth['prior_offserver_sha256']
 return auth
def collect(args):
 auth=guard_review(args)
 payload={n:base64.b64encode((H/n).read_bytes()).decode() for n in ['collect_delta.py','pinned_collect_one.py','AUTHORIZED_SNAPSHOT.json','PREVIOUS_ROOT_ADOPTION.json','PREVIOUS_OFFSERVER_ACCEPTANCE.json']}
 pins={n:sha(H/n) for n in payload}
 code="""from pathlib import Path
import base64,hashlib,os,json,subprocess,sys
busy=[];helpers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if any('celeba_gradient64_delta' in x and 'collect_delta.py' in x for x in argv):helpers.append(int(p.name))
  for t in (p/'task').iterdir():
   try:a=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(a)<=16 and 106 in a:busy.append(dict(pid=int(p.name),tid=int(t.name),cpus=sorted(a)))
 except (FileNotFoundError,ProcessLookupError):continue
assert not busy and not helpers,dict(busy=busy,helpers=helpers)
b=Path(%r);assert not b.exists() and not b.is_symlink();b.mkdir()
payload=%r;pins=%r
for n,data in payload.items():
 raw=base64.b64decode(data);assert hashlib.sha256(raw).hexdigest()==pins[n]
 with (b/n).open('xb') as f:f.write(raw)
cmd=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','taskset','-c','106','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(b/'collect_delta.py')]
p=subprocess.run(cmd,capture_output=True)
(b/'COLLECT.stdout').write_bytes(p.stdout);(b/'COLLECT.stderr').write_bytes(p.stderr)
sys.stdout.buffer.write(p.stdout);sys.stderr.buffer.write(p.stderr);sys.exit(p.returncode)
"""%(REMOTE,payload,pins)
 receipt=json.loads(run('COLLECT',SSH+['python -B -'],code))
 assert receipt['status']=='REMOTE_STRICT_AND_DELTA_ARCHIVE_PASS' and receipt['receipt']['accepted_new_ids']==auth['authorized_ids']
 save('SERVER_COLLECTOR_RECEIPT.json',receipt)
 print(json.dumps(dict(server_strict='PASS',new=len(auth['authorized_ids']),archive_sha256=receipt['receipt']['archive_sha256'],members=receipt['receipt']['members'])),flush=True)
def download(args):
 guard_review(args);names=['gradient64_delta_after1.tar.gz','backup_receipt.json','ORIGINAL_STRICT.json']
 code="from pathlib import Path\nimport hashlib,json\nb=Path(%r)\nprint(json.dumps({n:dict(sha256=hashlib.sha256((b/n).read_bytes()).hexdigest(),bytes=(b/n).stat().st_size) for n in %r}))\n"%(REMOTE+'/batch',names)
 pins=json.loads(run('REMOTE_ARTIFACTS',SSH+['python -B -'],code,60))
 storage=load(R/'tmp/guardfed_local_storage.py','bulk_storage')
 save('F_VOLUME_BEFORE_DOWNLOAD.json',storage.check_bulk_storage(sum(x['bytes'] for x in pins.values())))
 assert not F.exists() and not F.is_symlink() and F.resolve().drive.upper()=='F:';F.mkdir(parents=True)
 run('SCP',['scp','-q','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15']+['root@89.22.197.55:'+REMOTE+'/batch/'+n for n in names]+[str(F)],timeout=180)
 for n,pin in pins.items():assert sha(F/n)==pin['sha256'] and (F/n).stat().st_size==pin['bytes']
 save('DOWNLOAD.json',dict(archive_path=(F/names[0]).as_posix(),archive_sha256=sha(F/names[0]),receipt_path=(F/'backup_receipt.json').as_posix(),receipt_sha256=sha(F/'backup_receipt.json'),strict_sha256=sha(F/'ORIGINAL_STRICT.json'),files=pins,all_raw_direct_F=True,internal_fallback=False))
def close(args):
 guard_review(args)
 code="""from pathlib import Path
import os,json,subprocess,hashlib
busy=[];helpers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if any(%r in x and 'collect_delta.py' in x for x in argv):helpers.append(dict(pid=int(p.name),argv=argv))
  for t in (p/'task').iterdir():
   try:a=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(a)<=16 and 106 in a:busy.append(dict(pid=int(p.name),tid=int(t.name),cpus=sorted(a)))
 except (FileNotFoundError,ProcessLookupError):continue
assert not busy and not helpers
service=subprocess.run(['supervisorctl','status','guardfed_celeba_gradient_screen64_v2a'],capture_output=True,text=True)
assert service.returncode==0 and 'RUNNING' in service.stdout
print(json.dumps(dict(CPU106_released=True,restricted_owners=busy,collector_processes=helpers,service=service.stdout,guide_sha256=hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest())))
"""%REMOTE
 save('COLLECTOR_RELEASE.json',json.loads(run('RELEASE',SSH+['python -B -'],code,60)))
 storage=load(R/'tmp/guardfed_local_storage.py','bulk_storage_close')
 save('RAW_STORAGE_INDEX.json',dict(status='ACTUAL_F_RAW_MEMBER_SHA_INDEX',raw_storage_root=F.as_posix(),volume=storage.check_bulk_storage(0),files={p.relative_to(F).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(F.rglob('*')) if p.is_file()}))
if __name__=='__main__':
 try:
  p=argparse.ArgumentParser();p.add_argument('phase',choices=['collect','download','verify','close']);p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True);args=p.parse_args()
  if args.phase=='verify':
   guard_review(args);print(run('VERIFY',[sys.executable,'-B',str(H/'verify_delta.py')],timeout=180).decode())
  else:globals()[args.phase](args)
 except BaseException as e:
  save('TRANSPORT_FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
