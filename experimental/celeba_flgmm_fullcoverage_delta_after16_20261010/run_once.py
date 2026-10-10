"""One fixed FL96 delta; reuse sealed strict/archive/offserver/tensor readers."""
from pathlib import Path
import ast, base64, datetime, hashlib, json, subprocess, sys, traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
B=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
O=R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010'
REMOTE='/workspace/guardfed_checks/'+H.name
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
PREFIX='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 107 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -'
IDS=[f'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in (91008,91009)]
PACKAGE='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
PREVIOUS='8ca5e7afa10527fd01607082b0a461f0a226a4941091f182ba8fd966cd09197d'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def run(name,cmd,code=None,timeout=600):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=subprocess.run(cmd,input=code.encode() if code is not None else None,capture_output=True,timeout=timeout)
 for suffix,data in [('STDOUT.txt',p.stdout),('STDERR.txt',p.stderr)]:
  with (H/(name+'_'+suffix)).open('xb') as f:f.write(data)
 save(name+'_COMMAND.json',dict(command=cmd,exit_code=p.returncode,start=start,finish=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(code.encode()).hexdigest() if code else None))
 p.check_returncode();return p.stdout
def prepare():
 latest=read(B/'LATEST_BACKUP.json');assert latest['accepted_total']==16
 assert sha(R/latest['root_adoption_path'])==latest['root_adoption_sha256']=='9d587b8261c2d1340339e1905db133bc113aeff7b7150db95cbe260103c77617'
 assert sha(R/latest['next_collector_previous_path'])==latest['next_collector_previous_sha256']==PREVIOUS
 assert sha(B/'FILES_SHA256.json')=='f6de59a56de25a8d316cf7e05c44eafc9751041757e4dc61c88f7bccfd988472'
 for n,pin in read(B/'FILES_SHA256.json')['files'].items():assert sha(B/n)==pin['sha256']
 snap=R/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/observation_20261010T021408184632Z/SNAPSHOT.json'
 assert sha(snap)=='4b5f171949b35c086e0e6c453b3bbc9508a0836be228cea8a8232a5f0e7a1fbf'
 s=read(snap);prior=read(R/latest['next_collector_previous_path']);active={x['id'] for x in s['queue']['active']}
 assert s['source_members_match'] and not s['changed_source_members'] and not s['failure_paths'] and not s['queue']['failed']
 terminal=[x['id'] for x in s['rows'] if x['kind']=='new' and x['result_present'] and x['acceptance_present'] and x['progress']['round']==70 and x['id'] not in active]
 assert len(terminal)==18 and len(prior['accepted_job_ids'])==len(set(prior['accepted_job_ids']))==16
 assert set(prior['accepted_job_ids'])<=set(terminal) and [x for x in terminal if x not in prior['accepted_job_ids']]==IDS
 for n,p in [('PREVIOUS_LATEST.json',B/'LATEST_BACKUP.json'),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',R/latest['next_collector_previous_path']),('SNAPSHOT.json',snap)]:
  with (H/n).open('xb') as f:f.write(p.read_bytes())
 save('AUTHORIZED_SNAPSHOT.json',dict(status='FIXED_ONE_SNAPSHOT_EXACT_DELTA_NOT_ACCEPTANCE',snapshot_sha256=sha(snap),snapshot_utc=s['utc'],prior_count=16,authorized_ids=IDS,source_hashes=s['source_hashes'],prior_root_path=latest['root_adoption_path'],prior_root_sha256=latest['root_adoption_sha256'],prior_offserver_sha256=PREVIOUS,no_future_terminal_ids_allowed=True,root_adoption_required=True))
 old=O/'collect_delta.py';assert sha(old)=='575ecbd7f948ad66d58d28cb336100da1efc965559ee05ba500f3ef259d98c20'
 oldline="    authorized_ids="+repr([f'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in (91006,91007)])+'\n'
 newline='    authorized_ids='+repr(IDS)+'\n';text=old.read_text('utf8');assert text.count(oldline)==1
 effective=text.replace(oldline,newline)
 def loop(t):
  node=next(x for x in ast.parse(t).body if isinstance(x,ast.FunctionDef) and x.name=='execute')
  n=next(x for x in node.body if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='identity')
  return ast.get_source_segment(t,n)
 assert loop(effective)==loop(text)
 wrapper="from pathlib import Path\nimport hashlib\np=Path(%r)\ns=p.read_text('utf8')\nassert hashlib.sha256(p.read_bytes()).hexdigest()==%r\na=%r\nb=%r\nassert s.count(a)==1\nexec(compile(s.replace(a,b),__file__,'exec'),globals())\n"%('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py',sha(old),oldline,newline)
 with (H/'collect_delta.py').open('x',encoding='utf8',newline='\n') as f:f.write(wrapper)
 save('SOURCE_REUSE.json',dict(parent_collector_path=old.relative_to(R).as_posix(),parent_collector_sha256=sha(old),thin_entry_sha256=sha(H/'collect_delta.py'),effective_collector_sha256=hashlib.sha256(effective.encode()).hexdigest(),sole_effective_change='authorized_ids91006/91007 -> fixed authorized91008/91009',per_ID_scientific_loop_bytes_exact=True,original_verifier_path=(B/'verify_delta_offserver.py').relative_to(R).as_posix(),original_verifier_sha256=sha(B/'verify_delta_offserver.py'),source_seal_sha256=sha(B/'FILES_SHA256.json'),scientific_body_unchanged=True))
 print('Prepared exact18-minus16:',IDS,flush=True)
def collect():
 raw=run('GUIDE',SSH+['cat /etc/vast-agents-guide.md'],timeout=30)
 with (H/'SERVER_GUIDE.md').open('xb') as f:f.write(raw)
 assert sha(H/'SERVER_GUIDE.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
 owner="""from pathlib import Path
import os,json,hashlib
busy=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  for t in (p/'task').iterdir():
   try:a=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(a)<=16 and 107 in a:busy.append(dict(pid=int(p.name),tid=int(t.name),affinity=sorted(a)))
 except (FileNotFoundError,ProcessLookupError):continue
print(json.dumps(dict(CPU107_free=not busy,busy=busy,parent_collector_sha256=hashlib.sha256(Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py').read_bytes()).hexdigest())))
"""
 ownership=json.loads(run('OWNER',SSH+['python -B -'],owner,30));save('OWNER.json',ownership)
 if not ownership['CPU107_free']:print('CPU107 occupied; no collector dispatched',flush=True);return
 assert ownership['parent_collector_sha256']==read(H/'SOURCE_REUSE.json')['parent_collector_sha256']
 preflight=(O/'preflight.py').read_text('utf8');assert sha(O/'preflight.py')=='7fc3d7a78ad71a3c9b48522020ad5d773aa9303791fc6ff9fa080a25648340d2'
 assert preflight.count('(91006,91007)')==1;preflight=preflight.replace('(91006,91007)','(91008,91009)')
 save('PREFLIGHT.json',json.loads(run('PREFLIGHT',SSH+[PREFIX],preflight)))
 print('Guide/source/data/CPU107 all-thread preflight PASS; invoking one collector.',flush=True)
 payload={n:base64.b64encode(p.read_bytes()).decode() for n,p in [('collect_delta.py',H/'collect_delta.py'),('verify_delta_offserver.py',B/'verify_delta_offserver.py'),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')]}
 pins={n:hashlib.sha256(base64.b64decode(data)).hexdigest() for n,data in payload.items()}
 code="""from pathlib import Path
import base64,hashlib,json,subprocess,sys
b=Path(%r);assert not b.exists() and not b.is_symlink() and b.parent.resolve()==b.parent
payload=%r;pins=%r;b.mkdir()
for n,v in payload.items():
 data=base64.b64decode(v);assert hashlib.sha256(data).hexdigest()==pins[n]
 with (b/n).open('xb') as f:f.write(data)
cmd=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','taskset','-c','107','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(b/'collect_delta.py'),'--out',str(b/'batch'),'--cpu','107','--source-sha256',pins['collect_delta.py'],'--previous',str(b/'PREVIOUS_OFFSERVER_ACCEPTANCE.json'),'--previous-sha256',pins['PREVIOUS_OFFSERVER_ACCEPTANCE.json']]
r=subprocess.run(cmd,capture_output=True)
(b/'COLLECT_STDOUT.txt').write_bytes(r.stdout);(b/'COLLECT_STDERR.txt').write_bytes(r.stderr)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
"""%(REMOTE,payload,pins)
 receipt=json.loads(run('COLLECT',SSH+['python -B -'],code));save('SERVER_COLLECTOR_RECEIPT.json',receipt)
 assert receipt['accepted_new_ids']==IDS and receipt['accepted_total']==18 and receipt['accepted_new']==2
 print('Original server strict/archive PASS:',receipt['archive_sha256'],flush=True)
def verify():
 names=['accepted_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json']
 code="from pathlib import Path\nimport hashlib,json\nb=Path(%r)\nprint(json.dumps({n:dict(sha256=hashlib.sha256((b/n).read_bytes()).hexdigest(),size=(b/n).stat().st_size) for n in %r}))\n"%(REMOTE+'/batch',names)
 pins=json.loads(run('SERVER_SHA',SSH+['python -B -'],code,30));save('SERVER_TRANSFER_SHA256.json',pins)
 batch=H/'batch';batch.mkdir()
 run('SCP',['scp','-q','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15']+['root@89.22.197.55:'+REMOTE+'/batch/'+n for n in names]+[str(batch)],timeout=120)
 for n,pin in pins.items():assert sha(batch/n)==pin['sha256'] and (batch/n).stat().st_size==pin['size']
 raw=run('VERIFY',[sys.executable,'-B',str(B/'verify_delta_offserver.py'),'--batch',str(batch),'--release',str(R/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/attempt_20261009T200518912319Z/verified_manual_v2/stage'),'--receipt-sha256',sha(batch/'BACKUP_SHA256.json')],timeout=120)
 print(raw.decode(),flush=True)
 tensor=(O/'check_saved_tensors.py').read_text('utf8');assert sha(O/'check_saved_tensors.py')=='877ab0396a9814fd8c7cb41f1fce03ebdf9b84a04b410edeb5b8663e882e3b8d'
 assert tensor.count("proof['accepted_total']==16")==1
 exec(compile(tensor.replace("proof['accepted_total']==16","proof['accepted_total']==18"),str(O/'check_saved_tensors.py'),'exec'),dict(__file__=str(H/'check_saved_tensors_reused.py'),__name__='__main__'))
 closed=(O/'close_observation.py').read_text('utf8').replace('celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py',H.name+'/collect_delta.py')
 save('COLLECTOR_CLOSED.json',json.loads(run('CLOSED',SSH+['python -B -'],closed,30)))
 print('Original local strict/member/tensor checks PASS; CPU107 released.',flush=True)
def main():
 phase=sys.argv[1];assert phase in ('prepare','collect','verify')
 {'prepare':prepare,'collect':collect,'verify':verify}[phase]()
if __name__=='__main__':
 try:main()
 except BaseException as error:
  if not isinstance(error,SystemExit):save('FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
  raise
