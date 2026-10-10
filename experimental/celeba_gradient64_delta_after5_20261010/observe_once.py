"""One original read-only v2a snapshot, rooted at actual accepted5; no source/queue mutation."""
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,subprocess,sys,traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
def run(n,command,code=None,timeout=60):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=subprocess.run(command,input=code.encode() if code else None,capture_output=True,timeout=timeout)
 for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]:
  with (H/(n+'.'+suffix)).open('xb') as f:f.write(data)
 save(n+'_COMMAND.json',dict(command=command,start=start,finish=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=p.returncode,source_sha256=hashlib.sha256(code.encode()).hexdigest() if code else None))
 p.check_returncode();return p.stdout
if __name__=='__main__':
 try:
  prior_dir=R/'tmp/celeba_gradient64_delta_after1_20261010';root=prior_dir/'ROOT_ADOPTION_REVIEW.json';previous=prior_dir/'OFFSERVER_ACCEPTANCE.json'
  prior=read(root);off=read(previous)
  assert prior['status']=='ROOT_GRADIENT64_EXACT4_ORIGINAL_STRICT_OFFSERVER_ADOPTED' and prior['accepted_total']==5
  assert sha(previous)==prior['offserver_sha256']=='b3250c5986728a734dfee38eceef63ab160e65222dc875e8a1f310fc6848e6dd'
  assert prior['accepted_ids']==off['accepted_job_ids'] and off['accepted_total']==5 and len(set(prior['accepted_ids']))==5
  assert sha(R/prior['handoff_path'])==prior['handoff_sha256']
  assert sha(Path(prior['archive_path']).parent/'backup_receipt.json')==read(prior_dir/'ROOT_READY_CHAIN_LINK.json')['receipt_sha256']
  spec=importlib.util.spec_from_file_location('storage_guard',R/'tmp/guardfed_local_storage.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
  save('F_VOLUME_INITIAL.json',guard.check_bulk_storage(0))
  guide=run('GUIDE',SSH+['cat /etc/vast-agents-guide.md'])
  assert hashlib.sha256(guide).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
  owner="""from pathlib import Path
import os,json
busy=[];helpers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if (p/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if any('celeba_gradient64_delta' in x and 'collect_delta.py' in x for x in argv):helpers.append(dict(pid=int(p.name),argv=argv))
  for t in (p/'task').iterdir():
   try:a=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(a)<=16 and 106 in a:busy.append(dict(pid=int(p.name),tid=int(t.name),cpus=sorted(a)))
 except (FileNotFoundError,ProcessLookupError):continue
print(json.dumps(dict(CPU106_free=not busy,narrow_owners=busy,existing_collectors=helpers)))
"""
  own=json.loads(run('OWNER',SSH+['python -B -'],owner));save('OWNER.json',own)
  assert own['CPU106_free'] and not own['existing_collectors'],'CPU106/collector conflict'
  original_observer=R/'tmp/celeba_gradient_screen64_v2_root_operations_20261010/observe_attempt2.py'
  assert sha(original_observer)=='baec9bd8c845fbe490e21fb9175035031eb71b537f8a770e7ae0b732be6c8b4a'
  node=next(n for n in ast.parse(original_observer.read_text('utf8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
  snapshot=json.loads(run('SNAPSHOT',SSH+['python -B -'],ast.literal_eval(node.value)));save('SNAPSHOT.json',snapshot)
  assert snapshot['service']['returncode']==0 and 'guardfed_celeba_gradient_screen64_v2a' in snapshot['service']['stdout'] and 'RUNNING' in snapshot['service']['stdout']
  assert not snapshot['failure_paths'] and snapshot['queue']['test'] is False
  pkg=R/'tmp/celeba_gradient_screen64_v2_20261010'
  assert sha(pkg/'FILES_SHA256.json')=='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
  for n,pin in read(pkg/'FILES_SHA256.json')['files'].items():assert sha(pkg/n)==pin
  manifest=read(pkg/'jobs/manifest.json');allids=[x['id'] for x in manifest['jobs']];completed=snapshot['queue']['strict_server_completed_ids']
  assert len(allids)==len(set(allids))==64 and len(completed)==len(set(completed)) and set(completed)<=set(allids)
  assert set(prior['accepted_ids'])<=set(completed)
  ids=[identity for identity in allids if identity in completed and identity not in prior['accepted_ids']]
  for identity in ids:
   row=next(x for x in snapshot['rows'] if x['id']==identity)
   assert row['result'] and row['accepted'] and row['progress']['round']==row['diagnostics_rounds']==70
  save('AUTHORIZED_SNAPSHOT.json',dict(status='PARENT_AUTHORIZED_FIXED_ONE_SNAPSHOT_TERMINAL_MINUS_ACCEPTED5_NOT_ADOPTION',snapshot_sha256=sha(H/'SNAPSHOT.json'),snapshot_unix=snapshot['at_unix'],accepted_prior_ids=prior['accepted_ids'],accepted_prior_count=5,authorized_ids=ids,terminal_count=len(completed),manifest_sha256=sha(pkg/'jobs/manifest.json'),source_seal_sha256=sha(pkg/'FILES_SHA256.json'),prior_root_path=root.relative_to(R).as_posix(),prior_root_sha256=sha(root),prior_offserver_path=previous.relative_to(R).as_posix(),prior_offserver_sha256=sha(previous),future_completions_excluded=True,parent_authorization='Explicit /root NEW_TASK: collect one latest closed snapshot delta after accepted5; no shared adoption or source change'))
  for name,path in [('PREVIOUS_ROOT_ADOPTION.json',root),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',previous)]:
   with (H/name).open('xb') as f:f.write(path.read_bytes())
  print(json.dumps(dict(terminal=len(completed),prior=5,new_ids=ids,CPU106_free=True,accepted_offserver=0)),flush=True)
 except BaseException as error:
  save('OBSERVATION_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
