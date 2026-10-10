"""One read-only gradient64 snapshot; no shared observer/latest write."""
from pathlib import Path
import ast, datetime, hashlib, json, subprocess, sys, traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; R=H.parents[1]
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def run(n,command,code=None,timeout=60):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 p=subprocess.run(command,input=code.encode() if code else None,capture_output=True,timeout=timeout)
 for suffix,b in [('stdout',p.stdout),('stderr',p.stderr)]:
  with (H/(n+'.'+suffix)).open('xb') as f:f.write(b)
 save(n+'_COMMAND.json',dict(command=command,start=start,finish=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=p.returncode,source_sha256=hashlib.sha256(code.encode()).hexdigest() if code else None))
 p.check_returncode();return p.stdout
if __name__=='__main__':
 try:
  root=R/'tmp/root_adopt_first_closed_20261010/GRADIENT1_ROOT_ADOPTION.json'
  assert sha(root)=='ca4a38076e5080d94069cdabd50a032d84a96a339914761ec568db44a43db978'
  prior=read(root);assert prior['accepted_count']==1
  first=R/'tmp/celeba_gradient64_first_closed_adoption_20261010'
  assert sha(first/'OFFSERVER_ACCEPTANCE.json')==prior['offserver_proof_sha256']=='35089298546d72ef6b6503b63c7887912c8cd88e97a87d297d766245193e7cfa'
  assert sha(first/'backup_receipt.json')==prior['receipt_sha256']
  pkg=R/'tmp/celeba_gradient_screen64_v2_20261010'
  assert sha(pkg/'FILES_SHA256.json')==prior['source_seal_sha256']=='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
  for name,pin in read(pkg/'FILES_SHA256.json')['files'].items():assert sha(pkg/name)==pin
  guide=run('GUIDE',SSH+['cat /etc/vast-agents-guide.md'])
  assert hashlib.sha256(guide).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
  owner="""from pathlib import Path
import os,json
busy=[];helpers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if any('celeba_gradient64_delta' in x and 'collect_delta.py' in x for x in argv):helpers.append(dict(pid=int(p.name),argv=argv))
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 106 in aff:busy.append(dict(pid=int(p.name),tid=int(t.name),affinity=sorted(aff)))
 except (FileNotFoundError,ProcessLookupError):continue
print(json.dumps(dict(CPU106_free=not busy,narrow_owners=busy,existing_collectors=helpers)))
"""
  own=json.loads(run('OWNER',SSH+['python -B -'],owner));save('OWNER.json',own)
  assert own['CPU106_free'] and not own['existing_collectors'],'Owner conflict; no snapshot/dispatch'
  path=R/'tmp/celeba_gradient_screen64_v2_root_operations_20261010/observe_attempt2.py'
  assert sha(path)=='baec9bd8c845fbe490e21fb9175035031eb71b537f8a770e7ae0b732be6c8b4a'
  node=next(n for n in ast.parse(path.read_text('utf8')).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
  s=json.loads(run('SNAPSHOT',SSH+['python -B -'],ast.literal_eval(node.value)))
  save('SNAPSHOT.json',s)
  assert not s['failure_paths'] and s['queue']['failed'] is False
  assert s['service']['returncode']==0 and 'RUNNING' in s['service']['stdout']
  completed=s['queue']['strict_server_completed_ids'];assert len(completed)==len(set(completed))
  manifest=read(pkg/'jobs/manifest.json');allids=[x['id'] for x in manifest['jobs']]
  assert len(allids)==len(set(allids))==64 and set(completed)<=set(allids) and set(prior['accepted_ids'])<=set(completed)
  ids=[x for x in allids if x in completed and x not in prior['accepted_ids']]
  for identity in ids:
   row=next(x for x in s['rows'] if x['id']==identity)
   assert row['result'] and row['accepted'] and row['progress']['round']==row['diagnostics_rounds']==70
  save('AUTHORIZED_SNAPSHOT.json',dict(status='FIXED_ONE_SNAPSHOT_TERMINAL_MINUS_ROOT1_NOT_ACCEPTANCE',snapshot_sha256=sha(H/'SNAPSHOT.json'),snapshot_unix=s['at_unix'],accepted_prior_ids=prior['accepted_ids'],accepted_prior_count=1,authorized_ids=ids,terminal_count=len(completed),manifest_sha256=sha(pkg/'jobs/manifest.json'),source_seal_sha256=sha(pkg/'FILES_SHA256.json'),prior_root_path=root.relative_to(R).as_posix(),prior_root_sha256=sha(root),prior_offserver_path=(first/'OFFSERVER_ACCEPTANCE.json').relative_to(R).as_posix(),prior_offserver_sha256=sha(first/'OFFSERVER_ACCEPTANCE.json'),prior_receipt_sha256=sha(first/'backup_receipt.json'),future_completions_excluded=True,root_source_review_required_before_collect=True))
  print(json.dumps(dict(terminal_count=len(completed),prior=1,exact_new_ids=ids,accepted_offserver=0)))
 except BaseException as e:
  save('OBSERVATION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
