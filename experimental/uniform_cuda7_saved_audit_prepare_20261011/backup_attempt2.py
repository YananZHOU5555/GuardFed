from pathlib import Path,PurePosixPath
import datetime,hashlib,json,subprocess,sys,tarfile
R=Path.cwd();H=R/'tmp/uniform_cuda7_saved_audit_prepare_20261011';sys.path.insert(0,str(R/'tmp'))
from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
volume=check_bulk_storage(50*1024**2);F=Path('F:/YananResearchStorage/GuardFed/uniform_cuda500_20261011/canary7_attempt2');assert not F.exists();F.mkdir(parents=True)
code=r'''
from pathlib import Path
import datetime,hashlib,json,tarfile,os
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert os.sched_getaffinity(0)=={110}
for proc in Path('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  for t in (proc/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   assert not (len(aff)<=16 and 110 in aff),('CPU110 occupied',proc.name,t.name)
 except (FileNotFoundError,ProcessLookupError):continue
B=Path('/workspace/guardfed_checks/uniform_cuda_formal_20261011');M=Path('/workspace/guardfed_checks/mechanism_cuda_formal_20261011')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
ids=read(B/'STARTED.json')['canaries'];assert ids==['GuardFed-AD2+_IID_Benign_seed91001','minus_U_IID_Benign_seed91001','minus_C_IID_Benign_seed91001','minus_A_IID_Benign_seed91001','minus_F_IID_Benign_seed91001','GuardFed-AD2+_non-IID_Sp-DFA_seed91004','minus_F_non-IID_Sp-DFA_seed91004']
assert sha(B/'DEPLOYMENT_SEAL.json')=='16632d53e61e5ca85086014ae192631a08be0c26c26e3708abf3632806290d20'
for rel,pin in read(B/'DEPLOYMENT_SEAL.json')['files'].items():assert sha(B/rel)==pin['sha256']
assert sha(B/'cohort.json')=='121cf15e0d6b5d0f956cf46d0432da4f7839b655bd62bb71430660613f105447'
files={}
for name in ['cohort.json','author.json','formal_author.json','DEPLOYMENT_SEAL.json','ROOT_QUEUE_AUTH.json','STARTED.json','ROOT_SOURCE_REVIEW.json','ROOT_FRESH_PREFLIGHT.json','MATERIALIZATION.json','dispatch.py']:
 files['base/'+name]=B/name
for root in ['common','scope']:
 for p in (B/root).iterdir():
  if p.is_file():files['base/'+root+'/'+p.name]=p
for identity in ids:
 a=read(B/'tasks'/identity/'ACCEPTED.json');assert a['id']==identity and a['acceptance']['native_comparison']['accepted'] and a['acceptance']['native_comparison']['max_abs_difference']<=1e-12
 full=identity.startswith('GuardFed-AD2+_');out=(B if full else M)/'runs'/identity
 assert sha(out/'receipt.json')==a['receipt']['sha256'] and sha(out/'validation_predictions.npz')==a['arrays']['sha256']
 assert a['cohort']=={'path':str(B/'cohort.json'),'sha256':sha(B/'cohort.json')} and sha(Path(a['approval']['path']))==a['approval']['sha256']
 receipt=read(out/'receipt.json');assert receipt['id']==identity and receipt['native_comparison']==a['acceptance']['native_comparison'] and receipt['views']==a['acceptance']['views'] and receipt['status']=='NATIVE_VALID_REPLAY_PASS'
 assert not any((out.parent/(identity+suffix)).exists() for suffix in ['.formal_failure.json','.bridge_failure.json'])
 for p in (B/'tasks'/identity).iterdir():
  if p.is_file():files['base/tasks/'+identity+'/'+p.name]=p
 for name in ['receipt.json','validation_predictions.npz',('formal_binding.json' if full else 'bridge_receipt.json')]:files[('base' if full else 'mechanism')+'/runs/'+identity+'/'+name]=out/name
 files['base/logs/'+identity+'.log']=B/'logs'/(identity+'.log')
# Preserve a later failure as separate evidence, never erase or promote it.
for rel in ['QUEUE_FAILURE.json','logs/GuardFed-AD2+_IID_Benign_seed91005.log','runs/GuardFed-AD2+_IID_Benign_seed91005.formal_failure.json','runs/GuardFed-AD2+_IID_Benign_seed91005/receipt.json']:
 if (B/rel).is_file():files['base/'+rel]=B/rel
files['source/metadata.npz']=Path('/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/metadata.npz')
members={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}
index=B/'CANARY7_ATTEMPT2_MEMBERS.json';assert not index.exists();index.write_text(json.dumps({'files':members,'ids':ids},indent=2)+chr(10))
archive=B/'CANARY7_ATTEMPT2.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for n,p in files.items():tar.add(p,arcname=n,recursive=False)
 tar.add(index,arcname='ROOT_MEMBERS.json',recursive=False)
assert all(sha(p)==members[n]['sha256'] for n,p in files.items())
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_path=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,archive_members=len(files)+1,payload_members=len(files),member_index_sha256=sha(index),ids=ids)))
'''
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 110 nice -n 10 ionice -c 3 python3 -B -']
p=subprocess.run(cmd,input=code.encode(),capture_output=True,timeout=90)
(H/'CANARY7_ATTEMPT2_BACKUP.stdout').write_bytes(p.stdout);(H/'CANARY7_ATTEMPT2_BACKUP.stderr').write_bytes(p.stderr)
put(H/'CANARY7_ATTEMPT2_BACKUP_COMMAND.json',dict(command=cmd,payload_sha256=hashlib.sha256(code.encode()).hexdigest(),exit_code=p.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(H/'CANARY7_ATTEMPT2_BACKUP.stdout'),stderr_sha256=sha(H/'CANARY7_ATTEMPT2_BACKUP.stderr')))
assert p.returncode==0,p.stderr.decode(errors='replace');remote=json.loads(p.stdout)
check_bulk_storage(remote['archive_bytes']*4)
archive=F/'CANARY7_ATTEMPT2.tar.gz';cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55:'+remote['archive_path'],str(archive)]
p=subprocess.run(cmd,capture_output=True,timeout=120)
(H/'CANARY7_ATTEMPT2_SCP.stdout').write_bytes(p.stdout);(H/'CANARY7_ATTEMPT2_SCP.stderr').write_bytes(p.stderr)
put(H/'CANARY7_ATTEMPT2_SCP_COMMAND.json',dict(command=cmd,exit_code=p.returncode,stdout_sha256=sha(H/'CANARY7_ATTEMPT2_SCP.stdout'),stderr_sha256=sha(H/'CANARY7_ATTEMPT2_SCP.stderr')))
assert p.returncode==0,p.stderr.decode(errors='replace');assert sha(archive)==remote['archive_sha256'] and archive.stat().st_size==remote['archive_bytes']
dest=F/'verified';dest.mkdir();all_members={}
with tarfile.open(archive,'r:gz') as tar:
 ms=tar.getmembers();assert len(ms)==len({m.name for m in ms})==remote['archive_members'] and all(m.isfile() for m in ms)
 raw=tar.extractfile('ROOT_MEMBERS.json').read();assert hashlib.sha256(raw).hexdigest()==remote['member_index_sha256'];index=json.loads(raw)
 assert {m.name for m in ms}==set(index['files'])|{'ROOT_MEMBERS.json'}
 for m in ms:
  rel=PurePosixPath(m.name);assert not rel.is_absolute() and '..' not in rel.parts and ':' not in m.name and '\\' not in m.name
  target=dest.joinpath(*rel.parts);assert target.resolve().is_relative_to(dest.resolve());data=tar.extractfile(m).read();digest=hashlib.sha256(data).hexdigest()
  if m.name!='ROOT_MEMBERS.json':assert index['files'][m.name]==dict(sha256=digest,bytes=len(data))
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(data)
  assert sha(target)==digest;all_members[m.name]=dict(sha256=digest,bytes=len(data))
proof=dict(status='PASS_ARCHIVE_AND_ALL_MEMBER_SHA',archive_path=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,verified_extract=str(dest),extract_root=str(dest),archive_members=len(all_members),payload_members=len(index['files']),member_index_sha256=remote['member_index_sha256'],members=index['files'],all_members_verified=True,actual_F_volume=volume,remote=remote,models_downloaded=0,images_downloaded=0,new_scientific_acceptance=0)
put(F/'TRANSPORT_VERIFICATION.json',proof);put(H/'CANARY7_ATTEMPT2_TRANSPORT_VERIFICATION.json',proof)
print(json.dumps(dict(status=proof['status'],path=str(H/'CANARY7_ATTEMPT2_TRANSPORT_VERIFICATION.json'),sha256=sha(H/'CANARY7_ATTEMPT2_TRANSPORT_VERIFICATION.json'),archive_sha256=sha(archive),members=len(all_members),bytes=archive.stat().st_size)))
