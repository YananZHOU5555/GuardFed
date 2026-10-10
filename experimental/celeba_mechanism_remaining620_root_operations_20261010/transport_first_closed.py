"""One actual reviewed first transport; weights remain in the native archive chain."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,tarfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tmp'))
from guardfed_local_storage import STORAGE_ROOT,check_bulk_storage
SOURCE=ROOT/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010'
REMOTE='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010'
QUEUE='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010'
PY='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEAL='1a021b707575292c33959c19fcfa2fa1ee8c7f285d20576c562e4843d1488fb3'
assert sha(SOURCE/'FILES_SHA256.json')==SEAL
independent=ROOT/'tmp/celeba_remaining620_transport_independent_review_20261010/REVIEW.json'
assert sha(independent)=='d614a28050774543597eabc98e38de30c74db0fbee0d07ecdd2d91399fc0b390'
assert not (HERE/'FIRST_TRANSPORT_COMMAND.json').exists()
actual=json.loads((HERE/'ROOT_STARTUP_REVIEW.json').read_bytes())
assert actual['remote_strict_closed']>=1 and actual['accepted_offserver']==0
identity=actual['closed'][0]['id']
files=json.loads((SOURCE/'FILES_SHA256.json').read_bytes())['files']
assert all(sha(SOURCE/n)==v['sha256'] for n,v in files.items())
storage=check_bulk_storage(sum(v['bytes'] for v in files.values())+1048576)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');tag='first_'+stamp
archive=STORAGE_ROOT/'deployments/celeba_remaining620_transport_20261010'/(stamp+'.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'x:gz') as t:
 for n in list(files)+['FILES_SHA256.json']:t.add(SOURCE/n,arcname=n,recursive=False)
remote_archive='/workspace/guardfed_checks/remaining620_transport_source_'+stamp+'.tar.gz'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
upload=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350',str(archive),'root@89.22.197.55:'+remote_archive],capture_output=True,timeout=90)
(HERE/'transport_upload.stdout').write_bytes(upload.stdout);(HERE/'transport_upload.stderr').write_bytes(upload.stderr);upload.check_returncode()
code=r'''from pathlib import Path
import datetime,hashlib,json,os,subprocess,tarfile
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
p=Path(ARCHIVE);base=Path(REMOTE)
assert hashlib.sha256(p.read_bytes()).hexdigest()==ARCHIVE_SHA and not base.exists()
owners=[]
for proc in Path('/proc').glob('[0-9]*/stat'):
 try:
  if proc.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  for t in (proc.parent/'task').iterdir():
   cpus=os.sched_getaffinity(int(t.name))
   if len(cpus)<=16 and 110 in cpus:owners.append(dict(pid=int(proc.parent.name),tid=int(t.name),cpus=sorted(cpus)))
 except (OSError,ValueError):pass
assert not owners,owners
with tarfile.open(p) as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
 base.mkdir();t.extractall(base,filter='data')
cmd=['taskset','-c','110','nice','-n','10','ionice','-c','3',PY,'-B',str(base/'transport.py'),'export',
 '--source',QUEUE,'--source-seal','a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03','--transport-seal',SEAL,
 '--runtime',QUEUE+'/attempt1','--review',QUEUE+'/ROOT_APPROVED.json','--review-sha256',APPROVAL_SHA,'--tag',TAG,'--ids',IDENTITY]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
receipt=json.loads(r.stdout) if r.returncode==0 else None
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
 receipt_sha256=hashlib.sha256((base/'exports'/TAG/'backup_receipt.json').read_bytes()).hexdigest() if receipt else None,
 archive_bytes=Path(receipt['archive']).stat().st_size if receipt else None,restricted_CPU110_owners=owners,argv=cmd)))
'''
for k,v in [('ARCHIVE_SHA',sha(archive)),('APPROVAL_SHA',actual['root_approval_sha256']),('ARCHIVE',remote_archive),('REMOTE',REMOTE),('QUEUE',QUEUE),('PY',PY),('SEAL',SEAL),('TAG',tag),('IDENTITY',identity)]:
 code=code.replace(k,repr(v))
r=subprocess.run(ssh+['python -B -'],input=code.encode(),capture_output=True,timeout=270)
(HERE/'first_transport.stdout').write_bytes(r.stdout);(HERE/'first_transport.stderr').write_bytes(r.stderr)
with (HERE/'FIRST_TRANSPORT_EXIT.json').open('x') as f:f.write(json.dumps(dict(returncode=r.returncode,tag=tag,automatic_retry=False),indent=2)+'\n')
r.check_returncode();d=json.loads(r.stdout)
with (HERE/'FIRST_TRANSPORT_COMMAND.json').open('x') as f:f.write(json.dumps(dict(**d,storage_preflight=storage,source_archive=str(archive),source_archive_sha256=sha(archive)),indent=2)+'\n')
assert d['returncode']==0,'Preserve actual transport failure; no blind retry'
receipt=json.loads(d['stdout']);bulk=STORAGE_ROOT/'remaining620'/tag
fresh=check_bulk_storage(d['archive_bytes']+1048576);bulk.mkdir(parents=True,exist_ok=False)
small=HERE/'first_transport';small.mkdir(exist_ok=False)
for remote,local in [(receipt['archive'],bulk/'incremental_valid_three_views.tar.gz'),(REMOTE+'/exports/'+tag+'/backup_receipt.json',small/'backup_receipt.json')]:
 fetch=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+remote,str(local)],capture_output=True,timeout=120)
 fetch.check_returncode()
assert sha(small/'backup_receipt.json')==d['receipt_sha256'] and sha(bulk/'incremental_valid_three_views.tar.gz')==receipt['archive_sha256']
verify=[sys.executable,'-B',str(SOURCE/'transport.py'),'verify','--source',str(ROOT/'tmp/celeba_mechanism_remaining_evaluation_v2_20261010'),
 '--source-seal','a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03','--transport-seal',SEAL,
 '--archive',str(bulk/'incremental_valid_three_views.tar.gz'),'--receipt',str(small/'backup_receipt.json'),'--receipt-sha256',d['receipt_sha256'],
 '--cache',str(ROOT/'tmp/celeba_final_valid_replay_20261009/verification_inputs/original_valid_cache.npz'),'--out',str(bulk/'verification')]
v=subprocess.run(verify,capture_output=True,timeout=120)
(small/'verify.stdout').write_bytes(v.stdout);(small/'verify.stderr').write_bytes(v.stderr)
with (small/'VERIFY_EXIT.json').open('x') as f:f.write(json.dumps(dict(returncode=v.returncode,automatic_retry=False),indent=2)+'\n')
v.check_returncode()
proof=bulk/'verification/OFFSERVER_TRANSPORT_VERIFICATION.json'
with (HERE/'FIRST_TRANSPORT_LOCATION.json').open('x') as f:
 f.write(json.dumps(dict(tag=tag,id=identity,archive=str(bulk/'incremental_valid_three_views.tar.gz'),archive_sha256=receipt['archive_sha256'],
  receipt=str(small/'backup_receipt.json'),receipt_sha256=d['receipt_sha256'],proof=str(proof),proof_sha256=sha(proof),
  fresh_F_preflight=fresh,source_seal_sha256=SEAL,accepted_offserver=0,root_adoption_pending=True),indent=2)+'\n')
print(json.dumps(dict(tag=tag,id=identity,archive_bytes=d['archive_bytes'],archive_sha256=receipt['archive_sha256'],proof_sha256=sha(proof),accepted_offserver=0,root_adoption_pending=True)))
