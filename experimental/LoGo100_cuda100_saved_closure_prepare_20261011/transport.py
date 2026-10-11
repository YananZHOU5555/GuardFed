from pathlib import Path,PurePosixPath
import argparse,base64,datetime,hashlib,json,subprocess,sys,tarfile
import metadata_gate as g
R=Path.cwd();H=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'tmp'))
from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);p.add_argument('--source-seal-sha256',required=True);a=p.parse_args()
assert __debug__ and sha(a.inputs)==a.inputs_sha256
inp=json.loads(a.inputs.read_bytes());g.input_gate(inp);assert inp['root_authorized_transport'] is True
assert sha(H/'FILES_SHA256.json')==a.source_seal_sha256
for n,item in json.loads((H/'FILES_SHA256.json').read_bytes())['files'].items():assert Path(n).name==n and sha(H/n)==item['sha256']
review=Path(inp['source_review']['path']);assert sha(review)==inp['source_review']['sha256'] and json.loads(review.read_bytes())['source_adoptable'] is True and json.loads(review.read_bytes())['source_seal_sha256']==a.source_seal_sha256
assert sha(R/'tmp/guardfed_local_storage.py')=='2482f2f46243abafd33ea2fadc94d555c0d80327fefe19caea3eb8f3f6bb3d63'
volume=check_bulk_storage(250*1024**2);F=Path('F:/YananResearchStorage/GuardFed/uniform_cuda_LoGo100_20261011/all100_delta98_saved_attempt001');assert not F.exists();F.mkdir(parents=True)
payload={'inputs':inp,'refs':json.loads((H/'EXACT100_REFS.json').read_bytes()),'gate':base64.b64encode((H/'metadata_gate.py').read_bytes()).decode(),'remote':base64.b64encode((H/'remote_export.py').read_bytes()).decode()}
code="import base64,json,sys,types\nx=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\ng=types.ModuleType('metadata_gate');g.__file__='/source-only-bound/metadata_gate.py';sys.modules['metadata_gate']=g\nexec(compile(base64.b64decode(x['gate']),g.__file__,'exec'),g.__dict__)\nns={'__file__':'/source-only-bound/remote_export.py'};exec(compile(base64.b64decode(x['remote']),ns['__file__'],'exec'),ns);ns['main'](x['inputs'],x['refs'])\n"
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 110 nice -n 10 ionice -c 3 python3 -B -']
p=subprocess.run(cmd,input=code.encode(),capture_output=True,timeout=300)
(H/'LOGO100_DELTA98_BACKUP.stdout').write_bytes(p.stdout);(H/'LOGO100_DELTA98_BACKUP.stderr').write_bytes(p.stderr)
put(H/'LOGO100_DELTA98_BACKUP_COMMAND.json',dict(command=cmd,payload_sha256=hashlib.sha256(code.encode()).hexdigest(),exit_code=p.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(H/'LOGO100_DELTA98_BACKUP.stdout'),stderr_sha256=sha(H/'LOGO100_DELTA98_BACKUP.stderr')))
assert p.returncode==0,p.stderr.decode(errors='replace');remote=json.loads(p.stdout)
check_bulk_storage(remote['archive_bytes']*4)
archive=F/'LOGO100_DELTA98.tar.gz';cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55:'+remote['archive_path'],str(archive)]
p=subprocess.run(cmd,capture_output=True,timeout=120)
(H/'LOGO100_DELTA98_SCP.stdout').write_bytes(p.stdout);(H/'LOGO100_DELTA98_SCP.stderr').write_bytes(p.stderr)
put(H/'LOGO100_DELTA98_SCP_COMMAND.json',dict(command=cmd,exit_code=p.returncode,stdout_sha256=sha(H/'LOGO100_DELTA98_SCP.stdout'),stderr_sha256=sha(H/'LOGO100_DELTA98_SCP.stderr')))
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
put(F/'TRANSPORT_VERIFICATION.json',proof);put(H/'LOGO100_DELTA98_TRANSPORT_VERIFICATION.json',proof)
print(json.dumps(dict(status=proof['status'],path=str(H/'LOGO100_DELTA98_TRANSPORT_VERIFICATION.json'),sha256=sha(H/'LOGO100_DELTA98_TRANSPORT_VERIFICATION.json'),archive_sha256=sha(archive),members=len(all_members),bytes=archive.stat().st_size)))
