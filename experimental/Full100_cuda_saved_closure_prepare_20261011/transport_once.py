"""Once-only root-authorized Full100 delta98 transport using the actual seven transport pattern."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,importlib.util,json,subprocess,sys,tarfile
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def need(ok,m):
 if not ok:raise ValueError(m)
def command(label,argv,payload=None,timeout=180):
 result=None;error=None
 try:
  with (H/(label+'.stdout')).open('xb') as out,(H/(label+'.stderr')).open('xb') as err:result=subprocess.run(argv,input=payload,stdout=out,stderr=err,timeout=timeout)
 except BaseException as exc:error=repr(exc);raise
 finally:put(H/(label+'_COMMAND.json'),{'argv':argv,'payload_sha256':hashlib.sha256(payload).hexdigest() if payload else None,'returncode':None if result is None else result.returncode,'exception':error,'stdout_sha256':sha(H/(label+'.stdout')),'stderr_sha256':sha(H/(label+'.stderr')),'automatic_retry':False})
 need(result.returncode==0,'Command failed; preserve old artifacts; no retry')
p=argparse.ArgumentParser();p.add_argument('--root-inputs',type=Path,required=True);p.add_argument('--root-inputs-sha256',required=True);p.add_argument('--source-seal-sha256',required=True);a=p.parse_args()
need(sha(a.root_inputs)==a.root_inputs_sha256 and sha(H/'FILES_SHA256.json')==a.source_seal_sha256,'Unbound inputs/source')
for rel,pin in json.loads((H/'FILES_SHA256.json').read_bytes())['files'].items():need(sha(H/rel)==pin['sha256'],'Source member changed')
inputs=json.loads(a.root_inputs.read_bytes());need(inputs['execution_authorized'] is True,'Root must explicitly authorize transport')
review=Path(inputs['source_review']['path']);need(sha(review)==inputs['source_review']['sha256'],'Wrong independent review')
r=json.loads(review.read_bytes());need(r['source_adoptable'] is True and r['source_seal_sha256']==a.source_seal_sha256,'Review not for this source')
pins=json.loads((H/'SOURCE_PINS.json').read_bytes());guard=R/pins['storage_guard']['path'];need(sha(guard)==pins['storage_guard']['sha256'],'Storage guard changed')
spec=importlib.util.spec_from_file_location('full100_storage',guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);volume=g.check_bulk_storage(100*1024**2)
prior=json.loads((H/'PRIOR_ADOPTED.json').read_bytes())
for pin in prior.values():need(sha(pin['path'])==pin['sha256'],'Prior actual evidence changed')
pt=json.loads(Path(prior['transport']['path']).read_bytes());need(pt['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and sha(pt['archive_path'])==pt['archive_sha256'],'Prior actual transport changed')
for rel,pin in pt['members'].items():need(sha(Path(pt['verified_extract'])/rel)==pin['sha256'],'Prior extracted member changed')
F=Path('F:/YananResearchStorage/GuardFed/uniform_cuda500_20261011/full100_delta98_attempt001');need(not F.exists() and not F.is_symlink(),'Fresh F transport namespace required');F.mkdir(parents=True)
binding={'execution_authorized':True,'scope':json.loads((H/'SCOPE.json').read_bytes()),'prior_members':pt['members'],'prior_transport_sha256':prior['transport']['sha256'],'prior_root_sha256':prior['root']['sha256']}
code=('BIND='+repr(binding)+'\n').encode()+(H/'collect_remote.py').read_bytes();compile(code,'<actual root Full100 capture>','exec');(H/'ACTUAL_COLLECT_SOURCE.py').write_bytes(code)
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 110 nice -n 10 ionice -c 3 python3 -B -']
command('COLLECT',cmd,code,240);remote=json.loads((H/'COLLECT.stdout').read_bytes())
g.check_bulk_storage(remote['archive_bytes']*4);archive=F/'FULL100_DELTA98.tar.gz'
cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55:'+remote['archive_path'],str(archive)];command('SCP',cmd,timeout=240)
need(sha(archive)==remote['archive_sha256'] and archive.stat().st_size==remote['archive_bytes'],'Archive transport differs')
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
put(F/'TRANSPORT_VERIFICATION.json',proof);put(H/'ACTUAL_TRANSPORT_VERIFICATION.json',proof)
print(json.dumps(dict(status=proof['status'],path=str(H/'ACTUAL_TRANSPORT_VERIFICATION.json'),sha256=sha(H/'ACTUAL_TRANSPORT_VERIFICATION.json'),archive_sha256=sha(archive),members=len(all_members),bytes=archive.stat().st_size)))
