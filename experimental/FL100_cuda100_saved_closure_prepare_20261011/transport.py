from pathlib import Path,PurePosixPath
import argparse,base64,datetime,hashlib,json,subprocess,sys,tarfile
R=Path.cwd();H=R/'tmp/FL100_cuda100_saved_closure_prepare_20261011';sys.path.insert(0,str(R/'tmp'))
from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--inputs-sha256',required=True);parser.add_argument('--source-seal-sha256',required=True)
a=parser.parse_args();assert __debug__ and sha(a.inputs)==a.inputs_sha256
inp=json.loads(a.inputs.read_bytes());template=json.loads((H/'ROOT_INPUTS.template.json').read_bytes())
assert inp['root_authorized_transport'] is True and inp['exact_ids']==template['exact_ids'] and inp['base']==json.loads((H/'SELECTED100_INPUTS.json').read_bytes())['planned_namespace']
assert sha(H/'FILES_SHA256.json')==a.source_seal_sha256
for rel,item in json.loads((H/'FILES_SHA256.json').read_bytes())['files'].items():
 assert Path(rel).name==rel and sha(H/rel)==item['sha256'] and (H/rel).stat().st_size==item['bytes']
review=Path(inp['source_review']['path']);assert sha(review)==inp['source_review']['sha256']
assert json.loads(review.read_bytes())['source_adoptable'] is True and json.loads(review.read_bytes())['source_seal_sha256']==a.source_seal_sha256
assert set(inp['accepted_sha256'])==set(template['exact_ids']) and all(isinstance(v,str) and len(v)==64 for v in inp['accepted_sha256'].values())
for key in ['deployment_sha256','cohort_sha256','canary_complete_sha256','remaining_complete_sha256']:assert isinstance(inp[key],str) and len(inp[key])==64
volume=check_bulk_storage(250*1024**2);F=Path('F:/YananResearchStorage/GuardFed/uniform_cuda_FL100_20261011/all100_delta97_attempt001');assert not F.exists();F.mkdir(parents=True)
code=r'''
from pathlib import Path
import base64,datetime,hashlib,json,tarfile,os
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
B=Path('/workspace/guardfed_checks/uniform_cuda_FL100_20261011_canary1')
inp=json.loads(base64.b64decode('__ROOT_INPUTS__'));frozen=json.loads(base64.b64decode('__FROZEN_INPUTS__'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
ids=inp['exact_ids'];assert ids==frozen['ids'] and inp['base']==str(B)==frozen['planned_namespace']
assert sha(B/'DEPLOYMENT_SEAL.json')==inp['deployment_sha256'];deployment=read(B/'DEPLOYMENT_SEAL.json')
for rel,pin in deployment['files'].items():assert sha(B/rel)==pin['sha256']
assert sha(B/'adapter/EXACT100.json')==inp['cohort_sha256']==frozen['cohort_sha256']
cohort=read(B/'adapter/EXACT100.json');auth=read(B/'ROOT_QUEUE_AUTH.json')
assert cohort['status']=='FIXED_FL100_ACCEPTED_NATIVE_REFERENCES_NOT_CUDA_RESULTS' and len(cohort['records'])==100
assert auth['execution_authorized'] is True and auth['canary_ids']==frozen['canary_ids'] and auth['ordered_ids']==[r['id'] for r in cohort['records']]
assert auth['exact_cohort']=={'path':str(B/'adapter/EXACT100.json'),'sha256':frozen['cohort_sha256']}
complete=read(B/'canary/COMPLETE.json');assert sha(B/'canary/COMPLETE.json')==inp['canary_complete_sha256']
assert complete['ids']==frozen['canary_ids'] and complete['server_saved_accepted'] is True and complete['cohort']==auth['exact_cohort'] and complete['root_offserver_adopted'] is False
assert not (B/'canary/QUEUE_FAILURE.json').exists()
remaining=read(B/'remaining/COMPLETE.json');assert sha(B/'remaining/COMPLETE.json')==inp['remaining_complete_sha256']
assert remaining['ids']==frozen['remaining_ids'] and remaining['server_saved_accepted'] is True and remaining['cohort']==auth['exact_cohort'] and remaining['root_offserver_adopted'] is False
assert not (B/'remaining/QUEUE_FAILURE.json').exists()

for proc in Path('/proc').glob('[0-9]*'):
 try:
  argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
  if str(B/'worker.py') in argv:
   assert '--approval' in argv and argv.index('--approval')+1<len(argv),('Unidentified FL worker',proc.name)
   assert read(argv[argv.index('--approval')+1])['id'] not in ids,('Selected producer still live',proc.name)
  assert not (str(B/'run_queue.py') in argv and any(phase in argv for phase in ('canary','remaining'))),('Canary queue not exited',proc.name)
 except (FileNotFoundError,ProcessLookupError):continue
export=B/'exports/FL100_all100_delta97_saved_audit_attempt001';assert not export.exists();export.mkdir(parents=True)
selection=dict(scope='FL100_ALL100_CUDA_SELECTION_NOT_ROOT_ADOPTED',frozen_selection=frozen,full_cohort_sha256=inp['cohort_sha256'])
selection_path=export/'COHORT_SELECTION.json';selection_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2,allow_nan=False)+chr(10))
files={'base/COHORT_SELECTION.json':selection_path}
for name in ['DEPLOYMENT_SEAL.json','ROOT_QUEUE_AUTH.json','guard.py','run_queue.py','worker.py','canary/COMPLETE.json','remaining/COMPLETE.json']:files['base/'+name]=B/name
seal=read(B/'adapter/FILES_SHA256.json');assert sha(B/'adapter/FILES_SHA256.json')==frozen['formal_seal']
for name,pin in seal['files'].items():
 assert Path(name).name==name and sha(B/'adapter'/name)==pin['sha256'];files['base/adapter/'+name]=B/'adapter'/name
files['base/adapter/FILES_SHA256.json']=B/'adapter/FILES_SHA256.json'
for selected in frozen['rows']:
 row=selected['original_row'];route=selected['route'];identity=row['id'];task=B/'tasks'/identity;out=B/'runs'/identity
 assert next(r for r in cohort['records'] if r['id']==identity)==route
 accepted=read(task/'ACCEPTED.json');assert sha(task/'ACCEPTED.json')==inp['accepted_sha256'][identity] and accepted['id']==identity and accepted['root_offserver_adopted'] is False
 for role,name in [('receipt','receipt.json'),('arrays','validation_predictions.npz'),('binding','formal_binding.json'),('acceptance','formal_saved_acceptance.json')]:assert accepted[role]=={'path':str(out/name),'sha256':sha(out/name)}
 assert accepted['approval']=={'path':str(task/'approval.json'),'sha256':sha(task/'approval.json')}
 approval=read(task/'approval.json');deps=approval['runtime_dependencies'];assert set(deps)==set(selected['package'])
 for role,pin in deps.items():
  assert pin['sha256']==selected['package'][role]['sha256'] and sha(pin['path'])==pin['sha256']
  if role not in {'manifest','proof_registry'}:files['packages/'+route['identity_source']+'/'+role]=Path(pin['path'])
 actual_manifest=read(deps['manifest']['path']);assert actual_manifest['records'][route['identity_index']]==row
 package_root=Path(deps['candidate']['path']).parent;package_seal=read(deps['package_seal']['path'])
 for rel,pin in package_seal['files'].items():assert sha(package_root/rel)==pin['sha256']
 for role in ['author_decision','resource_gate_source','source_seal','formal_GPU_body']:
  pin=approval[role];p=Path(pin['path']);assert p.is_relative_to(B) and sha(p)==pin['sha256'];files['base/'+p.relative_to(B).as_posix()]=p
 receipt=read(out/'receipt.json');binding=read(out/'formal_binding.json');acceptance=read(out/'formal_saved_acceptance.json')
 assert binding['scope']=='UNIFORM_CUDA_FL100_FORMAL_V1' and binding['id']==identity and binding['receipt_sha256']==sha(out/'receipt.json') and binding['approval_sha256']==sha(task/'approval.json')
 assert binding['input_scope_sha256']==frozen['cohort_sha256'] and binding['source_before']==binding['source_after'] and binding['artifact_before']==binding['artifact_after']
 assert receipt['scope']=='FORMAL_FL100_CUDA_PER_RECORD_V1' and receipt['status']=='NATIVE_VALID_REPLAY_PASS' and receipt['runtime']['device']=='cuda:0'
 assert acceptance['status']=='PER_RECORD_FL100_CUDA_SAVED_WHOLE_ACCEPTED' and acceptance['id']==identity and acceptance['binding_sha256']==sha(out/'formal_binding.json')
 assert acceptance['views']==receipt['views'] and acceptance['native_comparison']==receipt['native_comparison'] and receipt['native_comparison']['accepted'] is True and receipt['native_comparison']['max_abs_difference']<=1e-12
 assert not out.with_name(out.name+'.formal_failure.json').exists()
 for name in ['approval.json','preflight.json','preflight_command.json','worker_command.json','ACCEPTED.json']:files['base/tasks/'+identity+'/'+name]=task/name
 for name in ['receipt.json','validation_predictions.npz','formal_binding.json','formal_saved_acceptance.json']:files['base/runs/'+identity+'/'+name]=out/name
files['source/metadata.npz']=Path('/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/metadata.npz')
assert sha(files['source/metadata.npz'])=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
assert not any(n.endswith(('.pt','.pth','.npy')) or n.endswith('/model_inventory.json') for n in files)

# Exact prior3 plus original metadata remain in their adopted F archive; download only97 arrays.
old_ids=set(frozen['canary_ids'])
reused={n:p for n,p in files.items() if n=='source/metadata.npz' or any(n.startswith('base/tasks/'+i+'/') or n.startswith('base/runs/'+i+'/') for i in old_ids)}
reused_pins={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in reused.items()}
assert set(old_ids)==set(inp['prior3_accepted_sha256'])
for i in old_ids:assert inp['accepted_sha256'][i]==inp['prior3_accepted_sha256'][i]
files={n:p for n,p in files.items() if n not in reused}
assert len([n for n in files if n.endswith('/validation_predictions.npz')])==97
members={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}
index=export/'ROOT_MEMBERS.json';assert not index.exists();index.write_text(json.dumps({'files':members,'ids':ids},indent=2)+chr(10))
archive=export/'FL100_DELTA97.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for n,p in files.items():tar.add(p,arcname=n,recursive=False)
 tar.add(index,arcname='ROOT_MEMBERS.json',recursive=False)
assert all(sha(p)==members[n]['sha256'] for n,p in files.items())
assert all(sha(p)==reused_pins[n]['sha256'] for n,p in reused.items())
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_path=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,archive_members=len(files)+1,payload_members=len(files),member_index_sha256=sha(index),ids=ids,new_arrays=97,reused_arrays=3,reused_member_pins=reused_pins)))
'''
code=code.replace('__ROOT_INPUTS__',base64.b64encode(json.dumps(inp,allow_nan=False).encode()).decode()).replace('__FROZEN_INPUTS__',base64.b64encode((H/'SELECTED100_INPUTS.json').read_bytes()).decode())
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 110 nice -n 10 ionice -c 3 python3 -B -']
p=subprocess.run(cmd,input=code.encode(),capture_output=True,timeout=300)
(H/'FL100_DELTA97_BACKUP.stdout').write_bytes(p.stdout);(H/'FL100_DELTA97_BACKUP.stderr').write_bytes(p.stderr)
put(H/'FL100_DELTA97_BACKUP_COMMAND.json',dict(command=cmd,payload_sha256=hashlib.sha256(code.encode()).hexdigest(),exit_code=p.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(H/'FL100_DELTA97_BACKUP.stdout'),stderr_sha256=sha(H/'FL100_DELTA97_BACKUP.stderr')))
assert p.returncode==0,p.stderr.decode(errors='replace');remote=json.loads(p.stdout)
check_bulk_storage(remote['archive_bytes']*4)
archive=F/'FL100_DELTA97.tar.gz';cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55:'+remote['archive_path'],str(archive)]
p=subprocess.run(cmd,capture_output=True,timeout=120)
(H/'FL100_DELTA97_SCP.stdout').write_bytes(p.stdout);(H/'FL100_DELTA97_SCP.stderr').write_bytes(p.stderr)
put(H/'FL100_DELTA97_SCP_COMMAND.json',dict(command=cmd,exit_code=p.returncode,stdout_sha256=sha(H/'FL100_DELTA97_SCP.stdout'),stderr_sha256=sha(H/'FL100_DELTA97_SCP.stderr')))
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
put(F/'TRANSPORT_VERIFICATION.json',proof);put(H/'FL100_DELTA97_TRANSPORT_VERIFICATION.json',proof)
print(json.dumps(dict(status=proof['status'],path=str(H/'FL100_DELTA97_TRANSPORT_VERIFICATION.json'),sha256=sha(H/'FL100_DELTA97_TRANSPORT_VERIFICATION.json'),archive_sha256=sha(archive),members=len(all_members),bytes=archive.stat().st_size)))
