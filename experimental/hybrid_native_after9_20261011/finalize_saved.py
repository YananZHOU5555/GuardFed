from pathlib import Path
import ast,datetime,hashlib,json,math,os,statistics,subprocess,sys
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
H=Path(__file__).resolve().parent; R=H.parents[1]
F=Path('F:/YananResearchStorage/GuardFed/hybrid_native_after9_20261011');V=F/'verified';S=V/'stage'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
def copy(p,n):
 with (H/n).open('xb') as f:f.write(p.read_bytes())
assert not sys.flags.optimize and len(sys.argv)==2 and sha(H/'FILES_SHA256.json')==sys.argv[1]
assert not (H/'SAVED_TENSOR_IDENTITY.json').exists() and not (H/'ROOT_READY_HANDOFF.json').exists()
for n,p in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/n)==p['sha256']
delta=read(H/'DELTA_SCOPE.json');IDS=delta['ids'];off=read(V/'OFFSERVER_VERIFICATION.json');receipt=read(F/'BACKUP_RECEIPT.json');remote=read(F/'REMOTE_STRICT.json')
assert off['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON' and (off['accepted_total'],off['local']['accepted_before'],off['local']['accepted_new'])==(12,9,3)
assert off['member_count']==receipt['archive_members'] and receipt['content_members']+1==receipt['archive_members']
assert off['accepted_ids']==read(H/'PRIOR_ROOT_ADOPTION.json')['accepted_ids']+IDS
assert off['accepted_ids'][:9]==read(H/'PRIOR_OFFSERVER.json')['accepted_ids']
assert off['local']['accepted_new_ids']==IDS and off['archive_sha256']==sha(F/'delta.tar.gz')==receipt['archive_sha256']
assert remote['source_data_before']==remote['source_data_after'] and remote['server_runtime']['CPU']==[108]
for n in ['COLLECT','SCP','RESTORE_VERIFY']:assert read(H/(n+'_COMMAND.json'))['returncode']==0
for src,n in [(V/'OFFSERVER_VERIFICATION.json','OFFSERVER_VERIFICATION.json'),(F/'REMOTE_STRICT.json','REMOTE_STRICT.json'),(F/'BACKUP_RECEIPT.json','BACKUP_RECEIPT.json'),(F/'MEMBERS.json','MEMBERS.json')]:copy(src,n)
# Only one fresh read-only release query; it never touches other affinity groups or services.
code="""from pathlib import Path
import os,json,hashlib,datetime
g=Path('/etc/vast-agents-guide.md').read_bytes();assert hashlib.sha256(g).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
owners=[];collectors=[]
for p in Path('/proc').glob('[0-9]*'):
 if int(p.name)==os.getpid():continue
 try:
  a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\\0') if x]
  if any('native_after9_20261011/collect_once.py' in x for x in a):collectors.append({'pid':int(p.name),'argv':a})
  for t in (p/'task').iterdir():
   try:aff=os.sched_getaffinity(int(t.name))
   except ProcessLookupError:continue
   if len(aff)<=16 and 108 in aff:owners.append({'pid':int(p.name),'tid':int(t.name),'cpus':sorted(aff),'argv':a})
 except (FileNotFoundError,ProcessLookupError):continue
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU108_owners=owners,matching_collectors=collectors,CPU108_released=not owners and not collectors,guide_sha256=hashlib.sha256(g).hexdigest(),read_only=True)))
assert not owners and not collectors
"""
(H/'CPU_RELEASE_SOURCE.py').write_text(code,encoding='utf8',newline='\n')
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -']
p=subprocess.run(cmd,input=code.encode(),capture_output=True,timeout=60)
(H/'CPU_RELEASE.stdout').write_bytes(p.stdout);(H/'CPU_RELEASE.stderr').write_bytes(p.stderr)
save('CPU_RELEASE_COMMAND.json',dict(command=cmd,returncode=p.returncode,stdin_sha256=hashlib.sha256(code.encode()).hexdigest()))
p.check_returncode();release=json.loads(p.stdout);assert release['CPU108_released'];save('CPU_RELEASE.json',release)
# Extract the identical tensor digest from original source, never instantiate a model/forward/optimizer.
import torch
torch.set_num_threads(1);assert not torch.cuda.is_initialized()
src=(S/'body.py').read_text('utf8');node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='tensor_sha');fn=ast.get_source_segment(src,node)
assert hashlib.sha256(fn.encode()).hexdigest()=='9d7e4d44d49de4393d2ab8b607e32c427af95c381c25fa6bb7ae8175a830cbe4'
ns={'hashlib':hashlib};exec(compile(fn,'original_body.py:tensor_sha','exec'),ns)
tensor_rows=[];records=[];manifest=read(S/'manifest.json');byid={x['id']:x for x in manifest['jobs']}
for row in off['local']['records']:
 ID=row['id'];entry=byid[ID];d=S/entry['output'];a=read(d/'acceptance.json');result=read(d/'result.json');job=read(S/entry['job']);prov=read(d/'provenance.json')
 assert sha(d/'model.pt')==row['checkpoint_sha256'] and sha(d/'acceptance.json')==row['acceptance_sha256']
 assert sha(S/entry['job'])==entry['job_sha256']==a['job_sha256'] and result['config']==job['config']
 state=torch.load(d/'model.pt',map_location='cpu',weights_only=True);digest=ns['tensor_sha'](state)
 assert len(state)==8 and digest==a['checkpoint_tensor_sha256'] and all(torch.isfinite(x).all() for x in state.values())
 tensor_rows.append(dict(id=ID,checkpoint_sha256=row['checkpoint_sha256'],tensor_sha256=digest,tensors=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype),elements=v.numel(),finite=True) for k,v in state.items()]))
 records.append(dict(row,seed=result['seed'],job_sha256=entry['job_sha256'],rounds=result['rounds'],evaluation_stats=result['evaluation_stats'],data_contract=result['data_contract']['image_data_contract'],provenance=prov))
assert len(tensor_rows)==3 and not torch.cuda.is_initialized()
save('SAVED_TENSOR_IDENTITY.json',dict(status='PASS_ORIGINAL_TENSOR_DIGEST_FINITE_NO_FORWARD',models=3,tensors=24,original_tensor_function_sha256=hashlib.sha256(fn.encode()).hexdigest(),records=tensor_rows,CNN_calls=0,CUDA_initialized=False))
save('RECORDS3.json',dict(status='ORIGINAL_STRICT_OFFSERVER_EXACT3_ROOT_ADOPTION_PENDING',records=records,prior_root_sha256=sha(H/'PRIOR_ROOT_ADOPTION.json'),offserver_sha256=sha(H/'OFFSERVER_VERIFICATION.json'),root_adopted=False))
# No scene statistics here: the frozen IID F Flip scope has only three seeds.
files=[p for p in sorted(F.rglob('*')) if p.is_file()]
save('RAW_STORAGE_INDEX.json',dict(status='F_ONLY_NEW_EXACT3_ARCHIVE_RECOVERY_INDEX',bulk_root=F.as_posix(),files={p.relative_to(F).as_posix():dict(path=p.as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files},original_F_guard=read(H/'F_VOLUME_DOWNLOAD.json'),prior_models_repacked=0,reuse_models_repacked=0))
handoff=dict(status='ROOT_READY_EXACT3_HYBRID70_NATIVE_STRICT_OFFSERVER_NOT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_before=9,accepted_new=3,accepted_total=12,accepted_new_ids=IDS,accepted_ids=off['accepted_ids'],planned_new=96,reused_separate=4,planned_total=100,canaries_separate=7,root_adopted=False,root_adoption_required=True,prior_root_path='tmp/celeba_hybrid_native9_root_adoption_20261011/ROOT_ADOPTION.json',prior_root_sha256=sha(H/'PRIOR_ROOT_ADOPTION.json'),prior_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),archive_path=(F/'delta.tar.gz').as_posix(),archive_sha256=sha(F/'delta.tar.gz'),archive_bytes=(F/'delta.tar.gz').stat().st_size,archive_members=off['member_count'],receipt_path=(F/'BACKUP_RECEIPT.json').as_posix(),receipt_sha256=sha(F/'BACKUP_RECEIPT.json'),inventory_sha256=receipt['inventory_sha256'],offserver_path=(V/'OFFSERVER_VERIFICATION.json').as_posix(),offserver_sha256=sha(V/'OFFSERVER_VERIFICATION.json'),remote_strict_sha256=sha(F/'REMOTE_STRICT.json'),source_seal_sha256=sha(H/'FILES_SHA256.json'),delta_scope_sha256=sha(H/'DELTA_SCOPE.json'),package_sha256=receipt['package_sha256'],original_checked_source_sha256=remote['original_checked_source_sha256'],source_data_before_after_exact=True,same_terminal_checkpoint_all_metrics=True,rounds=70,n_eval=19867,saved_models=3,saved_tensors=24,tensor_proof_sha256=sha(H/'SAVED_TENSOR_IDENTITY.json'),raw_storage_index_sha256=sha(H/'RAW_STORAGE_INDEX.json'),server_runtime=remote['server_runtime'],local_runtime=off['local']['local_verification_runtime'],runtime_equivalence_claim=False,CPU108_released=release['CPU108_released'],release_sha256=sha(H/'CPU_RELEASE.json'),collector_runs=1,restore_runs=1,actual_exit_codes={n:read(H/(n+'_COMMAND.json'))['returncode'] for n in ['COLLECT','SCP','RESTORE_VERIFY','CPU_RELEASE']},new_CNN=0,new_fit=0,new_training=0,final_test=False,recipe_reselection=False,prediction_arrays_supplied=False,prediction_arrays_recomputed=False,negative_results_preserved=True,shared_STATE_LATEST_Git_changed=False,old9_ids_order_exact=off['accepted_ids'][:9]==read(H/'PRIOR_OFFSERVER.json')['accepted_ids'],old9_source_objects_bytes_exact=sha(H/'PRIOR_ROOT_ADOPTION.json')==read(H/'INPUT_PINS.json')['prior_root_sha256'],limitations=['IID F Flip frozen partial3 only;not a complete10seed scene or full100.','Saved original record/tensor checks;no numerical cross-runtime equivalence or prediction-array replay.','Old9 formal records,4 selected screen reuses,and7 three-round canaries remain separate and unrepacked.','Exact3 requires independent root adoption;no current acceptance claimed by this handoff.'])
save('ROOT_READY_HANDOFF.json',handoff)
save('ROOT_READY_CHAIN_LINK.json',dict(status='EXACT3_ROOT_ADOPTION_PENDING',accepted_before=9,accepted_new=3,accepted_total=12,accepted_ids=off['accepted_ids'],previous_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),previous_root_sha256=sha(H/'PRIOR_ROOT_ADOPTION.json'),next_prior_offserver_path=(V/'OFFSERVER_VERIFICATION.json').as_posix(),next_prior_offserver_sha256=sha(V/'OFFSERVER_VERIFICATION.json'),handoff_sha256=sha(H/'ROOT_READY_HANDOFF.json'),root_adoption_required=True,reused_separate=4))
save('DELIVERY_FILES_SHA256.json',dict(status='ACTUAL_COMPACT_EXACT3_ROOT_READY_DELIVERY',files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.iterdir()) if p.is_file()}))
print(json.dumps(dict(handoff_sha256=sha(H/'ROOT_READY_HANDOFF.json'),delivery_seal_sha256=sha(H/'DELIVERY_FILES_SHA256.json'),new_accepted_pending=3,total_if_root_adopted=12,CPU108_released=True),indent=2))
