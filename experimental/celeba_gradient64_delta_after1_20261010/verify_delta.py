"""Original v4 member verification and original checked_result for the frozen delta."""
from pathlib import Path, PurePosixPath
import ast, datetime, hashlib, importlib.util, json, os, platform, sys, tarfile, traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
F=Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def require(v,m):
 if not v:raise ValueError(m)
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
if __name__=='__main__':
 try:
  os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  auth=read(H/'AUTHORIZED_SNAPSHOT.json');receipt=read(F/'backup_receipt.json');archive=F/'gradient64_delta_after1.tar.gz'
  require(sha(F/'backup_receipt.json')==read(H/'DOWNLOAD.json')['receipt_sha256'],'Actual downloaded receipt changed')
  require(receipt['accepted_new_ids']==auth['authorized_ids'] and receipt['previous_root_sha256']==auth['prior_root_sha256'] and receipt['previous_offserver_sha256']==auth['prior_offserver_sha256'],'Delta/parent receipt mismatch')
  original=R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
  require(sha(original)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef','Original archive verifier changed')
  v=load(original,'archive_verifier_original');member_proof=v.verify_archive(archive,receipt)
  require(member_proof['pass'] and member_proof['different_host_observed'],'Actual offserver required')
  guard=load(R/'tmp/guardfed_local_storage.py','external_storage_guard')
  with tarfile.open(archive,'r:gz') as t:required=sum(m.size for m in t)
  save('F_VOLUME_BEFORE_RESTORE.json',guard.check_bulk_storage(required))
  out=F/'restored';out.mkdir(exist_ok=False)
  # Same safe-member checks/extraction as the original first1 offserver reader.
  with tarfile.open(archive,'r:gz') as t:
   for m in t:
    name=PurePosixPath(m.name)
    require(m.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in m.name and '\\' not in m.name,'Unsafe member')
    p=out.joinpath(*name.parts);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(t.extractfile(m).read())
  inventory=read(out/'backup_inventory.json')
  for n,row in inventory['members'].items():
   p=out/n;require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Extracted member changed: '+n)
  require(inventory['accepted_new_ids']==auth['authorized_ids'] and inventory['old_models_repacked']==0 and inventory['selected_only'],'Wrong archive scope')
  for n,pin in [('PREVIOUS_ROOT_ADOPTION.json',auth['prior_root_sha256']),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',auth['prior_offserver_sha256']),('AUTHORIZED_SNAPSHOT.json',sha(H/'AUTHORIZED_SNAPSHOT.json'))]:require(sha(out/'evidence'/n)==pin,'Parent/auth binding changed')
  prior=read(out/'evidence/PREVIOUS_ROOT_ADOPTION.json');previous=read(out/'evidence/PREVIOUS_OFFSERVER_ACCEPTANCE.json')
  require(prior['accepted_ids']==previous['accepted_new_ids']==auth['accepted_prior_ids'] and prior['accepted_count']==1,'Original1 parent mismatch')
  pkg=out/'source/package';seal=pkg/'FILES_SHA256.json'
  require(sha(seal)==auth['source_seal_sha256']=='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced','Scientific seal changed')
  for n,d in read(seal)['files'].items():require(sha(pkg/n)==d,'Scientific source member changed')
  stage=pkg/'snapshot/gradient_bridge_20261010';sys.path.insert(0,str(stage))
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  from accept_result import checked_result
  require(sha(stage/'accept_result.py')=='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9','Original strict function changed')
  remote=read(out/'evidence/ORIGINAL_STRICT.json');manifest=read(pkg/'jobs/manifest.json');ids=auth['authorized_ids']
  require(remote['accepted_new_ids']==ids and [x['id'] for x in remote['records']]==ids,'Server strict scope mismatch')
  require(len(manifest['jobs'])==len({x['id'] for x in manifest['jobs']})==64 and [x['id'] for x in manifest['jobs'] if x['id'] in ids]==ids,'Frozen64 grid/order mismatch')
  require(not set(ids)&set(prior['accepted_ids']) and len(ids)==len(set(ids)),'Repeated old/duplicate ID')
  records=[];tensor_rows=[]
  for ID in ids:
   entry=next(r for r in manifest['jobs'] if r['id']==ID);job=pkg/'jobs'/entry['job'];require(sha(job)==entry['job_sha256'],'Frozen job changed')
   result=checked_result(job,out/'runs'/ID);server=next(x for x in remote['records'] if x['id']==ID)
   require(result is not None and result['metrics']==server['metrics'] and result['provenance']==server['original_training_provenance'],'Local strict differs from original server strict')
   cp=out/'runs'/ID/'model.pt';require(sha(cp)==server['checkpoint_sha256'],'Terminal model changed')
   checkpoint=torch.load(cp,map_location='cpu',weights_only=True)
   # Supplementary original CNN state layout only: no forward, data or optimizer.
   cnn=R/'tmp/revision-publish-20260928/src/celeba_data.py';require(sha(cnn)==read(job)['source_hashes']['src/celeba_data.py']=='0f48fbc6d241e7d0cde692839659e817feab407991795a6f92a33052a9cc07ce','CNN source layout binding')
   cls=next(n for n in ast.parse(cnn.read_text('utf8')).body if isinstance(n,ast.ClassDef) and n.name=='CelebACNN')
   ns={'torch':torch,'nn':torch.nn};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(cnn)+'[CLASS_ONLY_NO_FORWARD]','exec'),ns)
   expected=ns['CelebACNN']().state_dict();require(tuple(checkpoint)==tuple(expected),'Incomplete terminal fullstate keys/order')
   tensors=[]
   for key,value in checkpoint.items():
    require(value.device.type=='cpu' and value.shape==expected[key].shape and value.dtype==expected[key].dtype and torch.isfinite(value).all().item(),'Saved state layout/dtype/finiteness')
    tensors.append(dict(name=key,shape=list(value.shape),dtype=str(value.dtype),elements=value.numel(),finite=True))
   require(sha(cp)==server['checkpoint_sha256'],'Saved fullstate changed during check')
   tensor_rows.append(dict(id=ID,checkpoint_sha256=sha(cp),tensors=tensors,tensor_count=len(tensors),elements=sum(x['elements'] for x in tensors)))
   records.append(dict(id=ID,checkpoint_sha256=sha(cp),job_sha256=sha(job),rounds=result['rounds'],metrics=result['metrics'],data_contract=result['data_contract'],original_training_provenance=result['provenance'],tensor_count=len(checkpoint),constant_negative_retained=result['metrics']==dict(accuracy=0.5166859616449389,aeod=0.0,aspd=0.0)))
  require(not torch.cuda.is_initialized(),'No CUDA compute permitted')
  save('SAVED_TENSOR_STATE_CHECK.json',dict(status='SAVED_FULL_STATE_LAYOUT_DTYPE_FINITE_PASS_NO_FORWARD',models=len(records),records=tensor_rows,CNN_forward_calls=0,optimizer_calls=0,data_loads=0,cnn_source_sha256=sha(cnn),runtime=dict(python=sys.version,torch=torch.__version__,cuda_initialized=False)))
  proof=dict(status='OFFSERVER_ORIGINAL_STRICT_DELTA_PASS',accepted_new_ids=ids,accepted_before=1,accepted_total=1+len(ids),accepted_job_ids=prior['accepted_ids']+ids,records=records,members_verified=member_proof['members_verified'],archive_verification=member_proof,archive_sha256=sha(archive),receipt_sha256=sha(F/'backup_receipt.json'),inventory_sha256=sha(out/'backup_inventory.json'),original_strict_sha256=sha(out/'evidence/ORIGINAL_STRICT.json'),scientific_package_sha256=sha(seal),original_validator_sha256=sha(stage/'accept_result.py'),previous_root_sha256=auth['prior_root_sha256'],previous_offserver_sha256=auth['prior_offserver_sha256'],authorized_snapshot_sha256=sha(H/'AUTHORIZED_SNAPSHOT.json'),old1_ordered_prefix_exact=True,local_runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,host=platform.node(),cuda_hidden=True),runtime_equivalence_claim=False,new_inference=0,new_training=0,old_models_repacked=0,root_adopted=False,screen64_completed_claim=False,method_champion_claim=False,archive=str(archive),restored=str(out),tensor_proof_sha256=sha(H/'SAVED_TENSOR_STATE_CHECK.json'),prediction_arrays_recomputed=False,shared_log_is_prefix_only=True,all_negative_results_retained=True)
  save('OFFSERVER_ACCEPTANCE.json',proof)
  print(json.dumps(dict(status=proof['status'],new=len(ids),total=proof['accepted_total'],members=proof['members_verified'],proof_sha256=sha(H/'OFFSERVER_ACCEPTANCE.json'))))
 except BaseException as e:
  save('OFFSERVER_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
