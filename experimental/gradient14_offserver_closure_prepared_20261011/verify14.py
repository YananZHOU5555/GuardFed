"""Original archive/checked/tensor/RNG code; private F restore paths, no inference/fit."""
from pathlib import Path,PurePosixPath
import argparse,ast,hashlib,importlib.util,json,os,sys,tarfile,textwrap,traceback,platform
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;R=HERE.parents[1]
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--comparison-sha256',required=True);p.add_argument('--restore',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
def require(v,m):
 if not v:raise ValueError(m)
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
require(not a.output.exists() and not a.output.with_suffix('.failure.json').exists(),'Preserve output/failure; no automatic retry')
require(a.output.resolve().is_relative_to(HERE.resolve()) and a.output.parent.is_dir(),'Create a compact output parent in the owned namespace first')
pins=read(HERE/'SOURCE_PINS.json')
for rel,digest in pins['sources'].items():require(sha(R/rel)==digest,'Original source changed: '+rel)
guard=load(R/'tmp/guardfed_local_storage.py','original_F_guard');volume=guard.check_bulk_storage(0)
for f in (a.archive,a.receipt,a.restore):require(f.resolve().is_relative_to(guard.STORAGE_ROOT.resolve()),'Bulk must remain on healthy F')
try:
 require(sha(a.receipt)==a.receipt_sha256,'Receipt changed');receipt=read(a.receipt)
 require(receipt['scope_sha256']==pins['scope_sha256'] and receipt['comparison_sha256']==a.comparison_sha256,'Scope/comparison changed')
 ids=[e['id'] for e in pins['ordered_jobs']];require(receipt['accepted_new_ids']==ids and len(ids)==14,'Fixed14 order changed')
 require(receipt['scientific_table_records']==0 and receipt['test_evaluated'] is False,'Short-gate scope changed')
 evidence=load(R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py','original_v4_archive')
 archive_proof=evidence.verify_archive(a.archive,receipt)
 require(archive_proof['pass'] and archive_proof['different_host_observed'] and archive_proof['members_verified']==receipt['members'],'Actual offserver/member verification required')
 with tarfile.open(a.archive,'r:gz') as t:required=sum(m.size for m in t)
 volume=guard.check_bulk_storage(required);a.restore.mkdir(exist_ok=False);out=a.restore
 # Original first1/after42 safe extraction loop; only output path is rebound.
 with tarfile.open(a.archive,'r:gz') as t:
  for m in t:
   name=PurePosixPath(m.name)
   require(m.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in m.name and '\\' not in m.name,'Unsafe member')
   p=out.joinpath(*name.parts);p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(t.extractfile(m).read())
 inventory=read(out/'backup_inventory.json')
 for n,row in inventory['members'].items():require((out/n).stat().st_size==row['bytes'] and sha(out/n)==row['sha256'],'Restored member changed')
 runtime_manifest=out/'closure_source/RUNTIME_MEMBERS.json'
 require(sha(runtime_manifest)==receipt['runtime_manifest_sha256'],'Runtime receipt manifest changed')
 for name,pin in read(runtime_manifest)['files'].items():require(inventory['members'].get(name)==pin,'Runtime receipt member mismatch')
 scope=read(out/'scope.json');require(sha(out/'scope.json')==pins['scope_sha256'] and scope['jobs']==pins['ordered_jobs'],'Original14 scope/jobs changed')
 require(inventory['accepted_new_ids']==ids and inventory['scientific_table_records']==0,'Inventory scope changed')
 require(sha(out/'GATE_COMPARISON.json')==a.comparison_sha256,'Original comparison changed');comparison=read(out/'GATE_COMPARISON.json')
 package=R/'tmp/celeba_gradient_fullcoverage_gates_prepare_20261010';sys.path.insert(0,str(package));import adapter
 require(Path(adapter.__file__).resolve()==package/'adapter.py','Wrong original adapter')
 # Only private paths/stage/helper globals. No load_context, external-path rewrite or shared-cache loading.
 gate=adapter.load(R/'tmp/celeba_gradient_realimage_gate_20261009/gate.py','offserver_original_gate');gate.HERE=out;gate.STAGE=adapter.STAGE
 gate.check_attack_assignment=adapter.check_attack_assignment
 derived,_=adapter.render_gate_functions();exec(compile(derived['checked'],'<original_adapter_checked>','exec'),gate.__dict__)
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 rows={};receipts={};tensor_rows=[];deliveries=list(out.glob('offserver_closure_*/strict_delivery.json'));require(len(deliveries)==1,'Exactly one server delivery required');remote=read(deliveries[0])
 require(remote['comparison_sha256']==a.comparison_sha256 and [r['id'] for r in remote['records']]==ids,'Server delivery changed')
 for entry in scope['jobs']:
  output=out/entry['output'];accepted=read(output/'acceptance.json');outer=read(output/'item_receipt.json')
  require(outer['status']=='CPU3_GATE_ORIGINAL_STRICT_AND_EXTERNAL_IDENTITY_PASS' and outer['scope_sha256']==pins['scope_sha256'] and outer['scientific_table_records']==0,'Item identity changed')
  require(outer['acceptance_sha256']==sha(output/'acceptance.json') and outer['rng_sha256']==sha(output/'rng_after.json'),'Item/RNG changed')
  result=gate.checked(entry,scope,accepted['dispatch_receipt']);require(result is not None,'Missing original strict result')
  job=out/entry['job'];require(sha(job)==entry['job_sha256'],'Frozen job changed');jobrow=read(job)
  key=(jobrow['method'],jobrow['attack'],jobrow['implementation']);require(key not in rows,'Duplicate case');rows[key]=(output,result)
  receipts[entry['id']]=sha(output/'item_receipt.json');ID=entry['id'];server=next(r for r in remote['records'] if r['id']==ID);cp=output/'model.pt'
  require(result['metrics']==server['metrics'] and sha(cp)==server['checkpoint_sha256'],'Server/offserver checkpoint or metrics changed')
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
  require(len(checkpoint)==8,"Original eight CNN tensors required")
 require(not torch.cuda.is_initialized(),'No CUDA computation')
 # Compile the original nested exact() and comparison loop without changing a byte of scientific source.
 comparison_source=(package/'compare_saved.py').read_text('utf8');tree=ast.parse(comparison_source)
 compare_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compare')
 exact_node=next(n for n in compare_node.body if isinstance(n,ast.FunctionDef) and n.name=='exact')
 exact_source=ast.get_source_segment(comparison_source,exact_node);env=dict(rows=rows,torch=torch,read=read,scope=scope)
 exec(compile(exact_source,'<original_compare_saved_exact>','exec'),env)
 start=comparison_source.index('    comparisons=[]');end=comparison_source.index('    before=read(',start)
 loop=textwrap.dedent(comparison_source[start:end]);exec(compile(loop,'<original_compare_saved_loop>','exec'),env)
 require(env['comparisons']==comparison['comparisons'] and receipts==comparison['strict_receipts_sha256'],'Original pairing/RNG/receipts differ')
 require(comparison['status']=='EXACT14_CPU3_GATE_SAVED_COMPARISON_PASS' and (comparison['checked_jobs'],comparison['rounds_per_job'],comparison['total_rounds'])==(14,3,42),'Wrong original comparison scope')
 proof=dict(status='EXACT14_OFFSERVER_ORIGINAL_SAVED_STRICT_MODEL_RNG_AND_MEMBER_PASS',root_adopted=False,all14_offserver_verified=True,accepted_ids=ids,
  scope_sha256=pins['scope_sha256'],comparison_sha256=a.comparison_sha256,archive_verification=archive_proof,archive_sha256=sha(a.archive),receipt_sha256=sha(a.receipt),
  members_verified=archive_proof['members_verified'],model_count=len(tensor_rows),tensors=tensor_rows,original_comparisons=env['comparisons'],
  local_runtime=dict(python=sys.version,torch=torch.__version__,cuda_initialized=False,host=platform.node()),fresh_F_volume=volume,
  checked_source_sha256=sha(package/'adapter.py'),original_compare_saved_sha256=sha(package/'compare_saved.py'),original_gate_sha256=sha(R/'tmp/celeba_gradient_realimage_gate_20261009/gate.py'),
  saved_source_data_receipt_lineage_only=True,complete_image_data_reread=False,scientific_table_records=0,new_inference=0,new_fit=0,new_training=0,test=False,
  coverage192_authorized=False,runtime_training_equivalence_claim=False,restore=str(out),source_pins_sha256=sha(HERE/'SOURCE_PINS.json'))
 with a.output.open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=proof['status'],models=len(tensor_rows),members=proof['members_verified'],proof_sha256=sha(a.output))))
except BaseException as error:
 target=a.output.with_suffix('.failure.json')
 if not target.exists():
  with target.open('x',encoding='utf8') as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False),f,indent=2)
 raise
