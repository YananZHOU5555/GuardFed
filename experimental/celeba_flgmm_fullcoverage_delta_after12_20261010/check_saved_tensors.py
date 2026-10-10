"""Supplementary saved-state layout/finite checks only; no forward or data loading."""
from pathlib import Path
import ast,datetime,hashlib,json,os,sys,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
try:
 import torch
 torch.set_num_threads(1)
 proof=read(H/'batch/OFFSERVER_ACCEPTANCE.json');assert proof['accepted_new']==2 and proof['accepted_total']==14
 cnn=R/'tmp/revision-publish-20260928/src/celeba_data.py'
 assert sha(cnn)==proof['records'][0]['source_hashes']['src/celeba_data.py']=='0f48fbc6d241e7d0cde692839659e817feab407991795a6f92a33052a9cc07ce'
 tree=ast.parse(cnn.read_text(encoding='utf8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CelebACNN')
 ns={'torch':torch,'nn':torch.nn};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(cnn)+'[CLASS_ONLY_NO_FORWARD]','exec'),ns)
 expected=ns['CelebACNN']().state_dict();records=[]
 for record in proof['records']:
  p=H/'batch/restored/runs'/record['id']/'model.pt';before=sha(p);assert before==record['checkpoint_sha256']
  state=torch.load(p,map_location='cpu',weights_only=True)
  assert isinstance(state,dict) and tuple(state)==tuple(expected)
  tensors=[]
  for key,value in state.items():
   assert isinstance(value,torch.Tensor) and value.device.type=='cpu'
   assert value.shape==expected[key].shape and value.dtype==expected[key].dtype
   assert torch.isfinite(value).all().item()
   tensors.append(dict(name=key,shape=list(value.shape),dtype=str(value.dtype),elements=value.numel(),finite=True))
  assert sha(p)==before
  records.append(dict(id=record['id'],checkpoint_sha256=before,tensor_count=len(tensors),elements=sum(x['elements'] for x in tensors),tensors=tensors,checkpoint_bytes_unchanged=True))
 assert not torch.cuda.is_initialized()
 out=dict(status='SAVED_FULL_STATE_LAYOUT_DTYPE_FINITE_PASS_NO_FORWARD',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),offserver_acceptance_sha256=sha(H/'batch/OFFSERVER_ACCEPTANCE.json'),cnn_source_path=cnn.relative_to(R).as_posix(),cnn_source_sha256=sha(cnn),original_class_only=True,CNN_forward_calls=0,optimizer_calls=0,data_loads=0,training_runtime_not_recreated=True,verification_runtime=dict(python=sys.version,torch=torch.__version__,cuda_initialized=False),records=records)
 with (H/'SAVED_TENSOR_STATE_CHECK.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=out['status'],models=len(records),tensors=sum(r['tensor_count'] for r in records),elements=sum(r['elements'] for r in records),sha256=sha(H/'SAVED_TENSOR_STATE_CHECK.json'))))
except BaseException as e:
 with (H/'SAVED_TENSOR_STATE_FAILURE.json').open('x',encoding='utf8') as f:json.dump(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),f,indent=2)
 raise
