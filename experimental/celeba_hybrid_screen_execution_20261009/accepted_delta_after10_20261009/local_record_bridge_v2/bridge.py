"""PREPARED CPU record-layer replay; actual CUDA runtime comes only from pinned server proof."""
from pathlib import Path
import hashlib,json,sys,os,types,ast
HERE=Path(__file__).resolve().parent;B=HERE.parent;H=B.parent
SERVER_SHA='2fcb8394c91b86b0cd86189b6f09d58deb4c12813cec8389160c26d290df596c'
ARCHIVE_SHA='1e1e9e502087191eeb2001a9e9776cc4d3f4f7c0e1d580abffde851af2786782'
INVENTORY_SHA='351988b6fdb2bf3780d7a1bafe15f97a03ae0472b4a8a26377b035d41fcfa237'
SEAL_SHA='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need_file(p,expected):assert sha(p)==expected, 'Bound evidence/source/artifact SHA mismatch: '+str(p)
def check_runtime(server):
 r=server['runtime'];assert server['status']=='PARTIAL_ORIGINAL_STRICT_ACCEPTED_OFFSERVER_PENDING' and server['no_CNN_training_or_test']
 assert r['torch']=='2.11.0+cu128' and r['cuda_build']=='12.8' and r['CPU']==[106] and r['threads']==1
 assert r['gpu_name_for_original_checker']=='NVIDIA GeForce RTX 5090'
 assert server['source_data_verified_before_after'] and server['source_seal_sha256']==SEAL_SHA
 return r

def replay():
 # All original checks remain; only three CURRENT-HOST runtime queries become bound original-server observations.
 need_file(B/'PARTIAL_ACCEPTANCE.json',SERVER_SHA);need_file(B/'hybrid_after10_delta.tar.gz',ARCHIVE_SHA);need_file(B/'MEMBERS.json',INVENTORY_SHA);need_file(H/'FILES_SHA256.json',SEAL_SHA)
 server=read(B/'PARTIAL_ACCEPTANCE.json');runtime=check_runtime(server);reuse=read(HERE/'SOURCE_REUSE.json');restore=B/'restored'
 for name,expected in read(H/'FILES_SHA256.json')['files'].items():need_file(H/name,expected)
 for name,row in read(B/'MEMBERS.json')['members'].items():need_file(restore/name,row['sha256'])
 for name,key in [('body.py','original_body_sha256'),('driver.py','original_driver_sha256'),('writer_policy.py','original_writer_policy_sha256')]:need_file(H/name,reuse[key])
 text=(H/'body.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='checked');source='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n';adapted=(HERE/'checked_record_body.py').read_text()
 assert hashlib.sha256(source.encode()).hexdigest()==reuse['original_checked_source_sha256'] and hashlib.sha256(adapted.encode()).hexdigest()==reuse['record_checked_source_sha256']
 reverted=adapted
 for a,b in reversed(reuse['replacements']):assert reverted.count(b)==1;reverted=reverted.replace(b,a)
 assert reverted==source and ast.dump(ast.parse(reverted))==ast.dump(ast.parse(source))
 os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 sys.path.insert(0,str(H));import body,driver,torch
 torch.set_num_threads(1)
 # torch.load receives an output path from HERE, so use a fully populated owned restore tree of original source bytes.
 for name,expected in read(H/'FILES_SHA256.json')['files'].items():
  target=restore/name
  if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((H/name).read_bytes())
  need_file(target,expected)
 context=dict(body.__dict__,HERE=restore,server_runtime=runtime);exec(compile(adapted,'checked_record_body.py','exec'),context)
 namespace=types.SimpleNamespace(**{k:v for k,v in body.__dict__.items() if not k.startswith('__')});namespace.checked=context['checked']
 scope=read(H/'screen_scope.json');proposal=scope['resource_proposal'];scope=dict(scope,runtime_cuda_visible_device=proposal['gpu_index'],runtime_gpu_uuid=proposal['gpu_uuid'])
 factory=driver.clone(driver.functions,HERE=restore);_,checked,_=factory(namespace,scope)
 results=[]
 for entry in scope['jobs']:
  if entry['id'] in server['accepted_new_ids']:
   result=checked(entry,scope);assert result is not None
   results.append({'id':entry['id'],'metrics':result['metrics'],'checkpoint_sha256':sha(restore/entry['output']/'model.pt')})
 assert not torch.cuda.is_initialized()
 return {'status':'RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS','local_torch':torch.__version__,'actual_server_runtime':runtime,'local_runtime_not_claimed_equal':True,'local_CUDA_initialized':False,'server_check_receipt_sha256':SERVER_SHA,'records':results}
if __name__=='__main__':print('PREPARED_ONLY: root review required before record replay; no CNN or inference')
