"""Offserver raw archive/member SHA and CPU tensor identity; no claim of CUDA runtime replay."""
from pathlib import Path
import hashlib,json,tarfile,os,sys,platform,datetime
B=Path(__file__).resolve().parent;H=B.parent
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
r=read(B/'BACKUP_SHA256.json');a=read(B/'PARTIAL_ACCEPTANCE.json')
assert sha(B/'hybrid_after15_delta.tar.gz')==r['archive_sha256']=='c65f31ce85734f308d9b93e35259d96d7a612499662e7473b1a1352881203575'
assert sha(B/'MEMBERS.json')==r['inventory_sha256'] and sha(B/'PARTIAL_ACCEPTANCE.json')==r['acceptance_sha256']
m=read(B/'MEMBERS.json')['members'];out=B/'restored';assert not out.exists()
with tarfile.open(B/'hybrid_after15_delta.tar.gz') as t:
 assert len(t.getnames())==len(set(t.getnames()))==r['member_count']==23 and set(t.getnames())==set(m)|{'MEMBERS.json'}
 for x in t.getmembers():
  assert x.isfile() and not x.name.startswith('/') and '..' not in Path(x.name).parts
  data=t.extractfile(x).read();expect={'sha256':r['inventory_sha256'],'size':(B/'MEMBERS.json').stat().st_size} if x.name=='MEMBERS.json' else m[x.name]
  assert len(data)==expect['size'] and hashlib.sha256(data).hexdigest()==expect['sha256']
  f=out/x.name;f.parent.mkdir(parents=True,exist_ok=True)
  with f.open('xb') as stream:stream.write(data)
assert sha(H/'FILES_SHA256.json')==r['source_seal_sha256']
for rel,expected in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/rel)==expected
sys.path.insert(0,str(H));import body,torch
torch.set_num_threads(1)
assert sha(Path(body.__file__))==a['body_sha256']
scope=read(H/'screen_scope.json');records=[]
for e in scope['jobs']:
 if e['id'] not in r['accepted_new_ids']:continue
 p=out/e['output'];receipt=read(p/'acceptance.json');state=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
 assert state and all(torch.isfinite(x).all() for x in state.values())
 assert body.tensor_sha(state)==receipt['checkpoint_tensor_sha256']
 assert sha(p/'model.pt')==next(row['model_sha256'] for row in a['records'] if row['id']==e['id'])
 assert read(p/'native_replay.json')['checkpoint_tensor_sha256']==body.tensor_sha(state)
 records.append({'id':e['id'],'checkpoint_tensor_sha256':body.tensor_sha(state),'model_sha256':sha(p/'model.pt')})
assert not torch.cuda.is_initialized() and a['source_host']!=platform.node()
proof={'status':'ORIGINAL_SERVER_STRICT_PLUS_OFFSERVER_ALL_MEMBERS_AND_CPU_TENSORS_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_accepted':15,'accepted_new_ids':r['accepted_new_ids'],'accepted_new':len(records),'records':records,'archive_sha256':r['archive_sha256'],'inventory_sha256':r['inventory_sha256'],'server_acceptance_sha256':r['acceptance_sha256'],'member_count':23,'local_torch':torch.__version__,'local_CUDA_initialized':False,'original_full_checked_replayed_locally':False,'limitation':'Original checked asserts actual cu128 runtime/device; CPU host verified every byte and CPU tensor identities, separate record-layer replay pending root review.','source_seal_sha256':r['source_seal_sha256'],'source_members_verified':len(read(H/'FILES_SHA256.json')['files']),'different_host':True,'source_host':a['source_host'],'verification_host':platform.node()}
with (B/'OFFSERVER_MEMBER_TENSOR_PROOF.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'proof_sha256':sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),'accepted_new':len(records),'runtime':proof['local_torch']}))
