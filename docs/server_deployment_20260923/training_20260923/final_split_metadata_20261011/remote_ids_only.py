
import ast,copy,datetime,hashlib,json,pathlib,sys,types
import numpy as actual_np
P=pathlib.Path
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
p=json.load(sys.stdin)
require(sha('/etc/vast-agents-guide.md')==p['guide_sha256'],'Guide changed')
repo=P('/workspace/GuardFed-celeba-expanded');cache=repo/'data/celeba/derived/rgb64_v1'
source=P(p['original_source']);require(sha(source)==p['original_source_sha256'],'Original source changed')
paths={repo/k:v for k,v in p['input_hashes'].items()}
before={str(k):sha(k) for k in paths}
require(before=={str(k):v for k,v in paths.items()},'Metadata identity drift')
tree=ast.parse(source.read_text());fns={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
fn=copy.deepcopy(fns['metadata'])
stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='read_prefix' for c in ast.walk(n)))
require(stop==8,'Original non-label prefix changed')
prefix=copy.deepcopy(fn.body[:stop])
require(not any(isinstance(n,ast.Constant) and n.value in ('Smiling','Male') for stmt in prefix for n in ast.walk(stmt)),'Label key in selected prefix')
fn.name='metadata_ids_only';fn.body=prefix+[ast.parse('return ids,split,manifest,cache').body[0]]
reads=[]
class RestrictedNpz:
 def __init__(self,*a,**kw):self.z=actual_np.load(*a,**kw)
 def __enter__(self):return self
 def __exit__(self,*args):self.z.close()
 def __getitem__(self,key):
  require(key in ('image_id','split'),'Label-array decode refused: '+str(key));reads.append(key);return self.z[key]
proxy=types.SimpleNamespace(load=RestrictedNpz,array_equal=actual_np.array_equal,arange=actual_np.arange,dtype=actual_np.dtype,flatnonzero=actual_np.flatnonzero,asarray=actual_np.asarray)
ns=dict(np=proxy,read=lambda q:json.loads(P(q).read_text()),require=require,hashlib=hashlib)
module=ast.fix_missing_locations(ast.Module(body=[fn,copy.deepcopy(fns['array_sha'])],type_ignores=[]))
exec(compile(module,'<original-eight-non-label-statements>','exec'),ns)
ids,split,manifest,cache=ns['metadata_ids_only'](repo)
require(reads==['image_id','split'],'Unexpected member reads')
available=actual_np.load(cache/'available.npy',mmap_mode='r',allow_pickle=False)
require(available.shape==(202599,) and available.all(),'Unavailable images; no silent omission')
lines=(repo/'data/celeba/list_eval_partition.txt').read_text().splitlines()
official=[line.split() for line in lines]
require(len(official)==len(ids),'Official row count')
require(all(int(name.split('.')[0])==int(ids[i]) and int(part)==int(split[i]) for i,(name,part) in enumerate(official)),'Official split/order mismatch')
parts={}
for k,name in ((0,'train'),(1,'valid'),(2,'candidate_official_test')):
 selected=ids[split==k]
 parts[name]=dict(partition=k,n=len(selected),first_image_id=int(selected[0]),last_image_id=int(selected[-1]),dtype=str(selected.dtype),ordered_image_ids_sha256=ns['array_sha'](selected))
require(parts['train']['n']==162770 and parts['train']['ordered_image_ids_sha256']=='46d42484d5b5f53af8747fcf44ee11a0b051af27f10383d32254faee0311bc99','Train identity mismatch')
require(parts['valid']['n']==19867 and parts['valid']['ordered_image_ids_sha256']=='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf','Valid identity mismatch')
require(parts['candidate_official_test']['n']==19962,'Candidate target size mismatch')
require(sum(x['n'] for x in parts.values())==len(ids) and len(actual_np.unique(ids))==len(ids),'Not a disjoint complete split')
after={str(k):sha(k) for k in paths};require(after==before,'Metadata changed during observation')
report=dict(status='READONLY_CANDIDATE_OFFICIAL_TARGET_ID_METADATA_MEASURED_NOT_FROZEN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 original_source_sha256=p['original_source_sha256'],original_non_label_prefix_statements=8,
 original_prefix_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False).encode()).hexdigest(),
 original_array_sha_AST_sha256=hashlib.sha256(ast.dump(fns['array_sha'],include_attributes=False).encode()).hexdigest(),
 input_hashes_before=before,input_hashes_after=after,decoded_npz_members=reads,
 metadata_zip_whole_file_hashed=True,label_array_values_decoded=False,label_array_values_accessed=False,
 full_celeba_loader_called=False,image_pixels_opened=False,model_loaded=False,root_fit=False,
 test_inference=False,test_metrics=False,new_training=False,primary_endpoint_selected=False,protocol_frozen=False,
 prior_official_test_exposure_unchanged=True,all_images_available=True,official_text_order_match=True,
 partition_count=3,all_partition_IDs_disjoint_and_complete=True,partitions=parts,no_remote_writes=True)
print(json.dumps(report))
