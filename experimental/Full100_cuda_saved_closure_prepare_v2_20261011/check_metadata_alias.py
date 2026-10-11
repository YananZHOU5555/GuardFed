"""Only metadata path fixtures: no filesystem scientific artifacts are accessed."""
from pathlib import Path
import ast,copy,hashlib,json
H=Path(__file__).resolve().parent
source=(H/'collect_remote.py').read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='add')
# Compile only the actual two metadata path statements; stop before hashing/member registration.
fn=copy.deepcopy(fn);fn.body=fn.body[:2]+[ast.Return(ast.Constant(True))];ast.fix_missing_locations(fn)
logical='/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/metadata.npz';canonical='/workspace/GuardFed-revision/data/celeba/derived/rgb64_v1/metadata.npz';digest='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
resolved={};links=set();hashes={};missing=set()
class FakePath:
 def __init__(self,p):self.p=p.p if isinstance(p,FakePath) else p
 def __str__(self):return self.p
 def __eq__(self,p):return isinstance(p,FakePath) and self.p==p.p
 def resolve(self):return FakePath(resolved.get(self.p,self.p))
 def is_file(self):return self.p not in missing
 def is_symlink(self):return self.p in links
def need(ok,m):
 if not ok:raise ValueError(m)
ns={'Path':FakePath,'sha':lambda p:hashes.get(str(p)),'need':need};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual add path gate','exec'),ns)
def reset():
 resolved.clear();resolved[logical]=canonical;links.clear();hashes.clear();hashes[canonical]=digest;missing.clear()
reset();assert ns['add'](logical,'source/metadata.npz') is True
assert ns['add']('/workspace/plain.py','source/plain.py') is True
negative=0
for kind in ['wrong_target','wrong_hash','canonical_alias','terminal_symlink','foreign_alias','missing']:
 reset()
 path=logical
 if kind=='wrong_target':resolved[logical]='/workspace/other/metadata.npz'
 elif kind=='wrong_hash':hashes[canonical]='0'*64
 elif kind=='canonical_alias':resolved[canonical]='/workspace/other/metadata.npz'
 elif kind=='terminal_symlink':links.add(logical)
 elif kind=='foreign_alias':path='/workspace/plain.py';resolved[path]='/workspace/other.py'
 else:missing.add(canonical)
 try:ns['add'](path,'synthetic')
 except ValueError:negative+=1
 else:raise AssertionError(kind+' accepted')
for f in H.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
old=H.parent/'Full100_cuda_saved_closure_prepare_20261011'
for n in ['audit_saved.py','saved_fit_binding.py','original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt','transport_once.py','SCOPE.json','PRIOR_ADOPTED.json']:
 assert (H/n).read_bytes()==(old/n).read_bytes(),n+' changed'
report={'status':'PASS_SOURCE_ONLY_METADATA_ALIAS_FIX','positive_fixtures':2,'negative_fixtures':negative,'audit_science_transport_scope_raw_unchanged':True,'original_logical_member_key_preserved':"remote[str(path)]=dict(meta,storage=storage)" in source,'source_sha256':hashlib.sha256((H/'collect_remote.py').read_bytes()).hexdigest(),'SSH_F_arrays_models_scientific_execution':False}
(H/'V2_SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
