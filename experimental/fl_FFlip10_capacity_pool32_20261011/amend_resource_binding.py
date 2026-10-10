"""Source-only strict resource-assertion binding correction. No live/scientific execution."""
from pathlib import Path
import ast,json,hashlib,marshal,difflib
H=Path(__file__).resolve().parent;O=H.parent/'fl_FFlip10_capacity_cpu11_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:dict(sha256=sha(p),bytes=Path(p).stat().st_size)
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def funcs(s):return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
oldpkg=sha(H/'FILES_SHA256.json');assert oldpkg=='90e7b54047b0a13e19bb58db1c025f298e56e26c2378c4be09977a0d90e8b418'
# Preserve source-only candidate before the discovered operational dependency correction.
changes=['candidate.py','FILES_SHA256.json','SOURCE_CHECK.json','INVERSE_EDITS.json','SOURCE_DIFF.patch','HANDOFF.json','DELIVERY_FILES_SHA256.json','saved_v2/STATIC_SOURCE_SHA256.json','saved_v2/contract.py','saved_v2/linux_saved_remote.py','saved_v2/transport_remote.py']
for rel in changes:
 out=H/'prepared_v1'/rel;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('xb') as f:f.write((H/rel).read_bytes())
before="    text = (HERE / 'originals/replay.py').read_text(encoding='utf-8')\n    tree = ast.parse(text)"
oldassert="require(len(os.sched_getaffinity(0)) == 8, 'Process must be bound to its eight coordinated CPUs')"
newassert="require(sorted(os.sched_getaffinity(0)) == list(range(32, 64)), 'Process must use fixed32-core scheduling pool with eight compute threads')"
after="    text = (HERE / 'originals/replay.py').read_text(encoding='utf-8')\n    resource_assertion = "+repr(oldassert)+"\n    original_gate = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'resource_gate')\n    require(text.count(resource_assertion) == 1 and resource_assertion in ast.get_source_segment(text, original_gate), 'Original operational resource assertion changed')\n    text = text.replace(resource_assertion, "+repr(newassert)+")\n    tree = ast.parse(text)"
s=(H/'candidate.py').read_text();assert s.count(before)==1;s=s.replace(before,after)
# Use original CRLF style; code only operational source binding changes.
(H/'candidate.py').write_bytes(s.replace('\n','\r\n').encode())
assert s.replace(after,before)==(H/'prepared_v1/candidate.py').read_text()
compile(s,'candidate.py','exec')
# Obtain only private AST construction functions; no candidate import, Torch/NumPy or evaluator execution.
def runtime_nodes(path):
 text=path.read_text();fs=funcs(text);ns={'HERE':H,'ast':ast}
 exec(fs['require']+'\n'+fs['runtime_nodes'],ns);return ns['runtime_nodes']()
a=runtime_nodes(O/'candidate.py');b=runtime_nodes(H/'candidate.py')
oldnodes={n.name:n for n in a.body};newnodes={n.name:n for n in b.body};assert oldnodes.keys()==newnodes.keys()
for n in oldnodes:
 if n=='resource_gate':
  old=ast.unparse(oldnodes[n]);new=ast.unparse(newnodes[n]);assert old.count(oldassert)==1 and old.replace(oldassert,newassert)==new
 else:assert ast.dump(oldnodes[n])==ast.dump(newnodes[n]),n
compile(b,'private_original_binding_source_only','exec')
# Original source and science remain byte-identical. No science function calls.
for rel in read(H/'FILES_SHA256.json')['files']:
 if rel.startswith('originals/'):assert (H/rel).read_bytes()==(O/rel).read_bytes()
obj=read(H/'FILES_SHA256.json');obj['files']['candidate.py']=pin(H/'candidate.py');save(H/'FILES_SHA256.json',obj);pkg=sha(H/'FILES_SHA256.json')
for rel in ['saved_v2/contract.py','saved_v2/linux_saved_remote.py','saved_v2/transport_remote.py']:
 p=H/rel;b=p.read_bytes();assert oldpkg.encode() in b;p.write_bytes(b.replace(oldpkg.encode(),pkg.encode()));compile(p.read_bytes(),rel,'exec')
obj=read(H/'saved_v2/STATIC_SOURCE_SHA256.json');obj['files']={r:pin(H/'saved_v2'/r) for r in obj['files']};save(H/'saved_v2/STATIC_SOURCE_SHA256.json',obj)
inv=read(H/'INVERSE_EDITS.json');inv['new_package_sha256']=pkg
inv['edits']['candidate.py'].append(dict(before=before,after=after,count=1))
for rel in ['saved_v2/contract.py','saved_v2/linux_saved_remote.py','saved_v2/transport_remote.py']:
 for change in inv['edits'][rel]:
  if change['after']==oldpkg:change['after']=pkg
save(H/'INVERSE_EDITS.json',inv)
# Recheck only changed source restoration/package-dependent consumer wiring, not old capacity/science fixtures.
for rel in ['candidate.py','saved_v2/contract.py','saved_v2/linux_saved_remote.py','saved_v2/transport_remote.py']:
 current=(H/rel).read_text()
 for item in reversed(inv['edits'][rel]):
  assert current.count(item['after'])==item['count'];current=current.replace(item['after'],item['before'])
 assert current==(O/rel).read_text(),rel
checker=read(H/'SOURCE_CHECK.json');checker['package_sha256']=pkg;checker['candidate_changed_only']=['runtime_nodes','authorize','exclusive_cpus','run'];checker['resource_binding_only_one_original_operational_assertion']=True;checker['other_private_runtime_function_AST_exact']=True;checker['original_resource_gate_file_bytes_unchanged']=True;checker['unbound_original_main_affinity8_not_executed']=True;checker['whole_saved_check_single_CPU_unchanged']=True;save(H/'SOURCE_CHECK.json',checker)
handoff=read(H/'HANDOFF.json');handoff['package_sha256']=pkg;handoff['saved_seal_sha256']=sha(H/'saved_v2/STATIC_SOURCE_SHA256.json');handoff['source_check_sha256']=sha(H/'SOURCE_CHECK.json');handoff['expected_root_source_review_schema']['package_sha256']=pkg;handoff['required_SMT_siblings']=handoff.pop('required_SMТ_siblings');handoff['resource_gate_private_binding_exact_fixed32_pool']=True;save(H/'HANDOFF.json',handoff)
diff=[]
for rel in inv['edits']:
 old=(O/rel).read_text();new=(H/rel).read_text()
 if old!=new:diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='cpu11/'+rel,tofile='pool32/'+rel))
(H/'SOURCE_DIFF.patch').write_text(''.join(diff),encoding='utf8')
with (H/'OPERATIONAL_AMENDMENT.md').open('a',encoding='utf8') as f:f.write('\n调用链补齐：原replay.resource_gate包含affinity恰8断言，已在candidate.runtime_nodes私有绑定里仅对该完整语句作严格唯一匹配，替换为固定32–63池。原replay文件字节保持；其它7个绑定函数AST原样，该resource_gate仅这一operational语句不同；Torch8/interop1、线程不逃逸及其它门均保持。原replay.main/build-plan的旧affinity8描述不在当前绑定入口执行；whole saved checker独立单CPU要求原样。初次未执行源留prepared_v1，修正未涉及实际任务或科学数据。\n')
# Validate final source seals without importing any runtime module.
for seal,prefix in [('FILES_SHA256.json',''),('runtime/SOURCE_ONLY_FILES_SHA256.json','runtime/'),('saved_v2/STATIC_SOURCE_SHA256.json','saved_v2/')]:
 for rel,v in read(H/seal)['files'].items():assert pin(H/(prefix+rel))==v
files={p.relative_to(H).as_posix():pin(p) for p in sorted(H.rglob('*')) if p.is_file() and p!=H/'DELIVERY_FILES_SHA256.json'};save(H/'DELIVERY_FILES_SHA256.json',dict(status='SOURCE_ONLY_POOL32_V3_DELIVERY_WITH_OPERATIONAL_BINDING_FIXED',files=files))
print(json.dumps(dict(package=pkg,candidate=sha(H/'candidate.py'),runtime=sha(H/'runtime/SOURCE_ONLY_FILES_SHA256.json'),saved=sha(H/'saved_v2/STATIC_SOURCE_SHA256.json'),source_check=sha(H/'SOURCE_CHECK.json'),handoff=sha(H/'HANDOFF.json'),delivery=sha(H/'DELIVERY_FILES_SHA256.json'),members=len(files))))
