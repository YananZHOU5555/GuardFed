"""Only source inversion, unchanged science-loop AST, and synthetic JSON scope checks."""
from pathlib import Path
import ast, difflib, hashlib, importlib.util, json, textwrap
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
adapt=read(H/'SOURCE_ADAPTATIONS.json');checks={};patch=[]
for n,pin in adapt.items():
 parent=R/pin['parent_path'];before=parent.read_bytes();after=(H/n).read_bytes()
 assert sha(parent)==pin['parent_sha256']
 compile(after,str(H/n),'exec')
 spans=[dict(a=a,b=b,c=c,d=d,old=before[a:b].hex(),new=after[c:d].hex()) for tag,a,b,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if tag!='equal']
 undo=after
 for e in reversed(spans):assert undo[e['c']:e['d']].hex()==e['new'];undo=undo[:e['c']]+bytes.fromhex(e['old'])+undo[e['d']:]
 assert undo==before
 pin.update(sha256=sha(H/n),raw_inverse_spans=spans)
 checks[n]=dict(compiled=True,raw_inverse_exact=True,parent_sha256=sha(parent),sha256=sha(H/n))
 patch.extend(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile=pin['parent_path'],tofile=str((H/n).relative_to(R))))

def loop(path,needle):
 s=Path(path).read_bytes().decode();lines=s.splitlines(True)
 matches=[n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.For) and needle in lines[n.lineno-1]]
 assert len(matches)==1,(path,needle,len(matches))
 n=matches[0];return ''.join(lines[n.lineno-1:n.end_lineno])

loops={}
for n,needles in {'join_saved.py':['for rec in records:'], 'adopt_root.py':['for na in index[','for rec in records:'],
                 'verify_numeric.py':['for row in rows:','for record in records:'],
                 'verify_saved.py':["for view in ('native','raw','shared_calibration'):","for panel in tables['panels']:"]}.items():
 for needle in needles:
  before=loop(R/adapt[n]['parent_path'],needle);after=loop(H/n,needle)
  if n=='adopt_root.py' and needle=='for na in index[':after=after.replace("row['attack']=='FedSA'","row['attack']=='F Flip'")
  assert before==after,(n,needle)
  assert ast.dump(ast.parse(textwrap.dedent(before)))==ast.dump(ast.parse(textwrap.dedent(after)))
  loops[n+':'+needle]=dict(raw_exact=True,AST_exact=True,sha256=hashlib.sha256(before.encode()).hexdigest())
for n in ['execute_once.py','remote_cpu111_export.py','cpu_release_readonly.py']:
 assert (H/n).read_bytes()==(R/adapt[n]['parent_path']).read_bytes(),n
before=(R/adapt['remote_preflight.py']['parent_path']).read_bytes().decode()
after=(H/'remote_preflight.py').read_bytes().decode()
remove=lambda s:'\n'.join(l for l in s.splitlines() if not l.startswith(('candidate_ids=',"assert latest['receipt']==","assert latest['receipt_sha256']==","assert latest['all_transported_ids']==")))
assert remove(before)==remove(after),'Remote resource gates changed'

# No file reads, numerical values or scientific claims in this synthetic fixture.
spec=importlib.util.spec_from_file_location('source_scope_fixture_only',H/'accepted_scope.py');scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
oldids=['old_native_'+str(i) for i in range(370)];full=['old_Full_'+str(i) for i in range(100)]
old=dict(accepted_new_ids=oldids,accepted_reused_ids=full,new_count=370,invalid=[],records=[dict(id=i,fixture=True) for i in oldids+full])
raw=dict(accepted_new_ids=oldids+scope.FIXED+['pending_extra'],accepted_reused_ids=full,new_count=381,invalid=[],records=old['records']+[dict(id=i,fixture=True) for i in scope.FIXED+['pending_extra']])
enc=lambda v:json.dumps(v,indent=2).encode()
derived=scope.derive_scope(enc(old),enc(raw));assert derived['new_count']==380 and derived['records'][:470]==old['records'] and derived['pending_closed_ids']==['pending_extra']
cases={}
for name,mutate in [('old_row_changed',lambda x:x['records'][0].update(fixture=False)),('missing_fixed',lambda x:x['accepted_new_ids'].remove(scope.FIXED[-1])),('duplicate_id',lambda x:x['accepted_new_ids'].append(x['accepted_new_ids'][0])),('invalid_present',lambda x:x['invalid'].append('fixture'))]:
 item=json.loads(enc(raw));mutate(item)
 try:scope.derive_scope(enc(old),enc(item))
 except AssertionError:cases[name]=True
 else:raise AssertionError(name)
try:scope.derive_scope(enc(old),enc(raw).replace(b'"fixture": true',b'"fixture" : true',1))
except AssertionError:cases['old_serialized_token_changed']=True
else:raise AssertionError('Token mutation accepted')
assert read(H/'ROOT_BINDING.template.json')['adoption'] is None
assert not any((H/n).exists() for n in ['EXECUTION_INPUTS.json','ROOT_BINDING.json','ROOT_ACTUAL_PINS.json','ACTUAL_HANDOFF.json','ROOT_ADOPTION.json','records.json','tables.json'])
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
for n,obj in [('SOURCE_ADAPTATIONS.json',adapt),('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_INVERSE_CORE_LOOPS_RESOURCE_GATES_AND_SYNTHETIC_METADATA_PASS_NOT_SCIENCE',files=checks,unchanged_loops=loops,resource_gate_parent_exact=True,synthetic_refusals=cases,future_native380=None,future_saved380=None,future_F80=None,SSH=False,F_access=False,scientific_execution=False))]:
 (H/n).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8',newline='\n')
(H/'SOURCE_DIFF.patch').write_bytes(''.join(patch).encode())
print(json.dumps(dict(status='SOURCE_ONLY_PASS',inverses=len(checks),raw_AST_loops=len(loops),synthetic_refusals=len(cases),future_native380=None)))
