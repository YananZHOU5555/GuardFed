from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys,textwrap
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;R=B.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
s=importlib.util.spec_from_file_location('p41check',B/'publish_increment41.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
c=read(B/'ACTUAL_CLOSED_INPUTS.json');d={k:read(R/v['path']) for k,v in c['roots'].items()};m.closed_guard(c);m.evidence_guard(d['native'],d['FL'],d['Hybrid'],d['state'])
refused=[]
for key,value in [('parent_commit','3601c9dfca63dc1c6203aceb2c7fc066faa630fa'),('counts',dict(m.COUNTS,three_view=180)),('roots',{})]:
 x=copy.deepcopy(c);x[key]=value
 try:m.closed_guard(x)
 except AssertionError:refused.append(key)
 else:raise AssertionError('bad closure accepted')
for key,value in [('total_new_strict_and_offserver',172),('new_ids',[]),('test',True)]:
 x=copy.deepcopy(d['native']);x[key]=value
 try:m.evidence_guard(x,d['FL'],d['Hybrid'],d['state'])
 except AssertionError:refused.append(key)
 else:raise AssertionError('bad native accepted')
source=m.BASE.read_text('utf8');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main');transport=next(n for n in node.body if isinstance(n,ast.Try));block=textwrap.dedent('\n'.join(source.splitlines()[transport.lineno-1:transport.end_lineno]))
assert ast.dump(ast.parse(block).body[0])==ast.dump(transport)
sp=importlib.util.spec_from_file_location('v41check',B/'verify_increment41.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
original=v.source.read_text('utf8');start="    payload = git('cat-file'";stop='    changed = set'
assert v.text[v.text.index(start):v.text.index(stop)]==original[original.index(start):original.index(stop)]
for f in B.glob('*.py'):ast.parse(f.read_text('utf8'))
proof=dict(status='PASS_MINIMAL_41_BOUNDARY_AND_ORIGINAL_TRANSPORT_SOURCE_CHECKS',positive_guard_calls=2,refusals=refused,transport_source_sha256=m.BASE_SHA,transport_block_sha256=hashlib.sha256(block.encode()).hexdigest(),transport_AST_exact=True,verifier_original_blob_loop_byte_exact=True,Git_mutations=0,SSH=0,CNN=0)
(B/'SOURCE_REUSE_CHECK.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(proof))
