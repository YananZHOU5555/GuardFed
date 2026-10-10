"""Validate literal root publication bindings without executing Git helpers."""
from pathlib import Path
import ast,hashlib,json,re
OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for name in ('commit_and_push.py','register_verification.py'):
    wrapper=OWN/name;tree=ast.parse(wrapper.read_text('utf8'))
    bindings=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bindings' for t in n.targets)))
    original=ROOT/'tmp/publication46_root_20261010'/name;source=original.read_text('utf8')
    for key in bindings:assert source.count(key)==1,key
    bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],source)
    ast.parse(bound)
    rows.append(dict(path=wrapper.relative_to(ROOT).as_posix(),sha256=sha(wrapper),original_sha256=sha(original),bindings=len(bindings),syntax_pass=True))
out=OWN/'SOURCE_BINDING_CHECK.json'
with out.open('x',encoding='utf8') as f:
    json.dump(dict(status='SOURCE_BINDING_ONLY_NO_PUBLICATION_EXECUTED',helpers=rows,Git_executed=False),f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),bindings=[r['bindings'] for r in rows])))
