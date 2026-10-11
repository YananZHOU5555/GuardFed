"""Independent root source review: no array/scientific execution or sealed-file write."""
from pathlib import Path
import ast,hashlib,json
R=Path.cwd();H=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011';D=R/'tmp/FL100_cuda3_saved_closure_root_review_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=H/'FILES_SHA256.json';expected='88d9b7d39df99fa1da79cc403e6009deef67f92c2f2691ad3f133fc7b7e35ac8'
assert sha(seal)==expected and not D.exists()
files=json.loads(seal.read_bytes())['files']
for n,p in files.items():assert Path(n).name==n and sha(H/n)==p['sha256'] and (H/n).stat().st_size==p['bytes']
source=(H/'check_source.py').read_text();tree=ast.parse(source)
removed=[];kept=[]
for n in tree.body:
 if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='write_text':
  assert "SOURCE_CHECK.json" in ast.unparse(n);removed.append(ast.unparse(n));continue
 kept.append(n)
assert len(removed)==1
ns={'__file__':str(H/'check_source.py'),'__name__':'__root_readonly_check__'}
exec(compile(ast.Module(body=kept,type_ignores=[]),'<readonly-source-check>','exec'),ns)
for n,p in files.items():assert sha(H/n)==p['sha256'],'Sealed file modified'
assert ns['report']['scientific_statements_executed']==ns['report']['scientific_imports']==ns['report']['arrays_read']==0
D.mkdir()
review=dict(status='ROOT_SOURCE_REVIEW_FL_FIXED3_SAVED_CLOSURE_PASS_RUNTIME_NOT_ADOPTED',source_adoptable=True,
 source_seal_sha256=expected,blocking_findings=[],sealed_members=len(files),
 checks=ns['report']['checks'],readonly_check_removed_only_source_check_write=True,
 reviewed=['transport current guide/CPU110/selected queue absence/F volume/all member SHA and exact fixed3 selected original rows',
 'metadata original FL11 roles and FFlip earlier3f1c/26cf package; original route ensure_asciiFalse vs receipt ensure_asciiTrue contracts',
 'actual CUDA/native/70round/split/checkpoint/source/binding and original Linux whole proof; all3 IDs fixed before outcome',
 'raw original seven statement science tail and original FL saved fit restoration; only offserver zero-fit, no refit/CNN'],
 science_executed=False,arrays_read=False,F_access=False,SSH=False,root_results_adopted=False)
with (D/'SOURCE_REVIEW.json').open('x',encoding='utf8',newline='\n') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps({'path':str(D/'SOURCE_REVIEW.json'),'sha256':sha(D/'SOURCE_REVIEW.json'),'source_adoptable':True,'sealed_members':len(files)}))
