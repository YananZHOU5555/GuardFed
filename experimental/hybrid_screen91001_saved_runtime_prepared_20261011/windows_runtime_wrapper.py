"""Windows-only engineering repair: original canonical helper, original saved checker.

The failed first attempt imported Linux resource through runtime_originals before
the first record. Windows consumes only rt.canonical; Linux metadata helpers are
not called. Keep that failure and use a new output; no package or science edits.
"""
from pathlib import Path
import ast,hashlib,json,runpy,sys,types

R=Path(__file__).resolve().parents[2]
C=R/'tmp/hybrid_screen91001_single_replay_prepared_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sys.platform=='win32' and '--mode' in sys.argv
assert sys.argv[sys.argv.index('--mode')+1]=='windows-saved-output'
assert '--allow-saved-output-zero-fit' in sys.argv and '--allow-original-cached-root-refit' not in sys.argv
assert sha(C/'FILES_SHA256.json')=='9be2ce96b4548144903e9e928567efa96565326917e2363f1ad11712ca44015d'
assert sha(C/'check_saved.py')=='9408f64db4fb68d53eee4d882d69f3d0a3a0025baedbcb163823b3794a328934'
assert sha(C/'originals/replay.py')=='8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803'
source=(C/'check_saved.py').read_text(encoding='utf8')
rt_fields=[n.attr for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='rt']
assert sorted(rt_fields)==['canonical','canonical','metadata']
tree=ast.parse(source)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
linux_branch=next(n for n in ast.walk(main) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='linux_mode' and any(isinstance(x,ast.Attribute) and x.attr=='metadata' for x in ast.walk(n)))
assert not any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='rt' for item in linux_branch.orelse for n in ast.walk(item))
node=next(n for n in ast.parse((C/'originals/replay.py').read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef) and n.name=='canonical')
ns={'hashlib':hashlib,'json':json}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged original canonical helper>','exec'),ns)
sys.path.insert(0,str(C))
import candidate as c
original=c.runtime_originals
def saved_runtime_only(evaluator):
    assert sys.platform=='win32'
    return types.SimpleNamespace(canonical=ns['canonical'])
c.runtime_originals=saved_runtime_only
try:
    sys.argv[0]=str(C/'check_saved.py')
    runpy.run_path(str(C/'check_saved.py'),run_name='__main__')
finally:
    c.runtime_originals=original
