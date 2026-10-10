"""Load the pinned accepted C70 table source with fixed C80 metadata changes."""
from pathlib import Path
import hashlib,json,sys,types
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def load(kind):
    if sys.flags.optimize:raise RuntimeError('Optimized Python is forbidden for evidence guards')
    if kind not in ('build','panels','verify_numeric'):raise ValueError('Unknown fixed source')
    spec=json.loads((HERE/'REBINDS.json').read_bytes())[kind]
    path=ROOT/spec['source']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise ValueError('Accepted C70 source changed')
    source=path.read_text(encoding='utf-8')
    for before,after,count in spec['replacements']:
        if source.count(before)!=count:raise ValueError('Source replacement boundary changed: '+before)
        source=source.replace(before,after)
    if hashlib.sha256(source.encode()).hexdigest()!=spec['effective_sha256']:raise ValueError('Fixed C80 source drift')
    module=types.ModuleType('C80_'+kind);module.__file__=str(HERE/(kind+'.py'))
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module

def expose(module,namespace):
    namespace.update({k:v for k,v in vars(module).items() if not k.startswith('__')})
