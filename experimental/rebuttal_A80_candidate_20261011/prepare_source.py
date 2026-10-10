"""Reuse A60 document helpers verbatim; prepare only the bounded A80 editorial source."""
from pathlib import Path
import ast,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
OLD=R/'tmp/rebuttal_A60_reader_integration_20261011/build_and_check.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=OLD.read_text('utf-8');tree=ast.parse(source)
def function(name):
    n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,n)
body=(H/'increment_body.py.txt').read_text('utf-8')
header='''"""A80 author-review increment; original document invariant checks reused unchanged."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,re,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
PINS=json.loads((HERE/'SOURCE_PINS.json').read_bytes())
DOCS=('rebuttal_integrated_20261011.md','manuscript_insertions_integrated_20261011.md')
BASE=str(Path(PINS['tables']['path']).parent.as_posix())+'/'
'''
helpers='\n\n'.join(function(n) for n in ('file_sha','read','save','original_checks'))
result=header+helpers+'\n\nREUSE=original_checks()\n\n'+body
compile(result,str(H/'build_and_check.py'),'exec')
for n in ('file_sha','read','save','original_checks'):
    node=next(n2 for n2 in ast.parse(result).body if isinstance(n2,ast.FunctionDef) and n2.name==n)
    assert ast.get_source_segment(result,node)==function(n)
with (H/'build_and_check.py').open('x',encoding='utf-8',newline='\n') as f:f.write(result)
with (H/'REUSE_CHECK.json').open('x',encoding='utf-8') as f:
    json.dump({'status':'SOURCE_ONLY_COMPILE_AND_HELPER_BYTE_EXACT_NOT_EXECUTED','parent':OLD.relative_to(R).as_posix(),
               'parent_sha256':sha(OLD),'generated_sha256':sha(H/'build_and_check.py'),
               'exact_reused_functions':['file_sha','read','save','original_checks'],
               'original_document_invariant_AST_reused_at_runtime':True,'actual_A80_pins_bound':False,
               'generated_author_drafts':False,'scientific_statistics_recomputed':False},f,indent=2);f.write('\n')
print(json.dumps({'build_source':(H/'build_and_check.py').relative_to(R).as_posix(),'sha256':sha(H/'build_and_check.py'),'compile':True,'actual_build_executed':False}))
