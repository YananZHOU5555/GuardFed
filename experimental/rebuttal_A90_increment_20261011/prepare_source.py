"""Reuse the adopted A80 editorial helpers; prepare the bounded A90 increment."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/rebuttal_A80_candidate_20261011'
DOC=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def pin(p):return dict(path=p.resolve().as_posix(),sha256=sha(p),bytes=p.stat().st_size)
source=(P/'build_and_check.py').read_text('utf8');tree=ast.parse(source)
def function(name):return ast.get_source_segment(source,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
root=json.loads((DOC/'ROOT_REVIEW.json').read_bytes())
assert root['author_review_only'] and root['A80_incorporated'] and root['original_comments']==24
oldpins=json.loads((P/'SOURCE_PINS.json').read_bytes())
fixed={k:oldpins[k] for k in ('A50_build_checker','original_writing_checker')}
fixed['reader_root']=pin(DOC/'ROOT_REVIEW.json')
for n in ('rebuttal_integrated_20261011.md','manuscript_insertions_integrated_20261011.md'):
 assert sha(DOC/n)==root['documents_sha256'][n]
 fixed[n]=pin(DOC/n)
fixed['prior_A80_table_root']=oldpins['A80_root']
for k,v in fixed.items():assert sha(Path(v['path']))==v['sha256'] and Path(v['path']).stat().st_size==v['bytes'],k
save('FIXED_INPUT_PINS.json',fixed)
header='''"""A90 author-review increment; unchanged original document-invariant checks."""
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,re,sys
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
PINS=json.loads((HERE/'SOURCE_PINS.json').read_bytes())
DOCS=('rebuttal_integrated_20261011.md','manuscript_insertions_integrated_20261011.md')
BASE=str(Path(PINS['tables']['path']).parent.as_posix())+'/'
'''
names=('file_sha','read','save','original_checks')
result=header+'\n\n'.join(function(n) for n in names)+'\n\nREUSE=original_checks()\n\n'+(H/'increment_body.py.txt').read_text('utf8')
compile(result,str(H/'build_and_check.py'),'exec')
for n in names:
 node=next(x for x in ast.parse(result).body if isinstance(x,ast.FunctionDef) and x.name==n)
 assert ast.get_source_segment(result,node)==function(n)
with (H/'build_and_check.py').open('x',encoding='utf8',newline='\n') as f:f.write(result)
with (H/'SOURCE_DIFF.patch').open('x',encoding='utf8',newline='\n') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),result.splitlines(True),fromfile=str(P/'build_and_check.py'),tofile=str(H/'build_and_check.py'))))
save('SOURCE_REUSE.json',dict(status='SOURCE_PREPARED_NO_DOCUMENTS_OR_STATISTICS_GENERATED',parent_path=(P/'build_and_check.py').relative_to(R).as_posix(),parent_sha256=sha(P/'build_and_check.py'),exact_reused_functions=list(names),original_document_invariant_statements_reused_unchanged=7,scope_change='A80 historical text retained; add only the adopted non-IID S-DFA scene and synchronize current A90 scope',generated_author_drafts=False,scientific_statistics_recomputed=False,SSH_CNN_fit_training_Git=0))
for n in ('prepare_source.py','bind_inputs.py','build_and_check.py'):compile((H/n).read_text('utf8'),str(H/n),'exec')
save('SOURCE_CHECK.json',dict(status='SOURCE_COMPILE_AND_EXACT_HELPER_REUSE_PASS_NOT_ACTUAL_DRAFT_CHECK',compiled=['prepare_source.py','bind_inputs.py','build_and_check.py'],exact_helpers=list(names),actual_A90_root_bound=False,author_draft_generated=False,source_statistics_recomputed=False))
names=['prepare_source.py','bind_inputs.py','build_and_check.py','increment_body.py.txt','FIXED_INPUT_PINS.json','SOURCE_DIFF.patch','SOURCE_REUSE.json','SOURCE_CHECK.json']
save('PREPARED_FILES_SHA256.json',dict(status='PREPARED_ONLY_ACTUAL_ROOT_REQUIRED_BEFORE_GENERATION',files={n:pin(H/n) for n in names}))
print(json.dumps(dict(status='SOURCE_PREPARED',source_seal_sha256=sha(H/'PREPARED_FILES_SHA256.json'),actual_generation=False)))
