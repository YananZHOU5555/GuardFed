from pathlib import Path
import json,hashlib,ast
R=Path.cwd();H=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011';F=R/'tmp/FL100_uniform_cuda_formal_adapter_prepare_20261011';P=R/'tmp/old800_cuda3_saved_audit_prepare_20261011';prep=json.loads((R/'tmp/FL100_cuda_runtime_caller_prepare_20261011/ROOT_PREPARATION.json').read_bytes());pins=json.loads((F/'SOURCE_PINS.json').read_bytes());cohort=json.loads((F/'EXACT100.json').read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for i in prep['canary_ids']:
 route=next(r for r in cohort['records'] if r['id']==i);pkg=pins['packages'][route['identity_source']];manifest=json.loads((R/pkg['manifest']['path']).read_bytes());row=manifest['records'][route['identity_index']];assert row['id']==i
 rows.append(dict(route=route,original_row=row,package=pkg))
put=lambda n,v:(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
put('SELECTED3_INPUTS.json',dict(status='FROZEN_FL_CANARY3_INPUTS_NOT_CUDA_RESULTS',ids=prep['canary_ids'],rows=rows,planned_namespace=prep['namespace'],formal_seal=prep['formal_source_seal_sha256'],cohort_sha256=prep['exact_cohort_sha256']))
# Full originals reused by exact bytes; small source files only.
for name in ['original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt']:(H/name).write_bytes((P/name).read_bytes())
s=(H/'original_fl_audit.py.txt').read_text();fn=next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='saved_fits_for_original_predict');(H/'saved_fit_binding.py').write_text('import copy,math\n\ndef need(ok,message):\n    if not ok:raise ValueError(message)\n\n'+fn+'\n',encoding='utf8')
p=json.loads((P/'SOURCE_PINS.json').read_bytes());p['formal_API_seal']={'path':str((F/'FILES_SHA256.json').relative_to(R)),'sha256':sha(F/'FILES_SHA256.json')};p['FL_original_saved_science']={'path':'tmp/fl_isolated_v3_saved_closure_prepare_20261011/saved_v2/saved_science.py','sha256':sha(R/'tmp/fl_isolated_v3_saved_closure_prepare_20261011/saved_v2/saved_science.py')};p['frozen_selection_sha256']=sha(H/'SELECTED3_INPUTS.json');put('SOURCE_PINS.json',p)
np={'path':None,'sha256':None};put('ROOT_INPUTS.template.json',dict(status='UNBOUND_NOT_AUTHORIZATION',root_authorized_transport=False,root_authorized_audit=False,base=None,exact_ids=prep['canary_ids'],source_review=np,transport=np,archive_sha256=None,deployment_sha256=None,cohort_sha256=None,canary_complete_sha256=None,accepted_sha256={i:None for i in prep['canary_ids']},future_root_adoption=np))
print(json.dumps({'selected':len(rows),'selection_sha256':sha(H/'SELECTED3_INPUTS.json')}))
