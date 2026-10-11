"""Bounded standard-library-only source check; no F, arrays, Torch, NumPy or audit execution."""
from pathlib import Path
import ast,copy,hashlib,json,textwrap
H=Path(__file__).resolve().parent
def need(ok,msg):
    if not ok:raise ValueError(msg)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def segment(path,name):
    s=path.read_text('utf8');n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(s,n),n
pins=json.loads((H/'SOURCE_PINS.json').read_bytes())
for n in ('original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt'):
    need(sha(H/n)==pins[n]['sha256'],'Changed original source')
old,_=segment(H/'original_fl_audit.py.txt','saved_fits_for_original_predict')
new,_=segment(H/'saved_fit_binding.py','saved_fits_for_original_predict')
inverse=json.loads((H/'FIT_HELPER_INVERSE.json').read_bytes());restored=new
for a,b in reversed(inverse['exact_replacements']):need(b in restored,'Missing inverse metadata edit');restored=restored.replace(b,a)
need(restored==old and hashlib.sha256(old.encode()).hexdigest()==inverse['original_function_sha256'],'Fit helper original inverse failed')
_,node=segment(H/'original_fl_audit.py.txt','saved_output_block')
ns=dict(ast=ast,copy=copy,need=need);exec(compile(ast.Module(body=[node],type_ignores=[]),'original AST selector only','exec'),ns)
original=ast.parse(textwrap.dedent((H/'original_full_saved_science.py.txt').read_text('utf8'))).body[0]
block=ns['saved_output_block'](original)
need([ast.dump(n,include_attributes=False) for n in block.body]==[ast.dump(original.body[i],include_attributes=False) for i in (0,1,4,5,6,7,8)],'Seven original science statements changed')
selected={'original_evaluator.py.txt':['canonical_sha','check_binary','group_metrics','_seal_fit','predict_views','evaluate_frozen_predictions'],'original_replay.py.txt':['array_sha','read_prefix','check_native']}
for file,names in selected.items():
    for name in names:
        _,node=segment(H/file,name)
        need(not any(isinstance(n,ast.Call) and ((isinstance(n.func,ast.Attribute) and n.func.attr in ('fit_views','fit_group_thresholds','model_margins','extract_and_predict')) or (isinstance(n.func,ast.Name) and n.func.id=='fit_views')) for n in ast.walk(node)),'Scientific fit/CNN in selected function')
for n in ('audit_saved.py','saved_fit_binding.py','check_source.py','transport_once.py','collect_remote.py'):compile((H/n).read_bytes(),str(H/n),'exec')
tree=ast.parse((H/'audit_saved.py').read_text('utf8'));imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
need(not any((isinstance(n,ast.Import) and any(a.name.split('.')[0]=='torch' for a in n.names)) or (isinstance(n,ast.ImportFrom) and (n.module or '').startswith('torch')) for n in imports),'Torch imported by audit')
template=json.loads((H/'ROOT_INPUTS.template.json').read_bytes());need(template['root_authorized_audit'] is False and all(v is None for v in template['accepted_sha256'].values()),'Future evidence accidentally filled')
scope=json.loads((H/'SCOPE.json').read_bytes())
need(len(scope['ids'])==len(set(scope['ids']))==100 and len(scope['delta_ids'])==98 and len(scope['reused_adopted_ids'])==2,'Full100/delta98 scope')
need(set(scope['delta_ids']).isdisjoint(scope['reused_adopted_ids']) and set(scope['delta_ids'])|set(scope['reused_adopted_ids'])==set(scope['ids']),'Not an exact difference')
need([len(g['full_ids']) for g in scope['groups']]==[5,47,48],'Mixed producer scope')
need(len({(r['distribution'],r['attack'],r['seed']) for r in scope['cells']})==100,'Duplicate seed/cell')
need({r['seed'] for r in scope['cells']}==set(range(91001,91011)),'Seed grid changed')
need(len({r['attack'] for r in scope['cells']})==5 and {r['distribution'] for r in scope['cells']}=={'IID','non-IID'},'Cell grid changed')
for n in ('original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt','saved_fit_binding.py'):
 need((H/n).read_bytes()==(H.parent/'uniform_cuda7_saved_audit_prepare_20261011'/n).read_bytes(),'Original7 science bytes changed')
# The actual numerical block is selected by the same raw original selector.
old_audit=(H.parent/'uniform_cuda7_saved_audit_prepare_20261011/audit_saved.py').read_text()
new_audit=(H/'audit_saved.py').read_text()
a=old_audit.index("            fits=saved_fits_for_original_predict");b=old_audit.index("            context=dict",a)
raw=old_audit[a:b];need(raw==''.join('    '+line+'\n' for line in new_audit[new_audit.index('        fits=saved_fits_for_original_predict'):new_audit.index('        context=dict')].splitlines()),'Saved fit/array scientific preconditions changed')
collect=(H/'collect_remote.py').read_text();need("os.sched_getaffinity(0)=={110}" in collect and 'len(aff)<=16 and siblings&aff' in collect,'CPU110/SMT guard lost')
need('supervisorctl' not in collect and "snapshot.get('active') not in group['full_ids']" in collect,'Wrong all500 terminal assumption')
need("==98,'Delta must contain exactly98 new arrays'" in collect,'Delta array invariant absent')
report=dict(status='PASS_SOURCE_ONLY_NOT_ARRAY_AUDIT',compiled=5,original7_science_files_raw_exact=True,Full100_grid=True,delta98=True,producer_counts=[5,47,48],worker_exit_evidence='Frozen parent append only after subprocess returncode0; no fabricated per-task exit artifact',original_functions=selected,original_function_AST_sha256={file:{name:hashlib.sha256(ast.dump(segment(H/file,name)[1],include_attributes=False).encode()).hexdigest() for name in names} for file,names in selected.items()},seven_original_saved_output_statements_AST_exact=True,selected_block_AST_sha256=hashlib.sha256(ast.dump(block,include_attributes=False).encode()).hexdigest(),fit_helper_exact_inverse=True,helper_changes='Fixed method FLGMM to GuardFed-AD2+; native and shared JSON group keys restored; native objective checked against original config.',future_pins_null=True,F_access=0,SSH=0,Torch_imports=0,NumPy_imports=0,array_reads=0,fit_calls=0,CNN=0,statistics=0,root_adopted=False)
with (H/'SOURCE_CHECK.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(status=report['status'],source_check_sha256=sha(H/'SOURCE_CHECK.json'))))
