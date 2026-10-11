from pathlib import Path
import ast,copy,hashlib,json,importlib.util,textwrap
R=Path.cwd();H=Path(__file__).resolve().parent;P=R/'tmp/old800_cuda3_saved_audit_prepare_20261011';F=R/'tmp/FL100_uniform_cuda_formal_adapter_prepare_20261011';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
for name in ['audit_saved.py','transport.py']:
 inv=json.loads((H/(name+'.inverse.json')).read_bytes());s=(H/name).read_text('utf8');restored=s
 for z in reversed(inv['spans']):assert restored.count(z['after'])==1;restored=restored.replace(z['after'],z['before'],1)
 assert restored.encode()==(P/name).read_bytes();compile(s,name,'exec');checks.append(name+' full raw inverse exact')
for p in H.glob('*.py'):compile(p.read_text('utf-8-sig'),str(p),'exec')
source=(H/'transport.py').read_text();tree=ast.parse(source);code=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets));compile(code,'Linux transport payload','exec');checks.append('all source and embedded Linux payload compile')
old=(P/'audit_saved.py').read_text();new=(H/'audit_saved.py').read_text();start='            with np.load(array_path,allow_pickle=False) as z:';end='        for rel,pin in members.items():'
# Whitespace variable spelling differs only outside science, so use exact local end marker.
end='        for rel,pin in members.items()' if '        for rel,pin in members.items()' in old else '        for rel,pin in members.items()'
def tail(s):
 a=s.index(start);b=s.index('        for rel,',a);return s[a:b]
assert tail(old)==tail(new);checks.append('saved NPZ/root-ID guard and seven-statement execution tail raw exact')
flsrc=(R/'tmp/fl_isolated_v3_saved_closure_prepare_20261011/saved_v2/audit_saved.py').read_text();fitsrc=(H/'saved_fit_binding.py').read_text()
def fn(s,name):return next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
assert fn(flsrc,'saved_fits_for_original_predict')==fn(fitsrc,'saved_fits_for_original_predict');checks.append('original FL saved-fit restoration whole function raw exact')
flscience=(R/'tmp/fl_isolated_v3_saved_closure_prepare_20261011/saved_v2/saved_science.py').read_text();block=next(n for n in ast.walk(ast.parse(flscience)) if isinstance(n,ast.With));original=ast.parse(textwrap.dedent((H/'original_full_saved_science.py.txt').read_text())).body[0]
assert ast.dump(block,include_attributes=False)==ast.dump(original,include_attributes=False);checks.append('original FL Linux saved block AST equals retained original nine statements')
# Execute only original AST selector (stdlib) to verify seven selected statements, no scientific statements run.
ns={'ast':ast,'copy':copy,'need':lambda ok,msg:None if ok else (_ for _ in ()).throw(ValueError(msg))};exec(compile(fn(flsrc,'saved_output_block'),'original AST selector','exec'),ns);saved=ns['saved_output_block'](original);assert len(saved.body)==7;checks.append('seven original statements retained; two refit statements excluded by original selector')
frozen=json.loads((H/'SELECTED3_INPUTS.json').read_bytes());formal=json.loads((F/'SOURCE_PINS.json').read_bytes());scope=json.loads((F/'EXACT100.json').read_bytes());actual=json.loads((R/'tmp/FL100_cuda_actual_root_20261011/OBSERVATION_20261011T080528Z.json').read_bytes());assert [r['id'] for r in actual['accepted']]==frozen['ids']
sp=importlib.util.spec_from_file_location('_FL_metadata_only',H/'metadata_gate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
for selected in frozen['rows']:
 row=selected['original_row'];route=selected['route'];pkg=selected['package'];assert next(r for r in scope['records'] if r['id']==row['id'])==route;assert pkg==formal['packages'][route['identity_source']]
 assert m.canonical(row['identity'])==route['identity_canonical_sha256'];projected=m.projected(row)
 # Bind adapter's exact record.update AST in a stdlib-only namespace and compare the projected metadata.
 src=(F/'adapter.py').read_text();node=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='record.update')
 env=dict(record=copy.deepcopy(row['identity']),identity=row['identity'],original_row=row,canonical=m.canonical);exec(compile(ast.Module(body=[node],type_ignores=[]),'original FL metadata projection','exec'),env);assert env['record']==projected
 assert hashlib.sha256(json.dumps(projected,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==m.original_receipt_canonical(projected)
 assert len(pkg)==11;accepted=next(v['accepted'] for v in actual['accepted'] if v['id']==row['id']);assert set(accepted)>={'receipt','arrays','binding','acceptance','approval'} and accepted['native_max_abs_difference']==0.0
 for role,name in [('receipt','receipt.json'),('arrays','validation_predictions.npz'),('binding','formal_binding.json'),('acceptance','formal_saved_acceptance.json')]:assert accepted[role]['path']==frozen['planned_namespace']+'/runs/'+row['id']+'/'+name
checks.append('three original package roles/identity projections equal actual adapter; actual snapshot pin paths match')
ff=frozen['rows'][1]['package'];assert ff['candidate']['sha256']=='3f1c2e91c9ee492bd617900b6d79fb75df044c719ed3bc52cc33308dd0e37cab' and ff['package_seal']['sha256']=='26cf63fb47d9b59965e6231912cd072de06e6fa1b3bfad13f17866f5bd60c675';checks.append('F-Flip frozen original 3f1c/26cf preserved')
t=json.loads((H/'ROOT_INPUTS.template.json').read_bytes());assert t['base'] is None and t['future_root_adoption']=={'path':None,'sha256':None} and all(v is None for v in t['accepted_sha256'].values());checks.append('all future execution/adoption pins remain null')
report=dict(status='PASS_SOURCE_ONLY_NO_SCIENTIFIC_EXECUTION',checks=checks,scientific_statements_executed=0,scientific_imports=0,arrays_read=0,SSH=0,F_access=0,metadata_projection_AST_executed=True,actual_snapshot={'path':'tmp/FL100_cuda_actual_root_20261011/OBSERVATION_20261011T080528Z.json','sha256':sha(R/'tmp/FL100_cuda_actual_root_20261011/OBSERVATION_20261011T080528Z.json')},root_adopted=False,science_tail_sha256=hashlib.sha256(tail(new).encode()).hexdigest())
(H/'SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
