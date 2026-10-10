"""Local actual binding/scope checks; no table, CNN, refitting or authority."""
from pathlib import Path
import ast,copy,json,sys
sys.dont_write_bytecode=True
import build as b
import loader
H=Path(__file__).resolve().parent
inputs=b.verify_inputs();current=b.read(b.C80/'inventory_actual180_Full100refs.json');prior=b.read(b.C70/'inventory_actual170_Full100refs.json')
controls=b.scope(prior,current);assert len(controls)==80
template=b.read(H/'C10_BINDING_TEMPLATE.json')
proof=b.R/'tmp/celeba_mechanism_valid_C_after70_20261010/execution_candidate/backups/incremental_20261010T025719Z/ROOT_ADOPTION_REVIEW.json'
assert b.sha(proof)=='3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615'
binding=dict(template,adoption=proof.relative_to(b.R).as_posix(),adoption_sha256=b.sha(proof))
b.verify_future_binding(binding)
refusals=[]
def refuse(n,fn):
 try:fn()
 except (ValueError,AssertionError,KeyError,TypeError):refusals.append(n);return
 raise AssertionError('Did not refuse '+n)
refuse('null adoption cannot generate table',lambda:b.verify_future_binding(template))
for key,bad in [('accepted_new',9),('prior_three_view_models',160),('original170_unchanged',False),('science_seal_sha256','0'*64),('test_inference',True)]:
 value=copy.deepcopy(b.read(proof));value[key]=bad
 refuse(key,lambda value=value:b.adoption_metadata(value,binding['science_sha256'],binding['execution_sha256'],inputs['prior170_adoption_sha256']))
def fn(source,name):return ast.get_source_segment(source,next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name))
reb=b.read(H/'REBINDS.json')
for kind in ['build','panels','verify_numeric']:
 spec=reb[kind];before=(b.R/spec['source']).read_text('utf-8');after=before
 for a,z,n in spec['replacements']:assert after.count(a)==n;after=after.replace(a,z)
 ast.parse(after)
 if kind=='panels':assert fn(before,'aggregate_panels')==fn(after,'aggregate_panels')
 if kind=='verify_numeric':assert fn(before,'verify_aggregate')==fn(after,'verify_aggregate')
assert len(b.read(b.OLD/'snapshot/records.json')['records'])==140
assert not (H/'snapshot').exists() and 'torch' not in sys.modules
with (H/'CHECK_SOURCE_RESULTS.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(dict(status='ACTUAL_C10_BINDING_AND_C80_SCOPE_SOURCE_PASS_NO_TABLE',input_pins=len(inputs['files']),actual_adoption_sha256=b.sha(proof),prior170_excluded=True,scope_exact10=b.EXPECTED,Full100_unchanged=True,original_aggregate_functions_source_exact=True,refusals=refusals,new_statistics=0,table_output=False,CNN=0,SSH=False),f,indent=2);f.write('\n')
with (H/'C10_BINDING.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(binding,f,indent=2);f.write('\n')
print(json.dumps({'status':'ACTUAL_BINDING_AND_SCOPE_PASS','input_pins':len(inputs['files']),'refusals':len(refusals)}))
