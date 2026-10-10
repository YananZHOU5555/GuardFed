"""Bounded no-CNN scope/binding checks. Synthetic metadata is never an adoption."""
from pathlib import Path
import ast,copy,hashlib,json,traceback
import build as b
H=Path(__file__).resolve().parent;R=H.parents[1]
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
try:
 inputs=b.verify_inputs();template=b.read(H/'C10_BINDING_TEMPLATE.json')
 prior=b.read(b.C60/'inventory_actual160_Full100refs.json');current=b.read(b.C70/'inventory_actual170_Full100refs.json');controls=b.scope(prior,current);assert len(controls)==70
 def refuse(label,fn):
  try:fn()
  except (ValueError,AssertionError,KeyError,TypeError):return label
  raise AssertionError('Did not refuse '+label)
 refusals=[refuse('null actual adoption cannot build',lambda:b.verify_future_binding(template))]
 # Explicit in-memory schema fixture; no receipt/closure file or scientific output.
 fixture=dict(status='ROOT_C_AFTER60_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',prior_three_view_models=160,accepted_new=10,cumulative_three_view_models=170,accepted_new_ids=b.EXPECTED,original160_unchanged=True,source_scope_complete=True,negative_results_preserved=True,science_seal_sha256=template['science_sha256'],execution_seal_sha256=template['execution_sha256'],prior160_root_adoption_sha256=inputs['prior160_adoption_sha256'],all_native_differences_zero=True,server_strict_bound_in_saved_receipts=True,new_training=0,new_Full_inference=0,test_inference=False)
 call=lambda p:b.adoption_metadata(p,template['science_sha256'],template['execution_sha256'],inputs['prior160_adoption_sha256'])
 call(fixture)
 for key,value in [('status','RUNNING'),('accepted_new',9),('prior_three_view_models',150),('accepted_new_ids',[b.EXPECTED[0]]*10),('science_seal_sha256','0'*64),('prior160_root_adoption_sha256','0'*64),('all_native_differences_zero',False),('negative_results_preserved',False),('new_Full_inference',1),('test_inference',True)]:
  bad=copy.deepcopy(fixture);bad[key]=value;refusals.append(refuse('adoption '+key,lambda bad=bad:call(bad)))
 for label,mutate in [('missing new seed',lambda d:d['records'].pop()),('old record drift',lambda d:d['records'][0].update(seed=-1)),('tolerance drift',lambda d:d.update(native_tolerance=1e-10)),('new split drift',lambda d:next(r for r in d['records'] if r['id']==b.EXPECTED[0]).update(original_split='test'))]:
  bad=copy.deepcopy(current);mutate(bad);refusals.append(refuse(label,lambda bad=bad:b.scope(prior,bad)))
 for name in ('build.py','panels.py','verify_numeric.py','prepare_source.py','check_prepared.py'):ast.parse((H/name).read_text(encoding='utf8'))
 reuse=b.read(H/'SOURCE_REUSE.json');assert reuse['numeric_guard_normalized_body_exact'] and reuse['verify_aggregate_function_byte_exact']
 def function(p,n):
  text=p.read_text(encoding='utf8');node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name==n);return ast.get_source_segment(text,node)
 old=R/'tmp/celeba_mechanism_C_six_scenes_prepared_20261010/verify_numeric.py';actual=function(H/'verify_numeric.py','verify')
 for before,after in reversed(reuse['numeric_guard_changes']):actual=actual.replace(after,before)
 assert actual==function(old,'verify') and function(H/'verify_numeric.py','verify_aggregate')==function(old,'verify_aggregate')
 records=b.read(b.OLD/'snapshot/records.json')['records'];assert len(records)==len({r['id'] for r in records})==120
 assert not (H/'snapshot').exists()
 save('CHECK_RESULTS.json',dict(status='PREPARED_SOURCE_AND_METADATA_GATES_PASS_NO_FUTURE_ADOPTION',actual_input_pins=len(inputs['files']),actual_native_inventory_scope=170,actual_old_native_inventory_scope=160,actual_C60_records_preserved=120,old160_records_exact=True,Full100_references_exact=True,scope_exact10=b.EXPECTED,synthetic_adoption_schema_positive=True,synthetic_is_not_actual_adoption=True,refusal_count=len(refusals),refusals=refusals,numeric_guard_normalized_body_exact=True,verify_aggregate_function_byte_exact=True,no_imbalanced7scene_aggregate=True,future_adoption_sha256=None,new_statistics=0,new_table_outputs=0,CNN=0,SSH=False,Git=False))
 print(json.dumps(dict(status='PREPARED_NO_OUTPUT',input_pins=len(inputs['files']),refusals=len(refusals),actual_new_three_view_acceptance=0)))
except BaseException as e:
 save('PREPARATION_CHECK_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),scientific_output_created=False,automatic_retry=False));raise
