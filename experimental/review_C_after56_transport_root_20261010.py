"""Verify the transport rebind; retain the timing of the actual deployment."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
R=Path(__file__).resolve().parents[1]
T=R/'tmp/celeba_mechanism_C_after56_root_operations_20261010'
O=R/'tmp/celeba_mechanism_C_after50_root_operations_20261010'
B=R/'tmp/celeba_mechanism_valid_C_after56_20261010'
H=R/'tmp/celeba_mechanism_C_after56_source_review_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(T/'FILES_SHA256.json')=='5fe4d6ba6f3587d386ed2a2ee184b06b8211102f19c5388b375df326ef7fd42c'
for name,row in read(T/'FILES_SHA256.json')['files'].items():
 assert sha(T/name)==row['sha256'] and (T/name).stat().st_size==row['bytes']
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def function(p,n):
 s=p.read_text('utf8');return ast.get_source_segment(s,next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==n))
A=load('root_after56_adopter',T/'adopt.py');K=load('root_after56_backup',T/'backup.py');D=load('root_after56_deploy',T/'deploy.py')
assert function(T/'adopt.py','expected_archive_names')==function(O/'adopt.py','expected_archive_names')
ids=read(B/'SCOPE.json')['selected_ids']
assert len(A.expected_archive_names(ids,read(B/'FILES_SHA256.json'),read(B/'execution_candidate/EXECUTION_SOURCE_SHA256.json')))==61
good=dict(service='guardfed_celeba_mechanism_valid_C_after56 EXITED',processes=[],batch_failure=None,batch_complete={'completed':ids},completed=[{'id':i} for i in ids])
checks=0
for m in (A,K):
 m.check_terminal(good,ids);checks+=1
 for k,v in [('service','RUNNING'),('processes',[1]),('batch_failure',{'error':'fixture'}),('batch_complete',None),('completed',good['completed'][:3]),('completed',[{'id':ids[0]}]*4),('completed',good['completed'][:3]+[{'id':'foreign'}])]:
  x=copy.deepcopy(good);x[k]=v
  try:m.check_terminal(x,ids)
  except AssertionError:checks+=1
  else:raise AssertionError(k)
review=H/'ROOT_INDEPENDENT_REVIEW.json'
assert sha(review)=='45ab6401154d3c08aeebb07096f27d62d6f07f2729fc4c2a9736e30c97dc2258'
D.validate_source_review(review,sha(review))
assert checks==16
proof=dict(status='ROOT_EXACT4_TRANSPORT_DIFF_SOURCE_REVIEW_AND_LAYOUT_PASS',transport_seal_sha256=sha(T/'FILES_SHA256.json'),source_review_sha256=sha(review),terminal_fixture_checks=16,expected_archive_members=61,original_archive_layout_function_exact=True,manual_diff_scope='Namespace, actual source pins, 156+4 boundary and prior accepted receipt only',SSH=False,execution_approval_created=False,new_CNN=0,local_seal_reader_failure_preserved=True,actual_deployment_receipt_already_exists=(B/'execution_candidate/deployment_receipt.json').exists(),review_timing='Supplemental transport fixtures ran after actual deployment; independent scientific source review and manual transport diff review preceded deployment.')
with (H/'ROOT_TRANSPORT_REVIEW.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
