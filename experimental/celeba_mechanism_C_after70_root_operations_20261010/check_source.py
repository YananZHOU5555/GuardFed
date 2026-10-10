"""Bounded local API/source checks only; no remote calls, authority or adoption."""
from pathlib import Path
import ast,hashlib,json,re,sys
from unittest.mock import patch
sys.dont_write_bytecode=True
import loader
H=Path(__file__).resolve().parent
m=loader.bindings();checks={};calls=[]
def forbidden(*a,**k):calls.append((a,k));raise AssertionError('External subprocess forbidden in source check')
import subprocess
with patch.object(subprocess,'run',forbidden):
 for op in ('deploy','observe','backup','adopt'):
  before=(loader.OLD/(op+'.py')).read_text(encoding='utf-8-sig');after=loader.patched_source(op)
  inverse={v:k for k,v in m['replacements'].items()};assert len(inverse)==len(m['replacements'])
  restored=re.sub(r'\b(170|180)\b',lambda z:{'170':'160','180':'170'}[z[0]],after)
  restored=re.sub('|'.join(re.escape(k) for k in sorted(inverse,key=len,reverse=True)),lambda z:inverse[z[0]],restored)
  assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(before),include_attributes=False)
  assert "'python -B -'" in after if op in ('deploy','observe','backup') else True
  assert 'CPU112' not in after or '112' in after
  assert "filter='data'" not in after if op=='adopt' else True
  namespace={'__name__':'source_check_'+op,'__file__':str(H/(op+'.py'))}
  if op!='observe':exec(compile(after,str(H/(op+'.py')),'exec'),namespace)
  checks[op]={'old_sha256':loader.sha(loader.OLD/(op+'.py')),'effective_source_sha256':hashlib.sha256(after.encode()).hexdigest(),'AST_equal_after_metadata_inverse':True,'import_without_main_pass':op!='observe','observe_AST_only_no_top_level_execution':op=='observe'}
  if op=='deploy':deploy=namespace
  if op=='adopt':
   science=loader.read(loader.ROOT/m['prepared']/'FILES_SHA256.json');execution=loader.read(loader.ROOT/m['prepared']/'execution_candidate/EXECUTION_SOURCE_SHA256.json')
   ids=loader.read(loader.ROOT/m['prepared']/'SCOPE.json')['selected_ids']
   assert len(namespace['expected_archive_names'](ids,science,execution))==103
assert not calls
# Synthetic source-review fixture is process memory only; no review/approval file is created.
ids=loader.read(loader.ROOT/m['prepared']/'SCOPE.json')['selected_ids']
fixture=dict(status='PASS_SOURCE_READY_FOR_ROOT_LINUX_PREFLIGHT_AND_EXACT10_APPROVAL',source_adoptable=True,actual_dispatch_authorized_by_this_review=False,science_seal_sha256=deploy['SCIENCE'],execution_seal_sha256=deploy['EXECUTION'],package_sha256=m['pins'][m['prepared']+'/PACKAGE_SHA256.json'],native_accepted_snapshot=180,excluded_prior_three_view_ids=170,old170_records_exact=True,Full100_references_exact=True,Full100_actual900_source_records_exact=True,new_three_view_accepted=0,positive_approval_exact10=True,exact_selected_ids=ids,actual_worker_pre_science_bind_ids=ids)
deploy['sha']=lambda p:'f'*64;deploy['read']=lambda p:fixture
assert deploy['validate_source_review'](Path('MEMORY_ONLY_NO_FILE'),'f'*64)==fixture
refusals=[]
for key,bad in [('native_accepted_snapshot',179),('excluded_prior_three_view_ids',160),('old170_records_exact',False),('package_sha256','0'*64),('exact_selected_ids',ids[:-1])]:
 good=fixture[key];fixture[key]=bad
 try:deploy['validate_source_review'](Path('MEMORY_ONLY_NO_FILE'),'f'*64)
 except AssertionError:refusals.append(key)
 else:raise AssertionError('Invalid fixture accepted: '+key)
 fixture[key]=good
try:deploy['validate_source_review'](Path('MEMORY_ONLY_NO_FILE'),'0'*64)
except AssertionError:refusals.append('external_review_sha256')
else:raise AssertionError('External SHA drift not refused')
assert not calls and 'torch' not in sys.modules
report=dict(status='SOURCE_ONLY_METADATA_REUSE_API_PASS_NO_SSH_OR_AUTHORITY',operations=checks,pinned_inputs=len(m['pins']),native_snapshot=180,prior_views_excluded=170,exact_selected_ids=ids,archive_expected_members=103,source_review_memory_positive=True,source_review_memory_refusals=refusals,actual_source_review_or_future_closure_created=False,external_subprocess_calls=0,SSH=False,CNN=0,new_accepted=0)
with (H/'CHECK_RESULTS.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'operations':len(checks),'refusals':len(refusals),'archive_expected_members':103}))
