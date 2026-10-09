"""No-network operation-helper checks. Does not call any operation main()."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/celeba_mechanism_valid_C_after12_20261009';E=P/'execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
source_before=read(P/'PACKAGE_SHA256.json')['members']
for n,pin in source_before.items():assert sha(P/n)==pin['sha256']
def load(name):
 spec=importlib.util.spec_from_file_location('root_operation_check_'+name,H/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
deploy,backup,adopt=load('deploy'),load('backup'),load('adopt')
review_path=R/'tmp/celeba_mechanism_C_after12_root_independent_20261009/ROOT_INDEPENDENT_REVIEW.json'
review_sha='3b2c936c91a78e6caa39fb007c8d485c6e7870ebfab3bae49831adc1c60c64db'
assert deploy.validate_source_review(review_path,review_sha)==read(review_path)
assert deploy.ROOT==R and backup.ROOT==R and adopt.ROOT==R
ids=read(P/'SCOPE.json')['selected_ids'];assert ids==[f'minus_C_IID_F Flip_seed{s}' for s in range(91003,91011)]
refusals=[]
def refused(name,call):
 try:call()
 except (AssertionError,ValueError,KeyError):refusals.append(name);return
 raise AssertionError('Not refused: '+name)
refused('external_review_SHA',lambda:deploy.validate_source_review(review_path,'0'*64))
for key,value in [('status','PREPARED'),('science_seal_sha256','0'*64),('execution_seal_sha256','0'*64),('package_sha256','0'*64),('actual_dispatch_authorized_by_this_review',True),('exact_selected_ids',ids[:-1]),('new_three_view_accepted',8)]:
 bad=copy.deepcopy(read(review_path));bad[key]=value
 with patch.object(deploy,'read',lambda _:bad):refused('review_'+key,lambda:deploy.validate_source_review(review_path,review_sha))
live={'service':'guardfed_celeba_mechanism_valid_C_after12 EXITED','processes':[],'batch_failure':None,'batch_complete':{'synthetic_fixture_only':True},'completed':[{'id':i} for i in ids]}
for m in (backup,adopt):
 m.check_terminal(live,ids)
 for name,value in [('missing',live['completed'][:-1]),('duplicate',live['completed'][:-1]+[live['completed'][0]]),('extra',live['completed']+[{'id':'foreign'}])]:
  bad=copy.deepcopy(live);bad['completed']=value;refused(m.__name__+'_'+name,lambda bad=bad:m.check_terminal(bad,ids))
 for key,value in [('service','RUNNING'),('processes',[{'pid':1}]),('batch_failure',{'error':'synthetic'}),('batch_complete',None)]:
  bad=copy.deepcopy(live);bad[key]=value;refused(m.__name__+'_'+key,lambda bad=bad:m.check_terminal(bad,ids))
names=adopt.expected_archive_names(ids,read(P/'FILES_SHA256.json'),read(E/'EXECUTION_SOURCE_SHA256.json'))
assert len(names)==89 and not any(n.endswith('model.pt') for n in names)
entry_checks=[]
for name in ['deploy','observe','backup','adopt']:
 for optimize in (False,True):
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None)
  proc=subprocess.run([sys.executable,'-B',*(['-O'] if optimize else []),str(H/(name+'.py')),'--help'],capture_output=True,text=True,env=env,timeout=15)
  if optimize:assert proc.returncode!=0 and 'Optimized Python is forbidden' in proc.stderr
  else:assert proc.returncode==0 and 'usage:' in proc.stdout
  entry_checks.append({'entry':name,'optimized':optimize,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
# Materialize actual embedded SSH-code expressions using explicit inert future
# values. Compile the strings only, never execute them or connect to the server.
embedded=[]
for name,variables in [('deploy',dict(GUIDE=deploy.GUIDE,REMOTE=deploy.REMOTE,remote_archive='/INERT_ONLY.tar.gz',archive_sha='0'*64,BASE=P,EX=E,sha=lambda _:'0'*64)),('observe',dict(REMOTE='/workspace/guardfed_checks/'+P.name+'/execution_candidate',deployment={'execution_seal_sha256':'0'*64,'root_approval_sha256':'1'*64})),('backup',dict(REMOTE=backup.REMOTE,EXECUTION=backup.EXECUTION,SCIENCE=backup.SCIENCE,deployment={'root_approval_sha256':'0'*64},live=live,expected=ids))]:
 tree=ast.parse((H/(name+'.py')).read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('preflight','code') for t in node.targets) and isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Mod):
   expr=ast.Expression(node.value);code=eval(compile(expr,'INERT_EMBEDDED_TEMPLATE','eval'),variables)
   ast.parse(code);embedded.append({'helper':name,'target':node.targets[0].id,'stdin_code_bytes':len(code.encode()),'argv_remote_command':'python -B -','executed':False})
assert len(embedded)==4
local_verifier=(E/'verify_backup.py').read_text()
assert 'extractall(' not in local_verifier and 'filter=' not in local_verifier
assert "filter='data'" in (H/'deploy.py').read_text() # Remote3.12 code only.
for name in ['backup','observe','deploy']:
 text=(H/(name+'.py')).read_text();assert "'python -B -'" in text and 'input=' in text
for n,pin in source_before.items():assert sha(P/n)==pin['sha256']
assert not any((E/n).exists() for n in ['ROOT_APPROVED.json','EXECUTION_DRAFT.json','deployment_receipt.json','ROOT_DEPLOYMENT_FAILURE.json'])
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
report={'status':'PREPARED_HELPERS_LOCAL_NO_NETWORK_CHECKS_PASS_NOT_OPERATION_APPROVAL','actual_source_review_sha256':review_sha,'exact8_ids':ids,'positive_external_review_schema':True,'terminal_positive_modules':2,'refusals':refusals,'refusal_count':len(refusals),'expected_archive_members':89,'expected_metrics_counts_rules':[72,192,24],'entry_checks':entry_checks,'embedded_code_compile_only':embedded,'short_ssh_argv_stdin':True,'local_manual_safe_extract_reused':True,'original_source_package39_members_unchanged':True,'future_runtime_SHA_not_invented':True,'root_authority_files_created':False,'network_calls':0,'CNN':False,'SSH':False,'scientific_acceptance_registered':False}
with (H/'CHECKS.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['entry_checks','refusals']}))
