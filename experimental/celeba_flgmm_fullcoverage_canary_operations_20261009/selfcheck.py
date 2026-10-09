"""Metadata/guard checks only. No SSH, scientific imports, supervisor or CNN calls."""
from pathlib import Path
import ast,copy,hashlib,json,subprocess,sys,tempfile
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x):
 if not x:raise RuntimeError('Selfcheck failed')
def main():
 results=[]
 for name in ('launch.py','remote_launch.py','main_health.py'):
  ast.parse((HERE/name).read_text());results.append('parse:'+name)
 lineage=json.loads((HERE/'HEALTH_LINEAGE.json').read_bytes())
 original=Path(lineage['source'])
 require(sha(original)==lineage['source_sha256'])
 old=ast.parse(original.read_text());new=ast.parse((HERE/'main_health.py').read_text())
 for name in ('need','main_health'):
  get=lambda tree:next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
  require(ast.dump(get(old))==ast.dump(get(new)));results.append('original_health_AST:'+name)
 ns={'__name__':'metadata_check'};exec(compile(new,'main_health.py','exec'),ns)
 for n in (1,8):
  ns['main_health']({'returncode':0,'stdout':'main RUNNING pid 123','stderr':''},{'failed':[],'active':[{}]*n},'fixture');results.append('health_positive:'+str(n))
 for queue in ({'failed':['bad'],'active':[{}]},{'failed':[],'active':[]},{'failed':[],'active':[{}]*9}):
  try:ns['main_health']({'returncode':0,'stdout':'main RUNNING pid 123','stderr':''},queue,'fixture')
  except ValueError:results.append('health_refusal')
  else:raise RuntimeError('Health mutation accepted')
 # Run the real local approval prefix, stopping before transfer/source payload or network.
 tree=ast.parse((HERE/'launch.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
 cut=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source' for t in x.targets))
 fn.body=fn.body[:cut]+[ast.Raise(exc=ast.Call(func=ast.Name(id='ReachedTransportBoundary',ctx=ast.Load()),args=[],keywords=[]),cause=None)]
 prefix=ast.Module(body=[x for x in tree.body if not isinstance(x,ast.If)]+[],type_ignores=[])
 class ReachedTransportBoundary(Exception):pass
 ns={'__name__':'guard_fixture','__file__':str(HERE/'launch.py'),'ReachedTransportBoundary':ReachedTransportBoundary}
 exec(compile(ast.fix_missing_locations(prefix),'approval_prefix','exec'),ns)
 with tempfile.TemporaryDirectory(prefix='local_fixture_',dir=HERE) as td:
  root=Path(td);ns['HERE']=root
  def write(name,v):p=root/name;p.write_text(json.dumps(v));return p
  write('FILES_SHA256.json',{'files':{}})
  proof=write('proof.json',{'package_sha256':'bound-package','status':'ROOT_ACTUAL_BOUND96_PLUS4_METADATA_MEMBER_AND_SCOPE_PASS','archive_members_verified':144})
  baseline=write('baseline.json',{'fixture':True})
  approval={'status':'ROOT_AUTHORIZED_SEVEN_FLGMM_CANARIES','scope':'seven_same_horizon_3round_canaries','package_sha256':'bound-package','helper_seal_sha256':sha(root/'FILES_SHA256.json'),'bound_offserver_sha256':sha(proof),'bound_root_review_sha256':sha(proof),'baseline_snapshot_sha256':sha(baseline),'service':'guardfed_celeba_flgmm_fullcoverage_canary','cpus':[102,103],'cpu_threads':1,'formal100_started':False,'final_test':False}
  mutations={'status':'PREPARED_NOT_APPROVED','scope':'96_new_70round_valid_only','package_sha256':'wrong','helper_seal_sha256':'wrong','bound_offserver_sha256':'wrong','bound_root_review_sha256':'wrong','baseline_snapshot_sha256':'wrong','service':'guardfed_celeba_mechanism_formal','cpus':[104,105],'cpu_threads':8,'formal100_started':True,'final_test':True}
  oldargv=sys.argv
  try:
   for key,value in [(None,None)]+list(mutations.items()):
    a=copy.deepcopy(approval)
    if key:a[key]=value
    ap=write('approval.json',a)
    sys.argv=['fixture','--approval',str(ap),'--approval-sha256',sha(ap),'--bound-offserver',str(proof),'--bound-offserver-sha256',sha(proof),'--bound-root-review',str(proof),'--bound-root-review-sha256',sha(proof),'--baseline',str(baseline),'--baseline-sha256',sha(baseline),'--package-sha256','bound-package','--helper-seal-sha256',sha(root/'FILES_SHA256.json')]
    try:ns['main']()
    except ReachedTransportBoundary:require(key is None);results.append('approval_positive_to_transport_boundary')
    except AssertionError:require(key is not None);results.append('approval_refusal:'+key)
    else:raise RuntimeError('Unexpected approval exit')
  finally:sys.argv=oldargv
 for name in ('launch.py','remote_launch.py'):
  p=subprocess.run([sys.executable,'-B','-O',str(HERE/name),'--help'],capture_output=True,text=True)
  require(p.returncode!=0 and 'Optimized Python forbidden' in p.stderr);results.append('optimized_python_refused:'+name)
 output={'status':'PASS_LOCAL_METADATA_ONLY','checks':results,'check_count':len(results),'SSH_calls':0,'CNN_calls':0,'services_modified':0,'limits':'No Linux /proc, real resource or supervisor execution; root review and fresh server checks remain required.'}
 with (HERE/'SELF_CHECK.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2);f.write('\n')
 print(json.dumps(output))
if __name__=='__main__':main()
