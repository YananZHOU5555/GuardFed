"""Execute real approval prefix through metadata transport boundary only; no network/CNN."""
from pathlib import Path
import ast,copy,hashlib,json,subprocess,sys,tempfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Boundary(Exception):pass
def main():
 tree=ast.parse((HERE/'launch.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source' for t in n.targets))
 fn.body=fn.body[:cut]+[ast.Raise(exc=ast.Call(func=ast.Name(id='Boundary',ctx=ast.Load()),args=[],keywords=[]),cause=None)]
 tree.body=[n for n in tree.body if not isinstance(n,ast.If)]
 ns={'__name__':'fixture','__file__':str(HERE/'launch.py'),'Boundary':Boundary};exec(compile(ast.fix_missing_locations(tree),'real_approval_prefix','exec'),ns)
 checks=[]
 with tempfile.TemporaryDirectory(dir=HERE,prefix='fixture_') as td:
  root=Path(td);ns['HERE']=root
  def write(n,o):p=root/n;p.write_text(json.dumps(o));return p
  seal=write('FILES_SHA256.json',{'files':{}});baseline=write('baseline.json',{'fixture':True})
  off0={'status':'PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON','local':{'package_sha256':'package','gate_sha256':'gate','accepted_new':5,'same_horizon_pairs':2,'rounds':3,'formal_table_samples':0},'receipt_sha256':'receipt','archive_sha256':'archive'}
  close0={'status':'ROOT_SEVEN_CANARY_CLOSURE_ADOPTED','package_sha256':'package','gate_sha256':'gate','receipt_sha256':'receipt','archive_sha256':'archive'}
  cases=[('positive',None,None),('approval_scope','scope','seven_same_horizon_3round_canaries'),('approval_count','planned_new',95),('approval_reuse','reused',3),('approval_total','planned_total',99),('approval_test','final_test',True),('wrong_service','service','guardfed_celeba_mechanism_formal'),('wrong_cpus','cpus',[104,105]),('missing_root_acceptance',None,None),('offserver_not_pass',None,None),('wrong_gate',None,None),('wrong_pair_count',None,None),('wrong_horizon',None,None),('missing_offserver',None,None),('wrong_archive',None,None)]
  oldargv=sys.argv
  try:
   for label,key,value in cases:
    off=copy.deepcopy(off0);close=copy.deepcopy(close0)
    if label=='missing_root_acceptance':close['status']='PREPARED_NOT_APPROVED'
    if label=='offserver_not_pass':off['status']='PENDING'
    if label=='wrong_gate':close['gate_sha256']='wrong'
    if label=='wrong_pair_count':off['local']['same_horizon_pairs']=1
    if label=='wrong_horizon':off['local']['rounds']=70
    if label=='wrong_archive':close['archive_sha256']='wrong'
    op=write('off.json',off);close['offserver_sha256']=sha(op);cp=write('closure.json',close)
    a={'status':'ROOT_AUTHORIZED_FLGMM96_VALID_ONLY','scope':'96_new_70round_valid_only','package_sha256':'package','helper_seal_sha256':sha(seal),'gate_offserver_sha256':sha(op),'root_closure_sha256':sha(cp),'baseline_snapshot_sha256':sha(baseline),'service':'guardfed_celeba_flgmm_fullcoverage','cpus':[102,103],'cpu_threads':1,'planned_new':96,'reused':4,'planned_total':100,'final_test':False,'gate_acceptance_sha256':'gate','canary_authorization_sha256':'prior'}
    if key:a[key]=value
    ap=write('approval.json',a)
    if label=='missing_offserver':op.unlink()
    sys.argv=['fixture','--approval',str(ap),'--approval-sha256',sha(ap),'--gate-offserver',str(op),'--gate-offserver-sha256',a['gate_offserver_sha256'],'--root-closure',str(cp),'--root-closure-sha256',sha(cp),'--baseline',str(baseline),'--baseline-sha256',sha(baseline),'--package-sha256','package','--helper-seal-sha256',sha(seal)]
    try:ns['main']()
    except Boundary:assert label=='positive';checks.append('PASS:'+label)
    except (AssertionError,FileNotFoundError):assert label!='positive';checks.append('REFUSED:'+label)
    else:raise RuntimeError('Did not reach a valid boundary')
  finally:sys.argv=oldargv
 old=HERE.parent/'celeba_flgmm_fullcoverage_canary_operations_20261009'
 assert (HERE/'main_health.py').read_bytes()==(old/'main_health.py').read_bytes();checks.append('main_health_byte_exact')
 for name in ('remote_launch.py','launch.py'):
  ast.parse((HERE/name).read_text())
  p=subprocess.run([sys.executable,'-B','-O',str(HERE/name),'--help'],capture_output=True,text=True)
  assert p.returncode!=0 and 'Optimized Python forbidden' in p.stderr;checks.append('optimization_refused:'+name)
 result={'status':'PASS_LOCAL_METADATA_ONLY','checks':checks,'SSH_calls':0,'CNN_calls':0,'actual_gate_adoption_present':False,'limitation':'Fixture verifies approval boundary only, not actual Linux resource or training behavior.'}
 with (HERE/'SELF_CHECK.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
