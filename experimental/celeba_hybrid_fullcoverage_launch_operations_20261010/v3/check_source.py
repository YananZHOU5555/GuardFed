"""No-SSH/no-Torch checks of the exact Hybrid96 lifecycle interface."""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import ast, copy, hashlib, json, sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'tmp/celeba_hybrid_fullcoverage_canary_operations_20261010/v3'
IMPL=ROOT/'tmp/celeba_hybrid_fullcoverage_implementation_v3_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def expect(name,fn,success):
 try:fn();actual=True
 except (AssertionError,ValueError,KeyError,BlockingIOError):actual=False
 assert actual==success,name
 checks.append(dict(name=name,expected=success,actual=actual))
trees={n:ast.parse((HERE/n).read_text(encoding='utf8')) for n in ('launch.py','remote_launch.py','main_health.py')}
assert (HERE/'main_health.py').read_bytes()==(OLD/'main_health.py').read_bytes()
functions=[n for n in trees['remote_launch.py'].body if isinstance(n,ast.FunctionDef) and n.name in ('check_closure','unlocked_canary_file')]
ns={'SERVICE':'guardfed_celeba_hybrid_fullcoverage'}
exec(compile(ast.Module(body=functions,type_ignores=[]),'<original-helper-functions>','exec'),ns)
approval=dict(status='ROOT_AUTHORIZED_HYBRID96_VALID_ONLY',scope='96_new_70round_valid_only',service=ns['SERVICE'],cpus=[104],cpu_threads=1,planned_new=96,reused=4,planned_total=100,final_test=False,
 package_sha256='a'*64,implementation_source_seal_sha256='b'*64,gate_acceptance_sha256='c'*64,canary_authorization_sha256='d'*64)
gate=dict(status='SEVEN_HYBRID_CANARIES_STRICT_PASS_BACKUP_PENDING',package_sha256='a'*64,accepted_ids=['MOCK'+str(n) for n in range(7)],pairs=[{},{}],scientific70records=0,test=False)
closure=dict(status='ROOT_SEVEN_HYBRID_CANARIES_OFFSERVER_ADOPTED',package_sha256='a'*64,gate_sha256='c'*64)
off=dict(status='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON',local=dict(package_sha256='a'*64,gate_sha256='c'*64,accepted_new=7,same_horizon_pairs=2,total_runs=7,rounds=3,formal_table_samples=0,CNN_calls=0,local_CUDA_initialized=False))
expect('exact original7 closure contract',lambda:ns['check_closure'](approval,gate,off,closure),True)
for name,which,key,value in [('six IDs partial','gate','accepted_ids',gate['accepted_ids'][:6]),('wrong gate status','gate','status','PASS'),('wrong root status','closure','status','ROOT_SEVEN_CANARY_CLOSURE_ADOPTED'),
                            ('wrong package','closure','package_sha256','e'*64),('wrong gate SHA','closure','gate_sha256','e'*64),('wrong CPU105','approval','cpus',[105])]:
 values={k:copy.deepcopy(v) for k,v in dict(approval=approval,gate=gate,off=off,closure=closure).items()};values[which][key]=value
 expect(name,lambda values=values:ns['check_closure'](values['approval'],values['gate'],values['off'],values['closure']),False)

# Evaluate the actual new authorization dictionary, then call the unchanged common.authorized.
main=next(n for n in trees['remote_launch.py'].body if isinstance(n,ast.FunctionDef) and n.name=='main')
auth_value=next(n.value for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='authorization' for t in n.targets))
fake_here=PurePosixPath('/MOCK/fullcoverage_operations');stage=PurePosixPath('/MOCK/stage')
resource=dict(observed_unix=100,utc='MOCK',original_screen_exited=True,no_duplicate_workers=True,no_restricted_CPU_overlap=True,protected_main_healthy=True,protected_main_max_workers=8,
 planned_total_cpu_threads=120,cpu_quota_cores=122.88,free_memory_bytes=8*1024**3,gpu_free_memory_mib=4096,gpu_recovery_action='None')
hashes={str(fake_here/'ROOT_CLOSURE.json'):'e'*64,str(fake_here/'RESOURCE.json'):'f'*64,str(fake_here/'APPROVAL.json'):'1'*64,str(stage/'PACKAGE_SHA256.json'):'a'*64,
 str(stage/'GATE_ACCEPTANCE.json'):'c'*64,str(stage/'gate_runs/MOCK'):'2'*64}
authorization=eval(compile(ast.Expression(auth_value),'<actual-authorization-dict>','eval'),dict(approval=approval,closure=closure,resource_receipt=resource,HERE=fake_here,sha=lambda p:hashes[str(p)]))
common=ast.parse((IMPL/'common.py').read_text(encoding='utf8'));authorized=next(n for n in common.body if isinstance(n,ast.FunctionDef) and n.name=='authorized')
def require(ok,message):
 if not ok:raise ValueError(message)
class CgroupPath:
 def __init__(self,path):self.path=path
 def read_text(self):
  assert self.path=='/sys/fs/cgroup/cpu.max'
  return '12288000 100000'
gate_for_common=dict(gate,artifact_hashes={'gate_runs/MOCK':'2'*64})
def run_authorized(auth):
 documents={str(stage/'EXECUTION_AUTHORIZATION.json'):auth,str(fake_here/'RESOURCE.json'):resource,str(stage/'GATE_ACCEPTANCE.json'):gate_for_common,str(fake_here/'ROOT_CLOSURE.json'):closure}
 env=dict(HERE=stage,sys=SimpleNamespace(platform='linux'),os=SimpleNamespace(sched_getaffinity=lambda pid:{104},getpriority=lambda kind,pid:10,PRIO_PROCESS=0),Path=CgroupPath,
          local_identity=lambda:None,require=require,read=lambda p:documents[str(p)],digest=lambda p:hashes[str(p)])
 exec(compile(ast.Module(body=[authorized],type_ignores=[]),'<unchanged-common-authorized>','exec'),env)
 return env['authorized']('96_new_70round_valid_only',fresh=False)
expect('actual authorization accepted by unchanged common.authorized',lambda:run_authorized(authorization),True)
for name,key,value in [('max_workers2','max_workers',2),('automatic retry','automatic_retry',True),('wrong allowed CPU','allowed_cpus',[105]),('missing gate root path','gate_root_closure_path',None)]:
 a=copy.deepcopy(authorization)
 if value is None:del a[key]
 else:a[key]=value
 expect('common refuses '+name,lambda a=a:run_authorized(a),False)

# Execute only the exact lock function with in-memory fake file/flock; no real lock or process.
events=[]
class File:
 def __enter__(self):return self
 def __exit__(self,*args):pass
class LockPath:
 def is_file(self):return True
 def is_symlink(self):return False
 def open(self,mode):assert mode=='r+';events.append('open_existing');return File()
fcntl=SimpleNamespace(LOCK_EX=2,LOCK_NB=4,LOCK_UN=8,flock=lambda f,flag:events.append(flag))
prior_fcntl=sys.modules.get('fcntl');sys.modules['fcntl']=fcntl
try:
 ns['unlocked_canary_file'](LockPath());assert events==['open_existing',6,8]
 def blocked(f,flag):raise BlockingIOError('MOCK current lock holder')
 fcntl.flock=blocked
 expect('held canary lock refuses without retry',lambda:ns['unlocked_canary_file'](LockPath()),False)
finally:
 if prior_fcntl is None:del sys.modules['fcntl']
 else:sys.modules['fcntl']=prior_fcntl
text=(HERE/'remote_launch.py').read_text(encoding='utf8')
assert text.index("open('xb') as f:f.write(previous)")<text.index('os.replace(temporary,STAGE/')<text.index("command(['supervisorctl','start',SERVICE])")
assert "'coordinator.lock')" not in next(ast.get_source_segment(text,n) for n in main.body if isinstance(n,ast.For) and 'Prior fullcoverage output preserved' in ast.get_source_segment(text,n))
for token in ('run_fullcoverage.py run --repo','taskset -c 104','autorestart=false','autostart=false','startretries=0',"resource_guard_inputs=getattr(e,'resource_guard_inputs',None)"):
 assert token in text,token
assert "ssh+['python -B -']" in (HERE/'launch.py').read_text(encoding='utf8')
result=dict(status='SOURCE_ONLY_NO_SSH_NO_TORCH_CHECKS_PASS',checks=checks,check_n=len(checks),unchanged_common_authorized_sha256=sha(IMPL/'common.py'),original_main_health_sha256=sha(OLD/'main_health.py'),
 new_authorization_fields=sorted(authorization),main_health_bytes_exact=True,existing_unlocked_canary_lock_file_accepted=True,old_authorization_preserved_before_atomic_replace=True,
 source_parsed=3,actual_proof_created=False,SSH=False,process_started=False,Torch=False,CNN=False)
with (HERE/'SOURCE_CHECK.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],checks=len(checks))))
