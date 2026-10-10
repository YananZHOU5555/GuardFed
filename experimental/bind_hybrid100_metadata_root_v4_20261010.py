"""One actual source-reviewed metadata bind; no Torch, gates or service changes."""
from pathlib import Path
import ast,base64,datetime,hashlib,json,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
SOURCE=ROOT/'tmp/celeba_hybrid_fullcoverage_implementation_v3_20261010'
REMOTE='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')

def main():
    approval=read(HERE/'BIND_APPROVAL.json')
    assert approval['status']=='ROOT_APPROVED_HYBRID100_METADATA_BINDING' and approval['execute_authorized'] is False and approval['final_test'] is False
    assert sha(SOURCE/'FILES_SHA256.json')==approval['source_seal_sha256']
    inputs={'SUMMARY32.json':ROOT/'tmp/celeba_hybrid32_final_collection_20261010/SUMMARY32.json',
            'ROOT32.json':ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after27_20261010/ROOT32_SUMMARY_ADOPTION.json',
            'SOURCE_REVIEW.json':ROOT/approval['independent_source_review_path'],'BIND_APPROVAL.json':HERE/'BIND_APPROVAL.json'}
    assert sha(inputs['SUMMARY32.json'])==approval['summary_sha256'] and sha(inputs['ROOT32.json'])==approval['root32_sha256']
    assert sha(inputs['SOURCE_REVIEW.json'])==approval['independent_source_review_sha256']
    files={'source_prepared/'+n:SOURCE/n for n in read(SOURCE/'FILES_SHA256.json')['files']}
    files['source_prepared/FILES_SHA256.json']=SOURCE/'FILES_SHA256.json'
    files.update({'inputs/'+n:p for n,p in inputs.items()})
    for n,pin in read(SOURCE/'FILES_SHA256.json')['files'].items():assert sha(SOURCE/n)==pin['sha256'] and (SOURCE/n).stat().st_size==pin['bytes']
    members={n:dict(sha256=sha(p),bytes=p.stat().st_size) for n,p in files.items()}
    assert sum(p['bytes'] for p in members.values())<2*1024**2
    payload={n:base64.b64encode(p.read_bytes()).decode() for n,p in files.items()}
    script='''from pathlib import Path
import base64,datetime,hashlib,json,os,subprocess,sys,traceback
base=Path(%r);members=json.loads(%r);payload=json.loads(%r)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not base.exists() and not base.is_symlink() and base.parent.resolve()==base.parent
assert set(members)==set(payload)
for name,value in payload.items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 b=base64.b64decode(value,validate=True);assert hashlib.sha256(b).hexdigest()==members[name]['sha256'] and len(b)==members[name]['bytes']
base.mkdir()
for name,value in payload.items():
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(value,validate=True))
try:
 owners=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=p.read_bytes().decode(errors='replace').split('\\0')
   if not argv or 'python' not in Path(argv[0]).name or p.parent.name==str(os.getpid()):continue
   assert not any(str(base/'stage') in x for x in argv),'Duplicate stage worker'
   for t in (p.parent/'task').iterdir():
    try:cpus=sorted(os.sched_getaffinity(int(t.name)))
    except ProcessLookupError:continue
    if len(cpus)<=16 and 107 in cpus:owners.append(dict(pid=p.parent.name,tid=t.name,cpus=cpus))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 assert not owners,owners
 assert 'EXITED' in subprocess.run(['supervisorctl','status','guardfed_celeba_hybrid_screen32'],capture_output=True,text=True).stdout
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',GUARDFED_CPU_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 source=base/'source_prepared';stage=base/'stage';inputs=base/'inputs';approval=read(inputs/'BIND_APPROVAL.json')
 command=['taskset','-c','107','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(source/'bind_stage.py'),
 '--old-screen','/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009','--repo','/workspace/GuardFed-celeba-expanded',
 '--fl-runner','/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage/run_fullcoverage.py',
 '--summary',str(inputs/'SUMMARY32.json'),'--summary-sha256',sha(inputs/'SUMMARY32.json'),
 '--root32',str(inputs/'ROOT32.json'),'--root32-sha256',sha(inputs/'ROOT32.json'),
 '--approval',str(inputs/'BIND_APPROVAL.json'),'--approval-sha256',sha(inputs/'BIND_APPROVAL.json'),'--output',str(stage)]
 r=subprocess.run(command,env=env,capture_output=True,text=True)
 (base/'BIND_COMMAND.json').write_text(json.dumps(dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\\n')
 r.check_returncode();bound=json.loads(r.stdout);assert (bound['new'],bound['reused'],bound['canaries'],bound['execution_authorized'])==(96,4,7,False)
 package=read(stage/'PACKAGE_SHA256.json')
 for name,digest in package['files'].items():assert sha(stage/name)==digest
 protected=read(stage/'full_scope.json')['protected_source_hashes'];repo=Path('/workspace/GuardFed-celeba-expanded')
 for name,digest in protected.items():assert sha(repo/name)==digest
 assert not any((stage/n).exists() for n in ('EXECUTION_AUTHORIZATION.json','GATE_ACCEPTANCE.json','gate_runs','runs','queue_progress.json'))
 receipt=dict(bound,status='ACTUAL_HYBRID100_METADATA_BOUND_NO_EXECUTION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source_seal_sha256=sha(source/'FILES_SHA256.json'),summary_sha256=sha(inputs/'SUMMARY32.json'),root32_sha256=sha(inputs/'ROOT32.json'),
 approval_sha256=sha(inputs/'BIND_APPROVAL.json'),source_data_hashes=protected,CPU107_owners=owners,helper_CPU=107,test=False,services_changed=False)
 (base/'BOUND_IDENTITY.json').write_text(json.dumps(receipt,indent=2)+'\\n')
 names=['BOUND_IDENTITY.json','BIND_COMMAND.json']+[p.relative_to(base).as_posix() for p in stage.rglob('*') if p.is_file()]
 assert sum((base/n).stat().st_size for n in names)<3*1024**2
 print(json.dumps(dict(receipt=receipt,files={n:dict(sha256=sha(base/n),bytes=(base/n).stat().st_size,data=base64.b64encode((base/n).read_bytes()).decode()) for n in names})))
except BaseException as e:
 (base/'BIND_FAILURE.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),indent=2)+'\\n');raise
'''%(REMOTE,json.dumps(members),json.dumps(payload))
    ast.parse(script);save(HERE/'TRANSFER_MEMBERS.json',dict(members=members))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',"env CUDA_VISIBLE_DEVICES='' GUARDFED_CPU_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 107 ionice -c 3 nice -n 10 python -B -"],input=script.encode(),capture_output=True,timeout=180)
    save(HERE/'BIND_TRANSPORT_COMMAND.json',dict(returncode=result.returncode,stderr=result.stderr.decode(errors='replace')))
    if result.returncode:save(HERE/'BIND_FAILURE_OUTPUT.json',dict(stdout=result.stdout.decode(errors='replace')))
    result.check_returncode();response=json.loads(result.stdout)
    target=HERE/'bound_metadata';target.mkdir()
    for name,pin in response['files'].items():
        p=target/name;assert p.resolve().is_relative_to(target.resolve());p.parent.mkdir(parents=True,exist_ok=True)
        b=base64.b64decode(pin['data'],validate=True);assert hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes']
        with p.open('xb') as f:f.write(b)
    save(HERE/'BOUND_TRANSFER_VERIFICATION.json',dict(status='BOUND_METADATA_MEMBERS_RECEIVED_NOT_TRAINING',members=len(response['files']),
        package_sha256=response['receipt']['package_sha256'],files={n:{k:v for k,v in p.items() if k!='data'} for n,p in response['files'].items()},final_test=False))
    print(json.dumps(response['receipt']))
if __name__=='__main__':main()
