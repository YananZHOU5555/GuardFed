
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=lambda p:json.loads(Path(p).read_bytes())
assert H('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
stage=Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011')
runtime=stage/'runtime';package=Path('/workspace/GuardFed-celeba-expanded/tmp/celeba_gradient_fullcoverage_gates_prepare_20261010')
python='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
scope=R(stage/'scope.json');assert H(stage/'scope.json')=='22eba673a701d569b35736875245373d15696bca6a1ee9421165a5ed0e7232ff'
assert H(runtime/'ROOT_AUTHORIZATION.json')=='d695be6b9930cf5d57b9038834e98df24ffbddb0cfe0a190e9c33a04bb5f4ab8'
index=1;entry=scope['jobs'][index];job=entry['id']
out=runtime/('job_01');out.mkdir(exist_ok=False)
previous=runtime/'PREVIOUS_MAIN.json' if index==0 else Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011/runtime/job_00_attempt002/RESOURCE.json')
receipt=out/'RESOURCE.json'
def run(name,argv,env=None):
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 r=subprocess.run(argv,env=env,capture_output=True)
 (out/(name+'.stdout')).write_bytes(r.stdout);(out/(name+'.stderr')).write_bytes(r.stderr)
 (out/(name+'_COMMAND.json')).write_text(json.dumps(dict(argv=argv,start=start,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=r.returncode,automatic_retry=False),indent=2)+'\n')
 assert r.returncode==0,dict(phase=name,returncode=r.returncode,stderr=r.stderr.decode(errors='replace')[-6000:])
 return r
preflight=['nice','-n','10','ionice','-c','3',python,'-B',str(runtime/'preflight.py'),'--stage',str(stage),'--repo','/workspace/GuardFed-celeba-expanded','--package',str(package),'--authorization',str(runtime/'ROOT_AUTHORIZATION.json'),'--authorization-sha256',H(runtime/'ROOT_AUTHORIZATION.json'),'--job',job,'--previous-main',str(previous),'--previous-main-sha256',H(previous),'--out',str(receipt)]
run('PREFLIGHT',preflight)
r=R(receipt);receipt_sha=H(receipt)
assert r['job_id']==job==r['one_job_runtime_scope']['job_id']
assert r['job_sha256']==entry['job_sha256']==r['one_job_runtime_scope']['job_sha256']
assert r['one_job_runtime_scope']['sequential_index']==index and r['one_job_runtime_scope']['fresh_receipt_required_for_next_job'] is True
assert r['scope_sha256']==H(stage/'scope.json') and r['jobs']=={e['id']:e['job_sha256'] for e in scope['jobs']}
assert 0<=time.time()-r['measured_unix']<=120
assert r['exclusive_cpu_ids']==list(range(88,96)) and not r['coverage192_authorized'] and not r['test_authorized']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
cmd=['taskset','-c','88-95','nice','-n','10','ionice','-c','3',python,'-B',str(package/'adapter.py'),'--stage',str(stage),'--repo','/workspace/GuardFed-celeba-expanded','--job',job,'--approval',str(receipt),'--approval-sha256',receipt_sha]
run('ORIGINAL_ADAPTER',cmd,env)
assert H(receipt)==receipt_sha
result=stage/entry['output'];item=R(result/'item_receipt.json');accepted=R(result/'acceptance.json')
assert item['status']=='CPU3_GATE_ORIGINAL_STRICT_AND_EXTERNAL_IDENTITY_PASS' and accepted['status']=='PASS'
assert accepted['job_sha256']==entry['job_sha256'] and accepted['scope_sha256']==H(stage/'scope.json')
assert H(result/'acceptance.json')==item['acceptance_sha256'] and H(result/'rng_after.json')==item['rng_sha256']
print(json.dumps(dict(status='ONE_ACTUAL_ORIGINAL_CPU3_GATE_PASS_NOT_WHOLE_ACCEPTANCE',index=index,job_id=job,resource_sha256=receipt_sha,acceptance=accepted,item_receipt=item,rng_after=R(result/'rng_after.json'),main_snapshot=r['main_snapshot'],test=False,scientific_table_records=0,coverage192_started=0)))
