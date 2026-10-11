"""Root-only future fixed3 deployment; source preparation does not invoke main."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, base64, hashlib, json, subprocess, time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
API = REPO / 'tmp/FL100_uniform_cuda_formal_adapter_prepare_20261011'
MAPPING = REPO / 'tmp/FL100_cuda_runtime_mapping_prepare_20261011'
H = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def save(p, v):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(v, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


# Derived from install_old800_cuda_canary_root_once_20261011.py's single
# small-payload/deployment-seal/preflight/supervisor path. No scope discovery.
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,time
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p)}
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def run(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=70)
 assert p.returncode==0,(argv,p.returncode,p.stdout,p.stderr)
 return p.stdout
assert __debug__ and sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
payload=json.loads(base64.b64decode('__PAYLOAD__'))
prepared=json.loads(base64.b64decode('__PREPARED__'))
actual=json.loads(base64.b64decode('__ACTUAL__'))
base=Path(prepared['namespace']);repo=Path(actual['repo']);service=prepared['service']
py='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
conf=Path('/etc/supervisor/conf.d/'+service+'.conf')
assert not base.exists() and not conf.exists(),'Preserve prior namespace/service'
assert actual['execution_authorized'] and time.time()<actual['deadline_unix']
mapping=json.loads(base64.b64decode(payload['RUNTIME_DEPENDENCIES.json']))
cohort=json.loads(base64.b64decode(payload['adapter/EXACT100.json']))
sourcepins=json.loads(base64.b64decode(payload['adapter/SOURCE_PINS.json']))
assert len(cohort['records'])==100 and len({r['id'] for r in cohort['records']})==100
assert set(mapping)==set(sourcepins['packages'])
external={}
for identity_source,roles in mapping.items():
 assert set(roles)==set(sourcepins['packages'][identity_source]) and len(roles)==11
 for role,item in roles.items():
  p=Path(item['path']);assert p.is_absolute() and ':' not in str(p) and p.resolve()==p and not p.is_symlink()
  assert item['sha256']==sourcepins['packages'][identity_source][role]['sha256'] and sha(p)==item['sha256'],str(p)
  assert str(p) not in external or external[str(p)]==item['sha256']
  external[str(p)]=item['sha256']
original=json.loads(base64.b64decode(payload['ORIGINAL_INPUTS.json']))
assert original['repo']==str(repo) and len(original['source_and_data'])==11
author=Path(actual['author_decision']['path'])
assert sha(author)==actual['author_decision']['sha256']==sourcepins['author_sha256']
decision=read(author)
assert decision['approved']['uniform_CUDA_evaluation'] and decision['approved']['corresponding_Full_included'] and not decision['existing_diagnostic_retroactively_adopted']
base.mkdir()
for rel,b64 in payload.items():
 p=base/rel;assert '..' not in Path(rel).parts and p.is_relative_to(base)
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(base64.b64decode(b64))
with (base/'author.json').open('xb') as f:f.write(author.read_bytes())
gpu=actual['gpu_uuid'];protected=actual['protected_services']
approval=read(base/'adapter/ROOT_INPUTS.template.json')
approval.update(status='ROOT_AUTHORIZED_UNIFORM_CUDA_FL100_FORMAL_V1',execution_authorized=True,
 author_decision=pin(base/'author.json'),exact_cohort=pin(base/'adapter/EXACT100.json'),
 adapter=pin(base/'adapter/adapter.py'),adapter_sha256=sha(base/'adapter/adapter.py'),
 source_seal=pin(base/'adapter/FILES_SHA256.json'),formal_GPU_body=pin(base/'adapter/gpu_body.py'),
 resource_gate_source=pin(base/'guard.py'),repo=str(repo),gpu_uuid=gpu,allowed_cpus=[111],
 output_namespace=str(base),protected_services=protected)
auth={'execution_authorized':True,'stage':'FL100_CUDA_REEVALUATION_FIXED3_CANARY_FIRST',
 'ordered_ids':[r['id'] for r in cohort['records']],'canary_ids':prepared['canary_ids'],
 'exact_cohort':pin(base/'adapter/EXACT100.json'),'adapter':pin(base/'adapter/adapter.py'),
 'gpu_uuid':gpu,'allowed_cpus':[111],'deadline_unix':actual['deadline_unix'],'protected_services':protected,
 'approval_base':approval,'training':False,'test':False,'automatic_retry':False,
 'root_install_inputs_sha256':'__ACTUAL_SHA__',
 'source_review_pins':{k:pin(base/'reviews'/(k+'.json')) for k in actual['reviews']},'original_review_refs':actual['reviews']}
save(base/'ROOT_QUEUE_AUTH.json',auth)
# Hash original data once, in a single low-priority CPU111 subprocess after the
# exact original resources() ownership/budget gate; no scientific import.
hash_helper="""import guard,worker,os,time\nfrom pathlib import Path\nfrom datetime import datetime,timezone\nq=guard.read(guard.BASE/'ROOT_QUEUE_AUTH.json')\nassert time.time()<q['deadline_unix']\nowner=guard.resources(dict(q,id=q['canary_ids'][0]))\nf=os.statvfs(guard.BASE);assert f.f_bavail*f.f_frsize>=10*1024**3\noriginal=guard.read(guard.BASE/'ORIGINAL_INPUTS.json');files={}\nfor name,h in original['source_and_data'].items():\n p=Path(name);before=p.stat();assert worker.sha(p)==h,'Original source/data changed: '+name\n after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)\n files[name]={'sha256':h,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'ino':after.st_ino}\nassert time.time()<q['deadline_unix']\nprint(guard.json.dumps({'status':'ORIGINAL_SOURCE_DATA_FULL_HASH_BEFORE_START','utc':datetime.now(timezone.utc).isoformat(),'files':files,'owner_resource_check':owner},allow_nan=False))\n"""
hash_result=subprocess.run(['nice','-n','10','ionice','-c','3','taskset','-c','111',py,'-B','-c',hash_helper],cwd=base,capture_output=True,timeout=240)
save(base/'SOURCE_DATA_HASH_COMMAND.json',{'exit_code':hash_result.returncode,'stderr':hash_result.stderr.decode(errors='replace')})
assert hash_result.returncode==0,hash_result.stderr.decode(errors='replace')
save(base/'SOURCE_DATA_FINGERPRINT.json',json.loads(hash_result.stdout))
save(base/'DEPLOYMENT_SEAL.json',{'status':'ROOT_DEPLOYED_CODE_AND_COHORT_NOT_SCIENTIFIC_RESULTS',
 'files':{rel:{'sha256':sha(base/rel),'bytes':(base/rel).stat().st_size} for rel in
 list(payload)+['author.json','ROOT_QUEUE_AUTH.json','SOURCE_DATA_FINGERPRINT.json','SOURCE_DATA_HASH_COMMAND.json']},'external_code_pins':external})
first=prepared['canary_ids'][0]
fresh=json.loads(run(['nice','-n','10','ionice','-c','3','taskset','-c','111',py,str(base/'guard.py'),'--id',first]))
assert fresh['pass'] and fresh['gpu_uuid']==gpu and fresh['allowed_cpus']==[111]
save(base/'ROOT_INSTALL_PREFLIGHT.json',fresh)
# The lightweight controller keeps a broad mask; only its one preflight/worker
# child uses CPU111. Pinning the controller there would fail the ownership gate.
text='[program:'+service+']\ncommand=/usr/bin/nice -n 10 /usr/bin/ionice -c 3 '+py+' -B '+str(base/'run_queue.py')+' canary\ndirectory='+str(base)+'\nautostart=false\nautorestart=false\nstartsecs=1\nstartretries=0\nstopasgroup=true\nkillasgroup=true\nenvironment=OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1"\nstdout_logfile='+str(base/'canary.supervisor.stdout')+'\nstderr_logfile='+str(base/'canary.supervisor.stderr')+'\n'
with conf.open('x',encoding='utf-8') as f:f.write(text)
commands=[]
for argv in [['supervisorctl','reread'],['supervisorctl','update',service],['supervisorctl','start',service],['supervisorctl','status',service]]:
 commands.append({'command':argv,'stdout':run(argv)})
report={'status':'ROOT_INSTALLED_AND_STARTED_FIXED3_FL_INTERFACE_CANARIES_NOT100_COMPLETE',
 'utc':datetime.now(timezone.utc).isoformat(),'namespace':str(base),'service':service,'canary_ids':prepared['canary_ids'],
 'cohort':pin(base/'adapter/EXACT100.json'),'auth':pin(base/'ROOT_QUEUE_AUTH.json'),'seal':pin(base/'DEPLOYMENT_SEAL.json'),
 'preflight':pin(base/'ROOT_INSTALL_PREFLIGHT.json'),'commands':commands,'scientific_results_adopted':0,'remaining97_started':False}
save(base/'ROOT_START.json',report)
print(json.dumps(report,allow_nan=False))
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--inputs-sha256', required=True)
    p.add_argument('--source-seal-sha256', required=True)
    p.add_argument('--receipt-dir', type=Path, required=True)
    a = p.parse_args()
    assert __debug__ and H(a.inputs.read_bytes()) == a.inputs_sha256
    assert H((HERE/'FILES_SHA256.json').read_bytes()) == a.source_seal_sha256
    seal = read(HERE/'FILES_SHA256.json')
    for n, item in seal['files'].items():
        assert Path(n).name == n and H((HERE/n).read_bytes()) == item['sha256']
    actual, prepared = read(a.inputs), read(HERE/'ROOT_PREPARATION.json')
    assert actual['status']=='ROOT_AUTHORIZED_FL100_FIXED3_CALLER_V1' and actual['execution_authorized'] is True
    assert actual['caller_source_seal_sha256']==a.source_seal_sha256
    assert actual['formal_source_seal_sha256']==prepared['formal_source_seal_sha256']
    assert actual['mapping_source_seal_sha256']==prepared['mapping_source_seal_sha256']
    assert actual['canary_ids']==prepared['canary_ids'] and actual['allowed_cpus']==[111] and actual['GPU_index']==0
    assert isinstance(actual['gpu_uuid'],str) and actual['gpu_uuid'].startswith('GPU-')
    assert time.time()<actual['deadline_unix']<=time.time()+72*3600
    assert actual['repo']=='/workspace/GuardFed-celeba-expanded'
    assert actual['source_reviews_accepted_by_root'] is True and set(actual['reviews'])=={'formal_api','caller','runtime_mapping'}
    assert isinstance(actual['protected_services'],dict) and actual['protected_services']
    payload = {}
    for role, item in actual['reviews'].items():
        b=Path(item['path']).read_bytes();assert len(b)<2*1024**2 and H(b)==item['sha256']
        payload['reviews/'+role+'.json']=base64.b64encode(b).decode()
    source_seal=read(API/'FILES_SHA256.json')
    assert H((API/'FILES_SHA256.json').read_bytes())==prepared['formal_source_seal_sha256']
    for n, item in source_seal['files'].items():
        b=(API/n).read_bytes();assert Path(n).name==n and H(b)==item['sha256'] and len(b)==item['bytes']
        payload['adapter/'+n]=base64.b64encode(b).decode()
    payload['adapter/FILES_SHA256.json']=base64.b64encode((API/'FILES_SHA256.json').read_bytes()).decode()
    for n, item in prepared['caller_sources'].items():
        b=(HERE/n).read_bytes();assert H(b)==item['sha256'] and len(b)==item['bytes']
        payload[n]=base64.b64encode(b).decode()
    assert H((MAPPING/'FILES_SHA256.json').read_bytes())==prepared['mapping_source_seal_sha256']
    mapping_seal=read(MAPPING/'FILES_SHA256.json')['files']
    for n in ['RUNTIME_DEPENDENCIES.json','ROOT_ARGS.template.json']:
        assert H((MAPPING/n).read_bytes())==mapping_seal[n]['sha256']
    payload['RUNTIME_DEPENDENCIES.json']=base64.b64encode((MAPPING/'RUNTIME_DEPENDENCIES.json').read_bytes()).decode()
    original=read(MAPPING/'ROOT_ARGS.template.json')['original_inputs']
    payload['ORIGINAL_INPUTS.json']=base64.b64encode((json.dumps(original,indent=2)+'\n').encode()).decode()
    remote=REMOTE
    for token, value in [('__PAYLOAD__',payload),('__PREPARED__',prepared),('__ACTUAL__',actual)]:
        remote=remote.replace(token,base64.b64encode(json.dumps(value).encode()).decode())
    remote=remote.replace('__ACTUAL_SHA__',a.inputs_sha256)
    compile(remote,'<FL100-fixed3-install>','exec')
    assert not a.receipt_dir.exists(),'Preserve first attempt; use a root-reviewed fresh namespace for recovery'
    a.receipt_dir.mkdir(parents=True)
    command=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','python3 -']
    save(a.receipt_dir/'COMMAND.json',dict(utc=datetime.now(timezone.utc).isoformat(),command=command,
         installer_sha256=H(Path(__file__).read_bytes()),inputs_sha256=a.inputs_sha256,
         source_seal_sha256=a.source_seal_sha256,remote_script_sha256=H(remote.encode())))
    (a.receipt_dir/'REMOTE.py').write_bytes(remote.encode())
    try:
        result=subprocess.run(command,input=remote.encode(),capture_output=True,timeout=600)
        (a.receipt_dir/'STDOUT').write_bytes(result.stdout);(a.receipt_dir/'STDERR').write_bytes(result.stderr)
        save(a.receipt_dir/'EXIT.json',dict(exit_code=result.returncode,utc=datetime.now(timezone.utc).isoformat(),
             stdout_sha256=H(result.stdout),stderr_sha256=H(result.stderr)))
        assert result.returncode==0,result.stderr.decode(errors='replace')
        print(result.stdout.decode())
    except BaseException as exc:
        if isinstance(exc,subprocess.TimeoutExpired):
            (a.receipt_dir/'STDOUT').write_bytes(exc.stdout or b'');(a.receipt_dir/'STDERR').write_bytes(exc.stderr or b'')
        save(a.receipt_dir/'FAILURE.json',dict(error_type=type(exc).__name__,error=str(exc),automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
