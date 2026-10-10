"""Bind actual reviews and launch one exact-three CPU interface gate, once."""
from pathlib import Path
import argparse, base64, datetime, hashlib, json, shlex, subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/celeba_added_cnn_exact3_root_execution_20261010'
SRC=ROOT/'tmp/celeba_added_cnn_three_view_gate_preparation_20261010'
BASE='/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010'
PACKAGE='49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434'
PROGRAM='guardfed_added_cnn_exact3_gate'
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',type=Path,required=True); ap.add_argument('--preflight-sha256',required=True); a=ap.parse_args()
    assert not (OUT/'START_RECEIPT.json').exists() and not (OUT/'START_FAILURE.json').exists(), 'Preserve attempt; no automatic retry'
    pbytes=a.preflight.read_bytes(); p=read(a.preflight)
    assert sha(pbytes)==a.preflight_sha256
    assert p['status']=='ROOT_LINUX_EXACT3_PREFLIGHT_PASS'
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(p['utc'])).total_seconds()
    assert 0<=age<=220, ('Refresh resource/process/stat identity only before launch',age)
    assert p['cpu_affinity']==list(range(120,128))
    for k in ('no_duplicate_gate','all_thread_cpus_free','source_model_data_hashes_verified','selected_producers_quiescent','services_healthy','gpu_health_verified','cgroup_and_memory_headroom_verified','storage_headroom_verified'):
        assert p[k] is True,k
    assert p['nominal_reserved_cores_including_gate']<=p['actual_quota_cores']
    assert set(range(120,128))<=set(p['eligible_cpus'])
    review=(OUT/'ROOT_SOURCE_REVIEW.json').read_bytes(); review_sha=sha(review)
    assert review_sha=='e81d178f0390ca9c5be0939a0c62ff9da762b2b3dc59f51b9b8d82b26f878323'
    manifest=(SRC/'MANIFEST.json').read_bytes()
    auth=dict(status='ROOT_AUTHORIZED_EXACT3_VALID_IMAGE_INTERFACE_GATE',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), package_sha256=PACKAGE,
        manifest_sha256=sha(manifest),source_review_sha256=review_sha,linux_preflight_sha256=sha(pbytes),
        exact_ids=[r['id'] for r in read(SRC/'MANIFEST.json')['records']],cpu_affinity=list(range(120,128)),
        device='cpu',threads=8,max_processes=1,test=False,training=False,
        scope='Exactly three already accepted terminal70 checkpoints; real valid images, original root reconstruction, original native replay and shared calibration interface only',
        no_change_to_training_queues=True,not_full100=True,not_final_endpoint=True,
        failure_policy='Preserve scientific mismatch and all preflight/log evidence; no tolerance change or automatic retry')
    save(OUT/'AUTHORIZATION.json',auth); abytes=(OUT/'AUTHORIZATION.json').read_bytes()
    cmd=['/usr/bin/env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','OPENBLAS_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','-c','120-127','/usr/bin/ionice','-c','3','/usr/bin/nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',BASE+'/source/candidate.py','--package-sha256',PACKAGE,'--source-review',BASE+'/ROOT_SOURCE_REVIEW.json','--source-review-sha256',review_sha,'--preflight',BASE+'/LINUX_PREFLIGHT.json','--preflight-sha256',sha(pbytes),'--authorization',BASE+'/AUTHORIZATION.json','--authorization-sha256',sha(abytes),'--output',BASE+'/outputs/attempt001']
    config='\n'.join(['[program:'+PROGRAM+']','command='+shlex.join(cmd),'directory=/workspace/GuardFed-celeba-expanded','autostart=false','autorestart=false','startsecs=2','startretries=0','stopsignal=TERM','stopasgroup=true','killasgroup=true','stdout_logfile='+BASE+'/execution/stdout.log','stderr_logfile='+BASE+'/execution/stderr.log','stdout_logfile_maxbytes=10MB','stdout_logfile_backups=1','stderr_logfile_maxbytes=10MB','stderr_logfile_backups=1',''])
    (OUT/(PROGRAM+'.conf')).write_text(config,encoding='utf-8')
    payload={'base':BASE,'program':PROGRAM,'package':PACKAGE,'files':{name:{'sha256':sha(b),'bytes':len(b),'base64':base64.b64encode(b).decode()} for name,b in [('LINUX_PREFLIGHT.json',pbytes),('AUTHORIZATION.json',abytes)]},'config':config,'config_sha256':sha(config.encode()),'source_review_sha256':review_sha}
    remote=r'''
import base64,datetime,hashlib,json,pathlib,subprocess,sys
p=json.load(sys.stdin); base=pathlib.Path(p['base']); program=p['program']
assert str(base)=='/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010' and program=='guardfed_added_cnn_exact3_gate'
config=pathlib.Path('/etc/supervisor/conf.d')/(program+'.conf')
assert not config.exists() and not (base/'outputs/attempt001').exists() and not (base/'START_RECEIPT.json').exists(), 'Existing attempt; no blind retry'
assert hashlib.sha256((base/'source/FILES_SHA256.json').read_bytes()).hexdigest()==p['package']
assert hashlib.sha256((base/'ROOT_SOURCE_REVIEW.json').read_bytes()).hexdigest()==p['source_review_sha256']
for name,pin in p['files'].items():
 assert name in ('LINUX_PREFLIGHT.json','AUTHORIZATION.json')
 b=base64.b64decode(pin['base64'],validate=True); assert hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes']
 assert not (base/name).exists(); (base/name).write_bytes(b)
pre=json.loads((base/'LINUX_PREFLIGHT.json').read_text()); age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(pre['utc'])).total_seconds()
assert 0<=age<=280, 'Preflight expired before supervisor launch'
assert hashlib.sha256(p['config'].encode()).hexdigest()==p['config_sha256']
(base/'execution').mkdir(exist_ok=False)
config.write_text(p['config'])
commands=[]
try:
 for argv in (['supervisorctl','reread'],['supervisorctl','add',program],['supervisorctl','start',program],['supervisorctl','status',program]):
  c=subprocess.run(argv,capture_output=True,text=True,timeout=25); commands.append({'argv':argv,'exit':c.returncode,'stdout':c.stdout,'stderr':c.stderr})
  assert c.returncode==0, commands[-1]
 r={'status':'ROOT_EXACT3_SUPERVISOR_STARTED_NOT_SCIENTIFIC_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'program':program,'commands':commands,'config_sha256':p['config_sha256'],'authorization_sha256':p['files']['AUTHORIZATION.json']['sha256'],'linux_preflight_sha256':p['files']['LINUX_PREFLIGHT.json']['sha256'],'package_sha256':p['package'],'autostart':False,'autorestart':False,'startretries':0,'new_scientific_acceptances':0}
except BaseException:
 import traceback
 r={'status':'ROOT_EXACT3_START_FAILURE_PRESERVED_NO_RETRY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':commands,'traceback':traceback.format_exc()}
(base/'START_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r))
if 'FAILURE' in r['status']: sys.exit(1)
'''
    (OUT/'start_remote.py').write_text(remote,encoding='utf-8')
    remote_cmd='env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -c '+shlex.quote(remote)
    c=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',remote_cmd],input=json.dumps(payload).encode(),capture_output=True,timeout=80)
    (OUT/'START_STDOUT.txt').write_bytes(c.stdout); (OUT/'START_STDERR.txt').write_bytes(c.stderr)
    if c.returncode:
        save(OUT/'START_FAILURE.json',{'status':'ROOT_EXACT3_START_FAILURE_PRESERVED_NO_RETRY','exit':c.returncode,'stdout':c.stdout.decode(errors='replace'),'stderr':c.stderr.decode(errors='replace')})
        print(c.stdout.decode(errors='replace')); print(c.stderr.decode(errors='replace')); raise SystemExit(c.returncode)
    r=json.loads(c.stdout); save(OUT/'START_RECEIPT.json',r)
    print(json.dumps(r))

if __name__=='__main__': main()
