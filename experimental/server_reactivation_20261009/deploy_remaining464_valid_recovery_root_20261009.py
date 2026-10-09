"""Root-reviewed exact464 deployment and targeted launch after import11 closes."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'tmp/celeba_valid_gpu_remaining464_prepared_20261009'
LOCAL=ROOT/'tmp/celeba_valid_gpu_remaining464_execution_20261009'
REMOTE='/workspace/guardfed_checks'
REVIEW=REMOTE+'/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_REMAINING464.json'
PROGRAM='guardfed_celeba_valid_gpu_remaining464_20261009'
CONFIG='/etc/supervisor/conf.d/'+PROGRAM+'.conf'
PACKAGE='fa5626ad0ab8be12ac501aea531d7b8ad2f2c05b1a18937dd86fb3708acc8d6b'
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
SHA=lambda b:hashlib.sha256(b).hexdigest()
sha=lambda p:SHA(p.read_bytes())
read=lambda p:json.loads(p.read_bytes())
save=lambda p,v:p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
resume_prepared='--resume-prepared' in sys.argv
assert sha(PKG/'PACKAGE_SHA256.json')==PACKAGE
assert (LOCAL.exists() and not (LOCAL/'ROOT_REMOTE_DEPLOYMENT.json').exists()) if resume_prepared else not LOCAL.exists()
assert read(ROOT/'tmp/celeba_valid_gpu_recovery_execution_20261009/preserved11_import/ROOT_OFFSERVER_IMPORT_VERIFICATION.json')['new_CNN_inference']==0
current=read(ROOT/'tmp/celeba_valid_gpu_recovery_execution_20261009/cumulative_436_accepted.json')
manifest=read(PKG/'manifest.json')
inventory=read(ROOT/'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json')
assert len(set(manifest['ids']))==464 and set(manifest['ids'])=={r['id'] for r in inventory['records']}-set(current['accepted_ids'])
payload={}
for name,row in read(PKG/'PACKAGE_SHA256.json')['members'].items():
    rel=PurePosixPath(name); path=PKG/name
    assert not rel.is_absolute() and '..' not in rel.parts and path.resolve().is_relative_to(PKG.resolve()) and not path.is_symlink()
    data=path.read_bytes(); assert SHA(data)==row['sha256'] and len(data)==row['bytes']
    payload[PKG.name+'/'+name]=data
payload[PKG.name+'/PACKAGE_SHA256.json']=(PKG/'PACKAGE_SHA256.json').read_bytes()
if not resume_prepared:
    check=subprocess.run([sys.executable,str(PKG/'selfcheck.py')],capture_output=True,check=True)
    LOCAL.mkdir(); (LOCAL/'ROOT_LOCAL_CHECK.log').write_bytes(check.stdout+check.stderr)
approval=read(PKG/'ROOT_REVIEW_TEMPLATE.json')
approval.update(status='ROOT_APPROVED_GPU_VALID_RECOVERY_V1',queue_package_sha256=PACKAGE,
                approved_ids=manifest['ids'],execute_new465=True,execute_remaining464=True,
                gpu_uuid='GPU-da357477-30a7-fddc-344b-a20513b9a2d0',
                approved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                current436_collector_sha256=sha(ROOT/'tmp/celeba_valid_gpu_recovery_execution_20261009/cumulative_436_accepted.json'),
                review_scope='Exact remaining464 complements current436. Frozen425 is historical lineage; import11 accepted separately. No new methods, changes to science, training, test or numerical retry.')
review=LOCAL/'ROOT_REVIEW_REMAINING464.json'
if resume_prepared:
    approval=read(review)
    assert approval['approved_ids']==manifest['ids'] and approval['queue_package_sha256']==PACKAGE
else: save(review,approval)
review_sha=sha(review)
config=(PKG/(PROGRAM+'.conf.template')).read_bytes().replace(b'__REVIEW_SHA256__',review_sha.encode()).replace(b'__QUEUE_PACKAGE_SHA256__',PACKAGE.encode())
assert b'__' not in config and b'autorestart=false' in config and b'stopasgroup=true' in config
if resume_prepared: assert (LOCAL/(PROGRAM+'.conf')).read_bytes()==config
else: (LOCAL/(PROGRAM+'.conf')).write_bytes(config)
payload['celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_REMAINING464.json']=review.read_bytes()
archive=LOCAL/'deployment.tar.gz'
if not resume_prepared:
    with tarfile.open(archive,'w:gz') as bundle:
        for name,data in sorted(payload.items()):
            member=tarfile.TarInfo(name); member.size=len(data); member.mode=0o644; bundle.addfile(member,io.BytesIO(data))
archive_sha=sha(archive); specs={n:{'sha256':SHA(b),'bytes':len(b)} for n,b in payload.items()}
payload_record={'archive_sha256':archive_sha,'members':specs,'config_sha256':SHA(config)}
if resume_prepared: assert read(LOCAL/'DEPLOYMENT_PAYLOAD.json')==payload_record
else: save(LOCAL/'DEPLOYMENT_PAYLOAD.json',payload_record)
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
def remote(code,values,timeout=60):
    source='\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+code
    result=subprocess.run(ssh+['python -c '+shlex.quote(source)],capture_output=True,timeout=timeout)
    if result.returncode:
        index=len(list(LOCAL.glob('REMOTE_FAILURE_*.json')))
        save(LOCAL/('REMOTE_FAILURE_%03d.json'%index),{'returncode':result.returncode,'stdout':result.stdout.decode(errors='replace'),'stderr':result.stderr.decode(errors='replace'),'no_numerical_retry':True})
        result.check_returncode()
    return result
pre="""from pathlib import Path
import hashlib,json,subprocess,os
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==GUIDE
assert not Path(PKG).exists() and not Path(REVIEW).exists() and not Path(CONFIG).exists() and not Path(OUT).exists()
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace'); affinity=os.sched_getaffinity(int(p.name))
        if 'python' in cmd and '/workspace/' in cmd and len(affinity)<=16:
            assert not affinity & {105,106}, 'CPU105/106 occupied: '+cmd
            processes.append({'pid':int(p.name),'affinity':sorted(affinity)})
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
observed=subprocess.run(['supervisorctl','status'],capture_output=True,text=True)
assert observed.returncode in (0,3), observed.stderr
status=observed.stdout
assert PROGRAM not in status and 'sglang' in status and any('sglang' in line and 'STOPPED' in line for line in status.splitlines())
print(json.dumps({'status':'FRESH_TARGET_NO_DUPLICATE_CPU105_106','services':status,'processes':processes}))
"""
values=dict(GUIDE=GUIDE,PKG=REMOTE+'/'+PKG.name,REVIEW=REVIEW,CONFIG=CONFIG,OUT=approval['output_parent'],PROGRAM=PROGRAM)
(LOCAL/'REMOTE_PREFLIGHT.json').write_bytes(remote(pre,values).stdout)
remote_archive=REMOTE+'/remaining464_deployment_'+archive_sha+'.tar.gz'
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350',str(archive),'root@89.22.197.55:'+remote_archive],check=True,timeout=120)
install="""from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==GUIDE
assert hashlib.sha256(Path(ARCHIVE).read_bytes()).hexdigest()==ARCHIVE_SHA and not Path(PKG).exists() and not Path(REVIEW).exists() and not Path(CONFIG).exists()
with tarfile.open(ARCHIVE) as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==len(SPECS) and set(bundle.getnames())==set(SPECS)
    for member in bundle.getmembers():
        name=PurePosixPath(member.name); data=bundle.extractfile(member).read(); row=SPECS[member.name]
        assert member.isfile() and not name.is_absolute() and '..' not in name.parts and len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    Path(PKG).mkdir()
    for member in bundle.getmembers():
        target=Path('/workspace/guardfed_checks')/member.name; target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as out:out.write(bundle.extractfile(member).read())
    with Path(CONFIG).open('xb') as out:out.write(CONFIG_BYTES)
for name,row in SPECS.items():assert hashlib.sha256((Path('/workspace/guardfed_checks')/name).read_bytes()).hexdigest()==row['sha256']
assert hashlib.sha256(Path(CONFIG).read_bytes()).hexdigest()==CONFIG_SHA
cmd=['/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',PKG+'/remaining.py','inspect','--review',REVIEW,'--review-sha256',REVIEW_SHA,'--package-sha256',PACKAGE]
checked=subprocess.run(cmd,capture_output=True,text=True,check=True)
print(json.dumps({'status':'EXACT464_DEPLOYED_INSPECT_PASS_NOT_STARTED','inspect':checked.stdout,'package_sha256':PACKAGE,'review_sha256':REVIEW_SHA,'archive_sha256':ARCHIVE_SHA,'config_sha256':CONFIG_SHA}))
"""
values.update(ARCHIVE=remote_archive,ARCHIVE_SHA=archive_sha,SPECS=specs,CONFIG_BYTES=config,CONFIG_SHA=SHA(config),REVIEW_SHA=review_sha,PACKAGE=PACKAGE)
(LOCAL/'ROOT_REMOTE_DEPLOYMENT.json').write_bytes(remote(install,values).stdout)
launch="""from pathlib import Path
import hashlib,json,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==GUIDE
assert hashlib.sha256(Path(CONFIG).read_bytes()).hexdigest()==CONFIG_SHA and not Path(OUT).exists()
logs=[]
for command in [['supervisorctl','reread'],['supervisorctl','update',PROGRAM],['supervisorctl','start',PROGRAM],['supervisorctl','status',PROGRAM]]:
    result=subprocess.run(command,capture_output=True,text=True,check=True);logs.append({'command':command,'stdout':result.stdout,'stderr':result.stderr})
assert 'RUNNING' in logs[-1]['stdout']
print(json.dumps({'status':'TARGETED464_SERVICE_START_OBSERVED','logs':logs,'review_sha256':REVIEW_SHA,'queue_package_sha256':PACKAGE,'accepted_new_n':0}))
"""
(LOCAL/'ROOT_LAUNCH.json').write_bytes(remote(launch,values).stdout)
print(json.dumps({'status':'ROOT_EXACT464_DEPLOYED_INSPECTED_TARGETED_START','review_sha256':review_sha,'queue_package_sha256':PACKAGE,'service':PROGRAM,'accepted_total_before_queue':436,'accepted_new_by_launch':0}))
