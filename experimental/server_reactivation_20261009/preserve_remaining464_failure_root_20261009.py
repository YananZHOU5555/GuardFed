"""Use the unchanged original archiver to preserve the stopped third chunk."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/celeba_valid_gpu_remaining464_execution_20261009/failure_chunk002'
assert not OUT.exists(); OUT.mkdir()
BASE='/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009'
STAGE=BASE+'/remaining464_attempt1/chunk_002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
code=r'''from pathlib import Path
import hashlib,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert hashlib.sha256(Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009/PACKAGE_SHA256.json').read_bytes()).hexdigest()=='6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56'
command=['taskset','-c','105','nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009/recovery.py','backup','--stage',STAGE,'--chunk-index','2','--failure','--review',BASE+'/ROOT_REVIEW_REMAINING464.json','--review-sha256','2b646f10bb47e2e94e1421fd045564b7392b69962b2b51d455c7c0e85420bfac','--package-sha256','6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56']
raise SystemExit(subprocess.run(command).returncode)
'''
bound='BASE='+repr(BASE)+'\nSTAGE='+repr(STAGE)+'\n'+code
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -c '+shlex.quote(bound)],capture_output=True,timeout=90)
(OUT/'REMOTE_STDOUT.json').write_bytes(result.stdout); (OUT/'REMOTE_STDERR.log').write_bytes(result.stderr)
assert result.returncode==0, 'Preserve failure backup attempt; never blind retry'
for name in ('failure_chunk_evidence.tar.gz','failure_remote_archive_inventory.json'):
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+STAGE+'/'+name,str(OUT/name)],check=True,timeout=120)
inventory=json.loads((OUT/'failure_remote_archive_inventory.json').read_bytes())
assert inventory['status']=='PRESERVED_FAILURE_NOT_OFFSERVER_ACCEPTED' and not inventory['accepted_ids'] and inventory['old_model_files_archived']==0
archive=OUT/'failure_chunk_evidence.tar.gz'
assert sha(archive)==inventory['sha256'] and archive.stat().st_size==inventory['bytes']
with tarfile.open(archive) as bundle:
    members=bundle.getmembers()
    assert len(members)==len(set(bundle.getnames()))==inventory['member_n']
    for member in members:
        path=PurePosixPath(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts and member.name in inventory['members']
        data=bundle.extractfile(member).read(); row=inventory['members'][member.name]
        assert hashlib.sha256(data).hexdigest()==row['sha256'] and len(data)==row['bytes']
proof=dict(status='ROOT_FAILURE_CHUNK002_ARCHIVE_MEMBER_OFFSERVER_PASS_NOT_ACCEPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    archive_sha256=sha(archive),inventory_sha256=sha(OUT/'failure_remote_archive_inventory.json'),member_n=len(members),
    accepted_new_n=0,completed_partial_unregistered=2,failure_before_CNN=True,
    observed_failure='Protected main800 health failed in resource_preflight',instant_main_queue_snapshot_missing=True,
    unique_turnover_cause_proved=False,numerical_mismatch_observed=False,new_CNN_inference=0,test=False,old_models_repacked=0)
with (OUT/'ROOT_OFFSERVER_FAILURE_VERIFICATION.json').open('x',encoding='utf8',newline='\n') as stream:
    json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof))
