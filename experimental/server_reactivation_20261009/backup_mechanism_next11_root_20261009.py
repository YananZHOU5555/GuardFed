"""Transfer the completed frozen eleven once, using the original backup/verifier."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate'
REMOTE='/workspace/guardfed_checks/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(EX/'EXECUTION_SOURCE_SHA256.json')=='9f1252dd7c11abfe7cee297b2028ca9c58f179d964efb39ee6008ea860508b15'
assert not (EX/'BACKUP_LATEST.json').exists(), 'Already backed up; inspect the existing receipt instead of retrying'
progress=sorted(EX.glob('ROOT_PROGRESS_*.json'))
progress=[p for p in progress if not p.name.endswith('.RAW.json')]
latest=progress[-1];live=read(latest)
assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
assert len(live['completed'])==11 and live['batch_complete']
deployment=read(EX/'deployment_receipt.json')
code="""
from pathlib import Path
import hashlib,json,subprocess
H=Path(%r)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(H/'EXECUTION_SOURCE_SHA256.json')==%r and sha(H/'ROOT_APPROVED.json')==%r
for row in read(H/'EXECUTION_SOURCE_SHA256.json')['members']:assert sha(H/row['path'])==row['sha256']
assert (H/'batch_complete.json').is_file() and not (H/'batch_failure.json').exists()
assert len(list(H.glob('completed_*.json')))==11 and not (H/'BACKUP_LATEST.json').exists()
result=subprocess.run(['taskset','-c','107','ionice','-c','3','nice','-n','10',
 '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(H/'backup_completed.py')],check=True,capture_output=True,text=True)
print(result.stdout,end='')
"""%(REMOTE,sha(EX/'EXECUTION_SOURCE_SHA256.json'),deployment['root_approval_sha256'])
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',
 'python -c '+shlex.quote(code)],capture_output=True,check=True,timeout=120)
with (EX/'ROOT_BACKUP_COMMAND_STDOUT.json').open('xb') as stream:stream.write(result.stdout)
receipt=json.loads(result.stdout)
assert len(receipt['accepted_new_ids'])==len(receipt['all_accepted_ids'])==11
assert receipt['previous_backup_receipt_sha256'] is None
remote_archive=receipt['archive'];stamp=Path(remote_archive).parent.name
dest=EX/'backups'/stamp;dest.mkdir(parents=True,exist_ok=False)
for name in ('incremental_valid_three_views.tar.gz','backup_receipt.json'):
 subprocess.run(['scp','-q','-P','60350','root@89.22.197.55:'+str(Path(remote_archive).parent.as_posix())+'/'+name,
  str(dest/name)],check=True,timeout=180)
assert sha(dest/'incremental_valid_three_views.tar.gz')==receipt['archive_sha256']
assert read(dest/'backup_receipt.json')==receipt
verified=subprocess.run([sys.executable,'-B',str(EX/'verify_backup.py'),str(dest)],capture_output=True,check=True,timeout=120)
with (dest/'ROOT_LOCAL_VERIFY_STDOUT.json').open('xb') as stream:stream.write(verified.stdout)
print(json.dumps({'delta':stamp,'receipt_sha256':sha(dest/'backup_receipt.json'),
 'offserver_proof_sha256':sha(dest/'OFFSERVER_VERIFICATION.json'),'verification':json.loads(verified.stdout),
 'terminal_progress_sha256':sha(latest)}))
