"""ROOT-only completed after82_v2 exact10 transfer; original backup/verifier, no CNN."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009';EX=BASE/'execution_candidate'
REMOTE='/workspace/guardfed_checks/celeba_mechanism_valid_incremental_after82_v2_20261009/execution_candidate'
SCIENCE='b95801ac039bb39276e79012393792adfde290b92091ff45c0a4323bd4fdd8f0'
EXECUTION='94d99842b334346ae8a6715b84f7fddea35574a77fbf51f9460b25301c6ccae6'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def check_terminal(live,expected):
    assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
    assert live['batch_complete'] and len(live['completed'])==10
    ids=[row['id'] for row in live['completed']]
    assert len(ids)==len(set(ids))==len(expected)==10 and set(ids)==set(expected)

def main():
    assert not sys.flags.optimize and sha(BASE/'FILES_SHA256.json')==SCIENCE
    assert sha(EX/'EXECUTION_SOURCE_SHA256.json')==EXECUTION
    assert sha(BASE/'inventory_actual92_Full100refs.json')=='bedb867ae9bdf965ef7143a5b4f077620911ec26d93346337307046e64b5c9da'
    assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')=='a7fe2e43cadd5f148055720cb9fc00ea5a82e5f133611225187bc07419f0229b'
    for folder,seal in ((BASE,BASE/'FILES_SHA256.json'),(EX,EX/'EXECUTION_SOURCE_SHA256.json')):
        for row in read(seal)['members']:assert sha(folder/row['path'])==row['sha256']
    assert not (EX/'BACKUP_LATEST.json').exists(), 'Already backed up; inspect existing evidence'
    assert not (EX/'ROOT_BACKUP_ATTEMPT.json').exists(), 'Prior attempt preserved; no blind retry'
    expected=read(BASE/'SCOPE.json')['selected_ids'];assert len(expected)==len(set(expected))==10
    progress=[p for p in EX.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    assert progress, 'Actual terminal observation is required; startup is insufficient'
    latest=max(progress);live=read(latest);check_terminal(live,expected)
    deployment=read(EX/'deployment_receipt.json')
    assert sha(EX/'ROOT_APPROVED.json')==deployment['root_approval_sha256']=='34e5d9ec34a1ddad4bda4e49f746ffeb59545e064311b4b05e8d6d082731e5a0'
    code="""
from pathlib import Path
import hashlib,json,subprocess
H=Path(%r)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(H/'EXECUTION_SOURCE_SHA256.json')==%r and sha(H/'ROOT_APPROVED.json')==%r
assert sha(H.parent/'FILES_SHA256.json')==%r
for folder,seal in ((H,H/'EXECUTION_SOURCE_SHA256.json'),(H.parent,H.parent/'FILES_SHA256.json')):
 for row in read(seal)['members']:assert sha(folder/row['path'])==row['sha256']
assert (H/'batch_complete.json').is_file() and not (H/'batch_failure.json').exists()
assert read(H/'batch_complete.json')==%r
complete=[read(p)['id'] for p in H.glob('completed_*.json')]
assert len(complete)==len(set(complete))==10 and set(complete)==set(%r)
assert not (H/'BACKUP_LATEST.json').exists()
result=subprocess.run(['taskset','-c','107','ionice','-c','3','nice','-n','10',
 '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(H/'backup_completed.py')],check=True,capture_output=True,text=True)
print(result.stdout,end='')
"""%(REMOTE,EXECUTION,deployment['root_approval_sha256'],SCIENCE,live['batch_complete'],expected)
    with (EX/'ROOT_BACKUP_ATTEMPT.json').open('x',encoding='utf-8') as f:
        json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal_progress=str(latest),
            terminal_progress_sha256=sha(latest),execution_seal_sha256=EXECUTION,selected_ids=expected),f,indent=2);f.write('\n')
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',
        'python -B -'],input=code.encode(),capture_output=True,timeout=120)
    with (EX/'ROOT_BACKUP_COMMAND_RESULT.json').open('x',encoding='utf-8') as f:
        json.dump(dict(returncode=result.returncode,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace')),f,indent=2);f.write('\n')
    result.check_returncode()
    with (EX/'ROOT_BACKUP_COMMAND_STDOUT.json').open('xb') as stream:stream.write(result.stdout)
    receipt=json.loads(result.stdout)
    assert receipt['accepted_new_ids']==receipt['all_accepted_ids']==expected
    assert receipt['previous_backup_receipt_sha256'] is None
    remote_archive=receipt['archive'];stamp=Path(remote_archive).parent.name
    assert Path(remote_archive).parent.parent.as_posix()==REMOTE+'/backups'
    dest=EX/'backups'/stamp;dest.mkdir(parents=True,exist_ok=False)
    for name in ('incremental_valid_three_views.tar.gz','backup_receipt.json'):
        subprocess.run(['scp','-q','-P','60350','root@89.22.197.55:'+str(Path(remote_archive).parent.as_posix())+'/'+name,
            str(dest/name)],check=True,timeout=180)
    assert sha(dest/'incremental_valid_three_views.tar.gz')==receipt['archive_sha256']
    assert read(dest/'backup_receipt.json')==receipt
    verified=subprocess.run([sys.executable,'-B',str(EX/'verify_backup.py'),str(dest)],capture_output=True,timeout=120)
    with (dest/'ROOT_LOCAL_VERIFY_COMMAND_RESULT.json').open('x',encoding='utf-8') as f:
        json.dump(dict(returncode=verified.returncode,stdout=verified.stdout.decode(errors='replace'),stderr=verified.stderr.decode(errors='replace')),f,indent=2);f.write('\n')
    verified.check_returncode()
    with (dest/'ROOT_LOCAL_VERIFY_STDOUT.json').open('xb') as stream:stream.write(verified.stdout)
    print(json.dumps(dict(delta=str(dest),receipt_sha256=sha(dest/'backup_receipt.json'),
        offserver_proof_sha256=sha(dest/'OFFSERVER_VERIFICATION.json'),verification=json.loads(verified.stdout),
        terminal_progress_sha256=sha(latest))))

if __name__=='__main__':main()
