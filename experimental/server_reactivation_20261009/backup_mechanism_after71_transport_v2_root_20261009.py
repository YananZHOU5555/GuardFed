"""ROOT-only after71 transport-v2; fixed short SSH argv, preserved old failure, no CNN."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_valid_incremental_after71_20261009';EX=BASE/'execution_candidate'
REMOTE='/workspace/guardfed_checks/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate'
SCIENCE='d05a0b81620d1791858a9f71405252443e593600f390359175ff961dde0e2bec'
EXECUTION='0fe232aab75a870b7840fd6ef0bba85b3d12c541ca49a0c0976f3f8a7095352f'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def check_terminal(live,expected):
    assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
    assert live['batch_complete'] and len(live['completed'])==11
    ids=[row['id'] for row in live['completed']]
    assert len(ids)==len(set(ids))==len(expected)==11 and set(ids)==set(expected)

def recovery_review():
    old=ROOT/'tmp/backup_mechanism_after71_root_20261009.py'
    assert sha(old)=='09f4f9d015dc2a745edc8a82f54043130928fe730a42b226f9da6f8ffd64e6dd'
    assert sha(EX/'ROOT_BACKUP_ATTEMPT.json')=='cd01f77444fd21c40e6565dfc026e83593b88ca5d94d0459506e3287d2041058'
    path=EX/'ROOT_TRANSPORT_RECOVERY_REVIEW.json'
    assert sha(path)=='7c9cf002a689d8d418cfa5694e1d5751e4ffdec6d158050380cce297f7cadefe'
    review=read(path)
    assert review['status']=='ROOT_WINDOWS_ARGV206_PRE_SSH_FAILURE_AND_EMPTY_REMOTE_BACKUP_CONFIRMED'
    assert review['old_helper_sha256']==sha(old) and review['old_attempt_sha256']==sha(EX/'ROOT_BACKUP_ATTEMPT.json')
    assert review['old_command_result_exists'] is review['old_command_stdout_exists'] is False
    remote=review['remote_readonly']
    assert remote['backup_latest_exists'] is remote['batch_failure_exists'] is False
    assert remote['backup_directories']==[] and remote['completed']==11
    assert remote['batch_complete_sha256']=='1a66d99505715ae501b9fb080c22f026fcd910436a36b0b986bede068329770a'
    assert review['new_inference']==0 and review['automatic_retry'] is False
    return review

def main():
    recovery=recovery_review()
    assert not sys.flags.optimize and sha(BASE/'FILES_SHA256.json')==SCIENCE
    assert sha(EX/'EXECUTION_SOURCE_SHA256.json')==EXECUTION
    assert sha(BASE/'inventory_actual82_Full100refs.json')=='72aca76f626580faf465c61f35b2274068e76782eb1bcef543e98da23c2e6d1e'
    assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')=='a95018b7ccc62472d7b1f796e7a2d1307a72bb4bd4905397c9f623ca6a4d2f60'
    for folder,seal in ((BASE,BASE/'FILES_SHA256.json'),(EX,EX/'EXECUTION_SOURCE_SHA256.json')):
        for row in read(seal)['members']:assert sha(folder/row['path'])==row['sha256']
    assert not (EX/'BACKUP_LATEST.json').exists(), 'Already backed up; inspect existing evidence'
    assert not (EX/'ROOT_BACKUP_TRANSPORT_ATTEMPT.json').exists(), 'Transport attempt preserved; no blind retry'
    assert not (EX/'ROOT_BACKUP_COMMAND_RESULT.json').exists() and not (EX/'ROOT_BACKUP_COMMAND_STDOUT.json').exists(), 'Old failure state changed'
    expected=read(BASE/'SCOPE.json')['selected_ids'];assert len(expected)==len(set(expected))==11
    progress=[p for p in EX.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    assert progress, 'Actual terminal observation is required; startup is insufficient'
    latest=max(progress);live=read(latest);check_terminal(live,expected)
    assert read(EX/'ROOT_BACKUP_ATTEMPT.json')['terminal_progress_sha256']==sha(latest)=='4751c834ec25225ba531dc4ba6059baf85f55b85ceee225100091de7509bc4a4'
    deployment=read(EX/'deployment_receipt.json')
    assert sha(EX/'ROOT_APPROVED.json')==deployment['root_approval_sha256']=='0438a11351de81103f52983dd86b5eee5f5ecbabdef4585dbb44d01602af838d'
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
assert sha(H/'batch_complete.json')==%r
assert read(H/'batch_complete.json')==%r
complete=[read(p)['id'] for p in H.glob('completed_*.json')]
assert len(complete)==len(set(complete))==11 and set(complete)==set(%r)
assert not (H/'BACKUP_LATEST.json').exists()
assert not (H/'backups').exists() or not any((H/'backups').iterdir())
result=subprocess.run(['taskset','-c','107','ionice','-c','3','nice','-n','10',
 '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(H/'backup_completed.py')],check=True,capture_output=True,text=True)
print(result.stdout,end='')
"""%(REMOTE,EXECUTION,deployment['root_approval_sha256'],SCIENCE,recovery['remote_readonly']['batch_complete_sha256'],live['batch_complete'],expected)
    with (EX/'ROOT_BACKUP_TRANSPORT_ATTEMPT.json').open('x',encoding='utf-8') as f:
        json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal_progress=str(latest),
            terminal_progress_sha256=sha(latest),execution_seal_sha256=EXECUTION,selected_ids=expected,
            recovery_review_sha256=sha(EX/'ROOT_TRANSPORT_RECOVERY_REVIEW.json'),
            original_attempt_sha256=sha(EX/'ROOT_BACKUP_ATTEMPT.json'),old_helper_sha256=recovery['old_helper_sha256'],
            transport='fixed short ssh argv; complete script via stdin'),f,indent=2);f.write('\n')
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',
        'python -B -'],input=code.encode(),capture_output=True,timeout=120)
    with (EX/'ROOT_BACKUP_TRANSPORT_COMMAND_RESULT.json').open('x',encoding='utf-8') as f:
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
        terminal_progress_sha256=sha(latest),recovery_review_sha256=sha(EX/'ROOT_TRANSPORT_RECOVERY_REVIEW.json'),
        transport_attempt_sha256=sha(EX/'ROOT_BACKUP_TRANSPORT_ATTEMPT.json'))))

if __name__=='__main__':main()
