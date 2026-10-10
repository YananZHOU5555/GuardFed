"""ROOT-only completed C_after50 exact6 transfer; original backup/verifier, no CNN."""
from pathlib import Path
import argparse,datetime,hashlib,json,shlex,subprocess,sys
sys.dont_write_bytecode=True
if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden for root operation guards')
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp/celeba_mechanism_valid_C_after50_20261010';EX=BASE/'execution_candidate'
REMOTE='/workspace/guardfed_checks/celeba_mechanism_valid_C_after50_20261010/execution_candidate'
SCIENCE='6e79e65b34a1ff78628884993aa6089fdc8c5d07951ed97ce016ad9b3ca6aaf0'
EXECUTION='073bfde67b2286f6e29fe5f6e2c1f7580f74465b4b0a4e42c583ab0332aa6595'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def check_terminal(live,expected):
    assert 'EXITED' in live['service'] and not live['processes'] and not live['batch_failure']
    assert live['batch_complete'] and len(live['completed'])==6
    ids=[row['id'] for row in live['completed']]
    assert len(ids)==len(set(ids))==len(expected)==6 and set(ids)==set(expected)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--startup-sha256',required=True)
    parser.add_argument('--deployment-sha256',required=True)
    parser.add_argument('--terminal-progress',required=True,type=Path)
    parser.add_argument('--terminal-progress-sha256',required=True)
    args=parser.parse_args()
    assert not sys.flags.optimize and sha(BASE/'FILES_SHA256.json')==SCIENCE
    assert sha(EX/'EXECUTION_SOURCE_SHA256.json')==EXECUTION
    assert sha(BASE/'inventory_actual156_Full100refs.json')=='2131f2386cb3a851f990d6ed4baa38f5ea90c9acadcd62fad50d24bf97277bcd'
    assert sha(EX/'ROOT_STARTUP_OBSERVATION.json')==args.startup_sha256
    assert sha(EX/'deployment_receipt.json')==args.deployment_sha256
    for folder,seal in ((BASE,BASE/'FILES_SHA256.json'),(EX,EX/'EXECUTION_SOURCE_SHA256.json')):
        for row in read(seal)['members']:assert sha(folder/row['path'])==row['sha256']
    assert not (EX/'BACKUP_LATEST.json').exists(), 'Already backed up; inspect existing evidence'
    assert not (EX/'ROOT_BACKUP_ATTEMPT.json').exists(), 'Prior attempt preserved; no blind retry'
    expected=read(BASE/'SCOPE.json')['selected_ids'];assert len(expected)==len(set(expected))==6
    latest=args.terminal_progress.resolve()
    assert latest.parent==EX.resolve() and latest.name.startswith('ROOT_PROGRESS_') and not latest.name.endswith('.RAW.json')
    assert sha(latest)==args.terminal_progress_sha256
    live=read(latest);check_terminal(live,expected)
    deployment=read(EX/'deployment_receipt.json')
    assert sha(EX/'ROOT_APPROVED.json')==deployment['root_approval_sha256']
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
assert len(complete)==len(set(complete))==6 and set(complete)==set(%r)
assert not (H/'BACKUP_LATEST.json').exists()
result=subprocess.run(['taskset','-c','110','ionice','-c','3','nice','-n','10',
 '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(H/'backup_completed.py')],check=True,capture_output=True,text=True)
print(result.stdout,end='')
"""%(REMOTE,EXECUTION,deployment['root_approval_sha256'],SCIENCE,live['batch_complete'],expected)
    with (EX/'ROOT_BACKUP_ATTEMPT.json').open('x',encoding='utf-8') as f:
        json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal_progress=str(latest),
            terminal_progress_sha256=sha(latest),execution_seal_sha256=EXECUTION,selected_ids=expected,
            startup_sha256=args.startup_sha256,deployment_sha256=args.deployment_sha256),f,indent=2);f.write('\n')
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
