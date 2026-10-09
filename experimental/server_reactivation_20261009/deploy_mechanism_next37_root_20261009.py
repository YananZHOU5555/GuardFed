"""Deploy only independently reviewed next37 source and explicit root authority once."""
from pathlib import Path, PurePosixPath
import argparse,datetime,hashlib,json,shlex,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[1]
HOST='root@89.22.197.55'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(p,value):
    with p.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
parser=argparse.ArgumentParser();parser.add_argument('--execution-seal',required=True)
parser.add_argument('--scope',type=int,choices=(37,11),default=37);args=parser.parse_args()
BASE=ROOT/f'tmp/celeba_mechanism_valid_incremental_next{args.scope}_20261009';EX=BASE/'execution_candidate'
REMOTE=f'/workspace/guardfed_checks/celeba_mechanism_valid_incremental_next{args.scope}_20261009'
science_sha={37:'95978fa42c28e9b4ff5b855b33c2dda56edc2b14fcfd56c3a29b0a9ba98135fd',11:'65f706e8c7c7d7e18e76c8a300dd845297c97b3bd5c6c182bbbb8aad0103b5ff'}[args.scope]
source_review_sha={37:'1477fdfeb7709f459993b112045c3d5d3a5c0b16e87660ab5b74382090ce0cdb',11:'b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14'}[args.scope]
assert sha(BASE/'FILES_SHA256.json')==science_sha
assert sha(EX/'EXECUTION_SOURCE_SHA256.json')==args.execution_seal
review=read(EX/'ROOT_EXECUTION_REVIEW.json')
assert review['status']==f'ROOT_NEXT{args.scope}_EXECUTION_SOURCE_REVIEW_PASS_NOT_DISPATCHED'
assert review['execution_seal_sha256']==args.execution_seal and review['science_seal_sha256']==sha(BASE/'FILES_SHA256.json')
assert not review['CNN_executed'] and review['scientific_functions_unchanged']
assert sha(BASE/'root_source_review/ROOT_REVIEW.json')==source_review_sha
members={}
for folder,seal,relative in ((BASE,'FILES_SHA256.json',Path('.')),(EX,'EXECUTION_SOURCE_SHA256.json',Path('execution_candidate'))):
    for row in read(folder/seal)['members']:
        assert sha(folder/row['path'])==row['sha256'] and (folder/row['path']).stat().st_size==row['size']
        members[(relative/row['path']).as_posix()]=folder/row['path']
    members[(relative/seal).as_posix()]=folder/seal
authority=read(EX/'ROOT_REVIEW_TEMPLATE.json')
authority.update(status=f'ROOT_REVIEW_PASS_BOUNDED_NEXT{args.scope}_VALID_REPLAY',execution_authorized_within_existing_user_request=True,
    reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_seal_sha256=args.execution_seal,
    scientific_source_review_sha256=sha(BASE/'root_source_review/ROOT_REVIEW.json'),execution_review_sha256=sha(EX/'ROOT_EXECUTION_REVIEW.json'),
    authorization='Existing user request to finish rebuttal experiments; unchanged accepted-terminal valid-only evaluation, no new recipe or endpoint.')
save(EX/'ROOT_APPROVED.json',authority)
draft=read(EX/'APPROVED_TEMPLATE.json')
draft.update(status=f'APPROVED_NEXT{args.scope}_MECHANISM_VALID_REPLAY_ONLY',root_approval_sha256=sha(EX/'ROOT_APPROVED.json'),
    execution_seal_sha256=args.execution_seal)
save(EX/'EXECUTION_DRAFT.json',draft)
for name in ('ROOT_APPROVED.json','EXECUTION_DRAFT.json','ROOT_EXECUTION_REVIEW.json'):
    members['execution_candidate/'+name]=EX/name
archive=EX/'root_deployment_source.tar.gz'
with tarfile.open(archive,'x:gz') as bundle:
    for name,p in sorted(members.items()):
        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
        bundle.add(p,arcname=BASE.name+'/'+name,recursive=False)
archive_sha=sha(archive)
remote_archive=f'/workspace/guardfed_checks/next{args.scope}_source_root_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.tar.gz'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350',HOST]
preflight="""from pathlib import Path
import hashlib,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert not Path(%r).exists()
s=subprocess.run(['supervisorctl','status','sglang'],capture_output=True,text=True).stdout.strip();assert 'STOPPED' in s,s
s=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_valid_incremental15'],capture_output=True,text=True).stdout.strip();assert 'EXITED' in s,s
print('FRESH_NEXT37_NAMESPACE_SGLANG_STOPPED_PRIOR15_EXITED')
"""%REMOTE
if args.scope==11:
    preflight=preflight.replace("print('FRESH_NEXT37_NAMESPACE_SGLANG_STOPPED_PRIOR15_EXITED')", "s=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_valid_next37'],capture_output=True,text=True).stdout.strip();assert 'EXITED' in s,s\nprint('FRESH_NEXT11_NAMESPACE_SGLANG_STOPPED_PRIOR15_AND37_EXITED')")
pre=subprocess.run(ssh+['python -c '+shlex.quote(preflight)],capture_output=True,check=True,timeout=30)
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350',str(archive),HOST+':'+remote_archive],check=True,timeout=90)
code="""from pathlib import Path,PurePosixPath
import hashlib,json,subprocess,tarfile
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
archive=Path(%r);assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
target=Path(%r);assert not target.exists()
with tarfile.open(archive) as bundle:
    for item in bundle:
        rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]==%r
    bundle.extractall(target.parent,filter='data')
cmd=['ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',str(target/'execution_candidate/install_once.py'),'--draft-sha256',%r]
result=subprocess.run(cmd,capture_output=True,text=True)
print(json.dumps(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
"""%(remote_archive,archive_sha,REMOTE,BASE.name,sha(EX/'EXECUTION_DRAFT.json'))
result=subprocess.run(ssh+['python -c '+shlex.quote(code)],capture_output=True,check=True,timeout=180)
installed=json.loads(result.stdout)
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_archive_sha256=archive_sha,
    source_archive_members=len(members),source_archive_remote=remote_archive,science_seal_sha256=sha(BASE/'FILES_SHA256.json'),
    execution_seal_sha256=args.execution_seal,root_approval_sha256=sha(EX/'ROOT_APPROVED.json'),
    external_draft_sha256=sha(EX/'EXECUTION_DRAFT.json'),empty_namespace_prerequisite=pre.stdout.decode().strip(),
    remote_installation=installed,root_remote_stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
    new_training=0,new_Full_inference=0,test_inference=False)
save(EX/'deployment_receipt.json',receipt)
assert installed['returncode']==0,installed
print(json.dumps({'status':f'NEXT{args.scope}_INSTALLED_ACTUAL_STARTUP_OBSERVATION_PENDING','installation':installed,
                  'deployment_receipt_sha256':sha(EX/'deployment_receipt.json')}))
