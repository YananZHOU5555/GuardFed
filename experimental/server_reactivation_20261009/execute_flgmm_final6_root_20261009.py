"""One root-authorized final-six strict backup; requires an actual complete32 snapshot."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,importlib.util,json,subprocess,sys,tarfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_flgmm_final6_closure_20261009'
PARENT=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True,type=Path);parser.add_argument('--snapshot-sha256',required=True);args=parser.parse_args()
 assert not sys.flags.optimize
 assert sha(BASE/'FILES_SHA256.json')=='1d7cc11f95727a57478dd8575f65170024b36df98187030ac2de05e1e08cf6b9'
 for name,pin in read(BASE/'FILES_SHA256.json')['files'].items():
  assert sha(BASE/name)==pin['sha256'] and (BASE/name).stat().st_size==pin['bytes']
 assert sha(args.snapshot)==args.snapshot_sha256
 spec=importlib.util.spec_from_file_location('guard_final6',BASE/'closure_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
 guard.validate_snapshot(read(args.snapshot),read(BASE/'manifest.json'))
 assert not read(args.snapshot)['recent_log_error_matches'] and not read(args.snapshot)['restricted_CPU106_Python']
 assert sha(PARENT/'LATEST_BACKUP.json')==sha(BASE/'PREVIOUS_LATEST.json')
 chain=PARENT/read(PARENT/'LATEST_BACKUP.json')['chain_file'];assert sha(chain)==guard.EXPECTED_CHAIN
 assert guard.validate_previous(read(chain),sha(chain),guard.EXPECTED_PACKAGE,read(BASE/'manifest.json'))==read(BASE/'EXACT_DELTA.json')['selected_ids']
 ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');attempt=BASE/('actual_'+stamp)
 remote='/workspace/guardfed_checks/celeba_flgmm_screen_20261009/final6_closure_20261009/'+attempt.name
 pre="from pathlib import Path\nimport hashlib\nassert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\np=Path(%r)\nassert p.parent==Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/final6_closure_20261009')\nassert not p.exists() and not Path(str(p)+'.tar.gz').exists() and not p.parent.is_symlink()\np.parent.mkdir(exist_ok=True)\nassert p.parent.resolve()==p.parent\n"%remote
 subprocess.run(ssh+['python -B -'],input=pre.encode(),check=True,capture_output=True,timeout=30)
 attempt.mkdir(exist_ok=False)
 names=['collect_delta.py','verify_delta_offserver.py','closure_guard.py','PREVIOUS_CHAIN.json','PREVIOUS_LATEST.json','EXACT_DELTA.json','COLLECTOR_DIFF.patch','SOURCE_RECEIPT.json']
 for name in names:
  (attempt/name).write_bytes((BASE/name).read_bytes());assert sha(attempt/name)==sha(BASE/name)
 (attempt/'AUTHORIZED_SNAPSHOT.json').write_bytes(args.snapshot.read_bytes())
 binding=read(BASE/'EXECUTION_BINDINGS_TEMPLATE.json');binding.update(status='ROOT_AUTHORIZED_FIXED_FINAL6_CLOSURE',authorized_snapshot_sha256=args.snapshot_sha256,
  root_authorized_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prepared_seal_sha256=sha(BASE/'FILES_SHA256.json'),
  authorization='Existing user full rebuttal experiment request; exact final6 terminal acceptance only, no training/test')
 save(attempt/'EXECUTION_BINDINGS.json',binding)
 names+=['AUTHORIZED_SNAPSHOT.json','EXECUTION_BINDINGS.json']
 archive=attempt/'root_transfer.tar.gz'
 with tarfile.open(archive,'x:gz') as tar:
  for name in names:tar.add(attempt/name,arcname=name,recursive=False)
 remote_archive=remote+'.tar.gz'
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350',str(archive),'root@89.22.197.55:'+remote_archive],check=True,timeout=90)
 code="""from pathlib import Path,PurePosixPath
import hashlib,json,os,subprocess,tarfile
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()==%r
archive=Path(%r);assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
target=Path(%r);assert not target.exists();target.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive) as bundle:
 assert set(bundle.getnames())==set(%r)
 for item in bundle:
  rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
 bundle.extractall(target,filter='data')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',GUARDFED_CPU_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
cmd=['taskset','-c','106','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(target/'collect_delta.py')]
result=subprocess.run(cmd,env=env,capture_output=True,text=True)
print(json.dumps(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
"""%('42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa',remote_archive,sha(archive),remote,names)
 save(attempt/'ROOT_EXECUTION_ATTEMPT.json',dict(utc=binding['root_authorized_utc'],remote=remote,source_archive_sha256=sha(archive),snapshot_sha256=args.snapshot_sha256,
  previous_chain_sha256=sha(chain),selected_ids=read(BASE/'EXACT_DELTA.json')['selected_ids'],automatic_retry=False))
 try:result=subprocess.run(ssh+['python -B -'],input=code.encode(),capture_output=True,timeout=240)
 except subprocess.TimeoutExpired as error:
  save(attempt/'ROOT_TRANSPORT_TIMEOUT.json',dict(error=str(error),remote=remote,automatic_retry=False,
   stdout=(error.stdout or b'').decode(errors='replace'),stderr=(error.stderr or b'').decode(errors='replace'),
   next='Inspect this exact remote attempt and producer before any transfer or execution recovery; remote collector may still be active'))
  raise
 save(attempt/'ROOT_EXECUTION_COMMAND.json',dict(returncode=result.returncode,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace')))
 result.check_returncode();installed=json.loads(result.stdout);assert installed['returncode']==0,installed
 receipt=json.loads(installed['stdout']);assert receipt['accepted_new']==6 and receipt['accepted_total']==32
 for name in ['accepted_final6_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json']:
  subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+remote+'/'+name,str(attempt/name)],check=True,timeout=180)
 assert read(attempt/'BACKUP_SHA256.json')==receipt and sha(attempt/'accepted_final6_delta.tar.gz')==receipt['archive_sha256']
 verified=subprocess.run([sys.executable,'-B',str(attempt/'verify_delta_offserver.py')],capture_output=True,timeout=180)
 save(attempt/'ROOT_LOCAL_VERIFY_COMMAND.json',dict(returncode=verified.returncode,stdout=verified.stdout.decode(errors='replace'),stderr=verified.stderr.decode(errors='replace')))
 verified.check_returncode()
 print(json.dumps(dict(directory=str(attempt),receipt_sha256=sha(attempt/'BACKUP_SHA256.json'),offserver_proof_sha256=sha(attempt/'OFFSERVER_ACCEPTANCE.json'),new_accepted_pending_root_adoption=6)))
if __name__=='__main__':main()
