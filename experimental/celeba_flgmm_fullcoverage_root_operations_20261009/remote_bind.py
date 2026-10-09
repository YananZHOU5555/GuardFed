"""One metadata binding and original four-record acceptance, never dispatch."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,tarfile,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
BASE=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009')
SOURCE=BASE/'source_prepared';STAGE=BASE/'stage';OLD=Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2')
REPO=Path('/workspace/GuardFed-celeba-expanded')
PYTHON='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--cpu',type=int,required=True);a=p.parse_args()
 assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
 assert set(os.sched_getaffinity(0))=={a.cpu} and os.getpriority(os.PRIO_PROCESS,0)>=10
 assert 'idle' in subprocess.check_output(['ionice','-p',str(os.getpid())],text=True)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert all(os.environ.get(k)=='1' for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
 assert not STAGE.exists() and not (BASE/'BOUND_IDENTITY.json').exists() and not list(BASE.glob('BIND_FAILURE*.json'))
 inputs=read(BASE/'TRANSFER_MEMBERS.json')['members']
 for name,pin in inputs.items():assert sha(BASE/name)==pin['sha256'] and (BASE/name).stat().st_size==pin['bytes']
 review=read(BASE/'inputs/SOURCE_REVIEW.json');approval=read(BASE/'inputs/BIND_APPROVAL.json')
 assert review['v2_source_seal_sha256']==sha(SOURCE/'FILES_SHA256.json')==approval['source_seal_sha256']
 assert sha(BASE/'inputs/SOURCE_REVIEW.json')==approval['independent_source_review_sha256']
 assert review['status']=='PASS_LIMITED_V2_SOURCE_REVIEW_READY_FOR_ROOT_BINDING_AND_REAL_CANARIES' and not review['new_blocking_findings']
 command=[PYTHON,'-B',str(SOURCE/'bind_stage.py'),'--old-release',str(OLD),
  '--summary',str(BASE/'inputs/SUMMARY32.json'),'--summary-sha256',sha(BASE/'inputs/SUMMARY32.json'),
  '--final-root-proof',str(BASE/'inputs/FINAL32_ROOT.json'),'--final-root-proof-sha256',sha(BASE/'inputs/FINAL32_ROOT.json'),
  '--adoption',str(BASE/'inputs/BIND_APPROVAL.json'),'--adoption-sha256',sha(BASE/'inputs/BIND_APPROVAL.json'),
  '--source-seal-sha256',sha(SOURCE/'FILES_SHA256.json'),'--output',str(STAGE)]
 result=subprocess.run(command,capture_output=True,text=True)
 save(BASE/'BIND_COMMAND.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
 result.check_returncode();bound=json.loads(result.stdout)
 assert (bound['new'],bound['reused'],bound['gates'],bound['execute_authorized'])==(96,4,5,False)
 sys.path.insert(0,str(STAGE))
 from screen_common import local_identity,repo_identity
 protocol,manifest=local_identity();before=repo_identity(REPO,protocol)
 from run_fullcoverage import reused_records
 rows=reused_records(manifest);assert len(rows)==4
 assert repo_identity(REPO,protocol)==before;local_identity()
 assert not any((STAGE/n).exists() for n in ('EXECUTION_AUTHORIZATION.json','GATE_ACCEPTANCE.json','runs','preflight','queue_progress.json'))
 mapping=read(STAGE/'REPO_SYMLINK_TARGETS.json')['entries']
 targets={n:dict(resolved=str((REPO/n).resolve()),sha256=sha(REPO/n)) for n in mapping}
 assert len(targets)==6 and all(targets[n]==dict(resolved=mapping[n]['resolved_target'],sha256=mapping[n]['sha256']) for n in mapping)
 save(BASE/'BOUND_IDENTITY.json',dict(status='BOUND_METADATA_AND_ORIGINAL4_CHECKED_NO_EXECUTION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  package_sha256=sha(STAGE/'PACKAGE_SHA256.json'),manifest_sha256=sha(STAGE/'manifest.json'),protocol_sha256=sha(STAGE/'source/protocol.json'),
  source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'),source_review_sha256=sha(BASE/'inputs/SOURCE_REVIEW.json'),
  source_data_before=before,source_data_after=before,exact_six_targets=targets,strict_reused_records=rows,
  planned_new=96,reused=4,planned_total=100,prepared_new_canaries=5,run_canaries=0,new_training=0,test=False,services_changed=False,helper_cpu=a.cpu))
 names=['BOUND_IDENTITY.json','BIND_COMMAND.json','TRANSFER_MEMBERS.json']
 names += [p.relative_to(BASE).as_posix() for p in (BASE/'inputs').iterdir() if p.is_file()]
 names += [p.relative_to(BASE).as_posix() for p in STAGE.rglob('*') if p.is_file()]
 members={n:dict(sha256=sha(BASE/n),bytes=(BASE/n).stat().st_size) for n in names}
 save(BASE/'BOUND_MEMBERS.json',dict(members=members))
 archive=BASE/'bound_metadata.tar.gz'
 with tarfile.open(archive,'x:gz') as t:
  for n in names+['BOUND_MEMBERS.json']:t.add(BASE/n,arcname=n,recursive=False)
 with tarfile.open(archive) as t:
  assert len(t.getnames())==len(set(t.getnames()))==len(members)+1
  for item in t:
   b=t.extractfile(item).read();pin=members.get(item.name,dict(sha256=sha(BASE/'BOUND_MEMBERS.json'),bytes=(BASE/'BOUND_MEMBERS.json').stat().st_size))
   assert hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes']
 receipt=dict(status='BOUND_METADATA_ARCHIVED_NO_EXECUTION',archive_sha256=sha(archive),inventory_sha256=sha(BASE/'BOUND_MEMBERS.json'),
  archive_bytes=archive.stat().st_size,members=len(members)+1,package_sha256=bound['package_sha256'],no_checkpoint_repacked=True)
 save(BASE/'BOUND_BACKUP.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  failure=BASE/('BIND_FAILURE_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
  if BASE.is_dir():save(failure,dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
  raise
