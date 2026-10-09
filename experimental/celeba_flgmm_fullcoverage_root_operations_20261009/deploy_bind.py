"""Explicit one-shot root operation: upload reviewed source, bind metadata, return evidence. No dispatch."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,os,subprocess,sys,tarfile,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'tmp/celeba_flgmm_fullcoverage_source_v2_20261009'
FINAL=ROOT/'tmp/celeba_flgmm_final6_closure_20261009'
BASE='/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009'
PINS=dict(source='fc9cd4133345d74949d22cddca12868393b8ac6ac6920d2f1deb293611584c30',
 summary='d8f32441402f0ce7870bacc34e94de2a334133b2881d0a59ccf5123ef0781291',
 final_root='66097564f346b0dd5a0194ea7bd8c1413d51936819db086283f1d9a99a85738e',
 summary_adoption='e602761016e199da157862da3f24c9f9d0f191cfde10fad49074540c672b4a7f',
 review='a22118cfc7ba4f7cae592fa9131a596731c38c4d5a97f4efb2466f830b681e3b')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--source-review',type=Path,required=True);parser.add_argument('--source-review-sha256',required=True)
 parser.add_argument('--binding-approval',type=Path,default=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/BIND_APPROVAL.json');parser.add_argument('--binding-approval-sha256',required=True)
 parser.add_argument('--helper-seal-sha256',required=True);parser.add_argument('--cpu',type=int,required=True)
 a=parser.parse_args();assert a.cpu>=0
 assert sha(HERE/'FILES_SHA256.json')==a.helper_seal_sha256
 for n,v in read(HERE/'FILES_SHA256.json')['files'].items():assert sha(HERE/n)==v['sha256']
 assert sha(SOURCE/'FILES_SHA256.json')==PINS['source']
 for n,v in read(SOURCE/'FILES_SHA256.json')['files'].items():assert sha(SOURCE/n)==v['sha256'] and (SOURCE/n).stat().st_size==v['bytes']
 assert sha(a.source_review)==a.source_review_sha256==PINS['review']
 assert sha(a.binding_approval)==a.binding_approval_sha256
 sources={'SUMMARY32.json':FINAL/'summary32_final/SUMMARY32.json','FINAL32_ROOT.json':FINAL/'actual_20261009T194424Z/ROOT_ADOPTION_REVIEW.json',
          'SUMMARY_ADOPTION.json':FINAL/'ROOT_SUMMARY_ADOPTION.json','SOURCE_REVIEW.json':a.source_review,'BIND_APPROVAL.json':a.binding_approval}
 for n,k in [('SUMMARY32.json','summary'),('FINAL32_ROOT.json','final_root'),('SUMMARY_ADOPTION.json','summary_adoption')]:assert sha(sources[n])==PINS[k]
 review=read(a.source_review);approval=read(a.binding_approval);adoption=read(sources['SUMMARY_ADOPTION.json'])
 assert review['status']=='PASS_LIMITED_V2_SOURCE_REVIEW_READY_FOR_ROOT_BINDING_AND_REAL_CANARIES' and not review['new_blocking_findings']
 assert review['v2_source_seal_sha256']==PINS['source']
 assert approval['status']=='ROOT_APPROVED_FLGMM100_RECIPE_BINDING' and approval['scope']=='96_new_plus_4_reused_70round_valid_only'
 assert approval['source_seal_sha256']==PINS['source'] and approval['summary_sha256']==PINS['summary'] and approval['final_root_proof_sha256']==PINS['final_root']
 assert approval['independent_source_review_sha256']==a.source_review_sha256
 assert approval['summary_adoption_sha256']==PINS['summary_adoption']
 assert approval['selected_recipe']==adoption['selected_recipe'] and approval['output']==BASE+'/stage'
 assert approval['execute_authorized'] is False and approval['final_test'] is False
 assert approval['helper_seal_sha256']==a.helper_seal_sha256 and approval['helper_cpu']==a.cpu
 attempt=HERE/('attempt_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));attempt.mkdir()
 ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
 def command(label,argv,**kwargs):
  try:r=subprocess.run(argv,capture_output=True,**kwargs)
  except subprocess.TimeoutExpired as e:
   save(attempt/(label+'_TIMEOUT.json'),dict(error=repr(e),stdout=(e.stdout or b'').decode(errors='replace'),stderr=(e.stderr or b'').decode(errors='replace'),automatic_retry=False,remote=BASE,next='Inspect the exact namespace before any recovery; remote helper may still be active'))
   raise
  save(attempt/(label+'.json'),dict(returncode=r.returncode,stdout=r.stdout.decode(errors='replace'),stderr=r.stderr.decode(errors='replace')))
  r.check_returncode();return r
 try:
  files={'source_prepared/'+n:SOURCE/n for n in read(SOURCE/'FILES_SHA256.json')['files']}
  files['source_prepared/FILES_SHA256.json']=SOURCE/'FILES_SHA256.json'
  files.update({'inputs/'+n:path for n,path in sources.items()});files['operations/remote_bind.py']=HERE/'remote_bind.py'
  inventory={n:dict(sha256=sha(path),bytes=path.stat().st_size) for n,path in files.items()}
  save(attempt/'TRANSFER_MEMBERS.json',dict(members=inventory))
  archive=attempt/'transfer.tar.gz'
  with tarfile.open(archive,'x:gz') as t:
   for n,path in files.items():t.add(path,arcname=n,recursive=False)
   t.add(attempt/'TRANSFER_MEMBERS.json',arcname='TRANSFER_MEMBERS.json',recursive=False)
  pre="""from pathlib import Path
import hashlib,os
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
p=Path(%r);assert p.parent==Path('/workspace/guardfed_checks') and p.parent.resolve()==p.parent
assert not p.exists() and not p.is_symlink()
assert %r in os.sched_getaffinity(0)
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=proc.read_bytes().decode(errors='replace').split('\\0')
  if argv and 'python' in Path(argv[0]).name:
   cpus=os.sched_getaffinity(int(proc.parent.name));assert not (len(cpus)<=16 and %r in cpus),'Requested helper CPU already occupied'
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
p.mkdir();assert p.resolve()==p
"""%(BASE,a.cpu,a.cpu)
  command('REMOTE_PRECHECK',ssh+['python -B -'],input=pre.encode(),timeout=45)
  command('UPLOAD',['scp','-q','-o','BatchMode=yes','-P','60350',str(archive),'root@89.22.197.55:'+BASE+'/transfer.tar.gz'],timeout=90)
  script="""from pathlib import Path,PurePosixPath
import hashlib,json,os,subprocess,tarfile
base=Path(%r)
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
archive=base/'transfer.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
assert base.resolve()==base and not (base/'source_prepared').exists() and not (base/'stage').exists()
with tarfile.open(archive) as t:
 assert len(t.getnames())==len(set(t.getnames()))
 raw=t.extractfile('TRANSFER_MEMBERS.json').read();assert hashlib.sha256(raw).hexdigest()==%r
 members=json.loads(raw)['members'];assert set(t.getnames())==set(members)|{'TRANSFER_MEMBERS.json'}
 for m in t:
  path=PurePosixPath(m.name);assert m.isfile() and not path.is_absolute() and '..' not in path.parts
  data=t.extractfile(m).read()
  if m.name!='TRANSFER_MEMBERS.json':assert len(data)==members[m.name]['bytes'] and hashlib.sha256(data).hexdigest()==members[m.name]['sha256']
 t.extractall(base,filter='data')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',GUARDFED_CPU_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
cmd=['taskset','-c',str(%r),'ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(base/'operations/remote_bind.py'),'--cpu',str(%r)]
r=subprocess.run(cmd,env=env,capture_output=True,text=True)
print(json.dumps(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
"""%(BASE,sha(archive),sha(attempt/'TRANSFER_MEMBERS.json'),a.cpu,a.cpu)
  r=command('REMOTE_BIND',ssh+['python -B -'],input=script.encode(),timeout=360)
  result=json.loads(r.stdout);assert result['returncode']==0,result
  receipt=json.loads(result['stdout'])
  for n in ('bound_metadata.tar.gz','BOUND_MEMBERS.json','BOUND_BACKUP.json'):
   command('DOWNLOAD_'+n.replace('.','_'),['scp','-q','-o','BatchMode=yes','-P','60350','root@89.22.197.55:'+BASE+'/'+n,str(attempt/n)],timeout=120)
  assert read(attempt/'BOUND_BACKUP.json')==receipt
  assert sha(attempt/'bound_metadata.tar.gz')==receipt['archive_sha256'] and (attempt/'bound_metadata.tar.gz').stat().st_size==receipt['archive_bytes']
  assert sha(attempt/'BOUND_MEMBERS.json')==receipt['inventory_sha256'];members=read(attempt/'BOUND_MEMBERS.json')['members']
  output=attempt/'verified';assert not output.exists()
  with tarfile.open(attempt/'bound_metadata.tar.gz') as t:
   assert len(t.getnames())==len(set(t.getnames()))==receipt['members'] and set(t.getnames())==set(members)|{'BOUND_MEMBERS.json'}
   for m in t:
    assert m.isfile() and (output/m.name).resolve().is_relative_to(output.resolve())
    data=t.extractfile(m).read();pin=members.get(m.name,dict(sha256=receipt['inventory_sha256'],bytes=(attempt/'BOUND_MEMBERS.json').stat().st_size))
    assert hashlib.sha256(data).hexdigest()==pin['sha256'] and len(data)==pin['bytes']
   output.mkdir();t.extractall(output,filter='data')
  package=read(output/'stage/PACKAGE_SHA256.json');assert sha(output/'stage/PACKAGE_SHA256.json')==receipt['package_sha256']
  for n,digest in package['files'].items():assert sha(output/'stage'/n)==digest
  manifest=read(output/'stage/manifest.json');assert (len(manifest['jobs']),len(manifest['reused_jobs']),len(manifest['preflight_jobs']))==(96,4,5)
  for item in manifest['jobs']+manifest['preflight_jobs']:assert sha(output/'stage/jobs'/item['job'])==item['job_sha256']
  save(attempt/'OFFSERVER_BOUND_METADATA.json',dict(status='PASS_METADATA_PACKAGE_AND_MEMBER_SHA_NO_EXECUTION',archive_sha256=receipt['archive_sha256'],package_sha256=receipt['package_sha256'],members=receipt['members'],server_original_four_acceptance=True,offserver_model_recheck=False,checkpoints_repacked=0,canaries_started=0,training_started=0))
  print(json.dumps(dict(attempt=str(attempt),package_sha256=receipt['package_sha256'],offserver_receipt_sha256=sha(attempt/'OFFSERVER_BOUND_METADATA.json'))))
 except BaseException as e:
  save(attempt/'OPERATION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,remote=BASE));raise

if __name__=='__main__':main()
