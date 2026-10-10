"""Reviewed code only to the server, then an actual read-only Linux check."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp'))
from guardfed_local_storage import STORAGE_ROOT,check_bulk_storage
SOURCE=ROOT/'tmp/celeba_mechanism_remaining_evaluation_v2_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE/'FILES_SHA256.json')=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
review=ROOT/'tmp/celeba_mechanism_remaining620_v2_independent_review_20261010/REVIEW.json'
assert sha(review)=='e54a0c2f9e0182cc6f45f4757513fb3932706d55e30039a5fe1e79450b5afd4c'
files=json.loads((SOURCE/'FILES_SHA256.json').read_bytes())['files']
assert all(sha(SOURCE/n)==v['sha256'] for n,v in files.items())
storage=check_bulk_storage(sum(v['bytes'] for v in files.values())+1048576)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
archive=STORAGE_ROOT/'deployments/celeba_remaining620_v2_20261010'/(stamp+'.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'x:gz') as t:
 for n in list(files)+['FILES_SHA256.json','HANDOFF.json']:t.add(SOURCE/n,arcname=n,recursive=False)
 t.add(HERE/'remote_preflight.py',arcname='root_operations/remote_preflight.py',recursive=False)
remote='/workspace/guardfed_checks/remaining620_v2_source_'+stamp+'.tar.gz'
r=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350',str(archive),'root@89.22.197.55:'+remote],capture_output=True,timeout=90)
(HERE/'source_upload.stdout').write_bytes(r.stdout);(HERE/'source_upload.stderr').write_bytes(r.stderr);r.check_returncode()
code="""from pathlib import Path
import hashlib,json,subprocess,tarfile
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
p=Path(ARCHIVE);base=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010')
assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA and not base.exists()
with tarfile.open(p) as t:
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers())
 base.mkdir();t.extractall(base,filter='data')
r=subprocess.run(['/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(base/'root_operations/remote_preflight.py')],capture_output=True,text=True,timeout=55)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
""".replace('ARCHIVE',repr(remote)).replace('SHA',repr(sha(archive)))
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],input=code.encode(),capture_output=True,timeout=75)
(HERE/'preflight.stdout').write_bytes(r.stdout);(HERE/'preflight.stderr').write_bytes(r.stderr);r.check_returncode();d=json.loads(r.stdout)
(HERE/'SOURCE_AND_PREFLIGHT.json').write_text(json.dumps(dict(**d,local_archive=str(archive),archive_sha256=sha(archive),storage_preflight=storage,
 independent_review_sha256=sha(review),new_CNN_calls=0,actual_dispatch=False),indent=2)+'\n',encoding='utf8')
assert d['returncode']==0,'Preserve preflight failure; do not start'
print(d['stdout'])
