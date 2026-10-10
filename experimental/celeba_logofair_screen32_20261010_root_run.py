"""Execute one reviewed source-only command, with bulk logs exclusively on F."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp'))
from guardfed_local_storage import STORAGE_ROOT,check_bulk_storage
action=sys.argv[1]
assert action in ('stage','queue','summary')
SOURCE=ROOT/'tmp/celeba_logofair_screen32_20261010'
META=ROOT/'tmp/celeba_logofair_screen32_root_execution_20261010'
META.mkdir(exist_ok=True)
review=ROOT/'tmp/celeba_logofair_screen32_independent_review_20261010/REVIEW.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(review)=='418868c324b3931ae590b590a72badaee240829a8c09d21476944827f88b27fe'
assert sha(SOURCE/'FILES_SHA256.json')=='accd5cb8582a344f870188f1e70661b6dc9dc948cc88c9e6f6651e451607bc49'
storage=check_bulk_storage(128*1024**2)
DEST=STORAGE_ROOT/'logofair_screen32_20261010'
logs=DEST/'transport';logs.mkdir(parents=True,exist_ok=True)
exe=ROOT/'tmp/celeba_baselines/remaining_20261009/group_a/.venv/Scripts/python.exe'
if action=='stage':
    args=[str(exe),'-B',str(SOURCE/'stage_inputs.py'),'--out',str(DEST/'inputs')]
elif action=='queue':
    assert json.loads((META/'stage_EXIT.json').read_bytes())['exit_code']==0
    args=[str(exe),'-B',str(SOURCE/'run_queue.py'),'--repo',str(ROOT/'tmp/revision-publish-20260928'),
          '--inputs',str(DEST/'inputs'),'--out',str(DEST/'attempt001')]
else:
    assert json.loads((META/'queue_EXIT.json').read_bytes())['exit_code']==0
    args=[str(exe),'-B',str(SOURCE/'summarize.py'),'--index',str(DEST/'attempt001/STRICT32_INDEX.json'),
          '--out',str(DEST/'attempt001/SUMMARY32.json')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
         MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
with (logs/(action+'.stdout')).open('xb') as out,(logs/(action+'.stderr')).open('xb') as err:
    child=subprocess.Popen(args,stdout=out,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    record=dict(pid=child.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                argv=args,storage_preflight=storage,source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'),
                independent_review_sha256=sha(review),logs=str(logs),no_auto_retry=True)
    (META/(action+'_LIVE_HANDLE.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(action=action,pid=child.pid,logs=str(logs))),flush=True)
    code=child.wait()
(META/(action+'_EXIT.json')).write_text(json.dumps(dict(exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source_seal_sha256=record['source_seal_sha256'],logs=str(logs)),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(action=action,exit_code=code)),flush=True)
sys.exit(code)
