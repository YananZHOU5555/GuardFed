"""Execute one root-approved frozen repair source; retain prior failed environment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
auth=H/'ROOT_ENV_REPAIR_AUTH.json';a=json.loads(auth.read_bytes());source=H/'env_repair_source/create_cpu_environment_once.sh'
assert a['execution_authorized'] is True and a['source_sha256']==sha(source)=='2fe9b555d8aa50d8f2048f90584551a8e2ce39ff4b36f3c990f2ca6c3bb57b3f'
assert (datetime.now(timezone.utc)-datetime.fromisoformat(a['fresh_utc'])).total_seconds()<300
command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 nice -n 10 ionice -c 3 taskset -c 113 /bin/sh -s'
argv=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',command]
start=datetime.now(timezone.utc).isoformat();rc=None;error=None
try:
    with (H/'ENV_REPAIR.stdout').open('xb') as out,(H/'ENV_REPAIR.stderr').open('xb') as err:
        r=subprocess.run(argv,input=source.read_bytes(),stdout=out,stderr=err,timeout=900)
    rc=r.returncode
except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
finally:
    record={'argv':argv,'source_sha256':sha(source),'authorization_sha256':sha(auth),'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'returncode':rc,'exception':error,'stdout_sha256':sha(H/'ENV_REPAIR.stdout'),'stderr_sha256':sha(H/'ENV_REPAIR.stderr'),'automatic_retry':False}
    with (H/'ENV_REPAIR_COMMAND.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))
assert rc==0,'Repair failed: preserve evidence and stop without retry'
