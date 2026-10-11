"""Root-delegated one-time execution wrapper; original reviewed sources unchanged."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent
V2=HERE.parent/'LoGo100_cuda_actual_prerequisites_v2_20261011'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['transfer','environment']);args=p.parse_args()
    label=args.action.upper();auth=HERE/('ROOT_TRANSFER_AUTH.json' if args.action=='transfer' else 'ROOT_ENV_AUTH.json');a=json.loads(auth.read_bytes())
    assert a['execution_authorized'] is True
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(a['fresh_utc'])).total_seconds()<300,'Resource observation expired'
    input_bytes=None
    if args.action=='transfer':
        source=V2/'transfer_originals_proposal.py';assert sha(source)==a['sender_sha256']
        argv=[sys.executable,str(source),'--root-approval',str(auth),'--root-approval-sha256',sha(auth)]
    else:
        source=V2/'create_cpu_environment_proposal.sh';assert sha(source)==a['source_sha256'];input_bytes=source.read_bytes()
        command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 nice -n 10 ionice -c 3 taskset -c 113 /bin/sh -s'
        argv=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',command]
    start=datetime.now(timezone.utc).isoformat();rc=None;error=None
    stdout=HERE/(label+'.stdout');stderr=HERE/(label+'.stderr')
    try:
        with stdout.open('xb') as out,stderr.open('xb') as err:
            result=subprocess.run(argv,input=input_bytes,stdout=out,stderr=err,timeout=900)
        rc=result.returncode
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
        record={'argv':argv,'source_sha256':sha(source),'authorization_sha256':sha(auth),'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'returncode':rc,'exception':error,'stdout_sha256':sha(stdout) if stdout.exists() else None,'stderr_sha256':sha(stderr) if stderr.exists() else None,'automatic_retry':False}
        with (HERE/(label+'_COMMAND.json')).open('x',encoding='utf8') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(record))
    assert rc==0,'One-shot failed; preserve evidence, no retry'
if __name__=='__main__':main()
