"""Root-only local launcher/observer. All remote stdout/stderr retained exclusively."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','observe']);p.add_argument('--output',type=Path,required=True);p.add_argument('--root-inputs',type=Path);p.add_argument('--root-inputs-sha256');a=p.parse_args()
 source=HERE/('start_remote.py' if a.mode=='start' else 'observe_remote.py');raw=source.read_bytes()
 if a.mode=='start':
  need(a.root_inputs is not None and sha(a.root_inputs)==a.root_inputs_sha256,'Actual root start input SHA required')
  binding=json.loads(a.root_inputs.read_text());need(binding['execution_authorized'] is True,'Template cannot execute')
  need(binding['start_source_sha256']==sha(source) and binding['launcher_sha256']==sha(__file__),'Source changed')
  raw=('BINDING='+repr(binding)+'\n').encode()+raw
 compile(raw,str(source),'exec')
 need(not a.output.exists() and not a.output.is_symlink(),'New local command-evidence directory required');a.output.mkdir()
 (a.output/'REMOTE_EXECUTED.py').write_bytes(raw)
 command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 ionice -c 3 python3 -B -'
 argv=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',command]
 report={'argv':argv,'mode':a.mode,'start_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(source),'launcher_sha256':sha(__file__),'expanded_source_sha256':hashlib.sha256(raw).hexdigest(),'root_inputs_sha256':a.root_inputs_sha256,'returncode':None,'exception':None,'automatic_retry':False}
 try:
  with (a.output/'stdout').open('xb') as out,(a.output/'stderr').open('xb') as err:r=subprocess.run(argv,input=raw,stdout=out,stderr=err,timeout=240 if a.mode=='start' else 60)
  report['returncode']=r.returncode
 except BaseException as exc:report['exception']=repr(exc);raise
 finally:
  report['end_utc']=datetime.now(timezone.utc).isoformat()
  for n in ['stdout','stderr']:
   if (a.output/n).exists():report[n+'_sha256']=sha(a.output/n)
  with (a.output/'COMMAND.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
 need(report['returncode']==0,'Remote command failed; preserve evidence; no automatic retry')
if __name__=='__main__':main()
