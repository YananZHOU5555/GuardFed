"""Execute the approved fixed candidate intersection once; never retries."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime,traceback
H=Path(__file__).resolve().parent
def save(n,x):
 with (H/n).open('x',encoding='utf8') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 assert not (H/'PREFLIGHT_COMMAND.json').exists()
 exe=json.loads((H/'EXECUTION_INPUTS.json').read_bytes());assert exe['native251_root_verified']
 cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 111 nice -n 10 ionice -c 3 python -B -']
 code=(H/'remote_preflight.py').read_bytes();save('PREFLIGHT_COMMAND.json',dict(argv=cmd,source_sha256=hashlib.sha256(code).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False))
 p=subprocess.run(cmd,input=code,capture_output=True);(H/'PREFLIGHT.stdout.json').write_bytes(p.stdout);(H/'PREFLIGHT.stderr.txt').write_bytes(p.stderr);save('PREFLIGHT_EXIT.json',dict(returncode=p.returncode));p.check_returncode()
 pre=json.loads(p.stdout);print(json.dumps(dict(step='one_snapshot_closed_intersection',ids=pre['selected_ids'],status=pre['status'])),flush=True)
 if not pre['selected_ids']:return
 for n in ('export_once.py','download_verify_once.py'):
  p=subprocess.run([sys.executable,'-B',str(H/n)]);p.check_returncode()
if __name__=='__main__':
 try:main()
 except BaseException as e:save('EXECUTION_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
