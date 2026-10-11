
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
stage=Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011')
assert H('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert H(stage/'closure_source/export14.py')=='347a1447565811592e1a81bce19de724d357623e8ec684f288d70a64225196fa'
assert H(stage/'closure_source/SOURCE_PINS.json')=='220ee52596eeb8ff3c2b7bb10d90061088c67ffc4afcc722c13712be40779976'
assert H(stage/'GATE_COMPARISON.json')=='4a805cef7e01cb3a9cf4b55411a7a0efa33d0a1eeea8c886c851f0d43ca0b8d4'
assert H(stage/'closure_source/RUNTIME_MEMBERS.json')=='aec1f60e5e02e42f7739d61928a0141944186e5362723ad9b96f503609ad9933'
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=stage/('offserver_closure_'+tag)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
cmd=['taskset','-c','110','nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(stage/'closure_source/export14.py'),'--comparison-sha256','4a805cef7e01cb3a9cf4b55411a7a0efa33d0a1eeea8c886c851f0d43ca0b8d4','--runtime-manifest',str(stage/'closure_source/RUNTIME_MEMBERS.json'),'--runtime-manifest-sha256','aec1f60e5e02e42f7739d61928a0141944186e5362723ad9b96f503609ad9933','--destination',str(dest),'--cpu','110']
start=datetime.datetime.now(datetime.timezone.utc).isoformat();r=subprocess.run(cmd,env=env,capture_output=True)
out=stage/'runtime/root_export_actual001';out.mkdir()
(out/'stdout.json').write_bytes(r.stdout);(out/'stderr.txt').write_bytes(r.stderr)
(out/'COMMAND.json').write_text(json.dumps({'argv':cmd,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'automatic_retry':False},indent=2)+'\n')
assert r.returncode==0,r.stderr.decode(errors='replace')[-4000:]
v=json.loads(r.stdout);archive=Path(v['backup']);receipt=archive.parent/'backup_receipt.json';b=json.loads(receipt.read_bytes())
assert H(archive)==b['archive_sha256'] and b['comparison_sha256']=='4a805cef7e01cb3a9cf4b55411a7a0efa33d0a1eeea8c886c851f0d43ca0b8d4'
print(json.dumps({'status':v['status'],'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':H(archive),'receipt':str(receipt),'receipt_sha256':H(receipt),'receipt_json':b,'members':v['members'],'records':len(v['records']),'tag':tag,'source_sha256':H(stage/'closure_source/export14.py')}))
