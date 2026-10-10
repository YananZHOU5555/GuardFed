from pathlib import Path
import base64,hashlib,json,subprocess,datetime,sys
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();remote='/workspace/guardfed_checks/'+H.name
names=['collect_delta.py','verify_delta_offserver.py','PREVIOUS_OFFSERVER_ACCEPTANCE.json'];payload={n:base64.b64encode((H/n).read_bytes()).decode() for n in names};pins={n:sha(H/n) for n in names}
code="""from pathlib import Path
import base64,hashlib,json,subprocess,sys
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
b=Path(%r);assert not b.exists() and not b.is_symlink() and b.parent.resolve()==b.parent
payload=%r;pins=%r
b.mkdir()
for n,v in payload.items():
 data=base64.b64decode(v);assert hashlib.sha256(data).hexdigest()==pins[n]
 with (b/n).open('xb') as f:f.write(data)
cmd=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','taskset','-c','106','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(b/'collect_delta.py'),'--out',str(b/'batch'),'--cpu','106','--source-sha256',pins['collect_delta.py'],'--previous',str(b/'PREVIOUS_OFFSERVER_ACCEPTANCE.json'),'--previous-sha256',pins['PREVIOUS_OFFSERVER_ACCEPTANCE.json']]
r=subprocess.run(cmd,capture_output=True)
(b/'COLLECT_STDOUT.txt').write_bytes(r.stdout);(b/'COLLECT_STDERR.txt').write_bytes(r.stderr)
receipt=dict(exit_code=r.returncode,command=cmd,source_pins=pins,stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest())
(b/'COLLECT_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt));sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)
"""%(remote,payload,pins)
(H/'DISPATCH.py').write_text(code,encoding='utf8')
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -']
r=subprocess.run(cmd,input=code.encode(),capture_output=True,timeout=180)
(H/'DISPATCH_STDOUT.txt').write_bytes(r.stdout);(H/'DISPATCH_STDERR.txt').write_bytes(r.stderr)
(H/'DISPATCH_RECEIPT.json').write_text(json.dumps(dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(H/'DISPATCH.py'))),encoding='utf8');r.check_returncode()
print(r.stdout.decode())
