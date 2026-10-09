"""Upload only new delta metadata, run one original strict collector, then stop."""
from pathlib import Path
import datetime,hashlib,io,json,shlex,subprocess,tarfile
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')
names=['collect_once.py','PREVIOUS_CHAIN.json','PREVIOUS_LATEST.json','AUTHORIZED_SNAPSHOT.json','EXACT_DELTA.json','COLLECTOR_DIFF.patch','SOURCE_RECEIPT.json']
pins={name:sha(B/name) for name in names}
bundle=B/'collector_transport.tar'
with tarfile.open(bundle,'x') as archive:
 for name in names:archive.add(B/name,arcname=name,recursive=False)
remote="""from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile
B=Path('/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/accepted_delta_after14_20261009')
pins=PINS
payload=sys.stdin.buffer.read()
assert not B.exists()
with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
 assert len(archive.getnames())==len(pins) and set(archive.getnames())==set(pins)
 data={}
 for member in archive.getmembers():
  assert member.isfile() and member.name in pins
  value=archive.extractfile(member).read();assert hashlib.sha256(value).hexdigest()==pins[member.name]
  data[member.name]=value
B.mkdir()
for name,value in data.items():
 with (B/name).open('xb') as f:f.write(value)
cmd=['ionice','-c','3','nice','-n','10','taskset','-c','110','env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=0','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(B/'collect_once.py')]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
with (B/'COLLECT_STDOUT.txt').open('xb') as f:f.write(r.stdout)
with (B/'COLLECT_STDERR.txt').open('xb') as f:f.write(r.stderr)
receipt=dict(command=cmd,exit_code=r.returncode,transport_members=pins,collector_stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),collector_stderr_sha256=hashlib.sha256(r.stderr).hexdigest())
with (B/'REMOTE_COLLECT_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\\n')
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr)
print(json.dumps(receipt));sys.exit(r.returncode)
""".replace('PINS',repr(pins))
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55','python3 -B -c '+shlex.quote(remote)]
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(cmd,input=bundle.read_bytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(B/'COLLECT_TRANSPORT_STDOUT.txt').write_bytes(r.stdout);(B/'COLLECT_TRANSPORT_STDERR.txt').write_bytes(r.stderr)
save('COLLECT_TRANSPORT_RECEIPT.json',dict(started_utc=started,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=r.returncode,transport_sha256=sha(bundle),transport_members=pins,no_automatic_retry=True))
print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'))
assert r.returncode==0,'Preserve failed attempt; no strict or numerical retry'
