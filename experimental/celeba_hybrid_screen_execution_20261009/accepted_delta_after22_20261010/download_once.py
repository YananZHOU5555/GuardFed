"""Download this already closed exact-delta archive and raw receipts once."""
from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path(__file__).resolve().parent
remote='root@89.22.197.55:/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/accepted_delta_after22_20261010/'
names=['hybrid_after22_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json','service_log_prefix.txt','COLLECT_STDOUT.txt','COLLECT_STDERR.txt','REMOTE_COLLECT_RECEIPT.json']
assert all(not (B/name).exists() for name in names)
cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15',*[remote+name for name in names],str(B)]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(B/'SCP_STDOUT.txt').write_bytes(r.stdout);(B/'SCP_STDERR.txt').write_bytes(r.stderr)
with (B/'SCP_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
assert r.returncode==0,'Preserve transfer failure; no automatic retry'
expected={'hybrid_after22_delta.tar.gz': '06bbab6b219b3314d8becbb1901f0d0b24c5d084be60cf2bf59e420aea579853', 'MEMBERS.json': '48bcfd5d07edf815eafe59ab32569b06c3b14becc5c3025fa835761a37b36db7', 'PARTIAL_ACCEPTANCE.json': '197c4b0f6d71ebc1890586406dd43d0fe514ad457bedb5beb0dfe24abfa01a4f'}
for name,digest in expected.items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(exit_code=r.returncode,raw_server_bound_hashes=expected)))
