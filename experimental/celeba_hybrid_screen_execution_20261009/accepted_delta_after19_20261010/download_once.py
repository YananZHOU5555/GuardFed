"""Download this already closed exact-delta archive and raw receipts once."""
from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path(__file__).resolve().parent
remote='root@89.22.197.55:/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/accepted_delta_after19_20261010/'
names=['hybrid_after19_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json','service_log_prefix.txt','COLLECT_STDOUT.txt','COLLECT_STDERR.txt','REMOTE_COLLECT_RECEIPT.json']
assert all(not (B/name).exists() for name in names)
cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15',*[remote+name for name in names],str(B)]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(B/'SCP_STDOUT.txt').write_bytes(r.stdout);(B/'SCP_STDERR.txt').write_bytes(r.stderr)
with (B/'SCP_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
assert r.returncode==0,'Preserve transfer failure; no automatic retry'
expected={'hybrid_after19_delta.tar.gz': 'a49637ceb7d89ae41e0b84a353b2918f613fe37332e7f223a5f726e8b8d98e64', 'MEMBERS.json': '7a8ce005e6c6be731f74ca5bd23834c8e6ad7e2dd7180d96096302f654a0a87d', 'PARTIAL_ACCEPTANCE.json': '96555b4c6ea58ed74a8dafa910cc8a84bad3ba7eb96c479bd35e95a60a03c855'}
for name,digest in expected.items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(exit_code=r.returncode,raw_server_bound_hashes=expected)))
