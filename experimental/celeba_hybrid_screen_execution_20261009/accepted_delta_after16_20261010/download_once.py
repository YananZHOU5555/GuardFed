"""Download this already closed exact-delta archive and raw receipts once."""
from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path(__file__).resolve().parent
remote='root@89.22.197.55:/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/accepted_delta_after16_20261010/'
names=['hybrid_after16_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json','service_log_prefix.txt','COLLECT_STDOUT.txt','COLLECT_STDERR.txt','REMOTE_COLLECT_RECEIPT.json']
assert all(not (B/name).exists() for name in names)
cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15',*[remote+name for name in names],str(B)]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(B/'SCP_STDOUT.txt').write_bytes(r.stdout);(B/'SCP_STDERR.txt').write_bytes(r.stderr)
with (B/'SCP_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
assert r.returncode==0,'Preserve transfer failure; no automatic retry'
expected={'hybrid_after16_delta.tar.gz': 'cd00667fd14364be9c270a8408d677750c7c1a3843ab1c632addb5c8c50e70ee', 'MEMBERS.json': '91da67ac20b75452ea42230859c1a8ccb31112bd5561dd2b08f684265dcb5407', 'PARTIAL_ACCEPTANCE.json': 'b4bb801a0a4e02309c9947b481658ef581893b29ac5c48014b814bd82555f54b'}
for name,digest in expected.items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(exit_code=r.returncode,raw_server_bound_hashes=expected)))
