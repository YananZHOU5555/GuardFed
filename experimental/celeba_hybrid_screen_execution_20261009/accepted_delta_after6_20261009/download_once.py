"""Download this already closed exact-delta archive and raw receipts once."""
from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path(__file__).resolve().parent
remote='root@89.22.197.55:/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/accepted_delta_after6_20261009/'
names=['hybrid_after6_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json','service_log_prefix.txt','COLLECT_STDOUT.txt','COLLECT_STDERR.txt','REMOTE_COLLECT_RECEIPT.json']
assert all(not (B/name).exists() for name in names)
cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15',*[remote+name for name in names],str(B)]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(B/'SCP_STDOUT.txt').write_bytes(r.stdout);(B/'SCP_STDERR.txt').write_bytes(r.stderr)
with (B/'SCP_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
assert r.returncode==0,'Preserve transfer failure; no automatic retry'
expected={'hybrid_after6_delta.tar.gz':'efffebcaecacace9143e579f91d315ab13dd16fd20cce90481a3ba3072dd3c9d','MEMBERS.json':'4a76135321d4da21b5a08e70b615da90b345fc3a507b1fb06044ed71a39ddbaf','PARTIAL_ACCEPTANCE.json':'c796e51e497e6f8d03d337da019beeb4d24550095dece1bd9cf3d3c9a871d722'}
for name,digest in expected.items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(exit_code=r.returncode,raw_server_bound_hashes=expected)))
