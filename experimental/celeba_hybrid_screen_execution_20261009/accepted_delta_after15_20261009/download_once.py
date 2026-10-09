"""Download this already closed exact-delta archive and raw receipts once."""
from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path(__file__).resolve().parent
remote='root@89.22.197.55:/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/accepted_delta_after15_20261009/'
names=['hybrid_after15_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json','service_log_prefix.txt','COLLECT_STDOUT.txt','COLLECT_STDERR.txt','REMOTE_COLLECT_RECEIPT.json']
assert all(not (B/name).exists() for name in names)
cmd=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15',*[remote+name for name in names],str(B)]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
(B/'SCP_STDOUT.txt').write_bytes(r.stdout);(B/'SCP_STDERR.txt').write_bytes(r.stderr)
with (B/'SCP_RECEIPT.json').open('x',encoding='utf8') as f:json.dump(dict(command=cmd,exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f,indent=2);f.write('\n')
assert r.returncode==0,'Preserve transfer failure; no automatic retry'
expected={'hybrid_after15_delta.tar.gz':'c65f31ce85734f308d9b93e35259d96d7a612499662e7473b1a1352881203575','MEMBERS.json':'64ead37e2841a56c9924b160b00f04d17100009c43a9a728e7b3cb22a5c88b2e','PARTIAL_ACCEPTANCE.json':'8bc3b23887b156c8c2a10c2180e6e02ddf5473d2f7115599ce53199c0475cb44'}
for name,digest in expected.items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==digest,name
print(json.dumps(dict(exit_code=r.returncode,raw_server_bound_hashes=expected)))
