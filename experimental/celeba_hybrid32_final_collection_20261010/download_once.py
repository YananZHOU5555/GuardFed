"""One transfer of the original exact5 archive/receipts directly to guarded F storage."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess,sys
B=Path(__file__).resolve().parent;ROOT=B.parents[1]
sys.path.insert(0,str(ROOT/'tmp'));from guardfed_local_storage import check_bulk_storage,STORAGE_ROOT
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')
names=['hybrid_final5_delta.tar.gz','BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json','service_log_prefix.txt','COLLECT_STDOUT.txt','COLLECT_STDERR.txt','REMOTE_COLLECT_RECEIPT.json']
remote='/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/hybrid32_final_collection_20261010'
code="from pathlib import Path; import json,hashlib; b=Path("+repr(remote)+"); names="+repr(names)+"; print(json.dumps({n:{'sha256':hashlib.sha256((b/n).read_bytes()).hexdigest(),'bytes':(b/n).stat().st_size} for n in names}))"
cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55','python3 -B -c '+shlex.quote(code)]
result=subprocess.run(cmd,capture_output=True)
with (B/'REMOTE_ARTIFACT_METADATA_STDERR.txt').open('xb') as f:f.write(result.stderr)
assert result.returncode==0,'Preserve metadata transport failure; no automatic retry'
metadata=json.loads(result.stdout);save('REMOTE_ARTIFACT_METADATA.json',metadata)
server=next(json.loads(line) for line in (B/'COLLECT_TRANSPORT_STDOUT.txt').read_text('utf8').splitlines() if line.startswith('{') and 'archive_sha256' in line)
for name,key in [('hybrid_final5_delta.tar.gz','archive_sha256'),('MEMBERS.json','inventory_sha256'),('PARTIAL_ACCEPTANCE.json','acceptance_sha256')]:assert metadata[name]['sha256']==server[key]
volume=check_bulk_storage(sum(x['bytes'] for x in metadata.values()))
destination=STORAGE_ROOT/'celeba_hybrid32_final_collection_20261010';assert not destination.exists()
destination.mkdir()
command=['scp','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15',*['root@89.22.197.55:'+remote+'/'+name for name in names],str(destination)]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();result=subprocess.run(command,capture_output=True)
for name,data in [('SCP_STDOUT.txt',result.stdout),('SCP_STDERR.txt',result.stderr)]:
    with (B/name).open('xb') as f:f.write(data)
save('SCP_RECEIPT.json',dict(command=command,started_utc=start,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode,volume=volume,metadata=metadata,no_automatic_retry=True))
assert result.returncode==0,'Preserve SCP partial state; no automatic retry'
for name,pin in metadata.items():assert sha(destination/name)==pin['sha256'] and (destination/name).stat().st_size==pin['bytes']
for name in ('BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','live_snapshot.json','REMOTE_COLLECT_RECEIPT.json'):
    with (B/name).open('xb') as f:f.write((destination/name).read_bytes())
save('RAW_STORAGE_LOCATION.json',dict(status='EXACT5_ORIGINAL_ARCHIVE_AND_RECEIPTS_F_BYTES_VERIFIED',directory=str(destination),files=metadata,archive_sha256=server['archive_sha256'],archive_members=server['member_count'],accepted_new_ids=server['accepted_new_ids'],all_bulk_on_F=True))
print(json.dumps(dict(status='SCP_ONCE_F_HASHES_PASS',directory=str(destination),archive_sha256=server['archive_sha256'],members=server['member_count'])))
