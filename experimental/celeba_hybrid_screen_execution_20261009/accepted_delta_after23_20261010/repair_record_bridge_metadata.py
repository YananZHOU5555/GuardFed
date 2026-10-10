"""Preserve the failed self-inclusive helper seal; execute only the previously unrun record body."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
B=Path(__file__).resolve().parent;ROOT=B.parents[1]
sys.path.insert(0,str(ROOT/'tmp'));from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
F=Path(json.loads((B/'RAW_STORAGE_LOCATION.json').read_bytes())['directory'])
assert (F/'OFFSERVER_MEMBER_TENSOR_PROOF.json').exists() and not (F/'LOCAL_RECORD_CHECKS.json').exists()
old=B/'local_record_bridge_v2';new=B/'local_record_bridge_v2_verified';assert not new.exists();new.mkdir()
names=['bridge.py','checked_record_body.py','SOURCE_REUSE.json']
for name in names:
    with (new/name).open('xb') as f:f.write((old/name).read_bytes())
rows=[dict(path=name,sha256=sha(new/name),size=(new/name).stat().st_size) for name in names]
with (new/'FILES_SHA256.json').open('x',encoding='utf8',newline='\n') as f:json.dump(dict(members=rows),f,indent=2);f.write('\n')
assert all(sha(new/r['path'])==r['sha256'] for r in rows)
original=(B/'run_record_checks.py').read_text('utf8')
source=original.replace("parent/'local_record_bridge_v2';","parent/'local_record_bridge_v2_verified';").replace("'LOCAL_RECORD_FAILURE.json'","'LOCAL_RECORD_FAILURE_V2.json'")
assert source!=original and source.replace("parent/'local_record_bridge_v2_verified';","parent/'local_record_bridge_v2';").replace("'LOCAL_RECORD_FAILURE_V2.json'","'LOCAL_RECORD_FAILURE.json'")==original
path=B/'run_record_checks_v2.py';compile(source,str(path),'exec')
with path.open('x',encoding='utf8',newline='\n') as f:f.write(source)
volume=check_bulk_storage(4*1024**2)
correction=dict(status='AUXILIARY_SEAL_SELF_INCLUSION_FIXED_EXPLICIT_THREE_MEMBERS',failed_seal_sha256=sha(old/'FILES_SHA256.json'),corrected_seal_sha256=sha(new/'FILES_SHA256.json'),failed_original_record_command_receipt_sha256=sha(B/'run_record_checks_COMMAND_RECEIPT.json'),failed_record_log_path=str(F/'LOCAL_RECORD_FAILURE.json'),source_bytes_unchanged=all(sha(old/name)==sha(new/name) for name in names),record_body_first_failure_reached=False,server_strict_rerun=False,archive_verifier_rerun=False,F_guard=volume)
with (B/'METADATA_SEAL_CORRECTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(correction,f,indent=2);f.write('\n')
start=datetime.datetime.now(datetime.timezone.utc).isoformat();result=subprocess.run([sys.executable,'-B',str(path)],capture_output=True)
for name,data in [('run_record_checks_v2_STDOUT.txt',result.stdout),('run_record_checks_v2_STDERR.txt',result.stderr)]:
    with (B/name).open('xb') as f:f.write(data)
with (B/'run_record_checks_v2_COMMAND_RECEIPT.json').open('x',encoding='utf8',newline='\n') as f:json.dump(dict(start_utc=start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode,source_sha256=sha(path),metadata_correction_only=True,no_automatic_retry=True),f,indent=2);f.write('\n')
print(result.stdout.decode(errors='replace'));print(result.stderr.decode(errors='replace'))
assert result.returncode==0,'Preserve v2 failure and stop'
