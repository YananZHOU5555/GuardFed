"""Independently verify deployed source archives and actual bounded startup receipts."""
from pathlib import Path
import datetime
import hashlib
import json
import tarfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_archive(path,expected,rows):
    assert sha(path)==expected
    with tarfile.open(path) as t:
        members=t.getmembers()
        assert len(members)==len({m.name for m in members})==len(rows)
        for member in members:
            assert member.isfile() and member.name in rows
            row=rows[member.name]
            assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==(row['sha256'] if isinstance(row,dict) else row)
            if isinstance(row,dict):assert member.size==row['bytes']
    return len(rows)
p=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
chain=read(p/'BACKUP_CHAIN.json')
a=chain['source_archive']; assert verify_archive(p/a['file'],a['sha256'],a['member_hashes'])==67
for name,expected in chain['actual_receipts_sha256'].items():assert sha(p/name)==expected
authorization=read(p/'EXECUTION_AUTHORIZATION.json');live=read(p/'FIRST_PROGRESS_V2.json')
assert authorization['package_sha256']==live['package_sha256']==chain['release_seal_sha256']
assert authorization['scope']=='32_valid_only_screen' and authorization['jobs']==32 and authorization['rounds']==70
assert not authorization['final_test'] and not authorization['formal100'] and not authorization['automatic_retry']
assert len(live['active'])==2 and not live['failures'] and live['completed']==0
assert all(r['nice']==10 and r['environment']['cpu_threads']==1 and r['progress']['round']>=1 for r in live['active'])
assert not chain['accepted_job_ids']
proof=dict(status='ROOT_FLGMM32_SOURCE_AND_STARTUP_VERIFIED_NOT_COMPLETION',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_archive_members=67,
    receipt_members=4,source_archive_sha256=a['sha256'],first_progress_sha256=sha(p/'FIRST_PROGRESS_V2.json'),
    package_sha256=live['package_sha256'],observed_utc=live['utc'],accepted70round_jobs=0,test_started=False)
with (OUT/'FLGMM32_STARTUP_ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
p=ROOT/'tmp/celeba_final_valid_replay_20261009/v4/remaining872_execution_20261009'
files=read(p/'deployment_files.json'); assert verify_archive(p/files['archive'],files['sha256'],files['members'])==38
launch=read(p/'launch_receipt.json');inspection=read(p/'inspect_receipt.json');live=read(p/'live_start_sample.json')
assert launch['inspect_receipt_sha256']==sha(p/'inspect_receipt.json') and inspection['returncode']==0
assert not inspection['new_image_inference'] and not launch['remaining872_complete']
assert launch['execution_manifest_sha256']=='ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43'
assert launch['source_manifest_sha256']=='8f4d9504444210a0ad6622ed6afa825913c59dad1fde01685330433f91c5efe7'
assert launch['workers_max']==len(live['workers'])==11 and live['outer_nice']==0
assert all(r['nice']==10 for r in live['workers']) and live['formal_failed_n']==0
assert not launch['new_test_inference'] and not launch['new_training'] and not launch['autostart'] and not launch['autorestart']
proof=dict(status='ROOT_REMAINING872_DEPLOYMENT_AND_REAL_STARTUP_VERIFIED_NOT_COMPLETION',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_members=38,
    archive_sha256=files['sha256'],launch_receipt_sha256=sha(p/'launch_receipt.json'),
    live_sample_sha256=sha(p/'live_start_sample.json'),actual_workers=11,outer_nice=0,worker_nice=10,
    cpu_effective_cores=live['global_effective_cpu_cores'],cpu_quota_cores=live['quota_cores'],
    actual_new_replay_acceptances_at_observation=0,test_started=False)
with (OUT/'REMAINING872_STARTUP_ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
