"""Append one verified delta to the local accepted-ID backup chain."""
import hashlib
import json
from pathlib import Path

B=Path(__file__).resolve().parent
BASE=B.parents[1]
H=BASE.parent/'celeba_hybrid_screen_execution_20261009/results_incremental_20261009T1133Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')

previous=read(B/'PREVIOUS_CHAIN.json'); receipt=read(B/'BACKUP_SHA256.json'); proof=read(B/'OFFSERVER_ACCEPTANCE.json')
assert proof['original_checked_result_replayed_locally'] and proof['different_host_observed']
assert proof['accepted_new']==4 and proof['accepted_total']==6 and proof['archived_member_count']==45
assert proof['archive_sha256']==receipt['archive_sha256']==sha(B/'accepted_delta_four.tar.gz')
new=receipt['accepted_new_ids']; old=previous['accepted_job_ids']
assert len(set(new+old))==6 and not set(new)&set(old)
pointer=read(BASE/'LATEST_BACKUP.json')
assert pointer['chain_sha256']==sha(B/'PREVIOUS_CHAIN.json') and pointer['accepted']==2
(B/'PREVIOUS_LATEST_BACKUP.json').write_bytes((BASE/'LATEST_BACKUP.json').read_bytes())
chain_name='BACKUP_CHAIN_increment_20261009T1133Z.json'
chain=dict(status='PARTIAL_RESULTS_OFFSERVER_VERIFIED',accepted=6,planned=32,complete=False,
    utc=proof['utc'],snapshot_utc=read(B/'live_snapshot.json')['utc'],
    previous_chain=dict(file=pointer['chain_file'],sha256=pointer['chain_sha256']),
    source_archive_reused=previous['source_archive_reused'],package_sha256=previous['package_sha256'],
    accepted_job_ids=old+new,
    new_batch=dict(path='backups/increment_20261009T1133Z',archive='accepted_delta_four.tar.gz',
        archive_sha256=receipt['archive_sha256'],archive_members=45,inventory_sha256=receipt['inventory_sha256'],
        server_backup_receipt_sha256=sha(B/'BACKUP_SHA256.json'),offserver_acceptance_sha256=sha(B/'OFFSERVER_ACCEPTANCE.json'),
        accepted_new_ids=new),
    no_old_models_repackaged=True,candidate_selection_performed=False,final_test=False,formal100=False,
    next='Observe unchanged32 queue, accept only IDs not in this chain, no new scope or automatic retry.')
write(BASE/chain_name,chain)
new_pointer=dict(chain_file=chain_name,chain_sha256=sha(BASE/chain_name),accepted=6,planned=32,original_startup_chain_preserved=True)
(BASE/'LATEST_BACKUP.json').write_text(json.dumps(new_pointer,indent=2)+'\n',encoding='utf8',newline='\n')
start=read(H/'live_snapshot.json'); end=read(H/'closing_live_snapshot.json')
ha=[r for r in end['hybrid']['rows'] if r['progress']]
fa=[r for r in end['flgmm']['rows'] if r['progress'] and not r['terminal_acceptance']]
status=dict(status='HEALTHY_IN_PROGRESS_NO_NEW_HYBRID_TERMINAL',snapshot_utc=start['utc'],closing_utc=end['utc'],
    accepted=0,planned=32,active=1,pending=31,failed=0,
    active_id=ha[0]['id'],round_at_snapshot=48,round_at_closing=ha[0]['progress']['round'],
    frozen_source_members_verified=read(B/'PARTIAL_ACCEPTANCE.json')['hybrid']['frozen_source_members_verified'],
    source_seal_sha256=end['hybrid']['source_seal_sha256'],scope_sha256=end['hybrid']['scope_sha256'],
    runtime_protocol_sha256=end['hybrid']['protocol_sha256'],cpu=104,threads=1,nice=10,physical_gpu=0,
    scientific_acceptance_or_backup_performed=False,
    next='Keep original queue; use original checked for next new70 terminal; final selection only after all32.',
    flgmm_delta_offserver_proof=dict(path=str(B/'OFFSERVER_ACCEPTANCE.json'),sha256=sha(B/'OFFSERVER_ACCEPTANCE.json')))
write(H/'STATUS.json',status)
delivery=dict(status='BOUNDED_SNAPSHOT_DELTA_CLOSED',snapshot_utc=start['utc'],closing_utc=end['utc'],
    Hybrid=dict(accepted=0,planned=32,failed=0,active=1,rounds=[48,ha[0]['progress']['round']],source_members=69),
    FLGMM=dict(accepted=6,planned=32,accepted_previous=2,new_strict_offserver_accepted=4,failed=0,active=2,pending=24,
        active_rounds=[r['progress']['round'] for r in fa],archive_sha256=receipt['archive_sha256'],archive_members=45,
        offserver_proof_sha256=sha(B/'OFFSERVER_ACCEPTANCE.json'),chain_file=chain_name,chain_sha256=sha(BASE/chain_name)),
    CPU105_untouched=True,helper_CPU106_single_thread=True,no_queue_or_frozen_source_changes=True,
    no_test_or_formal100=True,no_later_completed_ids_collected=True,
    preserved_prior_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),original_strict_accepted_locally_and_on_server=True)
write(B/'DELIVERY.json',delivery)
write(B/'FILES_SHA256.json',dict(files={p.name:sha(p) for p in sorted(B.iterdir()) if p.is_file()}))
write(H/'FILES_SHA256.json',dict(files={p.name:sha(p) for p in sorted(H.iterdir()) if p.is_file()}))
print(json.dumps(delivery))
