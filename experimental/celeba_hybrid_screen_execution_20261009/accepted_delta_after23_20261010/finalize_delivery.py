"""Seal the completed exact4 original strict/F-only offserver evidence; never adopt shared state."""
from pathlib import Path
import datetime,hashlib,json
B=Path(__file__).resolve().parent;ROOT=B.parents[1];H=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
raw=read(B/'RAW_STORAGE_LOCATION.json');F=Path(raw['directory']);backup=read(B/'BACKUP_SHA256.json');server=read(B/'PARTIAL_ACCEPTANCE.json')
prior=read(B/'PREVIOUS_CHAIN.json');latest=read(B/'PREVIOUS_LATEST.json');snapshot=read(B/'AUTHORIZED_SNAPSHOT.json');ids=read(B/'EXACT_DELTA.json')['selected_ids']
assert prior['accepted_total']==server['old_accepted']==23 and server['accepted_total']==27 and len(ids)==4
assert ids==backup['accepted_new_ids']==server['accepted_new_ids'] and not set(ids)&set(prior['accepted_job_ids'])
assert len(set(prior['accepted_job_ids']+ids))==27
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json') and sha(H/latest['chain_file'])==sha(B/'PREVIOUS_CHAIN.json')
assert sha(F/'hybrid_after23_delta.tar.gz')==backup['archive_sha256'] and backup['member_count']==59
off=read(F/'OFFSERVER_MEMBER_TENSOR_PROOF.json');record=read(F/'LOCAL_RECORD_CHECKS.json')
assert off['status']=='ORIGINAL_SERVER_STRICT_PLUS_OFFSERVER_ALL_MEMBERS_AND_CPU_TENSORS_VERIFIED'
assert record['status']=='RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS'
assert off['accepted_new']==4 and {r['id'] for r in off['records']}=={r['id'] for r in record['records']}==set(ids)
assert not off['local_CUDA_initialized'] and not record['local_CUDA_initialized'] and off['different_host']
assert server['runtime']['CPU']==[106] and server['runtime']['threads']==1
assert server['source_data_verified_before_after'] and server['source_seal_sha256']=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
observed={r['id']:r for r in snapshot['rows']}
assert all(r['rounds']==70 and r['seed']==91001 and r['acceptance_sha256']==observed[r['id']]['acceptance_sha256'] for r in server['records'])
assert all(r['alpha']=={'IID':5000.0,'non-IID':5.0}[r['distribution']] for r in server['records'])
assert record['runtime_refusals']==['CPU','torch','cuda_build','threads','gpu_name_for_original_checker']
assert read(B/'METADATA_SEAL_CORRECTION.json')['source_bytes_unchanged']
for name in ['OFFSERVER_MEMBER_TENSOR_PROOF.json','LOCAL_RECORD_CHECKS.json','LOCAL_RECORD_FAILURE.json']:
    with (B/name).open('xb') as f:f.write((F/name).read_bytes())
    assert sha(B/name)==sha(F/name)
rows={p.relative_to(F).as_posix():dict(path=p.as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(F.rglob('*')) if p.is_file()}
save('RAW_STORAGE_INDEX.json',dict(status='F_ONLY_EXACT4_RAW_ARTIFACT_INDEX',root=F.as_posix(),files=rows,member_count=len(rows),archive='hybrid_after23_delta.tar.gz',archive_sha256=backup['archive_sha256'],raw_archives_and_models_on_F=True,derived_restored_tree_on_F=True,internal_drive_fallback=False,shared_source_not_modified=True))
ready=read(H/'accepted_delta_after22_20261010/ROOT_READY_CHAIN_LINK.json')
ready.pop('CPU106_release_sha256',None)
ready.update(previous_accepted=23,accepted_new=4,accepted_total_if_root_adopts=27,accepted_new_ids=ids,accepted_job_ids=prior['accepted_job_ids']+ids,
    previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),previous_chain_file=latest['chain_file'],
    previous_offserver_proof_sha256=prior['offserver_proof_sha256'],previous_root_adoption_sha256=prior['root_adoption_sha256'],authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),
    archive_sha256=backup['archive_sha256'],archive_members=59,inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],
    offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),
    actual_server_runtime=server['runtime'],actual_local_torch=record['local_torch'],remaining_if_root_adopts=5,
    source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'),metadata_seal_correction_sha256=sha(B/'METADATA_SEAL_CORRECTION.json'),
    delta_dir='accepted_delta_after23_20261010',archive=(F/'hybrid_after23_delta.tar.gz').as_posix(),archive_filename='hybrid_after23_delta.tar.gz',archive_path=(F/'hybrid_after23_delta.tar.gz').as_posix(),
    raw_storage_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),raw_storage_directory=F.as_posix(),owned_delivery_directory=B.as_posix(),
    root_copy_target=(H/'accepted_delta_after23_20261010').as_posix(),root_copy_scope='Compact source/receipt/index/proof only; archive/models/logprefix/restored remain on F.',
    helper_process_completed=True,helper_completion_evidence='COLLECT_TRANSPORT_RECEIPT.json exit0; original subprocess waited for collector completion. No extra resource polling.',
    backup_receipt_path=(F/'BACKUP_SHA256.json').as_posix(),backup_receipt_sha256=sha(F/'BACKUP_SHA256.json'),archive_inventory_path=(F/'MEMBERS.json').as_posix(),
    accepted_offserver_by_root=0,root_adopted=False)
save('ROOT_READY_CHAIN_LINK.json',ready)
limits=['Frozen32 exploratory screen uses one seed91001 and four conditions per candidate. Partial27 does not permit final recipe selection or formal100/test claims.',
    'Server strict cu128/RTX5090 device-name CUDA metadata is retained. Local torch2.8cpu member/tensor/record checks do not reproduce server runtime or run CNN inference.',
    'No prediction/probability arrays are supplied. No independent saved-array metric recomputation is claimed; original strict/native_replay/result equality and writer/null checks are retained.',
    'Shared active-service log is a labelled prefix. Only the four frozen, producer-quiescent terminal result/model/config sets are backed up; no later arrivals are accepted.',
    'First local record gate stopped on an auxiliary self-inclusive FILES seal before scientific checked replay. The failed seal/log remain; an explicit-three-member metadata seal in a new directory passed, then original record body ran once. Successful server strict/SCP/archive verification were not retried.',
    'All new raw archives/models/logs/restored artifacts are on freshly checked F. Root must adopt the compact delivery and reference absolute F raw paths. LATEST/STATE/RUNNING/Git/queue/recipe remain unchanged.']
save('DELIVERY.json',dict(status='EXACT4_ORIGINAL_SERVER_STRICT_OFFSERVER_TENSORS_RECORD_PASS_ROOT_PENDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_accepted=23,accepted_new=4,total_if_root_adopts=27,planned=32,accepted_new_ids=ids,snapshot_utc=snapshot['utc'],snapshot_terminal=27,
    snapshot_active=[dict(id=r['id'],round=r['round']) for r in snapshot['rows'] if not r['terminal'] and r['round'] is not None],snapshot_service=snapshot['service'],
    ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),archive_path=(F/'hybrid_after23_delta.tar.gz').as_posix(),archive_sha256=backup['archive_sha256'],archive_bytes=backup['archive_size'],archive_members=59,
    receipt_sha256=sha(F/'BACKUP_SHA256.json'),inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],
    offserver_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),RAW_STORAGE_INDEX_sha256=sha(B/'RAW_STORAGE_INDEX.json'),
    server_runtime=server['runtime'],local_runtime=record['local_torch'],all_bulk_on_F=True,source_members_unchanged=69,old23_models_not_repacked=True,
    accepted_offserver_by_root=0,root_adopted=False,selected_recipe=None,formal100_started=False,test=False,new_CNN=0,new_training=0,negative_results_preserved=True,limits=limits))
text='# Hybrid exact delta23→27 — root review pending\n\nOriginal strict,59 archive members,4 CPU checkpoint tensor identities and original record-layer/writer checks passed. Frozen snapshot at '+snapshot['utc']+' had27 complete/1 active(round64)/4 pending/0 failures; exactly4 new IDs were collected once. No later completion was pursued.\n\n'
text+='\n'.join('- '+identity for identity in ids)+'\n\nRaw artifacts: `'+F.as_posix()+'`. `RAW_STORAGE_INDEX.json` provides actual per-file byte hashes and paths; `MEMBERS.json` preserves original59-member inventory. `ROOT_READY_CHAIN_LINK.json` binds accepted23 parent/source69/strict/offserver/record proofs. Root adoption is still required; the proposed total is27/32.\n\n'
text+='\n'.join('- '+limit for limit in limits)+'\n'
with (B/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write(text)
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')
assert not list(B.rglob('__pycache__')) and not list(F.rglob('__pycache__'))
members=[dict(path=p.relative_to(B).as_posix(),sha256=sha(p),size=p.stat().st_size) for p in sorted(B.rglob('*')) if p.is_file()]
save('DELIVERY_FILES_SHA256.json',dict(status='SEALED_EXACT4_HYBRID_DELIVERY_ROOT_PENDING',members=members,archive_members_verified=59,original_source_members_verified=69,raw_storage_index='RAW_STORAGE_INDEX.json',bulk_external_only=True))
assert all(sha(B/r['path'])==r['sha256'] for r in members)
print(json.dumps(dict(seal_sha256=sha(B/'DELIVERY_FILES_SHA256.json'),sealed_members=len(members),raw_files=len(rows),delivery_sha256=sha(B/'DELIVERY.json'),ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),raw_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),archive_sha256=backup['archive_sha256'],receipt_sha256=sha(F/'BACKUP_SHA256.json'),offserver_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),previous=23,new=4,total_if_root_adopts=27)))
