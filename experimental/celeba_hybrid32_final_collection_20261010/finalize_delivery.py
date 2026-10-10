"""Seal final5 recovery evidence, original complete32 selection, and the external F restore index."""
from pathlib import Path
import datetime,hashlib,json,runpy
B=Path(__file__).resolve().parent;A=B/'attempt_v2';ROOT=B.parents[1]
H=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
F=Path(read(A/'RAW_STORAGE_LOCATION.json')['directory'])
volume=runpy.run_path(str(ROOT/'tmp/guardfed_local_storage.py'))['check_bulk_storage'](0)
previous=read(B/'PREVIOUS_CHAIN.json');latest=read(B/'PREVIOUS_LATEST.json');snapshot=read(B/'AUTHORIZED_SNAPSHOT.json')
backup=read(A/'BACKUP_SHA256.json');server=read(A/'PARTIAL_ACCEPTANCE.json');summary=read(B/'SUMMARY32.json')
off=read(F/'OFFSERVER_MEMBER_TENSOR_PROOF.json');record=read(F/'LOCAL_RECORD_CHECKS.json');ids=read(B/'EXACT_DELTA.json')['selected_ids']
assert previous['accepted_total']==server['old_accepted']==27 and server['accepted_total']==32
assert len(ids)==5 and ids==backup['accepted_new_ids']==server['accepted_new_ids']
assert len(set(previous['accepted_job_ids']+ids))==32
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json') and sha(H/latest['chain_file'])==sha(B/'PREVIOUS_CHAIN.json')
assert sha(F/'hybrid_final5_delta.tar.gz')==backup['archive_sha256'] and backup['member_count']==71
assert off['accepted_new']==5 and {r['id'] for r in off['records']}=={r['id'] for r in record['records']}==set(ids)
assert off['different_host'] and not off['local_CUDA_initialized'] and not record['local_CUDA_initialized']
assert server['runtime']['CPU']==[107] and server['runtime']['threads']==1
assert read(A/'COLLECT_TRANSPORT_RECEIPT.json')['exit_code']==0
assert read(B/'COLLECT_TRANSPORT_RECEIPT.json')['exit_code']==1 and read(B/'REMOTE_COLLECT_RECEIPT.json')['exit_code']==1
assert not (B/'PARTIAL_ACCEPTANCE.json').exists() and (B/'FAILURE.json').exists()
assert summary['previous_root_adopted']==27 and summary['new_root_adopted']==0 and len(summary['all_candidates'])==8
counts=[]
for row in server['records']:
    out=F/'restored/screen_runs'/row['id'];result=read(out/'result.json');native=read(out/'native_replay.json');receipt=read(out/'acceptance.json')
    assert row['rounds']==70 and native['metrics']==result['metrics']==row['metrics']
    assert sum(native['evaluation_group_label_counts'].values())==native['prediction_count']==19867
    assert sum(native['root_group_label_counts'].values())==native['root_group_label_total']==16277
    assert sha(out/'model.pt')==row['model_sha256'] and sha(out/'acceptance.json')==row['acceptance_sha256']
    counts.append(dict(id=row['id'],rounds=70,prediction_count=19867,root_count=16277,evaluation_group_label_counts=native['evaluation_group_label_counts'],root_group_label_counts=native['root_group_label_counts'],metric_components_equal=3,model_sha256=row['model_sha256'],checkpoint_tensor_sha256=receipt['checkpoint_tensor_sha256'],native_replay_sha256=sha(out/'native_replay.json'),result_sha256=sha(out/'result.json'),acceptance_sha256=row['acceptance_sha256']))
save('SAVED_RECORD_COUNTS.json',dict(status='SAVED_RECORD_IDENTITIES_AND_COUNTS_EXACT_NO_PREDICTION_ARRAY_RECOMPUTATION',records=counts,new_models=5,rounds_per_model=70,metric_components_equal=15,valid_rows_per_model=19867,total_saved_prediction_count_metadata=99335,root_rows_per_model=16277,prediction_arrays_supplied=False))
for name in ('BACKUP_SHA256.json','MEMBERS.json','PARTIAL_ACCEPTANCE.json','OFFSERVER_MEMBER_TENSOR_PROOF.json','LOCAL_RECORD_CHECKS.json'):
    source=A/name if (A/name).is_file() else F/name
    with (B/name).open('xb') as f:f.write(source.read_bytes())
rows={p.relative_to(F).as_posix():dict(path=p.as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(F.rglob('*')) if p.is_file()}
save('RAW_STORAGE_INDEX.json',dict(status='F_ONLY_FINAL5_ORIGINAL_RAW_AND_RESTORED_INDEX',root=F.as_posix(),files=rows,member_count=len(rows),archive='hybrid_final5_delta.tar.gz',archive_sha256=backup['archive_sha256'],bulk_external_only=True,volume_at_final_index=volume))
ready=dict(status='ROOT_READY_COMPLETE32_SERVER_STRICT_OFFSERVER_TENSOR_RECORD_AND_FROZEN_SUMMARY_PASS',previous_accepted=27,accepted_new=5,accepted_total_if_root_adopts=32,accepted_new_ids=ids,accepted_job_ids=previous['accepted_job_ids']+ids,planned=32,
    previous_chain_file=latest['chain_file'],previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),previous_offserver_proof_sha256=previous['offserver_proof_sha256'],previous_root_adoption_sha256=previous['root_adoption_sha256'],authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),
    archive_sha256=backup['archive_sha256'],archive_members=71,inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),source_seal_sha256=server['source_seal_sha256'],source_members_verified=69,
    actual_server_runtime=server['runtime'],actual_local_torch=record['local_torch'],local_runtime_not_claimed_equal=True,original_checker_runtime_metadata_bound_to_actual_server_receipt=True,scientific_checked_body_exact_after_three_original_runtime_query_reversals=True,original_driver_writer_policy_and_null_semantics_unchanged=True,
    helper=dict(CPU=107,threads=1,nice=10,IO='idle'),helper_process_completed=True,helper_completion_evidence='Remote subprocess.wait returned0; COLLECT_TRANSPORT_RECEIPT exit0; no remote collector launched thereafter.',
    original_source_not_repackaged=True,no_old_models_repacked=True,canonical_latest_chain_not_changed=True,no_training_or_CNN=True,prediction_arrays_supplied=False,independent_array_metric_recompute_claim=False,
    recipe_selection=True,selected_recipe=summary['selected_recipe'],summary32_sha256=sha(B/'SUMMARY32.json'),final_test=False,formal100=False,root_adoption_required=True,accepted_offserver_by_root=0,root_adopted=False,remaining_if_root_adopts=0,
    source_reuse_verification_sha256=sha(A/'SOURCE_REUSE_VERIFICATION.json'),delta_dir='accepted_delta_after27_20261010',archive=(F/'hybrid_final5_delta.tar.gz').as_posix(),archive_path=(F/'hybrid_final5_delta.tar.gz').as_posix(),archive_filename='hybrid_final5_delta.tar.gz',backup_receipt_path=(F/'BACKUP_SHA256.json').as_posix(),backup_receipt_sha256=sha(F/'BACKUP_SHA256.json'),archive_inventory_path=(F/'MEMBERS.json').as_posix(),raw_storage_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),raw_storage_directory=F.as_posix(),owned_delivery_directory=B.as_posix(),root_copy_target=(H/'accepted_delta_after27_20261010').as_posix(),root_copy_scope='Compact sealed source/receipts/proofs/index; absolute F archive and restore tree remain external.',
    first_failure_sha256=sha(B/'FAILURE.json'),engineered_recovery='Parent-authorized independent attempt_v2 changes only supervisor EXITED rc3/RUNNING rc0 state reading and metadata bindings. Original strict per-job body is byte exact; first failure occurred before torch/strict/archive, after code/metadata upload. Original attempt retained.')
save('ROOT_READY_CHAIN_LINK.json',ready)
limits=['One seed91001; four validation conditions per candidate are not four independent seeds. No SD, significance, final test, or formal100 evidence.',
    'Original server strict uses torch2.11.0+cu128; CPU107/one thread/nice10/idle. CUDA0 is retained for original device-name metadata only. Local torch2.8.0+cpu verifies member bytes, checkpoint tensors and original record-body/writer/null guards; runtime equality is not claimed.',
    'No saved prediction/probability arrays are present. Fifteen saved metric components/native-replay identities and recorded counts are checked, without claiming prediction-array metric recomputation.',
    'The original archive retains a historical still-running log-prefix description. Actual authorized and collector snapshots show EXITED; the shared log bytes are captured and hashed, not claimed to be a closed per-job log.',
    'The first collector stopped before torch/strict/archive on normal EXITED status rc3. Source/transport metadata had already been written. It is preserved with the explicit parent-authorized v2 recovery; no scientific failure was retried.',
    'All32 summary uses original unmodified score/rank functions and all original accepted records. Old27 strict/model checks were not rerun or repackaged. The outcome awaits root adoption; no shared LATEST/STATE/recipe/Git or service was changed.']
save('HANDOFF.json',dict(status='EXACT5_FINAL32_STRICT_OFFSERVER_RECORD_AND_ORIGINAL_SUMMARY_ROOT_READY_NOT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_accepted=27,accepted_new=5,total_if_root_adopts=32,new_root_accepted=0,accepted_new_ids=ids,archive_path=ready['archive_path'],archive_sha256=backup['archive_sha256'],archive_members=71,archive_bytes=backup['archive_size'],backup_receipt_sha256=ready['backup_receipt_sha256'],inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_sha256=ready['offserver_tensor_proof_sha256'],record_sha256=ready['offserver_original_record_check_sha256'],ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),raw_storage_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),summary32_sha256=sha(B/'SUMMARY32.json'),saved_counts_sha256=sha(B/'SAVED_RECORD_COUNTS.json'),selected_recipe=summary['selected_recipe'],accuracy_champion=summary['accuracy_champion'],three_metric_Pareto=summary['three_metric_Pareto'],formal100_started=False,new_CNN=0,new_training=0,test=False,negative_results_preserved=True,limits=limits))
text='# Hybrid final5: 27 → 32 (root adoption pending)\n\nOne read-only snapshot confirmed all32 terminal, service EXITED, no Hybrid worker/failure, and CPU107 free. Five frozen new IDs passed original server strict, all71 archive members, CPU tensor identity, and original record-layer checks. `ROOT_READY_CHAIN_LINK.json` binds the unchanged adopted27 parent. `RAW_STORAGE_INDEX.json` contains exact F paths and per-file hashes.\n\n`SUMMARY32.json`/`.md` use the original sealed score/rank on all32 records: eight candidates, four conditions each, n=1. Root adoption remains required; no formal100 or test was started.\n\n'+ '\n'.join('- '+x for x in limits)+'\n'
with (B/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write(text)
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')
assert not list(B.rglob('__pycache__')) and not list(F.rglob('__pycache__'))
members=[dict(path=p.relative_to(B).as_posix(),sha256=sha(p),size=p.stat().st_size) for p in sorted(B.rglob('*')) if p.is_file()]
save('FILES_SHA256.json',dict(status='SEALED_FINAL5_HYBRID32_DELIVERY_ROOT_PENDING',members=members,archive_members_verified=71,original_source_members_verified=69,raw_storage_index='RAW_STORAGE_INDEX.json',bulk_external_only=True))
assert all(sha(B/r['path'])==r['sha256'] for r in members)
print(json.dumps(dict(seal_sha256=sha(B/'FILES_SHA256.json'),sealed_members=len(members),handoff_sha256=sha(B/'HANDOFF.json'),ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),summary_sha256=sha(B/'SUMMARY32.json'),raw_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),raw_files=len(rows),archive_sha256=backup['archive_sha256'],receipt_sha256=sha(F/'BACKUP_SHA256.json'),offserver_sha256=ready['offserver_tensor_proof_sha256'],record_sha256=ready['offserver_original_record_check_sha256'],old=27,new=5,total=32)))
