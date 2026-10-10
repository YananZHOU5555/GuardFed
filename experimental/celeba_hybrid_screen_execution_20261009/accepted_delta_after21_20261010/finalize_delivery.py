from pathlib import Path
import ast,datetime,hashlib,json,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;O=B.parent/'celeba_hybrid_delta_after19_20261010';H=B.parent/'celeba_hybrid_screen_execution_20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(name,value):
 with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
prior=read(B/'PREVIOUS_CHAIN.json');latest=read(B/'PREVIOUS_LATEST.json');delta=read(B/'EXACT_DELTA.json');ids=delta['selected_ids']
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')
assert sha(H/latest['chain_file'])==sha(B/'PREVIOUS_CHAIN.json')==latest['chain_sha256']
assert not list(B.rglob('__pycache__')) and not(B/'FAILURE.json').exists() and not(B/'LOCAL_RECORD_FAILURE.json').exists()
server=read(B/'PARTIAL_ACCEPTANCE.json');backup=read(B/'BACKUP_SHA256.json');off=read(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json');local=read(B/'LOCAL_RECORD_CHECKS.json');snapshot=read(B/'AUTHORIZED_SNAPSHOT.json')
assert len(ids)==len(set(ids))==1 and prior['accepted_total']==server['old_accepted']==21 and server['accepted_total']==22
assert not set(ids)&set(prior['accepted_job_ids']) and len(set(prior['accepted_job_ids']+ids))==22
assert set(ids)==set(server['accepted_new_ids'])=={r['id'] for r in off['records']}=={r['id'] for r in local['records']}
observed={r['id']:r for r in snapshot['rows']}
assert all(r['acceptance_sha256']==observed[r['id']]['acceptance_sha256'] for r in server['records'])
assert snapshot['CPU106_all_thread_restricted_owners_free'] and delta['producer_quiescence_selected_ids']
assert server['runtime']['CPU']==[106] and server['runtime']['threads']==1 and server['source_data_verified_before_after']
assert off['different_host'] and off['local_CUDA_initialized'] is False and local['local_CUDA_initialized'] is False
assert read(B/'CPU_RELEASE.json')['CPU106_released']
assert read(B/'SCP_RECEIPT.json')['exit_code']==read(B/'REMOTE_COLLECT_RECEIPT.json')['exit_code']==0
scope=read(H/'screen_scope.json');selected=[e for e in scope['jobs'] if e['id'] in ids];members=read(B/'MEMBERS.json')['members']
for name in members:assert name.startswith('accepted_delta_after21_20261010/') or any(name==e['job'] or name.startswith(e['output']+'/') for e in selected),name
for name,pin in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/name)==pin
# Reverse exactly the recorded collector bindings; scientific loop and every other source byte are original.
source=(B/'collect_once.py').read_text('utf8')
for a,b in reversed(read(B/'SOURCE_RECEIPT.json')['replacements']):assert b in source;source=source.replace(b,a)
assert source==(O/'collect_once.py').read_text('utf8')
old=read(O/'BACKUP_SHA256.json')
source=(B/'verify_offserver.py').read_text('utf8').replace('hybrid_after21_delta.tar.gz','hybrid_after19_delta.tar.gz').replace(backup['archive_sha256'],old['archive_sha256']).replace("'old_accepted':21","'old_accepted':19").replace('==23 and set(t.getnames())','==35 and set(t.getnames())').replace("'member_count':23","'member_count':35")
assert source==(O/'verify_offserver.py').read_text('utf8')
bridge=B/'local_record_bridge_v2';source=(bridge/'bridge.py').read_text('utf8').replace('hybrid_after21_delta.tar.gz','hybrid_after19_delta.tar.gz')
for key in ('archive_sha256','acceptance_sha256','inventory_sha256'):source=source.replace(backup[key],old[key])
assert source==(O/'local_record_bridge_v2/bridge.py').read_text('utf8')
for name in ['checked_record_body.py','SOURCE_REUSE.json']:assert sha(bridge/name)==sha(O/'local_record_bridge_v2'/name)
assert (B/'run_record_checks.py').read_text('utf8').replace('len(expected)==1','len(expected)==2')==(O/'run_record_checks.py').read_text('utf8')
sys.path.insert(0,str(bridge));import bridge as bound
try:bound.need_file(B/'PARTIAL_ACCEPTANCE.json','0'*64)
except AssertionError:pass
else:raise RuntimeError('Bad SHA accepted')
save('SOURCE_REUSE_VERIFICATION.json',dict(status='ORIGINAL_SCIENTIFIC_LOOP_AND_RECORD_BODY_EXACT',collector_metadata_reversal_byte_exact=True,verifier_binding_reversal_exact=True,bridge_binding_reversal_exact=True,checked_record_body_bytes_exact=True,writer_null_policy_unchanged=True,runtime_refusals=local['runtime_refusals'],wrong_SHA_refused=True,source_members_verified=69,canonical_chain_unchanged=True,only_changes='Actual previous21/new1/snapshot/namespace and bound actual receipt hashes. Original strict scientific loop and body are unchanged.'))
ready=read(O/'ROOT_READY_CHAIN_LINK.json')
ready.update(previous_accepted=21,accepted_new=1,accepted_total_if_root_adopts=22,accepted_new_ids=ids,accepted_job_ids=prior['accepted_job_ids']+ids,previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),previous_offserver_proof_sha256=prior['offserver_proof_sha256'],previous_root_adoption_sha256=prior['root_adoption_sha256'],authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),archive_sha256=backup['archive_sha256'],archive_members=backup['member_count'],inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),actual_server_runtime=server['runtime'],actual_local_torch=local['local_torch'],remaining_if_root_adopts=10,source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'),previous_chain_file=latest['chain_file'],delta_dir='accepted_delta_after21_20261010',archive='hybrid_after21_delta.tar.gz',owned_delivery_directory=B.as_posix(),root_copy_target=(H/'accepted_delta_after21_20261010').as_posix(),CPU106_release_sha256=sha(B/'CPU_RELEASE.json'))
save('ROOT_READY_CHAIN_LINK.json',ready)
limits=['Partial22/32 valid-only screen, seed91001; no recipe selection, formal100, test or significance claim.','Actual server original strict used torch2.11cu128/RTX5090 metadata; local torch2.8cpu record and tensor checks are disclosed separately, not equal-runtime reproduction.','No prediction arrays or independent saved-array metric recomputation are claimed. Original result/native replay/writer/null checks passed.','Shared active-service log is a labelled prefix. Only frozen exact1 terminal model/result/config included; source69 and previous21 models are referenced, not repackaged.','Initial local snapshot assembly used the wrong field name accepted_ids; exception retained, corrected to actual accepted_job_ids using the same successful snapshot. No collector or scientific retry.','Root independent adoption remains required; no shared LATEST/STATE/RUNNING/Git/service modifications.']
delivery=dict(status='EXACT1_SERVER_STRICT_OFFSERVER_TENSORS_RECORD_CHECKS_PASS_ROOT_PENDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_accepted=21,new_accepted=1,total_if_root_adopts=22,planned=32,selected_ids=ids,snapshot_utc=snapshot['utc'],snapshot_terminal=sum(r['terminal'] for r in snapshot['rows']),snapshot_active_rounds=[r['round'] for r in snapshot['rows'] if r['round'] and not r['terminal']],archive_sha256=backup['archive_sha256'],archive_members=backup['member_count'],archive_inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_bridge_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),root_ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),backup_receipt_sha256=sha(B/'BACKUP_SHA256.json'),CPU106_release_sha256=sha(B/'CPU_RELEASE.json'),source69_not_repackaged=True,oldmodels_not_repackaged=True,canonical_chain_unchanged=True,no_CNN_train_test=True,automatic_retry=False,limits=limits)
save('DELIVERY.json',delivery)
with (B/'README.md').open('x',encoding='utf8') as f:f.write('# Hybrid exact delta 21 to 22 — root adoption pending\n\n'+ '\n'.join('- '+x for x in limits)+'\n\nNew ID: '+ids[0]+'\n')
rows=[dict(path=p.relative_to(B).as_posix(),sha256=sha(p),size=p.stat().st_size) for p in sorted(B.rglob('*')) if p.is_file() and 'restored' not in p.relative_to(B).parts]
save('DELIVERY_FILES_SHA256.json',dict(members=rows,excluded_derived_tree='restored',archive_members_verified=23,original_source_members_verified=69))
for row in rows:assert sha(B/row['path'])==row['sha256']
print(json.dumps(dict(status=delivery['status'],seal_sha256=sha(B/'DELIVERY_FILES_SHA256.json'),sealed_members=len(rows),handoff_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),delivery_sha256=sha(B/'DELIVERY.json'),archive_sha256=backup['archive_sha256'],receipt_sha256=sha(B/'BACKUP_SHA256.json'),offserver_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),CPU_release_sha256=sha(B/'CPU_RELEASE.json')),indent=2))
