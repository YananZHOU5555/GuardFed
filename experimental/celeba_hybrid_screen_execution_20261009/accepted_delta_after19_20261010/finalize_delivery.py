from pathlib import Path
import ast,datetime,hashlib,json,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
H=Path('E:/OneDrive/文档/GuardFed/tmp/celeba_hybrid_screen_execution_20261009')
OLD=H/'accepted_delta_after18_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
 with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')=='7074d0f7c0b62bb2e2ddcc6a59900ee1590679b146b5cc6479ec15d7a56cb58c'
assert sha(H/'BACKUP_CHAIN_accepted_delta_after18_20261010.json')==sha(B/'PREVIOUS_CHAIN.json')=='da17267a5f543d566b1e6179a376d24c71abb34c415fe251c0da9f1863941b92'
assert not list(B.rglob('__pycache__')) and not (B/'FAILURE.json').exists() and not (B/'LOCAL_RECORD_FAILURE.json').exists()
prior=read(B/'PREVIOUS_CHAIN.json');delta=read(B/'EXACT_DELTA.json');ids=delta['selected_ids']
server=read(B/'PARTIAL_ACCEPTANCE.json');backup=read(B/'BACKUP_SHA256.json');off=read(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json');local=read(B/'LOCAL_RECORD_CHECKS.json');snapshot=read(B/'AUTHORIZED_SNAPSHOT.json')
assert len(ids)==len(set(ids))==2 and prior['accepted_total']==server['old_accepted']==19 and server['accepted_total']==21
assert not set(ids)&set(prior['accepted_job_ids']) and len(set(prior['accepted_job_ids']+ids))==21
assert set(ids)==set(server['accepted_new_ids'])=={r['id'] for r in off['records']}=={r['id'] for r in local['records']}
observed={r['id']:r for r in snapshot['rows']}
assert all(r['acceptance_sha256']==observed[r['id']]['acceptance_sha256'] for r in server['records'])
assert snapshot['CPU106_all_thread_restricted_owners_free'] and delta['producer_quiescence_selected_ids']
assert server['runtime']['CPU']==[106] and server['runtime']['threads']==1 and server['source_data_verified_before_after']
assert off['different_host'] and off['local_CUDA_initialized'] is False and local['local_CUDA_initialized'] is False
scope=read(H/'screen_scope.json');selected=[e for e in scope['jobs'] if e['id'] in ids];members=read(B/'MEMBERS.json')['members']
remote_prefix='accepted_delta_after19_20261010/'
for name in members:assert name.startswith(remote_prefix) or any(name==e['job'] or name.startswith(e['output']+'/') for e in selected),name
for name,pin in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/name)==pin
def loop(p):
 return next(n for n in ast.walk(ast.parse(p.read_text(encoding='utf8'))) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='e' and any(isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in v.targets) for v in n.body))
assert ast.dump(loop(B/'collect_once.py'),include_attributes=False)==ast.dump(loop(OLD/'collect_once.py'),include_attributes=False)
oldbackup=read(OLD/'BACKUP_SHA256.json')
hbinding="H=Path("+repr(H.as_posix())+")"
source=(B/'verify_offserver.py').read_text(encoding='utf8').replace('hybrid_after19_delta.tar.gz','hybrid_after18_delta.tar.gz').replace(backup['archive_sha256'],oldbackup['archive_sha256']).replace("'old_accepted':19","'old_accepted':18").replace('==35','==23').replace("'member_count':35","'member_count':23")
assert source==(OLD/'verify_offserver.py').read_text(encoding='utf8')
bridge=B/'local_record_bridge_v2'
source=(bridge/'bridge.py').read_text(encoding='utf8').replace('hybrid_after19_delta.tar.gz','hybrid_after18_delta.tar.gz').replace("r['CPU']==[106]","r['CPU']==[110]")
for key in ('archive_sha256','acceptance_sha256','inventory_sha256'):source=source.replace(backup[key],oldbackup[key])
assert source==(OLD/'local_record_bridge_v2/bridge.py').read_text(encoding='utf8')
assert sha(bridge/'checked_record_body.py')==sha(OLD/'local_record_bridge_v2/checked_record_body.py')
assert sha(bridge/'SOURCE_REUSE.json')==sha(OLD/'local_record_bridge_v2/SOURCE_REUSE.json')
assert (B/'run_record_checks.py').read_text('utf8').replace("len(proof['records'])==len(expected)==2","len(proof['records'])==len(expected)==1")==(OLD/'run_record_checks.py').read_text('utf8')
sys.path.insert(0,str(bridge));import bridge as bound
try:bound.need_file(B/'PARTIAL_ACCEPTANCE.json','0'*64)
except AssertionError:pass
else:raise RuntimeError('Bad SHA accepted')
source_check=dict(status='ORIGINAL_SCIENTIFIC_LOOP_AND_RECORD_BODY_EXACT',scientific_loop_AST_exact=True,verifier_binding_reversal_exact=True,bridge_binding_and_CPU106_role_reversal_exact=True,checked_record_body_bytes_exact=True,writer_null_policy_unchanged=True,runtime_refusals=local['runtime_refusals'],wrong_SHA_refused=True,source_members_verified=69,canonical_chain_unchanged=True,only_collector_changes='Frozen actual parent19/count2/snapshot/new output namespace; record-only CPU role110 to106. Inherited producer/all-thread guards and original scientific per-terminal loop unchanged.',local_only_path_adapter='Owned output is outside the original execution base, so local verifier/record bridge H explicitly binds the source69 directory. Reversing only this path and actual receipt/archive hashes recovers the parent sources byte-exactly.',CUDA_metadata_path_unchanged=True)
save('SOURCE_REUSE_VERIFICATION.json',source_check)
ready=read(OLD/'ROOT_READY_CHAIN_LINK.json')
ready.update(previous_accepted=19,accepted_new=2,accepted_total_if_root_adopts=21,accepted_new_ids=ids,accepted_job_ids=prior['accepted_job_ids']+ids,previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),previous_offserver_proof_sha256=prior['offserver_proof_sha256'],previous_root_adoption_sha256=prior['root_adoption_sha256'],authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),archive_sha256=backup['archive_sha256'],archive_members=backup['member_count'],inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),actual_server_runtime=server['runtime'],actual_local_torch=local['local_torch'],remaining_if_root_adopts=11,source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'),helper=dict(CPU=106,threads=1,nice=10,IO='idle'),previous_chain_file='BACKUP_CHAIN_accepted_delta_after18_20261010.json',delta_dir='accepted_delta_after19_20261010',archive='hybrid_after19_delta.tar.gz',owned_delivery_directory=B.as_posix(),root_copy_required_before_adoption=True,root_copy_target=(H/'accepted_delta_after19_20261010').as_posix(),prediction_arrays_supplied=False,independent_array_metric_recompute_claim=False)
save('ROOT_READY_CHAIN_LINK.json',ready)
limits=['Fixed32 screen uses one seed91001 and four conditions per candidate. Partial21 is not a complete search, recipe selection, across-seed sample SD, significance or formal100 result.','Original GPU-trained provenance and server strict cu128/RTX5090 device-name metadata are retained. Local torch2.8cpu is a separate record/tensor check, not an equal-runtime reproduction.','Archives provide native_replay.json and checkpoint tensors, not prediction/probability arrays. No independent saved-array metric recomputation is claimed. Original strict result/native-replay equality and writer/null checks passed.','Shared active service log is a labelled prefix; selected result/model/config bytes are terminal and SHA-stable. No later arrival is included.','Root adoption/copy is still required. LATEST, canonical chain, STATE, RUNNING, Git, service, queue and recipe are unchanged.']
delivery=dict(status='EXACT2_SERVER_STRICT_OFFSERVER_TENSORS_RECORD_CHECKS_PASS_ROOT_PENDING',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_accepted=19,accepted_new=2,accepted_total_if_root_adopts=21,planned=32,selected_ids=ids,snapshot_utc=snapshot['utc'],snapshot_terminal=sum(r['terminal'] for r in snapshot['rows']),snapshot_active_rounds=[r['round'] for r in snapshot['rows'] if r['round'] and not r['terminal']],snapshot_service=snapshot['service'],snapshot_cpu_quota=snapshot['cpu_quota'],snapshot_nominal_threads=snapshot['nominal_threads'],archive_sha256=backup['archive_sha256'],archive_members=backup['member_count'],archive_bytes=backup['archive_size'],archive_inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_bridge_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),root_ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),backup_receipt_sha256=sha(B/'BACKUP_SHA256.json'),remote_collector_receipt_sha256=sha(B/'REMOTE_COLLECT_RECEIPT.json'),source_members_not_repackaged=69,server_runtime=server['runtime'],local_runtime=local['local_torch'],shared_log_prefix_only=True,canonical_latest_chain_unchanged=True,new_CNN_training_test=False,no_recipe_selection=True,no_later_arrivals=True,automatic_retry=False,nulls_and_negative_results_preserved=True,limits=limits)
save('DELIVERY.json',delivery)
text=f"""# Hybrid fixed delta19 to21 — root adoption pending

One frozen source-bound snapshot at {snapshot['utc']} contained21 terminal jobs. New exact2 IDs: {', '.join(ids)}. Other future tasks remain pending; no later arrivals or recipe selection are included.

Original strict ran once on CPU106/one thread/nice10/idleIO after all-thread ownership and selected-producer quiescence checks. Original source/data checks passed before/after. Original CUDA_VISIBLE_DEVICES=0 is retained only for device-name metadata, without CNN or CUDA tensor computation. All35 members, both checkpoint tensors and the original scientific record bridge passed; five wrong-runtime mutations and one wrong-SHA mutation were refused.

Actual root-accepted19 parent is {sha(B/'PREVIOUS_CHAIN.json')}. Original accepted_total/accepted_job_ids schema is used. Only exact parent/count/IDs/snapshot/output and record-only CPU role110→106 were rebound; scientific per-terminal loop and checked_record_body bytes are unchanged. Local torch2.8cpu is distinct from server cu128 and original GPU-trained provenance. No prediction arrays or independent saved-array metric recomputation are claimed.

One auxiliary preparation assertion counted digits in SHA text; it failed before writing/running local verifier, is preserved in PREPARATION_ASSERTION_FAILURE.json, and was corrected to exact member-count field checks. Successful server strict was never retried; local verifier and bridge each ran once.

ROOT_READY_CHAIN_LINK.json proposes21/32 acceptance, pending root independent review and copy into original base/accepted_delta_after19_20261010. Old19 models/source69 are referenced, not repackaged; derived restored/ is excluded. LATEST/chain/STATE/RUNNING/Git/service/queue remain unchanged.

"""
text+='\n'.join('- '+x for x in limits)+'\n'
with (B/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write(text)
rows=[dict(path=p.relative_to(B).as_posix(),sha256=sha(p),size=p.stat().st_size) for p in sorted(B.rglob('*')) if p.is_file() and 'restored' not in p.relative_to(B).parts]
save('DELIVERY_FILES_SHA256.json',dict(members=rows,excluded_derived_tree='restored',archive_members_verified=35,original_source_members_verified=69))
for row in rows:assert sha(B/row['path'])==row['sha256']
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json') and sha(H/'BACKUP_CHAIN_accepted_delta_after18_20261010.json')==sha(B/'PREVIOUS_CHAIN.json')
print(json.dumps(dict(status='EXACT2_DELIVERED_ROOT_ADOPTION_PENDING',ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),delivery_sha256=sha(B/'DELIVERY.json'),seal_sha256=sha(B/'DELIVERY_FILES_SHA256.json'),sealed_files=len(rows),archive_sha256=backup['archive_sha256'],archive_inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],backup_receipt_sha256=sha(B/'BACKUP_SHA256.json'),remote_receipt_sha256=sha(B/'REMOTE_COLLECT_RECEIPT.json'),offserver_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_bridge_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),old=19,new=2,total_if_root_adopts=21,source69_unchanged=True,LATEST_chain_unchanged=True),indent=2))
