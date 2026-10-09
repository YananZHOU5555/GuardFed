from pathlib import Path
import ast,datetime,hashlib,json,re,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;H=B.parent;OLD=H/'accepted_delta_after14_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
 with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')=='d79b8136c8bbc29fdfda640bb99c7a94776d1165379dbdf3173f68a0e81730db'
assert sha(H/'BACKUP_CHAIN_accepted_delta_after14_20261009.json')==sha(B/'PREVIOUS_CHAIN.json')=='801c0f53211ee9c51d3666565660155a154dabbd6791c34d597a4aafb3fa3d99'
assert not list(B.rglob('__pycache__')) and not (B/'FAILURE.json').exists() and not (B/'LOCAL_RECORD_FAILURE.json').exists()
prior=read(B/'PREVIOUS_CHAIN.json');delta=read(B/'EXACT_DELTA.json');ids=delta['selected_ids'];server=read(B/'PARTIAL_ACCEPTANCE.json');backup=read(B/'BACKUP_SHA256.json');off=read(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json');local=read(B/'LOCAL_RECORD_CHECKS.json');snapshot=read(B/'AUTHORIZED_SNAPSHOT.json')
assert len(ids)==len(set(ids))==1 and server['old_accepted']==15 and server['accepted_total']==16
assert not set(ids)&set(prior['accepted_job_ids']) and len(set(prior['accepted_job_ids']+ids))==16
assert set(ids)==set(server['accepted_new_ids'])=={r['id'] for r in off['records']}=={r['id'] for r in local['records']}
observed={r['id']:r for r in snapshot['rows']}
assert all(r['acceptance_sha256']==observed[r['id']]['acceptance_sha256'] for r in server['records'])
assert snapshot['CPU110_all_thread_restricted_owners_free'] is True and delta['producer_quiescence_selected_ids'] is True
assert server['runtime']['CPU']==[110] and server['runtime']['threads']==1 and server['source_data_verified_before_after'] is True
assert off['different_host'] is True and off['local_CUDA_initialized'] is False and local['local_CUDA_initialized'] is False
scope=read(H/'screen_scope.json');selected=[e for e in scope['jobs'] if e['id'] in ids];members=read(B/'MEMBERS.json')['members']
for name in members:assert name.startswith(B.name+'/') or any(name==e['job'] or name.startswith(e['output']+'/') for e in selected),name
for name,pin in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/name)==pin
def loop(p):
 return next(n for n in ast.walk(ast.parse(p.read_text(encoding='utf8'))) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='e' and any(isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in v.targets) for v in n.body))
assert ast.dump(loop(B/'collect_once.py'),include_attributes=False)==ast.dump(loop(OLD/'collect_once.py'),include_attributes=False)
oldbackup=read(OLD/'BACKUP_SHA256.json')
source=(B/'verify_offserver.py').read_text(encoding='utf8').replace('hybrid_after15_delta.tar.gz','hybrid_after14_delta.tar.gz').replace(backup['archive_sha256'],oldbackup['archive_sha256']).replace("'old_accepted':15","'old_accepted':14")
assert source==(OLD/'verify_offserver.py').read_text(encoding='utf8')
bridge=B/'local_record_bridge_v2';source=(bridge/'bridge.py').read_text(encoding='utf8').replace('hybrid_after15_delta.tar.gz','hybrid_after14_delta.tar.gz')
for key in ('archive_sha256','acceptance_sha256','inventory_sha256'):source=source.replace(backup[key],oldbackup[key])
# Parent and this collector both use actual CPU110; no role substitution is needed.
assert source==(OLD/'local_record_bridge_v2/bridge.py').read_text(encoding='utf8')
assert sha(bridge/'checked_record_body.py')==sha(OLD/'local_record_bridge_v2/checked_record_body.py')
assert (B/'run_record_checks.py').read_text(encoding='utf8')==(OLD/'run_record_checks.py').read_text(encoding='utf8')
sys.path.insert(0,str(bridge));import bridge as bound
try:bound.need_file(B/'PARTIAL_ACCEPTANCE.json','0'*64)
except AssertionError:pass
else:raise RuntimeError('Bad SHA accepted')
source_check={'status':'ORIGINAL_SCIENTIFIC_LOOP_AND_RECORD_BODY_EXACT','scientific_loop_AST_exact':True,'verifier_binding_reversal_exact':True,'bridge_binding_and_CPU110_role_reversal_exact':True,'checked_record_body_bytes_exact':True,'writer_null_policy_unchanged':True,'runtime_refusals':local['runtime_refusals'],'wrong_SHA_refused':True,'source_members_verified':69,'canonical_chain_unchanged':True,'only_collector_changes':'Frozen parent15/count1/actual snapshot/new namespace/CPU110 and added terminal-producer guard; scientific per-terminal loop unchanged.','CUDA_metadata_path_unchanged':True}
save('SOURCE_REUSE_VERIFICATION.json',source_check)
ready=read(OLD/'ROOT_READY_CHAIN_LINK.json')
ready.update(previous_accepted=15,accepted_new=1,accepted_total_if_root_adopts=16,accepted_new_ids=ids,accepted_job_ids=prior['accepted_job_ids']+ids,previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),archive_sha256=backup['archive_sha256'],archive_members=backup['member_count'],inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),actual_server_runtime=server['runtime'],actual_local_torch=local['local_torch'],remaining_if_root_adopts=16,source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'),helper=dict(CPU=110,threads=1,nice=10,IO='idle'),previous_chain_file='BACKUP_CHAIN_accepted_delta_after14_20261009.json',delta_dir=B.name,archive='hybrid_after15_delta.tar.gz')
save('ROOT_READY_CHAIN_LINK.json',ready)
limits=['Fixed32 screen uses one seed91001, four conditions/candidate; partial16 is not a completed search, recipe selection, sample SD, significance or formal100 result.','Original GPU-trained/checkpoint provenance and server strict cu128/RTX5090 device-name metadata are retained. Local torch2.8cpu is a separate record/tensor check, not an equal-runtime reproduction.','Original Hybrid archives supply native_replay.json and checkpoint tensors, not saved prediction/probability arrays; this increment makes no independent array-derived metric-recomputation claim. Original strict result/native-replay equality and writer/null checks passed.','The shared active service log is a labelled prefix; selected result/model/config bytes are closed and SHA-stable. No later arrival is included.']
save('DELIVERY.json',{'status':'EXACT1_SERVER_STRICT_OFFSERVER_TENSORS_RECORD_CHECKS_PASS_ROOT_PENDING','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_accepted':15,'accepted_new':1,'accepted_total_if_root_adopts':16,'planned':32,'selected_ids':ids,'snapshot_utc':snapshot['utc'],'snapshot_terminal':sum(r['terminal'] for r in snapshot['rows']),'snapshot_active_rounds':[r['round'] for r in snapshot['rows'] if r['round'] and not r['terminal']],'snapshot_cpu_quota':snapshot['cpu_quota'],'snapshot_nominal_threads':snapshot['nominal_threads'],'archive_sha256':backup['archive_sha256'],'archive_members':backup['member_count'],'archive_bytes':backup['archive_size'],'server_strict_sha256':backup['acceptance_sha256'],'offserver_proof_sha256':sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),'record_bridge_sha256':sha(B/'LOCAL_RECORD_CHECKS.json'),'root_ready_sha256':sha(B/'ROOT_READY_CHAIN_LINK.json'),'source_members_not_repackaged':69,'server_runtime':server['runtime'],'local_runtime':local['local_torch'],'shared_log_prefix_only':True,'canonical_latest_chain_unchanged':True,'new_CNN_training_test':False,'no_recipe_selection':True,'no_later_arrivals':True,'automatic_retry':False,'nulls_and_negative_results_preserved':True,'limits':limits})
text=f"""# Hybrid fixed delta15 to16 — root adoption pending

One source-bound observation at {snapshot['utc']} contained{sum(r['terminal'] for r in snapshot['rows'])} terminals; active rounds were {[r['round'] for r in snapshot['rows'] if r['round'] and not r['terminal']]}. Only the ID in EXACT_DELTA.json was frozen: {ids[0]}. No waiting for a larger batch or inclusion of later arrivals occurred.

The original server strict loop ran once onCPU110, one thread, nice10/idleIO after an all-thread restricted-CPU ownership check; original source/data checks passed before and after. Root explicitly retained CUDA_VISIBLE_DEVICES=0 for the original device-name metadata query. No CNN, CUDA tensor computation, training or test inference was run. All23 archive members, one checkpoint tensor identity and the previously accepted original-science record bridge passed; five runtime mutations and one wrong-SHA mutation were rejected.

ROOT_READY_CHAIN_LINK.json is proposed evidence for root adoption. LATEST and the prior841f0a5a chain remain byte-unchanged. The original69 source members and old15 models were referenced, not repackaged. Original null/undefined sidecars and all negative measurements remain. The shared running log is a labelled prefix.

"""
text+='\n'.join('- '+x for x in limits)+'\n'
(B/'README.md').write_text(text,encoding='utf8',newline='\n')
rows=[{'path':p.relative_to(B).as_posix(),'sha256':sha(p),'size':p.stat().st_size} for p in sorted(B.rglob('*')) if p.is_file() and 'restored' not in p.relative_to(B).parts]
save('DELIVERY_FILES_SHA256.json',{'members':rows,'excluded_derived_tree':'restored','archive_members_verified':23,'original_source_members_verified':69})
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json') and sha(H/'BACKUP_CHAIN_accepted_delta_after14_20261009.json')==sha(B/'PREVIOUS_CHAIN.json')
print(json.dumps({'status':'EXACT1_DELIVERED_ROOT_ADOPTION_PENDING','ready_sha256':sha(B/'ROOT_READY_CHAIN_LINK.json'),'delivery_sha256':sha(B/'DELIVERY.json'),'seal_sha256':sha(B/'DELIVERY_FILES_SHA256.json'),'sealed_files':len(rows),'archive_sha256':backup['archive_sha256'],'server_strict_sha256':backup['acceptance_sha256'],'offserver_proof_sha256':sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),'record_bridge_sha256':sha(B/'LOCAL_RECORD_CHECKS.json'),'old':15,'new':1,'total_if_root_adopts':16,'source69_unchanged':True,'LATEST_chain_unchanged':True},indent=2))
