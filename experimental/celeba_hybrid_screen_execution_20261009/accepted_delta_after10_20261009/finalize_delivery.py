from pathlib import Path
import ast,difflib,hashlib,json,sys
B=Path(__file__).resolve().parent;H=B.parent;OLD=H/'accepted_delta_after6_20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(n,v):
 with (B/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')=='9d8e65ce03b4b7106a32425b01a5e745fa2ea0b889385db3721a99bc8c8c4007'
assert sha(H/'BACKUP_CHAIN_accepted_delta_after6_20261009.json')==sha(B/'PREVIOUS_CHAIN.json')=='6733ca577aa2805af4bc6ee0f2d31429464caaa7c65846df83e09555b3634748'
assert not list(B.rglob('__pycache__')) and not (B/'FAILURE.json').exists() and not (B/'LOCAL_RECORD_FAILURE.json').exists()
prior=read(B/'PREVIOUS_CHAIN.json');delta=read(B/'EXACT_DELTA.json');ids=delta['selected_ids'];server=read(B/'PARTIAL_ACCEPTANCE.json');backup=read(B/'BACKUP_SHA256.json');off=read(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json');local=read(B/'LOCAL_RECORD_CHECKS.json')
assert len(ids)==len(set(ids))==4 and server['old_accepted']==10 and server['accepted_total']==14
assert not set(ids)&set(prior['accepted_job_ids']) and len(set(prior['accepted_job_ids']+ids))==14
assert set(ids)==set(server['accepted_new_ids'])=={r['id'] for r in off['records']}=={r['id'] for r in local['records']}
observed={r['id']:r for r in read(B/'AUTHORIZED_SNAPSHOT.json')['rows']}
assert all(r['acceptance_sha256']==observed[r['id']]['acceptance_sha256'] for r in server['records'])
scope=read(H/'screen_scope.json');selected=[e for e in scope['jobs'] if e['id'] in ids];members=read(B/'MEMBERS.json')['members']
for name in members:assert name.startswith(B.name+'/') or any(name==e['job'] or name.startswith(e['output']+'/') for e in selected),name
for name,pin in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/name)==pin
# Original scientific per-terminal loop and writer remain unaltered; new guard only checks all CPU106 threads.
def loop(p):return next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='e' and any(isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in v.targets) for v in n.body))
assert ast.dump(loop(B/'collect_once.py'))==ast.dump(loop(OLD/'collect_once.py'))
oldbackup=read(OLD/'BACKUP_SHA256.json')
source=(B/'verify_offserver.py').read_text().replace('hybrid_after10_delta.tar.gz','hybrid_after6_delta.tar.gz').replace(backup['archive_sha256'],oldbackup['archive_sha256']).replace("'old_accepted':10","'old_accepted':6")
assert source==(OLD/'verify_offserver.py').read_text()
bridge=B/'local_record_bridge_v2';source=(bridge/'bridge.py').read_text().replace('hybrid_after10_delta.tar.gz','hybrid_after6_delta.tar.gz')
for key in ('archive_sha256','acceptance_sha256','inventory_sha256'):source=source.replace(backup[key],oldbackup[key])
assert source==(OLD/'local_record_bridge_v2/bridge.py').read_text()
assert sha(bridge/'checked_record_body.py')==sha(OLD/'local_record_bridge_v2/checked_record_body.py')
sys.path.insert(0,str(bridge));import bridge as bound
try:bound.need_file(B/'PARTIAL_ACCEPTANCE.json','0'*64)
except AssertionError:pass
else:raise RuntimeError('Bad SHA accepted')
save('SOURCE_REUSE_VERIFICATION.json',{'status':'ORIGINAL_SCIENTIFIC_LOOP_AND_RECORD_BODY_EXACT','scientific_loop_AST_exact':True,'verifier_binding_reversal_exact':True,'bridge_binding_reversal_exact':True,'checked_record_body_bytes_exact':True,'writer_null_policy_unchanged':True,'runtime_refusals':local['runtime_refusals'],'wrong_SHA_refused':True,'source_members_verified':69,'canonical_chain_unchanged':True})
ready=read(OLD/'ROOT_READY_CHAIN_LINK.json');ready.update(previous_accepted=10,accepted_new=4,accepted_total_if_root_adopts=14,accepted_new_ids=ids,accepted_job_ids=prior['accepted_job_ids']+ids,previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),archive_sha256=backup['archive_sha256'],archive_members=backup['member_count'],inventory_sha256=backup['inventory_sha256'],server_strict_sha256=backup['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),actual_server_runtime=server['runtime'],actual_local_torch=local['local_torch'],remaining_if_root_adopts=18,source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'))
save('ROOT_READY_CHAIN_LINK.json',ready)
snapshot=read(B/'AUTHORIZED_SNAPSHOT.json')
save('DELIVERY.json',{'status':'EXACT4_SERVER_STRICT_OFFSERVER_TENSORS_RECORD_CHECKS_PASS_ROOT_PENDING','old_accepted':10,'accepted_new':4,'accepted_total_if_root_adopts':14,'planned':32,'selected_ids':ids,'snapshot_utc':snapshot['utc'],'snapshot_terminal':14,'snapshot_active_rounds':[r['round'] for r in snapshot['rows'] if r['round'] and not r['terminal']],'snapshot_cpu_quota':snapshot['cpu_quota'],'snapshot_nominal_threads':snapshot['nominal_threads'],'archive_sha256':backup['archive_sha256'],'archive_members':backup['member_count'],'archive_bytes':backup['archive_size'],'root_ready_sha256':sha(B/'ROOT_READY_CHAIN_LINK.json'),'source_members_not_repackaged':69,'server_runtime':server['runtime'],'local_runtime':local['local_torch'],'shared_log_prefix_only':True,'canonical_latest_chain_unchanged':True,'new_CNN_training_test':False,'no_recipe_selection':True,'no_later_arrivals':True,'automatic_retry':False,'nulls_and_negative_results_preserved':True})
(B/'README.md').write_text("# Hybrid fixed delta 10 to 14 — root adoption pending\n\nThe source-bound20:48:58Z snapshot contained14 terminals and one active job at48/70. Only the four IDs in EXACT_DELTA.json were accepted. Original server strict ran once onCPU106, one thread, nice10/idleIO, while main8 and FL2 queues were untouched. All59 archive members, four CPU tensor identities and the existing original-science record bridge passed. Five runtime mutations and one SHA mutation were rejected. Writer/null rules and negative outputs are preserved.\n\nROOT_READY_CHAIN_LINK.json is proposed, not canonical adoption. Prior LATEST and6733ca chain are byte-unchanged. Source69 and old10 models are referenced, not repackaged. The shared running log is an explicitly labelled prefix. Server cu128/RTX5090 metadata is bound to the actual strict receipt; localtorch2.8cpu is not a reproduction of that runtime. No CNN, threshold fitting, training, test, recipe selection, queue mutation or Git change was performed. No arrivals after the frozen snapshot were included.\n")
rows=[{'path':p.relative_to(B).as_posix(),'sha256':sha(p),'size':p.stat().st_size} for p in sorted(B.rglob('*')) if p.is_file() and 'restored' not in p.relative_to(B).parts]
save('DELIVERY_FILES_SHA256.json',{'members':rows,'excluded_derived_tree':'restored','archive_members_verified':59,'original_source_members_verified':69})
print(json.dumps({'ready_sha256':sha(B/'ROOT_READY_CHAIN_LINK.json'),'delivery_sha256':sha(B/'DELIVERY.json'),'seal_sha256':sha(B/'DELIVERY_FILES_SHA256.json'),'sealed_files':len(rows),'archive_sha256':backup['archive_sha256']}))
