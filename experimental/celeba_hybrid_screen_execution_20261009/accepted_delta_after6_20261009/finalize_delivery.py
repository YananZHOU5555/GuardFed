"""Seal the fixed four-ID evidence delivery without adopting or changing a chain."""
from pathlib import Path
import ast,hashlib,json,sys
B=Path(__file__).resolve().parent;H=B.parent;OLD=H/'accepted_delta_after4_20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(name,value):
 with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
assert not list(B.rglob('__pycache__')) and not any((B/name).exists() for name in ('FAILURE.json','LOCAL_RECORD_FAILURE.json'))
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')=='9be6240c0d28ffa973654cadb00ebe0abd7bfdabc52db34b2da355af19a6c807'
assert sha(H/'BACKUP_CHAIN_accepted_delta_after4_20261009.json')==sha(B/'PREVIOUS_CHAIN.json')=='231dda94d276174aaf488bb03aa71a8072b2bd0b4a0f63c241fa412e72d3fcd7'
delta=read(B/'EXACT_DELTA.json');previous=read(B/'PREVIOUS_CHAIN.json');wanted=delta['selected_ids'];old_wanted=read(OLD/'EXACT_DELTA.json')['selected_ids']
server=read(B/'PARTIAL_ACCEPTANCE.json');backup=read(B/'BACKUP_SHA256.json');offserver=read(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json');local=read(B/'LOCAL_RECORD_CHECKS.json')
assert set(server['accepted_new_ids'])==set(wanted)=={r['id'] for r in offserver['records']}=={r['id'] for r in local['records']}
assert server['old_accepted']==6 and server['accepted_new']==4 and server['accepted_total']==10 and len(wanted)==len(set(wanted))==4
assert not set(wanted).intersection(previous['accepted_job_ids']) and len(set(previous['accepted_job_ids']+wanted))==10
observed={r['id']:r for r in read(B/'AUTHORIZED_SNAPSHOT.json')['rows']}
assert all(r['acceptance_sha256']==observed[r['id']]['acceptance_sha256'] for r in server['records'])
scope=read(H/'screen_scope.json');selected=[e for e in scope['jobs'] if e['id'] in wanted];members=read(B/'MEMBERS.json')['members']
for name in members:
 assert name.startswith(B.name+'/') or any(name==e['job'] or name.startswith(e['output']+'/') for e in selected),name
assert all(not any(name.startswith(e['output']+'/') for name in members) for e in scope['jobs'] if e['id'] not in wanted)
for name,digest in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/name)==digest
source=(B/'collect_once.py').read_text(encoding='utf8')
fields={n.slice.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='previous' and isinstance(n.slice,ast.Constant)}
assert fields=={'accepted_job_ids','accepted_total'} and fields<=set(previous)
reverted=source.replace(repr(wanted),repr(old_wanted)).replace(sha(B/'PREVIOUS_CHAIN.json'),'5357d4ba81bbdf50964cb655fd19bb01eb6618a920dc0b2e683d9e4f25c014c2').replace(sha(B/'AUTHORIZED_SNAPSHOT.json'),'5bfe255142c9a6798f82a1331d83aaa45917b5e27781c4dac965028552257b84').replace("previous['accepted_total']==6","previous['accepted_total']==4").replace('old_accepted=6,accepted_total=6+len(wanted)','old_accepted=4,accepted_total=4+len(wanted)').replace('accepted_delta_after6_20261009','accepted_delta_after4_20261009').replace('hybrid_after6_delta.tar.gz','hybrid_after4_delta.tar.gz')
assert reverted==(OLD/'collect_once.py').read_text(encoding='utf8')
reverted=(B/'verify_offserver.py').read_text(encoding='utf8').replace('hybrid_after6_delta.tar.gz','hybrid_after4_delta.tar.gz').replace(backup['archive_sha256'],'261596ba4a4a11d5877630a436fc0649f857e3f0746263060bd944075bd6afa9').replace('==59','==35').replace("'member_count':59","'member_count':35").replace("'old_accepted':6","'old_accepted':4")
assert reverted==(OLD/'verify_offserver.py').read_text(encoding='utf8')
bridge=B/'local_record_bridge_v2';reverted=(bridge/'bridge.py').read_text(encoding='utf8').replace('hybrid_after6_delta.tar.gz','hybrid_after4_delta.tar.gz').replace(backup['archive_sha256'],'261596ba4a4a11d5877630a436fc0649f857e3f0746263060bd944075bd6afa9').replace(backup['acceptance_sha256'],'143a489bcb28dc5234ba275137daae944dd95e20f26723240e8d583b27baff41').replace(backup['inventory_sha256'],'20ea9d3a05368fd0aedd83172417d6b4102b9254f8c924ee7a3656467e64671e')
assert reverted==(OLD/'local_record_bridge_v2/bridge.py').read_text(encoding='utf8')
assert sha(bridge/'checked_record_body.py')==sha(OLD/'local_record_bridge_v2/checked_record_body.py')
sys.path.insert(0,str(bridge));import bridge as bound
try:bound.need_file(B/'PARTIAL_ACCEPTANCE.json','0'*64)
except AssertionError:wrong_hash_refused=True
else:raise AssertionError('Wrong evidence SHA was accepted')
save('SOURCE_REUSE_VERIFICATION.json',dict(status='EXACT_ORIGINAL_COLLECTOR_VERIFIERS_AFTER_BINDING_REVERSAL_PASS',
 previous_schema_fields=sorted(fields),previous_schema_fields_exist=True,no_schema_adapter_required=True,
 collector_binding_reversal_exact=True,verifier_binding_reversal_exact=True,bridge_binding_reversal_exact=True,
 checked_record_body_bytes_unchanged=True,scientific_three_runtime_query_inverse_and_AST_exact=True,
 original_driver_writer_and_null_policy_unchanged=True,wrong_evidence_sha_refused=wrong_hash_refused,
 original_source_members_verified=69,previous_latest_and_chain_unchanged=True))
ready=read(OLD/'ROOT_READY_CHAIN_LINK.json')
ready.update(previous_accepted=6,accepted_new=4,accepted_total_if_root_adopts=10,accepted_new_ids=wanted,
 accepted_job_ids=previous['accepted_job_ids']+wanted,previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),
 previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),
 archive_sha256=backup['archive_sha256'],archive_members=backup['member_count'],inventory_sha256=backup['inventory_sha256'],
 server_strict_sha256=backup['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),
 offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),source_seal_sha256=server['source_seal_sha256'],
 actual_server_runtime=server['runtime'],actual_local_torch=local['local_torch'],root_adoption_required=True,
 frozen_delta_only=True,remaining_if_root_adopts=22,source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'))
save('ROOT_READY_CHAIN_LINK.json',ready)
snapshot=read(B/'live_snapshot.json')
delivery=dict(status='SEALED_FIXED_FOUR_ID_DELTA_ROOT_ADOPTION_PENDING',previous_accepted=6,accepted_new=4,total_if_root_adopts=10,planned=32,
 accepted_new_ids=wanted,root_ready_chain_link_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),
 archive_sha256=backup['archive_sha256'],archive_size=backup['archive_size'],member_count=backup['member_count'],
 server_strict_sha256=backup['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),
 record_bridge_proof_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'),
 previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),
 source_seal_sha256=server['source_seal_sha256'],source_members=69,service_observation_utc=snapshot['utc'],service=snapshot['service'],
 backup_helper=dict(cpu=106,threads=1,nice=10,IO='idle'),original_training_role=dict(cpu=104,threads=1,gpu_index=0),
 observed_gpu=read(B/'AUTHORIZED_SNAPSHOT.json')['gpu'],
 server_runtime=server['runtime'],local_torch=local['local_torch'],local_runtime_not_equal=True,
 all_original_negative_results_and_undefined_sidecars_preserved=True,
 limitations=['Partial 10/32 only after root adoption; no completed four-condition candidate selection here.',
 'Shared still-running producer stdout prefix; no closed per-job log claim.',
 'CPU torch2.8.0+cpu is not server torch2.11.0+cu128/RTX5090 runtime replay; original runtime is bound to the actual SHA-pinned server receipt.',
 'Three original host introspection queries only use that receipt in the existing record bridge; scientific checked body reverses exactly and driver/writer/null semantics stay unchanged.',
 'n=1 validation search only; no sample SD/significance, selected recipe, test or formal100.',
 'Original metadata loader may materialize test partition metadata; no test image inference/fitting/selection is performed by this backup.'],
 no_CNN_or_training=True,no_old_model_repack=True,no_active_result_included=True,no_recipe_selection=True,
 original_latest_chain_canonical_state_and_Git_unchanged=True,SSH_transient_retries_used=0,automatic_numerical_or_logical_retry=False)
save('DELIVERY.json',delivery)
text='''# Hybrid fixed terminal delta: 6 -> 10, root adoption pending

Only the four IDs frozen in EXACT_DELTA.json are included. The 17:43:57Z source/scope/approval snapshot had ten 70-round terminals and the next job at round7. Original driver/body strict passed once, SCP returned zero, all 59 archive members passed raw SHA/size checks, and four CPU tensor identities plus the existing record bridge passed. Five wrong-runtime records and a wrong evidence hash were refused. No later arrival was added.

Collector, archive verifier and bridge differ from accepted after4 only in cohort/count/path/evidence bindings; reversing these changes restores each exact original source. The actual prior root-adopted chain has both previous fields used by the collector; no schema adapter was needed. Original 69 source members, scientific checked body, driver, writer/null policy, thresholds and recipes remain unchanged. All raw/negative outputs and undefined sidecars are retained.

ROOT_READY_CHAIN_LINK.json is a proposed link, not root adoption. DELIVERY.json lists actual hashes, resources and limits. The old LATEST_BACKUP and 231dda chain retain their exact bytes. Original source members are referenced, not repackaged in the delta archive. The local restored tree is a derived verification cache bound by the archive/source seals and is excluded from the delivery member seal.

Service resources were observed, not changed: training CPU104 x1/GPU0; backup CPU106 x1/nice10/idle IO. The log is a shared running-producer prefix. Server torch2.11.0+cu128/5090 and local torch2.8.0+cpu are unequal; only original runtime metadata is bound through the already adopted three-query record bridge. No CNN/training/test, new parameter, restart, recipe selection, formal100 or canonical/Git edit occurred. This is a partial n=1 validation search, without sample SD or significance.
'''
with (B/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write(text)
rows=[dict(path=p.relative_to(B).as_posix(),sha256=sha(p),size=p.stat().st_size) for p in sorted(B.rglob('*')) if p.is_file() and 'restored' not in p.relative_to(B).parts]
save('DELIVERY_FILES_SHA256.json',dict(members=rows,excluded_derived_tree='restored',source_and_raw_tree_bindings='Original69 source seal and59-member archive verified above'))
assert all(sha(B/row['path'])==row['sha256'] and (B/row['path']).stat().st_size==row['size'] for row in read(B/'DELIVERY_FILES_SHA256.json')['members'])
print(json.dumps(dict(status=delivery['status'],delivery_seal_sha256=sha(B/'DELIVERY_FILES_SHA256.json'),delivery_sha256=sha(B/'DELIVERY.json'),ready_link_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),sealed_files=len(rows),accepted_new_ids=wanted,total_if_root_adopts=10)))
