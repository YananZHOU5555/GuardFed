"""Close the one actual delta using existing strict/member/tensor/record receipts. No shared writes."""
from pathlib import Path
import datetime,hashlib,json,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;O=B.parent/'celeba_hybrid_delta_after21_20261010';H=B.parent/'celeba_hybrid_screen_execution_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(n,d):
 with (B/n).open('x',encoding='utf8',newline='\n') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
prior=read(B/'PREVIOUS_CHAIN.json');latest=read(B/'PREVIOUS_LATEST.json');d=read(B/'EXACT_DELTA.json');ids=d['selected_ids'];a=read(B/'PARTIAL_ACCEPTANCE.json');bk=read(B/'BACKUP_SHA256.json');off=read(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json');rec=read(B/'LOCAL_RECORD_CHECKS.json');snapshot=read(B/'AUTHORIZED_SNAPSHOT.json')
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json') and sha(H/latest['chain_file'])==sha(B/'PREVIOUS_CHAIN.json')==latest['chain_sha256']
assert prior['accepted_total']==a['old_accepted']==22 and a['accepted_total']==23 and len(ids)==len(set(ids))==1
assert not set(ids)&set(prior['accepted_job_ids']) and ids==a['accepted_new_ids']==bk['accepted_new_ids']
assert set(ids)=={r['id'] for r in off['records']}=={r['id'] for r in rec['records']}
assert read(B/'SCP_RECEIPT.json')['exit_code']==read(B/'REMOTE_COLLECT_RECEIPT.json')['exit_code']==0 and read(B/'CPU_RELEASE.json')['CPU106_released']
assert a['source_data_verified_before_after'] and a['runtime']['CPU']==[106] and a['runtime']['threads']==1 and not off['local_CUDA_initialized'] and not rec['local_CUDA_initialized']
assert bk['member_count']==23 and bk['source_not_repackaged'] and bk['original_strict_server_PASS']
observed={r['id']:r for r in snapshot['rows']};assert all(r['acceptance_sha256']==observed[r['id']]['acceptance_sha256'] for r in a['records'])
selected=[e for e in read(H/'screen_scope.json')['jobs'] if e['id'] in ids]
for n in read(B/'MEMBERS.json')['members']:assert n.startswith('accepted_delta_after22_20261010/') or any(n==e['job'] or n.startswith(e['output']+'/') for e in selected)
for n,p in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/n)==p
s=(B/'collect_once.py').read_text('utf8')
for old,new in reversed(read(B/'SOURCE_RECEIPT.json')['replacements']):s=s.replace(new,old)
assert s==(O/'collect_once.py').read_text('utf8')
ob=read(O/'BACKUP_SHA256.json');s=(B/'verify_offserver.py').read_text('utf8').replace('hybrid_after22_delta','hybrid_after21_delta').replace(bk['archive_sha256'],ob['archive_sha256']).replace("'old_accepted':22","'old_accepted':21")
assert s==(O/'verify_offserver.py').read_text('utf8')
s=(B/'local_record_bridge_v2/bridge.py').read_text('utf8').replace('hybrid_after22_delta','hybrid_after21_delta')
for k in ['archive_sha256','inventory_sha256','acceptance_sha256']:s=s.replace(bk[k],ob[k])
assert s==(O/'local_record_bridge_v2/bridge.py').read_text('utf8')
for n in ['checked_record_body.py','SOURCE_REUSE.json']:assert sha(B/'local_record_bridge_v2'/n)==sha(O/'local_record_bridge_v2'/n)
assert sha(B/'run_record_checks.py')==sha(O/'run_record_checks.py')
save('SOURCE_REUSE_VERIFICATION.json',dict(status='ORIGINAL_COLLECTOR_AND_RECORD_CHECKER_BODY_UNCHANGED',collector_binding_reversal_exact=True,verifier_binding_reversal_exact=True,record_binding_reversal_exact=True,scientific_body_and_null_policy_byte_exact=True,source_members_verified=69,new_tests_added=0))
r=read(O/'ROOT_READY_CHAIN_LINK.json')
r.update(previous_accepted=22,accepted_new=1,accepted_total_if_root_adopts=23,accepted_new_ids=ids,accepted_job_ids=prior['accepted_job_ids']+ids,previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'),previous_latest_sha256=sha(B/'PREVIOUS_LATEST.json'),previous_offserver_proof_sha256=prior['offserver_proof_sha256'],previous_root_adoption_sha256=prior['root_adoption_sha256'],authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),archive_sha256=bk['archive_sha256'],archive_members=bk['member_count'],inventory_sha256=bk['inventory_sha256'],server_strict_sha256=bk['acceptance_sha256'],offserver_tensor_proof_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),offserver_original_record_check_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),actual_server_runtime=a['runtime'],actual_local_torch=rec['local_torch'],remaining_if_root_adopts=9,source_reuse_verification_sha256=sha(B/'SOURCE_REUSE_VERIFICATION.json'),previous_chain_file=latest['chain_file'],delta_dir='accepted_delta_after22_20261010',archive='hybrid_after22_delta.tar.gz',owned_delivery_directory=B.as_posix(),root_copy_target=(H/'accepted_delta_after22_20261010').as_posix(),CPU106_release_sha256=sha(B/'CPU_RELEASE.json'))
save('ROOT_READY_CHAIN_LINK.json',r)
save('DELIVERY.json',dict(status='EXACT1_ORIGINAL_STRICT_OFFSERVER_TENSOR_RECORD_PASS_ROOT_PENDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_accepted=22,accepted_new=1,total_if_root_adopts=23,accepted_new_ids=ids,ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),archive_sha256=bk['archive_sha256'],receipt_sha256=sha(B/'BACKUP_SHA256.json'),offserver_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'),CPU106_release_sha256=sha(B/'CPU_RELEASE.json'),limits=['n=1 fixed validation search remains incomplete; no recipe selection or formal100/test.','Original cu128/5090 server device metadata and separate local CPU record/tensor verification retained. No new CNN or saved-array recomputation.','Shared active-service log is a labelled prefix. Source69/old22 models were not repackaged. Shared LATEST/STATE/Git unchanged.']))
rows=[dict(path=p.relative_to(B).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(B.rglob('*')) if p.is_file() and 'restored' not in p.relative_to(B).parts]
save('DELIVERY_FILES_SHA256.json',dict(members=rows,excluded_derived_tree='restored',archive_members_verified=23,original_source_members_verified=69))
print(json.dumps(dict(ready_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),delivery_sha256=sha(B/'DELIVERY.json'),seal_sha256=sha(B/'DELIVERY_FILES_SHA256.json'),sealed_members=len(rows),archive_sha256=bk['archive_sha256'],receipt_sha256=sha(B/'BACKUP_SHA256.json'),offserver_sha256=sha(B/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_sha256=sha(B/'LOCAL_RECORD_CHECKS.json'))))
