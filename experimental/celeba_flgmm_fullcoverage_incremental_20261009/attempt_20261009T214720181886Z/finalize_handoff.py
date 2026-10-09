from pathlib import Path
import json,hashlib,datetime
b=Path(__file__).resolve().parent;base=b.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert read(b/'VERIFY_COMMAND.json')['exit_code']==read(b/'SCP_RECEIPT.json')['exit_code']==read(b/'DISPATCH_RECEIPT.json')['exit_code']==0
assert sha(base/'LATEST_BACKUP.json')==sha(b/'PREVIOUS_LATEST.json')
prior=read(b/'PREVIOUS_OFFSERVER_ACCEPTANCE.json');proof=read(b/'batch/OFFSERVER_ACCEPTANCE.json');receipt=read(b/'batch/BACKUP_SHA256.json');server=read(b/'batch/PARTIAL_ACCEPTANCE.json');snap=read(b/'batch/restored/live_snapshot.json')
assert prior['accepted_total']==2 and proof['accepted_new']==1 and proof['accepted_total']==3
assert set(proof['accepted_job_ids'])-set(prior['accepted_job_ids'])==set(proof['new_ids']) and proof['new_ids']==receipt['accepted_new_ids']
assert receipt['previous_chain_sha256']==sha(b/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')=='9e9591bf3bede29b74a6ea34a944ef1b63b0584edd3ad95802d3725f77bd510e'
assert sha(base/'FILES_SHA256.json')=='f6de59a56de25a8d316cf7e05c44eafc9751041757e4dc61c88f7bccfd988472'
for n,pin in read(base/'FILES_SHA256.json')['files'].items():assert sha(base/n)==pin['sha256']
ready={'status':'ROOT_READY_FL96_INCREMENT_STRICT_OFFSERVER_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_new_ids':proof['new_ids'],'prior_accepted_new':2,'accepted_new':1,'accepted_new_cumulative':3,'planned_new':96,'separately_reused70round':4,'planned_total':100,'single_collector_snapshot':True,'source_package_sha256':proof['package_sha256'],'prepared_helper_seal_sha256':sha(base/'FILES_SHA256.json'),'offserver_acceptance_sha256':sha(b/'batch/OFFSERVER_ACCEPTANCE.json'),'server_backup_receipt_sha256':sha(b/'batch/BACKUP_SHA256.json'),'archive_sha256':receipt['archive_sha256'],'archive_bytes':receipt['archive_size'],'archive_members':receipt['archived_member_count'],'inventory_sha256':receipt['inventory_sha256'],'server_strict_sha256':receipt['acceptance_sha256'],'previous_actual_offserver_sha256':receipt['previous_chain_sha256'],'previous_latest_sha256':sha(b/'PREVIOUS_LATEST.json'),'actual_snapshot_utc':snap['utc'],'records':proof['records'],'helper_CPU':106,'helper_threads':1,'nice':10,'IO':'idle','server_verification_runtime':server['acceptance_runtime'],'local_verification_runtime':proof['verification_runtime'],'training_runtime_not_recreated':True,'root_adoption_required':True,'no_old_models_repacked':True,'no_CNN_training_test':True,'canonical_state_git_queue_unchanged':True}
with (b/'ROOT_READY_HANDOFF.json').open('x',encoding='utf8') as f:json.dump(ready,f,indent=2);f.write('\n')
files={p.relative_to(b).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(b.rglob('*')) if p.is_file() and 'restored' not in p.relative_to(b).parts}
with (b/'DELIVERY_FILES_SHA256.json').open('x',encoding='utf8') as f:json.dump({'status':'ACTUAL_INCREMENT_ROOT_PENDING','files':files,'derived_restored_tree_excluded':True},f,indent=2);f.write('\n')
print(json.dumps({'handoff_sha256':sha(b/'ROOT_READY_HANDOFF.json'),'delivery_seal_sha256':sha(b/'DELIVERY_FILES_SHA256.json'),'sealed_files':len(files),'offserver_sha256':sha(b/'batch/OFFSERVER_ACCEPTANCE.json'),'snapshot_utc':snap['utc'],'metrics':proof['records'][0]['metrics']}))
