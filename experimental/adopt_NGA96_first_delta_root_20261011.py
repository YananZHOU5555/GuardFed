"""Root acceptance of one closed coverage result, preserving all original outcomes."""
from pathlib import Path
import datetime,hashlib,json,tarfile
R=Path(__file__).resolve().parents[1];H=R/'tmp/NGA96_incremental_acceptance_prepare_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
assert not (H/'ROOT_ADOPTION.json').exists()
auth=read(H/'AUTHORIZED_SNAPSHOT.json');off=read(H/'OFFSERVER_ACCEPTANCE.json');download=read(H/'DOWNLOAD.json');release=read(H/'COLLECTOR_RELEASE.json')
assert sha(H/'ROOT_REVIEW.json')=='f8ea2d2824cd1404ad91c702e7c9043ac4ec25221fc34c8b7d96b50039a1ed1b'
assert sha(H/'OFFSERVER_ACCEPTANCE.json')=='0c64649b140870fec95624b2343b300abf210d3d81e2bf8c3ba1200c96005cda'
ids=auth['authorized_ids'];assert ids==['FedNGA_eta0.001_IID_Benign_seed91002_coverage'] and auth['accepted_prior_ids']==[] and auth['accepted_prior_count']==0
assert off['status']=='OFFSERVER_ORIGINAL_STRICT_DELTA_PASS' and off['accepted_new_ids']==off['accepted_job_ids']==ids and off['accepted_before']==0 and off['accepted_total']==1
assert off['root_adopted'] is False and off['coverage100_completed_claim'] is False and off['all_negative_results_retained'] is True
assert off['new_inference']==off['new_training']==off['old_models_repacked']==off['reused_screen4_repacked']==0 and off['reused_screen4_count']==4
assert off['scientific_package_sha256']=='06e3d638aea70123d7fcf73e5a219a0c0d3165c8cf60acb9dee4eace5faad09b' and off['original_validator_sha256']=='d00dd2f3c0d8486f3e32044c3553972d1ff98515b54834a0a8b51048d2a5555e'
assert off['archive_verification']['pass'] and off['archive_verification']['different_host_observed'] and release['CPU110_released'] and not release['collector_processes']
for phase in ['COLLECT','REMOTE_ARTIFACTS','SCP','VERIFY','RELEASE']:assert read(H/(phase+'_COMMAND.json'))['exit_code']==0,phase
archive=Path(off['archive']);assert archive.drive.upper()=='F:' and sha(archive)==off['archive_sha256']==download['archive_sha256']
assert sha(H/'SAVED_TENSOR_STATE_CHECK.json')==off['tensor_proof_sha256']
tensor=read(H/'SAVED_TENSOR_STATE_CHECK.json');assert tensor['models']==1 and tensor['CNN_forward_calls']==tensor['optimizer_calls']==tensor['data_loads']==0 and tensor['records'][0]['tensor_count']==8
with tarfile.open(archive,'r:gz') as t:
 inventory_raw=t.extractfile('backup_inventory.json').read();inventory=json.loads(inventory_raw)
 assert hashlib.sha256(inventory_raw).hexdigest()==off['inventory_sha256'] and inventory['accepted_new_ids']==ids
 assert len(t.getmembers())==len(inventory['members'])+1==off['members_verified']==131
 for name,pin in inventory['members'].items():
  m=t.getmember(name);assert m.isfile() and m.size==pin['bytes'];assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==pin['sha256'],name
assert len(off['records'])==1 and off['records'][0]['rounds']==70 and off['records'][0]['constant_negative_retained']
proof=dict(status='ROOT_NGA96_FIRST_COVERAGE_DELTA_STRICT_OFFSERVER_FULLSTATE_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_adopted=True,test=False,accepted_ids=ids,accepted_total=1,accepted_new_ids=ids,accepted_before=0,scientific_package_sha256=off['scientific_package_sha256'],original_validator_sha256=off['original_validator_sha256'],offserver=ref(H/'OFFSERVER_ACCEPTANCE.json'),authorization=ref(H/'AUTHORIZED_SNAPSHOT.json'),root_review=ref(H/'ROOT_REVIEW.json'),collector_release=ref(H/'COLLECTOR_RELEASE.json'),raw_storage_index=ref(H/'RAW_STORAGE_INDEX.json'),archive_path=archive.as_posix(),archive_sha256=sha(archive),root_rehashed_members=131,saved_tensor_state_proof=ref(H/'SAVED_TENSOR_STATE_CHECK.json'),records=off['records'],prior_root_sha256=None,prior_offserver_sha256=None,reused_screen4_count_separate=4,reused_screen4_repacked=0,coverage100_completed_claim=False,primary_endpoint_selected=False,final_evaluation=False,all_negative_results_retained=True,constant_predictor_not_fairness_success=True,whole_rebuttal_complete=False,new_inference=0,new_training=0)
with (H/'ROOT_ADOPTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(root_sha256=sha(H/'ROOT_ADOPTION.json'),accepted_new=1,coverage96_remaining=95,reused4_separate=True)))
