"""Compact handoff only; numerical validation and native join already ran once."""
from pathlib import Path
import datetime,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (H/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
P=read(H/'PREPARED.json');E=read(H/'EXECUTION_INPUTS.json');raw=read(H/'RAW_STORAGE_INDEX.json');j=read(H/'NATIVE_IDENTITY_JOIN.json');idx=read(H/'MECHANISM_INDEX.json')
assert raw['accepted_new_ids']==P['candidate_ids']==idx['new_ids']==j['new_ids'] and len(P['candidate_ids'])==8
assert (raw['metrics'],raw['counts'],raw['rules'])==(72,192,24)
assert sha(P['prior_index_path'])==P['prior_index_sha256'] and sha(P['prior_root_path'])==P['prior_root_sha256']
prior=read(P['prior_index_path']);assert idx['all_ids'][:280]==prior['all_ids'] and len(idx['all_ids'])==len(set(idx['all_ids']))==288
assert idx['prior_index_sha256']==P['prior_index_sha256'] and idx['prior_adoption_sha256']==P['prior_root_sha256']
assert len(raw['all_transported_ids'])==108 and raw['all_transported_ids'][:100]==P['previous_all_transported_ids']
assert j['prior280_objects_and_order_unchanged'] and j['old380_native_records_exact'] and j['old_native_ledger_entries_exact']
assert j['native_max_abs_difference']==0 and j['native_model_result_members_rehashed']==16
for name in ['PREFLIGHT','EXPORT','SCP','OFFSERVER']:assert read(H/(name+'_EXIT.json'))['returncode']==0
for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():assert sha(H/name)==pin['sha256']
# Actual metadata pins only: no rerun of original saved science or archive extraction.
assert sha(H/'SOURCE_FILES_SHA256.json')=='dba101d091a50bc8a3bd1f69abc83d4617e8f6331752fff6c122976c0122bdd9'
assert sha(H/'DELIVERY_FILES_SHA256.json')=='4ee8081a64e3dc1181e5bdc1deb837b684b30f2160077a4f7aa805f055b156f1'
for name,pin in read(H/'DELIVERY_FILES_SHA256.json')['files'].items():assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
assert raw['archive_sha256']=='fc6b2bb90bd7f7b15b33bd63b3da90f02f84616982c9dfa6846d76dca1c031c5' and raw['archive_members']==74
assert raw['offserver_verification_sha256']=='2d8cc9c55a0e84e410b36a0e1cb96701ad3fb9ec8a60665a82938bc09512a013'
assert sha(H/'MECHANISM_INDEX.json')=='15676070f43acab92b7d72aea2235432c459a95a3c6d33c08368f23ad19bb48d'
assert sha(H/'NATIVE_IDENTITY_JOIN.json')=='fc8c96d1fa4f52f90bfc465485cb017b49a23ff3258c8bf60d2c11c4cd23e342'
for name in ['ACTUAL_EXECUTE','ACTUAL_JOIN','CPU_RELEASE']:assert read(H/(name+'_EXIT.json'))['returncode']==0
assert E['native_root_sha256']=='deeee0f4c37546de88e3dd06ca26c3b847a04d3c09a5f6879d8c8f658bb5f3f7'

dep=read(R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/INPUTS.json')['dependencies']['saved_verifier']
assert sha(R/dep['local_relative'])==dep['sha256']
release=read(H/'CPU_RELEASE.json');assert not release['CPU111_restricted_owners'] and not release['exporters']
save('ACTUAL_HANDOFF.json',dict(status='EXACT8_ORIGINAL_SAVED_ARRAY_TRANSPORT_AND_NATIVE288_JOIN_PASS_PENDING_ROOT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 selected_ids=P['candidate_ids'],prior_root_accepted_replays=280,new_transported=8,replay_total_if_root_adopts=288,prior_transported=100,total_transported=108,
 accepted_offserver=0,root_adopted=0,prior_root=P['prior_root_path'],prior_root_sha256=P['prior_root_sha256'],prior_index=P['prior_index_path'],prior_index_sha256=P['prior_index_sha256'],
 prior_receipt_sha256=P['previous_receipt_sha256'],native_root=E['native_root_path'],native_root_sha256=E['native_root_sha256'],native_inspection=E['native_inspection_path'],native_inspection_sha256=E['native_inspection_sha256'],native_ledger=E['native_ledger_path'],native_ledger_sha256=E['native_ledger_sha256'],
 native_identity_join_sha256=sha(H/'NATIVE_IDENTITY_JOIN.json'),proposed_index_path=(H/'MECHANISM_INDEX.json').relative_to(R).as_posix(),proposed_index_sha256=sha(H/'MECHANISM_INDEX.json'),
 archive_path=raw['archive'],archive_sha256=raw['archive_sha256'],archive_bytes=raw['archive_bytes'],receipt_path=raw['receipt'],receipt_sha256=raw['receipt_sha256'],offserver_path=raw['offserver_verification'],offserver_sha256=raw['offserver_verification_sha256'],archive_member_manifest=raw['archive_member_manifest'],archive_member_manifest_sha256=raw['archive_member_manifest_sha256'],archive_members=raw['archive_members'],
 metrics=72,counts=192,rules=24,native_max_abs_difference=0,native_tolerance=1e-12,native_model_result_members_rehashed=16,old280_index_objects_and_order_exact=True,old380_native_records_exact=True,old40_native_ledger_entries_exact=True,
 source_seal_sha256=sha(H/'SOURCE_FILES_SHA256.json'),original_transport_source_sha256='f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7',original_saved_verifier_sha256=dep['sha256'],CPU111_memory_bridge_sha256=sha(H/'remote_cpu111_export.py'),CPU_release_sha256=sha(H/'CPU_RELEASE.json'),
 original_prepared_handoff_sha256=sha(H/'HANDOFF.json'),original_prepared_delivery_seal_sha256=sha(H/'DELIVERY_FILES_SHA256.json'),root_execute_exit_code=0,root_join_exit_code=0,CPU_release_exit_code=0,CPU111_restricted_owners=release['CPU111_restricted_owners'],exporters=release['exporters'],
 export_once=True,automatic_retry=False,new_CNN=0,new_fit=0,new_training=0,Full_inference=0,Full_weights_repacked=0,test=False,canonical_STATE_written=False,canonical_ledger_written=False,Git_mutations=0,
 limitation='Only exact8 saved arrays. Native total288 is adopted; proposed three-view288 remains pending root adoption. Original saved checker independently recomputed valid metrics/counts/prediction rules; it did not rerun root fitting (strict server bridge previously checked it). Root scientific adoption remains pending.'))
print(json.dumps({'handoff_sha256':sha(H/'ACTUAL_HANDOFF.json'),'index_sha256':sha(H/'MECHANISM_INDEX.json'),'new':8,'proposed_total':288,'native_total':288}))
