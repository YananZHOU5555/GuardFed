"""Compact handoff only; numerical validation and native join already ran once."""
from pathlib import Path
import argparse,datetime,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (H/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
parser=argparse.ArgumentParser()
for flag in ('archive','offserver','index','join'):parser.add_argument('--'+flag+'-sha256',required=True)
args=parser.parse_args()
P=read(H/'PREPARED.json');E=read(H/'EXECUTION_INPUTS.json');raw=read(H/'RAW_STORAGE_INDEX.json');j=read(H/'NATIVE_IDENTITY_JOIN.json');idx=read(H/'MECHANISM_INDEX.json')
assert raw['accepted_new_ids']==P['candidate_ids']==idx['new_ids']==j['new_ids'] and len(P['candidate_ids'])==10
assert (raw['metrics'],raw['counts'],raw['rules'])==(90,240,30)
assert sha(P['prior_index_path'])==P['prior_index_sha256'] and sha(P['prior_root_path'])==P['prior_root_sha256']
prior=read(P['prior_index_path']);assert idx['all_ids'][:320]==prior['all_ids'] and len(idx['all_ids'])==len(set(idx['all_ids']))==330
assert idx['prior_index_sha256']==P['prior_index_sha256'] and idx['prior_adoption_sha256']==P['prior_root_sha256']
assert len(raw['all_transported_ids'])==150 and raw['all_transported_ids'][:140]==P['previous_all_transported_ids']
assert j['prior320_objects_and_order_unchanged'] and j['old420_native_records_exact'] and j['old_native_ledger_entries_exact']
assert j['native_max_abs_difference']==0 and j['native_model_result_members_rehashed']==20
for name in ['PREFLIGHT','EXPORT','SCP','OFFSERVER']:assert read(H/(name+'_EXIT.json'))['returncode']==0
for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():assert sha(H/name)==pin['sha256']
# Actual metadata pins only: no rerun of original saved science or archive extraction.
assert sha(H/'SOURCE_FILES_SHA256.json')==E['source_seal_sha256']
assert raw['archive_sha256']==args.archive_sha256 and raw['archive_members']==92
assert raw['offserver_verification_sha256']==args.offserver_sha256
assert sha(H/'MECHANISM_INDEX.json')==args.index_sha256
assert sha(H/'NATIVE_IDENTITY_JOIN.json')==args.join_sha256
for name in ['ACTUAL_EXECUTE','ACTUAL_JOIN','CPU_RELEASE']:assert read(H/(name+'_EXIT.json'))['returncode']==0
assert sha(Path(E['native_root_path']))==E['native_root_sha256']
assert read(Path(E['native_root_path']))['total_new_strict_and_offserver']==E['root_native_total']>=330

dep=read(R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/INPUTS.json')['dependencies']['saved_verifier']
assert sha(R/dep['local_relative'])==dep['sha256']
release=read(H/'CPU_RELEASE.json');assert not release['CPU111_restricted_owners'] and not release['exporters']
save('ACTUAL_HANDOFF.json',dict(status='EXACT10_ORIGINAL_SAVED_ARRAY_TRANSPORT_AND_NATIVE_MIN333_JOIN_PASS_PENDING_ROOT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 selected_ids=P['candidate_ids'],prior_root_accepted_replays=320,new_transported=10,replay_total_if_root_adopts=330,prior_transported=140,total_transported=150,
 accepted_offserver=0,root_adopted=0,prior_root=P['prior_root_path'],prior_root_sha256=P['prior_root_sha256'],prior_index=P['prior_index_path'],prior_index_sha256=P['prior_index_sha256'],
 prior_receipt_sha256=P['previous_receipt_sha256'],native_root=E['native_root_path'],native_root_sha256=E['native_root_sha256'],native_inspection=E['native_inspection_path'],native_inspection_sha256=E['native_inspection_sha256'],native_ledger=E['native_ledger_path'],native_ledger_sha256=E['native_ledger_sha256'],
 native_identity_join_sha256=sha(H/'NATIVE_IDENTITY_JOIN.json'),proposed_index_path=(H/'MECHANISM_INDEX.json').relative_to(R).as_posix(),proposed_index_sha256=sha(H/'MECHANISM_INDEX.json'),
 archive_path=raw['archive'],archive_sha256=raw['archive_sha256'],archive_bytes=raw['archive_bytes'],receipt_path=raw['receipt'],receipt_sha256=raw['receipt_sha256'],offserver_path=raw['offserver_verification'],offserver_sha256=raw['offserver_verification_sha256'],archive_member_manifest=raw['archive_member_manifest'],archive_member_manifest_sha256=raw['archive_member_manifest_sha256'],archive_members=raw['archive_members'],
 metrics=90,counts=240,rules=30,native_max_abs_difference=0,native_tolerance=1e-12,native_model_result_members_rehashed=20,old320_index_objects_and_order_exact=True,old420_native_records_exact=True,old46_native_ledger_entries_exact=True,
 source_seal_sha256=sha(H/'SOURCE_FILES_SHA256.json'),original_transport_source_sha256='f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7',original_saved_verifier_sha256=dep['sha256'],CPU111_memory_bridge_sha256=sha(H/'remote_cpu111_export.py'),CPU_release_sha256=sha(H/'CPU_RELEASE.json'),
 original_prepared_handoff_sha256=sha(H/'HANDOFF.json'),original_prepared_delivery_seal_sha256=sha(H/'SOURCE_FILES_SHA256.json'),root_execute_exit_code=0,root_join_exit_code=0,CPU_release_exit_code=0,CPU111_restricted_owners=release['CPU111_restricted_owners'],exporters=release['exporters'],
 export_once=True,automatic_retry=False,new_CNN=0,new_fit=0,new_training=0,Full_inference=0,Full_weights_repacked=0,test=False,canonical_STATE_written=False,canonical_ledger_written=False,Git_mutations=0,
 limitation='Only exact10 saved arrays. Actual CLI-bound native total>=330 is adopted; proposed three-view330 remains pending root adoption. Original saved checker independently recomputed valid metrics/counts/prediction rules; it did not rerun root fitting (strict server bridge previously checked it). Root scientific adoption remains pending.'))
print(json.dumps({'handoff_sha256':sha(H/'ACTUAL_HANDOFF.json'),'index_sha256':sha(H/'MECHANISM_INDEX.json'),'new':10,'proposed_total':330,'native_total':E['root_native_total']}))
