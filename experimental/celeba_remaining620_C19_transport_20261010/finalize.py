"""Compact links to actual original proofs; no new numerical science or adoption."""
from pathlib import Path
import datetime,hashlib,json,tarfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
canonical=lambda d:hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def save(name,d):
    with (HERE/name).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
ix=read(HERE/'RAW_STORAGE_INDEX.json');dest=Path(ix['directory']);extract=dest/'verification/verified_extract'
proof=read(ix['offserver_verification']);saved=proof['original_saved_array_verification'];snapshot=read(extract/'snapshot_inventory.json')
native_base=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T045743Z'
native_root=read(native_base/'ROOT_DELTA_VERIFICATION.json');native_archive=Path(native_root['archive_local_path'])
assert native_root['total_new_strict_and_offserver']==200 and native_root['test'] is False
assert sha(native_archive)==native_root['archive_sha256']=='ce1e364bc102eae140fcc23b299270b4f7793854434a8c9a5196ebdc2e00b0cb'
with tarfile.open(native_archive,'r:gz') as t:
    name=next(n for n in t.getnames() if n=='inspection.json' or n.endswith('/inspection.json'))
    data=t.extractfile(name).read();assert hashlib.sha256(data).hexdigest()==native_root['inspection_sha256']=='de989e02e299d3b4ff26f91fe704d93c1c71b21ac9b1a508fa0d6f813352962d'
    inspection=json.loads(data)
native={r['id']:r for r in inspection['records']};pre=read(HERE/'PREFLIGHT.stdout.json');pre_rows={r['id']:r for r in pre['closed']}
ids=pre['selected_ids'];assert snapshot['selected_replay_ids']==ix['accepted_new_ids']==ids and len(ids)==19
rows=[];metrics={r['id']:r for r in saved['records']}
for r in snapshot['records']:
    identity=r['id'];v4=native[identity]
    assert r['accepted_v4_row']==v4, 'Native200 full original record differs: '+identity
    assert r['checkpoint']['sha256']==pre_rows[identity]['checkpoint_sha256']==v4['checkpoint_sha256']==metrics[identity]['checkpoint_sha256']
    assert (r['variant'],r['distribution'],r['attack'],r['seed'])==(v4['variant'],v4['distribution'],v4['attack'],v4['seed'])
    assert r['variant']=='minus_C' and r['distribution']=='non-IID' and r['actual_alpha']==5.0
    assert r['terminal_round']==70 and r['original_split']=='valid' and r['original_n_eval']==19867
    assert r['original_job']==v4['job']==r['manifest_entry']['job']
    assert r['raw_job']['sha256']==r['manifest_entry']['job_sha256']==v4['files'][v4['job']]
    assert canonical(r['config'])==r['config_canonical_sha256']
    assert r['paired_full']['replay_required_here'] is False and r['paired_full']['weights_repacked_here'] is False
    assert metrics[identity]['native_max_abs_difference']==0
    rows.append(dict(id=identity,checkpoint_sha256=r['checkpoint']['sha256'],original_result_sha256=r['result']['sha256'],
        original_rawjob_sha256=r['raw_job']['sha256'],original_native200_record_canonical_sha256=canonical(v4),
        config_canonical_sha256=r['config_canonical_sha256'],source_hashes=r['source_hashes'],adapter_source_hashes=r['adapter_source_hashes'],
        data_contract_canonical_sha256=canonical(r['data_contract']),training_torch=r['training_torch'],actual_alpha=r['actual_alpha'],
        paired_full=r['paired_full'],saved_array_sha256=metrics[identity]['prediction_arrays_sha256'],native_max_abs_difference=0,
        original70_valid_identity_exact=True))
first_path=ROOT/'tmp/celeba_remaining620_first_transport_recovery_20261010/HANDOFF.json';first=read(first_path)
assert sha(first['receipt'])==first['receipt_sha256']=='71e1c6782e175f81b89776d2004bde255d7d09f3bfa99117d5af6e2545bb00fe'
assert ix['all_transported_ids']==[first['id'],*ids] and len(set(ix['all_transported_ids']))==20
assert first['checkpoint_sha256']==native[first['id']]['checkpoint_sha256']
assert sha(ix['archive'])==ix['archive_sha256'] and sha(ix['receipt'])==ix['receipt_sha256'] and sha(ix['offserver_verification'])==ix['offserver_verification_sha256']
save('NATIVE200_CHECKPOINT_BINDINGS.json',dict(status='EXACT19_ORIGINAL_NATIVE200_RECORDS_SOURCE_CONFIG_DATA_CHECKPOINT_PASS_NO_ADOPTION',
    native_ROOT_DELTA_path=str(native_base/'ROOT_DELTA_VERIFICATION.json'),native_ROOT_DELTA_sha256=sha(native_base/'ROOT_DELTA_VERIFICATION.json'),
    native_archive=str(native_archive),native_archive_sha256=sha(native_archive),inspection_member=name,inspection_sha256=hashlib.sha256(data).hexdigest(),
    new_count=19,records=rows,previous_first_checkpoint_native200_join=True,accepted_offserver=0,root_adopted=0))
# Byte-identical compact copy; the complete extraction and saved arrays stay on F.
(HERE/'OFFSERVER_TRANSPORT_VERIFICATION.json').write_bytes(Path(ix['offserver_verification']).read_bytes())
save('ROOT_READY_CHAIN_LINK.json',dict(status='EXACT19_TRANSPORT_AND_ORIGINAL_OFFSERVER_PASS_ROOT_ADOPTION_PENDING',
    new_ids=ids,all_transported_ids=ix['all_transported_ids'],new_count=19,cumulative_transported_count=20,
    previous_local_receipt=first['receipt'],previous_receipt_sha256=first['receipt_sha256'],
    previous_recovery_handoff=str(first_path),previous_recovery_handoff_sha256=sha(first_path),
    previous_original_failure=first['preserved_original_failure'],previous_original_failure_sha256=first['preserved_original_failure_sha256'],
    previous_root_authorized_recovery=first['root_authorized_recovery_receipt'],previous_root_authorized_recovery_sha256=first['root_authorized_recovery_receipt_sha256'],
    new_archive=ix['archive'],archive_sha256=ix['archive_sha256'],receipt=ix['receipt'],receipt_sha256=ix['receipt_sha256'],
    offserver_proof=ix['offserver_verification'],offserver_proof_sha256=ix['offserver_verification_sha256'],
    native_binding_path=str(HERE/'NATIVE200_CHECKPOINT_BINDINGS.json'),native_binding_sha256=sha(HERE/'NATIVE200_CHECKPOINT_BINDINGS.json'),
    source_seal_sha256=pre['source_seal_sha256'],transport_seal_sha256=pre['transport_seal_sha256'],accepted_offserver=0,root_adopted=0))
save('HANDOFF.json',dict(ix,status='ACTUAL_EXACT19_C_REPLAY_TRANSPORT_OFFSERVER_VERIFIED_PENDING_ROOT_ADOPTION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=pre['source_seal_sha256'],transport_seal_sha256=pre['transport_seal_sha256'],
    new19_original_native200_rows_exact=True,native_max_abs_difference=0,first_plus_new_chain_unique20=True,
    source_data_config_checkpoint_binding=str(HERE/'NATIVE200_CHECKPOINT_BINDINGS.json'),source_data_config_checkpoint_binding_sha256=sha(HERE/'NATIVE200_CHECKPOINT_BINDINGS.json'),
    chain_link_sha256=sha(HERE/'ROOT_READY_CHAIN_LINK.json'),CPU110_preflight_all_thread_narrow_owners=[],export_exit=read(HERE/'EXPORT_EXIT.json'),
    offserver_exit=read(HERE/'OFFSERVER_EXIT.json'),new_CNN=0,new_fits=0,new_training=0,Full_inference=0,models_repacked=0,final_test=False,
    limitations=['Transport and original offserver verification do not adopt IDs; accepted_offserver/root remain0 for root independent adoption.',
        'Exactly frozen19 only; previousfirst linked without re-export or new saved-array verification, no future remaining620 progress pursued.',
        'Native/archive/source/model restore links are retained; native200 archive is not repacked in this replay transport.',
        'Mixed training/replay environments, valid selection history, historical test metadata exposure, AEOD as absolute TPR gap and pending primary endpoint unchanged.',
        'Original first export failure and authorized recovery stay unchanged and explicitly linked.']))
print(json.dumps(dict(status='EXACT19_READY',handoff_sha256=sha(HERE/'HANDOFF.json'),chain_sha256=sha(HERE/'ROOT_READY_CHAIN_LINK.json'),
    bindings_sha256=sha(HERE/'NATIVE200_CHECKPOINT_BINDINGS.json'),max_native_difference=0,new=19,transported=20)))
