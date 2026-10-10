"""Bind the actual exact8 saved-array delivery to native228 and adopted replay220; no adoption."""
from pathlib import Path
import datetime,hashlib,json,runpy,traceback
B=Path(__file__).resolve().parent;ROOT=B.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(name,value):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    pins=read(B/'INPUT_BINDINGS.json');raw=read(B/'RAW_STORAGE_INDEX.json');pre=read(B/'PREFLIGHT.stdout.json');export=read(B/'EXPORT.stdout.json')
    ids=pins['selected_ids'];F=Path(raw['directory']);extract=F/'verification/verified_extract'
    volume=runpy.run_path(str(ROOT/'tmp/guardfed_local_storage.py'))['check_bulk_storage'](0)
    for name in ('EXPORT_EXIT.json','SCP_EXIT.json','OFFSERVER_EXIT.json'):assert read(B/name)['returncode']==0
    assert sha(pins['prior_root_path'])==pins['prior_root_sha256'] and sha(pins['prior_index_path'])==pins['prior_index_sha256']
    prior=read(pins['prior_index_path']);assert len(prior['all_ids'])==220 and not set(ids)&set(prior['all_ids'])
    assert sha(pins['native_root_path'])==pins['native_root_sha256'] and sha(pins['native_inspection_path'])==pins['native_inspection_sha256'] and sha(pins['native_ledger_path'])==pins['native_ledger_sha256']
    native=read(pins['native_inspection_path']);by={r['id']:r for r in native['records']}
    assert len(native['accepted_new_ids'])==228 and read(pins['native_root_path'])['new_ids']==ids
    assert sha(raw['archive'])==raw['archive_sha256']==export['archive_sha256'] and sha(raw['receipt'])==raw['receipt_sha256']
    assert sha(raw['offserver_verification'])==raw['offserver_verification_sha256']
    proof=read(raw['offserver_verification']);saved=proof['original_saved_array_verification']
    assert proof['accepted_offserver']==0 and proof['root_adoption_pending'] and proof['archive_member_verification']['members_verified']==74
    assert saved['accepted_n']==8 and (saved['independent_metric_checks'],saved['independent_confusion_count_checks'],saved['prediction_rule_checks'])==(72,192,24)
    assert {r['id'] for r in saved['records']}==set(ids)
    assert export['accepted_new_ids']==ids and len(set(export['all_transported_ids']))==48
    assert export['all_transported_ids']==pre['previous_receipt']['all_transported_ids']+ids
    assert len(set(prior['all_ids']+ids))==228
    joined=[]
    for identity in ids:
        binding=read(extract/'runtime'/identity/'binding.json');record=binding['record'];accepted=next(r for r in saved['records'] if r['id']==identity)
        assert record['accepted_v4_row']==binding['native_acceptance']['row']==by[identity]
        assert identity==record['id']==binding['native_acceptance']['id']==by[identity]['id']
        assert record['terminal_round']==70 and record['original_split']=='valid' and record['original_n_eval']==19867
        assert record['variant']=='minus_A' and record['distribution']=='IID' and record['attack']=='FedSA' and record['actual_alpha']==5000.0
        assert record['seed']==by[identity]['seed'] and record['seed'] in range(91001,91009)
        assert record['checkpoint']['sha256']==binding['checkpoint_sha256']==accepted['checkpoint_sha256']==by[identity]['checkpoint_sha256']
        assert accepted['native_max_abs_difference']<=1e-12
        joined.append(dict(id=identity,native_inspection_row_exact=True,checkpoint_sha256=accepted['checkpoint_sha256'],original_files=by[identity]['files'],config_canonical_sha256=record['config_canonical_sha256'],source_hashes=record['source_hashes'],adapter_source_hashes=record['adapter_source_hashes'],data_contract=record['data_contract'],paired_full_reference=record['paired_full'],binding_path=str(extract/'runtime'/identity/'binding.json'),binding_sha256=sha(extract/'runtime'/identity/'binding.json'),native_acceptance_canonical_sha256=binding['native_acceptance_canonical_sha256'],native_max_abs_difference=accepted['native_max_abs_difference']))
    assert sha(ROOT/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py')==pins['transport_py_sha256']
    maximum=max(r['native_max_abs_difference'] for r in saved['records'])
    save('NATIVE228_IDENTITY_JOIN.json',dict(status='EIGHT_BOUND_RECORDS_EXACT_TO_ACCEPTED_NATIVE228_INSPECTION_ROOT_ADOPTION_PENDING',native_root_path=pins['native_root_path'],native_root_sha256=pins['native_root_sha256'],native_archive_path=pins['native_archive_path'],native_archive_sha256=pins['native_archive_sha256'],native_inspection_path=pins['native_inspection_path'],native_inspection_sha256=pins['native_inspection_sha256'],native_ledger_path=pins['native_ledger_path'],native_ledger_sha256=pins['native_ledger_sha256'],prior_replay220_index=pins['prior_index_path'],prior_replay220_index_sha256=pins['prior_index_sha256'],new8_ids=ids,records=joined,native_row_exact=8,old220_excluded=True,models_repacked=0,accepted_offserver=0,root_adopted=0))
    handoff=dict(status='EXACT8_ORIGINAL_TRANSPORT_AND_OFFSERVER_SAVED_ARRAY_CHECKS_PASS_PENDING_ROOT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_ids=ids,prior_transported=40,new_transported=8,total_transported=48,prior_root_accepted_replays=220,replay_total_if_root_adopts=228,accepted_offserver=0,root_adopted=0,
        prior_root=pins['prior_root_path'],prior_root_sha256=pins['prior_root_sha256'],prior_index=pins['prior_index_path'],prior_index_sha256=pins['prior_index_sha256'],prior_receipt_sha256=pins['previous_receipt_sha256'],prior_offserver_sha256=pins['previous_offserver_sha256'],
        native228_root=pins['native_root_path'],native228_root_sha256=pins['native_root_sha256'],native228_inspection_sha256=pins['native_inspection_sha256'],native228_ledger_sha256=pins['native_ledger_sha256'],native228_archive_path=pins['native_archive_path'],native228_archive_sha256=pins['native_archive_sha256'],native_identity_join_sha256=sha(B/'NATIVE228_IDENTITY_JOIN.json'),
        raw_storage_index=str(B/'RAW_STORAGE_INDEX.json'),raw_storage_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),archive_path=raw['archive'],archive_sha256=raw['archive_sha256'],archive_bytes=raw['archive_bytes'],receipt_path=raw['receipt'],receipt_sha256=raw['receipt_sha256'],offserver_path=raw['offserver_verification'],offserver_sha256=raw['offserver_verification_sha256'],archive_member_manifest=raw['archive_member_manifest'],archive_member_manifest_sha256=raw['archive_member_manifest_sha256'],archive_members=74,metrics=72,counts=192,rules=24,native_max_abs_difference=maximum,native_tolerance=1e-12,
        scientific_source_unchanged=True,source_seal_sha256=pre['source_seal_sha256'],transport_seal_sha256=pre['transport_seal_sha256'],transport_py_sha256=pins['transport_py_sha256'],original_export_cli_unmodified=True,original_saved_array_verifier_unmodified_except_original_transport_inventory_pin_projection=True,
        snapshot_utc=pre['utc'],snapshot_remote_closed=pre['actual_remote_closed'],CPU=110,CPU107_used=False,export_wait_returned0=True,remote_collector_completed=True,all_new_bulk_on_F=True,fresh_F_volume=volume,
        negative_results_preserved=True,new_CNN=0,new_fit=0,new_training=0,Full_inference=0,test_inference=0,models_repacked=0,scene_aggregate_created=False,A_complete=False,partial_A_scene=dict(distribution='IID',attack='FedSA',seeds=list(range(91001,91009)),n=8,target_n=10),
        canonical_STATE_latest_Git_modified=False,original_transport_internal_receipt_chain_updated=True,
        limits=['FedSA IID is only8/10 seeds; no scenario mean, A100, full mechanism completion, significance or final-test claim.','Raw/native/shared saved arrays and original9/24/3 checks are retained per ID; native tolerance stays1e-12. No CNN, root refit, model repack, Full or old220 replay occurred.','Native source/config/data/checkpoint rows exactly match the fixed native228 inspection and its original recovery archive. This transport report does not independently adopt native or replay evidence.','Original transport internal TRANSPORT_LATEST receipt pointer advances normally; no shared native ledger, canonical acceptance/state, recipe or Git was changed.'],
        root_next='Independently join native228 recovery members/checkpoints and adopt these exact8 on the unchanged replay220 index. Do not create a FedSA10 scene table from this8-seed partial subset.')
    save('HANDOFF.json',handoff)
    with (B/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write('# A28 exact8 transport — root adoption pending\n\nFixed minus_A / IID / FedSA seeds91001–91008 passed the original export and offserver saved-array verifier once:74 members,72 metrics,192 counts,24 rules; native maximum difference '+str(maximum)+' at unchanged1e-12 tolerance. Native228 exact rows/checkpoints are bound in `NATIVE228_IDENTITY_JOIN.json`; prior root-adopted replay220 is excluded. All new archive/arrays/extraction stay on F.\n\nThis is FedSA8/10, not a complete scenario; no mean/table or new inference/fit/training/test was produced. Accepted_offserver/root_adopted remain0 until root review. Existing scientific transport, sources, prior evidence and shared state/Git remain unchanged. Original internal transport receipt chain advanced normally.\n')
    assert not list(B.rglob('__pycache__'))
    members=[dict(path=p.relative_to(B).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(B.rglob('*')) if p.is_file()]
    save('FILES_SHA256.json',dict(status='SEALED_EXACT8_A28_ORIGINAL_TRANSPORT_DELIVERY_PENDING_ROOT',files={r['path']:dict(sha256=r['sha256'],bytes=r['bytes']) for r in members},raw_storage_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),bulk_external_only=True))
    assert all(sha(B/r['path'])==r['sha256'] for r in members)
    print(json.dumps(dict(seal_sha256=sha(B/'FILES_SHA256.json'),members=len(members),handoff_sha256=sha(B/'HANDOFF.json'),native_join_sha256=sha(B/'NATIVE228_IDENTITY_JOIN.json'),archive_sha256=raw['archive_sha256'],receipt_sha256=raw['receipt_sha256'],offserver_sha256=raw['offserver_verification_sha256'],raw_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),native_max_abs_difference=maximum)))
if __name__=='__main__':
    try:main()
    except BaseException as error:save('FINALIZER_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
