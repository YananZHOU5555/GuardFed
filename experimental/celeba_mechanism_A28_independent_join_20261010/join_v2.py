"""One local, record-only A28 join; proposes an index and never adopts it."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DELIVERY = ROOT / 'tmp/celeba_remaining620_A28_transport_20261010'
EXPECTED = [f'minus_A_IID_FedSA_seed{s}' for s in range(91001, 91009)]
read = lambda p: json.loads(Path(p).read_bytes())
canonical = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(name, value):
    with (HERE / name).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def pin(path, wanted):
    assert sha(path) == wanted, str(path)
    return read(path)

def run():
    seal = pin(DELIVERY / 'FILES_SHA256.json', 'f2f5479caeb1b69b70651d7a815145e951bc4befe50b4b371af0eada58069b17')
    assert len(seal['files']) == 32
    for name, row in seal['files'].items():
        p = DELIVERY / name
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name
    handoff = pin(DELIVERY / 'HANDOFF.json', 'd8effc1729631747c7f7bd225c00e3dda32c4ae6c3de0fc479adaed6a350da58')
    storage = pin(DELIVERY / 'RAW_STORAGE_INDEX.json', 'b82fb5a87fff6fde70908bda335f9ea781a1c18a7257526672b307b72b2ccb0b')
    assert handoff['selected_ids'] == storage['accepted_new_ids'] == EXPECTED
    assert storage['accepted_offserver'] == storage['root_adopted'] == 0
    assert (storage['metrics'], storage['counts'], storage['rules'], storage['archive_members']) == (72, 192, 24, 74)
    receipt = pin(storage['receipt'], '977c97a1c05ba78bc25b7402fb5c313dcd9c286b140ce4dbf9129bf7bd92aa30')
    off = pin(storage['offserver_verification'], '6c6cfba336d6da107a8413289d0dfee6c8522e58c23ba45ede720b3fe491e759')
    assert off['actual_transport_archive_sha256'] == receipt['archive_sha256'] == storage['archive_sha256'] == '09e5eec9f4f9ac266f2ada3449e11658a020e702a880000a183dd049652d9d4d'
    assert off['receipt_sha256'] == storage['receipt_sha256'] == sha(storage['receipt'])
    assert off['accepted_offserver'] == 0 and off['root_adoption_pending'] is True
    assert off['source_seal_sha256'] == receipt['source_seal_sha256'] == handoff['source_seal_sha256'] == 'a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
    assert receipt['transport_source_seal_sha256'] == handoff['transport_seal_sha256'] == '1a021b707575292c33959c19fcfa2fa1ee8c7f285d20576c562e4843d1488fb3'
    assert sha(storage['archive_member_manifest']) == receipt['inventory_sha256'] == storage['archive_member_manifest_sha256']
    original = ROOT / 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
    assert sha(original) == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
    spec = importlib.util.spec_from_file_location('original_A28_archive_verifier', original)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    archive_check = verifier.verify_archive(Path(storage['archive']), receipt)
    assert archive_check['members_verified'] == 74 and archive_check['different_host_observed']
    saved = off['original_saved_array_verification']; records = saved['records']
    assert saved['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
    assert [r['id'] for r in records] == EXPECTED
    assert (saved['independent_metric_checks'], saved['independent_confusion_count_checks'], saved['prediction_rule_checks']) == (72, 192, 24)
    prior_proof_path = Path(handoff['prior_root']); prior_path = Path(handoff['prior_index'])
    prior_proof = pin(prior_proof_path, 'e088871fbd98cbc9415cc79a44626667a532389ce0b6dddbaaf2ab25f72a4979')
    prior = pin(prior_path, 'af414a0ac6705230c53d324cd1f51e7a76b893dabd7416bb6914a6690b6fdddc')
    assert prior_proof['cumulative_accepted'] == len(prior['all_ids']) == 220
    assert prior_proof['records_index_sha256'] == sha(prior_path)
    assert not set(EXPECTED) & set(prior['all_ids'])
    previous = read(ROOT / 'tmp/celeba_remaining620_A20_transport_20261010/RAW_STORAGE_INDEX.json')
    assert sha(previous['receipt']) == storage['previous_receipt_sha256'] == receipt['previous_backup_receipt_sha256'] == '428fa89806721228f9deca04030218b819102d5d4f6c7595b23a29dba2fe7669'
    assert sha(previous['offserver_verification']) == handoff['prior_offserver_sha256']
    assert storage['all_transported_ids'] == receipt['all_transported_ids'] == previous['all_transported_ids'] + EXPECTED
    native_root_path = Path(handoff['native228_root']); native_folder = native_root_path.parent
    native_proof = pin(native_root_path, '8df618fda9dde965a43d5a325fc4010a322189585dc45302d3166ab87ea1ed60')
    inspection_path = native_folder / 'inspection/inspection.json'
    inspection = pin(inspection_path, '9836296d09e11c6b95f394e0007d83b735bdb7c05e2fcb1c162cd0f9e3a26d5b')
    ledger_path = native_folder / 'verified_ledger.json'
    ledger = pin(ledger_path, 'ea513bdf4456a43b8bc84d37167edbb029999951a894e690139f56aa1ec09180')
    assert native_proof['new_ids'] == EXPECTED and native_proof['total_new_strict_and_offserver'] == 228
    assert native_proof['inspection_sha256'] == sha(inspection_path) and native_proof['ledger_sha256'] == sha(ledger_path)
    assert sha(native_folder / 'OFFSERVER_VERIFICATION.json') == native_proof['offserver_proof_sha256']
    old_inspection_path = ROOT / prior_proof['native_inspection_path']
    old_inspection = pin(old_inspection_path, prior_proof['native_inspection_sha256'])
    old_ledger_path = old_inspection_path.parent.parent / 'verified_ledger.json'
    old_ledger = pin(old_ledger_path, native_proof['previous_ledger_sha256'])
    assert len(old_ledger['entries']) == 33 and len(ledger['entries']) == 34
    assert ledger['entries'][:-1] == old_ledger['entries'] and ledger['manifest_sha256'] == old_ledger['manifest_sha256']
    assert len(old_inspection['records']) == 320 and len(inspection['records']) == 328
    assert [r for r in inspection['records'] if r['id'] not in EXPECTED] == old_inspection['records']
    assert inspection['records'][:220] == old_inspection['records'][:220]
    assert inspection['records'][228:] == old_inspection['records'][220:]
    native_rows = {r['id']: r for r in inspection['records']}
    assert len(native_rows) == 328
    baseline_path = ROOT / 'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json'
    baseline = pin(baseline_path, '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd')
    full_rows = {r['id']: r for r in baseline['records']}
    native_archive = Path(handoff['native228_archive_path'])
    assert sha(native_archive) == native_proof['archive_sha256'] == handoff['native228_archive_sha256'] == '3b794119e28381ed7ad209e261fe3f3dc51957888e8517a2400bc9e588526ecc'
    extract = Path(storage['offserver_verification']).parent / 'verified_extract'
    bindings = {}; binding_files = {}; artifacts = {}; member_checks = []
    with tarfile.open(native_archive, 'r:gz') as archive:
        inv = json.load(archive.extractfile('backup_inventory.json'))
        assert inv['accepted_new_ids'] == EXPECTED
        for rec in records:
            identity = rec['id']; runtime = extract / 'runtime' / identity; run_dir = extract / 'runs' / identity
            binding = read(runtime / 'binding.json'); r = binding['record']; native = native_rows[identity]
            assert binding['status'] == 'IMMUTABLE_TERMINAL_ONCE_BOUND' and r['accepted_v4_row'] == binding['native_acceptance']['row'] == native
            assert binding['checkpoint_sha256'] == r['checkpoint']['sha256'] == rec['checkpoint_sha256'] == native['checkpoint_sha256']
            assert (r['variant'], r['distribution'], r['attack'], r['actual_alpha'], r['terminal_round'], r['original_split'], r['original_n_eval']) == ('minus_A', 'IID', 'FedSA', 5000, 70, 'valid', 19867)
            assert r['seed'] == int(identity[-5:]) and r['config']['ablation_component'] == 'A' and canonical(r['config']) == r['config_canonical_sha256']
            payloads = {}
            for name, wanted in (('model.pt', r['checkpoint']['sha256']), ('result.json', r['result']['sha256'])):
                member = 'runs/' + identity + '/' + name; payload = archive.extractfile(member).read()
                actual = hashlib.sha256(payload).hexdigest()
                assert actual == wanted == inv['members'][member]['sha256'] and len(payload) == inv['members'][member]['bytes']
                member_checks.append(dict(member=member, sha256=actual, bytes=len(payload)))
                if name == 'result.json': payloads[name] = json.loads(payload)
            result = payloads['result.json']; revision = result['revision_job']
            assert result['config'] == r['config'] and result['rounds'] == 70 and result['seed'] == r['seed']
            assert revision['source_hashes'] == r['source_hashes'] and result['data_contract']['image_data_contract'] == r['data_contract']
            assert inv['members']['jobs/' + identity + '.json']['sha256'] == r['raw_job']['sha256']
            for item in (r['checkpoint'], r['result'], r['raw_job']):
                assert item['sha256'] in native['files'].values()
            paired = r['paired_full']; full = full_rows[paired['id']]
            assert (full['distribution'], full['attack'], full['seed']) == (r['distribution'], r['attack'], r['seed'])
            assert paired['baseline_record_canonical_sha256'] == canonical(full)
            assert all(paired[key + '_sha256'] == full[key]['sha256'] for key in ('checkpoint', 'result', 'raw_job'))
            assert not paired['replay_required_here'] and not paired['weights_repacked_here']
            strict = read(run_dir / 'strict_acceptance.json'); bridge = read(run_dir / 'bridge_receipt.json'); sci = read(run_dir / 'receipt.json')
            approval = read(runtime / 'APPROVED.json'); complete = read(runtime / 'REMOTE_COMPLETE.json'); inventory = read(runtime / 'inventory.json')
            assert strict['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['id'] == identity
            assert strict['views'] == sci['views'] == rec['views']
            assert bridge['views'] == list(strict['views'])
            assert strict['checkpoint_sha256'] == sci['checkpoint_sha256'] == complete['checkpoint_sha256'] == r['checkpoint']['sha256']
            assert strict['native_comparison']['tolerance'] == bridge['native_tolerance'] == approval['native_tolerance'] == 1e-12
            assert strict['native_comparison']['accepted'] and strict['native_comparison']['max_abs_difference'] == rec['native_max_abs_difference'] == complete['native_difference'] == 0
            assert strict['native_comparison']['expected'] == r['prior_validation_metrics']
            assert strict['bridge_receipt_sha256'] == sha(run_dir / 'bridge_receipt.json')
            assert bridge['scientific_body_receipt_sha256'] == sha(run_dir / 'receipt.json')
            assert bridge['source_before'] == bridge['source_after'] and bridge['artifact_before'] == bridge['artifact_after']
            assert sha(runtime / 'binding.json') == complete['binding_sha256'] == approval['binding_sha256']
            assert sha(runtime / 'inventory.json') == complete['inventory_sha256'] == strict['inventory_sha256'] == bridge['inventory_sha256'] == approval['inventory_sha256']
            assert sha(runtime / 'APPROVED.json') == bridge['approval_sha256'] and sha(run_dir / 'strict_acceptance.json') == complete['strict_sha256']
            assert inventory['records'] == [r] and approval['selected_ids'] == [identity]
            assert approval['device'] == 'cpu' and approval['compute_threads'] == 8 and approval['max_processes'] == 1 and approval['allowed_cpus'] == list(range(112, 120))
            assert sci['original_result_sha256'] == r['result']['sha256'] and sci['original_job_sha256'] == r['raw_job']['sha256'] and sci['config_canonical_sha256'] == r['config_canonical_sha256']
            assert sci['weights_before'] == sci['weights_after'] and not sci['optimizer_created'] and not sci['gradients_created']
            assert sci['valid_n'] == 19867 and sci['valid_image_ids_sha256'] == r['data_contract']['evaluation_image_ids_sha256']
            assert sci['root_reconstruction']['root_n'] == 16277 and sci['root_reconstruction']['root_image_ids_sha256'] == r['data_contract']['root_image_ids_sha256']
            assert sci['root_reconstruction']['client_sample_counts'] == r['data_contract']['client_sample_counts']
            assert (rec['independent_metric_checks'], rec['independent_confusion_count_checks'], rec['prediction_rule_checks']) == (9, 24, 3)
            assert complete['accepted_offserver'] == complete['new_training'] == complete['new_Full_inference'] == strict['new_training'] == 0
            assert not complete['test'] and not strict['test_inference'] and not sci['test_inference_performed']
            assert sha(run_dir / 'validation_predictions.npz') == rec['prediction_arrays_sha256'] == sci['prediction_arrays_sha256'] == complete['prediction_arrays_sha256']
            bindings[identity] = binding; binding_files[identity] = dict(path=str(runtime / 'binding.json'), sha256=sha(runtime / 'binding.json'))
            artifacts[identity] = {name: dict(path=str(path), sha256=sha(path)) for name, path in (
                ('scientific_receipt', run_dir / 'receipt.json'), ('bridge_receipt', run_dir / 'bridge_receipt.json'), ('strict_json', run_dir / 'strict_acceptance.json'),
                ('delegated_approval', runtime / 'APPROVED.json'), ('bound_inventory', runtime / 'inventory.json'), ('remote_complete', runtime / 'REMOTE_COMPLETE.json'))}
    assert len(member_checks) == 16
    all_ids = prior['all_ids'] + EXPECTED
    assert len(all_ids) == len(set(all_ids)) == 228 and all_ids[:220] == prior['all_ids']
    native_archives = [dict(path=str(native_archive), sha256=sha(native_archive), root_verification_path=native_root_path.relative_to(ROOT).as_posix(), root_verification_sha256=sha(native_root_path))]
    index = dict(status='PROPOSED228_REPLAY_ID_INDEX_PENDING_ROOT_ADOPTION', prior_index_path=prior_path.relative_to(ROOT).as_posix(), prior_index_sha256=sha(prior_path),
        prior_adoption_path=prior_proof_path.relative_to(ROOT).as_posix(), prior_adoption_sha256=sha(prior_proof_path), all_ids=all_ids, new_ids=EXPECTED, new_records=records,
        new_bindings=bindings, new_binding_files=binding_files, new_artifacts=artifacts, new_archive=storage, native_inspection_path=inspection_path.relative_to(ROOT).as_posix(),
        native_inspection_sha256=sha(inspection_path), native_archives=native_archives, native_members_rehashed=16, Full_inference=0, test=False, root_adopted=False)
    save('MECHANISM228_INDEX.json', index)
    proof = dict(status='INDEPENDENT_A28_IDENTITY_RESTORE_CHAIN_PASS_ROOT_ADOPTABLE_NO_ADOPTION', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), blocking_findings=[],
        selected_ids=EXPECTED, prior_accepted=220, checked_new_ids=8, proposed_cumulative=228, actual_adoption_performed=False, accepted_offserver_by_this_review=0,
        proposed_index_path=(HERE / 'MECHANISM228_INDEX.json').relative_to(ROOT).as_posix(), proposed_index_sha256=sha(HERE / 'MECHANISM228_INDEX.json'),
        delivery_seal_sha256=sha(DELIVERY / 'FILES_SHA256.json'), delivery_files_checked=32, handoff_sha256=sha(DELIVERY / 'HANDOFF.json'), source_sha256=sha(Path(__file__)),
        original_A12_join_source_sha256=sha(ROOT / 'tmp/adopt_mechanism_A12_root_20261010.py'), original_archive_verifier_sha256=sha(original), archive_check=archive_check,
        archive_sha256=storage['archive_sha256'], receipt_sha256=sha(storage['receipt']), offserver_sha256=sha(storage['offserver_verification']), native_archives=native_archives,
        native_inspection_sha256=sha(inspection_path), native_ledger_sha256=sha(ledger_path), ledger_entries=34, prior33_ledger_prefix_exact=True, old320_native_JSON_records_and_relative_order_exact=True,
        old220_native_control_prefix_exact=True, Full100_native_tail_exact=True, prior220_index_prefix_exact=True, new_source_config_data_checkpoint_receipt_identities_exact=8,
        native_model_result_members_rehashed=16, native_member_checks=member_checks, baseline_Full_reference_joins=8, Full_weights_read_or_repacked=0,
        original_saved_check_counts_bound_not_recomputed=dict(metrics=72, confusion_counts=192, prediction_rules=24), native_max_abs_difference=0, native_tolerance=1e-12,
        new_CNN=0, new_fit=0, torch_imported=False, arrays_loaded=False, statistics_created=False, shared_state_or_ledger_or_Git_modified=False,
        partial_A_scene=dict(distribution='IID', attack='FedSA', seeds=list(range(91001, 91009)), n=8, target_n=10),
        limits=['The proposed index requires independent root adoption; prior accepted220 remains unchanged.', 'FedSA8/10 has no scenario mean or A100 claim.', 'This review binds the existing72/192/24 saved-array checks; it does not recompute arrays, fit, metrics or statistics.', 'Validation-only; historical selection, test exposure, calibration and mixed runtime/device limitations remain.'])
    save('REVIEW.json', proof)
    print(json.dumps(dict(status=proof['status'], index_sha256=proof['proposed_index_sha256'], review_sha256=sha(HERE / 'REVIEW.json'), new=8, proposed=228)))

if __name__ == '__main__':
    try:
        run()
    except BaseException as exc:
        failure = dict(status='STOPPED_NO_ADOPTION', error=repr(exc), traceback=traceback.format_exc())
        if not (HERE / 'FAILURE_V2.json').exists(): save('FAILURE_V2.json', failure)
        raise
