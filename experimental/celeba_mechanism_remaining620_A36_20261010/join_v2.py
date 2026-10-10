"""One local, record-only A36 join; proposes an index and never adopts it."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DELIVERY = ROOT / 'tmp/celeba_mechanism_remaining620_A36_20261010'
EXPECTED = ['minus_A_IID_FedSA_seed91009', 'minus_A_IID_FedSA_seed91010', 'minus_A_IID_S-DFA_seed91001', 'minus_A_IID_S-DFA_seed91002', 'minus_A_IID_S-DFA_seed91003', 'minus_A_IID_S-DFA_seed91004', 'minus_A_IID_S-DFA_seed91005', 'minus_A_IID_S-DFA_seed91006']
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
    seal = pin(DELIVERY / 'FILES_SHA256.json', '0b15ff51b4d173fd1a0057dba1344154c59ca418faf03be428219879ae9a3a46')
    assert len(seal['files']) == 45
    for name, row in seal['files'].items():
        p = DELIVERY / name
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name
    handoff = pin(DELIVERY / 'HANDOFF.json', '308c08666f36d9baaf007de76f95e1957e4ed4b5a44a05f9eff242f1c0e808b9')
    storage = pin(DELIVERY / 'RAW_STORAGE_INDEX.json', 'ad24141572607833032236f3f56b10ef2567f215592e13d8f7707157220ae4f2')
    assert handoff['selected_ids'] == storage['accepted_new_ids'] == EXPECTED
    assert storage['accepted_offserver'] == storage['root_adopted'] == 0
    assert (storage['metrics'], storage['counts'], storage['rules'], storage['archive_members']) == (72, 192, 24, 74)
    receipt = pin(storage['receipt'], '44ff19d96074432a154788c3b27dfa982815087ea5d000888614d0e0711f50cc')
    off = pin(storage['offserver_verification'], 'b13d26ac5024660e2c0448dbfc84f11e7f8b49496d98e07203acefa816c304ab')
    assert off['actual_transport_archive_sha256'] == receipt['archive_sha256'] == storage['archive_sha256'] == '5eaa96193bb40d2aac6660e49b6065805861985d711ee85cbacb655cc6def8e4'
    assert off['receipt_sha256'] == storage['receipt_sha256'] == sha(storage['receipt'])
    assert off['accepted_offserver'] == 0 and off['root_adoption_pending'] is True
    assert off['source_seal_sha256'] == receipt['source_seal_sha256'] == handoff['source_seal_sha256'] == 'a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
    assert receipt['transport_source_seal_sha256'] == handoff['transport_seal_sha256'] == '1a021b707575292c33959c19fcfa2fa1ee8c7f285d20576c562e4843d1488fb3'
    assert sha(storage['archive_member_manifest']) == receipt['inventory_sha256'] == storage['archive_member_manifest_sha256']
    original = ROOT / 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
    assert sha(original) == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
    spec = importlib.util.spec_from_file_location('original_A36_archive_verifier', original)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    archive_check = verifier.verify_archive(Path(storage['archive']), receipt)
    assert archive_check['members_verified'] == 74 and archive_check['different_host_observed']
    saved = off['original_saved_array_verification']; records = saved['records']
    assert saved['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
    assert [r['id'] for r in records] == EXPECTED
    assert (saved['independent_metric_checks'], saved['independent_confusion_count_checks'], saved['prediction_rule_checks']) == (72, 192, 24)
    prior_proof_path = Path(handoff['prior_root']); prior_path = Path(handoff['prior_index'])
    prior_proof = pin(prior_proof_path, 'f6761dd1c2b844724aee91d235d71f6474511ac8aec63f3dc4fe0308272e0966')
    prior = pin(prior_path, '765ea715defea1e54aebbee0115f38c926ee022f47f520d151d1417c6c8592a9')
    assert prior_proof['cumulative_accepted'] == len(prior['all_ids']) == 228
    assert prior_proof['records_index_sha256'] == sha(prior_path)
    assert not set(EXPECTED) & set(prior['all_ids'])
    previous = read(ROOT / 'tmp/celeba_remaining620_A28_transport_20261010/RAW_STORAGE_INDEX.json')
    assert sha(previous['receipt']) == storage['previous_receipt_sha256'] == receipt['previous_backup_receipt_sha256'] == '977c97a1c05ba78bc25b7402fb5c313dcd9c286b140ce4dbf9129bf7bd92aa30'
    assert sha(previous['offserver_verification']) == handoff['prior_offserver_sha256']
    assert storage['all_transported_ids'] == receipt['all_transported_ids'] == previous['all_transported_ids'] + EXPECTED
    native_root_path = Path(handoff['native236_root']); native_folder = native_root_path.parent
    native_proof = pin(native_root_path, '6d9594230da5f11b26a3df74db1b698811e8f6cdbb3368a9b467cdf92b2467cd')
    inspection_path = native_folder / 'inspection/inspection.json'
    inspection = pin(inspection_path, '934efbe079005b272cd6bf5ade3168aefde14895d24bcc8e9b66c43ad3206a31')
    ledger_path = native_folder / 'verified_ledger.json'
    ledger = pin(ledger_path, '79e7b57db51dd23477715b102ee24afbbcf95f926b03aaa92285767da12321fa')
    assert native_proof['new_ids'] == EXPECTED and native_proof['total_new_strict_and_offserver'] == 236
    assert native_proof['inspection_sha256'] == sha(inspection_path) and native_proof['ledger_sha256'] == sha(ledger_path)
    assert sha(native_folder / 'OFFSERVER_VERIFICATION.json') == native_proof['offserver_proof_sha256']
    old_inspection_path = (ROOT / prior_proof['native_root_path']).parent / 'inspection/inspection.json'
    old_inspection = pin(old_inspection_path, prior_proof['native_inspection_sha256'])
    old_ledger_path = old_inspection_path.parent.parent / 'verified_ledger.json'
    old_ledger = pin(old_ledger_path, native_proof['previous_ledger_sha256'])
    assert len(old_ledger['entries']) == 34 and len(ledger['entries']) == 35
    assert ledger['entries'][:-1] == old_ledger['entries'] and ledger['manifest_sha256'] == old_ledger['manifest_sha256']
    assert len(old_inspection['records']) == 328 and len(inspection['records']) == 336
    assert [r for r in inspection['records'] if r['id'] not in EXPECTED] == old_inspection['records']
    assert inspection['records'][:228] == old_inspection['records'][:228]
    assert inspection['records'][236:] == old_inspection['records'][228:]
    native_rows = {r['id']: r for r in inspection['records']}
    assert len(native_rows) == 336
    baseline_path = ROOT / 'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json'
    baseline = pin(baseline_path, '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd')
    full_rows = {r['id']: r for r in baseline['records']}
    native_archive = Path(handoff['native236_archive_path'])
    assert sha(native_archive) == native_proof['archive_sha256'] == handoff['native236_archive_sha256'] == 'c578f6dc58566bf53bbab8863e42d962213b7b70462974a5b7b4daab24e322ee'
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
            assert (r['variant'], r['distribution'], r['attack'], r['actual_alpha'], r['terminal_round'], r['original_split'], r['original_n_eval']) == ('minus_A', 'IID', 'FedSA' if identity in EXPECTED[:2] else 'S-DFA', 5000, 70, 'valid', 19867)
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
    assert len(all_ids) == len(set(all_ids)) == 236 and all_ids[:228] == prior['all_ids']
    native_archives = [dict(path=str(native_archive), sha256=sha(native_archive), root_verification_path=native_root_path.relative_to(ROOT).as_posix(), root_verification_sha256=sha(native_root_path))]
    index = dict(status='PROPOSED236_REPLAY_ID_INDEX_PENDING_ROOT_ADOPTION', prior_index_path=prior_path.relative_to(ROOT).as_posix(), prior_index_sha256=sha(prior_path),
        prior_adoption_path=prior_proof_path.relative_to(ROOT).as_posix(), prior_adoption_sha256=sha(prior_proof_path), all_ids=all_ids, new_ids=EXPECTED, new_records=records,
        new_bindings=bindings, new_binding_files=binding_files, new_artifacts=artifacts, new_archive=storage, native_inspection_path=inspection_path.relative_to(ROOT).as_posix(),
        native_inspection_sha256=sha(inspection_path), native_archives=native_archives, native_members_rehashed=16, Full_inference=0, test=False, root_adopted=False)
    save('MECHANISM236_INDEX.json', index)
    proof = dict(status='INDEPENDENT_A36_IDENTITY_RESTORE_CHAIN_PASS_ROOT_ADOPTABLE_NO_ADOPTION', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), blocking_findings=[],
        selected_ids=EXPECTED, prior_accepted=228, checked_new_ids=8, proposed_cumulative=236, actual_adoption_performed=False, accepted_offserver_by_this_review=0,
        proposed_index_path=(HERE / 'MECHANISM236_INDEX.json').relative_to(ROOT).as_posix(), proposed_index_sha256=sha(HERE / 'MECHANISM236_INDEX.json'),
        delivery_seal_sha256=sha(DELIVERY / 'FILES_SHA256.json'), delivery_files_checked=45, handoff_sha256=sha(DELIVERY / 'HANDOFF.json'), source_sha256=sha(Path(__file__)),
        original_A12_join_source_sha256=sha(ROOT / 'tmp/adopt_mechanism_A12_root_20261010.py'), original_archive_verifier_sha256=sha(original), archive_check=archive_check,
        archive_sha256=storage['archive_sha256'], receipt_sha256=sha(storage['receipt']), offserver_sha256=sha(storage['offserver_verification']), native_archives=native_archives,
        native_inspection_sha256=sha(inspection_path), native_ledger_sha256=sha(ledger_path), ledger_entries=35, prior34_ledger_prefix_exact=True, old328_native_JSON_records_and_relative_order_exact=True,
        old228_native_control_prefix_exact=True, Full100_native_tail_exact=True, prior228_index_prefix_exact=True, new_source_config_data_checkpoint_receipt_identities_exact=8,
        native_model_result_members_rehashed=16, native_member_checks=member_checks, baseline_Full_reference_joins=8, Full_weights_read_or_repacked=0,
        original_saved_check_counts_bound_not_recomputed=dict(metrics=72, confusion_counts=192, prediction_rules=24), native_max_abs_difference=0, native_tolerance=1e-12,
        new_CNN=0, new_fit=0, torch_imported=False, arrays_loaded=False, statistics_created=False, shared_state_or_ledger_or_Git_modified=False,
        partial_A_scene=dict(distribution='IID', attack='S-DFA', seeds=list(range(91001, 91007)), n=6, target_n=10),
        limits=['The proposed index requires independent root adoption; prior accepted228 remains unchanged.', 'S-DFA6/10 has no scenario mean or A100 claim.', 'This review binds the existing72/192/24 saved-array checks; it does not recompute arrays, fit, metrics or statistics.', 'Validation-only; historical selection, test exposure, calibration and mixed runtime/device limitations remain.'])
    save('REVIEW.json', proof)
    print(json.dumps(dict(status=proof['status'], index_sha256=proof['proposed_index_sha256'], review_sha256=sha(HERE / 'REVIEW.json'), new=8, proposed=236)))

if __name__ == '__main__':
    try:
        run()
    except BaseException as exc:
        failure = dict(status='STOPPED_NO_ADOPTION', error=repr(exc), traceback=traceback.format_exc())
        if not (HERE / 'FAILURE_V3.json').exists(): save('FAILURE_V3.json', failure)
        raise
