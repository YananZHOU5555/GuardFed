"""Join twelve original saved-output checks to the accepted native212 restore chain."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'docs/server_deployment_20260923/training_20260923'
HERE = ROOT / 'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010'
DELIVERY = ROOT / 'tmp/celeba_remaining620_A12_transport_20261010'
NATIVE = TRAIN / 'server_reactivation_20261009/mechanism_science_backups_20261009'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
sys.path.insert(0, str(ROOT / 'tmp'))
from guardfed_local_storage import check_bulk_storage

def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')

volume = check_bulk_storage()
expected = [f'minus_A_IID_Benign_seed{s}' for s in range(91001, 91011)] + [f'minus_A_IID_F Flip_seed{s}' for s in (91001, 91002)]
prepared = read(DELIVERY / 'PREPARED.json')
assert prepared['selected_ids'] == expected and prepared['actual_export'] is False
for name, wanted in prepared['invocation_sha256'].items():
    assert sha(DELIVERY / name) == wanted
storage = read(DELIVERY / 'RAW_STORAGE_INDEX.json')
assert storage['accepted_new_ids'] == expected and storage['accepted_offserver'] == storage['root_adopted'] == 0
assert (storage['metrics'], storage['counts'], storage['rules']) == (108, 288, 36)
assert storage['archive_sha256'] == sha(storage['archive']) == '970f43d8dfbc0eff46189e71245b6af982d23e72fe980fcd6191316edb02625a'
assert sha(storage['receipt']) == storage['receipt_sha256']
assert sha(storage['offserver_verification']) == storage['offserver_verification_sha256'] == 'b62949a65a140921c22fe56acbd0906d5288d1b4aee78152342801ac3b1e0092'
off = read(storage['offserver_verification'])
saved = off['original_saved_array_verification']
records = saved['records']
assert off['accepted_offserver'] == 0 and off['root_adoption_pending'] is True
assert saved['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
assert [r['id'] for r in records] == expected
assert off['source_seal_sha256'] == 'a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
original = ROOT / 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
assert sha(original) == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
spec = importlib.util.spec_from_file_location('original_A12_archive_verifier', original)
v4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v4)
archive_check = v4.verify_archive(Path(storage['archive']), read(storage['receipt']))
assert archive_check['members_verified'] == 110 and archive_check['different_host_observed']

prior_proof_path = ROOT / 'tmp/celeba_mechanism_remaining620_C100_root_adoption_20261010/ROOT_ADOPTION.json'
assert sha(prior_proof_path) == '2ac1d2f200d9671de5271f24b0cbb3a0772afb88c16ae6ccf71805ed1588ee46'
prior_proof = read(prior_proof_path)
assert prior_proof['cumulative_accepted'] == 200 and prior_proof['C100_table_adopted'] is False
prior_path = ROOT / prior_proof['records_index_path']
assert sha(prior_path) == prior_proof['records_index_sha256']
prior = read(prior_path)
assert len(prior['all_ids']) == 200 and not set(expected) & set(prior['all_ids'])
assert read(TRAIN / 'TRAINING_STATE.json')['celeba_mechanism_v1']['three_view_accepted_ids'] == prior['all_ids']
previous = read(ROOT / 'tmp/celeba_remaining620_C19_transport_20261010/RAW_STORAGE_INDEX.json')
assert sha(previous['receipt']) == storage['previous_receipt_sha256'] == previous['receipt_sha256']
assert storage['all_transported_ids'] == previous['all_transported_ids'] + expected

native_archives = []; native_by_id = {}
for tag, wanted in (
    ('root_delta_20261010T060527Z', '7179d0d832392cf9f322098d8d28f5c53e3beec90791d7ecd997fc659dca7683'),
    ('root_delta_20261010T062622Z', 'b608f2188e8ecca16e514344e613337eb315cd478e90946a6bbada54fe5dfbe9'),
):
    folder = NATIVE / tag; proof_path = folder / 'ROOT_DELTA_VERIFICATION.json'; proof = read(proof_path)
    assert sha(proof_path) == {
        'root_delta_20261010T060527Z': '796fa3fc3a7b673c9267696e6b3e122e85fef2a2ec8db0dfbd6fcddaeb18c415',
        'root_delta_20261010T062622Z': '566b44092d4261afc63ccd512765408b74820721eb327b3179c61ffde585ae48',
    }[tag]
    archive = Path(proof['archive_local_path'])
    assert sha(archive) == proof['archive_sha256'] == wanted
    assert sha(folder / 'OFFSERVER_VERIFICATION.json') == proof['offserver_proof_sha256']
    native_archives.append(dict(path=str(archive), sha256=wanted,
        root_verification_path=proof_path.relative_to(ROOT).as_posix(), root_verification_sha256=sha(proof_path)))
    for identity in proof['new_ids']:
        assert identity not in native_by_id
        native_by_id[identity] = archive
assert set(native_by_id) == set(expected)
inspection_path = NATIVE / 'root_delta_20261010T062622Z/inspection/inspection.json'
inspection_sha = sha(inspection_path)
assert inspection_sha == '13894351b4ea323204ce7bc389ad478cd3fdc746c973d5119de70ebf6b23c0fb'
native_rows = {r['id']: r for r in read(inspection_path)['records']}
assert len(native_rows) == 312
extract = Path(storage['offserver_verification']).parent / 'verified_extract'
bindings = {}; binding_files = {}; artifacts = {}; member_checks = 0
for rec in records:
    identity = rec['id']; runtime = extract / 'runtime' / identity; run = extract / 'runs' / identity
    binding = read(runtime / 'binding.json'); record = binding['record']; native = native_rows[identity]
    assert record['accepted_v4_row'] == native
    assert binding['checkpoint_sha256'] == record['checkpoint']['sha256'] == rec['checkpoint_sha256'] == native['checkpoint_sha256']
    assert record['variant'] == 'minus_A' and record['distribution'] == 'IID' and record['actual_alpha'] == 5000
    assert record['seed'] in range(91001, 91011) and record['terminal_round'] == 70
    assert record['original_split'] == 'valid' and record['original_n_eval'] == 19867
    assert record['paired_full']['replay_required_here'] is False and record['paired_full']['weights_repacked_here'] is False
    strict = read(run / 'strict_acceptance.json')
    assert strict['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['id'] == identity
    assert strict['checkpoint_sha256'] == rec['checkpoint_sha256'] and strict['new_training'] == 0 and strict['test_inference'] is False
    assert sha(run / 'bridge_receipt.json') == strict['bridge_receipt_sha256']
    assert rec['native_max_abs_difference'] == 0
    assert (rec['independent_metric_checks'], rec['independent_confusion_count_checks'], rec['prediction_rule_checks']) == (9, 24, 3)
    assert all(abs(x) <= 1e-12 for values in rec['differences'].values() for x in values.values())
    assert sha(run / 'validation_predictions.npz') == rec['prediction_arrays_sha256']
    with tarfile.open(native_by_id[identity], 'r:gz') as archive:
        inventory = json.load(archive.extractfile('backup_inventory.json'))
        for name, wanted in (('model.pt', rec['checkpoint_sha256']), ('result.json', record['result']['sha256'])):
            member = 'runs/' + identity + '/' + name
            assert inventory['members'][member]['sha256'] == hashlib.sha256(archive.extractfile(member).read()).hexdigest() == wanted
            member_checks += 1
    bindings[identity] = binding
    binding_files[identity] = dict(path=str(runtime / 'binding.json'), sha256=sha(runtime / 'binding.json'))
    artifacts[identity] = {n: dict(path=str(p), sha256=sha(p)) for n, p in (
        ('scientific_receipt', run / 'receipt.json'), ('bridge_receipt', run / 'bridge_receipt.json'),
        ('strict_json', run / 'strict_acceptance.json'), ('delegated_approval', runtime / 'APPROVED.json'),
        ('bound_inventory', runtime / 'inventory.json'), ('remote_complete', runtime / 'REMOTE_COMPLETE.json'))}
assert member_checks == 24
all_ids = prior['all_ids'] + expected
assert len(all_ids) == len(set(all_ids)) == 212
HERE.mkdir(exist_ok=False)
index = dict(status='ACCEPTED212_REPLAY_ID_INDEX', prior_index_path=prior_path.relative_to(ROOT).as_posix(),
    prior_index_sha256=sha(prior_path), prior_adoption_path=prior_proof_path.relative_to(ROOT).as_posix(),
    prior_adoption_sha256=sha(prior_proof_path), all_ids=all_ids, new_ids=expected, new_records=records,
    new_bindings=bindings, new_binding_files=binding_files, new_artifacts=artifacts,
    new_archive=storage, native_inspection_path=inspection_path.relative_to(ROOT).as_posix(),
    native_inspection_sha256=inspection_sha, native_archives=native_archives, native_members_rehashed=member_checks,
    Full_inference=0, test=False)
index_path = HERE / 'MECHANISM212_INDEX.json'; save(index_path, index)
proof = dict(status='ROOT_A12_SAVED_ARRAYS_AND_NATIVE212_RESTORE_CHAIN_ADOPTED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), accepted_new_ids=expected, new_accepted=12,
    prior_accepted=200, cumulative_accepted=212, remaining620_new_accepted=32,
    records_index_path=index_path.relative_to(ROOT).as_posix(), records_index_sha256=sha(index_path),
    archive_path=storage['archive'], archive_sha256=storage['archive_sha256'], archive_members=110,
    offserver_proof_path=storage['offserver_verification'], offserver_proof_sha256=storage['offserver_verification_sha256'],
    native_inspection_path=inspection_path.relative_to(ROOT).as_posix(), native_inspection_sha256=inspection_sha,
    native_archives=native_archives, exact_native_records_checked=12, native_members_rehashed=24,
    independent_metrics=108, independent_counts=288, prediction_rules=36, native_max_abs_difference=0,
    original200_unchanged=True, source_seal_sha256=off['source_seal_sha256'],
    complete_A_scenes=[['IID', 'Benign']], partial_A_scenes=[dict(distribution='IID', attack='F Flip', seeds=[91001, 91002])],
    A_table_adopted=False, Full_inference=0, new_CNN=0, new_training=0, test=False,
    whole_rebuttal_complete=False, fresh_F_volume=volume)
save(HERE / 'ROOT_ADOPTION.json', proof)
print(json.dumps(dict(status=proof['status'], root_adoption_sha256=sha(HERE / 'ROOT_ADOPTION.json'),
    index_sha256=sha(index_path), accepted=212, A_table_adopted=False)))
