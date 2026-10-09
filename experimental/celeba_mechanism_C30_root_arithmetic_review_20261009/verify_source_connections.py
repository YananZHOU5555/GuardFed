"""Bounded C30 source connection supplement; no adoption or scientific recomputation."""
import collections, datetime, hashlib, json, sys, tarfile
from pathlib import Path
sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'tmp/celeba_mechanism_three_view_C_three_scenes_prepared_20261009'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
def main():
    output = HERE/'SOURCE_CONNECTION_REVIEW.json'
    assert not output.exists()
    handoff = read(BASE/'ACTUAL_HANDOFF.json')
    assert sha(BASE/'ACTUAL_HANDOFF.json') == '86b1a397e4c96adae15cf668b43b6ecbb3755bc669ab61bc669269889ecbf94f'
    assert sha(BASE/'ACTUAL_FILES_SHA256.json') == '46c405262070155a55dbeefd569d13af11d301113217f4b0f77f2437a981fef2'
    assert sha(HERE/'ROOT_ARITHMETIC_REVIEW.json') == '4c0816364469b16cdc1f76b6852633d5b4c9157af8161ae43b1cf2ada908e4a5'
    records = read(BASE/'snapshot/records.json')['records']
    partial = read(BASE/'snapshot/excluded_partial_C_records.json')['records']
    by = {r['id']:r for r in records+partial}; assert len(by) == 66
    invpath = ROOT/'tmp/celeba_mechanism_valid_C_after28_20261009/inventory_actual136_Full100refs.json'
    assert sha(invpath) == '7cf324b92be73420b7f4497673519c779666664d13a98ad93f4207434c1de635'
    inventory = read(invpath); native = {r['id']:r for r in inventory['records']}
    for record in records+partial:
        if record['variant'] != 'minus_C': continue
        original = native[record['id']]
        assert original['terminal_round'] == 70 and original['original_split'] == 'valid' and original['original_n_eval'] == 19867
        for k in ('distribution','attack','seed','data_contract'):
            assert record[k] == original[k]
        assert record['checkpoint_sha256'] == original['checkpoint']['sha256']
        assert record['config_sha256'] == original['config_canonical_sha256']
    archives = []; accepted = []
    for chain in handoff['new_archive_chains']:
        path = Path(chain['archive']); assert sha(path) == chain['archive_sha256']
        rootpath = path.parent/'ROOT_ADOPTION_REVIEW.json'; proof = read(rootpath)
        receiptpath = path.parent/'backup_receipt.json'; receipt = read(receiptpath)
        offpath = path.parent/'OFFSERVER_VERIFICATION.json'
        assert sha(rootpath) == chain['root_adoption_sha256']
        assert proof['archive_sha256'] == receipt['archive_sha256'] == chain['archive_sha256']
        assert proof['accepted_new_ids'] == receipt['accepted_new_ids'] == chain['accepted_ids']
        assert proof['backup_receipt_sha256'] == sha(receiptpath) and proof['offserver_verification_sha256'] == sha(offpath)
        with tarfile.open(path,'r:gz') as archive:
            members = archive.getmembers(); assert len(members) == len({m.name for m in members}) == chain['members_verified']
            assert all(m.isfile() for m in members)
            raw = archive.extractfile('backup_inventory.json').read(); inventory_sha = hashlib.sha256(raw).hexdigest()
            member_inventory = json.loads(raw); declared = member_inventory['members']
            assert set(m.name for m in members) == set(declared)|{'backup_inventory.json'}
            assert receipt['inventory_sha256'] == inventory_sha
            for name,pin in declared.items():
                data = archive.extractfile(name).read()
                assert hashlib.sha256(data).hexdigest() == pin['sha256'] and len(data) == pin['bytes']
            saved = {r['id']:r for r in member_inventory['records']}
            for id in chain['accepted_ids']:
                record = by[id]; provenance = record['provenance']; saved_record = saved[id]
                data = archive.extractfile(provenance['receipt_member']).read(); scientific = json.loads(data)
                assert hashlib.sha256(data).hexdigest() == provenance['receipt_sha256']
                assert scientific['id'] == id and scientific['checkpoint_sha256'] == record['checkpoint_sha256'] == saved_record['checkpoint_sha256']
                assert scientific['config_canonical_sha256'] == record['config_sha256']
                assert scientific['original_result_sha256'] == native[id]['result']['sha256'] and scientific['original_job_sha256'] == native[id]['raw_job']['sha256']
                assert scientific['root_reconstruction']['root_image_ids_sha256'] == native[id]['data_contract']['root_image_ids_sha256']
                assert scientific['valid_n'] == 19867 and scientific['valid_image_ids_sha256'] == native[id]['data_contract']['evaluation_image_ids_sha256']
                assert member_inventory['original_inventory_sha256'] == provenance['inventory_sha256']
                assert scientific['views'] == record['views'] == saved_record['views'] and scientific['fits'] == record['fits']
                assert scientific['prediction_arrays_sha256'] == provenance['prediction_arrays_sha256']
                assert saved_record['strict_acceptance_sha256'] == provenance['strict_acceptance_sha256']
                assert scientific['native_comparison']['accepted'] and scientific['native_comparison']['tolerance'] == 1e-12
                assert scientific['weights_before'] == scientific['weights_after'] and not scientific['optimizer_created'] and not scientific['gradients_created']
                assert not scientific['test_inference_performed'] and saved_record['native_difference'] == 0
                accepted.append(id)
        archives.append(dict(archive=str(path),archive_sha256=sha(path),root_sha256=sha(rootpath),receipt_sha256=sha(receiptpath),offserver_sha256=sha(offpath),members_verified=len(members),accepted_ids=chain['accepted_ids']))
    assert len(accepted) == len(set(accepted)) == 16
    assert {by[id]['attack'] for id in accepted} == {'FedSA','S-DFA'}
    full = [r for r in records if r['variant']=='Full']; reference = {r['id']:r for r in inventory['full_references']}
    path = Path(full[0]['provenance']['accepted900_path']); full900_sha = sha(path)
    assert full900_sha == '983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529'
    original900 = {r['id']:r for r in read(path)['records']}; assert len(original900) == 900
    for record in full:
        original = original900[record['id']]; ref = reference[record['id']]; native_record = original['original_inventory_record']
        assert record['provenance']['accepted900_sha256'] == full900_sha
        for key in ('checkpoint_sha256','views','fits','training_torch'):
            assert record[key] == original[key]
        assert record['config_sha256'] == original['config_canonical_sha256']
        assert record['data_contract'] == native_record['data_contract'] and record['replay_runtime'] == original['runtime']
        assert original['same_checkpoint_all_views'] and not original['test_evaluation_performed']
        assert original['native_comparison']['accepted'] and original['native_comparison']['tolerance'] == 1e-12
        assert original['model_inventory_record_sha256'] == ref['baseline_record_canonical_sha256']
        assert ref['checkpoint_sha256'] == native_record['checkpoint']['sha256'] == record['checkpoint_sha256']
        assert ref['result_sha256'] == native_record['result']['sha256'] and ref['raw_job_sha256'] == native_record['raw_job']['sha256']
        assert ref['root_image_ids_sha256'] == original['root_reconstruction']['root_image_ids_sha256']
        for key in ('receipt_sha256','prediction_arrays_sha256','source_binding'):
            assert record['provenance'][key] == original[key]
        for key in ('offserver_proof','archive_inventory'):
            pin = original['source_binding'].get(key)
            if pin and 'path' in pin: assert sha(ROOT/pin['path']) == pin['sha256']
    proof = dict(status='INDEPENDENT_C30_ARCHIVE_ROOT_FULL900_CONNECTION_PASS_NO_ADOPTION',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arithmetic_proof_sha256=sha(HERE/'ROOT_ARITHMETIC_REVIEW.json'),actual_handoff_sha256=sha(BASE/'ACTUAL_HANDOFF.json'),actual_delivery_seal_sha256=sha(BASE/'ACTUAL_FILES_SHA256.json'),
        archives=archives,new_archive_members_verified=sum(x['members_verified'] for x in archives),new_C_receipts_verified=16,new_C_complete_scene_records=10,excluded_S_DFA_receipts_verified=6,
        Full900_records_sha256=full900_sha,Full900_references_verified=len(full),actual136_inventory_sha256=sha(invpath),C_native_identity_records_verified=36,
        old40_records_reused_under_exact_arithmetic_regression=True,old_model_archives_repacked=False,old_model_archives_members_rescanned=False,
        canonical_modified=False,adoption_performed=False,new_CNN=0,new_training=0,test=False,primary_endpoint='PENDING_AUTHOR',source_sha256=sha(Path(__file__)))
    with output.open('x',encoding='utf-8') as stream: json.dump(proof,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output),status=proof['status'],members=proof['new_archive_members_verified'])))
if __name__ == '__main__': main()
