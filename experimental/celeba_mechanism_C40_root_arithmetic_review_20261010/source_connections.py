"""Only the new C4 archive and ten new Full table references; reuse prior proof."""
import hashlib,json,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent; R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def verify_connections(records,bindings,adopted,adoption):
    prior=R/'tmp/celeba_mechanism_C30_root_arithmetic_review_20261009/SOURCE_CONNECTION_REVIEW.json'
    assert sha(prior)=='e53dbbd8b8e7e4757d0104d2744af23edae50d41db749585f8b0bf6147915996'
    assert read(prior)['status']=='INDEPENDENT_C30_ARCHIVE_ROOT_FULL900_CONNECTION_PASS_NO_ADOPTION'
    invpath=R/'tmp/celeba_mechanism_valid_C_after36_20261010/inventory_actual140_Full100refs.json'
    assert sha(invpath)=='ddda34996da47f176520e782ca03ebf536bda7b1a361b6e34b70092e660299c1'
    inv=read(invpath); native={r['id']:r for r in inv['records']}; by={r['id']:r for r in records}
    ids=[f'minus_C_IID_S-DFA_seed{s}' for s in range(91007,91011)]
    chain=bindings['new_archive_chain']; archive=Path(chain['archive'])
    assert archive.parent==adoption.parent and chain['accepted_ids']==ids
    assert sha(archive)==chain['archive_sha256']==adopted['archive_sha256']
    assert chain['root_adoption_sha256']==sha(adoption)
    receiptpath=archive.parent/'backup_receipt.json'; receipt=read(receiptpath)
    offpath=archive.parent/'OFFSERVER_VERIFICATION.json'
    assert adopted['backup_receipt_sha256']==sha(receiptpath) and adopted['offserver_verification_sha256']==sha(offpath)
    assert receipt['archive_sha256']==sha(archive) and receipt['accepted_new_ids']==ids
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers(); assert len(members)==len({m.name for m in members})==61==chain['members_verified']
        assert all(m.isfile() for m in members)
        data=tar.extractfile('backup_inventory.json').read(); inventory_sha=hashlib.sha256(data).hexdigest(); inventory=json.loads(data)
        assert receipt['inventory_sha256']==inventory_sha and set(tar.getnames())==set(inventory['members'])|{'backup_inventory.json'}
        for name,pin in inventory['members'].items():
            raw=tar.extractfile(name).read(); assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
        saved={r['id']:r for r in inventory['records']}; assert set(saved)==set(ids)
        for id in ids:
            record=by[id]; original=native[id]; provenance=record['provenance']
            raw=tar.extractfile(provenance['receipt_member']).read(); scientific=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==provenance['receipt_sha256'] and scientific['id']==id
            assert record['checkpoint_sha256']==scientific['checkpoint_sha256']==saved[id]['checkpoint_sha256']==original['checkpoint']['sha256']
            assert record['config_sha256']==scientific['config_canonical_sha256']==original['config_canonical_sha256']
            assert record['views']==scientific['views']==saved[id]['views'] and record['fits']==scientific['fits']
            assert scientific['original_result_sha256']==original['result']['sha256'] and scientific['original_job_sha256']==original['raw_job']['sha256']
            assert scientific['root_reconstruction']['root_image_ids_sha256']==original['data_contract']['root_image_ids_sha256']
            assert scientific['valid_n']==19867 and scientific['valid_image_ids_sha256']==original['data_contract']['evaluation_image_ids_sha256']
            assert scientific['prediction_arrays_sha256']==provenance['prediction_arrays_sha256'] and saved[id]['strict_acceptance_sha256']==provenance['strict_acceptance_sha256']
            assert inventory['original_inventory_sha256']==provenance['inventory_sha256']==sha(invpath)
            assert record['data_contract']==original['data_contract'] and (original['terminal_round'],original['original_split'],original['original_n_eval'])==(70,'valid',19867)
            assert scientific['native_comparison']['accepted'] and scientific['native_comparison']['tolerance']==1e-12 and saved[id]['native_difference']==0
            assert scientific['weights_before']==scientific['weights_after'] and not scientific['optimizer_created'] and not scientific['gradients_created'] and not scientific['test_inference_performed']
    full=[r for r in records if r['variant']=='Full' and r['attack']=='S-DFA']; assert len(full)==10
    path=Path(full[0]['provenance']['accepted900_path']); full_sha=sha(path)
    assert full_sha=='983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529'
    source={r['id']:r for r in read(path)['records']}; refs={r['id']:r for r in inv['full_references']}
    for record in full:
        original=source[record['id']]; ref=refs[record['id']]; baseline=original['original_inventory_record']
        for key in ('checkpoint_sha256','views','fits','training_torch'): assert record[key]==original[key]
        assert record['config_sha256']==original['config_canonical_sha256'] and record['replay_runtime']==original['runtime'] and record['data_contract']==baseline['data_contract']
        for key in ('receipt_sha256','prediction_arrays_sha256','source_binding'): assert record['provenance'][key]==original[key]
        assert record['provenance']['accepted900_sha256']==full_sha and original['model_inventory_record_sha256']==ref['baseline_record_canonical_sha256']
        assert ref['checkpoint_sha256']==baseline['checkpoint']['sha256'] and ref['result_sha256']==baseline['result']['sha256'] and ref['raw_job_sha256']==baseline['raw_job']['sha256']
        assert ref['root_image_ids_sha256']==original['root_reconstruction']['root_image_ids_sha256']
        assert original['same_checkpoint_all_views'] and not original['test_evaluation_performed'] and original['native_comparison']['accepted'] and original['native_comparison']['tolerance']==1e-12
    return dict(prior_C30_source_proof_sha256=sha(prior),prior60_record_and_prior_S_DFA6_provenance_reused=True,
        new_archive_sha256=sha(archive),new_archive_members_verified=61,new_C_receipts_verified=4,new_C_adoption_sha256=sha(adoption),
        new_Full_table_references_verified=10,Full900_records_sha256=full_sha,old30_Full_references_reused=True,
        old900_models_rechecked=False,old_archive_members_rescanned=False,new_inference=0,threshold_refits=0,models_repacked=0)
