"""Read the Sp7/Sp3 scientific receipts and ten new Full900 identities, no tensor oracle."""
import hashlib,json,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent; R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def verify_connections(records,bindings,adopted,adoption):
    prior=R/'tmp/celeba_mechanism_C40_root_arithmetic_review_20261010/ROOT_ARITHMETIC_REVIEW.json'
    assert sha(prior)=='17ff4cebf29bd7c80c06125fd925f02e2ba727964d217a2585565cfe94233803'
    assert read(prior)['status']=='INDEPENDENT_C40_FOUR_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
    inv150=R/'tmp/celeba_mechanism_valid_C_after47_20261010/inventory_actual150_Full100refs.json'
    inv147=R/'tmp/celeba_mechanism_valid_C_after40_20261010/inventory_actual147_Full100refs.json'
    assert sha(inv150)=='fc5f9b31aef29b1f3c43537301c4eacac40cd4d789114a3238f41e5c6d58cab4'
    assert sha(inv147)=='c65a6e8a88ef1e97a687bbf259840582c3d1f4c26e8e09deb604e94a4cc5b10f'
    current=read(inv150); previous=read(inv147)
    assert current['full_references']==previous['full_references']
    native={r['id']:r for r in current['records']}; before={r['id']:r for r in previous['records']}
    assert len(native)==150 and len(before)==147 and all(native[k]==v for k,v in before.items())
    by={r['id']:r for r in records}; chains=[]
    prior_adoption=R/'tmp/celeba_mechanism_valid_C_after40_20261010/execution_candidate/backups/incremental_20261009T233028Z/ROOT_ADOPTION_REVIEW.json'
    assert sha(prior_adoption)=='64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea'
    assert adopted['prior147_root_adoption_sha256']==sha(prior_adoption)
    for chain,path,invpath,ids,counts in [
        (bindings['prior_Sp_DFA7_archive_chain'],prior_adoption,inv147,[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001,91008)],(140,7,147)),
        (bindings['new_archive_chain'],adoption,inv150,[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91008,91011)],(147,3,150))]:
        proof=read(path); archive=Path(chain['archive'])
        assert (proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models'])==counts
        assert proof['accepted_new_ids']==chain['accepted_ids']==ids and proof['negative_results_preserved'] and proof['source_scope_complete']
        assert archive.resolve().parent==path.resolve().parent
        assert sha(archive)==chain['archive_sha256']==proof['archive_sha256']
        assert chain['root_adoption_sha256']==sha(path)
        receiptpath=path.parent/'backup_receipt.json'; receipt=read(receiptpath)
        offpath=path.parent/'OFFSERVER_VERIFICATION.json'
        assert proof['backup_receipt_sha256']==sha(receiptpath) and proof['offserver_verification_sha256']==sha(offpath)
        assert receipt['archive_sha256']==sha(archive) and receipt['accepted_new_ids']==ids
        assert proof['all_native_differences_zero'] and proof['server_strict_bound_in_saved_receipts']
        assert proof['new_training']==proof['new_Full_inference']==0 and proof['test_inference'] is False
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers(); assert len(members)==len({m.name for m in members})==chain['members_verified']==proof['archive_members_verified']
            assert len(members)==(7*len(ids)+10+11+12) and all(m.isfile() for m in members)
            raw=tar.extractfile('backup_inventory.json').read(); inventory=json.loads(raw)
            assert hashlib.sha256(raw).hexdigest()==receipt['inventory_sha256']
            assert set(tar.getnames())==set(inventory['members'])|{'backup_inventory.json'}
            saved={r['id']:r for r in inventory['records']}; assert set(saved)==set(ids)
            for key in ids:
                record=by[key]; original=native[key]; provenance=record['provenance']
                raw=tar.extractfile(provenance['receipt_member']).read(); scientific=json.loads(raw)
                assert hashlib.sha256(raw).hexdigest()==provenance['receipt_sha256']==inventory['members'][provenance['receipt_member']]['sha256']
                assert scientific['id']==key
                assert record['checkpoint_sha256']==scientific['checkpoint_sha256']==saved[key]['checkpoint_sha256']==original['checkpoint']['sha256']
                assert record['config_sha256']==scientific['config_canonical_sha256']==original['config_canonical_sha256']
                assert record['views']==scientific['views']==saved[key]['views'] and record['fits']==scientific['fits']
                assert scientific['original_result_sha256']==original['result']['sha256'] and scientific['original_job_sha256']==original['raw_job']['sha256']
                assert scientific['root_reconstruction']['root_image_ids_sha256']==original['data_contract']['root_image_ids_sha256']
                assert scientific['valid_n']==19867 and scientific['valid_image_ids_sha256']==original['data_contract']['evaluation_image_ids_sha256']
                assert scientific['prediction_arrays_sha256']==provenance['prediction_arrays_sha256']
                assert saved[key]['strict_acceptance_sha256']==provenance['strict_acceptance_sha256']
                assert inventory['original_inventory_sha256']==provenance['inventory_sha256']==sha(invpath)
                assert record['data_contract']==original['data_contract']
                assert (original['terminal_round'],original['original_split'],original['original_n_eval'])==(70,'valid',19867)
                assert scientific['native_comparison']['accepted'] and scientific['native_comparison']['tolerance']==1e-12 and saved[key]['native_difference']==0
                assert scientific['weights_before']==scientific['weights_after']
                assert not scientific['optimizer_created'] and not scientific['gradients_created'] and not scientific['test_inference_performed']
                full=next(r for r in records if (r['variant'],r['distribution'],r['attack'],r['seed'])==('Full','IID','Sp-DFA',original['seed']))
                assert full['data_contract']==record['data_contract'] and original['paired_full']['id']==full['id']
                for m in ('accuracy','aeod','aspd'): assert abs(record['views']['native'][m]-original['prior_validation_metrics'][m])<=1e-12
        chains.append(dict(root_adoption_sha256=sha(path),archive_sha256=sha(archive),archive_members=chain['members_verified'],scientific_receipts_connected=len(ids),accepted_ids=ids))
    full=[r for r in records if r['variant']=='Full' and r['attack']=='Sp-DFA']; assert len(full)==10
    path=R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json'
    full_sha=sha(path); assert full_sha=='983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529'
    source={r['id']:r for r in read(path)['records']}; refs={r['id']:r for r in current['full_references']}
    for record in full:
        original=source[record['id']]; ref=refs[record['id']]; baseline=original['original_inventory_record']
        assert Path(record['provenance']['accepted900_path']).resolve()==path.resolve()
        for key in ('checkpoint_sha256','views','fits','training_torch'): assert record[key]==original[key]
        assert record['config_sha256']==original['config_canonical_sha256'] and record['replay_runtime']==original['runtime'] and record['data_contract']==baseline['data_contract']
        for key in ('receipt_sha256','prediction_arrays_sha256','source_binding'): assert record['provenance'][key]==original[key]
        assert record['provenance']['accepted900_sha256']==full_sha and original['model_inventory_record_sha256']==ref['baseline_record_canonical_sha256']
        assert ref['checkpoint_sha256']==baseline['checkpoint']['sha256'] and ref['result_sha256']==baseline['result']['sha256'] and ref['raw_job_sha256']==baseline['raw_job']['sha256']
        assert ref['root_image_ids_sha256']==original['root_reconstruction']['root_image_ids_sha256']
        assert original['same_checkpoint_all_views'] and not original['test_evaluation_performed'] and original['native_comparison']['accepted'] and original['native_comparison']['tolerance']==1e-12
    return dict(prior_C40_proof_sha256=sha(prior),prior80_record_provenance_reused=True,Sp_DFA7_plus3_chains=chains,
        new_C_scientific_receipts_connected=10,new_Full_table_references_verified=10,Full900_records_sha256=full_sha,
        old900_models_rechecked=False,old_checkpoint_tensors_rechecked=False,archive_member_content_verification_reused_from_root_adoptions=True,
        prediction_arrays_recomputed=False,new_inference=0,threshold_refits=0,models_repacked=0)
