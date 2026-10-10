"""Read the non-IID F Flip10 scientific receipts and ten new Full900 identities, no tensor oracle."""
import hashlib,json,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent; R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def verify_connections(records,bindings,adopted,adoption):
    prior=R/'tmp/celeba_mechanism_C60_root_arithmetic_review_20261010/ROOT_ARITHMETIC_REVIEW.json'
    assert sha(prior)=='7273c2f3340a1c62c3bb9874117b5bff0bbe447751e0621d2c737de8ff9189ec'
    assert read(prior)['status']=='INDEPENDENT_C60_SIX_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
    inv170=R/'tmp/celeba_mechanism_valid_C_after60_20261010/inventory_actual170_Full100refs.json'
    inv160=R/'tmp/celeba_mechanism_valid_C_after56_20261010/inventory_actual160_Full100refs.json'
    assert sha(inv170)=='c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef'
    assert sha(inv160)=='302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4'
    current=read(inv170); previous=read(inv160)
    assert current['full_references']==previous['full_references']
    native={r['id']:r for r in current['records']}; before={r['id']:r for r in previous['records']}
    assert len(native)==170 and len(before)==160 and all(native[k]==v for k,v in before.items())
    by={r['id']:r for r in records}; chains=[]
    prior_adoption=R/'tmp/celeba_mechanism_valid_C_after56_20261010/execution_candidate/backups/incremental_20261010T005530Z/ROOT_ADOPTION_REVIEW.json'
    assert sha(prior_adoption)=='21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e'
    assert adopted['prior160_root_adoption_sha256']==sha(prior_adoption)
    for chain,path,invpath,ids,counts in [
        (bindings['new10_archive_chain'],adoption,inv170,[f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)],(160,10,170))]:
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
                assert provenance['root_adoption_sha256']==sha(path) and provenance['offserver_sha256']==sha(offpath)
                assert Path(provenance['archive']).resolve()==archive.resolve() and provenance['archive_sha256']==sha(archive)
                raw=tar.extractfile(provenance['receipt_member']).read(); scientific=json.loads(raw)
                assert hashlib.sha256(raw).hexdigest()==provenance['receipt_sha256']==inventory['members'][provenance['receipt_member']]['sha256']
                assert scientific['id']==key
                assert (record['distribution'],record['attack'],record['seed'])==('non-IID','F Flip',original['seed'])
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
                full=next(r for r in records if (r['variant'],r['distribution'],r['attack'],r['seed'])==('Full','non-IID','F Flip',original['seed']))
                assert full['data_contract']==record['data_contract'] and original['paired_full']['id']==full['id']
                for m in ('accuracy','aeod','aspd'): assert abs(record['views']['native'][m]-original['prior_validation_metrics'][m])<=1e-12
        chains.append(dict(root_adoption_sha256=sha(path),archive_sha256=sha(archive),archive_members=chain['members_verified'],scientific_receipts_connected=len(ids),accepted_ids=ids))
    full=[r for r in records if r['variant']=='Full' and r['distribution']=='non-IID' and r['attack']=='F Flip']; assert len(full)==10
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
    return dict(prior_C60_proof_sha256=sha(prior),prior120_record_provenance_reused=True,nonIID_F_Flip10_chain=chains,
        new_C_scientific_receipts_connected=10,new_Full_table_references_verified=10,Full900_records_sha256=full_sha,
        old900_models_rechecked=False,old_checkpoint_tensors_rechecked=False,archive_member_content_verification_reused_from_root_adoption=True,
        prediction_arrays_recomputed=False,new_inference=0,threshold_refits=0,models_repacked=0)
