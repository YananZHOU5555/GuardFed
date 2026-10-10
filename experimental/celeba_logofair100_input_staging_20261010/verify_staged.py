"""Independent byte/identity and ID-population support checks; no scores, fitting or CNN."""
from pathlib import Path
import hashlib, json, sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'tmp/celeba_logofair_fullcoverage_20261010'
sys.path.insert(0,str(SOURCE))
import population
import numpy as np
digest=population.digest;read=population.read;require=population.require


def main():
    require(not (HERE/'STAGING_FAILURE.json').exists() and not (HERE/'VERIFICATION.json').exists(),'Preserve failed or completed check; no retry')
    population.verify_sources();result=read(HERE/'STAGING_RESULT.json')
    receipt_path=Path(result['F_receipt']);require(digest(receipt_path)==result['F_receipt_sha256'],'Actual input receipt changed')
    receipt=read(receipt_path);require(receipt['root_approved'] is False and not receipt['recipe_selected'],'Staging cannot approve mapping/recipe')
    rows=read(SOURCE/'CACHE_IDENTITIES100.json')['references'];records={r['id']:r for r in receipt['references']}
    require(len(records)==100 and set(records)=={r['id'] for r in rows},'Original100 set differs')
    for e in receipt['files']:
        require(Path(e['path']).resolve().drive.upper()=='F:' and digest(e['path'])==e['sha256'] and Path(e['path']).stat().st_size==e['bytes'],'Restored/reused file differs')
    f=population.original_functions();supports=[];common_valid=None;root_hashes=set();cache_n=0
    for seed in range(91001,91011):
        group=[r for r in rows if r['seed']==seed];require(len(group)==10,'Seed does not cover all10 conditions')
        pin=receipt['mappings'][str(seed)];meta=read(pin['metadata'])
        require(digest(pin['path'])==pin['sha256'] and digest(pin['metadata'])==pin['metadata_sha256'],'Mapping bytes changed')
        original=read(population.SCREEN/'mapping_metadata.json')
        require(all(meta[k]==original[k] for k in ('semantics','cohorts','domain_hex','rule','input_fields_for_mapping','not_mapping_inputs','author_decision_sha256')),'Population rule/source changed')
        if seed!=91001:
            require(meta['status']=='PREPARED_NOT_APPROVED' and meta['approved'] is False and meta['execution_authorized'] is False,'New mapping must remain unapproved')
            array=pin['accepted_ID_arrays'];require(digest(array['path'])==array['sha256']==meta['accepted_id_array_sha256'],'Actual accepted ID member changed')
            with np.load(array['path'],allow_pickle=False) as z:source_root=z['root_image_ids'].copy();source_valid=z['valid_image_ids'].copy()
        else:require(pin['sha256']==original['mapping_sha256'] and pin['metadata_sha256']==digest(population.SCREEN/'mapping_metadata.json'),'Original91001 mapping must remain exact')
        with np.load(pin['path'],allow_pickle=False) as z:mapping={k:z[k].copy() for k in z.files}
        f['check_mapping'](mapping,group[0]['root_image_ids_sha256'],group[0]['valid_image_ids_sha256'])
        require(len(mapping['root_image_id'])==16277 and len(mapping['valid_image_id'])==19867,'Wrong mapping sample counts')
        if seed!=91001:require(np.array_equal(mapping['root_image_id'],source_root) and np.array_equal(mapping['valid_image_id'],source_valid),'Mapping differs from accepted image order')
        if common_valid is None:common_valid={k:mapping[k].copy() for k in ('valid_image_id','valid_client_id')}
        else:require(all(np.array_equal(mapping[k],common_valid[k]) for k in common_valid),'Valid image/cohort population changed across seeds')
        root_hashes.add(f['arrsha'](mapping['root_image_id']));baseline_root=None;baseline_valid_sensitive=None
        for row in group:
            require((row['root_image_ids_sha256'],row['valid_image_ids_sha256'])==(meta['root_image_ids_sha256'],meta['valid_image_ids_sha256']),'Condition mapping identity differs')
            folder=Path(records[row['id']]['path']);native=read(folder/'result.json');job=read(folder/'source_job.json')
            require(native['rounds']==70 and [r['round'] for r in native['round_summaries']]==list(range(1,71)) and native['seed']==seed
                and (native['method'],native['distribution'],native['attack'])==('FedAvg',row['distribution'],row['attack'])
                and all(native['revision_job'][k]==job[k] for k in job),'Native recipe/seed/source/terminal differs')
            contract=native['data_contract']['image_data_contract']
            require(contract['evaluation_split']=='valid' and contract['actual_train_rows']==162770 and contract['actual_evaluation_rows']==19867
                and contract['train_eval_disjoint'] and contract['root_client_disjoint'],'Original train/valid contract differs')
            # Membership is already fixed solely by accepted IDs. Only support fields are decoded here.
            with np.load(folder/'margins.npz',allow_pickle=False) as z:ry=z['root_y'].copy();rs=z['root_sensitive'].copy();vs=z['valid_sensitive'].copy()
            require(ry.shape==rs.shape==(16277,) and vs.shape==(19867,) and set(ry)==set(rs)==set(vs)=={0,1},'Root/valid support fields differ')
            if baseline_root is None:baseline_root=(ry,rs);baseline_valid_sensitive=vs
            else:require(np.array_equal(ry,baseline_root[0]) and np.array_equal(rs,baseline_root[1]) and np.array_equal(vs,baseline_valid_sensitive),'Support fields differ across same-seed conditions')
            cache_n+=1
        ry,rs=baseline_root;counts=[]
        for cid in range(20):
            root_mask=mapping['root_client_id']==cid;valid_mask=mapping['valid_client_id']==cid
            cells={f'root_y{y}_Male{s}':int(np.sum(root_mask&(ry==y)&(rs==s))) for y in (0,1) for s in (0,1)}
            counts.append(dict(cohort=cid,root_n=int(root_mask.sum()),valid_n=int(valid_mask.sum()),**cells,
                valid_Male0=int(np.sum(valid_mask&(baseline_valid_sensitive==0))),valid_Male1=int(np.sum(valid_mask&(baseline_valid_sensitive==1)))))
        require(sum(r['root_n'] for r in counts)==16277 and sum(r['valid_n'] for r in counts)==19867,'Population totals differ')
        complete=all(all(r[f'root_y{y}_Male{s}']>0 for y in (0,1) for s in (0,1)) and r['valid_Male0']>0 and r['valid_Male1']>0 for r in counts)
        require(complete,'Missing root two-label/group or valid sensitive support; preserve failure, no remapping')
        supports.append(dict(seed=seed,status=meta['status'],mapping_sha256=pin['sha256'],mapping_metadata_sha256=pin['metadata_sha256'],root_image_ids_sha256=meta['root_image_ids_sha256'],valid_image_ids_sha256=meta['valid_image_ids_sha256'],
            root_n=16277,valid_n=19867,cohorts=20,all_root_group_label_and_valid_sensitive_support_complete=complete,counts=counts))
    require(cache_n==100 and len(root_hashes)==10,'All100 caches and10 original root populations required')
    population.verify_sources()
    proof=dict(status='STAGED100_IDENTITY_AND_FIXED10_POPULATION_SUPPORT_PASS_NEW9_NOT_APPROVED',
        F_receipt=str(receipt_path),F_receipt_sha256=digest(receipt_path),source_seal_sha256=digest(SOURCE/'FILES_SHA256.json'),
        original_inventory_sha256=digest(SOURCE/'CACHE_IDENTITIES100.json'),references_verified=100,reference_files_verified=400,
        ID_members_verified=9,new_mappings_PREPARED_NOT_APPROVED=9,original91001_mapping_unchanged=True,root_populations=10,valid_population=1,
        population_support=supports,decoded_cache_fields=['root_y','root_sensitive','valid_sensitive'],valid_labels_or_scores_decoded=False,
        mapping_rule_input_fields=['image_id'],mapping_rule_original_pure_functions_reused=True,root_approved=False,recipe_selected=False,
        new_CNN_calls=0,new_fits=0,new_score_caches=0,torch_imported='torch' in sys.modules,
        limit='Support completeness does not establish Beta MLE convergence, absence of exact threshold ties or useful performance. Root must approve actual inputs/mappings; 32 recipe selection is separate.')
    with (HERE/'VERIFICATION.json').open('x',encoding='utf8') as out:out.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=proof['status'],references=100,cache_support_checks=cache_n,new_mappings=9,root_approved=False)))


if __name__=='__main__':main()
