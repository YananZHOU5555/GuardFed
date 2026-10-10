"""Consume the adopted200 small-JSON index; no model/array reads or inference."""
import argparse
import json
from pathlib import Path
from common import H,R,OLD,FULL,EXPECTED,need,sha,read,verify_inputs,scientific,require_exact20
from assemble import assemble


def pinned(path,digest):
    path=Path(path)
    need(len(digest)==64 and sha(path)==digest,'Actual accepted source SHA differs: '+str(path))
    return read(path)


def load_records(root_path,root_sha,index_path,index_sha):
    verify_inputs();science,functionproof=scientific()
    adoption=pinned(root_path,root_sha);index=pinned(index_path,index_sha)
    need(root_sha=='2ac1d2f200d9671de5271f24b0cbb3a0772afb88c16ae6ccf71805ed1588ee46'
         and index_sha=='0940513f702c42ce9451d42ba6d66cbc9ab70868c90ca2137612b8555ca5dc96','Only actual adopted200 permitted')
    need(adoption['status']=='ROOT_C19_SAVED_ARRAYS_AND_NATIVE200_RESTORE_CHAIN_ADOPTED'
         and index['status']=='ACCEPTED200_REPLAY_ID_INDEX','Actual root adoption required')
    need(adoption['records_index_sha256']==index_sha and (R/adoption['records_index_path']).resolve()==index_path.resolve(),'Root/index link differs')
    need((adoption['prior_accepted'],adoption['new_accepted'],adoption['cumulative_accepted'])==(181,19,200)
         and adoption['accepted_new_ids']==index['new_ids']==EXPECTED[1:],'Exact181+19 required')
    need(adoption['original180_unchanged'] and adoption['first181_unchanged'] and adoption['native_max_abs_difference']==0,'Accepted prior/native integrity failed')
    need(adoption['Full_inference']==adoption['new_CNN']==adoption['new_training']==0 and adoption['test'] is False and index['test'] is False,'Forbidden scientific operation')
    pins=read(H/'INPUT_PINS.json')
    first=pinned(R/index['first_adoption'],index['first_adoption_sha256'])
    need(index['first_adoption_sha256']==pins['first1_adoption_sha256'] and first['new_accepted']==1 and first['cumulative_accepted']==181,'First1 adopted chain differs')
    first_index=pinned(R/index['first181_index'],index['first181_index_sha256'])
    prior=read(R/'tmp/celeba_mechanism_valid_C_after70_20261010/inventory_actual180_Full100refs.json')
    need(set(first_index['prior180_ids'])=={r['id'] for r in prior['records']}
         and index['all_ids']==first_index['all_ids']+EXPECTED[1:],'Prior180/first181 ID order changed')
    need(len(index['all_ids'])==len(set(index['all_ids']))==200,'Duplicate adopted model')
    need(index['prior180_root_adoption_sha256']==first_index['prior180_root_adoption_sha256']=='3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615','Prior180 root differs')
    for field in ('new_bindings','new_binding_files','new_artifacts'):require_exact20(list(index[field]))
    saved=[index['first_record']]+index['new_records'];require_exact20([r['id'] for r in saved])
    need(index['first_record']==first_index['new_records'][0],'First1 saved metrics changed')
    native=pinned(R/index['new_native_inspection'],index['new_native_inspection_sha256'])
    need(index['new_native_inspection_sha256']==adoption['native200_inspection_sha256'] and native['new_count']==200 and len(native['records'])==300,'Native200 inspection required')
    native_by={r['id']:r for r in native['records']}
    archive=index['new_archive']
    need(archive['archive_sha256']==adoption['archive_sha256'] and archive['accepted_new_ids']==EXPECTED[1:],'Exact19 archive chain differs')
    receipt=pinned(archive['receipt'],archive['receipt_sha256'])
    offserver=pinned(archive['offserver_verification'],archive['offserver_verification_sha256'])
    need(archive['receipt_sha256']==adoption['receipt_sha256'] and archive['offserver_verification_sha256']==adoption['offserver_proof_sha256'],'Original receipt/offserver differs')
    need(archive['previous_receipt_sha256']==first['receipt_sha256'],'First1→19 receipt chain differs')
    # The root adoption already rehashed whole archives and native model/result members.
    # This reader rechecks their small accepted JSON products rather than extracting arrays.
    refs={r['id']:r for r in prior['full_references']}
    full900={r['id']:r for r in read(FULL/'records_three_views_900.json')['records']}
    added=[];full=[];identity_checks=[]
    for item in saved:
        identity=item['id'];binding=index['new_bindings'][identity];bpin=index['new_binding_files'][identity]
        need(pinned(bpin['path'],bpin['sha256'])==binding,'Binding JSON differs from adopted index')
        record=binding['record'];apins=index['new_artifacts'][identity]
        artifacts={k:pinned(v['path'],v['sha256']) for k,v in apins.items()}
        scientific_receipt=artifacts['scientific_receipt'];strict=artifacts['strict_json'];bridge=artifacts['bridge_receipt']
        need(binding['status']=='IMMUTABLE_TERMINAL_ONCE_BOUND' and binding['id']==record['id']==identity,'Once-bound identity changed')
        need(record['accepted_v4_row']==native_by[identity],'Native200 source record differs')
        need(record['variant']=='minus_C' and record['terminal_round']==70 and record['original_split']=='valid'
             and record['original_n_eval']==19867 and record['actual_alpha']==5.0 and record['config']['ablation_component']=='C','Scientific scope differs')
        need(record['data_contract']['actual_train_rows']==162770 and record['data_contract']['actual_evaluation_rows']==19867,'Full train/valid split differs')
        science.receipt_identity(scientific_receipt,record,science)
        need(strict['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and bridge['status']=='MECHANISM_NATIVE_VALID_REPLAY_PASS','Original strict/bridge not accepted')
        need(strict['id']==bridge['id']==identity and strict['checkpoint_sha256']==item['checkpoint_sha256']==record['checkpoint']['sha256'],'Mixed terminal checkpoint')
        need(strict['views']==item['views']==scientific_receipt['views'],'Saved-array/scientific/strict metrics differ')
        need(strict['bridge_receipt_sha256']==apins['bridge_receipt']['sha256']
             and strict['inventory_sha256']==bridge['inventory_sha256']==apins['bound_inventory']['sha256']
             and bridge['approval_sha256']==apins['delegated_approval']['sha256'],'Runtime strict source links differ')
        ref=record['paired_full'];need(ref==refs[ref['id']],'Original Full100 reference differs')
        full.append(science.full_record(full900[ref['id']],ref))
        provenance=dict(root200_path=str(root_path),root200_sha256=root_sha,index200_sha256=index_sha,binding=bpin,artifacts=apins,native200_inspection_sha256=index['new_native_inspection_sha256'],saved_array_record=item)
        added.append(science.normalized(scientific_receipt,record,'minus_C',provenance))
        identity_checks.append(dict(id=identity,checkpoint_sha256=record['checkpoint']['sha256'],scientific_receipt_sha256=apins['scientific_receipt']['sha256'],native_max_abs_difference=scientific_receipt['native_comparison']['max_abs_difference']))
    old=read(OLD/'snapshot/records.json')['records']
    bindings=dict(prepared_input_pins=pins['files'],actual_root200_path=str(root_path),actual_root200_sha256=root_sha,index200_path=str(index_path),index200_sha256=index_sha,source_functions=functionproof,new20_identity_checks=identity_checks,new20_artifacts=index['new_artifacts'],new20_binding_files=index['new_binding_files'],new_archive_chain=archive,first1_adoption_sha256=index['first_adoption_sha256'],original_C80_root_sha256=sha(OLD/'ROOT_VERIFICATION.json'),no_model_or_array_reads=True,new_inference=0)
    return old+full+added,bindings


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root-adoption',type=Path,required=True);ap.add_argument('--root-adoption-sha256',required=True)
    ap.add_argument('--index',type=Path,required=True);ap.add_argument('--index-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    need(not a.output.exists(),'Existing evidence must not be overwritten')
    records,bindings=load_records(a.root_adoption,a.root_adoption_sha256,a.index,a.index_sha256)
    print(json.dumps(assemble(records,bindings,a.output)))

if __name__=='__main__':main()
