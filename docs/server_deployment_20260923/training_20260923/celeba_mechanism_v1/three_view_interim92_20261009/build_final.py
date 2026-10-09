"""Offline adopted-receipt join only; required external root receipt SHA, no CNN."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
R=H.parents[2]
OLD=R/'tmp/celeba_mechanism_three_view_paired71_20261009'
P82=R/'tmp/celeba_mechanism_valid_incremental_after71_20261009'
P92=R/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009'
FULL=R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
B82=P82/'execution_candidate/backups/incremental_20261009T163050Z'
VIEWS=('native','raw','shared_calibration')

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rawsha(b):return hashlib.sha256(b).hexdigest()
def read(p,expected=None):
    if expected:need(sha(p)==expected,'SHA mismatch '+str(p))
    return json.loads(Path(p).read_bytes())
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def funcs(p,names,ns):
    source=Path(p).read_text(encoding='utf-8');nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes}==set(names),'Missing exact source functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return {n.name:rawsha(ast.get_source_segment(source,n).encode()) for n in nodes}
def write(p,j):
    with p.open('x',encoding='utf-8') as f:f.write(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def accepted_increment(folder,adoption_sha,inventory,inventory_path,bridge_path,prior,total,identity,normalized,canonical):
    """Existing strict/offserver evidence only; no rerun of its evaluator."""
    a=read(folder/'ROOT_ADOPTION_REVIEW.json',adoption_sha)
    n=total-prior;ids=inventory['selected_replay_ids']
    need(a['prior_three_view_models']==prior and a['cumulative_three_view_models']==total and a['accepted_new']==n,'Wrong adopted count')
    need(a['accepted_new_ids']==ids and len(ids)==len(set(ids))==n,'Wrong exact adopted increment')
    need(a['all_native_differences_zero'] and a['source_scope_complete'] and a[f'original{prior}_unchanged'],'Incomplete native/source/prior adoption')
    need(a['science_seal_sha256']==sha(inventory_path.parent/'FILES_SHA256.json') and a['execution_seal_sha256']==sha(inventory_path.parent/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),'Adopted science/execution seal drift')
    need(a['new_training']==0 and a['new_Full_inference']==0 and a['test_inference'] is False,'Forbidden new science')
    b=read(folder/'backup_receipt.json',a['backup_receipt_sha256'])
    o=read(folder/'OFFSERVER_VERIFICATION.json',a['offserver_verification_sha256'])
    need(b['accepted_new_ids']==o['accepted_new_ids']==ids and o['accepted_n']==n,'Backup/offserver cohort drift')
    need(o['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS','Offserver incomplete')
    need(o['independent_metric_checks']==9*n and o['independent_confusion_count_checks']==24*n and o['prediction_rule_checks']==3*n,'Offserver checks incomplete')
    archive=folder/'incremental_valid_three_views.tar.gz'
    need(sha(archive)==b['archive_sha256']==o['archive_sha256']==a['archive_sha256'],'Archive identity drift')
    actual={r['id']:r for r in inventory['records']};off={r['id']:r for r in o['records']};output=[]
    need(set(off)==set(ids) and len(o['records'])==n,'Duplicate/offserver ID')
    with tarfile.open(archive) as tar:
        payload=tar.extractfile('backup_inventory.json').read();need(rawsha(payload)==b['inventory_sha256']==o['inventory_sha256'],'Member manifest drift')
        members=json.loads(payload)['members'];names=tar.getnames()
        need(len(names)==len(set(names))==b['members'] and set(names)==set(members)|{'backup_inventory.json'},'Archive membership drift')
        for name,pin in members.items():
            raw=tar.extractfile(name).read();need(len(raw)==pin['bytes'] and rawsha(raw)==pin['sha256'],'Member drift '+name)
        for rid in ids:
            def member(suffix):
                name='runs/'+rid+'/'+suffix;raw=tar.extractfile(name).read();return json.loads(raw),members[name]['sha256']
            receipt,receipt_sha=member('receipt.json');bridge,bridge_sha=member('bridge_receipt.json');strict,strict_sha=member('strict_acceptance.json')
            record=actual[rid];identity(receipt,record,SimpleNamespace(canonical=canonical))
            need(strict['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and bridge['status']=='MECHANISM_NATIVE_VALID_REPLAY_PASS','Scientific/strict incomplete')
            need(strict['id']==bridge['id']==receipt['id']==rid and strict['variant']==bridge['variant']=='minus_U','Mixed ID/variant')
            need(strict['scope']==bridge['scope']==inventory['scope'],'Old failed scope cannot enter new cohort')
            need(strict['inventory_sha256']==bridge['inventory_sha256']==sha(inventory_path),'Inventory bytes differ')
            need(bridge['inventory_record_sha256']==canonical(record) and bridge['bridge_source_sha256']==sha(bridge_path),'Record/source drift')
            need(strict['bridge_receipt_sha256']==bridge_sha and bridge['scientific_body_receipt_sha256']==receipt_sha,'Receipt chain drift')
            need(strict['checkpoint_sha256']==off[rid]['checkpoint_sha256']==receipt['checkpoint_sha256'],'Mixed model')
            need(strict['views']==off[rid]['views']==receipt['views'],'Views disagree with strict/offserver')
            need(strict['native_comparison']==receipt['native_comparison'],'Strict/native comparison drift')
            need(members['runs/'+rid+'/validation_predictions.npz']['sha256']==receipt['prediction_arrays_sha256'],'Array member does not match scientific receipt')
            need(off[rid]['native_max_abs_difference']<=1e-12 and off[rid]['prediction_arrays_sha256']==receipt['prediction_arrays_sha256'],'Saved array/native drift')
            need(bridge['source_before']==bridge['source_after'] and bridge['artifact_before']==bridge['artifact_after'],'Mutated science inputs')
            need(bridge['paired_full_reference']==strict['paired_full_reference']==record['paired_full'],'Paired Full differs')
            need(receipt['runtime']['device']=='cpu' and receipt['runtime']['cuda_device_count']==0 and not receipt['test_labels_accessed'],'Wrong replay runtime/split')
            provenance={'archive':str(archive),'archive_sha256':b['archive_sha256'],'receipt_member':'runs/'+rid+'/receipt.json','receipt_sha256':receipt_sha,'bridge_receipt_sha256':bridge_sha,'strict_acceptance_sha256':strict_sha,'root_adoption_sha256':adoption_sha,'offserver_sha256':sha(folder/'OFFSERVER_VERIFICATION.json'),'inventory_sha256':sha(inventory_path),'prediction_arrays_sha256':receipt['prediction_arrays_sha256']}
            output.append(normalized(receipt,record,'minus_U',provenance))
    return output,{'archive':str(archive),'archive_sha256':b['archive_sha256'],'root_adoption_sha256':adoption_sha,'accepted_ids':ids,'members_verified':len(names)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root-review',type=Path,required=True)
    ap.add_argument('--root-review-sha',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    need(args.root_review.name=='ROOT_ADOPTION_REVIEW.json' and len(args.root_review_sha)==64,'Explicit actual root adoption required')
    need(args.root_review.resolve().is_relative_to((P92/'execution_candidate/backups').resolve()),'Only v2 backup allowed')
    need(args.output.resolve().parent==H.resolve() and not args.output.exists(),'Fresh owned snapshot only')
    pins=read(H/'INPUTS.json')['files']
    for name,pin in pins.items():need(sha(R/name)==pin['sha256'] and (R/name).stat().st_size==pin['bytes'],'Pinned source changed '+name)
    adoption=read(args.root_review,args.root_review_sha)
    need(adoption['cumulative_three_view_models']==92 and adoption['prior_three_view_models']==82 and adoption['accepted_new']==10,'Exact new10 adoption required')
    i82=read(P82/'inventory_actual82_Full100refs.json');i92=read(P92/'inventory_actual92_Full100refs.json')
    need(sha(P92/'inventory_actual92_Full100refs.json')=='bedb867ae9bdf965ef7143a5b4f077620911ec26d93346337307046e64b5c9da','Wrong v2 inventory')
    actual={r['id']:r for r in i92['records']};prior={r['id']:r for r in i82['records']}
    need(len(actual)==92 and len(prior)==82 and all(actual[k]==r for k,r in prior.items()),'Original82 changed')
    need(set(actual)-set(prior)==set(i92['selected_replay_ids']),'Wrong92 minus82')
    need(adoption['prior82_root_adoption_sha256']==sha(B82/'ROOT_ADOPTION_REVIEW.json'),'Root prior82 chain differs')
    ns={'need':need,'VIEWS':VIEWS,'hashlib':hashlib,'json':json}
    function_proof=funcs(OLD/'inputs.py',['receipt_identity','normalized'],ns)
    function_proof.update(funcs(P92/'bridge.py',['canonical'],ns))
    canonical=ns['canonical']
    original=module('unchanged_original_statistics',R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py')
    pure=module('nine_scene_scope_draft',H.parent/'panels_draft.py')
    independent=module('independent_fsum_table_checks',H.parent/'verify_numeric_draft.py')
    oldrecords=read(OLD/'snapshot_724_full100_mechanism71/records.json')['records']
    controls=[r for r in oldrecords if r['variant']=='minus_U'];need(len(controls)==71,'Old71 controls missing')
    for r in controls:
        inv=actual[r['id']];need(r['checkpoint_sha256']==inv['checkpoint']['sha256'] and r['config_sha256']==inv['config_canonical_sha256'] and r['data_contract']==inv['data_contract'],'Old accepted71 identity drift')
        need(all(abs(r['views']['native'][m]-inv['prior_validation_metrics'][m])<=1e-12 for m in ['accuracy','aeod','aspd']),'Old native drift')
    add11,chain11=accepted_increment(B82,sha(B82/'ROOT_ADOPTION_REVIEW.json'),i82,P82/'inventory_actual82_Full100refs.json',P82/'bridge.py',71,82,ns['receipt_identity'],ns['normalized'],canonical)
    add10,chain10=accepted_increment(args.root_review.parent,args.root_review_sha,i92,P92/'inventory_actual92_Full100refs.json',P92/'bridge.py',82,92,ns['receipt_identity'],ns['normalized'],canonical)
    controls+=add11+add10;need(len(controls)==len({r['id'] for r in controls})==92 and {r['id'] for r in controls}==set(actual),'Duplicate/missing controls')
    full900=read(FULL/'records_three_views_900.json');byid={r['id']:r for r in full900['records']};full=[]
    refs={r['id']:r for r in i92['full_references']};matched={r['paired_full']['id'] for r in actual.values()}
    for rid in sorted(matched):
        r=byid[rid];ref=refs[rid];inv=r['original_inventory_record']
        need(r['checkpoint_sha256']==ref['checkpoint_sha256'] and r['model_inventory_record_sha256']==ref['baseline_record_canonical_sha256'],'Full900 identity drift')
        need(r['same_checkpoint_all_views'] and not r['test_evaluation_performed'],'Incomplete Full views')
        full.append(dict(id=rid,variant='Full',method=r['method'],distribution=r['distribution'],attack=r['attack'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],config_sha256=r['config_canonical_sha256'],data_contract=inv['data_contract'],training_torch=r['training_torch'],replay_runtime=r['runtime'],views=r['views'],fits=r['fits'],provenance={'accepted900_path':str(FULL/'records_three_views_900.json'),'accepted900_sha256':sha(FULL/'records_three_views_900.json'),'receipt_sha256':r['receipt_sha256'],'prediction_arrays_sha256':r['prediction_arrays_sha256'],'source_binding':r['source_binding']}))
    records=full+controls;need(len(records)==184,'Exact92 pairs required')
    fullcells={(r['distribution'],r['attack'],r['seed']):r for r in full}
    for r in controls:need(r['data_contract']==fullcells[r['distribution'],r['attack'],r['seed']]['data_contract'],'Paired data contract differs')
    panels,coverage,paired=pure.panels(records,original);verification=independent.verify(records,panels)
    oldtables=read(OLD/'snapshot_724_full100_mechanism71/tables.json')
    indexed={(p['view'],p['label'],r['distribution'],r['attack'],r['variant']):r for p in panels for r in p['rows']}
    for p in oldtables['panels']:
        for r in p['rows']:need(indexed[p['view'],p['label'],r['distribution'],r['attack'],r['variant']]==r,'Old seven scene statistic changed')
    native=read(R/'tmp/celeba_mechanism_native92_tables_20261009/tables.json')
    for p in native['panels']:
        for r in p['rows']:
            new=indexed['native',p['label'],r['distribution'],r['attack'],r['variant']]
            for m in original.METRICS:
                for k in ['mean','sample_sd_ddof1']:need(abs(new[m][k]-r[m][k])<=1e-12,'Native92 mismatch')
    complete=[r for r in records if not(r['distribution']=='non-IID' and r['attack']=='Sp-DFA')]
    result={'status':'NINE_COMPLETE_SCENES_THREE_VIEWS_PENDING_ROOT_TABLE_REVIEW','complete_scenes':9,'paired_checkpoints':92,'preserved_records':184,'table_model_records':180,'incomplete_pairs':2,'panels':panels,'native_shared_identical_records':sum(r['views']['native']==r['views']['shared_calibration'] for r in records),'replay_devices':{v:dict(Counter(r['replay_runtime']['device'] for r in complete if r['variant']==v)) for v in ['Full','minus_U']},'training_torch':{v:dict(Counter(r['training_torch'] for r in complete if r['variant']==v)) for v in ['Full','minus_U']},'primary_endpoint_selected':False,'final_test':False,'new_inference':0,'new_training':0,'Full900_records_sha256':sha(FULL/'records_three_views_900.json')}
    texts=['# CelebA Full-minus_U: nine complete native/raw/shared validation scenes','','Three views of the same terminal checkpoints; no primary endpoint selected.','']
    for p in panels:
        texts+=['## '+p['view']+' - '+p['label'],'','| Distribution | Scenario | Variant / paired difference | n | ACC (%) | AEOD | ASPD |','|---|---|---|---:|---:|---:|---:|']
        for r in p['rows']:
            vals=[f"{r[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {r[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in original.METRICS]
            texts.append('| '+' | '.join([r['distribution'],r['attack'],r['variant'],str(r['n']),*vals])+' |')
        texts.append('')
    texts+=['Mean ± sample SD, ddof=1; paired differences are minus_U minus Full (ACC: percentage points). AEOD is absolute TPR gap. Native includes each procedure’s own calibration; raw is uncalibrated; shared uses original root-only group thresholds.','Mixed CPU/GPU and cu128/cu130/driver provenance retained; no environment-equivalence or aggregation-only causal claim. Seed91001 selection exposure and prior validation exposure remain. No significance/CI or best-seed selection. All negative outcomes retained.','non-IID Sp-DFA has only two paired seeds91001/91002; all four records are retained separately and excluded from every mean table. The other seven mechanism variants remain outside this snapshot. No final-test or overall completion claim.','']
    display_lines=[line for line in texts if line.startswith('| ') and ' ± ' in line]
    flat=[row for panel in panels for row in panel['rows']]
    need(len(display_lines)==len(flat)==243,'Missing displayed rows')
    display_cells=0
    for line,row in zip(display_lines,flat):
        cells=line.strip('| ').split(' | ')
        for index,metric in enumerate(original.METRICS,4):
            precision=3 if metric=='accuracy_pct' else 5
            need(cells[index]==f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}",'Display value mismatch')
            display_cells+=1
    verification.update(display_cells_verified=display_cells,old_seven_scene_rows_exact=189,native92_rows_matched=81,new_increment_archive_chains=[chain11,chain10],source_functions_exact=function_proof,negative_results_preserved=True)
    args.output.mkdir()
    write(args.output/'tables.json',result);write(args.output/'records.json',{'records':records});write(args.output/'paired_per_seed.json',paired);write(args.output/'coverage.json',coverage)
    write(args.output/'incomplete_pairs.json',{'records':[r for r in records if r not in complete],'included_in_mean_tables':False})
    write(args.output/'verification.json',verification);write(args.output/'SOURCE_BINDINGS.json',{'prepared_input_pins':pins,'new_root_adoption':str(args.root_review),'new_root_adoption_sha256':args.root_review_sha,'builder_sha256':sha(__file__)})
    (args.output/'TABLES.md').write_text('\n'.join(texts),encoding='utf-8')
    print(json.dumps({'status':result['status'],'scenes':9,'model_records':184,'scalar_checks':verification['mean_sd_scalars']}))

if __name__=='__main__':main()
