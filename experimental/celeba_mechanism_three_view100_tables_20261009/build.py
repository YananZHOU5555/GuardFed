"""Join externally adopted after92 saved results; no CNN, labels or fitting."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
R=H.parents[1]
PRIOR=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim92_20261009'
P92=R/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009'
P100=R/'tmp/celeba_mechanism_valid_incremental_after92_20261009'
FULL=R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
OLD=R/'tmp/celeba_mechanism_three_view92_tables_20261009/final_builder_v2'
EXPECTED=[f'minus_U_non-IID_Sp-DFA_seed{s}' for s in range(91003,91011)]
VIEWS=('native','raw','shared_calibration')

def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(p,j):
    with p.open('x',encoding='utf-8') as f:f.write(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def scope(i92,i100):
    prior={r['id']:r for r in i92['records']};actual={r['id']:r for r in i100['records']}
    need(len(prior)==len(i92['records'])==92 and len(actual)==len(i100['records'])==100,'Duplicate/missing scientific records')
    need(i100['selected_replay_ids']==EXPECTED and set(actual)-set(prior)==set(EXPECTED),'Only exact after92 eight allowed')
    need(set(i100['excluded_prior_replay_ids'])==set(prior),'Closed92 exclusion drift')
    need(all(actual[k]==r for k,r in prior.items()),'Original92 record changed')
    need(i100['full_references']==i92['full_references'] and len(i100['full_references'])==100,'Full100 references changed')
    need(i100['native_tolerance']==i92['native_tolerance']==1e-12,'Native tolerance changed')
    need(all(r['variant']=='minus_U' and r['original_split']=='valid' and r['terminal_round']==70 and r['original_n_eval']==19867 for r in actual.values()),'Variant/split/terminal drift')
    cells={(r['distribution'],r['attack'],r['seed']) for r in actual.values()}
    expected={(d,a,s) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA') for s in range(91001,91011)}
    need(cells==expected,'Incomplete ten-scene/ten-seed scope')
    return actual

def adoption_gate(proof,path,expected_sha):
    need(path.name=='ROOT_ADOPTION_REVIEW.json' and path.resolve().is_relative_to((P100/'execution_candidate/backups').resolve()),'Only actual after92 root adoption allowed')
    need(len(expected_sha)==64 and sha(path)==expected_sha,'External root adoption SHA mismatch')
    need(proof['prior_three_view_models']==92 and proof['accepted_new']==8 and proof['cumulative_three_view_models']==100,'Actual adopted92+8 required')
    need(proof['accepted_new_ids']==EXPECTED and proof['original92_unchanged'] and proof['source_scope_complete'],'Exact adopted eight/source/prior required')
    previous=P92/'execution_candidate/backups/incremental_20261009T174922Z/ROOT_ADOPTION_REVIEW.json'
    need(proof['prior92_root_adoption_sha256']==sha(previous),'Prior92 adoption chain drift')
    need(proof['new_training']==proof['new_Full_inference']==0 and proof['test_inference'] is False,'Forbidden science operation')

def full_record(r,ref):
    need(r['checkpoint_sha256']==ref['checkpoint_sha256'] and r['model_inventory_record_sha256']==ref['baseline_record_canonical_sha256'],'Full900 identity drift')
    need(r['same_checkpoint_all_views'] and not r['test_evaluation_performed'],'Full three views incomplete')
    inv=r['original_inventory_record']
    return dict(id=r['id'],variant='Full',method=r['method'],distribution=r['distribution'],attack=r['attack'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],config_sha256=r['config_canonical_sha256'],data_contract=inv['data_contract'],training_torch=r['training_torch'],replay_runtime=r['runtime'],views=r['views'],fits=r['fits'],provenance={'accepted900_path':str(FULL/'records_three_views_900.json'),'accepted900_sha256':sha(FULL/'records_three_views_900.json'),'receipt_sha256':r['receipt_sha256'],'prediction_arrays_sha256':r['prediction_arrays_sha256'],'source_binding':r['source_binding']})

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root-review',type=Path,required=True);ap.add_argument('--root-review-sha',required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    need(args.output.resolve().parent==H.resolve() and not args.output.exists(),'Fresh owned snapshot only')
    pins=read(H/'INPUTS.json')['files']
    for name,pin in pins.items():need(sha(R/name)==pin['sha256'] and (R/name).stat().st_size==pin['bytes'],'Pinned input changed '+name)
    i92=read(P92/'inventory_actual92_Full100refs.json');i100=read(P100/'inventory_actual100_Full100refs.json');actual=scope(i92,i100)
    adoption=read(args.root_review);adoption_gate(adoption,args.root_review,args.root_review_sha)
    reused=module('accepted92_join_unchanged',OLD/'build_final.py')
    ns={'need':need,'VIEWS':VIEWS,'hashlib':hashlib,'json':json}
    functions=reused.funcs(R/'tmp/celeba_mechanism_three_view_paired71_20261009/inputs.py',['receipt_identity','normalized'],ns)
    functions.update(reused.funcs(P100/'bridge.py',['canonical'],ns));canonical=ns['canonical']
    oldrecords=read(PRIOR/'snapshot92/records.json')['records'];need(len(oldrecords)==len({r['id'] for r in oldrecords})==184,'Prior accepted184 missing')
    added,chain=reused.accepted_increment(args.root_review.parent,args.root_review_sha,i100,P100/'inventory_actual100_Full100refs.json',P100/'bridge.py',92,100,ns['receipt_identity'],ns['normalized'],canonical)
    baseline=read(FULL/'records_three_views_900.json');byid={r['id']:r for r in baseline['records']}
    refs={r['id']:r for r in i100['full_references']};oldids={r['id'] for r in oldrecords};controls=[r for r in oldrecords if r['variant']=='minus_U']+added
    matched={r['paired_full']['id'] for r in actual.values()};need(matched==set(refs) and len(matched)==100,'Full100 pairing incomplete')
    extra=[full_record(byid[rid],refs[rid]) for rid in sorted(matched-oldids)]
    need(len(extra)==8,'Only eight absent Full records may be referenced')
    records=oldrecords+added+extra;need(len(records)==len({r['id'] for r in records})==200,'Exact100 pairs required')
    fullcells={(r['distribution'],r['attack'],r['seed']):r for r in records if r['variant']=='Full'}
    for r in records:
        if r['variant']=='Full':
            frozen=full_record(byid[r['id']],refs[r['id']]);need(r==frozen,'Old/new Full900 views/fit/checkpoint changed')
        else:
            inv=actual[r['id']]
            need(r['checkpoint_sha256']==inv['checkpoint']['sha256'] and r['config_sha256']==inv['config_canonical_sha256'] and r['data_contract']==inv['data_contract'],'Control scientific identity changed')
            need(all(abs(r['views']['native'][m]-inv['prior_validation_metrics'][m])<=1e-12 for m in ('accuracy','aeod','aspd')),'Original native mismatch')
            need(r['data_contract']==fullcells[r['distribution'],r['attack'],r['seed']]['data_contract'],'Paired root/valid/train IDs differ')
    original=module('original_statistics_unchanged',R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py')
    pure=module('ten_scene_scope',H/'panels.py');independent=module('independent_numeric_checks',H/'verify_numeric.py')
    panels,coverage,paired=pure.panels(records,original);verification=independent.verify(records,panels)
    indexed={(p['view'],p['label'],r['distribution'],r['attack'],r['variant']):r for p in panels for r in p['rows']}
    oldtables=read(PRIOR/'snapshot92/tables.json');old_rows=0
    for p in oldtables['panels']:
        for r in p['rows']:need(indexed[p['view'],p['label'],r['distribution'],r['attack'],r['variant']]==r,'Old nine-scene row changed');old_rows+=1
    need(old_rows==243 and records[:184]==oldrecords,'Old rows/records changed')
    native=read(R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/tables.json')
    for p in native['panels']:
        for r in p['rows']:
            new=indexed['native',p['label'],r['distribution'],r['attack'],r['variant']]
            for m in original.METRICS:
                for k in ('mean','sample_sd_ddof1'):need(abs(new[m][k]-r[m][k])<=1e-12,'Native100 table mismatch')
    texts=['# CelebA Full-minus_U: ten complete validation scenes, three views','','100 paired terminal checkpoints; valid-only; primary endpoint remains pending.','']
    for p in panels:
        texts+=['## '+p['view']+' - '+p['label'],'','| Distribution | Scenario | Variant / paired difference | n | ACC (%) | AEOD | ASPD |','|---|---|---|---:|---:|---:|---:|']
        for r in p['rows']:
            vals=[f"{r[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {r[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in original.METRICS]
            texts.append('| '+' | '.join([r['distribution'],r['attack'],r['variant'],str(r['n']),*vals])+' |')
        texts.append('')
    oldlines=[s for s in (PRIOR/'snapshot92/TABLES.md').read_text(encoding='utf-8').splitlines() if s.startswith('| ') and ' ± ' in s]
    newlines=[s for s in texts if s.startswith('| ') and ' ± ' in s]
    need(len(oldlines)==243 and len(newlines)==270 and all(s in newlines for s in oldlines),'Prior nine-scene display changed')
    display_cells=0
    for line,row in zip(newlines,[r for p in panels for r in p['rows']]):
        cells=line.strip('| ').split(' | ')
        for index,metric in enumerate(original.METRICS,4):
            precision=3 if metric=='accuracy_pct' else 5
            need(cells[index]==f"{row[metric]['mean']:.{precision}f} ± {row[metric]['sample_sd_ddof1']:.{precision}f}",'Display value mismatch')
            display_cells+=1
    need(display_cells==810,'Missing table cell')
    texts+=['Mean ± sample SD, ddof=1. Paired differences: minus_U minus Full; ACC in percentage points. AEOD is absolute TPR gap, not full equalized odds. Native includes each method\'s original root-only calibration; shared uses the frozen common root-only rule.','Same terminal checkpoint per record/view; no new Full inference or threshold refitting here. Mixed CPU/GPU replay and training CUDA/driver provenance retained, not a uniform-device final fairness comparison. Seed91001 selection and prior validation exposure remain; 9/6-seed panels are descriptive sensitivity panels. No significance, necessity, or final-test claim.','Native104 consists of minus_U100 plus minus_C4. This three-view snapshot contains only Full100/minus_U100; C4 and all other variants are excluded. Mechanism900 three-view completion and native/shared primary endpoint selection remain pending.','']
    verification.update(old_nine_scene_rows_exact=243,old_nine_display_cells_exact=729,old_records_exact=184,display_cells_verified=display_cells,new_increment_archive_chain=chain,source_functions_exact=functions)
    result=dict(status='TEN_COMPLETE_SCENES_THREE_VIEWS_PENDING_ROOT_TABLE_REVIEW',complete_scenes=10,paired_checkpoints=100,preserved_records=200,table_model_records=200,incomplete_pairs=0,panels=panels,native_shared_identical_records=sum(r['views']['native']==r['views']['shared_calibration'] for r in records),replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_U')},training_torch={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_U')},native_accepted_U=100,native_accepted_other_variants={'minus_C':4},other_variants_excluded=True,primary_endpoint_selected=False,final_test=False,new_inference=0,new_training=0)
    args.output.mkdir();write(args.output/'tables.json',result);write(args.output/'records.json',{'records':records});write(args.output/'paired_per_seed.json',paired);write(args.output/'coverage.json',coverage);write(args.output/'verification.json',verification)
    write(args.output/'SOURCE_BINDINGS.json',dict(prepared_input_pins=pins,new_root_adoption=str(args.root_review),new_root_adoption_sha256=args.root_review_sha,builder_sha256=sha(__file__)))
    (args.output/'TABLES.md').write_text('\n'.join(texts),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],scenes=10,model_records=200,scalar_checks=verification['mean_sd_scalars'])))

if __name__=='__main__':main()
