"""Prepared record-only six-scene table; no statistics before actual exact4 adoption."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
OLD=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010'
C6=R/'tmp/celeba_mechanism_valid_C_after50_20261010'
BENIGN=OLD.with_name('three_view_C_Benign10_20261009')
EXPECTED=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
TEN=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91001,91011)]
METRICS=['accuracy_pct','aeod','aspd']

def need(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def verify_inputs():
    inputs=read(H/'INPUTS.json')
    for name,pin in inputs['files'].items():need(sha(R/name)==pin['sha256'] and (R/name).stat().st_size==pin['bytes'],'Pinned input changed '+name)
    return inputs

def scope(prior,current):
    a={r['id']:r for r in prior['records']};b={r['id']:r for r in current['records']}
    need(len(a)==len(prior['records'])==156 and len(b)==len(current['records'])==160,'Exact156/160 inventories required')
    need(set(b)-set(a)==set(EXPECTED) and all(b[k]==v for k,v in a.items()),'Original156 or exact4 delta changed')
    need(current['selected_replay_ids']==EXPECTED and set(current['excluded_prior_replay_ids'])==set(a),'Only exact4 after156 scope')
    need(current['full_references']==prior['full_references'] and len(current['full_references'])==100,'Full100 references changed')
    need(current['native_tolerance']==prior['native_tolerance']==1e-12,'Tolerance changed')
    controls={k:r for k,r in b.items() if r['variant']=='minus_C'}
    scenes=[('IID',a) for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']]+[('non-IID','Benign')]
    need(len(controls)==60 and {(r['distribution'],r['attack'],r['seed']) for r in controls.values()}=={(d,a,s) for d,a in scenes for s in range(91001,91011)},'Exact six complete C scenes required; partials forbidden')
    for r in controls.values():
        need((r['terminal_round'],r['original_split'],r['original_n_eval'])==(70,'valid',19867),'Terminal/valid identity changed')
        need(r['config']['ablation_component']=='C' and r['actual_alpha']==(5000 if r['distribution']=='IID' else 5),'Variant/partition identity changed')
    return controls

def adoption_metadata(proof,science,execution,prior_sha):
    need(proof['status']=='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS','Actual after56 ROOT adoption required')
    need((proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models'])==(156,4,160),'Actual156+4 required')
    need(proof['accepted_new_ids']==EXPECTED and proof['original156_unchanged'] and proof['source_scope_complete'] and proof['negative_results_preserved'],'Exact4/prior/source boundary changed')
    need(proof['science_seal_sha256']==science and proof['execution_seal_sha256']==execution and proof['prior156_root_adoption_sha256']==prior_sha,'Source/prior pins changed')
    need(proof['all_native_differences_zero'] and proof['server_strict_bound_in_saved_receipts'],'Native/strict not accepted')
    need(proof['new_training']==proof['new_Full_inference']==0 and proof['test_inference'] is False,'Forbidden science')

def verify_future_binding(binding):
    need(set(binding)=={'stage','science_sha256','execution_sha256','inventory_sha256','adoption','adoption_sha256'},'Exact external future binding keys required')
    stage=(R/binding['stage']).resolve();path=(R/binding['adoption']).resolve()
    need(stage==(R/'tmp/celeba_mechanism_valid_C_after56_20261010').resolve(),'Only independent after56 stage permitted')
    need(path.name=='ROOT_ADOPTION_REVIEW.json' and path.parent.parent==(stage/'execution_candidate/backups').resolve(),'Actual backup adoption path required')
    files={'science_sha256':stage/'FILES_SHA256.json','execution_sha256':stage/'execution_candidate/EXECUTION_SOURCE_SHA256.json','inventory_sha256':stage/'inventory_actual160_Full100refs.json','adoption_sha256':path}
    for k,p in files.items():
        value=binding[k];need(isinstance(value,str) and len(value)==64 and set(value)<=set('0123456789abcdef') and sha(p)==value,'Explicit actual future SHA required '+k)
    for folder,seal in [(stage,files['science_sha256']),(stage/'execution_candidate',files['execution_sha256'])]:
        for row in read(seal)['members']:
            f=folder/row['path'];need(f.resolve().is_relative_to(folder) and sha(f)==row['sha256'] and f.stat().st_size==row['size'],'Future source member changed')
    inputs=read(H/'INPUTS.json');adoption_metadata(read(path),binding['science_sha256'],binding['execution_sha256'],inputs['prior_C6_adoption_sha256'])
    return stage,path

def displayed_cells(text,panels):
    lines=[x for x in text.splitlines() if x.startswith('| ') and ' ± ' in x];rows=[r for p in panels for r in p['rows']]
    need(len(lines)==len(rows)==162,'Exact162 displayed rows required')
    for line,row in zip(lines,rows):
        cells=line.strip('| ').split(' | ')
        for k,m in enumerate(METRICS,3):need(cells[k]==f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}",'Display changed')
    return len(rows)*3

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--C4-binding',type=Path,required=True);p.add_argument('--C4-binding-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    need(args.output.resolve().parent==H.resolve() and not args.output.exists(),'Fresh owned output only')
    need(sha(args.C4_binding)==args.C4_binding_sha256,'Externally pinned C4 binding required')
    inputs=verify_inputs();binding=read(args.C4_binding);stage,adoption=verify_future_binding(binding)
    for row in read(H/'FILES_SHA256.json')['members']:need(sha(H/row['path'])==row['sha256'],'Prepared source changed')
    prior=read(C6/'inventory_actual156_Full100refs.json');current=read(stage/'inventory_actual160_Full100refs.json');controls=scope(prior,current)
    old_builder=module('accepted_C50_builder',OLD/'build.py');old_builder.R=R
    oldspans=old_builder.record_spans((C6/'inventory_actual156_Full100refs.json').read_text('utf8'))
    spans=old_builder.record_spans((stage/'inventory_actual160_Full100refs.json').read_text('utf8'))
    need([s for s in spans if json.loads(s)['id'] not in EXPECTED]==oldspans,'Original156 raw record bytes/order changed')
    accepted=module('accepted_C_join',BENIGN/'build.py');accepted.R=R;accepted.C11=R/'tmp/celeba_mechanism_valid_C_after1_20261009'
    basis=read(BENIGN/'INPUTS.json');join,scientific,functionproof=accepted.source_functions(basis)
    first_path=R/inputs['prior_C6_adoption']
    first,chain6=join(first_path.parent,inputs['prior_C6_adoption_sha256'],prior,C6/'inventory_actual156_Full100refs.json',C6/'bridge.py',150,156,scientific['receipt_identity'],scientific['normalized'],scientific['canonical'])
    added,chain4=join(adoption.parent,binding['adoption_sha256'],current,stage/'inventory_actual160_Full100refs.json',stage/'bridge.py',156,160,scientific['receipt_identity'],scientific['normalized'],scientific['canonical'])
    need([r['id'] for r in first]==TEN[:6] and [r['id'] for r in added]==EXPECTED,'Exact6+4 actual source records required')
    old=read(OLD/'snapshot/records.json')['records'];need(len(old)==100,'Old100 required')
    source=read(R/inputs['full900_path']);byid={r['id']:r for r in source['records']};refs={r['id']:r for r in current['full_references']}
    full_builder=module('accepted_Full_normalizer',R/basis['parent_builder'])
    full=[full_builder.full_record(byid[controls[i]['paired_full']['id']],refs[controls[i]['paired_full']['id']]) for i in TEN]
    records=old+full+first+added;need(len(records)==len({r['id'] for r in records})==120,'Exact120 unique records required')
    raw=json.dumps(dict(records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    need(old_builder.record_spans(raw)[:100]==old_builder.record_spans((OLD/'snapshot/records.json').read_text('utf8')),'Old100 JSON bytes/order changed')
    cells={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    for r in records:
        need(r['variant'] in ['Full','minus_C'],'Wrong variant')
        if r['variant']=='minus_C':
            inv=controls[r['id']];f=cells['Full',r['distribution'],r['attack'],r['seed']]
            need(r['checkpoint_sha256']==inv['checkpoint']['sha256'] and r['config_sha256']==inv['config_canonical_sha256'],'Checkpoint/config changed')
            need(r['data_contract']==inv['data_contract']==f['data_contract'],'Paired data identity changed')
            need(all(abs(r['views']['native'][m]-inv['prior_validation_metrics'][m])<=1e-12 for m in ['accuracy','aeod','aspd']),'Native record mismatch')
    original=module('original_statistics',R/basis['evidence']);pure=module('C60_panels',H/'panels.py');verify=module('C60_numeric',H/'verify_numeric.py')
    panels,coverage,paired=pure.panels(records,original);checks=verify.verify(records,panels)
    oldpanels=read(OLD/'snapshot/tables.json')['panels'];filtered=[dict(p,rows=[r for r in p['rows'] if r['distribution']=='IID']) for p in panels]
    need(filtered==oldpanels,'Old810 statistics changed')
    oldverifier=module('accepted_C50_numeric',OLD/'verify_numeric.py');regression=oldverifier.verify(old,filtered)
    need(regression=={k:v for k,v in read(OLD/'snapshot/verification.json').items() if k in regression},'Old numeric regression changed')
    aggregate=(OLD/'snapshot/cross_scene_seed_first.json').read_bytes();aggregate_check=verify.verify_aggregate(old,json.loads(aggregate)['panels'])
    text=['# CelebA Full–minus_C: five IID scenes and non-IID Benign, three views','','Six complete scenes, sixty matched pairs; valid-only19,867, round70. Mean ± sample SD(ddof1); differences minus_C − Full. ACC percent; ΔACC percentage points. ACC higher/gaps lower better.','']
    for panel in panels:
        text+=['## '+panel['view']+' — '+panel['label'],'','| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            vals=[f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in METRICS]
            text.append('| '+' | '.join([row['distribution']+' '+row['attack'],row['variant'],str(row['n']),*vals])+' |')
        text.append('')
    text+=['AEOD is absolute TPR gap, not full equalized odds. Native includes original root-only calibration; raw is uncalibrated; shared uses the frozen common root-only rule. Same terminal checkpoint for all metrics/views. No inference/refit here.','Mixed CPU/GPU replay and training CUDA/driver provenance, seed91001 configuration selection and historical validation/test exposure are retained. Panels10/9/6 are descriptive, not untouched confirmation. All negative/zero outcomes retained; no necessity/aggregation-causal or significance claim.','IID five scenes are complete; only non-IID Benign is complete. Other four non-IID C scenes and overall mechanism/rebuttal remain incomplete. Preserved seed-first aggregate covers only the original five IID scenes. No imbalanced six-scene global average, new endpoint or final-test claim.','']
    rendered='\n'.join(text);display=displayed_cells(rendered,panels)
    oldlines=[x for x in (OLD/'snapshot/TABLES.md').read_text('utf8').splitlines() if x.startswith('| ') and ' ± ' in x];newlines=[x for x in rendered.splitlines() if x.startswith('| IID ') and ' ± ' in x]
    need(oldlines==newlines and len(oldlines)*3==405,'Old405 cells/order changed')
    verify_inputs();verify_future_binding(binding);args.output.mkdir()
    write(args.output/'records.json',dict(records=records));write(args.output/'tables.json',dict(status='C_SIX_COMPLETE_SCENES_THREE_VIEWS_PENDING_ROOT_REVIEW',unique_records=120,paired_models=60,complete_scenes=6,panels=panels,nonIID_complete_scenes=['Benign'],full_nonIID_coverage=False,primary_endpoint_selected=False,final_test=False,new_inference=0))
    (args.output/'cross_scene_seed_first.json').write_bytes(aggregate)
    write(args.output/'coverage.json',coverage);write(args.output/'paired_per_seed.json',paired)
    write(args.output/'verification.json',dict(checks,display_mean_sd_cells=display,old100_records_bytes_exact=True,old810_statistics_exact=True,old405_cells_exact=True,old162_IID_seed_first_bytes_exact=True,preserved_IID_seed_first=aggregate_check))
    write(args.output/'SOURCE_BINDINGS.json',dict(prepared_input_pins=inputs['files'],actual_C4_binding=binding,external_binding_sha256=args.C4_binding_sha256,prior6_archive_chain=chain6,new4_archive_chain=chain4,source_functions=functionproof,original_C50_root_sha256=sha(OLD/'ROOT_VERIFICATION.json'),new_inference=0))
    (args.output/'TABLES.md').write_text(rendered,encoding='utf8',newline='\n')
    write(args.output/'FILES_SHA256.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(status='BUILT_PENDING_INDEPENDENT_ROOT_REVIEW',records=120,mean_SD_scalars=972,cells=486,count_metrics=1080,old_IID_seed_first_scalars_preserved=162)))

if __name__=='__main__':main()
