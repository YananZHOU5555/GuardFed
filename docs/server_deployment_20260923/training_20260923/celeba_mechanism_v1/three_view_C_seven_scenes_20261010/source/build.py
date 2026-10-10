"""Record-only C70 extension; actual exact10 offserver root adoption is mandatory."""
from pathlib import Path
import argparse,importlib.util,json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
OLD=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010'
C60=R/'tmp/celeba_mechanism_valid_C_after56_20261010';C70=R/'tmp/celeba_mechanism_valid_C_after60_20261010'
BENIGN=OLD.with_name('three_view_C_Benign10_20261009')
EXPECTED=[f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
SCENES=[('IID',a) for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']]+[('non-IID','Benign'),('non-IID','F Flip')]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('accepted_C60_helpers',OLD/'build.py');base.R=R
need,read,sha,write=base.need,base.read,base.sha,base.write
METRICS=base.METRICS
def verify_inputs():
 inputs=read(H/'INPUTS.json')
 for n,pin in inputs['files'].items():need(sha(R/n)==pin['sha256'] and (R/n).stat().st_size==pin['bytes'],'Pinned input changed '+n)
 return inputs
def scope(prior,current):
 a={r['id']:r for r in prior['records']};b={r['id']:r for r in current['records']}
 need(len(a)==len(prior['records'])==160 and len(b)==len(current['records'])==170,'Exact160/170 inventories required')
 need(set(b)-set(a)==set(EXPECTED) and all(b[k]==v for k,v in a.items()),'Old160 or exact10 changed')
 need(current['selected_replay_ids']==EXPECTED and set(current['excluded_prior_replay_ids'])==set(a),'Exact10 after160 only')
 need(current['full_references']==prior['full_references'] and len(current['full_references'])==100,'Full100 changed')
 need(current['native_tolerance']==prior['native_tolerance']==1e-12,'Tolerance changed')
 controls={k:r for k,r in b.items() if r['variant']=='minus_C'}
 need(len(controls)==70 and {(r['distribution'],r['attack'],r['seed']) for r in controls.values()}=={(d,a,s) for d,a in SCENES for s in range(91001,91011)},'Only seven complete scenes; no partials')
 for r in controls.values():
  need((r['terminal_round'],r['original_split'],r['original_n_eval'])==(70,'valid',19867),'Terminal/valid changed')
  need(r['config']['ablation_component']=='C' and r['actual_alpha']==(5000 if r['distribution']=='IID' else 5),'Variant/partition changed')
 return controls
def adoption_metadata(proof,science,execution,prior_sha):
 need(proof['status']=='ROOT_C_AFTER60_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS','Actual after60 ROOT adoption required')
 need((proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models'])==(160,10,170),'Actual160+10 required')
 need(proof['accepted_new_ids']==EXPECTED and proof['original160_unchanged'] and proof['source_scope_complete'] and proof['negative_results_preserved'],'Exact10/prior/source changed')
 need(proof['science_seal_sha256']==science and proof['execution_seal_sha256']==execution and proof['prior160_root_adoption_sha256']==prior_sha,'Source/prior changed')
 need(proof['all_native_differences_zero'] and proof['server_strict_bound_in_saved_receipts'],'Native/strict not accepted')
 need(proof['new_training']==proof['new_Full_inference']==0 and proof['test_inference'] is False,'Forbidden science')
def verify_future_binding(binding):
 need(set(binding)=={'stage','science_sha256','execution_sha256','inventory_sha256','adoption','adoption_sha256'},'Exact external binding keys required')
 need(isinstance(binding['stage'],str) and isinstance(binding['adoption'],str),'Actual adopted path missing; prepare is not acceptance')
 stage=(R/binding['stage']).resolve();path=(R/binding['adoption']).resolve()
 need(stage==C70.resolve(),'Only fixed after60 stage')
 need(path.name=='ROOT_ADOPTION_REVIEW.json' and path.parent.parent==(stage/'execution_candidate/backups').resolve(),'Actual backup adoption path required')
 files={'science_sha256':stage/'FILES_SHA256.json','execution_sha256':stage/'execution_candidate/EXECUTION_SOURCE_SHA256.json','inventory_sha256':stage/'inventory_actual170_Full100refs.json','adoption_sha256':path}
 for k,p in files.items():
  v=binding[k];need(isinstance(v,str) and len(v)==64 and set(v)<=set('0123456789abcdef') and sha(p)==v,'Explicit actual SHA required '+k)
 for folder,seal in [(stage,files['science_sha256']),(stage/'execution_candidate',files['execution_sha256'])]:
  for row in read(seal)['members']:
   f=folder/row['path'];need(f.resolve().is_relative_to(folder) and sha(f)==row['sha256'] and f.stat().st_size==row['size'],'Source member changed')
 adoption_metadata(read(path),binding['science_sha256'],binding['execution_sha256'],read(H/'INPUTS.json')['prior160_adoption_sha256'])
 return stage,path
def displayed_cells(text,panels):
 lines=[x for x in text.splitlines() if x.startswith('| ') and ' ± ' in x];rows=[r for p in panels for r in p['rows']]
 need(len(lines)==len(rows)==189,'Exact189 display rows required')
 for line,row in zip(lines,rows):
  cells=line.strip('| ').split(' | ')
  for k,m in enumerate(METRICS,3):need(cells[k]==f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}",'Display changed')
 return len(rows)*3
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--C10-binding',type=Path,required=True);p.add_argument('--C10-binding-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 need(a.output.resolve().parent==H.resolve() and not a.output.exists(),'Fresh owned output only')
 need(sha(a.C10_binding)==a.C10_binding_sha256,'External binding SHA required');inputs=verify_inputs();binding=read(a.C10_binding);stage,adoption=verify_future_binding(binding)
 for row in read(H/'FILES_SHA256.json')['members']:need(sha(H/row['path'])==row['sha256'],'Prepared source changed')
 prior=read(C60/'inventory_actual160_Full100refs.json');current=read(stage/'inventory_actual170_Full100refs.json');controls=scope(prior,current)
 prior_builder=module('accepted_C50_span_reader',OLD.with_name('three_view_C_five_scenes_20261010')/'build.py');prior_builder.R=R
 spans=prior_builder.record_spans((stage/'inventory_actual170_Full100refs.json').read_text('utf8'));oldspans=prior_builder.record_spans((C60/'inventory_actual160_Full100refs.json').read_text('utf8'))
 need([s for s in spans if json.loads(s)['id'] not in EXPECTED]==oldspans,'Old160 raw bytes/order changed')
 accepted=module('accepted_C_join',BENIGN/'build.py');accepted.R=R;accepted.C11=R/'tmp/celeba_mechanism_valid_C_after1_20261009'
 basis=read(BENIGN/'INPUTS.json');join,scientific,functionproof=accepted.source_functions(basis)
 added,chain=join(adoption.parent,binding['adoption_sha256'],current,stage/'inventory_actual170_Full100refs.json',stage/'bridge.py',160,170,scientific['receipt_identity'],scientific['normalized'],scientific['canonical'])
 need([r['id'] for r in added]==EXPECTED,'Exact10 adopted source records required')
 old=read(OLD/'snapshot/records.json')['records'];need(len(old)==120,'Old120 required')
 byid={r['id']:r for r in read(R/inputs['full900_path'])['records']};refs={r['id']:r for r in current['full_references']};full_builder=module('accepted_Full_normalizer',R/basis['parent_builder'])
 full=[full_builder.full_record(byid[controls[i]['paired_full']['id']],refs[controls[i]['paired_full']['id']]) for i in EXPECTED]
 records=old+full+added;need(len(records)==len({r['id'] for r in records})==140,'Exact140 unique records required')
 raw=json.dumps(dict(records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n';need(prior_builder.record_spans(raw)[:120]==prior_builder.record_spans((OLD/'snapshot/records.json').read_text('utf8')),'Old120 JSON bytes/order changed')
 cells={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
 for r in records:
  need(r['variant'] in ['Full','minus_C'],'Wrong variant')
  if r['variant']=='minus_C':
   inv=controls[r['id']];f=cells['Full',r['distribution'],r['attack'],r['seed']]
   need(r['checkpoint_sha256']==inv['checkpoint']['sha256'] and r['config_sha256']==inv['config_canonical_sha256'],'Checkpoint/config changed')
   need(r['data_contract']==inv['data_contract']==f['data_contract'],'Paired data changed')
   need(all(abs(r['views']['native'][m]-inv['prior_validation_metrics'][m])<=1e-12 for m in ['accuracy','aeod','aspd']),'Native mismatch')
 original=module('original_statistics',R/basis['evidence']);pure=module('C70_panels',H/'panels.py');numeric=module('C70_numeric',H/'verify_numeric.py');panels,coverage,paired=pure.panels(records,original);checks=numeric.verify(records,panels)
 filtered=[dict(p,rows=[r for r in p['rows'] if (r['distribution'],r['attack'])!=('non-IID','F Flip')]) for p in panels];need(filtered==read(OLD/'snapshot/tables.json')['panels'],'Old972 statistics changed')
 oldcheck=base.module('accepted_C60_numeric',OLD/'verify_numeric.py');need(oldcheck.verify(old,filtered)=={k:v for k,v in read(OLD/'snapshot/verification.json').items() if k in checks},'Old arithmetic regression changed')
 aggregate=(OLD/'snapshot/cross_scene_seed_first.json').read_bytes();aggregate_check=numeric.verify_aggregate([r for r in old if r['distribution']=='IID'],json.loads(aggregate)['panels'])
 text=['# CelebA Full–minus_C: five IID and two non-IID scenes, three views','','Seven complete scenes,70 matched pairs; valid19867,round70. Mean ± sampleSD(ddof1); differences minus_C−Full. ACC percent; ΔACC pp.','']
 for panel in panels:
  text+=['## '+panel['view']+' — '+panel['label'],'','| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---|---:|---:|---:|---:|']
  for row in panel['rows']:
   vals=[f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in METRICS];text.append('| '+' | '.join([row['distribution']+' '+row['attack'],row['variant'],str(row['n']),*vals])+' |')
  text.append('')
 text+=['AEOD is absoluteTPRgap, not full equalized odds. Raw/native/shared are same-checkpoint parallel outputs; native includes original root-only calibration, shared uses the frozen common fitting rule. No inference/refit here.','Mixed CPU/GPU replay, training CUDA/driver histories, recipe-selectionseed91001 and prior valid/officialtest exposure remain in each record. All10/9/6 panels and unfavorable/zero results retained; no necessity, significance, pure aggregation causality or chosen primary endpoint.','Five IID scenes plus non-IID Benign/F Flip are complete; three non-IID C attack scenes and six other image controls remain incomplete. Original162-scalar seed-first aggregate remains fiveIID only; no imbalanced seven-scene global mean or finaltest claim.','']
 rendered='\n'.join(text);display=displayed_cells(rendered,panels);oldlines=[x for x in (OLD/'snapshot/TABLES.md').read_text('utf8').splitlines() if x.startswith('| ') and ' ± ' in x];kept=[x for x in rendered.splitlines() if x.startswith('| ') and ' ± ' in x and not x.startswith('| non-IID F Flip |')];need(oldlines==kept and len(kept)*3==486,'Old486 display cells/order changed')
 verify_inputs();verify_future_binding(binding);a.output.mkdir()
 write(a.output/'records.json',dict(records=records));write(a.output/'tables.json',dict(status='C_SEVEN_COMPLETE_SCENES_THREE_VIEWS_PENDING_ROOT_REVIEW',unique_records=140,paired_models=70,complete_scenes=7,panels=panels,nonIID_complete_scenes=['Benign','F Flip'],full_nonIID_coverage=False,primary_endpoint_selected=False,final_test=False,new_inference=0))
 (a.output/'cross_scene_seed_first.json').write_bytes(aggregate);write(a.output/'coverage.json',coverage);write(a.output/'paired_per_seed.json',paired)
 write(a.output/'verification.json',dict(checks,display_mean_sd_cells=display,old120_records_bytes_exact=True,old972_statistics_exact=True,old486_cells_exact=True,old162_IID_seed_first_bytes_exact=True,preserved_IID_seed_first=aggregate_check))
 write(a.output/'SOURCE_BINDINGS.json',dict(prepared_input_pins=inputs['files'],actual_C10_binding=binding,external_binding_sha256=a.C10_binding_sha256,new10_archive_chain=chain,source_functions=functionproof,original_C60_root_sha256=sha(OLD/'ROOT_VERIFICATION.json'),new_inference=0));(a.output/'TABLES.md').write_text(rendered,encoding='utf8',newline='\n')
 write(a.output/'FILES_SHA256.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(a.output.iterdir()) if p.is_file()}));print(json.dumps(dict(status='BUILT_PENDING_INDEPENDENT_ROOT_REVIEW',records=140,mean_SD_scalars=1134,cells=567,count_metrics=1260,old_IID_aggregate_scalars_preserved=162)))
if __name__=='__main__':main()
