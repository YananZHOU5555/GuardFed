"""Preserve adopted Benign10; append adopted F Flip10 with original statistics, no fitting."""
from pathlib import Path
from collections import Counter
import ast,hashlib,json,statistics,subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
ROOT=R/'tmp/hybrid_FFlip10_root_adoption_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
OLD=R/'outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011'
EVIDENCE=R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
METRICS=['accuracy_pct','aeod','aspd'];VIEWS=['native','raw','shared_calibration']
PANELS=[('All 10 seeds',list(range(91001,91011))),('Exclude selection seed: 9 seeds',list(range(91002,91011))),('Seeds 91005–91010: 6 seeds',list(range(91005,91011)))]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def need(ok,message):
 if not ok:raise ValueError(message)
def save(name,value):
 with (H/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
 need(__debug__ and not (H/'records.json').exists(),'Fresh output required; no overwrite or retry')
 need(sha(ROOT)=='ed315fe982eae5a51fca97b658898b90eb7150cc9b29df8bc0597122d3544c94','Adopted root changed')
 inputs=read(H/'ROOT_INPUTS.json');need(sha(H/'ROOT_INPUTS.json')==BINDING_SHA and inputs['execution_authorized'] is True,'Actual root authorization/binding required')
 need(inputs['FFlip_root_sha256']==sha(ROOT) and inputs['Benign_table_root_sha256']==sha(OLD/'ROOT_VERIFICATION.json'),'Actual root bindings changed')
 for n,pin in read(OLD/'ROOT_VERIFICATION.json')['files'].items():need(sha(OLD/n)==pin['sha256'],'Old adopted table member changed')
 adopted=read(ROOT);need(adopted['root_adoption'] is True and adopted['new_three_view_records_accepted']==10 and adopted['Hybrid_total_three_view_records']==20,'Root scope differs')
 volume=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True));need(volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy','F missing/unhealthy')
 accepted=adopted['new_records']
 need(adopted['old_Benign10_root']['sha256']=='02b5f2c8fd26808f62a9e94d1901f4587d09d13dcce29b118113ccc0e1374e06','Old Benign root changed')
 need(adopted['Linux_whole_original_saved_check_pass'] and adopted['Windows_saved_outputs_audit_pass'] and adopted['Windows_saved_outputs_audit_fit_calls']==0 and adopted['cross_platform_bitwise_recalibration_claimed'] is False,'Complementary saved proof roles changed')
 expected=['CosineFairness_lam20.0_tau0.1_lr0.001_IID_F Flip_seed'+str(s)+'_fullcoverage' for s in range(91001,91011)]
 byid={r['id']:r for r in accepted};need(len(accepted)==len(byid)==10 and set(byid)==set(expected),'Exact ten unique checkpoints required')
 reuse=read(OLD/'ENVIRONMENT_SCOPE.json')
 transport_pin=next(x for x in adopted['proof_files'] if x['path'].endswith('/TRANSPORT_VERIFICATION.json'));need(sha(R/transport_pin['path'])==transport_pin['sha256'],'Adopted transport changed')
 transport=read(R/transport_pin['path']);need(transport['archive_sha256']==adopted['archive_sha256'] and transport['member_count']==adopted['archive_members']==24,'Adopted saved member scope changed')
 old_records=read(OLD/'records.json')['records'];old_tables=read(OLD/'tables.json');need(len(old_records)==10 and len(old_tables['panels'])==9,'Old Benign table scope differs')
 oldby={r['id']:{'checkpoint_sha256':r['checkpoint_sha256'],'metrics':r['native_comparison']['expected']} for r in accepted}
 records=[];bindings=[]
 for rid in expected:
  a=byid[rid];seed=a['seed'];phase='fullcoverage'
  member='bundle/'+rid+'/receipt.json';pin=transport['members'][member];path=Path(transport['verified_extract'])/member;role='actual_root_adopted_FFlip10'
  need(path.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) and sha(path)==a['receipt_sha256']==pin['sha256'] and path.stat().st_size==pin['bytes'],'Adopted receipt identity changed')
  receipt=read(path)
  need(receipt['id']==rid and receipt['method']=='CosineFairnessHybrid' and receipt['distribution']=='IID' and receipt['attack']=='F Flip' and receipt['seed']==seed,'Receipt cell differs')
  need(receipt['checkpoint_sha256']==a['checkpoint_sha256']==oldby[rid]['checkpoint_sha256'] and receipt['prediction_arrays_sha256']==a['array_sha256'],'Checkpoint/array provenance differs')
  need(receipt['valid_n']==19867 and receipt['valid_image_ids_sha256']=='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf' and receipt['test_labels_accessed'] is False and receipt['test_inference_performed'] is False,'Valid/test identity differs')
  need(receipt['weights_before']==receipt['weights_after'] and all(p['dtype']=='<f4' for p in receipt['weights_before'].values()) and receipt['runtime']['device']=='cpu','Same FP32 checkpoint across views required')
  need(receipt['native_comparison']['accepted'] and receipt['native_comparison']['tolerance']==1e-12 and receipt['native_comparison']['max_abs_difference']<=1e-12,'Accepted native replay missing')
  root=receipt['root_reconstruction'];need(root['root_n']+sum(root['client_sample_counts'])==162770 and root['train_eval_disjoint'] and root['root_client_disjoint'],'Train/root/valid partition identity differs')
  need(set(receipt['views'])==set(VIEWS),'Missing view')
  fits=receipt['fits']
  need(all(fits[v]['rule']=='argmax_margin_strictly_positive' and fits[v]['fit_data']=='none' for v in ['native','raw']) and fits['shared_calibration']['rule']=='group_margin_greater_equal' and fits['shared_calibration']['fit_data']=='clean_train_root_only','Frozen prediction/fit policy differs')
  need({k:receipt['views']['native'][k] for k in ['accuracy','aeod','aspd']}==oldby[rid]['metrics'],'Old native per-checkpoint metrics changed')
  record={k:receipt[k] for k in ['id','method','distribution','attack','seed','checkpoint_sha256','original_result_sha256','original_job_sha256','config_canonical_sha256','original_training_torch','runtime','root_reconstruction','valid_n','valid_image_ids_sha256','fits','views','native_comparison','prediction_arrays_sha256','weights_before','weights_after','zero_margin_count']}
  record.update(terminal_round=70,split='valid',phase=phase,selection_seed=seed==91001,source_role=role,source_receipt={'path':path.as_posix(),'sha256':sha(path),'bytes':path.stat().st_size},adopted_root_sha256=sha(ROOT))
  records.append(record);bindings.append({'id':rid,'seed':seed,'phase':phase,'checkpoint_sha256':a['checkpoint_sha256'],'receipt_sha256':a['receipt_sha256'],'array_sha256':a['array_sha256'],'source_receipt_path':path.as_posix(),'adopted_source_pointer':'/new_records/'+str(adopted['new_records'].index(a))})
 need(sha(EVIDENCE)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef','Original statistics source changed')
 source=EVIDENCE.read_text(encoding='utf8');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='statistic')
 ns={'require':need,'statistics':statistics,'METRICS':METRICS};exec(compile(ast.Module(body=[node],type_ignores=[]),str(EVIDENCE),'exec'),ns)
 panels=list(old_tables['panels'])
 for view in VIEWS:
  for label,seeds in PANELS:
   selected=[r for r in records if r['seed'] in seeds];need([r['seed'] for r in selected]==seeds,'Same-seed panel incomplete')
   flat=[{'seed':r['seed'],'accuracy_pct':100*r['views'][view]['accuracy'],'aeod':r['views'][view]['aeod'],'aspd':r['views'][view]['aspd']} for r in selected]
   row=dict(method='CosineFairnessHybrid',distribution='IID',attack='F Flip',**ns['statistic'](flat,len(seeds)))
   row['display']={m:f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in METRICS}
   panels.append({'view':view,'label':label,'seeds':seeds,'ids':[r['id'] for r in selected],'checkpoint_sha256':[r['checkpoint_sha256'] for r in selected],'rows':[row]})
 records=old_records+records
 environment={'recipe':read(OLD/'ENVIRONMENT_SCOPE.json')['recipe'],'selection_seed':91001,'selection_conditions':'Original four-condition exposed-valid search; selected recipe unchanged.','prior_official_test_exposure':True,'new_final_test_evaluation':False,'training_torch':dict(Counter(r['original_training_torch'] for r in records)),'replay_torch':dict(Counter(r['runtime']['torch'] for r in records)),'replay_devices':dict(Counter(r['runtime']['device'] for r in records)),'replay_threads':dict(Counter(str(r['runtime']['torch_threads']) for r in records)),'native_raw_exact_all_records':all(r['views']['native']==r['views']['raw'] for r in records),'root_only_shared_calibration_original_recipe':reuse['root_only_shared_calibration_original_recipe'],'Linux_whole_original_saved_check_pass':True,'Windows_saved_outputs_zero_fit_pass':True,'Windows_whole_combined_pass':False,'cross_platform_bitwise_recalibration_claimed':False,'original_Windows8_pre_record_failure_preserved':True,'exact_diagnostic_maps':{'Benign':reuse['exact_diagnostic_maps'],'FFlip':adopted['exact_root_receipt_differences']},'new_fit':0,'new_CNN':0,'new_training':0,'table_root_adopted':False,'final_primary_endpoint_decided':False,'significance_or_necessity_or_causality_claimed':False}
 text=['# CosineFairnessHybrid: two IID scenes, three validation views','',
 'Twenty fixed round-70 checkpoints across two separately reported scenes, 19,867 validation images; raw/native/shared use the same checkpoint and seed panel. Mean ± sample SD (ddof=1). ACC (%) ↑; AEOD and ASPD ↓. AEOD is the absolute TPR gap, not full equalized odds. Formal primary endpoint remains pending.','',
 '## IID / Benign', '', '| View | Fixed seed panel | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |','|---|---|---:|---:|---:|---:|']
 for index,p in enumerate(panels):
  if index==9:text += ['', '## IID / F Flip', '', '| View | Fixed seed panel | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |','|---|---|---:|---:|---:|---:|']
  row=p['rows'][0];text.append('| '+' | '.join([p['view'],p['label'],str(row['n']),*[row['display'][m] for m in METRICS]])+' |')
 text += ['',
 'Benign seed91001 is the originally selected screen checkpoint; its other nine and all ten F Flip checkpoints retain their formal-coverage phase. Seed91001 was exposed during recipe selection in both conditions. The all10 panel retains the selected seed; n9 excludes only91001; n6 fixes91005–91010. Validation was exposed during development, including development after the initial official-test results were viewed. These descriptive subsets are not untouched confirmation sets; no new final-test evaluation was performed.',
 'Raw and native retain margin > 0 (zero-margin ties predict0); shared_calibration uses the frozen common clean-training-root-only thresholds with margin ≥ group threshold. This table only reads saved accepted receipts: it performs no calibration fit, forward pass or training. Any raw/native equality is retained as measured, not treated as independent confirmation. The shared view is a diagnostic under the fixed recipe, not a newly selected endpoint.',
 f"All20 training receipts report {environment['training_torch']}; replay receipts report {environment['replay_devices']} and {environment['replay_torch']}, with8 Torch threads. Original CUDA training and CPU replay remain distinct environments; no equal-device trajectories or cross-platform bitwise recalibration are claimed.",
 'Linux whole supplied the original cached-root-refit checks; Windows supplied zero-fit saved-output audits. The original Windows8 import failure (completed0) remains preserved. For F Flip seed91005, the /server_sampling_audit/group_kl difference is −2.168404344971009e−19; the original exact diagnostic maps are retained separately. This difference does not alter the saved threshold/prediction/metric/count audit and is not relabelled as cross-platform bitwise identity.',
 'These are IID Benign and IID F Flip only, with nineteen formal endpoints plus one selected screen checkpoint. No across-scene mean is formed. It is not full100, a Full-versus-ablation pairing, all methods, causal isolation, statistical significance or final-test completion. Other results and negative outcomes are not filtered.','']
 save('records.json',{'records':records,'all_metrics_same_checkpoint_across_views':True,'same_seed_panels':True,'adopted_root_sha256':sha(ROOT)})
 save('tables.json',{'status':'ACTUAL_SAVED_RECEIPT_TABLE_ROOT_TABLE_REVIEW_PENDING','complete_scenes':2,'unique_checkpoints':20,'panels':panels,'units':{'accuracy_pct':'percent','aeod':'fraction;absolute TPR gap','aspd':'fraction'},'ddof':1,'mean_SD_scalars':108,'display_cells':54,'new_fit':0,'new_inference':0,'new_training':0,'final_test':False})
 save('coverage.json',{'complete_scenes':2,'distribution':'IID','attacks':['Benign','F Flip'],'seeds':list(range(91001,91011)),'formal_checkpoints':19,'selected_screen_checkpoints':1,'seed_panels':[s for _,s in PANELS],'ids':[r['id'] for r in records],'old_Benign10_records_panels_and_display_exact':True,'other_scenes_excluded':True,'full100_complete':False})
 save('ENVIRONMENT_SCOPE.json',environment)
 save('SOURCE_BINDINGS.json',{'root_path':ROOT.relative_to(R).as_posix(),'root_sha256':sha(ROOT),'old_Benign_table_root_sha256':sha(OLD/'ROOT_VERIFICATION.json'),'old_native_seal_sha256':sha(OLD/'FILES_SHA256.json'),'old_native_table_preserved_not_overwritten':True,'original_statistic_source':EVIDENCE.relative_to(R).as_posix(),'original_statistic_file_sha256':sha(EVIDENCE),'original_statistic_function_source_sha256':hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),'original_statistic_AST_unmodified':True,'receipt_bindings':read(OLD/'SOURCE_BINDINGS.json')['receipt_bindings']+bindings,'F_read_volume':volume,'no_arrays_models_or_ZIP_read':True})
 with (H/'TABLES.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text))
 print(json.dumps({'status':'HYBRID_TWO_SCENES20_THREE_VIEW_TABLE_BUILT_ROOT_TABLE_REVIEW_PENDING','checkpoints':20,'panels':18,'mean_SD_scalars':108,'display_cells':54,'native_raw_equal':environment['native_raw_exact_all_records'],'training_torch':environment['training_torch'],'replay_devices':environment['replay_devices']}))
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--binding-sha256',required=True);BINDING_SHA=parser.parse_args().binding_sha256;main()
