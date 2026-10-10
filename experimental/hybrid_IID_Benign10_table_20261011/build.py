"""Read ten root-adopted receipts; reuse original statistic without fitting."""
from pathlib import Path
from collections import Counter
import ast,hashlib,json,statistics,subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
ROOT=R/'tmp/hybrid_missing8_root_adoption_prepared_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
OLD=R/'outputs/guardfed_tables/celeba_hybrid_IID_Benign10_20261011'
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
 need(sha(ROOT)=='02b5f2c8fd26808f62a9e94d1901f4587d09d13dcce29b118113ccc0e1374e06','Adopted root changed')
 adopted=read(ROOT);need(adopted['root_adoption'] is True and adopted['new_three_view_records_accepted']==9 and adopted['Hybrid_total_three_view_records']==10,'Root scope differs')
 volume=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True));need(volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy','F missing/unhealthy')
 prior=adopted['prior_interface_explicitly_reused'];accepted=adopted['records']+prior
 expected=['CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed'+str(s)+('_screen' if s==91001 else '_fullcoverage') for s in range(91001,91011)]
 byid={r['id']:r for r in accepted};need(len(accepted)==len(byid)==10 and set(byid)==set(expected),'Exact ten unique checkpoints required')
 reuse=read(R/'tmp/hybrid_missing8_pool32_runtime_prepared_20261011/proofs/existing_replay_reuse_check.json')
 transports={8:read(R/adopted['proof_files']['F_transport']['path']),1:read(R/adopted['proof_files']['screen_F_transport']['path'])}
 old_records=read(OLD/'records.json')['records'];oldby={r['id']:r for r in old_records};need(set(oldby)==set(expected),'Old native ten identity grid differs')
 records=[];bindings=[]
 for rid in expected:
  a=byid[rid];seed=a['seed'];phase='screen' if seed==91001 else 'fullcoverage'
  if seed==91002:
   pin=reuse['existing_saved_members']['receipt.json'];path=Path(pin['path']);role='prior_root_adopted_interface_reuse'
  else:
   t=transports[1 if seed==91001 else 8];member='bundle/'+rid+'/receipt.json';pin=t['members'][member];path=Path(t['verified_extract'])/member;role='actual_selected_screen1' if seed==91001 else 'actual_root_adopted_formal8'
  need(path.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) and sha(path)==a['receipt_sha256']==pin['sha256'] and path.stat().st_size==pin['bytes'],'Adopted receipt identity changed')
  receipt=read(path)
  need(receipt['id']==rid and receipt['method']=='CosineFairnessHybrid' and receipt['distribution']=='IID' and receipt['attack']=='Benign' and receipt['seed']==seed,'Receipt cell differs')
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
  records.append(record);bindings.append({'id':rid,'seed':seed,'phase':phase,'checkpoint_sha256':a['checkpoint_sha256'],'receipt_sha256':a['receipt_sha256'],'array_sha256':a['array_sha256'],'source_receipt_path':path.as_posix(),'adopted_source_pointer':'/prior_interface_explicitly_reused/0' if seed==91002 else '/records/'+str(adopted['records'].index(a))})
 need(sha(EVIDENCE)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef','Original statistics source changed')
 source=EVIDENCE.read_text(encoding='utf8');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='statistic')
 ns={'require':need,'statistics':statistics,'METRICS':METRICS};exec(compile(ast.Module(body=[node],type_ignores=[]),str(EVIDENCE),'exec'),ns)
 panels=[]
 for view in VIEWS:
  for label,seeds in PANELS:
   selected=[r for r in records if r['seed'] in seeds];need([r['seed'] for r in selected]==seeds,'Same-seed panel incomplete')
   flat=[{'seed':r['seed'],'accuracy_pct':100*r['views'][view]['accuracy'],'aeod':r['views'][view]['aeod'],'aspd':r['views'][view]['aspd']} for r in selected]
   row=dict(method='CosineFairnessHybrid',distribution='IID',attack='Benign',**ns['statistic'](flat,len(seeds)))
   row['display']={m:f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in METRICS}
   panels.append({'view':view,'label':label,'seeds':seeds,'ids':[r['id'] for r in selected],'checkpoint_sha256':[r['checkpoint_sha256'] for r in selected],'rows':[row]})
 environment={'recipe':read(OLD/'ENVIRONMENT_SCOPE.json')['recipe'],'selection_seed':91001,'selection_conditions':'Original four-condition exposed-valid search; selected recipe unchanged.','prior_official_test_exposure':True,'new_final_test_evaluation':False,'training_torch':dict(Counter(r['original_training_torch'] for r in records)),'replay_torch':dict(Counter(r['runtime']['torch'] for r in records)),'replay_devices':dict(Counter(r['runtime']['device'] for r in records)),'replay_threads':dict(Counter(str(r['runtime']['torch_threads']) for r in records)),'native_raw_exact_all_records':all(r['views']['native']==r['views']['raw'] for r in records),'root_only_shared_calibration_original_recipe':reuse['shared_calibration'],'Linux_whole_original_saved_check_pass':True,'Windows_saved_outputs_zero_fit_pass':True,'Windows_whole_combined_pass':False,'cross_platform_bitwise_recalibration_claimed':False,'original_Windows8_pre_record_failure_preserved':True,'exact_diagnostic_maps':{'formal8':adopted['exact_diagnostic_map_sha256'],'screen1':adopted['proof_files']['screen_exact_diagnostic_map']['sha256']},'new_fit':0,'new_CNN':0,'new_training':0,'table_root_adopted':False,'final_primary_endpoint_decided':False,'significance_or_necessity_or_causality_claimed':False}
 text=['# CosineFairnessHybrid: IID Benign, three validation views','',
 'Ten fixed round-70 checkpoints, 19,867 validation images; raw/native/shared use the same checkpoint and seed panel. Mean ± sample SD (ddof=1). ACC (%) ↑; AEOD and ASPD ↓. AEOD is the absolute TPR gap, not full equalized odds. Formal primary endpoint remains pending.','',
 '| View | Fixed seed panel | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |','|---|---|---:|---:|---:|---:|']
 for p in panels:
  row=p['rows'][0];text.append('| '+' | '.join([p['view'],p['label'],str(row['n']),*[row['display'][m] for m in METRICS]])+' |')
 text += ['',
 'Seed91001 is the originally selected screen checkpoint; seeds91002–91010 are formal-coverage checkpoints. The all10 panel retains the selected seed; n9 excludes only91001; n6 fixes91005–91010. Validation was exposed during development, including development after the initial official-test results were viewed. These descriptive subsets are not untouched confirmation sets; no new final-test evaluation was performed.',
 'Raw and native retain margin > 0 (zero-margin ties predict0); shared_calibration uses the frozen common clean-training-root-only thresholds with margin ≥ group threshold. This table only reads saved accepted receipts: it performs no calibration fit, forward pass or training. Raw/native equality is retained as measured, not treated as independent confirmation. The shared view is a diagnostic under the fixed recipe, not a newly selected endpoint.',
 f"All10 training receipts report {environment['training_torch']}; replay receipts report {environment['replay_devices']} and {environment['replay_torch']}, with8 Torch threads. Original CUDA training and CPU replay remain distinct environments; no equal-device trajectories or cross-platform bitwise recalibration are claimed.",
 'Linux whole supplied the original cached-root-refit checks; Windows supplied zero-fit saved-output audits. The original Windows8 import failure (completed0) remains preserved. The exact seed91005 group-KL diagnostic difference is retained separately; it does not alter the saved threshold/prediction/metric/count audit.',
 'This is one IID Benign scene only, with nine formal endpoints plus one selected screen checkpoint. It is not full100, a Full-versus-ablation pairing, all methods, causal isolation, statistical significance or final-test completion. Other results and negative outcomes are not filtered.','']
 save('records.json',{'records':records,'all_metrics_same_checkpoint_across_views':True,'same_seed_panels':True,'adopted_root_sha256':sha(ROOT)})
 save('tables.json',{'status':'ACTUAL_SAVED_RECEIPT_TABLE_ROOT_TABLE_REVIEW_PENDING','complete_scenes':1,'unique_checkpoints':10,'panels':panels,'units':{'accuracy_pct':'percent','aeod':'fraction;absolute TPR gap','aspd':'fraction'},'ddof':1,'mean_SD_scalars':54,'display_cells':27,'new_fit':0,'new_inference':0,'new_training':0,'final_test':False})
 save('coverage.json',{'complete_scenes':1,'distribution':'IID','attack':'Benign','seeds':list(range(91001,91011)),'formal_checkpoints':9,'selected_screen_checkpoints':1,'seed_panels':[s for _,s in PANELS],'ids':expected,'old_native_checkpoint_metrics_exact':True,'other_scenes_excluded':True,'full100_complete':False})
 save('ENVIRONMENT_SCOPE.json',environment)
 save('SOURCE_BINDINGS.json',{'root_path':ROOT.relative_to(R).as_posix(),'root_sha256':sha(ROOT),'old_native_seal_sha256':sha(OLD/'FILES_SHA256.json'),'old_native_table_preserved_not_overwritten':True,'original_statistic_source':EVIDENCE.relative_to(R).as_posix(),'original_statistic_file_sha256':sha(EVIDENCE),'original_statistic_function_source_sha256':hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),'original_statistic_AST_unmodified':True,'receipt_bindings':bindings,'F_read_volume':volume,'no_arrays_models_or_ZIP_read':True})
 with (H/'TABLES.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text))
 print(json.dumps({'status':'HYBRID_BENIGN10_THREE_VIEW_TABLE_BUILT_ROOT_TABLE_REVIEW_PENDING','checkpoints':10,'panels':9,'mean_SD_scalars':54,'display_cells':27,'native_raw_equal':environment['native_raw_exact_all_records'],'training_torch':environment['training_torch'],'replay_devices':environment['replay_devices']}))
if __name__=='__main__':main()
