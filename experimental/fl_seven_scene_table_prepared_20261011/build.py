"""Build descriptive FLGMM tables from 71 root-adopted compact saved receipts only."""
from pathlib import Path
import argparse
from input_contract import validate_binding, validate_new_rows
from collections import defaultdict,Counter
from statistics import mean,stdev
import ast,copy,csv,datetime,hashlib,json,subprocess
H=Path(__file__).resolve().parent;R=H.parents[1]
O=H/'candidate'
OLD_TABLE=R/'outputs/guardfed_tables/celeba_flgmm_six_scenes60_20261011'
ROOT61=R/'tmp/fl_three_view_after48_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
ROOT61_SHA='d6c7bdadb05ffcf8786221ed15a16125cd0fe84d1745fc15f9b7e6cc0a2f68d6'
ROOT48=R/'tmp/celeba_flgmm_closed47_root_execution_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
ROOT48_SHA='22fc113add73814ae40accf563ce5f63bd63d50bcf53b7262bdb2b996b016dfe'
VIEWS=['raw','native','shared_calibration'];METRICS=[('accuracy','ACC (%) ↑',100,2),('aeod','AEOD ↓',1,4),('aspd','ASPD ↓',1,4)]
SCENES=[('IID',a) for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']]+[('non-IID','Benign'),('non-IID','F Flip')]
PARTIAL='FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed91001_screen'
pins={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def read(p,expected=None):
 p=Path(p);b=p.read_bytes();h=hashlib.sha256(b).hexdigest();need(expected is None or h==expected,'Changed source: '+str(p))
 pins[p.resolve().as_posix()]={'sha256':h,'bytes':len(b)};return json.loads(b)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(name,j):
 with (O/name).open('x',encoding='utf8',newline='\n') as f:json.dump(j,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def main():
 need(not O.exists(),'Fresh output directory only; preserve existing attempt')
 volume=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True));need(volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy','Required F evidence volume unavailable')
 ap=argparse.ArgumentParser();ap.add_argument('--binding-sha256',required=True);args=ap.parse_args()
 need(sha(H/'ROOT_BINDING.json')==args.binding_sha256,'Actual adopted71 binding differs')
 binding=read(H/'ROOT_BINDING.json');validate_binding(binding)
 root=read(R/binding['root71'],binding['root71_sha256']);old=read(ROOT61,ROOT61_SHA)
 need(root['root_adoption'] is True and root['FLGMM_total_three_view_records']==71 and root['new_three_view_records_accepted']==10,'Actual71 adoption required')
 need(root['Linux_whole_original_saved_check_pass'] and root['Windows_saved_outputs_audit_pass'] and root['Windows_saved_outputs_audit_fit_calls']==0 and root['Windows_whole_saved_check_pass'] is False,'Complementary evidence incomplete')
 need(root['records'][:60]==old['records'] and root['prior_interface_explicitly_reused']==old['prior_interface_explicitly_reused'],'Prior61 root objects/order differ')
 expected=['FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed'+str(seed)+'_fullcoverage' for seed in range(91001,91011)]
 validate_new_rows(root['new_records']);need([r['id'] for r in root['new_records']]==expected and root['records']==old['records']+root['new_records'],'Exact new10 root list required')
 oldroot=read(OLD_TABLE/'ROOT_VERIFICATION.json','a7ed22f1f6366135b361433581ad59a76aa52271ba0516cd0bb11efc18e9eef0')
 for name in ('records61.json','tables.json','TABLES.md','cells.csv','CAPTION.md'):need(sha(OLD_TABLE/name)==oldroot['adopted_outputs'][name],'Prior adopted table bytes changed')
 olddata=read(OLD_TABLE/'records61.json','aecacb19c1b984a44c0f0f805e9921f52b2cde0b5fda063a7d4183c2d0c67b6b')
 oldtables=read(OLD_TABLE/'tables.json','c20b22f8d42b2672aee2ff8706efeb18c2ed2ee1a5c9581b62bf9f4fc04dcaf1')
 need([r['root_adoption_record'] for r in olddata['records']]==old['records']+old['prior_interface_explicitly_reused'],'Accepted old table root/order differs')
 adopted=old['records']+old['prior_interface_explicitly_reused']+root['new_records']
 need(len(adopted)==len({x['id'] for x in adopted})==71,'Unique accepted71 required')
 source_groups=[(R/binding['candidate'],binding['candidate_seal_sha256'],R/binding['transport'],binding['transport_sha256'],expected)]
 accepted={x['id']:x for x in adopted};records={r['id']:copy.deepcopy(r) for r in olddata['records']}
 for base,seal_sha,transport_path,transport_sha,ids in source_groups:
  seal=read(base/'FILES_SHA256.json',seal_sha);m=read(base/'MANIFEST.json',seal['files']['MANIFEST.json']['sha256']);byid={x['id']:x for x in m['records']}
  transport=read(transport_path,transport_sha);fbase=Path(transport['verified_extract']).resolve();need(fbase.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()),'Non-F receipt storage refused')
  for rid in ids:
   a=accepted[rid];row=byid[rid];rel='bundle/'+rid+'/receipt.json';receipt_path=fbase/rel;receipt_pin=transport['members'][rel]
   need(receipt_pin['sha256']==a['receipt_sha256'],'Root/transport receipt mismatch');r=read(receipt_path,a['receipt_sha256']);need(receipt_path.stat().st_size==receipt_pin['bytes'],'Receipt size differs')
   need(rid not in records and (r['method'],r['id'],r['distribution'],r['attack'],r['seed'])==('FLGMM',rid,a['distribution'],a['attack'],a['seed']),'Duplicate or identity mismatch')
   need(r['checkpoint_sha256']==a['checkpoint_sha256']==row['identity']['checkpoint']['sha256'],'Checkpoint mismatch')
   need(r['prediction_arrays_sha256']==a['array_sha256']==transport['members']['bundle/'+rid+'/validation_predictions.npz']['sha256'],'Accepted array pin mismatch')
   identity=copy.deepcopy(row['identity']);identity.update(distribution=row['distribution'],attack=row['attack'],seed=row['seed'],result=identity['original_artifact_pins']['result'],raw_job=identity['original_artifact_pins']['job'],config_canonical_sha256=canonical(identity['config']),training_torch=row['original_training_torch'])
   need(canonical(identity)==r['external_identity_record_sha256'],'Original source identity mismatch')
   cfg=identity['config'];need(cfg['rounds']==row['terminal_round']==70 and cfg['celeba_evaluation_split']==row['split']=='valid' and row['n_eval']==r['valid_n']==19867,'Nonterminal/subset/test refused')
   need(cfg['celeba_train_limit']==cfg['celeba_eval_limit']==0 and cfg['learning_rate']==.001,'Fixed full-data recipe differs')
   need(cfg['client_alpha']=={'IID':5000.,'non-IID':5.}[row['distribution']],'Distribution alpha differs')
   need(r['original_result_sha256']==identity['result']['sha256'] and r['config_canonical_sha256']==canonical(cfg),'Original result/config changed')
   need(r['weights_before']==r['weights_after'] and r['valid_image_ids_sha256']=='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf','Weight/sample identity differs')
   need(all(r[k] is False for k in ['optimizer_created','gradients_created','test_labels_accessed','test_inference_performed']),'Forbidden operation in receipt')
   need(set(r['views'])==set(r['fits'])==set(VIEWS),'Missing views');need(r['native_comparison']['accepted'] and r['native_comparison']['max_abs_difference']==0,'Native strict result differs')
   need(r['views']['native']==r['views']['raw'],'FLGMM native/raw must remain identical')
   for view,fit in r['fits'].items():need(canonical({k:v for k,v in fit.items() if k!='fit_sha256'})==fit['fit_sha256'],'Fit payload pin differs')
   records[rid]={'id':rid,'method':'FLGMM','distribution':r['distribution'],'attack':r['attack'],'seed':r['seed'],'checkpoint_sha256':r['checkpoint_sha256'],'receipt_path':receipt_path.as_posix(),'receipt_sha256':a['receipt_sha256'],'array_sha256':a['array_sha256'],'source_manifest_path':(base/'MANIFEST.json').resolve().as_posix(),'source_manifest_sha256':sha(base/'MANIFEST.json'),'external_identity_record_sha256':r['external_identity_record_sha256'],'config_canonical_sha256':r['config_canonical_sha256'],'terminal_round':70,'evaluation_split':'valid','valid_n':19867,'valid_image_ids_sha256':r['valid_image_ids_sha256'],'training_torch':r['original_training_torch'],'original_training_device':row['original_training_device'],'runtime':r['runtime'],'views':r['views'],'fits':r['fits'],'native_comparison':r['native_comparison'],'root_adoption_record':a,'included_complete_scene':(r['distribution'],r['attack']) in SCENES,'same_checkpoint_all_views':True}
 ordered=[records[x['id']] for x in adopted];complete=[r for r in ordered if r['included_complete_scene']];partial=[r for r in ordered if not r['included_complete_scene']]
 need(len(complete)==70 and [r['id'] for r in partial]==[PARTIAL],'Exactly seven full scenes plus retained nonIID S-DFA seed1')
 for d,a in SCENES:need(sorted(r['seed'] for r in complete if (r['distribution'],r['attack'])==(d,a))==list(range(91001,91011)),'Incomplete/duplicate scene seeds')
 seed_source=R/'tmp/celeba_nine_method_three_view_tables_20261009/build_three_view_tables.py';need(sha(seed_source)=='589394fbf1264d172257647b9ff2ce5b81dcda6e14aead637dd79fad2ea5b83f','Original fixed panel source differs')
 seed_node=next(n for n in ast.parse(seed_source.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SEEDS' for t in n.targets));ns={};exec(compile(ast.Module(body=[seed_node],type_ignores=[]),'<original fixed SEEDS>','exec'),ns);seeds=ns['SEEDS']
 renderer=R/'outputs/guardfed_tables/celeba_nine_method_final_20261004/build_tables.py';need(sha(renderer)=='4ae22c85c3fcc1f8e01ac9b3fc4dbf0ce45ea676f39a677e337782cb0d2b3529','Original renderer differs')
 source=renderer.read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='values_for');fn_text=ast.get_source_segment(source,fn)
 summaries=[]
 for view in VIEWS:
  groups=defaultdict(list)
  for r in complete:
   if r['id'] not in expected:continue
   groups['FLGMM',r['distribution'],r['attack']].append(dict(r,**{k:r['views'][view][k] for k,_,_,_ in METRICS}))
  fn_ns={'groups':groups,'mean':mean,'stdev':stdev,'metrics':METRICS};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<unchanged original values_for>','exec'),fn_ns)
  for panel,ss in seeds.items():
   cells=[]
   for d,a in SCENES[-1:]:
    rows=sorted([r for r in groups['FLGMM',d,a] if r['seed'] in ss],key=lambda r:r['seed']);need([r['seed'] for r in rows]==ss,'Panel seed membership differs')
    display=fn_ns['values_for']('FLGMM',d,a,ss,True)
    cells.append({'distribution':d,'attack':a,'n':len(ss),'ids':[r['id'] for r in rows],'metrics':{k:{'mean':mean(r[k] for r in rows),'sample_sd':stdev(r[k] for r in rows),'display':display[i]} for i,(k,_,_,_) in enumerate(METRICS)}})
   prior=next(p for p in oldtables['panels'] if p['view']==view and p['panel']==panel)
   need(prior['seeds']==ss and prior['n']==len(ss),'Old fixed panel changed')
   summaries.append(dict(copy.deepcopy(prior),scenes=copy.deepcopy(prior['scenes'])+cells))
 protocol=R/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/attempt_20261009T200518912319Z/verified_manual_v2/stage/source/protocol.json';p=read(protocol,'1006bcc7c614e2111490f0c57949107d23a52cc5301c849578768273dd6964f4')
 need(p['selected_recipe']['id']=='FLGMM_Tg20_L2.0_lr0.001','Recipe source differs')
 O.mkdir(parents=True)
 for name in ('records61.json','tables.json','TABLES.md','cells.csv','CAPTION.md'):
  (O/('PREVIOUS60_'+name)).write_bytes((OLD_TABLE/name).read_bytes())
 need(ordered[:61]==olddata['records'],'Old61 normalized objects/order changed')
 for newpan,oldpan in zip(summaries,oldtables['panels']):need(dict(newpan,scenes=newpan['scenes'][:6])==oldpan,'Old six scene statistics/order changed')
 write('records71.json',{'root71_sha256':binding['root71_sha256'],'prior61_sha256':ROOT61_SHA,'old61_order_preserved_then_exact10_appended':True,'records':ordered,'table_records':70,'retained_partial_ids':[PARTIAL]})
 write('tables.json',{'method':'FLGMM','scenes':[list(x) for x in SCENES],'views':VIEWS,'fixed_seed_panels':seeds,'panels':summaries,'statistics':'arithmetic mean and sample SD (ddof=1)','final_test':False})
 notes=['Terminal round70; validation only,19,867 images; IID/non-IID use frozen Dirichlet alpha5000/5; all three metrics and views use the same checkpoint per ID. Mean ± sample SD (ddof=1); ACC in percent, AEOD/ASPD on [0,1]. AEOD is the absolute TPR gap, not full equalized odds.',
 'Panels are fixed:10 seeds91001–91010;9 seeds91002–91010;6 seeds91005–91010. No per-cell best-seed selection. The9-seed panel excludes selection seed91001; it does not undo repeated validation exposure. The6-seed panel preserves the earlier common-seed comparison subset; all71 FLGMM records themselves use the same declared cu128 version.',
 'Selected recipe: warmup20, control width2.0, local learning rate0.001. Recipe selection used seed91001 validation scores across IID/non-IID Benign/S-DFA; three of the four reused screen checkpoints enter these complete tables. The fourth, non-IID S-DFA seed91001, is retained in records71.json but excluded from scene statistics.',
 'FLGMM follows the frozen author-code adaptation, including its largest-cluster choice, upstream bounds behavior and declared zero-standard-deviation extension. Native and raw are identical uncalibrated margin>0 outputs, not independent replications. Shared calibration applies the existing common clean train-root group thresholds (margin≥threshold), not FLGMM-native calibration.',
 'All71 accepted records were trained on CUDA with torch2.11.0+cu128 and evaluated on CPU with torch2.11.0+cu128; this is not a CPU/GPU training-equivalence claim. Windows saved-output audit runtime is separately retained in the root proofs.',
 'Original Linux whole checks supply root-refit evidence. The original Windows47 exact-refit/whole failure remains preserved; The adopted13 and this new10 Windows saved-output checks use zero refits; Linux original whole checks provide cached root-refit evidence. The prior61 group-KL audit micro-differences and original Windows47 failure are retained; new10 diagnostics must remain explicit in the adopted71 proof. Saved-output agreement is not whole Windows root-reconstruction equality. No universal bitwise recalibration claim.',
 'This is seven complete FLGMM scenes (70 records) plus one retained partial record, not FLGMM100, a17-method comparison or final test. Values are descriptive; all outcomes are retained, without a superiority or significance claim.']
 # Environment and preserved audit counts below are measured, not guessed.
 env={'all71_training_torch':dict(Counter(r['training_torch'] for r in ordered)),'all71_training_device':dict(Counter(r['original_training_device'] for r in ordered)),'all71_inference_torch':dict(Counter(r['runtime']['torch'] for r in ordered)),'all71_inference_device':dict(Counter(r['runtime']['device'] for r in ordered))}
 need(env['all71_training_torch']==env['all71_inference_torch']=={'2.11.0+cu128':71} and env['all71_inference_device']=={'cpu':71},'Environment note differs from actual records')
 blocks=['# FLGMM: seven complete CelebA validation scenes','']
 for pan in summaries:
  blocks+=['## '+pan['view']+' · '+str(pan['n'])+' fixed seeds','', '| Metric | '+' | '.join(d+': '+a.replace('F Flip','F-Flip') for d,a in SCENES)+' |','| --- | '+' | '.join(['---']*7)+' |']
  for metric,label,_,_ in METRICS:blocks.append('| '+label+' | '+' | '.join(cell['metrics'][metric]['display'] for cell in pan['scenes'])+' |')
  blocks.append('')
 blocks+=['## Scope and interpretation','']+['- '+s for s in notes]
 (O/'TABLES.md').write_text('\n'.join(blocks)+'\n',encoding='utf8',newline='\n')
 with (O/'cells.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['view','panel','n','distribution','attack','metric','mean','sample_sd','display'])
  for pan in summaries:
   for cell in pan['scenes']:
    for k,_,_,_ in METRICS:w.writerow([pan['view'],pan['panel'],pan['n'],cell['distribution'],cell['attack'],k,cell['metrics'][k]['mean'],cell['metrics'][k]['sample_sd'],cell['metrics'][k]['display']])
 write('SOURCE_BINDINGS.json',{'root71':{'path':str(R/binding['root71']),'sha256':binding['root71_sha256']},'prior61_sha256':ROOT61_SHA,'prior61_table_records_sha256':'aecacb19c1b984a44c0f0f805e9921f52b2cde0b5fda063a7d4183c2d0c67b6b','prior_six_scene_tables_sha256':'c20b22f8d42b2672aee2ff8706efeb18c2ed2ee1a5c9581b62bf9f4fc04dcaf1','old61_objects_order_unchanged':True,'old324_statistics_162_cells_unchanged':True,'new_scene_statistical_scalars':54,'new_scene_display_cells':27,'old_scene_statistics_recomputed':False,'sources':pins,'renderer':{'path':renderer.as_posix(),'sha256':sha(renderer),'values_for_source_sha256':hashlib.sha256(fn_text.encode()).hexdigest()},'panel_source':{'path':seed_source.as_posix(),'sha256':sha(seed_source)},'environment':env,'selection_rule':p['selection_rule'],'selected_recipe':p['selected_recipe'],'limitations':notes,'partial_retained_excluded':PARTIAL,'arrays_read':False,'new_fit':0,'new_CNN':0,'test':False,'builder_sha256':sha(__file__),'input_contract_sha256':sha(H/'input_contract.py'),'F_volume':volume})
 print(json.dumps({'status':'FLGMM_SEVEN_SCENE_TABLE_BUILT_NOT_ROOT_TABLE_ADOPTED','records':71,'complete_records':70,'retained_partial':1,'mean_SD_pairs':189,'numeric_scalars':378,'panels':9,'new_fit':0,'new_CNN':0}))
if __name__=='__main__':main()
