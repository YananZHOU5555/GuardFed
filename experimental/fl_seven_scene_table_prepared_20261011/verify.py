"""Independent saved-count, fsum two-pass sample SD and rendered-cell verification."""
from pathlib import Path
import csv,hashlib,json,math
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'candidate';OLD_TABLE=R/'outputs/guardfed_tables/celeba_flgmm_six_scenes60_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def main():
 need(not (O/'VERIFICATION.json').exists(),'Fresh independent verification only')
 b=read(O/'SOURCE_BINDINGS.json');binding=read(H/'ROOT_BINDING.json')
 need(binding['root_adopted'] is True and sha(b['root71']['path'])==b['root71']['sha256']==binding['root71_sha256'],'Actual71 changed')
 for p,pin in b['sources'].items():need(sha(p)==pin['sha256'] and Path(p).stat().st_size==pin['bytes'],'Source identity changed')
 root=read(b['root71']['path']);old=read(R/'tmp/fl_three_view_after48_20261011/ROOT_SCIENTIFIC_ADOPTION.json')
 need(root['root_adoption'] and root['FLGMM_total_three_view_records']==71 and root['new_three_view_records_accepted']==10,'Actual71 required')
 accepted=old['records']+old['prior_interface_explicitly_reused']+root['new_records']
 data=read(O/'records71.json');rows=data['records'];need([r['root_adoption_record'] for r in rows]==accepted,'Accepted objects/order differ')
 need(len(rows)==len({r['id'] for r in rows})==71,'Accepted71 not unique')
 need(rows[:61]==read(OLD_TABLE/'records61.json')['records'],'Old61 objects/order changed')
 scene_list=[('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA'),('non-IID','Benign'),('non-IID','F Flip')]
 views=['raw','native','shared_calibration'];panels={'ten':list(range(91001,91011)),'nonselection_nine':list(range(91002,91011)),'matching_six':list(range(91005,91011))}
 partial=[r['id'] for r in rows if (r['distribution'],r['attack']) not in scene_list]
 need(partial==data['retained_partial_ids']==['FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed91001_screen'],'Partial boundary differs')
 need(sum(r['included_complete_scene'] for r in rows)==70,'Complete70 boundary differs')
 verified_metrics=0;verified_counts=0;max_count_delta=0.;environments={};raw_native=0
 for r in rows[61:]:
  receipt=read(r['receipt_path']);need(sha(r['receipt_path'])==r['receipt_sha256'] and receipt['checkpoint_sha256']==r['checkpoint_sha256'] and receipt['views']==r['views'] and receipt['fits']==r['fits'],'Saved accepted receipt differs')
  need(r['terminal_round']==70 and r['evaluation_split']=='valid' and r['valid_n']==19867 and r['same_checkpoint_all_views'],'Round/split mismatch')
  need(receipt['weights_before']==receipt['weights_after'] and receipt['valid_n']==19867,'Weight/count differs')
  need(r['views']['native']==r['views']['raw'],'Native/raw scientific fields differ');raw_native+=1
  for v in views:
   s=r['views'][v];counts=s['group_confusion_counts'];need(set(counts)=={'0','1'},'Protected groups differ')
   for group in counts.values():
    ints=[group[k] for k in ['tp','fp','tn','fn']];need(all(type(n) is int and n>=0 for n in ints),'Invalid base count')
    tp,fp,tn,fn=ints;verified_counts+=4
    need(group['n']==tp+fp+tn+fn and group['positives']==tp+fn and group['negatives']==tn+fp,'Confusion marginal differs')
    need(group['positives']>0 and group['negatives']>0,'Undefined subgroup metric')
    for key,value in [('tpr',tp/(tp+fn)),('fpr',fp/(fp+tn)),('positive_rate',(tp+fp)/(tp+fp+tn+fn))]:need(abs(value-group[key])<=1e-15,'Group rate/count mismatch')
   g0,g1=counts['0'],counts['1'];n=g0['n']+g1['n'];need(n==19867==s['prediction_count'],'Different evaluation denominator')
   recomputed={'accuracy':(g0['tp']+g0['tn']+g1['tp']+g1['tn'])/n,'aeod':abs(g0['tp']/g0['positives']-g1['tp']/g1['positives']),'aspd':abs((g0['tp']+g0['fp'])/g0['n']-(g1['tp']+g1['fp'])/g1['n'])}
   for key,value in recomputed.items():
    delta=abs(value-s[key]);max_count_delta=max(max_count_delta,delta);need(math.isfinite(s[key]) and 0<=s[key]<=1 and delta<=1e-15,'Receipt metric/count mismatch');verified_metrics+=1
   positives=g0['tp']+g0['fp']+g1['tp']+g1['fp'];need(s['constant_predictions']==(positives in (0,n)),'Constant prediction flag changed')
   fit=r['fits'][v];payload={k:x for k,x in fit.items() if k!='fit_sha256'};need(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==fit['fit_sha256'],'Saved fit identity changed')
  key=(r['training_torch'],r['original_training_device'],r['runtime']['torch'],r['runtime']['device']);environments[key]=environments.get(key,0)+1
 table=read(O/'tables.json');need(table['scenes']==[list(x) for x in scene_list] and table['views']==views and table['fixed_seed_panels']==panels,'Table scope changed')
 need([(p['view'],p['panel']) for p in table['panels']]==[(v,p) for v in views for p in panels],'Panel order differs')
 md=(O/'TABLES.md').read_text(encoding='utf8');expected_display=[];csv_expect=[];scalars=0;max_stats_delta=0.
 for pan in table['panels']:
  need(pan['seeds']==panels[pan['panel']] and pan['n']==len(pan['seeds']),'Fixed panel seed set differs')
  need([(c['distribution'],c['attack']) for c in pan['scenes']]==scene_list,'Scene labels/order differ')
  md_rows={k:[] for k in ['accuracy','aeod','aspd']}
  for cell in pan['scenes']:
   if (cell['distribution'],cell['attack'])!=('non-IID','F Flip'):
    for metric in ['accuracy','aeod','aspd']:
     actual=cell['metrics'][metric];display=actual['display'];md_rows[metric].append(display)
     csv_expect.append([pan['view'],pan['panel'],str(cell['n']),cell['distribution'],cell['attack'],metric,str(actual['mean']),str(actual['sample_sd']),display])
    continue
   chosen=sorted([r for r in rows if (r['distribution'],r['attack'])==(cell['distribution'],cell['attack']) and r['seed'] in pan['seeds']],key=lambda r:r['seed'])
   need([r['seed'] for r in chosen]==pan['seeds'] and [r['id'] for r in chosen]==cell['ids'],'Selected seed identity/order differs')
   n=len(chosen);need(n==pan['n']==cell['n'],'Cell n differs')
   for metric in ['accuracy','aeod','aspd']:
    vals=[r['views'][pan['view']][metric] for r in chosen];avg=math.fsum(vals)/n;sd=math.sqrt(math.fsum((x-avg)**2 for x in vals)/(n-1))
    actual=cell['metrics'][metric]
    for k,value in [('mean',avg),('sample_sd',sd)]:
     delta=abs(actual[k]-value);max_stats_delta=max(max_stats_delta,delta);need(delta<=1e-14,'Independent fsum/sample SD differs');scalars+=1
    scale,precision=(100,2) if metric=='accuracy' else (1,4);display=f'{avg*scale:.{precision}f} ± {sd*scale:.{precision}f}'
    need(display==actual['display'],'Display rounding differs');md_rows[metric].append(display)
    csv_expect.append([pan['view'],pan['panel'],str(n),cell['distribution'],cell['attack'],metric,str(actual['mean']),str(actual['sample_sd']),display])
  expected_display += ['| '+label+' | '+' | '.join(md_rows[k])+' |' for k,label in [('accuracy','ACC (%) ↑'),('aeod','AEOD ↓'),('aspd','ASPD ↓')]]
 actual_display=[line for line in md.splitlines() if line.startswith(('| ACC (%) ↑ |','| AEOD ↓ |','| ASPD ↓ |'))];need(actual_display==expected_display,'Markdown labels/value/order differs')
 with (O/'cells.csv').open(encoding='utf8',newline='') as f:csv_rows=list(csv.reader(f))
 need(csv_rows[1:]==csv_expect,'CSV value/order differs')
 need(verified_metrics==90 and verified_counts==240 and scalars==54 and len(csv_expect)==189,'Verification denominator changed')
 for newpan,oldpan in zip(table['panels'],read(OLD_TABLE/'tables.json')['panels']):need(dict(newpan,scenes=newpan['scenes'][:6])==oldpan,'Old324 statistics/162 cells/order changed')
 for name in ('records61.json','tables.json','TABLES.md','cells.csv','CAPTION.md'):need((O/('PREVIOUS60_'+name)).read_bytes()==(OLD_TABLE/name).read_bytes(),'Old table bytes differ')
 report={'status':'INDEPENDENT_RECEIPT_COUNTS_FSUM_SAMPLE_SD_DISPLAY_PASS','records':71,'complete_scene_records':70,'retained_partial_records':1,'scene_count':7,'panels':9,'mean_SD_pairs':189,'statistical_scalars':378,'new_scene_statistical_scalars_recomputed':54,'old324_statistics_preserved_not_recomputed':True,'metrics_recomputed_from_counts':90,'integer_base_counts_verified':240,'native_raw_equal_records':raw_native,'max_count_metric_absolute_difference':max_count_delta,'max_fsum_statistics_absolute_difference':max_stats_delta,'Markdown_cells_verified':189,'CSV_cells_verified':189,'partial_record_excluded_from_all_panels':True,'all_adopted_records_retained':True,'source_identity_pins_verified':len(b['sources']),'environment_scope':'new10 only; prior61 retained in source table','environment':[dict(zip(['training_torch','training_device','inference_torch','inference_device','n'],(*k,v))) for k,v in environments.items()],'arrays_read':False,'new_inference':0,'new_fit':0,'new_training':0,'test':False,'root_table_adopted':False,'verifier_sha256':sha(__file__),'outputs_sha256':{name:sha(O/name) for name in ['records71.json','tables.json','TABLES.md','cells.csv','SOURCE_BINDINGS.json']}}
 with (O/'VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':main()
