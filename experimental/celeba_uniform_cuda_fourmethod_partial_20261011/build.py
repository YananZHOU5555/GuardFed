"""Root-authorized partial four-method CUDA-native tables from adopted compact JSON only."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,sys
from statistics_original import METHODS,DISTS,ATTACKS,METRICS,SEEDS,summaries,markdown,NOTES
H=Path(__file__).resolve().parent;R=H.parents[1]
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def local(pin):
 need(isinstance(pin,dict) and isinstance(pin.get('path'),str) and isinstance(pin.get('sha256'),str) and len(pin['sha256'])==64,'Unbound compact evidence')
 p=Path(pin['path']);p=p if p.is_absolute() else R/p;p=p.resolve()
 need(p.is_relative_to(R.resolve()) and p.drive.upper()!='F:' and p.suffix=='.json','Only workspace compact JSON inputs')
 need(sha(p)==pin['sha256'],'Changed compact input: '+str(p));return p
def pinned(pin):return read(local(pin))
def verified_inputs(binding):
 """Identity-only gate: no mean/SD, arrays, fit, inference or checkpoint load."""
 pins=read(H/'INPUT_PINS.json');need(binding['roots']==pins['roots'],'Changed four adopted roots')
 prior=pinned(pins['metadata_root']);need(prior['status']=='ROOT_ELEVEN_METHOD_NATIVE1100_DESCRIPTIVE_VALIDATION_TABLE_ADOPTED' and prior['records']==1100,'Wrong metadata-only parent')
 prior_rows=pinned(pins['metadata'])['records'];need(len(prior_rows)==1100,'Original metadata incomplete')
 selected={r['id']:(j,r) for j,r in enumerate(prior_rows) if r['method'] in METHODS}
 need(len(selected)==400,'Original four-method metadata incomplete/duplicate')
 expected={(m,d,a,s) for m in METHODS for d in DISTS for a in ATTACKS for s in SEEDS['ten']}
 need({(r['method'],r['distribution'],r['attack'],r['seed']) for _,r in selected.values()}==expected,'Missing metadata cells')
 documents={};audits={}
 for method in METHODS:
  doc=pinned(binding['roots'][method]);documents[method]=doc
  need(doc['status']==pins['root_statuses'][method] and doc['accepted']==100 and doc['test'] is False and doc['new_training'] is False and doc['parameters_changed'] is False and doc['tolerance_changed'] is False,'Non-adopted/mixed root')
  ids=[i for i,(_,r) in selected.items() if r['method']==method]
  need(len(doc['ids'])==len(set(doc['ids']))==len(doc['records'])==100 and set(doc['ids'])==set(ids)==set(doc['records']),'Missing/duplicate CUDA100')
  ap=doc['offserver_audit'] if method=='LoGoFair-DP' else doc['audit'];audit=pinned(ap);transport=pinned(doc['transport']);audits[method]=(ap,audit)
  need(transport['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and transport['all_members_verified'] is True,'Root has no verified transport')
  need(transport['archive_sha256']==doc['archive']['sha256'],'Root/transport archive differs')
  au={v['post_id'] if method=='LoGoFair-DP' else v['id']:v for v in audit['records']}
  need(len(audit['records'])==len(au)==100 and set(au)==set(ids),'Audit100 mismatch')
  for i in ids:
   _,old=selected[i];current=doc['records'][i];a=au[i]
   need(current['checkpoint_sha256']==old['checkpoint_sha256'],'Old metadata/new CUDA checkpoint differs')
   if method=='LoGoFair-DP':
    need(current['metrics']==a['metrics'] and current['scores_sha256']==a['scores_sha256'] and current['task_acceptance']==a['Linux_whole_acceptance'],'LoGo root/audit identity differs')
    need(current['frozen_post_state_sha256']==a['frozen_post_state_sha256'],'Different frozen LoGo state')
   else:
    need(current['views']==a['views'] and current['receipt_sha256']==a['receipt_sha256'] and current['arrays_sha256']==a['arrays_sha256'] and current['task_acceptance_sha256']==a['accepted_sha256'] and current['checkpoint_sha256']==a['checkpoint_sha256'],'Root/saved audit identity differs')
    if method=='GuardFed-AD2+':
     need((current['distribution'],current['attack'],current['seed'])==(old['distribution'],old['attack'],old['seed']),'Full root/cell metadata differs')
     need(current['native_comparison']['accepted'] and current['native_comparison']['max_abs_difference']<=1e-12,'Full native gate missing')
 return pins,selected,documents,audits
def collect_rows(binding):
 pins,metadata,docs,audits=verified_inputs(binding);rows=[]
 for method in METHODS:
  doc=docs[method]
  for identity in doc['ids']:
   j,old=metadata[identity];current=doc['records'][identity]
   metrics=current['metrics'] if method=='LoGoFair-DP' else current['views']['native']
   need(metrics['prediction_count']==19867 and all(isinstance(metrics[k],(int,float)) and not isinstance(metrics[k],bool) and math.isfinite(metrics[k]) and 0<=metrics[k]<=1 for k,_,_,_ in METRICS),'Nonfinite or invalid native metric')
   counts=current['group_confusion_counts'] if method=='LoGoFair-DP' else metrics['group_confusion_counts']
   positives=sum(counts[g]['tp']+counts[g]['fp'] for g in ['0','1']);constant=0 if positives==0 else (1 if positives==19867 else None)
   need(sum(counts[g][k] for g in ['0','1'] for k in ['tp','fp','tn','fn'])==19867,'Wrong adopted prediction count')
   if method=='LoGoFair-DP':need(current['constant_prediction']==constant,'LoGo constant flag differs')
   else:need(metrics['constant_predictions']==(constant is not None),'Native constant flag differs')
   rows.append(dict(id=identity,method=method,distribution=old['distribution'],attack=old['attack'],seed=old['seed'],checkpoint_sha256=current['checkpoint_sha256'],metrics={k:metrics[k] for k,_,_,_ in METRICS},positive_rate=metrics['positive_rate'],constant_prediction=constant,view='native',root_source=binding['roots'][method],root_record_pointer='/records/'+identity.replace('~','~0').replace('/','~1'),metadata_source=dict(**pins['metadata'],pointer='/records/'+str(j)),old_metrics_used=False))
 need(len(rows)==400,'Wrong four-method count');return rows
def save(p,value):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def write_csv(p,fields,rows):
 with p.open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True);parser.add_argument('--source-seal-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 need(__debug__ and not os.environ.get('PYTHONOPTIMIZE'),'Optimized Python refused')
 need(sha(H/'FILES_SHA256.json')==args.source_seal_sha256,'Wrong package seal')
 for n,p in read(H/'FILES_SHA256.json')['files'].items():need(sha(H/n)==p['sha256'],'Source changed')
 need(sha(args.inputs)==args.inputs_sha256,'Changed actual table inputs');binding=read(args.inputs)
 need(binding['status']=='ROOT_AUTHORIZED_PARTIAL_FOUR_METHOD_UNIFORM_CUDA_NATIVE_TABLE' and binding['root_execution_authorized'] is True and binding['view']=='native' and binding['final_test'] is False and binding['new_fit']==binding['new_inference']==0,'Statistics require separate explicit root authority')
 review=pinned(binding['root_source_review']);need(review['source_adoptable'] is True and review['source_seal_sha256']==args.source_seal_sha256,'Unreviewed table source')
 output=args.output.resolve();need(output.is_relative_to(H.resolve()) and not output.exists(),'New owned output directory required')
 rows=collect_rows(binding);panels,aggregates=summaries(rows)
 constants=[]
 for panel,seeds in SEEDS.items():
  for method in METHODS:
   for dist in DISTS:
    for attack in ATTACKS:
     items=[r for r in rows if (r['method'],r['distribution'],r['attack'])==(method,dist,attack) and r['seed'] in seeds]
     cs=[r for r in items if r['constant_prediction'] is not None]
     constants.append(dict(panel=panel,method=method,distribution=dist,attack=attack,n=len(items),constant_count=len(cs),constant_ids=[r['id'] for r in cs]))
 text=markdown(panels,aggregates)+'\n## Constant-prediction outcomes retained\n\n| Panel | Method | Distribution | Scenario | Constant / n |\n|---|---|---|---|---:|\n'
 for c in constants:text+='| '+ ' | '.join([c['panel'],c['method'],c['distribution'],c['attack'],str(c['constant_count'])+'/'+str(c['n'])])+' |\n'
 output.mkdir();save(output/'records_native400.json',dict(status='PARTIAL_FOUR_METHOD_UNIFORMLY_CUDA_VALIDATION_NATIVE_ONLY',records=rows));save(output/'summary_statistics.json',panels);save(output/'seed_first_aggregates.json',aggregates);save(output/'constant_predictions.json',constants)
 with (output/'TABLES.md').open('x',encoding='utf8',newline='\n') as f:f.write(text)
 long=[]
 for panel,items in panels.items():
  for r in items:
   for k,_,_,_ in METRICS:long.append(dict(panel=panel,method=r['method'],distribution=r['distribution'],attack=r['attack'],n=r['n'],metric=k,mean=r[k]['mean'],sample_sd=r[k]['sample_sd'],display=r[k]['display']))
 write_csv(output/'summary_statistics.csv',['panel','method','distribution','attack','n','metric','mean','sample_sd','display'],long)
 ag=[]
 for panel,items in aggregates.items():
  for r in items:
   for k,_,_,_ in METRICS:ag.append(dict(panel=panel,method=r['method'],scope=r['scope'],n=r['n'],metric=k,mean=r[k]['mean'],sample_sd=r[k]['sample_sd']))
 write_csv(output/'seed_first_aggregates.csv',['panel','method','scope','n','metric','mean','sample_sd'],ag)
 save(output/'BUILD_RECEIPT.json',dict(status='PARTIAL_FOUR_METHOD_UNIFORMLY_CUDA_VALIDATION_NOT_FINAL_COMPARISON',inputs_sha256=args.inputs_sha256,source_seal_sha256=args.source_seal_sha256,records=400,methods=METHODS,panels=SEEDS,view='native',ddof=1,cross_scene_rule='mean within seed first, then across-seed mean/sample SD',new_fit=0,new_inference=0,test=False,score=False,best_seed=False,new_primary_endpoint=False,old_tables_modified=False,files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()}))
 print(json.dumps(dict(status='PARTIAL_FOUR_METHOD_TABLE_CREATED',records=400,output=str(output),receipt_sha256=sha(output/'BUILD_RECEIPT.json'))))
if __name__=='__main__':main()
