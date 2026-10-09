"""Independently reconstruct counts/statistics and promote the closed nine-scene table."""
from pathlib import Path
import datetime, hashlib, json, math, shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_three_view92_tables_20261009/final_builder_v2'
DEST=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim92_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(BASE/'FILES_SHA256.json')=='4ae84bc75ac08902ea6787bd525270950945f6e5166887010b41b5d40df61c7d'
seal=read(BASE/'FILES_SHA256.json')['files']
assert len(seal)==12 and not DEST.exists()
for name,row in seal.items():
 p=BASE/name;assert p.resolve().is_relative_to(BASE.resolve()) and sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
for name,row in read(BASE/'INPUTS.json')['files'].items():
 p=ROOT/name;assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
adopt=ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009/execution_candidate/backups/incremental_20261009T174922Z/ROOT_ADOPTION_REVIEW.json'
assert sha(adopt)=='b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683'
assert read(adopt)['cumulative_three_view_models']==92
records=read(BASE/'snapshot92/records.json')['records'];tables=read(BASE/'snapshot92/tables.json')
assert len(records)==184 and len({r['id'] for r in records})==184
assert (tables['complete_scenes'],tables['paired_checkpoints'],tables['table_model_records'],tables['incomplete_pairs'])==(9,92,180,2)
assert not tables['primary_endpoint_selected'] and not tables['final_test'] and tables['new_inference']==tables['new_training']==0
bycell={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
assert len(bycell)==184
baseline=read(ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json')['records']
full={(r['distribution'],r['attack'],r['seed']):r for r in baseline if r['method']=='GuardFed-AD2+'}
inventory=read(ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009/inventory_actual92_Full100refs.json')
original={r['id']:r for r in inventory['records']}
assert {r['id'] for r in records if r['variant']=='minus_U'}==set(original)
metric_count=0
for r in records:
 assert r['views']['native']==r['views']['shared_calibration']
 if r['variant']=='Full':
  old=full[(r['distribution'],r['attack'],r['seed'])]
  assert r['checkpoint_sha256']==old['checkpoint_sha256'] and r['config_sha256']==old['config_canonical_sha256']
  assert r['views']==old['views'] and r['fits']==old['fits']
 else:
  old=original[r['id']]
  assert r['checkpoint_sha256']==old['checkpoint']['sha256'] and r['config_sha256']==old['config_canonical_sha256']
  for k in ('accuracy','aeod','aspd'):assert abs(r['views']['native'][k]-old['prior_validation_metrics'][k])<=1e-12
 for view in r['views'].values():
  groups=view['group_confusion_counts'];a,b=groups['0'],groups['1'];n=a['n']+b['n']
  assert n==view['prediction_count']==19867
  metrics=dict(accuracy=(a['tp']+a['tn']+b['tp']+b['tn'])/n,
   aeod=abs(a['tp']/(a['tp']+a['fn'])-b['tp']/(b['tp']+b['fn'])),
   aspd=abs((a['tp']+a['fp'])/a['n']-(b['tp']+b['fp'])/b['n']))
  for k,v in metrics.items():assert abs(view[k]-v)<=1e-12;metric_count+=1
panels=tables['panels'];assert len(panels)==9
scalar_count=0;maximum=0.;complete_scenes=set()
for panel in panels:
 seeds=panel['seeds'];assert seeds in [list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
 assert len(panel['rows'])==27
 for row in panel['rows']:
  d,a,v=row['distribution'],row['attack'],row['variant'];complete_scenes.add((d,a))
  assert (d,a)!=('non-IID','Sp-DFA') and row['seeds']==seeds and row['n']==len(seeds)
  for key,metric,scale in [('accuracy_pct','accuracy',100),('aeod','aeod',1),('aspd','aspd',1)]:
   values=[]
   for seed in seeds:
    if v=='minus_U minus Full':
     x=bycell[('minus_U',d,a,seed)]['views'][panel['view']][metric]-bycell[('Full',d,a,seed)]['views'][panel['view']][metric]
    else:x=bycell[(v,d,a,seed)]['views'][panel['view']][metric]
    values.append(x*scale)
   mean=math.fsum(values)/len(values)
   sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
   for expected,actual in [(mean,row[key]['mean']),(sd,row[key]['sample_sd_ddof1'])]:
    difference=abs(expected-actual);assert difference<=1e-12
    maximum=max(maximum,difference);scalar_count+=1
assert scalar_count==1458 and len(complete_scenes)==9 and metric_count==1656
old_tables=read(ROOT/'tmp/celeba_mechanism_three_view_paired71_20261009/snapshot_724_full100_mechanism71/tables.json')
old_rows=0
for before,after in zip(old_tables['panels'],panels):
 assert before['view']==after['view'] and before['seeds']==after['seeds']
 lookup={(r['distribution'],r['attack'],r['variant']):r for r in after['rows']}
 for r in before['rows']:assert lookup[(r['distribution'],r['attack'],r['variant'])]==r;old_rows+=1
assert old_rows==189
verification=read(BASE/'snapshot92/verification.json')
assert verification['display_cells_verified']==729 and verification['native92_rows_matched']==81
for name,row in seal.items():
 target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target);assert sha(target)==row['sha256']
shutil.copyfile(BASE/'FILES_SHA256.json',DEST/'FILES_SHA256.json')
proof=dict(status='ROOT_THREE_VIEW92_NINE_SCENE_COUNTS_PAIRED_STATISTICS_AND_PRIOR_ROWS_PASS',
 checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(BASE/'FILES_SHA256.json'),
 source_entry=BASE.relative_to(ROOT).as_posix(),accepted_three_view92=92,paired_checkpoints=92,preserved_records=184,
 complete_scenes=9,complete_paired_checkpoints=90,incomplete_paired_checkpoints=2,display_cells_verified=729,
 independently_reconstructed_metrics=metric_count,independent_mean_sd_scalars=scalar_count,max_abs_statistic_difference=maximum,
 old_seven_scene_statistic_rows_exact=old_rows,native92_display_rows_matched=81,
 original_Full900_view_fit_checkpoint_config_exact=True,original_native92_control_identity_exact=True,
 native_shared_identical_records=184,new10_root_adoption_sha256=sha(adopt),
 tables_sha256=sha(DEST/'snapshot92/tables.json'),records_sha256=sha(DEST/'snapshot92/records.json'),
 verification_sha256=sha(DEST/'snapshot92/verification.json'),source_helper_sha256=sha(Path(__file__)),
 new_inference=0,new_training=0,primary_endpoint_selected=False,test=False,negative_results_preserved=True)
with (DEST/'ROOT_REVIEW.json').open('x',encoding='utf8',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],root_proof_sha256=sha(DEST/'ROOT_REVIEW.json'),scalars=scalar_count,
 reconstructed_metrics=metric_count,canonical=str(DEST))))
