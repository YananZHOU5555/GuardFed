"""Root arithmetic/provenance review and canonical delivery of the actual30 paired C models."""
from pathlib import Path
import collections,datetime,hashlib,json,math,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_three_view_C_three_scenes_prepared_20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
seal=BASE/'ACTUAL_FILES_SHA256.json';assert sha(seal)=='46c405262070155a55dbeefd569d13af11d301113217f4b0f77f2437a981fef2'
files=read(seal)['files'];assert len(files)==29
for n,pin in files.items():assert sha(BASE/n)==pin['sha256'] and (BASE/n).stat().st_size==pin['bytes']
assert sha(BASE/'ACTUAL_HANDOFF.json')=='86b1a397e4c96adae15cf668b43b6ecbb3755bc669ab61bc669269889ecbf94f'
handoff=read(BASE/'ACTUAL_HANDOFF.json')
independent_path=ROOT/'tmp/celeba_mechanism_C30_root_arithmetic_review_20261009/ROOT_ARITHMETIC_REVIEW.json'
assert sha(independent_path)=='4c0816364469b16cdc1f76b6852633d5b4c9157af8161ae43b1cf2ada908e4a5'
independent=read(independent_path)
assert independent['actual_delivery_seal_sha256']==sha(seal)
assert (independent['unique_records'],independent['mean_SD_scalars_recomputed'],independent['display_cells'],independent['count_metrics_recomputed'])==(60,486,243,540)
assert independent['old40_record_JSON_bytes_and_order_exact'] and independent['old324_scalars_exact'] and independent['old162_display_cells_exact'] and independent['S_DFA_six_preserved_excluded']
for n,pin in handoff['actual_closure_pins'].items():assert sha(ROOT/n)==pin['sha256']
assert handoff['actual_C8_root_sha256']=='cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0'
records=read(BASE/'snapshot/records.json')['records'];tables=read(BASE/'snapshot/tables.json')
assert len(records)==len({r['id'] for r in records})==60
by={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
assert set(by)=={(v,'IID',a,s) for v in ['Full','minus_C'] for a in ['Benign','F Flip','FedSA'] for s in range(91001,91011)}
old=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_two_scenes_20261009/snapshot'
assert all(by[(r['variant'],r['distribution'],r['attack'],r['seed'])]==r for r in read(old/'records.json')['records'])
errors=[];metric_checks=0
for r in records:
    assert r['data_contract']['evaluation_split']=='valid' and r['data_contract']['actual_evaluation_rows']==19867
    assert r['views']['native']==r['views']['shared_calibration']
    for view in r['views'].values():
        g=list(view['group_confusion_counts'].values());assert len(g)==2
        calc=dict(accuracy=sum(x['tp']+x['tn'] for x in g)/sum(x['n'] for x in g),aeod=abs(g[0]['tp']/(g[0]['tp']+g[0]['fn'])-g[1]['tp']/(g[1]['tp']+g[1]['fn'])),aspd=abs((g[0]['tp']+g[0]['fp'])/g[0]['n']-(g[1]['tp']+g[1]['fp'])/g[1]['n']))
        for k,v in calc.items():assert abs(view[k]-v)<=1e-12;metric_checks+=1
for panel in tables['panels']:
    assert len(panel['rows'])==9
    for row in panel['rows']:
        for metric in ['accuracy_pct','aeod','aspd']:
            def value(v,s):return by[(v,row['distribution'],row['attack'],s)]['views'][panel['view']]['accuracy' if metric=='accuracy_pct' else metric]*(100 if metric=='accuracy_pct' else 1)
            xs=[value('minus_C',s)-value('Full',s) if row['variant']=='minus_C minus Full' else value(row['variant'],s) for s in panel['seeds']]
            mu=math.fsum(xs)/len(xs);sd=math.sqrt(math.fsum((v-mu)**2 for v in xs)/(len(xs)-1))
            errors.extend([abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])])
assert len(errors)==486 and max(errors)<=1e-12 and metric_checks==540
assert read(BASE/'snapshot/verification.json')['display_mean_sd_cells']==243
partial=read(BASE/'snapshot/excluded_partial_C_records.json')['records']
assert len(partial)==len({r['id'] for r in partial})==6 and all(r['attack']=='S-DFA' and r['variant']=='minus_C' for r in partial)
assert not set(r['id'] for r in partial)&set(r['id'] for r in records)
for old_panel,new_panel in zip(read(old/'tables.json')['panels'],tables['panels']):
    assert old_panel['rows']==[r for r in new_panel['rows'] if r['attack'] in ['Benign','F Flip']]
destination=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009'
assert not destination.exists();destination.mkdir()
for n in list(files)+['ACTUAL_FILES_SHA256.json']:
    target=destination/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/n,target);assert sha(target)==sha(BASE/n)
proof=dict(status='ROOT_C30_THREE_SCENE_THREE_VIEW_TABLES_ADOPTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    independent_review_path=independent_path.relative_to(ROOT).as_posix(),independent_review_sha256=sha(independent_path),
    actual_delivery_seal_sha256=sha(seal),actual_members_verified=29,C8_root_adoption_sha256=handoff['actual_C8_root_sha256'],
    records_sha256=sha(BASE/'snapshot/records.json'),tables_sha256=sha(BASE/'snapshot/tables.json'),display_sha256=sha(BASE/'snapshot/TABLES.md'),
    unique_records=60,paired_models=30,complete_scenes=3,mean_SD_scalars_recomputed=486,display_cells=243,count_metrics_recomputed=540,max_abs_difference=max(errors),
    original_two_scene40_records_exact=True,original_two_scene324_statistics_exact=True,original_two_scene162_cells_preserved=True,excluded_partial_C_records=6,
    replay_devices=handoff['replay_devices'],training_torch=handoff['training_torch'],native_shared_identical_records=60,
    seed_panels=[10,9,6],all_negative_results_retained=True,new_CNN=0,new_training=0,new_Full_inference=0,test=False,
    primary_endpoint='PENDING_AUTHOR',other_C_scenes_complete=False,whole_rebuttal_complete=False,
    canonical_table=(destination/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
with (destination/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(root_path=str(destination/'ROOT_VERIFICATION.json'),root_sha256=sha(destination/'ROOT_VERIFICATION.json'),table=proof['canonical_table'])))
