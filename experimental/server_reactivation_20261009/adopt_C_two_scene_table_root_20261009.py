"""Root arithmetic/provenance review and canonical delivery of the actual20 paired C models."""
from pathlib import Path
import collections,datetime,hashlib,json,math,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_three_view_C_two_scenes_prepared_20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
seal=BASE/'ACTUAL_FILES_SHA256.json';assert sha(seal)=='4778f70cc126a3f81432f1920cdf0a2fafa53fa7703a046272654c10a59a5287'
files=read(seal)['files'];assert len(files)==27
for n,pin in files.items():assert sha(BASE/n)==pin['sha256'] and (BASE/n).stat().st_size==pin['bytes']
handoff=read(BASE/'ACTUAL_HANDOFF.json')
for n,pin in handoff['actual_closure_pins'].items():assert sha(ROOT/n)==pin['sha256']
assert handoff['actual_C8_root_sha256']=='817d5f8ebebb566ee4b851fd600edcddaf07a410d29e618d1e5a821c5748b775'
records=read(BASE/'snapshot/records.json')['records'];tables=read(BASE/'snapshot/tables.json')
assert len(records)==len({r['id'] for r in records})==40
by={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
assert set(by)=={(v,'IID',a,s) for v in ['Full','minus_C'] for a in ['Benign','F Flip'] for s in range(91001,91011)}
old=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_Benign10_20261009/snapshot'
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
    assert len(panel['rows'])==6
    for row in panel['rows']:
        for metric in ['accuracy_pct','aeod','aspd']:
            def value(v,s):return by[(v,row['distribution'],row['attack'],s)]['views'][panel['view']]['accuracy' if metric=='accuracy_pct' else metric]*(100 if metric=='accuracy_pct' else 1)
            xs=[value('minus_C',s)-value('Full',s) if row['variant']=='minus_C minus Full' else value(row['variant'],s) for s in panel['seeds']]
            mu=math.fsum(xs)/len(xs);sd=math.sqrt(math.fsum((v-mu)**2 for v in xs)/(len(xs)-1))
            errors.extend([abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])])
assert len(errors)==324 and max(errors)<=1e-12 and metric_checks==360
assert read(BASE/'snapshot/verification.json')['display_mean_sd_cells']==162
for old_panel,new_panel in zip(read(old/'tables.json')['panels'],tables['panels']):
    assert old_panel['rows']==[r for r in new_panel['rows'] if r['attack']=='Benign']
destination=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_two_scenes_20261009'
assert not destination.exists();destination.mkdir()
for n in list(files)+['ACTUAL_FILES_SHA256.json']:
    target=destination/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/n,target);assert sha(target)==sha(BASE/n)
proof=dict(status='ROOT_C20_TWO_SCENE_THREE_VIEW_TABLES_ADOPTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    actual_delivery_seal_sha256=sha(seal),actual_members_verified=27,C8_root_adoption_sha256=handoff['actual_C8_root_sha256'],
    records_sha256=sha(BASE/'snapshot/records.json'),tables_sha256=sha(BASE/'snapshot/tables.json'),display_sha256=sha(BASE/'snapshot/TABLES.md'),
    unique_records=40,paired_models=20,complete_scenes=2,mean_SD_scalars_recomputed=324,display_cells=162,count_metrics_recomputed=360,max_abs_difference=max(errors),
    original_Benign24_records_exact=True,original_Benign162_statistics_exact=True,original_Benign81_cells_preserved=True,
    replay_devices=handoff['replay_devices'],training_torch=handoff['training_torch'],native_shared_identical_records=40,
    seed_panels=[10,9,6],all_negative_results_retained=True,new_CNN=0,new_training=0,new_Full_inference=0,test=False,
    primary_endpoint='PENDING_AUTHOR',other_C_scenes_complete=False,whole_rebuttal_complete=False,
    canonical_table=(destination/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
with (destination/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(root_path=str(destination/'ROOT_VERIFICATION.json'),root_sha256=sha(destination/'ROOT_VERIFICATION.json'),table=proof['canonical_table'])))
