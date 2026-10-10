"""Independent seed arithmetic and original-record checks for the actual A table."""
from pathlib import Path
import argparse
from collections import Counter
import datetime, hashlib, itertools, json, math

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_Benign10_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
p=argparse.ArgumentParser();p.add_argument('--delivery-seal',required=True);a=p.parse_args()
assert sha(HERE/'FILES_SHA256.json')==a.delivery_seal
for name,pin in read(HERE/'FILES_SHA256.json')['files'].items():
    assert sha(HERE/name)==pin['sha256'] and (HERE/name).stat().st_size==pin['bytes']
adoption_path=ROOT/'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/ROOT_ADOPTION.json'
assert sha(adoption_path)=='1221482d564a2c735b0de0680fe8a42512c9fe5774a9d157bd3bfda5dd9c858b'
adoption=read(adoption_path);index=read(ROOT/adoption['records_index_path'])
assert sha(ROOT/adoption['records_index_path'])==adoption['records_index_sha256']
original_A={r['id']:r for r in index['new_records']}
full_path=ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json'
assert sha(full_path)=='983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529'
original_Full={r['id']:r for r in read(full_path)['records']}
records=read(HERE/'records.json')['records'];table=read(HERE/'tables.json')
assert len(records)==24 and len({r['id'] for r in records})==24
cells={(r['variant'],r['attack'],r['seed']):r for r in records}
expected={(v,'Benign',s) for v in ('Full','minus_A') for s in range(91001,91011)}|{(v,'F Flip',s) for v in ('Full','minus_A') for s in (91001,91002)}
assert set(cells)==expected and all(r['distribution']=='IID' for r in records)
assert table['complete_scenes']==1 and table['paired_models']==10 and table['preserved_records']==24
assert table['final_test'] is False and table['primary_endpoint_selected'] is False
assert table['new_threshold_fits']==table['new_inference']==table['new_training']==0
metric_checks=0;count_checks=0
for r in records:
    original=(original_Full if r['variant']=='Full' else original_A)[r['id']]
    assert r['checkpoint_sha256']==original['checkpoint_sha256'] and r['views']==original['views']
    if r['variant']=='Full':
        assert r['training_torch']==original['training_torch'] and r['fits']==original['fits']
    else:
        binding=index['new_bindings'][r['id']]['record']
        assert r['config_sha256']==binding['config_canonical_sha256'] and r['data_contract']==binding['data_contract']
        assert r['checkpoint_sha256']==binding['checkpoint']['sha256']
        receipt=r['provenance']['artifacts']['scientific_receipt']
        assert sha(receipt['path'])==receipt['sha256']
        original_receipt=read(receipt['path'])
        assert r['views']==original_receipt['views'] and r['fits']==original_receipt['fits']
    paired=cells['Full',r['attack'],r['seed']]
    assert r['data_contract']==paired['data_contract']
    for view in r['views'].values():
        left,right=(view['group_confusion_counts'][str(g)] for g in (0,1))
        for g in (left,right):
            assert all(type(g[k]) is int and g[k]>=0 for k in ('tp','fp','tn','fn'))
            assert g['tp']+g['fn']==g['positives'] and g['fp']+g['tn']==g['negatives']
            assert g['positives']+g['negatives']==g['n'];count_checks+=4
        n=left['n']+right['n'];assert n==view['prediction_count']==19867
        calculated=dict(accuracy=(left['tp']+left['tn']+right['tp']+right['tn'])/n,
            aeod=abs(left['tp']/left['positives']-right['tp']/right['positives']),
            aspd=abs((left['tp']+left['fp'])/left['n']-(right['tp']+right['fp'])/right['n']))
        for key,value in calculated.items():
            assert abs(value-view[key])<=1e-12;metric_checks+=1
views=('native','raw','shared_calibration')
seed_sets=[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
panels=table['panels']
assert len(panels)==9 and {(p['view'],tuple(p['seeds'])) for p in panels}==set(itertools.product(views,map(tuple,seed_sets)))
errors=[];all_rows=[]
for panel in panels:
    seeds=panel['seeds'];view=panel['view'];rows=panel['rows']
    assert len(rows)==3 and {r['variant'] for r in rows}=={'Full','minus_A','minus_A minus Full'}
    for row in rows:
        assert row['n']==row['expected_n']==len(seeds) and row['seeds']==seeds and row['complete']
        assert (row['distribution'],row['attack'])==('IID','Benign')
        for key in ('accuracy_pct','aeod','aspd'):
            def value(v,s):
                metrics=cells[v,'Benign',s]['views'][view]
                return 100*metrics['accuracy'] if key=='accuracy_pct' else metrics[key]
            xs=[value('minus_A',s)-value('Full',s) if row['variant']=='minus_A minus Full' else value(row['variant'],s) for s in seeds]
            mean=math.fsum(xs)/len(xs);sd=math.sqrt(math.fsum((x-mean)**2 for x in xs)/(len(xs)-1))
            errors.extend((abs(mean-row[key]['mean']),abs(sd-row[key]['sample_sd_ddof1'])))
        all_rows.append(row)
lines=[x for x in (HERE/'TABLES.md').read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
assert len(lines)==len(all_rows)==27
display=0
for line,row in zip(lines,all_rows):
    columns=line.strip('| ').split(' | ')
    assert columns[0]==row['variant'] and int(columns[1])==row['n']
    for offset,key in enumerate(('accuracy_pct','aeod','aspd'),2):
        digits=3 if key=='accuracy_pct' else 5
        assert columns[offset]==f"{row[key]['mean']:.{digits}f} ± {row[key]['sample_sd_ddof1']:.{digits}f}";display+=1
assert len(errors)==162 and max(errors)<=1e-12 and display==81 and metric_checks==216 and count_checks==576
proof=dict(status='ROOT_A12_SINGLE_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),paired_models=10,complete_scenes=1,
    preserved_records=24,partial_F_Flip_pairs=2,mean_SD_scalars_recomputed=162,display_cells=81,
    metrics_from_group_counts=216,base_integer_confusion_counts_checked=576,max_abs_difference=max(errors),
    original_Full12_records_exact=True,original_A12_saved_views_exact=True,
    replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v and r['attack']=='Benign')) for v in ('Full','minus_A')},
    source_acceptance_sha256=sha(adoption_path),delivery_seal_sha256=a.delivery_seal,
    files_sha256={n:sha(HERE/n) for n in ('records.json','tables.json','TABLES.md','SOURCE_BINDINGS.json')},
    canonical_table=(HERE/'TABLES.md').relative_to(ROOT).as_posix(),
    new_CNN=0,new_fits=0,new_training=0,test=False,whole_rebuttal_complete=False)
with (HERE/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as stream:
    json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],root_proof_sha256=sha(HERE/'ROOT_VERIFICATION.json'),mean_sd=162,cells=81,max_error=max(errors))))
