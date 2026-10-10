"""Independent arithmetic after strict receipt joining; not an acceptance tool."""
import itertools
import math

def verify(records,panels):
    assert len(records)==120 and len({r['id'] for r in records})==120
    bycell={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    assert len(bycell)==120
    scenes={('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA'),('non-IID','Benign')}
    views=['native','raw','shared_calibration'];seedsets=[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
    assert len(panels)==9 and {(p['view'],tuple(p['seeds'])) for p in panels}==set(itertools.product(views,map(tuple,seedsets)))
    partial=[r for r in records if (r['distribution'],r['attack']) not in scenes]
    assert not partial
    errors=[]
    for panel in panels:
        rows=panel['rows'];seeds=panel['seeds'];view=panel['view']
        assert len(rows)==18 and {(r['distribution'],r['attack'],r['variant']) for r in rows}=={(d,a,v) for d,a in scenes for v in ['Full','minus_C','minus_C minus Full']}
        for row in rows:
            assert row['n']==row['expected_n']==len(seeds) and row['complete'] and row['seeds']==seeds
            for metric in ['accuracy_pct','aeod','aspd']:
                def value(variant,seed):
                    r=bycell[variant,row['distribution'],row['attack'],seed]
                    return r['views'][view]['accuracy']*100 if metric=='accuracy_pct' else r['views'][view][metric]
                xs=[value('minus_C',s)-value('Full',s) if row['variant']=='minus_C minus Full' else value(row['variant'],s) for s in seeds]
                mu=math.fsum(xs)/len(xs);sd=math.sqrt(math.fsum((x-mu)**2 for x in xs)/(len(xs)-1))
                errors.extend([abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])])
    assert len(errors)==972 and max(errors)<=1e-12
    metric_checks=0;count_checks=0
    for record in records:
        assert set(record['views'])=={'native','raw','shared_calibration'}
        for view in record['views'].values():
            groups=view['group_confusion_counts'];a,b=groups['0'],groups['1'];n=a['n']+b['n']
            assert n==view['prediction_count']==19867
            for group in (a,b):
                assert all(isinstance(group[k],int) and group[k]>=0 for k in ('tp','fp','tn','fn'))
                assert group['tp']+group['fn']==group['positives'] and group['fp']+group['tn']==group['negatives']
                assert group['positives']+group['negatives']==group['n'];count_checks+=4
            metrics=dict(accuracy=(a['tp']+a['tn']+b['tp']+b['tn'])/n,
                aeod=abs(a['tp']/(a['tp']+a['fn'])-b['tp']/(b['tp']+b['fn'])),
                aspd=abs((a['tp']+a['fp'])/a['n']-(b['tp']+b['fp'])/b['n']))
            for key,value in metrics.items():assert abs(view[key]-value)<=1e-12;metric_checks+=1
    assert metric_checks==1080 and count_checks==2880
    return {'mean_sd_scalars':len(errors),'max_abs_difference':max(errors),'partial_pairs_excluded':0,
        'receipt_metrics_from_group_counts':metric_checks,'base_confusion_counts_structurally_checked':count_checks}



def verify_aggregate(records,panels):
    by={(r['variant'],r['attack'],r['seed']):r for r in records}
    scenes=['Benign','F Flip','FedSA','S-DFA','Sp-DFA']; errors=[]
    assert len(by)==100 and len(panels)==9
    expected_seeds=[tuple(range(91001,91011)),tuple(range(91002,91011)),tuple(range(91005,91011))]
    assert {(p['view'],tuple(p['seeds'])) for p in panels}==set(itertools.product(['native','raw','shared_calibration'],expected_seeds))
    for panel in panels:
        seeds=panel['seeds'];view=panel['view'];assert seeds in [list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
        assert len(panel['rows'])==3 and {r['variant'] for r in panel['rows']}=={'Full','minus_C','minus_C minus Full'}
        for row in panel['rows']:
            assert row['n']==row['expected_n']==len(seeds) and row['seeds']==seeds and row['complete']
            for metric in ['accuracy_pct','aeod','aspd']:
                def one(variant,seed):
                    xs=[by[variant,a,seed]['views'][view]['accuracy']*100 if metric=='accuracy_pct' else by[variant,a,seed]['views'][view][metric] for a in scenes]
                    return math.fsum(xs)/5
                xs=[one('minus_C',seed)-one('Full',seed) if row['variant']=='minus_C minus Full' else one(row['variant'],seed) for seed in seeds]
                mu=math.fsum(xs)/len(xs);sd=math.sqrt(math.fsum((x-mu)**2 for x in xs)/(len(xs)-1))
                errors.extend([abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])])
    assert len(errors)==162 and max(errors)<=1e-12
    return dict(mean_sd_scalars=162,max_abs_difference=max(errors),n_is_seed_count=True,scene_count_per_seed=5)

def main():
    import argparse,json
    from pathlib import Path
    import build as b
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);args=parser.parse_args()
    bindings=b.read(args.snapshot/'SOURCE_BINDINGS.json');b.verify_future_binding(bindings['actual_C4_binding'])
    b.verify_inputs()
    for name,pin in b.read(args.snapshot/'FILES_SHA256.json')['files'].items():b.need(b.sha(args.snapshot/name)==pin['sha256'],'Snapshot changed '+name)
    records=b.read(args.snapshot/'records.json')['records'];tables=b.read(args.snapshot/'tables.json')
    result=verify(records,tables['panels'])
    result['display_mean_sd_cells']=b.displayed_cells((args.snapshot/'TABLES.md').read_text('utf8'),tables['panels'])
    b.need(result['display_mean_sd_cells']==486,'Display scope changed')
    iid=[r for r in records if r['distribution']=='IID']
    aggregate=args.snapshot/'cross_scene_seed_first.json'
    b.need(aggregate.read_bytes()==(b.OLD/'snapshot/cross_scene_seed_first.json').read_bytes(),'Old IID aggregate bytes changed')
    result['preserved_IID_seed_first']=verify_aggregate(iid,b.read(aggregate)['panels'])
    result['status']='INDEPENDENT_STDLIB_NUMERIC_AND_DISPLAY_PASS_PENDING_ROOT_REVIEW'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
