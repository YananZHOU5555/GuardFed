"""Independent arithmetic after strict receipt joining; not an acceptance tool."""
import itertools
import math

def verify(records,panels):
    assert len(records)==40 and len({r['id'] for r in records})==40
    bycell={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    assert len(bycell)==40
    scenes={('IID','Benign'),('IID','F Flip')}
    views=['native','raw','shared_calibration'];seedsets=[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
    assert len(panels)==9 and {(p['view'],tuple(p['seeds'])) for p in panels}==set(itertools.product(views,map(tuple,seedsets)))
    partial=[r for r in records if (r['distribution'],r['attack']) not in scenes]
    assert not partial
    errors=[]
    for panel in panels:
        rows=panel['rows'];seeds=panel['seeds'];view=panel['view']
        assert len(rows)==6 and {(r['distribution'],r['attack'],r['variant']) for r in rows}=={(d,a,v) for d,a in scenes for v in ['Full','minus_C','minus_C minus Full']}
        for row in rows:
            assert row['n']==row['expected_n']==len(seeds) and row['complete'] and row['seeds']==seeds
            for metric in ['accuracy_pct','aeod','aspd']:
                def value(variant,seed):
                    r=bycell[variant,row['distribution'],row['attack'],seed]
                    return r['views'][view]['accuracy']*100 if metric=='accuracy_pct' else r['views'][view][metric]
                xs=[value('minus_C',s)-value('Full',s) if row['variant']=='minus_C minus Full' else value(row['variant'],s) for s in seeds]
                mu=math.fsum(xs)/len(xs);sd=math.sqrt(math.fsum((x-mu)**2 for x in xs)/(len(xs)-1))
                errors.extend([abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])])
    assert len(errors)==324 and max(errors)<=1e-12
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
    assert metric_checks==360 and count_checks==960
    return {'mean_sd_scalars':len(errors),'max_abs_difference':max(errors),'partial_pairs_excluded':0,
        'receipt_metrics_from_group_counts':metric_checks,'base_confusion_counts_structurally_checked':count_checks}


def main():
    import argparse,json
    from pathlib import Path
    import build as b
    parser=argparse.ArgumentParser(description='Independent stdlib arithmetic/count/display verification of an actual adopted two-scene snapshot')
    parser.add_argument('--snapshot',type=Path,required=True)
    args=parser.parse_args()
    bindings=b.read(args.snapshot/'SOURCE_BINDINGS.json')
    b.adoption_gate(Path(bindings['actual_C8_adoption']),bindings['actual_C8_adoption_sha256'])
    for name,pin in bindings['prepared_input_pins'].items():
        b.need(b.sha(b.R/name)==pin['sha256'],'Pinned source changed '+name)
    records=b.read(args.snapshot/'records.json')['records'];tables=b.read(args.snapshot/'tables.json')
    result=verify(records,tables['panels'])
    result['display_mean_sd_cells']=b.displayed_cells((args.snapshot/'TABLES.md').read_text('utf8'),tables['panels'])
    b.need(result['display_mean_sd_cells']==162,'Display scope changed')
    result['status']='INDEPENDENT_STDLIB_NUMERIC_AND_DISPLAY_PASS_PENDING_ROOT_REVIEW'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
