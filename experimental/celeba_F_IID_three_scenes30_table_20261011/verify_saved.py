"""Root review entry: original count/fsum loops + saved display/paired-source checks.
No invocation is authorized during source preparation. No model/arrays/fit import.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.dont_write_bytecode=True
from binding import load,need,IDS
from build import variant_module,function,FULL,sha,OLD
H=Path(__file__).resolve().parent
R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())

def main():
    need(not sys.flags.optimize,'Optimized Python forbidden')
    pins,root,index=load()
    records=read(H/'records.json')['records'];tables=read(H/'tables.json')
    need(len(records)==60 and len({r['id'] for r in records})==60,'Exact60 saved records required')
    need(records[:40]==read(OLD/'records.json')['records'], 'Old40 objects/order changed')
    need([r['id'] for r in records[40:] if r['variant']=='minus_F']==IDS,'Exact new FedSA seed order differs')
    need((tables['complete_scenes'],tables['paired_models'],tables['displayed_records'],tables['preserved_records'])==(3,30,60,60),'Wrong scene/pair cardinality')
    numeric=variant_module(R/pins['C_numeric'])
    checks=numeric.verify(records,tables['panels'])
    need(checks['mean_sd_scalars']==486 and checks['receipt_metrics_from_group_counts']==540 and checks['base_confusion_counts_structurally_checked']==1440,'Original numeric verification incomplete')
    source=read(H/'SOURCE_BINDINGS.json')
    need(source['actual_F30_root_adoption_sha256']==pins['files'][pins['adoption']]['sha256'] and source['accepted_index_sha256']==pins['files'][pins['index']]['sha256'],'Actual adoption binding differs')
    accepted={r['id']:r for r in index['new_records']}
    original_full=dict(need=need,FULL=FULL,sha=sha)
    function(R/pins['full_record_source'],['full_record'],original_full)
    baseline={r['id']:r for r in read(FULL/'records_three_views_900.json')['records']}
    refs={r['id']:r for r in read(R/pins['original_full_reference_inventory'])['full_references']}
    for r in records:
        need((r['distribution'],r['attack']) in [('IID','Benign'),('IID','F Flip'),('IID','FedSA')] and set(r['views'])=={'raw','native','shared_calibration'},'Other scene/view cannot enter table')
        if r['variant']=='minus_F' and r['attack'] in ('Benign','F Flip'):
            need(r in read(OLD/'records.json')['records'],'Old two-scene record changed')
        elif r['variant']=='minus_F':
            need(r['views']==accepted[r['id']]['views'] and r['checkpoint_sha256']==accepted[r['id']]['checkpoint_sha256'],'Accepted F checkpoint/views differ')
        else:
            need(r['variant']=='Full' and r==original_full['full_record'](baseline[r['id']],refs[r['id']]),'Original Full record differs')
    for current,previous in zip(tables['panels'],read(OLD/'tables.json')['panels']):
        need({k:current[k] for k in ('view','label','seeds')}=={k:previous[k] for k in ('view','label','seeds')},'Old panel identity changed')
        need([r for r in current['rows'] if r['attack'] in ('Benign','F Flip')]==previous['rows'],'Old two-scene324 scalars changed')
    fragments=dict(json=json)
    function(R/pins['record_fragment_source'],['record_fragments'],fragments)
    need(fragments['record_fragments']((H/'records.json').read_text('utf8'))[:40]==fragments['record_fragments']((OLD/'records.json').read_text('utf8')),'Old40 serialized bytes/order changed')
    paired=read(H/'paired_per_seed.json')
    full_by_seed={(r['distribution'],r['attack'],r['seed']):r for r in records if r['variant']=='Full'}
    for view in ('native','raw','shared_calibration'):
        expected=[]
        for r in records:
            if r['variant']!='minus_F':continue
            f=full_by_seed[r['distribution'],r['attack'],r['seed']]
            expected.append({k:r[k] for k in ('id','variant','distribution','attack','seed')} | dict(accuracy_pct=100*r['views'][view]['accuracy']-100*f['views'][view]['accuracy'],aeod=r['views'][view]['aeod']-f['views'][view]['aeod'],aspd=r['views'][view]['aspd']-f['views'][view]['aspd']))
        need(paired[view]==expected,'Saved paired differences differ or negative values omitted')
    # Match each displayed cell to its same saved row; independent arithmetic above.
    lines=[];cells=0
    for panel in tables['panels']:
        for row in panel['rows']:
            values=[f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in ('accuracy_pct','aeod','aspd')]
            lines.append('| '+' | '.join([row['distribution']+' / '+row['attack']+' / '+row['variant'],str(row['n']),*values])+' |');cells+=len(values)
    actual=[line for line in (H/'TABLES.md').read_text('utf8').splitlines() if line.startswith('| IID / ')]
    need(actual==lines and cells==243,'Saved Markdown cell/order differs')
    need(not tables['primary_endpoint_selected'] and not tables['final_test'] and tables['new_threshold_fits']==tables['new_inference']==tables['new_training']==0,'Unpermitted claim')
    print(json.dumps(dict(status='PASS_ORIGINAL_FSUM_COUNTS_AND_SAVED_DISPLAY_SCOPE',**checks,display_cells=cells,root_adopted_by_this_checker=False,fit=0,CNN=0,test=False),indent=2))

if __name__=='__main__':main()
