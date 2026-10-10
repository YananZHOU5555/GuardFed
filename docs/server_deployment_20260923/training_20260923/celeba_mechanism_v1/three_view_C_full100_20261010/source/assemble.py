"""Generate statistics only from externally accepted, identity-checked records."""
from collections import Counter
import json
from common import H,R,OLD,METRICS,SCENES,need,read,write,sha,module,record_spans,displayed_cells,verify_inputs


def assemble(records,bindings,output):
    need(output.resolve().parent==H.resolve() and not output.exists(),'Fresh owned snapshot required')
    verify_inputs()
    old=read(OLD/'snapshot/records.json')['records']
    need(len(records)==len({r['id'] for r in records})==200,'Exact200 unique records required')
    raw=json.dumps(dict(records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    need(record_spans(raw)[:160]==record_spans((OLD/'snapshot/records.json').read_text('utf8')),'Old160 record bytes/order changed')
    cells={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    expected={(v,d,a,s) for v in ('Full','minus_C') for d,a in SCENES for s in range(91001,91011)}
    need(set(cells)==expected,'Exact Full100/C100 ten-scene grid required')
    for r in records:
        if r['variant']=='minus_C':
            need(r['data_contract']==cells['Full',r['distribution'],r['attack'],r['seed']]['data_contract'],'Paired Full data contract differs')
    original=module('original_statistics',R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py')
    pure=module('C100_panels',H/'panels.py');numeric=module('C100_numeric',H/'verify_numeric.py')
    panels,coverage,paired=pure.panels(records,original);checks=numeric.verify(records,panels)
    newscenes={('non-IID','S-DFA'),('non-IID','Sp-DFA')}
    filtered=[dict(p,rows=[r for r in p['rows'] if (r['distribution'],r['attack']) not in newscenes]) for p in panels]
    need(filtered==read(OLD/'snapshot/tables.json')['panels'],'Old1296 statistic values/order changed')
    iid=[r for r in records if r['distribution']=='IID']
    aggregate=(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()
    aggregate_check=numeric.verify_aggregate(iid,json.loads(aggregate)['panels'])
    need(pure.aggregate_panels(iid,original)==json.loads(aggregate)['panels'],'Old IID162 aggregate differs')
    # Reuse the exact five-scene arithmetic on a metadata-only distribution projection.
    # Original records are immutable; only the aggregation lookup label is rebound.
    projected=[dict(r,distribution='IID') for r in records if r['distribution']=='non-IID']
    noniid=pure.aggregate_panels(projected,original)
    noniid_check=numeric.verify_aggregate(projected,noniid)
    noniid=[dict(p,rows=[dict(r,distribution='non-IID') for r in p['rows']]) for p in noniid]
    balanced=pure.aggregate_balanced_panels(records,original)
    balanced_check=numeric.verify_balanced_aggregate(records,balanced)
    text=['# CelebA Full–minus_C: complete IID/non-IID coverage, three views','',
          'Ten complete scenes,100 matched pairs; valid19867,round70. Mean ± sampleSD(ddof1); differences minus_C−Full. ACC percent; ΔACC pp. Higher ACC and lower gaps are favorable; a positive deletion-minus-Full gap favors Full.','']
    for panel in panels:
        text+=['## '+panel['view']+' — '+panel['label'],'','| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            vals=[f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in METRICS]
            text.append('| '+' | '.join([row['distribution']+' '+row['attack'],row['variant'],str(row['n']),*vals])+' |')
        text.append('')
    text+=['AEOD here is absolute TPR gap, not full equalized odds. Raw/native/shared are parallel outputs of the same checkpoint; native includes each method’s original root-only calibration and shared uses the frozen common fitting rule. No inference or refitting occurs in this table build.',
           'Full replay comprises5 CPU/95 GPU records; Full training comprises98 cu128/2 cu130 records. The100 minus_C replays are CPU. Exact runtime and training fields remain in records.json. Recipe-selection seed91001, validation exposure and historical official-test exposure remain disclosed; these are validation tables, not a newly untouched test.',
           'All10/9/6 panels and unfavorable/zero results are retained. Paired deletion contrasts do not establish necessity, pure aggregation causality or statistical significance. No primary endpoint is selected. This completes minus_C coverage only; it does not complete the other six control variants or the overall revision.',
           'cross_scene_seed_first.json preserves the original five-IID summary byte-for-byte. cross_scene_additional.json adds five-non-IID and balanced-ten-scene summaries: first equally average scenes within each seed, then summarize independent seed units.','']
    rendered='\n'.join(text);display=displayed_cells(rendered,panels)
    oldlines=[x for x in (OLD/'snapshot/TABLES.md').read_text('utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    kept=[x for x in rendered.splitlines() if x.startswith('| ') and ' ± ' in x and not any(x.startswith('| '+d+' '+a+' |') for d,a in newscenes)]
    need(oldlines==kept and len(kept)*3==648,'Old648 display cells/order changed')
    runtime={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_C')}
    training={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_C')}
    need(runtime['minus_C']=={'cpu':100},'C CPU runtime disclosure differs')
    need(runtime['Full'].get('cpu',0)==5 and sum(n for d,n in runtime['Full'].items() if d.startswith('cuda'))==95,'Full mixed runtime disclosure differs')
    need(sum(n for t,n in training['Full'].items() if t.endswith('+cu128'))==98 and sum(n for t,n in training['Full'].items() if t.endswith('+cu130'))==2,'Full CUDA training disclosure differs')
    verify_inputs();output.mkdir()
    write(output/'records.json',dict(records=records))
    write(output/'tables.json',dict(status='C100_TEN_COMPLETE_SCENES_THREE_VIEWS_PENDING_INDEPENDENT_ROOT_REVIEW',unique_records=200,paired_models=100,complete_scenes=10,panels=panels,full_nonIID_coverage=True,primary_endpoint_selected=False,final_test=False,new_inference=0))
    (output/'cross_scene_seed_first.json').write_bytes(aggregate)
    write(output/'cross_scene_additional.json',dict(nonIID_five_scene_panels=noniid,balanced_ten_scene_panels=balanced,n_is_seed_count=True,scene_mean_before_seed_statistics=True))
    write(output/'coverage.json',coverage);write(output/'paired_per_seed.json',paired)
    write(output/'verification.json',dict(checks,display_mean_sd_cells=display,old160_records_bytes_exact=True,old1296_statistics_exact=True,old648_cells_exact=True,old162_IID_seed_first_bytes_exact=True,preserved_IID_seed_first=aggregate_check,nonIID_seed_first=noniid_check,balanced_seed_first=balanced_check,replay_runtime=runtime,training_torch=training))
    write(output/'SOURCE_BINDINGS.json',bindings)
    (output/'TABLES.md').write_text(rendered,encoding='utf8',newline='\n')
    write(output/'FILES_SHA256.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()}))
    return dict(status='BUILT_PENDING_INDEPENDENT_ROOT_REVIEW',records=200,pairs=100,scenes=10,mean_sd_scalars=1620,cells=810,count_metrics=1800,count_checks=4800,additional_seed_first_scalars=324)
