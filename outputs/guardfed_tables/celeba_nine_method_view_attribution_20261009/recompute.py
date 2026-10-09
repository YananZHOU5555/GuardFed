"""Hash-bound accepted-number aggregation only; no model/arrays/cache access."""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'

def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    p=HERE/name
    if p.exists(): assert p.read_text(encoding='utf-8')==data, 'Refuse altered existing result '+name
    else: p.write_text(data,encoding='utf-8')

pins=load(HERE/'INPUTS.json')['inputs']
for name,pin in pins.items():
    p=ROOT/name; assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],name
seal=load(SOURCE/'FILES_SHA256.json');rootproof=load(SOURCE/'ROOT_REVIEW.json')
assert rootproof['source_seal_sha256']==sha(SOURCE/'FILES_SHA256.json') and rootproof['unique_records']==900
for name in ['records_three_views_900.json','build_three_view_tables.py','coverage_alias_environment.json']:
    assert sha(SOURCE/name)==seal[name]['sha256']
assert sha(SOURCE/'records_three_views_900.json')=='983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529'
# Reuse exact existing constants without executing renderer top-level code.
ns={}
for node in ast.parse((SOURCE/'build_three_view_tables.py').read_text(encoding='utf-8')).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['METHODS','DISTS','ATTACKS','METRICS','SEEDS','VIEWS'] for t in node.targets):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'sealed_renderer_constants','exec'),ns)
METHODS,DISTS,ATTACKS,METRICS,SEEDS,VIEWS=(ns[k] for k in ['METHODS','DISTS','ATTACKS','METRICS','SEEDS','VIEWS'])
records=load(SOURCE/'records_three_views_900.json')['records'];coverage=load(SOURCE/'coverage_alias_environment.json')
bycell={(r['method'],r['distribution'],r['attack'],r['seed']):r for r in records}
assert len(records)==len(bycell)==900 and set(bycell)==set(itertools.product(METHODS,DISTS,ATTACKS,SEEDS['ten']))
assert len({r['id'] for r in records})==900
for r in records:
    assert r['same_checkpoint_all_views'] and not r['test_evaluation_performed'] and set(r['views'])==set(VIEWS)
    assert all(math.isfinite(r['views'][v][k]) and 0<=r['views'][v][k]<=1 for v in VIEWS for k in METRICS)

series={};per_seed=[]
for method,view in itertools.product(METHODS,VIEWS):
    for seed in SEEDS['ten']:
        rs=[bycell[method,d,a,seed] for d,a in itertools.product(DISTS,ATTACKS)]
        scores={k:mean(r['views'][view][k] for r in rs) for k in METRICS}
        series[method,view,seed]=scores
        per_seed.append({'method':method,'view':view,'seed':seed,'scenarios':10,'source_ids':[r['id'] for r in rs],'metrics':scores})

def stats(values): return {'n':len(values),'mean':mean(values),'sample_sd':stdev(values)}
def vector(method,view,seeds,k): return [series[method,view,s][k] for s in seeds]

panels={};paired_values=[]
for panel,seeds in SEEDS.items():
    means={m:{v:{k:stats(vector(m,v,seeds,k)) for k in METRICS} for v in VIEWS} for m in METHODS}
    transitions={}
    for m in METHODS:
        transitions[m]={}
        for target in ['native','shared_calibration']:
            vals={k:[series[m,target,s][k]-series[m,'raw',s][k] for s in seeds] for k in METRICS}
            transitions[m][target]={k:stats(v) for k,v in vals.items()}
            for i,s in enumerate(seeds):paired_values.append({'panel':panel,'type':'target_minus_raw','method':m,'target':target,'seed':s,'metrics':{k:vals[k][i] for k in METRICS}})
    contrasts={}
    for m in METHODS[:-1]:
        contrasts[m]={}
        for view in VIEWS:
            vals={k:[(series['GuardFed-AD2+',view,s][k]-series[m,view,s][k])*(1 if k=='accuracy' else -1) for s in seeds] for k in METRICS}
            contrasts[m][view]={k:stats(v) for k,v in vals.items()}
            for i,s in enumerate(seeds):paired_values.append({'panel':panel,'type':'GuardFed_advantage','baseline':m,'view':view,'seed':s,'metrics':{k:vals[k][i] for k in METRICS}})
    panels[panel]={'seeds':seeds,'method_statistics':means,'view_changes_target_minus_raw':transitions,'GuardFed_advantage':contrasts,
       'GuardFed_positive_mean_advantage_baseline_count':{v:{k:sum(contrasts[m][v][k]['mean']>0 for m in contrasts) for k in METRICS} for v in VIEWS}}
output={'status':'ACCEPTED900_DESCRIPTIVE_VIEW_COMPARISON','source_records_sha256':sha(SOURCE/'records_three_views_900.json'),
 'scenario_aggregation':'Equal mean of IID/non-IID × Benign/F Flip/FedSA/S-DFA/Sp-DFA within each method/seed/view; then mean and sample SD across seeds.',
 'units':'All values retain original0..1 scale; report multiplies ACC and ACC deltas by100.',
 'view_change_direction':'target minus raw; positive ACC better, negative AEOD/ASPD better.',
 'GuardFed_advantage_direction':'ACC=GuardFed-baseline; AEOD/ASPD=baseline-GuardFed. Positive means better GuardFed for all3.',
 'method_aliases':coverage['method_aliases'],'panels':panels,'new_inference':0,'threshold_refit':False,'test':False,'significance_or_CI':False}
save('statistics.json',output);save('per_seed_ten_scene_means.json',per_seed);save('paired_per_seed.json',paired_values)
# Independent NumPy: rebuild arrays from original records, not stored series.
tensor=np.asarray([[[[[bycell[m,d,a,s]['views'][v][k] for k in METRICS] for d,a in itertools.product(DISTS,ATTACKS)] for s in SEEDS['ten']] for v in VIEWS] for m in METHODS],dtype=float)
independent=tensor.mean(axis=3);diffs=[];checked=0
for row in per_seed:
    actual=independent[METHODS.index(row['method']),VIEWS.index(row['view']),SEEDS['ten'].index(row['seed'])]
    diffs.extend(abs(actual[i]-row['metrics'][k]) for i,k in enumerate(METRICS));checked+=3
for name,panel in panels.items():
    ids=[SEEDS['ten'].index(s) for s in panel['seeds']]
    for mi,m in enumerate(METHODS):
        for vi,v in enumerate(VIEWS):
            a=independent[mi,vi,ids,:]
            for i,k in enumerate(METRICS):
                st=panel['method_statistics'][m][v][k];diffs.extend([abs(a[:,i].mean()-st['mean']),abs(a[:,i].std(ddof=1)-st['sample_sd'])]);checked+=2
        for v in ['native','shared_calibration']:
            a=independent[mi,VIEWS.index(v),ids,:]-independent[mi,0,ids,:]
            for i,k in enumerate(METRICS):
                st=panel['view_changes_target_minus_raw'][m][v][k];diffs.extend([abs(a[:,i].mean()-st['mean']),abs(a[:,i].std(ddof=1)-st['sample_sd'])]);checked+=2
    for mi,m in enumerate(METHODS[:-1]):
        for vi,v in enumerate(VIEWS):
            a=(independent[-1,vi,ids,:]-independent[mi,vi,ids,:])*np.asarray([1,-1,-1])
            for i,k in enumerate(METRICS):
                st=panel['GuardFed_advantage'][m][v][k];diffs.extend([abs(a[:,i].mean()-st['mean']),abs(a[:,i].std(ddof=1)-st['sample_sd'])]);checked+=2
assert max(diffs)<1e-12
old=load(ROOT/'outputs/guardfed_tables/celeba_nine_method_final_20261004/seed_paired_summary.json')
old_diffs=[]
for panel in SEEDS:
    for m,k in itertools.product(METHODS,METRICS):
        for stat in ['mean','sample_sd']:
            old_diffs.append(abs(old[panel]['methods'][m][k][stat]-panels[panel]['method_statistics'][m]['native'][k][stat]))
assert max(old_diffs)<1e-12
native_raw_exact={m:all(r['views']['native'][k]==r['views']['raw'][k] for r in records if r['method']==m for k in METRICS) for m in METHODS}
gf_same=all(r['views']['native']==r['views']['shared_calibration'] for r in records if r['method']=='GuardFed-AD2+')
save('verification.json',{'status':'INDEPENDENT_NUMPY_AND_PURE_STATISTICS_PASS','source_sha256':output['source_records_sha256'],'records':900,'complete_ten_scene_seed_groups':90,'scalar_checks':checked,'max_abs_difference':max(diffs),'native_prior_paired_summary_scalar_checks':len(old_diffs),'native_prior_paired_summary_max_difference':max(old_diffs),'baseline_native_raw_metric_identity':native_raw_exact,'GuardFed_all100_native_shared_full_view_dictionary_identity':gf_same,'archives_reaudited':0,'model_or_prediction_files_opened':0,'science_executed':False})
print(json.dumps({'verification_scalars':checked,'max_difference':max(diffs),'counts':{p:panels[p]['GuardFed_positive_mean_advantage_baseline_count'] for p in SEEDS},'GuardFed_native_shared_identical':gf_same},indent=2))
