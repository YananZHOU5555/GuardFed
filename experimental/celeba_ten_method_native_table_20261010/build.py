"""Offline native-only table candidate; requires actual root adoption of LoGoFair100."""
import argparse, ast, hashlib, json, math, os, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
METHODS = ['FedAvg', 'FairFed', 'Median', 'FLTrust', 'FedAA-DDPG', 'LASA', 'FairGuard', 'FLTrust+FairGuard', 'GuardFed-AD2+', 'LoGoFair-DP']
DISTS = ['IID', 'non-IID']
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
METRICS = [('accuracy', 'ACC (%) ↑', 100, 2), ('aeod', 'AEOD ↓', 1, 4), ('aspd', 'ASPD ↓', 1, 4)]
SEEDS = {'ten': list(range(91001, 91011)), 'nonselection_nine': list(range(91002, 91011)), 'matching_six': list(range(91005, 91011))}
LOGO_RECORDS_SHA = 'fdc7c4f2402e26fdaa7b34bbfa792ceccafc5e3d32759d77940def2ed1fbc98d'


def need(ok, message):
    if not ok: raise ValueError(message)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def location(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def pinned(path, expected):
    need(isinstance(expected, str) and len(expected) == 64 and digest(path) == expected, 'Input SHA mismatch: ' + str(path))
    return read(path)


def approval_gate(proof, records_sha, recipe):
    need(proof.get('accepted_count') == proof.get('root_adopted') == 100
         and proof.get('new_accepted') == 96 and proof.get('reused') == 4
         and proof.get('final_test') is False, 'Actual root-adopted100 (96+4) required')
    need(proof.get('records_sha256') == records_sha == LOGO_RECORDS_SHA
         and proof.get('selected_recipe') == recipe, 'Fixed accepted recipe/records binding required')
    need(bool(proof.get('records_path')) and bool(proof.get('independent_review_path'))
         and isinstance(proof.get('independent_review_sha256'), str)
         and len(proof['independent_review_sha256']) == 64, 'Independent actual review binding missing')


def source_inputs():
    need(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'Optimized Python refused')
    pins = read(HERE / 'INPUT_PINS.json')
    for name, expected in pins['files'].items(): need(digest(ROOT/name) == expected, 'Pinned source changed: ' + name)
    root900 = read(OLD/'ROOT_REVIEW.json')
    need(root900['status'] == 'ROOT_NINE_METHOD900_THREE_VIEW_RECEIPTS_COUNTS_AND_TABLE_STATISTICS_PASS'
         and root900['unique_records'] == 900 and root900['final_test'] is False, 'Accepted original900 required')
    return pins


def original_values_for(groups):
    pin = read(HERE/'INPUT_PINS.json')['renderer']
    path = ROOT/pin['path']; need(digest(path) == pin['sha256'], 'Original renderer changed')
    tree = ast.parse(path.read_text(encoding='utf8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'values_for')
    ns = dict(groups=groups, metrics=METRICS, mean=mean, stdev=stdev)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['values_for']


def join_records(old, logo):
    expected = {(m,d,a,s) for m in METHODS for d in DISTS for a in ATTACKS for s in SEEDS['ten']}
    need(len(old) == 900 and len(logo) == 100, 'Complete900+100 only; no partial cells')
    old_cells = {(r['method'],r['distribution'],r['attack'],r['seed']):r for r in old}
    need(len(old_cells) == 900, 'Duplicate original cell')
    rows = []
    for i,r in enumerate(old):
        need(r['same_checkpoint_all_views'] and r['test_evaluation_performed'] is False, 'Old native identity drift')
        need(r['checkpoint_sha256'] == r['original_inventory_record']['checkpoint']['sha256'], 'Old checkpoint drift')
        rows.append(dict(id=r['id'], method=r['method'], distribution=r['distribution'], attack=r['attack'], seed=r['seed'],
            checkpoint_sha256=r['checkpoint_sha256'], metrics=r['views']['native'],
            source_record_pointer='/records/'+str(i), source_records_sha256=read(HERE/'INPUT_PINS.json')['old_records_sha256'],
            native_definition='Accepted original method-native prediction; own calibration where applicable',
            original_receipt_sha256=r['receipt_sha256'], training_torch=r['training_torch'], inference_runtime=r['runtime']))
    for i,r in enumerate(logo):
        baseline = old_cells.get(('FedAvg',r['distribution'],r['attack'],r['seed']))
        need(baseline is not None and r['baseline_id'] == baseline['id']
             and r['checkpoint_sha256'] == baseline['checkpoint_sha256'], 'LoGo/FedAvg same-cell checkpoint mismatch')
        need(r['fit_seed'] == 1719 and r['id'].startswith('LoGoFair-DP_07_')
             and r['metrics']['prediction_count'] == 19867, 'LoGo fixed recipe/fitseed/valid count changed')
        rows.append(dict(id=r['id'], method='LoGoFair-DP', distribution=r['distribution'], attack=r['attack'], seed=r['seed'],
            checkpoint_sha256=r['checkpoint_sha256'], metrics=r['metrics'], source_record_pointer='/'+str(i),
            source_records_sha256=LOGO_RECORDS_SHA, native_definition='Fitted LoGoFair-DP composite output on20 virtual image-ID cohorts; not FedAvg raw/shared predictions',
            fit_seed=1719, mapping_sha256=r['mapping_sha256'], cache_sha256=r['cache_sha256'],
            result_sha256=r['result_sha256'], acceptance_sha256=r['acceptance_sha256'],
            baseline_source_record_pointer='/records/'+str(old.index(baseline)),
            inference_runtime=r['environment'], pretrained_source_runtime=r['pretrained_source_runtime']))
    need(len({r['id'] for r in rows}) == 1000 and {(r['method'],r['distribution'],r['attack'],r['seed']) for r in rows} == expected, '1000 unique complete scientific cells required')
    for r in rows:
        need(all(math.isfinite(r['metrics'][k]) and 0 <= r['metrics'][k] <= 1 for k,_,_,_ in METRICS), 'Invalid metric')
    return rows


def summaries(rows, old_summary):
    groups = defaultdict(list)
    for r in rows: groups[r['method'],r['distribution'],r['attack']].append(dict(r, **{k:r['metrics'][k] for k,_,_,_ in METRICS}))
    display = original_values_for(groups); panels = {}; aggregates = {}; old_checks = 0
    for panel,seeds in SEEDS.items():
        panel_rows = []
        for m in METHODS:
            for d in DISTS:
                for a in ATTACKS:
                    selected = [r for r in groups[m,d,a] if r['seed'] in seeds]
                    need(len(selected) == len(seeds), 'Incomplete seed panel')
                    cells = display(m,d,a,seeds)
                    row = dict(method=m, distribution=d, attack=a, seeds=seeds, IDs=[r['id'] for r in selected], n=len(seeds))
                    for i,(k,_,_,_) in enumerate(METRICS):
                        vals=[r[k] for r in selected]; row[k]=dict(mean=mean(vals),sample_sd=stdev(vals),display=cells[i])
                    panel_rows.append(row)
        need(panel_rows[:90] == old_summary[panel], 'Original nine-method native statistics/IDs/display changed')
        old_checks += 90*3
        panels[panel] = panel_rows
        aggregates[panel] = []
        for m in METHODS:
            for scope,distributions in [('IID',['IID']),('non-IID',['non-IID']),('balanced_all10',DISTS)]:
                seed_rows=[]
                for seed in seeds:
                    selected=[r for r in rows if r['method']==m and r['distribution'] in distributions and r['seed']==seed]
                    need(len(selected)==5*len(distributions), 'Unbalanced cross-scene scope')
                    seed_rows.append(dict(seed=seed, n_scenes=len(selected), **{k:mean(r['metrics'][k] for r in selected) for k,_,_,_ in METRICS}))
                aggregates[panel].append(dict(method=m,scope=scope,n=len(seeds),seed_first=seed_rows,
                    **{k:dict(mean=mean(r[k] for r in seed_rows),sample_sd=stdev(r[k] for r in seed_rows)) for k,_,_,_ in METRICS}))
    return panels,aggregates,old_checks


NOTES = ('Validation-only; backbone terminal round70, LoGoFair30 frozen postprocessing rounds. Mean ± sample SD(ddof1); ACC %, gaps [0,1]. '
 'AEOD is absolute TPR gap, not full equalized odds. LoGoFair-DP uses fitted native predictions, fixed fitseed1719 and20 image-ID virtual cohorts, not true training clients. '
 'FairFed/FairGuard/combination and FedAA/LASA remain documented project adaptations. Native predictions include each method\'s own postprocessing; differences do not isolate aggregation causality. '
 'The old900 retain434CPU/466GPU replay and886cu128/14cu130 training histories; LoGoFair uses those accepted FedAvg checkpoints/caches and CPU netcal1.3.6 fitting. No full runtime equivalence is claimed. '
 'Seed91001 participated in validation selection; other validation seeds and historical test metadata were exposed. n9/n6 subsets are sensitivity views, not an untouched test cohort. '
 'All outcomes/negative and constant predictions are retained. No score, significance or final-test claim; ten methods are not the full17; the primary manuscript endpoint remains the author\'s decision.')


def markdown(panels,aggregates):
    lines=['# CelebA native view — ten-method author-review candidate','',NOTES,'']
    for panel,table in panels.items():
        by={(r['method'],r['distribution'],r['attack']):r for r in table}
        for dist in DISTS:
            lines += ['## '+dist+' / '+panel,'','| Method | Metric | '+' | '.join(ATTACKS)+' |','|---|---|'+'---:|'*5]
            for method in METHODS:
                for k,label,_,_ in METRICS:
                    lines.append('| '+' | '.join([method,label]+[by[method,dist,a][k]['display'] for a in ATTACKS])+' |')
            lines.append('')
    lines += ['# Seed-first descriptive aggregates','','Each model seed is averaged across its five scenes per distribution or all ten balanced scenes, before cross-seed mean/SD; scenes are not independent seeds.','', '| Panel | Scope | Method | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |','|---|---|---|---:|---:|---:|---:|']
    for panel,items in aggregates.items():
        for r in items:
            cells=[f"{r[k]['mean']*scale:.{precision}f} ± {r[k]['sample_sd']*scale:.{precision}f}" for k,_,scale,precision in METRICS]
            lines.append('| '+' | '.join([panel,r['scope'],r['method'],str(r['n']),*cells])+' |')
    return '\n'.join(lines)+'\n'


def run(args):
    pins=source_inputs(); proof=pinned(args.logofair_root,args.logofair_root_sha256)
    recipe=read(ROOT/pins['screen32_adoption'])['selected_recipe']
    approval_gate(proof,LOGO_RECORDS_SHA,recipe)
    pinned(location(proof['independent_review_path']),proof['independent_review_sha256'])
    logo=pinned(location(proof['records_path']),LOGO_RECORDS_SHA)
    old=read(OLD/'records_three_views_900.json')['records']; previous=read(OLD/'summary_statistics.json')['native']
    rows=join_records(old,logo);panels,aggregates,n=summaries(rows,previous)
    need(args.out.resolve().is_relative_to(HERE) and not args.out.exists(), 'Fresh owned snapshot required')
    args.out.mkdir(parents=True)
    values={'records_native_1000.json':dict(records=rows,view='native',source900_sha256=pins['old_records_sha256'],source100_sha256=LOGO_RECORDS_SHA),
      'summary_statistics.json':panels,'seed_first_aggregates.json':aggregates,
      'SOURCE_BINDINGS.json':dict(fixed_inputs=pins,actual_logofair_root=dict(path=str(args.logofair_root.resolve()),sha256=args.logofair_root_sha256),
          actual_logofair_records=dict(path=proof['records_path'],sha256=LOGO_RECORDS_SHA),original900_unchanged=True,old_native_summary_metric_objects_exact=n,
          models=1000,methods=10,scenarios_per_method=10,final_test=False,new_inference=0,new_fit=0,main_endpoint_selected=False,full17_complete=False)}
    for name,value in values.items():(args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (args.out/'TABLES.md').write_text(markdown(panels,aggregates),encoding='utf8')
    print(json.dumps(dict(status='ACTUAL_NATIVE1000_TABLE_CANDIDATE_NOT_ROOT_ADOPTED',output=str(args.out),old_exact_metric_objects=n)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--logofair-root',type=Path,required=True);p.add_argument('--logofair-root-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
