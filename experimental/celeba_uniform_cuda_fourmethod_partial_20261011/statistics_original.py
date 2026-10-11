from pathlib import Path
from statistics import mean,stdev
from collections import defaultdict
import ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
METHODS=['FedAvg','GuardFed-AD2+','FLGMM','LoGoFair-DP']
DISTS=['IID','non-IID']
ATTACKS=['Benign','F Flip','FedSA','S-DFA','Sp-DFA']
METRICS=[('accuracy','ACC (%) ↑',100,2),('aeod','AEOD ↓',1,4),('aspd','ASPD ↓',1,4)]
SEEDS={'ten':list(range(91001,91011)),'nonselection_nine':list(range(91002,91011)),'matching_six':list(range(91005,91011))}
def need(ok,message):
 if not ok:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())

def original_values_for(groups):
    pin = read(HERE/'INPUT_PINS.json')['renderer']
    path = ROOT/pin['path']; need(digest(path) == pin['sha256'], 'Original renderer changed')
    tree = ast.parse(path.read_text(encoding='utf8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'values_for')
    ns = dict(groups=groups, metrics=METRICS, mean=mean, stdev=stdev)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['values_for']


def summaries(rows):
    groups = defaultdict(list)
    for r in rows: groups[r['method'],r['distribution'],r['attack']].append(dict(r, **{k:r['metrics'][k] for k,_,_,_ in METRICS}))
    display = original_values_for(groups); panels = {}; aggregates = {}
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
    return panels,aggregates


NOTES='Partial four-method uniformly-CUDA validation; not the final17-method comparison or a newly selected manuscript endpoint. Native predictions only; ACC in %, AEOD/ASPD on [0,1]. AEOD is the implemented absolute TPR gap, not full equalized odds. Mean ± sample SD(ddof1). Original70-round CNN checkpoints are reused; CNN/root+valid forward evaluation uses CUDA FP32, while each frozen method retains its original postprocessing implementation/device. LoGoFair-DP applies its unchanged fitted recipe07/fitseed1719 state to adopted FedAvg CUDA margins, with netcal1.3.6 CPU postprocessing and20 image-ID virtual cohorts, not real training clients; DP means demographic parity. No refit or new CNN is performed for LoGo. Original cu128/cu130 training histories and adaptation boundaries remain; CUDA evaluation is not a claim of training-runtime equivalence. Seed91001 participated in validation selection; n9/n6 are sensitivity subsets, not untouched test cohorts. All fixed seeds, negative outcomes and constant predictors are retained. No score, best-seed selection, significance or final-test claim. Each native pipeline includes its own postprocessing and does not isolate aggregation causality.'

def markdown(panels,aggregates):
    lines=['# partial four-method uniformly-CUDA validation','',NOTES,'']
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

