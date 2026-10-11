"""Offline native-only table candidate; requires actual root adoption of FLGMM96 plus four explicit screen records."""
import argparse, ast, hashlib, json, math, os, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'outputs/guardfed_tables/celeba_ten_method_native_20261010'
METHODS = ['FedAvg', 'FairFed', 'Median', 'FLTrust', 'FedAA-DDPG', 'LASA', 'FairGuard', 'FLTrust+FairGuard', 'GuardFed-AD2+', 'LoGoFair-DP', 'FLGMM']
DISTS = ['IID', 'non-IID']
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
METRICS = [('accuracy', 'ACC (%) ↑', 100, 2), ('aeod', 'AEOD ↓', 1, 4), ('aspd', 'ASPD ↓', 1, 4)]
SEEDS = {'ten': list(range(91001, 91011)), 'nonselection_nine': list(range(91002, 91011)), 'matching_six': list(range(91005, 91011))}
# Actual FLGMM evidence is late-bound; no future hash is predicted.


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


def approval_gate(binding, pins):
    need(binding['status']=='ROOT_REVIEWED_ACTUAL_FLGMM96_PLUS4_NATIVE_TABLE_READY'
         and binding['root_execution_authorized'] is True and binding['view']=='native'
         and binding['final_test'] is False, 'Explicit actual root binding required')
    ref=binding['flgmm96_root'];proof=pinned(location(ref['path']),ref['sha256'])
    need(proof['status']=='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
         and proof['accepted_total']==96 and proof['planned_new']==96 and proof['reused_separately']==4
         and proof['final_test'] is False, 'Actual96 strict/offserver root required, not running/partial')
    manifest=read(ROOT/pins['manifest_path'])
    jobs={j['id']:j for j in manifest['jobs']}
    need(len(jobs)==96 and len(proof['accepted_job_ids'])==96
         and set(proof['accepted_job_ids'])==set(jobs), 'Root96/manifest identity mismatch')
    need(binding['reused_screen_ids']==pins['reused_screen_ids'] and binding['new_expected']==96
         and binding['reused_expected']==4 and binding['new_fit']==binding['new_inference']==0, 'Fixed96+4 only')
    env=binding['environment_evidence'];pinned(location(env['path']),env['sha256'])
    need(isinstance(binding['environment_note'],str) and binding['environment_note'].strip(), 'Actual reviewed environment disclosure required')
    return manifest,proof


def source_inputs():
    need(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'Optimized Python refused')
    pins=read(HERE/'INPUT_PINS.json')
    for name,expected in pins['files'].items(): need(digest(ROOT/name)==expected,'Pinned source changed: '+name)
    proof=read(OLD/'ROOT_REVIEW.json')
    need(proof['status']=='ROOT_TEN_METHOD_NATIVE1000_DESCRIPTIVE_TABLE_ADOPTED'
         and proof['records']==1000 and proof['methods']==10 and proof['final_test'] is False,'Actual original1000 required')
    need(METHODS[:-1]==pins['native_methods'], 'Original method display order changed')
    return pins


def original_values_for(groups):
    pin = read(HERE/'INPUT_PINS.json')['renderer']
    path = ROOT/pin['path']; need(digest(path) == pin['sha256'], 'Original renderer changed')
    tree = ast.parse(path.read_text(encoding='utf8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'values_for')
    ns = dict(groups=groups, metrics=METRICS, mean=mean, stdev=stdev)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['values_for']


def join_records(old, binding, pins, manifest):
    expected={(m,d,a,s) for m in METHODS for d in DISTS for a in ATTACKS for s in SEEDS['ten']}
    need(len(old)==1000 and len({r['id'] for r in old})==1000, 'Actual original1000 required')
    jobs={j['id']:j for j in manifest['jobs']}; accepted={}; origins={}
    protocol=read(ROOT/pins['protocol_path'])
    need(isinstance(binding['native96_sources'],list) and binding['native96_sources'], 'Actual native96 source references required')
    for source in binding['native96_sources']:
        document=pinned(location(source['path']),source['sha256'])
        selected=source['record_ids'];need(len(selected)==len(set(selected)), 'Duplicate source selection')
        entries=document['records']; byid={r['id']:(i,r) for i,r in enumerate(entries)}
        need(len(byid)==len(entries) and set(selected)<=set(byid), 'Original record source IDs differ')
        for rid in selected:
            need(rid in jobs and rid not in accepted, 'Unexpected or repeated native ID')
            i,row=byid[rid]; accepted[rid]=row
            origins[rid]=dict(path=source['path'],sha256=source['sha256'],pointer='/records/'+str(i),reused_screen=False)
    need(set(accepted)==set(jobs), 'Exactly all96 original native records required')
    for j in manifest['reused_jobs']:
        need(j['id'] not in accepted,'Screen reused twice')
        accepted[j['id']]=j['accepted_record']
        origins[j['id']]=dict(path=pins['manifest_path'],sha256=pins['files'][pins['manifest_path']],
             pointer='/reused_jobs/'+str(manifest['reused_jobs'].index(j))+'/accepted_record',reused_screen=True)
        jobs[j['id']]=dict(j,seed=j['accepted_record']['seed'])
    need(len(accepted)==100, 'All100 FLGMM native records required')
    rows=list(old)  # Preserve every original1000 record object and its order.
    for d in DISTS:
        for a in ATTACKS:
            for seed in SEEDS['ten']:
                matches=[r for r in accepted.values() if (r['distribution'],r['attack'],r['seed'])==(d,a,seed)]
                need(len(matches)==1,'Missing/duplicate FLGMM cell');r=matches[0];j=jobs[r['id']]
                need(r['candidate']==pins['selected_recipe']['id'] and r['rounds']==70
                     and r['evaluation_stats']['prediction_count']==19867, 'Recipe/terminal/validation denominator differs')
                need(r['job_sha256']==j['job_sha256'] and (j['distribution'],j['attack'],j['seed'])==(d,a,seed),'Frozen job identity differs')
                need(set(r['metrics'])=={k for k,_,_,_ in METRICS}, 'Original native metric schema required; views cannot be substituted')
                for key in ('checkpoint_sha256','original_acceptance_sha256','screen_identity_sha256'):
                    need(isinstance(r[key],str) and len(r[key])==64,'Missing original checkpoint/acceptance identity')
                need(r['source_hashes']==protocol['source_hashes'] and bool(r['adapter_source_hashes']), 'Original frozen source/data identity differs')
                reference=manifest['reused_jobs'][0]['accepted_record']['adapter_source_hashes']
                for key in ('flgmm_adapter.py','sources/flgmm_pinned.py','sources/flgmm_fedavg.py'):
                    need(r['adapter_source_hashes'][key]==reference[key], 'Frozen FLGMM aggregation source differs')
                if not origins[r['id']]['reused_screen']:
                    job=pinned((ROOT/pins['manifest_path']).parent/'jobs'/j['job'],j['job_sha256']);cfg=job['config']
                    need(cfg['rounds']==70 and cfg['celeba_evaluation_split']=='valid' and cfg['seed']==seed
                         and cfg['client_alpha']=={'IID':5000.,'non-IID':5.}[d]
                         and cfg['celeba_train_limit']==cfg['celeba_eval_limit']==0,'Frozen job round/split/seed/alpha/data scope differs')
                rows.append(dict(id=r['id'],method='FLGMM',distribution=d,attack=a,seed=seed,
                    checkpoint_sha256=r['checkpoint_sha256'],metrics=r['metrics'],source_record=origins[r['id']],
                    original_native_record=r,native_definition='Original accepted FLGMM author-code native margin>0 output; no shared calibration',
                    terminal_round=70,evaluation_split='valid',test_evaluation_performed=False))
    need(rows[:1000]==old,'Original1000 objects changed')
    need(len(rows)==len({r['id'] for r in rows})==1100 and {(r['method'],r['distribution'],r['attack'],r['seed']) for r in rows}==expected,'1100 unique complete cells required')
    for r in rows:need(all(math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1 for k,_,_,_ in METRICS),'Invalid metric')
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
        need(panel_rows[:100] == old_summary[panel], 'Original ten-method native statistics/IDs/display changed')
        old_checks += 100*3
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
 'All outcomes/negative and constant predictions are retained. No score, significance or final-test claim; eleven methods are not the full17; the primary manuscript endpoint remains the author\'s decision.')


def markdown(panels,aggregates):
    lines=['# CelebA native view — eleven-method author-review candidate','',NOTES,'']
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
    global NOTES
    pins=source_inputs();binding=pinned(args.root_binding,args.root_binding_sha256)
    manifest,proof=approval_gate(binding,pins)
    old=read(OLD/'records_native_1000.json');need(old['view']=='native','Original view changed')
    rows=join_records(old['records'],binding,pins,manifest)
    panels,aggregates,n=summaries(rows,read(OLD/'summary_statistics.json'))
    old_aggregates=read(OLD/'seed_first_aggregates.json')
    for panel in SEEDS:need(aggregates[panel][:30]==old_aggregates[panel],'Original10 seed-first statistics changed')
    NOTES += ' FLGMM follows the author-code largest-cluster/full-model adaptation with declared zero-standard-deviation protection; native predictions are uncalibrated margin>0, not shared group thresholds. '
    NOTES += 'Its fixed recipe uses warmup20/control-width2.0/local-lr0.001 selected with seed91001 validation across IID/non-IID Benign/S-DFA. All four selected screen checkpoints are explicitly reused; the other96 are newly trained records. '
    NOTES += binding['environment_note']
    text=markdown(panels,aggregates)
    original_display=[line for line in (OLD/'TABLES.md').read_text(encoding='utf8').splitlines() if line.startswith('| ')]
    preserved_display=[line for line in text.splitlines() if line.startswith('| ') and not line.startswith('| FLGMM |')
         and not any(line.startswith('| '+panel+' | '+scope+' | FLGMM |') for panel in SEEDS for scope in ('IID','non-IID','balanced_all10'))]
    need(preserved_display==original_display,'Original table display/header/order changed')
    need(args.out.resolve().is_relative_to(HERE) and not args.out.exists(),'Fresh owned snapshot required')
    args.out.mkdir(parents=True)
    values={'records_native_1100.json':dict(records=rows,view='native',source1000_sha256=pins['old_records_sha256']),
        'summary_statistics.json':panels,'seed_first_aggregates.json':aggregates,
        'SOURCE_BINDINGS.json':dict(fixed_inputs=pins,actual_root_binding=dict(path=str(args.root_binding.resolve()),sha256=args.root_binding_sha256),
             actual_flgmm96_root=binding['flgmm96_root'],native96_sources=binding['native96_sources'],reused_screen_ids=pins['reused_screen_ids'],
             original1000_objects_exact=True,old_native_summary_metric_objects_exact=n,original_seed_first_aggregates_exact=True,original_display_lines_exact=True,
             environment_note=binding['environment_note'],environment_evidence=binding['environment_evidence'],
             models=1100,methods=11,scenarios_per_method=10,final_test=False,new_inference=0,new_fit=0,main_endpoint_selected=False,full17_complete=False)}
    for name,value in values.items():(args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (args.out/'TABLES.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(status='ACTUAL_NATIVE1100_TABLE_CANDIDATE_NOT_ROOT_ADOPTED',output=str(args.out),old_exact_metric_objects=n)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root-binding',type=Path,required=True);p.add_argument('--root-binding-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
