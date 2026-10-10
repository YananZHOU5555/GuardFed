"""Reuse accepted IID seed-first arithmetic; render a paper fragment, no inference."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
METRICS = ['accuracy_pct', 'aeod', 'aspd']

def need(ok, message):
    if not ok: raise ValueError(message)

def put(name, value):
    with (H/name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def display(row, metric, latex=False):
    digits = 3 if metric == 'accuracy_pct' else 5
    token = r'\pm' if latex else '±'
    return f"{row[metric]['mean']:.{digits}f} {token} {row[metric]['sample_sd_ddof1']:.{digits}f}"

def main():
    need(not sys.flags.optimize, 'Optimized Python forbidden')
    need(not (H/'IID_SEED_FIRST.json').exists(), 'No overwrite of actual aggregate')
    pins = read(H/'AGGREGATE_SOURCE_PINS.json')
    for name, pin in pins.items():
        need(sha(R/name) == pin['sha256'] and (R/name).stat().st_size == pin['bytes'], 'Source changed: '+name)
    base = runpy.run_path(str(H/'build.py'), run_name='reader_only')
    old = runpy.run_path(str(base['OLD']/'build.py'), run_name='reader_only')
    panel_path = R/'tmp/celeba_mechanism_C100_table_20261010/panels.py'
    check_path = R/'tmp/celeba_mechanism_C100_table_20261010/verify_numeric.py'
    panels = old['variant_module'](panel_path)
    numeric = old['variant_module'](check_path)
    spec = importlib.util.spec_from_file_location('original_stats', R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py')
    evidence = importlib.util.module_from_spec(spec); spec.loader.exec_module(evidence)
    records = read(H/'records.json')['records']; tables = read(H/'tables.json')
    need(len(records)==100 and all(r['distribution']=='IID' for r in records), 'Only exact five IID scenes')
    aggregate = panels.aggregate_panels(records, evidence)
    aggregate_check = numeric.verify_aggregate(records, aggregate)
    # Both functions are original source; variant_module only changes exact variant string constants.
    source_functions = {}
    for p, names in [(panel_path,['aggregate_panels']), (check_path,['verify_aggregate'])]:
        source = p.read_text(encoding='utf8')
        for node in ast.parse(source).body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                source_functions[node.name] = dict(path=p.relative_to(R).as_posix(),
                    sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest())
    put('IID_SEED_FIRST.json',dict(status='ACTUAL_FIVE_IID_SCENE_SEED_FIRST_CANDIDATE',
        scene_count_per_seed=5,distribution='IID',panels=aggregate,
        definition='For each model seed, equally average its five IID scenes; then mean/sample SD across paired model seeds.',
        not_all_ten_scenes=True,final_test=False))
    checks = read(H/'checks.json')
    put('FOCUSED_CHECKS.json',dict(status='PASS',per_scene=checks,seed_first=aggregate_check,
        total_statistic_scalars=checks['mean_sd_scalars']+aggregate_check['mean_sd_scalars'],
        scene_display_cells=405,aggregate_display_cells=81,
        source_functions=source_functions,source_pins=pins,
        variant_only_AST_inverse_exact=True,independent_reviewer=False,
        excluded_partial_id='minus_A_non-IID_Benign_seed91001',new_inference=0,new_fit=0,new_training=0,test=False))
    # Preserve every old display cell, and add a separate, explicitly IID aggregate.
    text=['\n# Five IID scenes: seed-first aggregate','',
        'AUTHOR-REVIEW CANDIDATE. Each model seed contributes once after its five IID scenes are equally averaged. This is not a balanced ten-scene or non-IID aggregate.','']
    for panel in aggregate:
        text += ['## '+panel['view']+' — '+panel['label'],'',
            '| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |', '|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            text.append('| '+' | '.join([row['variant'],str(row['n']),*[display(row,m) for m in METRICS]])+' |')
        text.append('')
    with (H/'TABLES.md').open('a',encoding='utf8',newline='\n') as f:f.write('\n'.join(text))
    tex=[r'% AUTHOR-REVIEW CANDIDATE — VALIDATION, not final test.',
        r'% Requires booktabs; input as a fragment. Nine panels preserve raw/native/shared and 10/9/6 seeds.',
        r'% Difference = minus_A - Full; ACC in percent / paired ACC in percentage points.',
        r'% AEOD = absolute TPR gap; ASPD = absolute positive-prediction-rate gap.',
        r'% All three metrics come from the same terminal round-70 checkpoint.']
    for panel in tables['panels'] + aggregate:
        label=(panel['view']+'; '+panel['label']).replace('_',r'\_')
        is_aggregate=panel['rows'][0]['attack'].startswith('Five-scene')
        tex += [r'\begin{table*}[t]',r'\centering',r'\small',
            r'\caption{'+('Five IID scenarios, seed-first mean. ' if is_aggregate else 'CelebA, five IID scenarios. ')+label+
            r'. Mean $\pm$ sample SD across paired model seeds; validation only.}',
            r'\begin{tabular}{llrrrr}',r'\toprule',
            r'Scenario & Variant & $n$ & ACC $\uparrow$ & AEOD $\downarrow$ & ASPD $\downarrow$ \\',r'\midrule']
        for row in panel['rows']:
            attack='Five-IID mean' if is_aggregate else row['attack']
            variant={'Full':'Full','minus_A':r'$-A$','minus_A minus Full':r'$\Delta(-A-\mathrm{Full})$'}[row['variant']]
            tex.append(' & '.join([attack,variant,str(row['n']),*['$'+display(row,m,True)+'$' for m in METRICS]])+r' \\')
        tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    with (H/'TABLES.tex').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(tex))
    directions=[]
    for scope, ps in [('per_scene',tables['panels']),('five_IID_seed_first',aggregate)]:
        for p in ps:
            for row in p['rows']:
                if row['variant']=='minus_A minus Full':
                    directions.append(dict(scope=scope,view=p['view'],seeds=p['seeds'],attack=row['attack'],
                        mean={m:row[m]['mean'] for m in METRICS},sample_sd={m:row[m]['sample_sd_ddof1'] for m in METRICS},
                        signs={m:('positive' if row[m]['mean']>0 else 'negative' if row[m]['mean']<0 else 'zero') for m in METRICS}))
    put('DIRECTIONS.json',dict(difference='minus_A minus Full',positive_ACC='higher accuracy after deleting A',
        positive_gap='larger disparity after deleting A',all_directions=directions,significance_claim=False))
    print(json.dumps(dict(status='PASS',per_scene_scalars=810,aggregate_scalars=162,
        total_scalars=972,total_display_cells=486,aggregate_max_error=aggregate_check['max_abs_difference'])))

if __name__=='__main__':main()
