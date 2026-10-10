"""Read adopted F10 receipts; reuse accepted A10/C single-scene science."""
import ast
import copy
from collections import Counter
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
VIEWS = ('native', 'raw', 'shared_calibration')
FULL = R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def need(ok, message):
    if not ok: raise ValueError(message)

def write(name, value):
    with (H/name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')

def function(path, names, namespace):
    source = path.read_text(encoding='utf8'); tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == set(names), 'Original source function missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {n.name: hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest() for n in nodes}

def variant_module(path):
    """Only exact variant string constants change; no count, metric or statistic changes."""
    class Variant(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str) and n.value in ('minus_C','minus_C minus Full'):
                return ast.copy_location(ast.Constant(n.value.replace('minus_C','minus_F')), n)
            return n
    original = ast.parse(path.read_text(encoding='utf8'))
    mapped = Variant().visit(copy.deepcopy(original)); ast.fix_missing_locations(mapped)
    class Undo(ast.NodeTransformer):
        def visit_Constant(self, n):
            if isinstance(n.value, str) and n.value in ('minus_F','minus_F minus Full'):
                return ast.copy_location(ast.Constant(n.value.replace('minus_F','minus_C')), n)
            return n
    need(ast.dump(Undo().visit(copy.deepcopy(mapped))) == ast.dump(original), 'Unexpected scientific AST difference')
    ns = {}; exec(compile(mapped, str(path), 'exec'), ns)
    return SimpleNamespace(**ns)

def main():
    need(not sys.flags.optimize, 'Optimized Python forbidden')
    need(not (H/'records.json').exists() and not (H/'tables.json').exists(), 'Fresh output required; no overwrite/retry')
    from binding import load, IDS
    pins, adopted, index = load()
    ids = list(IDS)
    accepted = {r['id']:r for r in index['new_records']}
    need(len(accepted)==len(index['new_records'])==10 and list(accepted)==ids, 'Exact adopted F10 receipts required')
    native = read(R/index['native_inspection_path']); native_rows={r['id']:r for r in native['records']}
    need(sha(R/index['native_inspection_path'])==index['native_inspection_sha256']==adopted['native_inspection_sha256'], 'Native snapshot drift')
    scientific = dict(need=need,VIEWS=VIEWS,json=json,hashlib=hashlib)
    funcs = function(R/pins['receipt_identity_source'], ['receipt_identity','normalized'], scientific)
    funcs.update(function(R/pins['canonical_source'], ['canonical'], scientific))
    original_full = dict(need=need,FULL=FULL,sha=sha)
    funcs.update(function(R/pins['full_record_source'], ['full_record'], original_full))
    baseline = read(FULL/'records_three_views_900.json'); byid={r['id']:r for r in baseline['records']}
    original_refs=read(R/pins['original_full_reference_inventory'])['full_references']; refs={r['id']:r for r in original_refs}
    need(len(refs)==len(original_refs)==100, 'Original Full100 references incomplete')
    records=[]; fulls={}; links=[]
    for rid in ids:
        binding=index['new_bindings'][rid]; bp=index['new_binding_files'][rid]
        need(sha(bp['path'])==bp['sha256'] and read(bp['path'])==binding, 'Immutable binding changed')
        row=binding['record']; artifacts=index['new_artifacts'][rid]; actual={}
        for kind,pin in artifacts.items():
            need(sha(pin['path'])==pin['sha256'], 'Accepted artifact changed: '+rid+'/'+kind)
            actual[kind]=read(pin['path'])
        receipt=actual['scientific_receipt']; bridge=actual['bridge_receipt']; strict=actual['strict_json']; inventory=actual['bound_inventory']
        need(inventory['records']==[row] and inventory['full_references']==original_refs and inventory['native_tolerance']==1e-12, 'Bound inventory/Full source changed')
        need(row['id']==rid and row['variant']=='minus_F' and row['config']['ablation_component']=='F'
            and row['terminal_round']==70 and row['original_split']=='valid' and row['original_n_eval']==19867, 'Wrong variant/terminal/split')
        need(row['accepted_v4_row']==native_rows[rid] and row['checkpoint']['sha256']==binding['checkpoint_sha256'], 'Native/checkpoint source mismatch')
        scientific['receipt_identity'](receipt,row,SimpleNamespace(canonical=scientific['canonical']))
        need(strict['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['checkpoint_sha256']==row['checkpoint']['sha256'], 'Original strict acceptance differs')
        need(bridge['source_before']==bridge['source_after'] and bridge['artifact_before']==bridge['artifact_after']
            and bridge['native_tolerance']==1e-12 and bridge['scientific_body_receipt_sha256']==artifacts['scientific_receipt']['sha256'], 'Bridge source/artifact drift')
        need(receipt['views']==accepted[rid]['views'] and receipt['prediction_arrays_sha256']==accepted[rid]['prediction_arrays_sha256'], 'Adopted array/receipt views differ')
        fid=row['paired_full']['id']; full=original_full['full_record'](byid[fid],refs[fid])
        need(row['paired_full']==bridge['paired_full_reference'] and row['data_contract']==full['data_contract'], 'Paired root/train/valid/client partition mismatch')
        need(full['seed']==row['seed'] and full['distribution']==row['distribution']=='IID' and full['attack']==row['attack']=='Benign' and row['actual_alpha']==5000, 'Exact paired seed/IID Benign/alpha5000 required')
        fulls[fid]=full
        provenance=dict(root_adoption=pins['adoption'],root_adoption_sha256=pins['files'][pins['adoption']]['sha256'],accepted_index=pins['index'],
            binding=bp,artifacts=artifacts,prediction_arrays_sha256=receipt['prediction_arrays_sha256'],record_source_root=index['record_source_roots'][rid])
        records.append(scientific['normalized'](receipt,row,'minus_F',provenance))
        links.append(dict(id=rid,Full=fid,checkpoint_sha256=row['checkpoint']['sha256'],Full_checkpoint_sha256=full['checkpoint_sha256'],
            data_contract_canonical_sha256=scientific['canonical'](row['data_contract']),terminal_round=70,evaluation_split='valid'))
    records=[fulls[k] for k in sorted(fulls)]+records
    spec=importlib.util.spec_from_file_location('original_evidence_statistics',R/pins['evidence']);evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
    panels_module=variant_module(R/pins['C_panels']); numeric=variant_module(R/pins['C_numeric'])
    panels,coverage,paired=panels_module.panels(records,evidence);checks=numeric.verify(records,panels)
    need(checks['mean_sd_scalars']==162 and checks['receipt_metrics_from_group_counts']==180, 'Required arithmetic incomplete')
    displayed=[r for r in records if r['attack']=='Benign'];counts={v:dict(Counter(r['replay_runtime']['device'] for r in displayed if r['variant']==v)) for v in ('Full','minus_F')}
    torches={v:dict(Counter(r['training_torch'] for r in displayed if r['variant']==v)) for v in ('Full','minus_F')}
    text=['# CelebA Full–minus_F: IID Benign, three views','',
        'Validation only (19,867 images), round70; IID is frozen Dirichlet alpha5000. Ten paired model seeds; each model uses the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1); paired difference = minus_F − Full. ACC is percent; ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.','']
    display_checks=0
    for panel in panels:
        text += ['## '+panel['view']+' — '+panel['label'],'','| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            values=[f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in evidence.METRICS]
            text.append('| '+' | '.join([row['variant'],str(row['n']),*values])+' |');display_checks+=len(values)
        text.append('')
    need(display_checks==81, 'Incomplete displayed cells')
    text += ['AEOD = absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0); native retains the original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. Native/shared metrics and counts are compared below, not treated as independent evidence of calibration gain. No threshold is refitted by this builder.',
        f'Complete-scene replay devices: Full {counts["Full"]}; minus_F {counts["minus_F"]}. Training Torch: Full {torches["Full"]}; minus_F {torches["minus_F"]}. Historical/current driver provenance is retained per source record. The broader Full100 history includes98 cu128 and2 cu130 records; that fact does not turn this selected scene into a unified-device comparison.',
        'Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets.',
        'Only IID Benign seeds91001–91010 enter this one-scene table. No cross-scene aggregate is computed or published; the other nine minus_F scenes and other control coverages remain incomplete. This is not F100 or all mechanism controls. No significance, necessity, causal-isolation, final-test or whole-rebuttal-completion claim is made.','']
    checks.update(display_mean_sd_cells=81,table_record_count=20,preserved_records=20,paired_models=10,
        native_shared_metrics_and_counts_exact=all(r['views']['native']==r['views']['shared_calibration'] for r in records),
        original_full_record_and_receipt_identity_sources_unmodified=True,variant_only_AST_rebind=True,new_threshold_fits=0,new_CNN=0,new_training=0,test=False)
    write('records.json',dict(records=records));write('tables.json',dict(status='ACTUAL_ONE_F_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING',complete_scenes=1,paired_models=10,
        displayed_records=20,preserved_records=20,partial_other_scene_pairs=0,panels=panels,replay_devices=counts,training_torch=torches,
        primary_endpoint_selected=False,final_test=False,new_threshold_fits=0,new_inference=0,new_training=0))
    write('coverage.json',coverage);write('paired_per_seed.json',paired);write('checks.json',checks)
    write('SOURCE_BINDINGS.json',dict(input_pins=pins['files'],actual_F10_root_adoption=pins['adoption'],actual_F10_root_adoption_sha256=pins['files'][pins['adoption']]['sha256'],
        accepted_index=pins['index'],accepted_index_sha256=pins['files'][pins['index']]['sha256'],original_function_source_sha256=funcs,
        variant_change='Only exact minus_C/minus_C minus Full string constants become minus_F/minus_F minus Full; statistics/independent arithmetic AST unchanged',paired_identity_links=links))
    with (H/'TABLES.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text))
    print(json.dumps(dict(status='ACTUAL_F_BENIGN10_THREE_VIEW_TABLE_BUILT_ROOT_REVIEW_PENDING',records=20,displayed_models=20,stats=162,cells=81,metrics_from_counts=180,count_checks=480,replay_devices=counts,training_torch=torches)))

if __name__=='__main__': main()
