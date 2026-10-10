"""Read actual adopted A51 receipts; display only complete IID A50; preserve original identity loop and arithmetic."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
OLD = R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_Benign10_20261010'
PREV = R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_four_scenes40_20261010'
FULL = R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
VIEWS = ('native', 'raw', 'shared_calibration')
SCENES = [('IID', 'Benign'), ('IID', 'F Flip'), ('IID', 'FedSA'), ('IID', 'S-DFA'), ('IID', 'Sp-DFA')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def need(ok, message):
    if not ok: raise ValueError(message)

def path(p):
    p = Path(p)
    return p if p.is_absolute() else R/p

def write(name, value):
    with (H/name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def record_fragments(text):
    """Retain exact serialized object bytes and order for the old80 projection."""
    pos = text.index('[', text.index('"records"')) + 1
    decoder = json.JSONDecoder(); result = []
    while True:
        while text[pos] in ' \t\r\n,': pos += 1
        if text[pos] == ']': return result
        record, end = decoder.raw_decode(text, pos)
        result.append((record['id'], text[pos:end].encode('utf8'))); pos = end

def scoped_module(old, source_path, replacements, changes):
    original = path(source_path).read_text(encoding='utf8'); mapped = original
    for before, after in replacements:
        need(mapped.count(before) == 1, 'Scope adapter must match exactly once: '+before)
        mapped = mapped.replace(before, after)
    undone = mapped
    for before, after in reversed(replacements): undone = undone.replace(after, before, 1)
    need(undone == original, 'Scope adapter changed scientific source')
    changes[source_path] = [dict(before=a, after=b) for a,b in replacements]
    class Reader:
        def read_text(self, **kwargs): return mapped
        def __str__(self): return source_path+' [scope-only in-memory adapter]'
    return old['variant_module'](Reader())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adoption', required=True); parser.add_argument('--adoption-sha256', required=True)
    parser.add_argument('--index', required=True); parser.add_argument('--index-sha256', required=True)
    args = parser.parse_args()
    need(not sys.flags.optimize, 'Optimized Python forbidden')
    need(not (H/'records.json').exists() and not (H/'tables.json').exists(), 'Fresh output required; no overwrite/retry')
    need(sha(OLD/'build.py')=='8969ce66af6cdaea912e89a7bc2df7823f862153b60b5b67fd1a5f2f3845cede', 'Original A source changed')
    old = runpy.run_path(str(OLD/'build.py'), run_name='accepted_A_reader_only')
    pins = read(OLD/'INPUTS.json'); input_files = dict(pins['files'])
    for name, pin in input_files.items():
        need(sha(path(name))==pin['sha256'] and path(name).stat().st_size==pin['bytes'], 'Original input/source drift: '+name)
    for name, expected in [(args.adoption,args.adoption_sha256),(args.index,args.index_sha256)]:
        need(sha(path(name))==expected, 'Actual external closure SHA mismatch: '+name)
        input_files[name] = dict(sha256=expected, bytes=path(name).stat().st_size)
    adopted = read(path(args.adoption)); new = read(path(args.index))
    new_ids = [f'minus_A_IID_Sp-DFA_seed{seed}' for seed in range(91001,91011)]+['minus_A_non-IID_Benign_seed91001']
    need(adopted['status']=='ROOT_AFTER240_EXACT11_SAVED_ARRAYS_NATIVE251_ADOPTED'
        and adopted['accepted_new_ids']==new['new_ids']==new_ids
        and adopted['new_accepted']==11 and adopted['prior_accepted']==240
        and adopted['cumulative_accepted']==251 and adopted['original240_unchanged']
        and adopted['native_max_abs_difference']==0, 'Actual exact11 root adoption required')
    need(adopted['Full_inference']==adopted['new_CNN']==adopted['new_training']==0
        and adopted.get('new_fit',0)==0 and adopted['test'] is False, 'Forbidden science operation')
    need(adopted['records_index_sha256']==args.index_sha256, 'Newest root/index binding differs')
    # Follow six SHA-bound adopted increments; stop at the original A12 index.
    # Every leaf is independently root-adopted; no runtime status label counts.
    batches=[]; current=(args.adoption,args.adoption_sha256,args.index,args.index_sha256)
    visited=set()
    while True:
        ap,asha,ip,isha=current
        ap=str(ap).replace('\\','/'); ip=str(ip).replace('\\','/')  # Preserve existing record provenance path spelling.
        need(ip not in visited and len(visited)<6, 'Cycle or unexpected A-chain depth')
        visited.add(ip)
        for name,expected in [(ap,asha),(ip,isha)]:
            need(sha(path(name))==expected, 'Adopted chain pin changed: '+name)
            input_files[name]=dict(sha256=expected,bytes=path(name).stat().st_size)
        root=read(path(ap)); index=read(path(ip)); ids=index['new_ids']
        need(root['records_index_sha256']==isha and root['accepted_new_ids']==ids
            and root['new_accepted']==len(ids) and root['cumulative_accepted']==len(index['all_ids'])
            and root['native_max_abs_difference']==0 and root['Full_inference']==root['new_CNN']==root['new_training']==0
            and root.get('new_fit',0)==0 and root['test'] is False, 'Unadopted/incompatible chain leaf')
        need(root['status'].startswith('ROOT_A') and root['status'].endswith('_ADOPTED'), 'Not a root adoption')
        batches.append((index,ids,ap,ip,root))
        if ip==pins['index']:
            need(ap==pins['adoption'] and isha==input_files[pins['index']]['sha256']
                and asha==input_files[pins['adoption']]['sha256'], 'Original A12 anchor changed')
            break
        prior=read(path(index['prior_index_path']))
        need(sha(path(index['prior_index_path']))==index['prior_index_sha256']
            and index['all_ids']==prior['all_ids']+ids and len(set(index['all_ids']))==len(index['all_ids'])
            and root['prior_accepted']==len(prior['all_ids']), 'Broken adopted prefix')
        current=(index['prior_adoption_path'],index['prior_adoption_sha256'],
                 index['prior_index_path'],index['prior_index_sha256'])
    batches.reverse()
    need([len(x[1]) for x in batches]==[12,8,8,8,4,11]
        and [len(x[0]['all_ids']) for x in batches]==[212,220,228,236,240,251], 'Wrong exact six A increments')
    wanted=[f'minus_A_{distribution}_{attack}_seed{seed}' for distribution,attack in SCENES for seed in range(91001,91011)]
    need([rid for _,ids,_,_,_ in batches for rid in ids if rid in set(wanted)]==wanted, 'Five complete IID scenes required')
    need(sha(PREV/'ROOT_VERIFICATION.json')=='edf4a520c3b797e2376005ef132bc21021ab5dbd0962b4105299d67309f8ebf9', 'Original A40 root adoption changed')
    for name in ['build.py','ROOT_VERIFICATION.json','FILES_SHA256.json','records.json','tables.json','TABLES.md']:
        p=PREV/name; input_files[p.relative_to(R).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
    for name,pin in read(PREV/'FILES_SHA256.json')['files'].items():
        need(sha(PREV/name)==pin['sha256'] and (PREV/name).stat().st_size==pin['bytes'], 'Original A40 sealed file drift: '+name)
    storage = runpy.run_path(str(R/'tmp/guardfed_local_storage.py'))['check_bulk_storage'](0)
    scientific = dict(need=need,VIEWS=VIEWS,json=json,hashlib=hashlib)
    funcs = old['function'](path(pins['receipt_identity_source']), ['receipt_identity','normalized'], scientific)
    funcs.update(old['function'](path(pins['canonical_source']), ['canonical'], scientific))
    original_full = dict(need=need,FULL=FULL,sha=sha)
    funcs.update(old['function'](path(pins['full_record_source']), ['full_record'], original_full))
    baseline = read(FULL/'records_three_views_900.json'); byid={r['id']:r for r in baseline['records']}
    original_refs=read(path(pins['original_full_reference_inventory']))['full_references']; refs={r['id']:r for r in original_refs}
    need(len(refs)==len(original_refs)==100, 'Original Full100 references incomplete')
    source = (OLD/'build.py').read_text(encoding='utf8'); main_ast = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop = next(n for n in main_ast.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='rid')
    loop_sha = hashlib.sha256(ast.get_source_segment(source,loop).encode()).hexdigest()
    loop_code = compile(ast.Module(body=[loop],type_ignores=[]),str(OLD/'build.py'),'exec')
    records=[]; fulls={}; links=[]
    shared=dict(need=need,sha=sha,read=read,SimpleNamespace=SimpleNamespace,scientific=scientific,
        original_full=original_full,byid=byid,original_refs=original_refs,refs=refs,records=records,fulls=fulls,links=links)
    for index, ids, adoption_path, index_path, batch_root in batches:
        need(set(index['new_bindings'])==set(index['new_binding_files'])==set(index['new_artifacts'])==set(ids), 'Exact batch bindings required')
        accepted={r['id']:r for r in index['new_records']}
        need(len(accepted)==len(index['new_records'])==len(ids) and set(accepted)==set(ids), 'Missing/duplicate adopted records')
        native_path=index['native_inspection_path']; native=read(path(native_path))
        need(sha(path(native_path))==index['native_inspection_sha256'], 'Native snapshot changed')
        input_files[native_path]=dict(sha256=sha(path(native_path)),bytes=path(native_path).stat().st_size)
        need(index['native_inspection_sha256']==batch_root['native_inspection_sha256'], 'Native snapshot differs from root proof')
        local_pins=dict(adoption=adoption_path,index=index_path,files=input_files)
        shared.update(index=index,ids=[rid for rid in ids if rid in set(wanted)],accepted=accepted,native_rows={r['id']:r for r in native['records']},pins=local_pins)
        exec(loop_code,shared)
    records=[fulls[k] for k in sorted(fulls)]+records
    need(len(records)==len({r['id'] for r in records})==100 and len(links)==50, 'Exactly50 A +50 paired Full required')
    changes={}
    panels_module=scoped_module(old,pins['C_panels'],[("expected=[('IID','Benign')]","expected=[('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA')]"),
        ('Only the exact C IID Benign ten-shared-seed scene is publishable','Only the exact five A IID ten-shared-seed scenes are publishable')],changes)
    numeric=scoped_module(old,pins['C_numeric'],[
        ("assert len(records)==24 and len({r['id'] for r in records})==24","assert len(records)==100 and len({r['id'] for r in records})==100"),
        ('assert len(bycell)==24','assert len(bycell)==100'),("scenes={('IID','Benign')}","scenes={('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA')}"),
        ("assert len(partial)==4 and {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in partial}=={(v,'IID','F Flip',s) for v in ['Full','minus_C'] for s in [91001,91002]}",'assert not partial'),
        ('assert len(rows)==3','assert len(rows)==15'),('assert len(errors)==162','assert len(errors)==810'),
        ('assert metric_checks==216 and count_checks==576','assert metric_checks==900 and count_checks==2400'),("'partial_pairs_excluded':2","'partial_pairs_excluded':0")],changes)
    spec=importlib.util.spec_from_file_location('accepted_original_evidence_statistics',path(pins['evidence'])); evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
    panels,coverage,paired=panels_module.panels(records,evidence); checks=numeric.verify(records,panels)
    old_records=read(PREV/'records.json')['records']; old_ids_set={r['id'] for r in old_records}
    old_projection=[r for r in records if r['id'] in old_ids_set]
    need(old_projection==old_records, 'Old80 records/order changed')
    serialized=json.dumps(dict(records=records),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    need([x for x in record_fragments(serialized) if x[0] in old_ids_set]==record_fragments((PREV/'records.json').read_text(encoding='utf8')), 'Old80 serialized object bytes/order changed')
    old_panels=read(PREV/'tables.json')['panels']; old_cells=0
    for current,previous in zip(panels,old_panels):
        need({k:current[k] for k in ('view','label','seeds')}=={k:previous[k] for k in ('view','label','seeds')}, 'Old seed panel changed')
        need([r for r in current['rows'] if r['attack'] in ('Benign','F Flip','FedSA','S-DFA')]==previous['rows'], 'Old A40 648 statistics changed')
        for row in previous['rows']:
            for metric in evidence.METRICS:
                digits=3 if metric=='accuracy_pct' else 5
                display=f"{row[metric]['mean']:.{digits}f} ± {row[metric]['sample_sd_ddof1']:.{digits}f}"
                need(display in (PREV/'TABLES.md').read_text(encoding='utf8'), 'Original display cell mismatch'); old_cells+=1
    need(old_cells==324 and checks['mean_sd_scalars']==810, 'Required original/new checks incomplete')
    counts={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_A')}
    torches={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_A')}
    text=['# CelebA Full–minus_A: five complete IID scenes, three views','',
        'IID Benign, F Flip, FedSA, S-DFA and Sp-DFA only. Validation (19,867 images), round70; ten paired model seeds per scene, with the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1). Paired difference = minus_A − Full; ACC is percent and ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.','']
    cells=0
    for panel in panels:
        text += ['## '+panel['view']+' — '+panel['label'],'','| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            values=[f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in evidence.METRICS]
            text.append('| '+' | '.join([row['distribution']+' / '+row['attack']+' / '+row['variant'],str(row['n']),*values])+' |'); cells+=len(values)
        text.append('')
    need(cells==405, 'Display cell count incomplete')
    text += ['AEOD is the absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0). Native retains each original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. The three views are parallel descriptions, with no endpoint selected or threshold refitted by this builder.',
        f'Actual replay devices across50 pairs: Full {counts["Full"]}; minus_A {counts["minus_A"]}. Training Torch: Full {torches["Full"]}; minus_A {torches["minus_A"]}. Per-record configuration, source, checkpoint, environment and driver provenance remain in records.json. Broader Full100 history includes98 cu128 and2 cu130 records; these50 actual source records, not that broader count, define this table.',
        'Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets. No final test was run for this table.',
        'All negative and constant outcomes are retained. These are all five IID minus_A scenes; all five non-IID scenes remain outside this delivery (one non-IID Benign seed is retained in the accepted source index only). The separate five-IID-scene aggregate first averages within each model seed, and scenes are not treated as independent model seeds. This is not A100 or completion of all mechanism controls. No significance, necessity, causal-isolation or whole-rebuttal-completion claim is made.','']
    checks.update(display_mean_sd_cells=cells,table_record_count=100,preserved_records=100,paired_models=50,complete_scenes=5,
        old80_record_JSON_bytes_and_order_exact=True,old648_scalars_exact=True,old324_display_cells_exact=True,
        native_shared_metrics_and_counts_exact=all(r['views']['native']==r['views']['shared_calibration'] for r in records),
        original_per_record_scientific_loop_source_sha256=loop_sha,scope_only_reversible_adapter=True,new_threshold_fits=0,new_CNN=0,new_training=0,test=False)
    write('INPUTS.json',dict(original_A_inputs=pins,adoption=args.adoption,index=args.index,files=input_files))
    with (H/'records.json').open('x',encoding='utf8',newline='\n') as f:f.write(serialized)
    write('tables.json',dict(status='ACTUAL_FIVE_A_IID_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING',complete_scenes=5,paired_models=50,
        displayed_records=100,preserved_records=100,partial_F_Flip_pairs=0,panels=panels,replay_devices=counts,training_torch=torches,
        primary_endpoint_selected=False,final_test=False,new_threshold_fits=0,new_inference=0,new_training=0))
    write('coverage.json',coverage);write('paired_per_seed.json',paired);write('checks.json',checks)
    write('SOURCE_BINDINGS.json',dict(input_pins=input_files,actual_A51_root_adoption=args.adoption,actual_A51_root_adoption_sha256=args.adoption_sha256,
        accepted_index=args.index,accepted_index_sha256=args.index_sha256,original_function_source_sha256=funcs,
        original_per_record_scientific_loop_source_sha256=loop_sha,scope_only_reversible_rebindings=changes,
        original_variant_AST_rebind='Exact minus_C/minus_C minus Full string constants only become minus_A/minus_A minus Full',
        original80_provenance_kept=True,paired_identity_links=links,fresh_F_volume=storage,
        limits=dict(new_CNN=0,new_training=0,new_fit=0,new_predictions=0,test=False,root_table_adopted=False,complete_scenes=SCENES,remaining_A_scenes=5,excluded_partial_id="minus_A_non-IID_Benign_seed91001")))
    with (H/'TABLES.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text))
    print(json.dumps(dict(status='ACTUAL_A50_FIVE_IID_SCENE_THREE_VIEW_TABLE_BUILT_ROOT_REVIEW_PENDING',records=100,pairs=50,stats=810,cells=405,metrics=900,counts=2400,max_abs_difference=checks['max_abs_difference'],replay_devices=counts,training_torch=torches)))

if __name__=='__main__': main()
