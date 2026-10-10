"""Read actual adopted A20 receipts; reuse the accepted A identity and C arithmetic."""
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
R = H.parents[4]
OLD = H.parent/'three_view_A_Benign10_20261010'
FULL = R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
VIEWS = ('native', 'raw', 'shared_calibration')
SCENES = [('IID', 'Benign'), ('IID', 'F Flip')]
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
    """Retain exact serialized object bytes and order for the old24 projection."""
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
    adopted = read(path(args.adoption)); new = read(path(args.index)); prior = read(path(pins['index']))
    new_ids = [f'minus_A_IID_F Flip_seed{s}' for s in range(91003,91011)]
    old_ids = [f'minus_A_IID_Benign_seed{s}' for s in range(91001,91011)] + [f'minus_A_IID_F Flip_seed{s}' for s in (91001,91002)]
    need(adopted['status']=='ROOT_A20_SAVED_ARRAYS_AND_NATIVE220_RESTORE_CHAIN_ADOPTED'
        and adopted['accepted_new_ids']==new['new_ids']==new_ids and adopted['new_accepted']==8
        and adopted['prior_accepted']==212 and adopted['cumulative_accepted']==220
        and adopted['original212_unchanged'] and adopted['native_max_abs_difference']==0, 'Actual exact8 root adoption required')
    need(adopted['Full_inference']==adopted['new_CNN']==adopted['new_training']==0 and adopted['test'] is False, 'Forbidden science operation')
    need(adopted['records_index_sha256']==args.index_sha256 and new['prior_index_path']==pins['index']
        and new['prior_index_sha256']==input_files[pins['index']]['sha256']
        and new['prior_adoption_path']==pins['adoption'] and new['prior_adoption_sha256']==input_files[pins['adoption']]['sha256'], 'Broken actual212→220 chain')
    need(new['all_ids']==prior['all_ids']+new_ids and len(set(new['all_ids']))==220, 'Original212 ID prefix or new8 changed')
    prior_adopted = read(path(pins['adoption']))
    need(prior_adopted['status']=='ROOT_A12_SAVED_ARRAYS_AND_NATIVE212_RESTORE_CHAIN_ADOPTED'
        and prior_adopted['accepted_new_ids']==prior['new_ids']==old_ids and prior_adopted['original200_unchanged'], 'Original A12 adoption changed')
    for name in ['build.py','ROOT_VERIFICATION.json','FILES_SHA256.json','records.json','tables.json','TABLES.md']:
        p = OLD/name; input_files[p.relative_to(R).as_posix()] = dict(sha256=sha(p),bytes=p.stat().st_size)
    need(sha(OLD/'ROOT_VERIFICATION.json')=='c3d65134f0e6eeda36fd37c64b6ac0df3794828af054d920d46775670a54df9d', 'Original table adoption changed')
    need(sha(OLD/'FILES_SHA256.json')=='c96af519a0fc8ef3eff780f2990c89d40022b6fc1eba25cc73675e3079fcdbb5', 'Original table seal changed')
    for name,pin in read(OLD/'FILES_SHA256.json')['files'].items():
        need(sha(OLD/name)==pin['sha256'] and (OLD/name).stat().st_size==pin['bytes'], 'Original sealed A table drift: '+name)
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
    for index, ids, adoption_path, index_path in [(prior,old_ids,pins['adoption'],pins['index']),(new,new_ids,args.adoption,args.index)]:
        need(set(index['new_bindings'])==set(index['new_binding_files'])==set(index['new_artifacts'])==set(ids), 'Exact batch bindings required')
        accepted={r['id']:r for r in index['new_records']}
        need(len(accepted)==len(index['new_records'])==len(ids) and set(accepted)==set(ids), 'Missing/duplicate adopted records')
        native_path=index['native_inspection_path']; native=read(path(native_path))
        need(sha(path(native_path))==index['native_inspection_sha256'], 'Native snapshot changed')
        input_files[native_path]=dict(sha256=sha(path(native_path)),bytes=path(native_path).stat().st_size)
        if index is new: need(index['native_inspection_sha256']==adopted['native_inspection_sha256'], 'Newest native snapshot differs from root proof')
        local_pins=dict(adoption=adoption_path,index=index_path,files=input_files)
        shared.update(index=index,ids=ids,accepted=accepted,native_rows={r['id']:r for r in native['records']},pins=local_pins)
        exec(loop_code,shared)
    records=[fulls[k] for k in sorted(fulls)]+records
    need(len(records)==len({r['id'] for r in records})==40 and len(links)==20, 'Exactly20 A +20 paired Full required')
    changes={}
    panels_module=scoped_module(old,pins['C_panels'],[("expected=[('IID','Benign')]","expected=[('IID','Benign'),('IID','F Flip')]"),
        ('Only the exact C IID Benign ten-shared-seed scene is publishable','Only the exact two A IID ten-shared-seed scenes are publishable')],changes)
    numeric=scoped_module(old,pins['C_numeric'],[
        ("assert len(records)==24 and len({r['id'] for r in records})==24","assert len(records)==40 and len({r['id'] for r in records})==40"),
        ('assert len(bycell)==24','assert len(bycell)==40'),("scenes={('IID','Benign')}","scenes={('IID','Benign'),('IID','F Flip')}"),
        ("assert len(partial)==4 and {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in partial}=={(v,'IID','F Flip',s) for v in ['Full','minus_C'] for s in [91001,91002]}",'assert not partial'),
        ('assert len(rows)==3','assert len(rows)==6'),('assert len(errors)==162','assert len(errors)==324'),
        ('assert metric_checks==216 and count_checks==576','assert metric_checks==360 and count_checks==960'),("'partial_pairs_excluded':2","'partial_pairs_excluded':0")],changes)
    spec=importlib.util.spec_from_file_location('accepted_original_evidence_statistics',path(pins['evidence'])); evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
    panels,coverage,paired=panels_module.panels(records,evidence); checks=numeric.verify(records,panels)
    old_records=read(OLD/'records.json')['records']; old_ids_set={r['id'] for r in old_records}
    old_projection=[r for r in records if r['id'] in old_ids_set]
    need(old_projection==old_records, 'Old24 records/order changed')
    serialized=json.dumps(dict(records=records),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    need([x for x in record_fragments(serialized) if x[0] in old_ids_set]==record_fragments((OLD/'records.json').read_text(encoding='utf8')), 'Old24 serialized object bytes/order changed')
    old_panels=read(OLD/'tables.json')['panels']; old_cells=0
    for current,previous in zip(panels,old_panels):
        need({k:current[k] for k in ('view','label','seeds')}=={k:previous[k] for k in ('view','label','seeds')}, 'Old seed panel changed')
        need([r for r in current['rows'] if r['attack']=='Benign']==previous['rows'], 'Old Benign162 statistics changed')
        for row in previous['rows']:
            for metric in evidence.METRICS:
                digits=3 if metric=='accuracy_pct' else 5
                display=f"{row[metric]['mean']:.{digits}f} ± {row[metric]['sample_sd_ddof1']:.{digits}f}"
                need(display in (OLD/'TABLES.md').read_text(encoding='utf8'), 'Original display cell mismatch'); old_cells+=1
    need(old_cells==81 and checks['mean_sd_scalars']==324, 'Required original/new checks incomplete')
    counts={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_A')}
    torches={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_A')}
    text=['# CelebA Full–minus_A: two complete IID scenes, three views','',
        'IID Benign and IID F Flip only. Validation (19,867 images), round70; ten paired model seeds per scene, with the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1). Paired difference = minus_A − Full; ACC is percent and ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.','']
    cells=0
    for panel in panels:
        text += ['## '+panel['view']+' — '+panel['label'],'','| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |','|---|---:|---:|---:|---:|']
        for row in panel['rows']:
            values=[f"{row[m]['mean']:.{3 if m=='accuracy_pct' else 5}f} ± {row[m]['sample_sd_ddof1']:.{3 if m=='accuracy_pct' else 5}f}" for m in evidence.METRICS]
            text.append('| '+' | '.join([row['distribution']+' / '+row['attack']+' / '+row['variant'],str(row['n']),*values])+' |'); cells+=len(values)
        text.append('')
    need(cells==162, 'Display cell count incomplete')
    text += ['AEOD is the absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0). Native retains each original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. The three views are parallel descriptions, with no endpoint selected or threshold refitted by this builder.',
        f'Actual replay devices across20 pairs: Full {counts["Full"]}; minus_A {counts["minus_A"]}. Training Torch: Full {torches["Full"]}; minus_A {torches["minus_A"]}. Per-record configuration, source, checkpoint, environment and driver provenance remain in records.json. Broader Full100 history includes98 cu128 and2 cu130 records; these20 actual source records, not that broader count, define this table.',
        'Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets. No final test was run for this table.',
        'All negative and constant outcomes are retained. These are two of ten minus_A scenes; the other eight scenes remain outside this delivery. No scene-pooled mean is reported, and scenes are not treated as independent model seeds. This is not A100 or completion of all mechanism controls. No significance, necessity, causal-isolation or whole-rebuttal-completion claim is made.','']
    checks.update(display_mean_sd_cells=cells,table_record_count=40,preserved_records=40,paired_models=20,complete_scenes=2,
        old24_record_JSON_bytes_and_order_exact=True,old162_scalars_exact=True,old81_display_cells_exact=True,
        native_shared_metrics_and_counts_exact=all(r['views']['native']==r['views']['shared_calibration'] for r in records),
        original_per_record_scientific_loop_source_sha256=loop_sha,scope_only_reversible_adapter=True,new_threshold_fits=0,new_CNN=0,new_training=0,test=False)
    write('INPUTS.json',dict(original_A_inputs=pins,adoption=args.adoption,index=args.index,files=input_files))
    with (H/'records.json').open('x',encoding='utf8',newline='\n') as f:f.write(serialized)
    write('tables.json',dict(status='ACTUAL_TWO_A_IID_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING',complete_scenes=2,paired_models=20,
        displayed_records=40,preserved_records=40,partial_F_Flip_pairs=0,panels=panels,replay_devices=counts,training_torch=torches,
        primary_endpoint_selected=False,final_test=False,new_threshold_fits=0,new_inference=0,new_training=0))
    write('coverage.json',coverage);write('paired_per_seed.json',paired);write('checks.json',checks)
    write('SOURCE_BINDINGS.json',dict(input_pins=input_files,actual_A20_root_adoption=args.adoption,actual_A20_root_adoption_sha256=args.adoption_sha256,
        accepted_index=args.index,accepted_index_sha256=args.index_sha256,original_function_source_sha256=funcs,
        original_per_record_scientific_loop_source_sha256=loop_sha,scope_only_reversible_rebindings=changes,
        original_variant_AST_rebind='Exact minus_C/minus_C minus Full string constants only become minus_A/minus_A minus Full',
        original24_provenance_kept=True,paired_identity_links=links,fresh_F_volume=storage,
        limits=dict(new_CNN=0,new_training=0,new_fit=0,new_predictions=0,test=False,root_table_adopted=False,complete_scenes=SCENES,remaining_A_scenes=8)))
    with (H/'TABLES.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(text))
    print(json.dumps(dict(status='ACTUAL_A20_TWO_IID_SCENE_THREE_VIEW_TABLE_BUILT_ROOT_REVIEW_PENDING',records=40,pairs=20,stats=324,cells=162,metrics=360,counts=960,max_abs_difference=checks['max_abs_difference'],replay_devices=counts,training_torch=torches)))

if __name__=='__main__': main()
