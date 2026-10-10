"""Create reversible source mappings and compile them; never run scientific code."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import runpy

H = Path(__file__).resolve().parent
R = H.parents[1]
A90 = R/'tmp/celeba_mechanism_A90_candidate_20261011'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(name, value):
    with (H/name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert not (H/'SOURCE_ADAPTATIONS.json').exists(), 'No overwrite/retry'
    source = runpy.run_path(str(A90/'source_adapter.py'))['mapped']
    scenes90 = [('IID',a) for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]+[('non-IID',a) for a in ('Benign','F Flip','FedSA','S-DFA')]
    scenes100 = scenes90+[('non-IID','Sp-DFA')]
    compact = lambda xs: '['+','.join("('%s','%s')"%x for x in xs)+']'
    old_filter = "r['distribution']=='IID' or (r['distribution']=='non-IID' and r['attack'] in ('Benign','F Flip','FedSA'))"
    new_filter = "r['distribution']=='IID' or (r['distribution']=='non-IID' and r['attack'] in ('Benign','F Flip','FedSA','S-DFA'))"
    specs = {'build.py': [
        ('Read actual adopted A90 receipts; display five IID plus four non-IID scenes;', 'Read actual adopted A100 receipts; display all ten IID/non-IID scenes;'),
        ('three_view_A_eight_scenes80_20261011', 'three_view_A_nine_scenes90_20261011'),
        ('SCENES = '+repr(scenes90), 'SCENES = '+repr(scenes100)),
        ("new_ids = [f'minus_A_non-IID_S-DFA_seed{seed}' for seed in (91009,91010)]+[f'minus_A_non-IID_Sp-DFA_seed{seed}' for seed in range(91001,91006)]", "new_ids = [f'minus_A_non-IID_Sp-DFA_seed{seed}' for seed in range(91006,91011)]"),
        ("and adopted['new_accepted']==7 and adopted['prior_accepted']==288", "and adopted['new_accepted']==5 and adopted['prior_accepted']==295"),
        ("and adopted['cumulative_accepted']==295 and adopted['original288_unchanged']", "and adopted['cumulative_accepted']==300 and adopted['original295_unchanged']"),
        ('Actual exact7 root adoption required', 'Actual exact5 root adoption required'),
        ('Follow ten SHA-bound adopted increments', 'Follow eleven SHA-bound adopted increments'),
        ('len(visited)<10', 'len(visited)<11'),
        ('[12,8,8,8,4,11,9,20,8,7]', '[12,8,8,8,4,11,9,20,8,7,5]'),
        ('[212,220,228,236,240,251,260,280,288,295]', '[212,220,228,236,240,251,260,280,288,295,300]'),
        ('Wrong exact ten A increments', 'Wrong exact eleven A increments'),
        ('Five complete IID scenes plus non-IID Benign/F Flip/FedSA/S-DFA required', 'All five IID plus five non-IID scenes required'),
        ('d2245e4d9de68b415931dccc7d1f70c39a871c226c59d9973675dfc1d3cc1bc7', '445904a761cad8de89b57a9e0dd65fab298dbd097d7a0bd5f3d75458a6f65cdd'),
        ('Original A80 root adoption changed', 'Original A90 root adoption changed'),
        ('Original root-adopted A80 file drift:', 'Original root-adopted A90 file drift:'),
        ("==180 and len(links)==90, 'Exactly90 A +90 paired Full required'", "==200 and len(links)==100, 'Exactly100 A +100 paired Full required'"),
        ('expected='+compact(scenes90), 'expected='+compact(scenes100)),
        ('Only five A IID scenes plus non-IID Benign/F Flip/FedSA/S-DFA with ten shared seeds are publishable', 'Only all ten A IID/non-IID scenes with ten shared seeds are publishable'),
        ("assert len(records)==180 and len({r['id'] for r in records})==180", "assert len(records)==200 and len({r['id'] for r in records})==200"),
        ('assert len(bycell)==180', 'assert len(bycell)==200'),
        ('scenes={'+compact(scenes90)[1:-1]+'}', 'scenes={'+compact(scenes100)[1:-1]+'}'),
        ('assert len(rows)==27', 'assert len(rows)==30'),
        ('assert len(errors)==1458', 'assert len(errors)==1620'),
        ('assert metric_checks==1620 and count_checks==4320', 'assert metric_checks==1800 and count_checks==4800'),
        ('Old160 records/order changed', 'Old180 records/order changed'),
        ('Old160 serialized object bytes/order changed', 'Old180 serialized object bytes/order changed'),
        (old_filter, new_filter),
        ('Old A80 1296 per-scene statistics changed', 'Old A90 1458 per-scene statistics changed'),
        ("old_cells==648 and checks['mean_sd_scalars']==1458", "old_cells==729 and checks['mean_sd_scalars']==1620"),
        ('# CelebA Full–minus_A: five IID plus four non-IID scenes, three views', '# CelebA Full–minus_A: all ten IID/non-IID scenes, three views'),
        ('IID Benign, F Flip, FedSA, S-DFA and Sp-DFA plus non-IID Benign, F Flip, FedSA and S-DFA only; non-IID Sp-DFA remains incomplete (five accepted seeds, excluded here).', 'IID and non-IID Benign, F Flip, FedSA, S-DFA and Sp-DFA: all ten scenes, 100 matched Full/minus_A pairs.'),
        ("need(cells==729, 'Display cell count incomplete')", "need(cells==810, 'Display cell count incomplete')"),
        ('Actual replay devices across90 pairs:', 'Actual replay devices across100 pairs:'),
        ('these90 actual source records', 'these100 actual source records'),
        ('These are all five IID minus_A scenes plus complete ten-seed non-IID Benign, F Flip, FedSA and S-DFA; non-IID Sp-DFA remains incomplete and outside this delivery. No mixed-distribution nine-scene aggregate is reported. The separate five-IID-scene aggregate first averages within each model seed, and scenes are not treated as independent model seeds. This is not A100 or completion of all mechanism controls.', 'These are all ten minus_A scenes. Separate five-IID, five-non-IID and balanced-ten-scene summaries first equally average scenes within each model seed, then summarize seeds; scenes are never treated as independent model seeds. This completes A100 coverage only, not the other five image-control variants or all800 controls.'),
        ('table_record_count=180,preserved_records=180,paired_models=90,complete_scenes=9', 'table_record_count=200,preserved_records=200,paired_models=100,complete_scenes=10'),
        ('old160_record_JSON_bytes_and_order_exact=True,old1296_scalars_exact=True,old648_display_cells_exact=True', 'old180_record_JSON_bytes_and_order_exact=True,old1458_scalars_exact=True,old729_display_cells_exact=True'),
        ("status='ACTUAL_A90_FIVE_IID_PLUS_FOUR_NONIID_CANDIDATE_ROOT_REVIEW_PENDING',complete_scenes=9,paired_models=90", "status='ACTUAL_A100_TEN_SCENE_CANDIDATE_ROOT_REVIEW_PENDING',complete_scenes=10,paired_models=100"),
        ('displayed_records=180,preserved_records=180', 'displayed_records=200,preserved_records=200'),
        ('actual_A90_root_adoption=args.adoption,actual_A90_root_adoption_sha256=args.adoption_sha256', 'actual_A100_root_adoption=args.adoption,actual_A100_root_adoption_sha256=args.adoption_sha256'),
        ('original160_provenance_kept=True', 'original180_provenance_kept=True'),
        ("remaining_A_scenes=1,excluded_partial_id=None,accepted_partial_ids_retained_upstream=[f'minus_A_non-IID_Sp-DFA_seed{s}' for s in range(91001,91006)]", "remaining_A_scenes=0,excluded_partial_id=None,accepted_partial_ids_retained_upstream=[]"),
        ("status='ACTUAL_A90_NINE_SCENE_THREE_VIEW_TABLE_BUILT_ROOT_REVIEW_PENDING',records=180,pairs=90,stats=1458,cells=729,metrics=1620,counts=4320", "status='ACTUAL_A100_TEN_SCENE_THREE_VIEW_TABLE_BUILT_ROOT_REVIEW_PENDING',records=200,pairs=100,stats=1620,cells=810,metrics=1800,counts=4800")
    ], 'finish.py': [
        ("need(len(records)==180, 'Exactly nine paired scenes required')", "need(len(records)==200, 'Exactly ten paired scenes required')"),
        ("for a in ('Benign','F Flip','FedSA','S-DFA')), 'Only original five IID scenes plus non-IID Benign/F Flip/FedSA/S-DFA'", "for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')), 'Only five complete IID plus five complete non-IID scenes'"),
        ('Original A80 IID aggregate statistics changed', 'Original A90 IID aggregate statistics changed'),
        ("    # Both functions are original source; variant_module only changes exact variant string constants.", "    # Same distribution projection and original calls as accepted C100 assemble.py.\n    projected=[dict(r,distribution='IID') for r in records if r['distribution']=='non-IID']\n    noniid=panels.aggregate_panels(projected,evidence)\n    noniid_check=numeric.verify_aggregate(projected,noniid)\n    noniid=[dict(p,rows=[dict(r,distribution='non-IID') for r in p['rows']]) for p in noniid]\n    balanced=panels.aggregate_balanced_panels(records,evidence)\n    balanced_check=numeric.verify_balanced_aggregate(records,balanced)\n    # Original functions; variant_module only changes exact variant string constants."),
        ("[(panel_path,['aggregate_panels']), (check_path,['verify_aggregate'])]", "[(panel_path,['aggregate_panels','aggregate_balanced_panels']), (check_path,['verify_aggregate','verify_balanced_aggregate'])]"),
        ("    checks = read(H/'checks.json')", "    put('CROSS_SCENE_ADDITIONAL.json',dict(nonIID_five_scene_panels=noniid,balanced_ten_scene_panels=balanced,n_is_seed_count=True,scene_mean_before_seed_statistics=True))\n    checks = read(H/'checks.json')"),
        ("total_statistic_scalars=checks['mean_sd_scalars']+aggregate_check['mean_sd_scalars']", "total_statistic_scalars=checks['mean_sd_scalars']+aggregate_check['mean_sd_scalars']+noniid_check['mean_sd_scalars']+balanced_check['mean_sd_scalars']"),
        ('scene_display_cells=729,aggregate_display_cells=81', 'scene_display_cells=810,aggregate_display_cells=243,nonIID_seed_first=noniid_check,balanced_seed_first=balanced_check'),
        ('This is not a balanced ten-scene or non-IID aggregate.', 'The separate five-non-IID and balanced-ten-scene summaries below use the same seed-first rule.'),
        ("    for panel in aggregate:\n        text +=", "    aggregate_groups=[('Five IID',aggregate),('Five non-IID',noniid),('Balanced ten-scene',balanced)]\n    for panel in aggregate+noniid+balanced:\n        text += ['### '+panel['rows'][0]['distribution']+' / '+panel['rows'][0]['attack'],'']\n        text +="),
        ("for panel in tables['panels'] + aggregate:", "for panel in tables['panels'] + aggregate + noniid + balanced:"),
        ("is_aggregate=panel['rows'][0]['attack'].startswith('Five-scene')", "is_aggregate=panel['rows'][0]['attack'].startswith(('Five-scene','Ten-scene'))"),
        ("('Five IID scenarios, seed-first mean. ' if is_aggregate else 'CelebA, five IID scenarios plus non-IID Benign, F Flip, FedSA and S-DFA. ')", "((panel['rows'][0]['distribution']+' scenes, seed-first mean. ') if is_aggregate else 'CelebA, all ten IID/non-IID scenarios. ')"),
        ("attack='Five-IID mean' if is_aggregate else", "attack=panel['rows'][0]['distribution']+' mean' if is_aggregate else"),
        ("[('per_scene',tables['panels']),('five_IID_seed_first',aggregate)]", "[('per_scene',tables['panels']),('five_IID_seed_first',aggregate),('five_nonIID_seed_first',noniid),('balanced_ten_scene_seed_first',balanced)]"),
        ("total_scalars=checks['mean_sd_scalars']+aggregate_check['mean_sd_scalars'],total_display_cells=729+81", "total_scalars=checks['mean_sd_scalars']+aggregate_check['mean_sd_scalars']+noniid_check['mean_sd_scalars']+balanced_check['mean_sd_scalars'],total_display_cells=810+243")
    ], 'verify_saved.py': [
        ("    result = dict(per_scene=numeric.verify(records,panels),seed_first=aggregate_numeric.verify_aggregate([r for r in records if r['distribution']=='IID'],aggregate))", "    additional=read(H/'CROSS_SCENE_ADDITIONAL.json')\n    projected=[dict(r,distribution='IID') for r in records if r['distribution']=='non-IID']\n    result = dict(per_scene=numeric.verify(records,panels),seed_first=aggregate_numeric.verify_aggregate([r for r in records if r['distribution']=='IID'],aggregate),nonIID_seed_first=aggregate_numeric.verify_aggregate(projected,additional['nonIID_five_scene_panels']),balanced_seed_first=aggregate_numeric.verify_balanced_aggregate(records,additional['balanced_ten_scene_panels']))"),
        ('len(previous)==160', 'len(previous)==180'),
        (old_filter.replace("r[", "row["), new_filter.replace("r[", "row[")),
        ('for panel in panels+aggregate:', "for panel in panels+aggregate+additional['nonIID_five_scene_panels']+additional['balanced_ten_scene_panels']:"),
        (r"cells==810 and tex.count(r'\begin{table*}')==18 and tex.count(r'\end{table*}')==18", r"cells==1053 and tex.count(r'\begin{table*}')==36 and tex.count(r'\end{table*}')==36"),
        ('len(records)==180', 'len(records)==200'),
        ("for a in ('Benign','F Flip','FedSA','S-DFA'))", "for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA'))"),
        ("all_scalars=result['per_scene']['mean_sd_scalars']+result['seed_first']['mean_sd_scalars']", "all_scalars=result['per_scene']['mean_sd_scalars']+result['seed_first']['mean_sd_scalars']+result['nonIID_seed_first']['mean_sd_scalars']+result['balanced_seed_first']['mean_sd_scalars']"),
        ('old160_object_bytes_order_exact=True,old1296_scalars_and648_cells_exact=True', 'old180_object_bytes_order_exact=True,old1458_scalars_and729_cells_exact=True'),
        ('tex_fragment_tables=18', 'tex_fragment_tables=36')
    ]}
    contracts={};diffs=[];report={}
    for name,replacements in specs.items():
        before=source(name);after=before
        for old,new in replacements:
            assert after.count(old)==1, (name,'non-unique old',old)
            after=after.replace(old,new,1)
        inverse=after
        for old,new in reversed(replacements):
            assert inverse.count(new)==1, (name,'non-unique inverse',new)
            inverse=inverse.replace(new,old,1)
        assert inverse==before
        compile(after,name+' [A100 source only]','exec')
        oldfuncs={n.name:n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef)}
        newfuncs={n.name:n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef)}
        assert oldfuncs.keys()==newfuncs.keys()
        unchanged=[]
        for key in oldfuncs.keys()-{'main'}:
            assert ast.dump(oldfuncs[key])==ast.dump(newfuncs[key]), (name,key)
            unchanged.append(key)
        if name=='build.py':
            loop=lambda f:next(n for n in f.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target)=='(index, ids, adoption_path, index_path, batch_root)')
            assert ast.dump(loop(oldfuncs['main']))==ast.dump(loop(newfuncs['main']))
        contracts[name]=dict(A90_mapped_source_sha256=hashlib.sha256(before.encode()).hexdigest(),replacements=replacements)
        report[name]=dict(mapped_sha256=hashlib.sha256(after.encode()).hexdigest(),reversible_replacements=len(replacements),inverse_A90_bytes_exact=True,helper_AST_exact=sorted(unchanged))
        diffs.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='A90_effective/'+name,tofile='A100_effective/'+name))
    pins={}
    for rel in [
        'tmp/celeba_mechanism_A90_candidate_20261011/'+n for n in ('FILES_SHA256.json','DELIVERY_FILES_SHA256.json','source_adapter.py','SOURCE_ADAPTATIONS.json','build.py','finish.py','verify_saved.py')
    ]+[
        'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011/ROOT_VERIFICATION.json',
        'tmp/celeba_mechanism_C100_table_20261010/panels.py',
        'tmp/celeba_mechanism_C100_table_20261010/verify_numeric.py',
        'tmp/celeba_mechanism_C100_table_20261010/assemble.py',
        'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
    ]:
        p=R/rel;pins[rel]=dict(sha256=sha(p),bytes=p.stat().st_size)
    assert pins['tmp/celeba_mechanism_A90_candidate_20261011/FILES_SHA256.json']['sha256']=='4054a32a1c39b39ab0b1f8943b28c12f2b2615d4ec8a9154eecdd500ac06159a'
    assert pins['tmp/celeba_mechanism_A90_candidate_20261011/DELIVERY_FILES_SHA256.json']['sha256']=='811065967b0515f1a8efee376091477378dcaa061387e5b6a9cb7b8adb14f8d5'
    assert pins['docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_nine_scenes90_20261011/ROOT_VERIFICATION.json']['sha256']=='445904a761cad8de89b57a9e0dd65fab298dbd097d7a0bd5f3d75458a6f65cdd'
    put('SOURCE_ADAPTATIONS.json',contracts)
    put('SOURCE_PINS.json',pins)
    agg={k:v for k,v in pins.items() if k.endswith(('/panels.py','/verify_numeric.py','/evidence_v4.py'))}
    put('AGGREGATE_SOURCE_PINS.json',agg)
    with (H/'A90_TO_A100_SOURCE.diff').open('x',encoding='utf8',newline='\n') as f:f.write(''.join(diffs))
    compiled=[]
    for p in sorted(H.glob('*.py')):
        compile(p.read_bytes(),str(p),'exec');compiled.append(p.name)
    assert not (H/'ROOT_BINDING.json').exists()
    assert not any((H/n).exists() for n in ('records.json','tables.json','TABLES.md','IID_SEED_FIRST.json','CROSS_SCENE_ADDITIONAL.json'))
    put('SOURCE_COMPILE_AND_DIFF.json',dict(status='PREPARED_SOURCE_COMPILED_NO_CHECKER_OR_STATISTICS_EXECUTED',mapped_sources=report,compiled_entry_files=compiled,dependency_pins=len(pins),original_per_record_loop_AST_exact=True,all_non_main_helpers_AST_exact=True,original_seed_first_functions_reused=['aggregate_panels','verify_aggregate','aggregate_balanced_panels','verify_balanced_aggregate'],nonIID_metadata_projection_matches_original_C100=True,actual_binding_present=False,scientific_checker_executed=False,statistics_executed=False,new_accepted=0,new_inference=0,new_fit=0,new_training=0,test=False))
    print(json.dumps(dict(status='PREPARED_ONLY',compiled=compiled,mapped_sources=report),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
