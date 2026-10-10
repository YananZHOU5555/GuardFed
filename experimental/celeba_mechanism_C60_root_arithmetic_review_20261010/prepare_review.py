"""Minimally adapt the accepted C50 independent checker; do not read future outputs."""
from pathlib import Path
import ast, difflib, hashlib, json, re
H=Path(__file__).resolve().parent;O=H.with_name('celeba_mechanism_C50_root_arithmetic_review_20261010')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(O/'review.py')=='13b67bdae74b43925c3d5ebfc51d016fce6af1420c05e1940b5062eea24b7d3d'
assert sha(O/'source_connections.py')=='d4fe902a14389157e4ca1498d67dd2e4bf8e1a25292e6855e81689039ded2c30'
def write(name,text):
    with (H/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def sub(s,a,b):
    assert a in s,a
    return s.replace(a,b)
s=(O/'review.py').read_text(encoding='utf-8-sig');original=s
changes={
 'Independent stdlib C50':'Independent stdlib C60',
 'celeba_mechanism_three_view_C_five_scenes_prepared_20261010':'celeba_mechanism_C_six_scenes_prepared_20261010',
 'three_view_C_four_scenes_20261010':'three_view_C_five_scenes_20261010',
 '06e58da21084ebc2b1c83d73cd10a1f01b9a6d4eed45baea6e99d4a3c4fd242f':'4a1a2f9d71bd432e58229f98cf7f98b1fdba9e8b60dad4c9338c8801b240067c',
 'eb08dbc328339e8293133bbd35f60e627ca70b7436ec3b20871bc149e6047172':'811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d',
 'INDEPENDENT_C50_FIVE_SCENE':'INDEPENDENT_C60_SIX_SCENE',
 'original_four_scene_root_sha256':'original_C50_root_sha256',
 'OLD80_RECORDS':'OLD100_RECORDS',
 'old80_record_JSON_bytes_and_order_exact':'old100_record_JSON_bytes_and_order_exact',
 'old648_scalars_exact':'old810_scalars_exact','old324_display_cells_exact':'old405_display_cells_exact',
}
s=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m[0]],s)
s=sub(s,"SCENES = ('Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA')", "SCENES = ('Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA')\nCELLS = [('IID',a) for a in SCENES]+[('non-IID','Benign')]")
s=sub(s,"source_files = read(BASE/'FILES_SHA256.json')['files']", "source_files = read(BASE/'FILES_SHA256.json')['members']")
s=sub(s,"verify_files(BASE,{r['path']:r for r in source_files})", "verify_files(BASE,{r['path']:dict(sha256=r['sha256'],bytes=r['size']) for r in source_files})")
s=sub(s,"handoff['unique_records'] == 100 and handoff['Full'] == handoff['minus_C'] == 50", "handoff['unique_records'] == 120 and handoff['Full'] == handoff['minus_C'] == 60")
s=sub(s,"    assert sha(OLD/'snapshot/records.json') == bindings['old80_records_sha256']\n", "")
start=s.index("    adoption = Path(bindings['actual_C3_adoption'])")
end=s.index("    records = read(snap/'records.json')['records']; tables",start)
s=s[:start]+"""    binding = bindings['actual_C4_binding']
    stage = (ROOT/binding['stage']).resolve()
    assert stage == (ROOT/'tmp/celeba_mechanism_valid_C_after56_20261010').resolve()
    adoption = (ROOT/binding['adoption']).resolve(); adopted = read(adoption)
    assert adoption.name == 'ROOT_ADOPTION_REVIEW.json' and adoption.parent.parent == stage/'execution_candidate/backups'
    assert sha(adoption) == binding['adoption_sha256'] == handoff['actual_C4_adoption_sha256']
    assert sha(stage/'FILES_SHA256.json') == binding['science_sha256'] == '2095ab384a7844355fc92453dbfa5d2922f88f378838e7b7eda2524578f3bbd6'
    assert sha(stage/'execution_candidate/EXECUTION_SOURCE_SHA256.json') == binding['execution_sha256'] == 'd1679c0bbd53bc66e4ea7ae792000d398efcafe5192bc5164ffd80e7a2eeb236'
    assert sha(stage/'inventory_actual160_Full100refs.json') == binding['inventory_sha256'] == '302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4'
    for folder,seal in [(stage,stage/'FILES_SHA256.json'),(stage/'execution_candidate',stage/'execution_candidate/EXECUTION_SOURCE_SHA256.json')]:
        for row in read(seal)['members']:
            path=folder/row['path'];assert sha(path)==row['sha256'] and path.stat().st_size==row['size']
    assert adopted['status'] == 'ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (adopted['prior_three_view_models'],adopted['accepted_new'],adopted['cumulative_three_view_models']) == (156,4,160)
    assert adopted['accepted_new_ids'] == [f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
    assert adopted['original156_unchanged'] and adopted['source_scope_complete'] and adopted['negative_results_preserved']
    assert adopted['all_native_differences_zero'] and adopted['server_strict_bound_in_saved_receipts']
    assert adopted['science_seal_sha256']==binding['science_sha256'] and adopted['execution_seal_sha256']==binding['execution_sha256']
    assert adopted['new_training'] == adopted['new_Full_inference'] == 0 and adopted['test_inference'] is False
"""+s[end:]
s=sub(s,"len(by) == 100", "len(by) == 120")
s=sub(s,"set(itertools.product(VARIANTS[:2],('IID',),SCENES,range(91001,91011)))", "{(v,d,a,seed) for v in VARIANTS[:2] for d,a in CELLS for seed in range(91001,91011)}")
s=sub(s,"len(old_records) == 80 and records[:80] == old_records", "len(old_records) == 100 and records[:100] == old_records")
s=sub(s,"raw_records(snap/'records.json')[:80]", "raw_records(snap/'records.json')[:100]")
s=sub(s,"metric_checks == 900 and count_checks == 2400", "metric_checks == 1080 and count_checks == 2880")
s=sub(s,"def value(variant,attack,seed,view,metric):\n        return by[variant,'IID',attack,seed]", "def value(variant,distribution,attack,seed,view,metric):\n        return by[variant,distribution,attack,seed]")
s=sub(s,"len(panel['rows']) == 15", "len(panel['rows']) == 18")
s=sub(s,"set(itertools.product(('IID',),SCENES,VARIANTS))", "{(d,a,v) for d,a in CELLS for v in VARIANTS}")
for expression in ["value('minus_C',row['attack']", "value('Full',row['attack']", "value(row['variant'],row['attack']"]:
    s=sub(s,expression,expression.replace("row['attack']","row['distribution'],row['attack']"))
s=sub(s,"attack=row['attack'],minus_C_minus_Full_mean", "distribution=row['distribution'],attack=row['attack'],minus_C_minus_Full_mean")
s=sub(s,"len(errors) == 810", "len(errors) == 972")
s=sub(s,"len(recomputed) == 135", "len(recomputed) == 162")
s=sub(s,"if r['attack'] in SCENES[:4]", "if r['distribution']=='IID'")
s=sub(s,"if any(x.startswith('| IID '+a+' |') for a in SCENES[:4])", "if x.startswith('| IID ')")
s=sub(s,"len(old_lines)*3 == 324", "len(old_lines)*3 == 405")
s=sub(s,"len(paired[view]) == 50", "len(paired[view]) == 60")
s=sub(s,"set(itertools.product(('IID',),SCENES,range(91001,91011)))", "{(d,a,seed) for d,a in CELLS for seed in range(91001,91011)}")
s=sub(s,"value('minus_C',pair['attack']", "value('minus_C',pair['distribution'],pair['attack']")
s=sub(s,"value('Full',pair['attack']", "value('Full',pair['distribution'],pair['attack']")
s=sub(s,"    cross=read(snap/'cross_scene_seed_first.json');", "    assert (snap/'cross_scene_seed_first.json').read_bytes()==(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()\n    cross=read(snap/'cross_scene_seed_first.json');")
s=sub(s,"value(variant,a,seed,panel['view'],metric)", "value(variant,'IID',a,seed,panel['view'],metric)")
s=sub(s,"actual_C3_root_adoption_sha256", "actual_C4_root_adoption_sha256")
s=sub(s,"unique_records=100,paired_models=50,complete_scenes=5", "unique_records=120,paired_models=60,complete_scenes=6")
s=sub(s,"mean_SD_scalars_recomputed=810,display_cells=405,count_metrics_recomputed=900,confusion_count_checks=2400", "mean_SD_scalars_recomputed=972,display_cells=486,count_metrics_recomputed=1080,confusion_count_checks=2880")
s=sub(s,"old100_record_JSON_bytes_and_order_exact=True,old810_scalars_exact=True,old405_display_cells_exact=True,", "old100_record_JSON_bytes_and_order_exact=True,old810_scalars_exact=True,old405_display_cells_exact=True,old162_IID_aggregate_bytes_exact=True,")
s=sub(s,"'Saved count metrics recomputed;", "'Six-scene rows remain separate; the preserved seed-first aggregate covers only five IID scenes.',\n            'Saved count metrics recomputed;")
ast.parse(s);write('review.py',s)
review_diff=''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='sealed_C50/review.py',tofile='C60/review.py'))
t=(O/'source_connections.py').read_text(encoding='utf-8-sig');original=t
changes={
 'Sp7/Sp3':'non-IID Benign6/4','celeba_mechanism_C40_root_arithmetic_review_20261010':'celeba_mechanism_C50_root_arithmetic_review_20261010',
 '17ff4cebf29bd7c80c06125fd925f02e2ba727964d217a2585565cfe94233803':'8903af110c3bf5727d7dbe5687711b0b2ee30c23ce0ef893c62c370ef2da69dc',
 'INDEPENDENT_C40_FOUR_SCENE':'INDEPENDENT_C50_FIVE_SCENE',
 'celeba_mechanism_valid_C_after47_20261010':'celeba_mechanism_valid_C_after56_20261010',
 'celeba_mechanism_valid_C_after40_20261010':'celeba_mechanism_valid_C_after50_20261010',
 'inventory_actual150_Full100refs.json':'inventory_actual160_Full100refs.json','inventory_actual147_Full100refs.json':'inventory_actual156_Full100refs.json',
 'fc5f9b31aef29b1f3c43537301c4eacac40cd4d789114a3238f41e5c6d58cab4':'302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4',
 'c65a6e8a88ef1e97a687bbf259840582c3d1f4c26e8e09deb604e94a4cc5b10f':'2131f2386cb3a851f990d6ed4baa38f5ea90c9acadcd62fad50d24bf97277bcd',
 'inv150':'inv160','inv147':'inv156','len(native)==150 and len(before)==147':'len(native)==160 and len(before)==156',
 'incremental_20261009T233028Z':'incremental_20261010T002635Z',
 '64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea':'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080',
 'prior147_root_adoption_sha256':'prior156_root_adoption_sha256',
 'prior_Sp_DFA7_archive_chain':'prior6_archive_chain','new_archive_chain':'new4_archive_chain',
 'minus_C_IID_Sp-DFA_seed':'minus_C_non-IID_Benign_seed',
 'range(91001,91008)':'range(91001,91007)','range(91008,91011)':'range(91007,91011)',
 '(140,7,147)':'(150,6,156)','(147,3,150)':'(156,4,160)',
 "('Full','IID','Sp-DFA',original['seed'])":"('Full','non-IID','Benign',original['seed'])",
 "r['variant']=='Full' and r['attack']=='Sp-DFA'":"r['variant']=='Full' and r['distribution']=='non-IID' and r['attack']=='Benign'",
 'prior_C40_proof_sha256':'prior_C50_proof_sha256','prior80_record_provenance_reused':'prior100_record_provenance_reused','Sp_DFA7_plus3_chains':'nonIID_Benign6_plus4_chains',
}
t=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m[0]],t)
ast.parse(t);write('source_connections.py',t)
write('MINIMAL_SOURCE_DIFF.patch',review_diff+''.join(difflib.unified_diff(original.splitlines(True),t.splitlines(True),fromfile='sealed_C50/source_connections.py',tofile='C60/source_connections.py')))
print(json.dumps(dict(status='C60_INDEPENDENT_REVIEW_SOURCE_PREPARED_NOT_EXECUTED',review_sha256=sha(H/'review.py'),source_connections_sha256=sha(H/'source_connections.py'),actual_handoff=None,actual_delivery_seal=None)))
