"""Static contract/reuse checks only; no future snapshot or statistics are read."""
from pathlib import Path
import ast, hashlib, json
H=Path(__file__).resolve().parent;R=H.parents[1];O=H.with_name('celeba_mechanism_C50_root_arithmetic_review_20261010')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def functions(p):
    text=p.read_text(encoding='utf-8-sig');tree=ast.parse(text)
    return text,tree,{n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
old,_,a=functions(O/'review.py');new,tree,b=functions(H/'review.py')
assert all(a[n]==b[n] for n in ('raw_records','verify_files','mean_sd'))
assert 'math.fsum(values)/len(values)' in b['mean_sd'] and '/(len(values)-1)' in b['mean_sd']
caller=R/'tmp/adopt_C60_table_root_20261010.py';_,ctree,cf=functions(caller)
guard=next(n for n in ctree.body if isinstance(n,ast.FunctionDef) and n.name=='review_guard')
keys={n.slice.value for n in ast.walk(guard) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='review' and isinstance(n.slice,ast.Constant)}
keys.update(['old100_record_JSON_bytes_and_order_exact','old810_scalars_exact','old405_display_cells_exact','old162_IID_aggregate_bytes_exact'])
expected={'unique_records':120,'paired_models':60,'complete_scenes':6,'mean_SD_scalars_recomputed':972,'display_cells':486,'count_metrics_recomputed':1080,'confusion_count_checks':2880,'cross_scene_mean_SD_scalars_recomputed':162}
verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify')
ret=next(n for n in verify.body if isinstance(n,ast.Return));keywords={k.arg:k.value for k in ret.value.keywords}
assert keys|set(expected)<=set(keywords)
for key,value in expected.items():assert ast.literal_eval(keywords[key])==value
assert ast.literal_eval(keywords['status'])=='INDEPENDENT_C60_SIX_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
assert "(snap/'cross_scene_seed_first.json').read_bytes()==(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()" in new
assert "value(variant,'IID',a,seed" in new and "for a in SCENES)/5" in new
assert "records[:100] == old_records" in new and "raw_records(snap/'records.json')[:100]" in new
connections=(H/'source_connections.py').read_text('utf-8');ast.parse(connections)
for marker in ("scientific['weights_before']==scientific['weights_after']", "scientific['native_comparison']['tolerance']==1e-12", "original['same_checkpoint_all_views']", "prior6_archive_chain", "new4_archive_chain"):
    assert marker in connections,marker
for p in H.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
prepared=R/'tmp/celeba_mechanism_C_six_scenes_prepared_20261010'
assert sha(prepared/'FILES_SHA256.json')=='4a1a2f9d71bd432e58229f98cf7f98b1fdba9e8b60dad4c9338c8801b240067c'
for row in json.loads((prepared/'FILES_SHA256.json').read_bytes())['members']:
    p=prepared/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['size']
report=dict(status='C60_REVIEW_SOURCE_CONTRACT_AND_C50_REUSE_CHECKS_PASS_NOT_ACTUAL_REVIEW', unchanged_functions=['raw_records','verify_files','mean_sd'],
    required_adopter_fields_present=True, expected_future_denominators=expected, source_members_verified=12,
    actual_handoff_read=False,actual_snapshot_read=False,actual_arithmetic_review_created=False,
    actual_handoff_sha256=None,actual_delivery_seal_sha256=None,caller_sha256=sha(caller),
    new_CNN=0,threshold_refits=0,SSH=False,Git_changed=False,canonical_changed=False)
with (H/'SOURCE_CHECK.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
