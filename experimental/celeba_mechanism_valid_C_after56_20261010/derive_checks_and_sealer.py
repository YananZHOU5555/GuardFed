"""Reuse the accepted parent's metadata checks and one-shot delivery sealer."""
from pathlib import Path
import ast, difflib, json, re
H=Path(__file__).resolve().parent; O=H.with_name('celeba_mechanism_valid_C_after50_20261010')
def transform(s,changes):
    return re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m[0]],s)
def put(name,s):
    with (H/name).open('x',encoding='utf-8',newline='\n') as f:f.write(s)
changes={
 'inventory_actual156_Full100refs.json':'inventory_actual160_Full100refs.json',
 'inventory_actual150_Full100refs.json':'inventory_actual156_Full100refs.json',
 'C_after47':'C_after50','C_after50':'C_after56','C_AFTER50':'C_AFTER56',
 '==156':'==160','==644':'==640','==150':'==156',
 'prior150_in_scope':'prior156_in_scope','closed150_must_not_replay':'closed156_must_not_replay',
 'range(91001,91007)':'range(91007,91011)',
 'bounded C6 check':'bounded C4 check',
 'len(chosen)==6':'len(chosen)==4',
 'exact6':'exact4',
 "'per_child_exact6_science_approval_positive':6":"'per_child_exact4_science_approval_positive':4",
 "'positive_inventory_records':156":"'positive_inventory_records':160",
 '3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8':'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080',
}
s=(O/'check_prepared.py').read_text(encoding='utf-8-sig'); before=s
# Preserve numeric/resource literals outside the exact metadata expressions.
s=transform(s,changes)
s=s.replace("len(scope['excluded_prior_ids'])==150", "len(scope['excluded_prior_ids'])==156")
s=s.replace('for n in (0,1,3,5,7,11):','for n in (0,1,3,5,6,11):')
ast.parse(s);put('check_prepared.py',s)
put('CHECKER_REBIND_SOURCE_DIFF.patch',''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='sealed_C_after50/check_prepared.py',tofile='C_after56/check_prepared.py')))
s=(O/'seal_delivery.py').read_text(encoding='utf-8-sig'); before=s
changes={
 'C_after47':'C_after50','C_after50':'C_after56',
 'inventory_actual150_Full100refs.json':'inventory_actual156_Full100refs.json','inventory_actual156_Full100refs.json':'inventory_actual160_Full100refs.json',
 'old150_':'old156_', 'actual_native156_':'actual_native160_', 'prior150_root_adoption_sha256':'prior156_root_adoption_sha256',
 '3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8':'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080',
 'native_global=156':'native_global=160','inventory_records=156':'inventory_records=160',
 'prior_three_views_excluded=150':'prior_three_views_excluded=156',
 'EXACT6_SOURCE':'EXACT4_SOURCE','native_accepted_snapshot=156':'native_accepted_snapshot=160',
 'three_view_accepted_unchanged=150':'three_view_accepted_unchanged=156',
 'C_nonIID_Benign_native_n=6,C_nonIID_Benign_scene_complete=False':'C_nonIID_Benign_native_n=10,C_nonIID_Benign_native_scene_complete=True,C_nonIID_Benign_three_view_scene_complete=False',
}
s=transform(s,changes);ast.parse(s);put('seal_delivery.py',s)
put('SEALER_REBIND_SOURCE_DIFF.patch',''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='sealed_C_after50/seal_delivery.py',tofile='C_after56/seal_delivery.py')))
cmd=(O/'COMMANDS.md').read_text(encoding='utf-8-sig')
cmd=transform(cmd,{
 'C_after47':'C_after50','C_after50':'C_after56','C_AFTER50':'C_AFTER56',
 'prior150':'prior156','closed150_must_not_replay':'closed156_must_not_replay',
 'exact6':'exact4','all6':'all4',
 '75 total members/74content,54 metrics/144counts/18rules':'61 total members/60content,36 metrics/96counts/12rules',
})
put('COMMANDS.md',cmd)
print(json.dumps(dict(status='CHECKER_SEALER_AND_ROOT_COMMANDS_REBOUND_NOT_EXECUTED')))
