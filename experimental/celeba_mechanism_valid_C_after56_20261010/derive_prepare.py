"""Derive the exact4 constructor by complete semantic rebindings, not digit replacement."""
from pathlib import Path
import ast, difflib, hashlib, json, re
D=Path(__file__).resolve().parent; O=D.with_name('celeba_mechanism_valid_C_after50_20261010'); R=D.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(name,value):
    with (D/name).open('x',encoding='utf-8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2)+'\n')
ids=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
B=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'; tag='root_delta_20261010T003657Z'
rp=B/tag/'ROOT_DELTA_VERIFICATION.json'; root=read(rp); review=read(D/'ROOT_NATIVE_INCREMENT_REVIEW.json')
assert root['new_ids']==review['added_ids']==ids and root['total_new_strict_and_offserver']==review['native_accepted']==160
files={}
for name,p,key in [('archive',B/(tag+'.tar.gz'),'archive_sha256'),('receipt',B/(tag+'.tar.gz.receipt.json'),'receipt_sha256'),('proof',B/(tag+'_offserver_verification.json'),'offserver_proof_sha256'),('ledger',B/tag/'verified_ledger.json','ledger_sha256'),('inspection',B/('mechanism_inspection_v4_'+tag)/'inspection.json','inspection_sha256')]:
    assert sha(p)==root[key]; files[name]={'path':p.relative_to(R).as_posix(),'sha256':sha(p)}
for name,p in [('root_delta',rp),('root_review',D/'ROOT_NATIVE_INCREMENT_REVIEW.json')]:files[name]={'path':p.relative_to(R).as_posix(),'sha256':sha(p)}
write('NATIVE_INPUTS.json',dict(status='ACTUAL_STRICT_OFFSERVER_NATIVE160_C_AFTER56_INPUTS',selected_ids=ids,files=files,actual_native_accepted=160,prior_three_views_accepted=156,CNN=False,dispatch=False))
write('PREPARATION_BASIS.json',dict(status='ACTUAL_NATIVE_INPUTS_BOUND_SOURCE_CONSTRUCTION_NOT_EXECUTION',actual_native=160,prior_three_views=156,selected_ids=ids,prior_adoption_sha256='a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080',native_independent_review_sha256=sha(D/'ROOT_NATIVE_INCREMENT_REVIEW.json'),initial_plan_status='PLAN_ONLY_WAITING_FOR_ROOT_ACTUAL_NATIVE_DELTA',CNN=False,SSH=False,dispatch=False))
s=(O/'prepare.py').read_text(encoding='utf-8-sig'); before=s
changes={
 'native156 minus closed150 C6':'native160 minus closed156 C4',
 'C_after40':'C_after47','C_after47':'C_after50','C_after50':'C_after56',
 'C_AFTER47':'C_AFTER50','C_AFTER50':'C_AFTER56',
 'inventory_actual150_Full100refs.json':'inventory_actual156_Full100refs.json',
 'inventory_actual156_Full100refs.json':'inventory_actual160_Full100refs.json',
 'aa5073948bd1c0854b3a4d760ee58b892909f702c31950e5ee80f0cf83b2efc0':'6e79e65b34a1ff78628884993aa6089fdc8c5d07951ed97ce016ad9b3ca6aaf0',
 'b647c5a6759e709ab4c4fdd9d7ba74361901f98973d67f69176569b1a14c8ef5':'073bfde67b2286f6e29fe5f6e2c1f7580f74465b4b0a4e42c583ab0332aa6595',
 'fc5f9b31aef29b1f3c43537301c4eacac40cd4d789114a3238f41e5c6d58cab4':'2131f2386cb3a851f990d6ed4baa38f5ea90c9acadcd62fad50d24bf97277bcd',
 'incremental_20261009T235311Z':'incremental_20261010T002635Z',
 '3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8':'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080',
 '64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea':'3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8',
 "adoption['cumulative_three_view_models']==150 and adoption['accepted_new']==3 and adoption['original147_unchanged']":"adoption['cumulative_three_view_models']==156 and adoption['accepted_new']==6 and adoption['original150_unchanged']",
 'NATIVE156_':'NATIVE160_',
 "['total_new_strict_and_offserver']==156":"['total_new_strict_and_offserver']==160",
 "review['ledger_accepted_unique_ids']==156":"review['ledger_accepted_unique_ids']==160",
 'original250_raw_json_record_bytes_exact':'original256_raw_json_record_bytes_exact',
 "inspection['new_count']==156":"inspection['new_count']==160",
 'len(rows)==156 and len(closed)==150':'len(rows)==160 and len(closed)==156',
 "if i.startswith('minus_C_')])==56":"if i.startswith('minus_C_')])==60",
 "proof['members_verified']==78":"proof['members_verified']==62",
 "'actual_accepted_new':156":"'actual_accepted_new':160",
 "'pending_new_without_checkpoint':644":"'pending_new_without_checkpoint':640",
 'exact6 native C minus closed U100+C50':'exact4 native C minus closed U100+C56',
 "'actual_strict_offserver':156":"'actual_strict_offserver':160",
 "'scope_bound_prior_and_new':156":"'scope_bound_prior_and_new':160",
 "'actual_global_pending_native':644":"'actual_global_pending_native':640",
 '644 not yet native accepted':'640 not yet native accepted',
 'PRIOR150_EXCLUDED':'PRIOR156_EXCLUDED',
 'prior147_root_adoption_sha256':'prior150_root_adoption_sha256',
 'len(records) == 150 and len({r[\'id\'] for r in records}) == 150':'len(records) == 156 and len({r[\'id\'] for r in records}) == 156',
 'len(records) == 156 and len({r[\'id\'] for r in records}) == 156':'len(records) == 160 and len({r[\'id\'] for r in records}) == 160',
 'exactly150 actual accepted terminals':'exactly156 actual accepted terminals',
 'exactly156 actual accepted terminals':'exactly160 actual accepted terminals',
 'Actual accepted150 snapshot':'Actual accepted156 snapshot','Actual accepted156 snapshot':'Actual accepted160 snapshot',
 'Excluded-prior147/selected3':'Excluded-prior150/selected6','Excluded-prior150/selected6':'Excluded-prior156/selected4',
 '== 650':'== 644','== 644':'== 640',
 'Only adopted U100+C47 plus exact three C terminals; no other scope':'Only adopted U100+C50 plus exact six C terminals; no other scope',
 'Only adopted U100+C50 plus exact six C terminals; no other scope':'Only adopted U100+C56 plus exact four C terminals; no other scope',
 'len(ids) == len(set(ids)) == 3':'len(ids) == len(set(ids)) == 6',
 'len(ids) == len(set(ids)) == 6':'len(ids) == len(set(ids)) == 4',
 'selected3_pairs':'selected6_pairs','selected6_pairs':'selected4_pairs',
 "scope.pop('prior150_root_adoption')":"scope.pop('prior150_root_adoption')",
 "scope.pop('prior147_root_adoption')":"scope.pop('prior150_root_adoption')",
 'actual_native_accepted_records=156':'actual_native_accepted_records=160',
 'prior150_root_adoption=inv':'prior156_root_adoption=inv',
 'pending_native_count_without_model=644':'pending_native_count_without_model=640',
 '==156\n':'==160\n',
 'require_approval_exact6_change':'require_approval_exact4_change','original150_records_exact':'original156_records_exact',
 "native_archive_members_verified':78":"native_archive_members_verified':62",
 "'members_verified_now_without_inference':78":"'members_verified_now_without_inference':62",
 'SELECTED_6.txt':'SELECTED_4.txt',
 "'closed147':'closed150'":"'closed150':'closed156'", "'Prior147':'Prior150'":"'Prior150':'Prior156'", "'prior147':'prior150'":"'prior150':'prior156'",
 "'prepared3 frozen':'prepared6 frozen'":"'prepared6 frozen':'prepared4 frozen'", "'reviewed3 root':'reviewed6 root'":"'reviewed6 root':'reviewed4 root'", "'Only3 IDs':'Only6 IDs'":"'Only6 IDs':'Only4 IDs'", "'exact3-scope':'exact6-scope'":"'exact6-scope':'exact4-scope'",
 "len(scope['excluded_prior_ids']) == 147":"len(scope['excluded_prior_ids']) == 150", "len(scope['excluded_prior_ids']) == 150":"len(scope['excluded_prior_ids']) == 156",
 'Prepared exact3-terminal':'Prepared exact6-terminal','Prepared exact6-terminal':'Prepared exact4-terminal',
 'len(chosen)==3':'len(chosen)==6','len(chosen)==6':'len(chosen)==4',
 'reviewed3 CPU':'reviewed6 CPU','reviewed6 CPU':'reviewed4 CPU',
 'Previously accepted C after47':'Previously accepted C after50','Previously accepted C after40':'Previously accepted C after47',
 'len(prior|set(ids))<3':'len(prior|set(ids))<6','len(prior|set(ids))<6':'len(prior|set(ids))<4',
 'ALL3_STRICT':'ALL6_STRICT','ALL6_STRICT':'ALL4_STRICT',
 'closed150_must_not_replay':'closed156_must_not_replay', "'selected':6":"'selected':4"
}
used={k:s.count(k) for k in changes}
s=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m[0]],s)
n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTED' for t in n.targets))
s=s.replace(ast.get_source_segment(s,n),'SELECTED = '+repr(ids),1)
ast.parse(s);write('prepare.py',s)
write('PREPARE_REBIND_SOURCE_DIFF.patch',''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='sealed_C_after50/prepare.py',tofile='C_after56/prepare.py')))
print(json.dumps(dict(status='PREPARE_SOURCE_DERIVED_NOT_EXECUTED',used={k:v for k,v in used.items() if v})))
