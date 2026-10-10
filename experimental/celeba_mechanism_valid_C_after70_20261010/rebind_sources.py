"""One-shot source rebind from sealed C_after60; no runtime/science execution."""
from pathlib import Path
import ast,difflib,hashlib,json,re
H=Path(__file__).resolve().parent;O=H.with_name('celeba_mechanism_valid_C_after60_20261010')
R=H.parents[1]
TAG='root_delta_20261010T023113Z'
BASE='docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
P=R/BASE/TAG
NATIVE_ROOT='1f1c7d12bf10aa457ccfdaa9cae48ab3a79843b4960d1621b5c05762d1b64772'
PRIOR_ADOPTION='7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(n,s):
 with (H/n).open('x',encoding='utf-8',newline='\n') as f:f.write(s)
def replace(s,a,b):
 assert a in s,repr(a);return s.replace(a,b)
def simultaneous(s,changes):
 return re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m[0]],s)
def source(name):return (O/name).read_text(encoding='utf-8-sig')
def diff(name,s):return ''.join(difflib.unified_diff(source(name).splitlines(True),s.splitlines(True),fromfile='sealed_C_after60/'+name,tofile='C_after70/'+name))
assert sha(P/'ROOT_DELTA_VERIFICATION.json')==NATIVE_ROOT
root=read(P/'ROOT_DELTA_VERIFICATION.json')
assert root['total_new_strict_and_offserver']==180 and root['new_ids']==[f'minus_C_non-IID_FedSA_seed{s}' for s in range(91001,91011)]
assert sha(O/'execution_candidate/backups/incremental_20261010T015546Z/ROOT_ADOPTION_REVIEW.json')==PRIOR_ADOPTION
oldni=read(O/'NATIVE_INPUTS.json')
hash_changes={oldni['files'][k]['sha256']:root[v] for k,v in [('archive','archive_sha256'),('receipt','receipt_sha256'),('proof','offserver_proof_sha256'),('ledger','ledger_sha256'),('inspection','inspection_sha256')]}
s=source('verify_native_snapshot.py')
s=simultaneous(s,hash_changes|{'root_delta_20261010T003657Z':'root_delta_20261010T013518Z','ffa58131d986fcdb6a3e580ce3cad5ef221016ce96109ce726ce32a7dd80c093':oldni['files']['ledger']['sha256'],'be2f7e00ef45349bd57050938195c5972c2688dd1b8b2e62706fdce4c6d6c6b3':oldni['files']['inspection']['sha256'],'native160':'native170','original260':'original270','OLD260':'OLD270','previous25':'previous26'})
s=replace(s,"minus_C_non-IID_F Flip_seed","minus_C_non-IID_FedSA_seed")
s=replace(s,'TOTAL==170','TOTAL==180');s=replace(s,'TOTAL==160+len(IDS)','TOTAL==170+len(IDS)')
s=replace(s,'len(oldids)==260','len(oldids)==270')
s=replace(s,"len(ledger['entries'])==26","len(ledger['entries'])==27");s=replace(s,"len(priorledger['entries'])==25","len(priorledger['entries'])==26")
s=replace(s,'ledger_entries_verified=26','ledger_entries_verified=27')
s=replace(s,"archive=B/(TAG+'.tar.gz');receipt_path=B/(TAG+'.tar.gz.receipt.json');proof_path=B/(TAG+'_offserver_verification.json')","archive=root_path.parent/(TAG+'.tar.gz');receipt_path=root_path.parent/(TAG+'.tar.gz.receipt.json');proof_path=root_path.parent/'OFFSERVER_VERIFICATION.json'")
s=replace(s,"inspection_path=B/('mechanism_inspection_v4_'+TAG)/'inspection.json'","inspection_path=root_path.parent/'inspection/inspection.json'")
s=replace(s,"ledger_path=B/TAG/'verified_ledger.json'","ledger_path=root_path.parent/'verified_ledger.json'")
s=replace(s,"p=B/Path(entry['receipt']).name;assert sha(p)==entry['receipt_sha256']","p=receipt_path if entry['receipt_sha256']==sha(receipt_path) else B/Path(entry['receipt']).name;assert sha(p)==entry['receipt_sha256']")
ast.parse(s);put('verify_native_snapshot.py',s);put('NATIVE_CHECKER_SOURCE_DIFF.patch',diff('verify_native_snapshot.py',s))

# Actual receipt fields are pinned; root_review is added only after independent verification.
files={k:{'path':str((P/n).relative_to(R)).replace('\\','/'),'sha256':root[v]} for k,n,v in [('archive',TAG+'.tar.gz','archive_sha256'),('receipt',TAG+'.tar.gz.receipt.json','receipt_sha256'),('proof','OFFSERVER_VERIFICATION.json','offserver_proof_sha256'),('ledger','verified_ledger.json','ledger_sha256'),('inspection','inspection/inspection.json','inspection_sha256')]}
files['root_delta']={'path':(P/'ROOT_DELTA_VERIFICATION.json').relative_to(R).as_posix(),'sha256':NATIVE_ROOT}
put('NATIVE_INPUTS_PENDING_REVIEW.json',json.dumps(dict(status='ACTUAL_NATIVE180_REVIEW_PENDING_NOT_PREPARED',selected_ids=root['new_ids'],files=files,actual_native_accepted=180,prior_three_views_accepted=170,CNN=False,dispatch=False),indent=2)+'\n')

s=source('prepare.py')
# Source metadata counts are rebound simultaneously, never as global numeral edits.
s=simultaneous(s,{'celeba_mechanism_valid_C_after56_20261010':'celeba_mechanism_valid_C_after60_20261010','C_AFTER60':'C_AFTER70','closed160':'closed170','closed156':'closed160','Prior156':'Prior160','prior156':'prior160','prior160':'prior170','inventory_actual160_Full100refs.json':'inventory_actual170_Full100refs.json','inventory_actual170_Full100refs.json':'inventory_actual180_Full100refs.json','native170':'native180','Native170':'Native180','NATIVE170':'NATIVE180','original260':'original270','original160':'original170','old160':'old170','Actual accepted170':'Actual accepted180','exactly170':'exactly180','selected4_pairs':'selected10_pairs','minus_C_non-IID_F Flip_seed':'minus_C_non-IID_FedSA_seed','C_after60_metadata':'C_after70_metadata','C_AFTER56':'C_AFTER60','valid_C_after56':'valid_C_after60','valid_C_after60':'valid_C_after70','U100+C60':'U100+C70','U100+C56':'U100+C60'})
s=simultaneous(s,{'2095ab384a7844355fc92453dbfa5d2922f88f378838e7b7eda2524578f3bbd6':sha(O/'FILES_SHA256.json'),'d1679c0bbd53bc66e4ea7ae792000d398efcafe5192bc5164ffd80e7a2eeb236':sha(O/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),'302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4':sha(O/'inventory_actual170_Full100refs.json'),'incremental_20261010T005530Z':'incremental_20261010T015546Z','21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e':PRIOR_ADOPTION})
s=replace(s,"adoption['cumulative_three_view_models']==160 and adoption['accepted_new']==4 and adoption['original156_unchanged']","adoption['cumulative_three_view_models']==170 and adoption['accepted_new']==10 and adoption['original160_unchanged']")
for a,b in [('==170','==180'),('==160','==170'),('==70','==80'),('== 160','== 170'),('== 170','== 180')]:
 # These exact guard expressions are replaced in one pass below to avoid cascading.
 pass
s=simultaneous(s,{'==170':'==180','==160':'==170','==70':'==80','== 160':'== 170','== 170':'== 180',"'actual_accepted_new':170":"'actual_accepted_new':180","'pending_new_without_checkpoint':630":"'pending_new_without_checkpoint':620","'actual_strict_offserver':170":"'actual_strict_offserver':180","'scope_bound_prior_and_new':170":"'scope_bound_prior_and_new':180","'actual_global_pending_native':630":"'actual_global_pending_native':620","'630 not yet":"'620 not yet",'actual_native_accepted_records=170':'actual_native_accepted_records=180','pending_native_count_without_model=630':'pending_native_count_without_model=620'})
# Preserve comparisons against the old170 source while changing generated bridge to180.
s=replace(s,"after=after.replace(\"len(records) == 180 and len({r['id'] for r in records}) == 180\",\"len(records) == 180 and len({r['id'] for r in records}) == 180\")","after=after.replace(\"len(records) == 170 and len({r['id'] for r in records}) == 170\",\"len(records) == 180 and len({r['id'] for r in records}) == 180\")")
s=replace(s,".replace('== 640','== 630')",".replace('== 630','== 620')")
s=replace(s,"assert after.count('len(ids) == len(set(ids)) == 4')==1;after=after.replace('len(ids) == len(set(ids)) == 4','len(ids) == len(set(ids)) == 10')","assert after.count('len(ids) == len(set(ids)) == 10')==1")
s=replace(s,"assert of['require_approval'].replace('len(ids) == len(set(ids)) == 4','len(ids) == len(set(ids)) == 10')==nf['require_approval']","assert of['require_approval']==nf['require_approval']")
s=replace(s,'Excluded-prior160/selected10\'','Excluded-prior170/selected10\'') if "Excluded-prior160/selected10'" in s else s
# Execution inherits exact10 already; old after56 service becomes after60, with current root proof.
s=replace(s,"'closed160':'closed170','Prior160':'Prior160','prior160':'prior170'","'closed160':'closed170','Prior160':'Prior170','prior160':'prior170'") if "'closed160':'closed170','Prior160':'Prior160','prior160':'prior170'" in s else s
s=replace(s,"len(scope['excluded_prior_ids']) == 156","len(scope['excluded_prior_ids']) == 160")
s=replace(s,".replace(\"len(scope['excluded_prior_ids']) == 160\",\"len(scope['excluded_prior_ids']) == 180\")",".replace(\"len(scope['excluded_prior_ids']) == 160\",\"len(scope['excluded_prior_ids']) == 170\")")
s=replace(s,"s=s.replace(\"service = 'guardfed_celeba_mechanism_valid_C_after50'\",\"service = 'guardfed_celeba_mechanism_valid_C_after60'\")","s=s.replace(\"service = 'guardfed_celeba_mechanism_valid_C_after56'\",\"service = 'guardfed_celeba_mechanism_valid_C_after60'\")")
s=replace(s,"/celeba_mechanism_valid_C_after50_20261010/execution_candidate/batch.py","/celeba_mechanism_valid_C_after56_20261010/execution_candidate/batch.py")
s=replace(s,'Previously accepted C after50','Previously accepted C after56')
s=replace(s,"'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080'","'21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e'")
# Replace stale count4 transformations with identities: count remains10 on both sides.
s=s.replace(".replace('len(chosen)==4','len(chosen)==10')",'')
s=s.replace("s=s.replace('len(prior|set(ids))<4','len(prior|set(ids))<10').replace('ALL4_STRICT','ALL10_STRICT')","s=s.replace('ALL10_STRICT','ALL10_STRICT')")
ast.parse(s);put('prepare.py',s);put('PREPARE_REBIND_SOURCE_DIFF.patch',diff('prepare.py',s))
print(json.dumps({'status':'CONSTRUCTION_SOURCE_WRITTEN_AWAIT_NATIVE_REVIEW','root_delta_sha256':NATIVE_ROOT,'prior170_root_adoption_sha256':PRIOR_ADOPTION}))
