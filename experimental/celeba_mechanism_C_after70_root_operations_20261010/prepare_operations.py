"""Create fixed metadata and tiny operation entries; no external operation."""
from pathlib import Path
import ast,difflib,hashlib,json,re
H=Path(__file__).resolve().parent;R=H.parents[1]
O=R/'tmp/celeba_mechanism_C_after60_root_operations_20261010'
P=R/'tmp/celeba_mechanism_valid_C_after70_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(n,s):
 with (H/n).open('x',encoding='utf-8',newline='\n') as f:f.write(s if isinstance(s,str) else json.dumps(s,indent=2)+'\n')
assert sha(P/'PACKAGE_SHA256.json')=='370e75fc5e015048c3cba5c8c13e856a8bbcf8b5a8a46cb158a0468981303159'
parent=R/'tmp/celeba_mechanism_valid_C_after60_20261010'
prior=parent/'execution_candidate/backups/incremental_20261010T015546Z/ROOT_ADOPTION_REVIEW.json'
assert sha(prior)=='7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'
replacements={
 'C_after60':'C_after70','C_after56':'C_after60','C_AFTER60':'C_AFTER70','C_AFTER56':'C_AFTER60',
 'inventory_actual170_Full100refs.json':'inventory_actual180_Full100refs.json',
 'old160':'old170','original160':'original170','prior160':'prior170',
 'minus_C_non-IID_F Flip_seed':'minus_C_non-IID_FedSA_seed',
 'incremental_20261010T005530Z':'incremental_20261010T015546Z',
 '21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e':sha(prior),
 'ed8ecc84781b799e205c9e139dce8e753cb5545139b7e48ef23d5f8d07a50a77':sha(P/'FILES_SHA256.json'),
 '12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896':sha(P/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),
 '08aaf7a04ca819b32f1adcf102079a7953f9e928e00bd48ff5c0370e1f17958e':sha(P/'PACKAGE_RECEIPT.json'),
 '5d35280b387c7ff621d94c00fb5983284edf06f272dba4f5a01dc78fb297f824':sha(P/'PACKAGE_SHA256.json'),
 'c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef':sha(P/'inventory_actual180_Full100refs.json'),
 '== 160':'== 170','==160':'==170','== 170':'== 180',
 'prior_three_view_models=160':'prior_three_view_models=170',
 'cumulative_three_view_models=170':'cumulative_three_view_models=180',
}
pins={}
paths=[P/n for n in ['PACKAGE_SHA256.json','PACKAGE_RECEIPT.json','FILES_SHA256.json','execution_candidate/EXECUTION_SOURCE_SHA256.json','inventory_actual180_Full100refs.json','NATIVE_INPUTS.json','SCOPE.json']]+[parent/'inventory_actual170_Full100refs.json',prior]
for p in paths:pins[p.relative_to(R).as_posix()]=sha(p)
ops=read(O/'TRANSPORT_REBIND.json')['members'];original={p['path'][:-3]:p['sha256'] for p in ops}
for name,pin in original.items():assert sha(O/(name+'.py'))==pin
put('BINDINGS.json',dict(status='SOURCE_ONLY_ROOT_OPERATIONS_PREPARED_NO_AUTHORITY',prepared=P.relative_to(R).as_posix(),prior_inventory=(parent/'inventory_actual170_Full100refs.json').relative_to(R).as_posix(),prior_adoption=prior.relative_to(R).as_posix(),pins=pins,original_operations=original,replacements=replacements,source_review='External --review and --review-sha256 supplied by root at deploy invocation; not guessed',SSH=False,approval_created=False,new_accepted=0))
diff=[]
for name in original:
 before=(O/(name+'.py')).read_text(encoding='utf-8-sig')
 after=re.sub('|'.join(re.escape(k) for k in sorted(replacements,key=len,reverse=True)),lambda m:replacements[m[0]],before)
 ast.parse(after)
 diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='sealed_C_after60/'+name+'.py',tofile='in_memory_C_after70/'+name+'.py'))
 put(name+'.py',f'"""Explicit root {name} entry; original pinned body is loaded only when invoked."""\nfrom loader import run\nif __name__ == "__main__":\n    run("{name}")\n')
put('SOURCE_DIFF.patch',''.join(diff))
print(json.dumps({'status':'FOUR_THIN_ENTRIES_AND_METADATA_READY_NO_EXECUTION','original_operation_bytes':sum((O/(n+'.py')).stat().st_size for n in original),'entries_bytes':sum((H/(n+'.py')).stat().st_size for n in original)}))
