"""Correct only the failed source-construction assertions; no scientific execution."""
import ast,json
from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'rebind_sources.py';tree=ast.parse(p.read_text(encoding='utf-8'));nodes=[]
for n in tree.body:
 if n.lineno>=55:break
 if isinstance(n,ast.Expr) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='put' for x in ast.walk(n)):continue
 nodes.append(n)
ns={'__file__':str(p.resolve())};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
s=ns['s'];fix=ns['replace'];put=ns['put']
edits={
 "adoption['cumulative_three_view_models']==180":"adoption['cumulative_three_view_models']==170",
 "schema='celeba_mechanism_valid_replay_inventory_C_after60'":"schema='celeba_mechanism_valid_replay_inventory_C_after70'",
 "'PRIOR160_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND'":"'PRIOR170_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND'",
 "'exactly160 actual accepted terminals'":"'exactly170 actual accepted terminals'",
 "'Actual accepted160 snapshot'":"'Actual accepted170 snapshot'",
 ".replace('Excluded-prior160/selected4','Excluded-prior170/selected10')":".replace('Excluded-prior160/selected10','Excluded-prior170/selected10')",
 ".replace('== 640','== 630')":".replace('== 630','== 620')",
 "'Only adopted U100+C60 plus exact four C terminals; no other scope'":"'Only adopted U100+C60 plus exact ten C terminals; no other scope'",
 "assert after.count('len(ids) == len(set(ids)) == 4')==1;after=after.replace('len(ids) == len(set(ids)) == 4','len(ids) == len(set(ids)) == 10')":"assert after.count('len(ids) == len(set(ids)) == 10')==1",
 "assert of['require_approval'].replace('len(ids) == len(set(ids)) == 4','len(ids) == len(set(ids)) == 10')==nf['require_approval']":"assert of['require_approval']==nf['require_approval']",
 "'Prior160':'Prior160'":"'Prior160':'Prior170'",
 "len(scope['excluded_prior_ids']) == 156":"len(scope['excluded_prior_ids']) == 160",
 "service = 'guardfed_celeba_mechanism_valid_C_after50'":"service = 'guardfed_celeba_mechanism_valid_C_after56'",
 '/celeba_mechanism_valid_C_after50_20261010/execution_candidate/batch.py':'/celeba_mechanism_valid_C_after56_20261010/execution_candidate/batch.py',
 'Previously accepted C after50':'Previously accepted C after56',
 'a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080':'21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e',
}
for a,b in edits.items():s=fix(s,a,b)
ast.parse(s);put('prepare.py',s);put('PREPARE_REBIND_SOURCE_DIFF.patch',ns['diff']('prepare.py',s))
print(json.dumps({'status':'SOURCE_CONSTRUCTOR_READY_FOR_ACTUAL_NATIVE_REVIEW','failure_preserved':'REBIND_COMMAND.json','root_delta_sha256':ns['NATIVE_ROOT'],'prior170_root_adoption_sha256':ns['PRIOR_ADOPTION']}))
