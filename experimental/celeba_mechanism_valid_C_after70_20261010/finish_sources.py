"""Rebind the original metadata check and package writer; no science/runtime."""
from pathlib import Path
import ast,difflib,re,json
H=Path(__file__).resolve().parent;O=H.with_name('celeba_mechanism_valid_C_after60_20261010')
def put(n,s):
 with (H/n).open('x',encoding='utf-8',newline='\n') as f:f.write(s)
def rebind(s,changes):
 return re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m[0]],s)
def old(n):return (O/n).read_text(encoding='utf-8-sig')
common={'celeba_mechanism_valid_C_after56_20261010':'celeba_mechanism_valid_C_after60_20261010','celeba_mechanism_valid_C_after60_20261010':'celeba_mechanism_valid_C_after70_20261010','C_AFTER60':'C_AFTER70','valid_C_after56':'valid_C_after60','valid_C_after60':'valid_C_after70','inventory_actual160_Full100refs.json':'inventory_actual170_Full100refs.json','inventory_actual170_Full100refs.json':'inventory_actual180_Full100refs.json','old160':'old170','original160':'original170','closed160':'closed170','native170':'native180','native160':'native170','prior160':'prior170','C_nonIID_FFlip':'C_nonIID_FedSA','minus_C_non-IID_F Flip_seed':'minus_C_non-IID_FedSA_seed','21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e':'7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'}
s=rebind(old('check_prepared.py'),common|{'==170':'==180','==630':'==620','==160':'==170',"'positive_inventory_records':170":"'positive_inventory_records':180"})
# The prior service string has no directory name and must remain after60.
s=s.replace("service = 'guardfed_celeba_mechanism_valid_C_after56'","service = 'guardfed_celeba_mechanism_valid_C_after60'")
ast.parse(s);put('check_prepared.py',s)
put('CHECKER_REBIND_SOURCE_DIFF.patch',''.join(difflib.unified_diff(old('check_prepared.py').splitlines(True),s.splitlines(True),fromfile='sealed_C_after60/check_prepared.py',tofile='C_after70/check_prepared.py')))
s=rebind(old('seal_delivery.py'),common|{"native_global=170":"native_global=180","inventory_records=170":"inventory_records=180","prior_three_views_excluded=160":"prior_three_views_excluded=170","native_accepted_snapshot=170":"native_accepted_snapshot=180","three_view_accepted_unchanged=160":"three_view_accepted_unchanged=170","guardfed_celeba_mechanism_valid_C_after60'":"guardfed_celeba_mechanism_valid_C_after70'","guardfed_celeba_mechanism_valid_C_after56'":"guardfed_celeba_mechanism_valid_C_after60'"})
ast.parse(s);put('seal_delivery.py',s)
put('SEALER_REBIND_SOURCE_DIFF.patch',''.join(difflib.unified_diff(old('seal_delivery.py').splitlines(True),s.splitlines(True),fromfile='sealed_C_after60/seal_delivery.py',tofile='C_after70/seal_delivery.py')))
print(json.dumps({'status':'CHECKER_AND_SEALER_SOURCE_READY_NO_RUNTIME','new_outputs':['check_prepared.py','seal_delivery.py']}))
