"""Exact13 scope/path adapters; no scientific execution."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
H=Path(__file__).resolve().parent;R=H.parents[1]
O=R/'tmp/celeba_flgmm_closed47_saved_acceptance_preparation_20261011/attempt_v2'
S=H/'saved';S.mkdir(exist_ok=False)
oldpackage='43b53caf0d6a0b4ac268fdad6597065cd76eee26d52f6d4929d45478e47cf023'
package=hashlib.sha256((H/'FILES_SHA256.json').read_bytes()).hexdigest()
ids=json.loads((H/'MANIFEST.json').read_bytes())['exact_ids']
pins=json.loads((O/'FILES_SHA256.json').read_bytes())['files']
changes=[]
for name in ['contract.py','check_linux.py','linux_saved_remote.py','transport.py','transport_remote.py','verify_arrays.py']:
 p=O/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==pins[name]['sha256']
 old=p.read_text(encoding='utf8');s=old
 s=s.replace('celeba_flgmm_three_view_closed_batch_preparation_20261011','fl_three_view_after48_20261011').replace('celeba_flgmm_three_view_closed_batch_20261011','fl_three_view_after48_20261011')
 s=s.replace(oldpackage,package).replace('guardfed_flgmm_closed47_valid','guardfed_flgmm_after48_exact13_valid')
 s=s.replace('FLGMM47','FLGMM13').replace('EXACT47','EXACT13').replace('exact47','exact13').replace('Exact47','Exact13').replace('flgmm47_saved','flgmm13_saved')
 s=s.replace('==47','==13').replace('== 47','== 13').replace(':47',':13').replace(': 47',': 13')
 s=s.replace('==98','==30').replace('=98','=30').replace(':98',':30')
 if name=='transport_remote.py':
  start=s.index('ids=[');end=s.index('\n',start);s=s[:start]+'ids='+repr(ids)+s[end:]
 ast.parse(s)
 (S/name).write_text(s,encoding='utf8',newline='\n')
 changes.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='original47/'+name,tofile='exact13/'+name))
(S/'FIXED_IDS.json').write_text(json.dumps({'exact_ids':ids},indent=2)+'\n')
# Original complete check_saved file, loaded by Linux without changing one line.
science=R/'tmp/celeba_flgmm_three_view_closed_batch_preparation_20261011/originals/saved_science.py'
assert hashlib.sha256(science.read_bytes()).hexdigest()=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
shutil.copyfile(science,S/'saved_science.py')
old=(R/'tmp/celeba_flgmm47_saved_outputs_audit_source_20261011/audit_saved.py').read_text()
assert hashlib.sha256((R/'tmp/celeba_flgmm47_saved_outputs_audit_source_20261011/audit_saved.py').read_bytes()).hexdigest()=='6fa4cd1f5d9917f28c22fa5b61f0e06d45bada72cf4fa2e068451d90f3121e35'
s=old.replace('Exact47 saved-output audit.','Exact13 saved-output audit after accepted48.')
s=s.replace('ROOT = HERE.parents[1]','ROOT = HERE.parents[2]')
s=s.replace("ACTUAL = ROOT / 'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001'", "ACTUAL = ROOT / 'tmp/fl_three_view_after48_20261011/actual/saved001'\nHISTORICAL = ROOT / 'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001'")
s=s.replace("PREPARED = ROOT / 'tmp/celeba_flgmm_closed47_saved_acceptance_preparation_20261011/attempt_v2'",'PREPARED = HERE')
s=s.replace("read(ACTUAL / 'OFFSERVER_ARRAY_REFIT_CHECK.failure.json')", "read(HISTORICAL / 'OFFSERVER_ARRAY_REFIT_CHECK.failure.json')")
s=s.replace('== 47','== 13').replace('Require exact47','Require exact13').replace("'metric_values_checked': 423, 'integer_base_counts_checked': 1128, 'prediction_rules_checked': 141", "'metric_values_checked': 117, 'integer_base_counts_checked': 312, 'prediction_rules_checked': 39")
ast.parse(s);(S/'audit_saved.py').write_text(s,encoding='utf8',newline='\n')
changes.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='original47/audit_saved.py',tofile='exact13/audit_saved.py'))
# The three bodies making numerical/saved-fit assertions are byte-identical.
names=['saved_fits_for_original_predict','saved_output_block']
for name in names:
 a=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name==name)
 b=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.get_source_segment(old,a)==ast.get_source_segment(s,b)
(S/'SCOPE_DIFF.patch').write_text(''.join(changes),encoding='utf8',newline='\n')
print(json.dumps({'status':'SAVED13_SCOPE_DERIVED_NO_FIT','Linux_original_whole_source_exact':True,'saved_fits_and_original_block_adapter_exact':True,'expected_members':30,'expected_records':13}))
