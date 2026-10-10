"""One-shot source preparation; no future receipt, table or statistic is produced."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'tmp/celeba_mechanism_C_six_scenes_prepared_20261010'
C60=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010';C70=R/'tmp/celeba_mechanism_valid_C_after60_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
def write(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def function(p,n):
 s=p.read_text(encoding='utf8');node=next(a for a in ast.parse(s).body if isinstance(a,ast.FunctionDef) and a.name==n);return ast.get_source_segment(s,node)
panel_old=(O/'panels.py').read_text(encoding='utf8');old_expected="('non-IID','Benign')]";new_expected="('non-IID','Benign'),('non-IID','F Flip')]"
assert panel_old.count(old_expected)==1
panel_new=panel_old.replace(old_expected,new_expected).replace('Only exact five IID scenes plus non-IID Benign10 are publishable','Only five IID plus non-IID Benign/F Flip10 scenes are publishable')
assert panel_new.replace(new_expected,old_expected).replace('Only five IID plus non-IID Benign/F Flip10 scenes are publishable','Only exact five IID scenes plus non-IID Benign10 are publishable')==panel_old
with (H/'panels.py').open('x',encoding='utf8',newline='\n') as f:f.write(panel_new)
old_verify=function(O/'verify_numeric.py','verify');new_verify=old_verify
replacements=[('==120','==140'),("('non-IID','Benign')}","('non-IID','Benign'),('non-IID','F Flip')}"),('len(rows)==18','len(rows)==21'),('len(errors)==972','len(errors)==1134'),('metric_checks==1080 and count_checks==2880','metric_checks==1260 and count_checks==3360')]
for old,new in replacements:assert old in new_verify;new_verify=new_verify.replace(old,new)
reverse=new_verify
for old,new in reversed(replacements):reverse=reverse.replace(new,old)
assert reverse==old_verify
aggregate=function(O/'verify_numeric.py','verify_aggregate')
tail='''
def main():
 import argparse,json
 from pathlib import Path
 import build as b
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);a=p.parse_args()
 bindings=b.read(a.snapshot/'SOURCE_BINDINGS.json');b.verify_future_binding(bindings['actual_C10_binding']);b.verify_inputs()
 for n,pin in b.read(a.snapshot/'FILES_SHA256.json')['files'].items():b.need(b.sha(a.snapshot/n)==pin['sha256'],'Snapshot changed '+n)
 records=b.read(a.snapshot/'records.json')['records'];tables=b.read(a.snapshot/'tables.json');result=verify(records,tables['panels'])
 result['display_mean_sd_cells']=b.displayed_cells((a.snapshot/'TABLES.md').read_text('utf8'),tables['panels']);b.need(result['display_mean_sd_cells']==567,'Display scope changed')
 aggregate=a.snapshot/'cross_scene_seed_first.json';b.need(aggregate.read_bytes()==(b.OLD/'snapshot/cross_scene_seed_first.json').read_bytes(),'Old IID aggregate changed')
 result['preserved_IID_seed_first']=verify_aggregate([r for r in records if r['distribution']=='IID'],b.read(aggregate)['panels']);result['status']='INDEPENDENT_STDLIB_NUMERIC_AND_DISPLAY_PASS_PENDING_ROOT_REVIEW';print(json.dumps(result,indent=2))
if __name__=='__main__':main()
'''
numeric='"""C60 arithmetic unchanged; only exact seven-scene/cardinality guards rebound."""\nimport itertools\nimport math\n\n'+new_verify+'\n\n'+aggregate+'\n'+tail
with (H/'verify_numeric.py').open('x',encoding='utf8',newline='\n') as f:f.write(numeric)
assert function(H/'verify_numeric.py','verify_aggregate')==aggregate
inputs=read(O/'INPUTS.json');pins=dict(inputs['files'])
extra=[C60/'ROOT_VERIFICATION.json',C60/'build.py',C60/'verify_numeric.py',C60/'panels.py',O/'FILES_SHA256.json',O/'INPUTS.json',O/'ACTUAL_FILES_SHA256.json',O/'ACTUAL_HANDOFF.json',R/'tmp/celeba_mechanism_valid_C_after56_20261010/inventory_actual160_Full100refs.json',R/'tmp/celeba_mechanism_valid_C_after56_20261010/bridge.py',R/'tmp/celeba_mechanism_valid_C_after56_20261010/execution_candidate/backups/incremental_20261010T005530Z/ROOT_ADOPTION_REVIEW.json',C70/'FILES_SHA256.json',C70/'execution_candidate/EXECUTION_SOURCE_SHA256.json',C70/'inventory_actual170_Full100refs.json',C70/'HANDOFF.json',C70/'bridge.py']+list((C60/'snapshot').iterdir())
for p in extra:
 if p.is_file():pins[p.relative_to(R).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
for n,pin in pins.items():assert sha(R/n)==pin['sha256'] and (R/n).stat().st_size==pin['bytes']
root=read(C60/'ROOT_VERIFICATION.json');assert root['unique_records']==120 and root['paired_models']==60 and root['complete_scenes']==6
parent=R/'tmp/celeba_mechanism_valid_C_after56_20261010/execution_candidate/backups/incremental_20261010T005530Z/ROOT_ADOPTION_REVIEW.json';assert sha(parent)=='21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e'
handoff=read(C70/'HANDOFF.json');expected=[f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)];assert handoff['selected_ids']==expected and handoff['native_accepted_snapshot']==170 and handoff['new_three_view_accepted']==0
write('INPUTS.json',dict(status='SOURCE_PREPARED_WAIT_ACTUAL_EXACT10_OFFSERVER_ROOT_ADOPTION',full900_path=inputs['full900_path'],files=pins,prior160_adoption=parent.relative_to(R).as_posix(),prior160_adoption_sha256=sha(parent),old_C60_root_sha256=sha(C60/'ROOT_VERIFICATION.json'),expected_added_ids=expected,expected_records=140,expected_pairs=70,expected_scenes=7,expected_scene_scalars=1134,expected_cells=567,expected_receipt_metrics=1260,expected_confusion_counts=3360,old120_records_preserved=True,old972_scalars_preserved=True,old486_cells_preserved=True,old162_IID_seed_first_preserved=True,new_statistics_generated=False,new_three_view_acceptance=0))
write('C10_BINDING_TEMPLATE.json',dict(stage=C70.relative_to(R).as_posix(),science_sha256=sha(C70/'FILES_SHA256.json'),execution_sha256=sha(C70/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),inventory_sha256=sha(C70/'inventory_actual170_Full100refs.json'),adoption=None,adoption_sha256=None))
diff=''.join(difflib.unified_diff(panel_old.splitlines(True),panel_new.splitlines(True),fromfile='C60/panels.py',tofile='C70/panels.py'))+'\n'+''.join(difflib.unified_diff((O/'verify_numeric.py').read_text(encoding='utf8').splitlines(True),numeric.splitlines(True),fromfile='C60/verify_numeric.py',tofile='C70/verify_numeric.py'))+'\n'+''.join(difflib.unified_diff((O/'build.py').read_text(encoding='utf8').splitlines(True),(H/'build.py').read_text(encoding='utf8').splitlines(True),fromfile='C60/build.py',tofile='C70/build.py'))
with (H/'SOURCE_DIFF.patch').open('x',encoding='utf8') as f:f.write(diff)
write('SOURCE_REUSE.json',dict(original_C60_builder_sha256=sha(O/'build.py'),original_C60_panels_sha256=sha(O/'panels.py'),original_C60_numeric_sha256=sha(O/'verify_numeric.py'),numeric_guard_changes=replacements,numeric_guard_normalized_body_exact=True,verify_aggregate_function_byte_exact=True,panels_only_expected_scene_and_message_rebind=True,original_statistic_summarize_loaded_from_evidence=True,strict_receipt_join_and_Full_normalizer_original=True,old_statistics_not_recomputed_during_preparation=True,future_actual_adoption_sha256=None))
print(json.dumps(dict(status='SOURCE_PREPARED_NO_TABLE_OUTPUT',input_pins=len(pins),future_adoption=None,expected_ids=expected)))
