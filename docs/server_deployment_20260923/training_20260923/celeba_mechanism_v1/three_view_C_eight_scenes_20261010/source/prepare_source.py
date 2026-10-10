"""Prepare the exact C80 record-only extension; no future adoption/table output."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
OLD=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_seven_scenes_20261010'
P=R/'tmp/celeba_mechanism_valid_C_after70_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(n,v):
 with (H/n).open('x',encoding='utf-8',newline='\n') as f:f.write(v if isinstance(v,str) else json.dumps(v,indent=2)+'\n')
assert sha(OLD/'ROOT_VERIFICATION.json')=='abab0188adfa2d857316b23a282fd104c50dd282ca00ffed628b78b2accaea70'
root=read(OLD/'ROOT_VERIFICATION.json');assert (root['unique_records'],root['paired_models'],root['complete_scenes'])==(140,70,7)
assert sha(OLD/'source/FILES_SHA256.json')=='758d49ac894a423f151f64cb0619f630bb6fdb697d9e8d16949b163ba09f336e'
assert sha(P/'PACKAGE_SHA256.json')=='370e75fc5e015048c3cba5c8c13e856a8bbcf8b5a8a46cb158a0468981303159'
prior=R/'tmp/celeba_mechanism_valid_C_after60_20261010/execution_candidate/backups/incremental_20261010T015546Z/ROOT_ADOPTION_REVIEW.json'
assert sha(prior)=='7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'
reb={};diff=[];proof={}
for kind in ('build','panels','verify_numeric'):
 source=OLD/'source'/(kind+'.py');before=source.read_text(encoding='utf-8');after=before;edits=[]
 def change(a,b,count=None):
  global after
  n=after.count(a);assert n and (count is None or n==count),(kind,a,n,count)
  edits.append([a,b,n]);after=after.replace(a,b)
 if kind=='build':
  # Rename stage variables simultaneously through nonnumeric placeholders.
  for a,b in [('C60','PREVIOUS_STAGE_TOKEN'),('C70','CURRENT_STAGE_TOKEN'),('PREVIOUS_STAGE_TOKEN','C70'),('CURRENT_STAGE_TOKEN','C80')]:change(a,b)
  change('three_view_C_six_scenes_20261010','three_view_C_seven_scenes_20261010')
  change('celeba_mechanism_valid_C_after56_20261010','PREVIOUS_AFTER_STAGE_TOKEN')
  change('celeba_mechanism_valid_C_after60_20261010','celeba_mechanism_valid_C_after70_20261010')
  change('PREVIOUS_AFTER_STAGE_TOKEN','celeba_mechanism_valid_C_after60_20261010')
  change("base=module('accepted_C70_helpers',OLD/'build.py')","base=module('accepted_original_C60_helpers',R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010/build.py')")
  change("OLD/'verify_numeric.py'","OLD/'source/verify_numeric.py'")
  change("minus_C_non-IID_F Flip_seed","minus_C_non-IID_FedSA_seed")
  change("('non-IID','Benign'),('non-IID','F Flip')]","('non-IID','Benign'),('non-IID','F Flip'),('non-IID','FedSA')]")
  for a,b in [('==160','==PRIOR_COUNT_TOKEN'),('==170','==180'),('==PRIOR_COUNT_TOKEN','==170'),('inventory_actual160_','inventory_actualPRIOR_TOKEN_'),('inventory_actual170_','inventory_actual180_'),('inventory_actualPRIOR_TOKEN_','inventory_actual170_')]:change(a,b)
  change('len(controls)==70','len(controls)==80')
  change('Only seven complete scenes','Only eight complete scenes')
  change("ROOT_C_AFTER60_INCREMENT_","ROOT_C_AFTER70_INCREMENT_")
  change("==(160,10,170)","==(170,10,180)")
  for a,b in [('Old160','Old170'),('after160','after170'),('Exact160/170','Exact170/180'),('Actual160+10','Actual170+10'),('original160','original170'),('prior160','prior170')]:change(a,b)
  change("'Only fixed after60 stage'","'Only fixed after70 stage'")
  change("'Only fixed after60 stage'","'Only fixed after70 stage'") if "'Only fixed after60 stage'" in after else None
  # Existing capitalization is checked exactly below.
  change("'Only fixed after60 stage'","'Only fixed after70 stage'") if "'Only fixed after60 stage'" in after else None
  change('stage/\'bridge.py\',160,170','stage/\'bridge.py\',170,180')
  change('len(old)==120','len(old)==140');change('Old120','Old140');change('[:120]','[:140]')
  change("==140,'Exact140 unique records required'","==160,'Exact160 unique records required'")
  change("('non-IID','F Flip')]) for p in panels]","('non-IID','FedSA')]) for p in panels]")
  change('Old972','Old1134')
  change('==189','==216');change('Exact189','Exact216')
  change('five IID and two non-IID scenes','five IID and three non-IID scenes')
  change('Seven complete scenes,70 matched pairs','Eight complete scenes,80 matched pairs')
  change('non-IID Benign/F Flip are complete; three non-IID C attack scenes','non-IID Benign/F Flip/FedSA are complete; two non-IID C attack scenes')
  change('seven-scene global mean','eight-scene global mean')
  change("not x.startswith('| non-IID F Flip |')","not x.startswith('| non-IID FedSA |')")
  change('==486','==567');change('Old486','Old567')
  change('C_SEVEN_COMPLETE_SCENES','C_EIGHT_COMPLETE_SCENES')
  change('unique_records=140,paired_models=70,complete_scenes=7','unique_records=160,paired_models=80,complete_scenes=8')
  change("nonIID_complete_scenes=['Benign','F Flip']","nonIID_complete_scenes=['Benign','F Flip','FedSA']")
  change('old120_records','old140_records');change('old972_statistics','old1134_statistics');change('old486_cells','old567_cells')
  change('records=140,mean_SD_scalars=1134,cells=567,count_metrics=1260','records=160,mean_SD_scalars=1296,cells=648,count_metrics=1440')
 elif kind=='panels':
  change("('non-IID','Benign'),('non-IID','F Flip')]","('non-IID','Benign'),('non-IID','F Flip'),('non-IID','FedSA')]")
  change('Only five IID plus non-IID Benign/F Flip10 scenes','Only five IID plus non-IID Benign/F Flip/FedSA10 scenes')
 else:
  change('==140','==160');change("('non-IID','F Flip')}","('non-IID','F Flip'),('non-IID','FedSA')}")
  change('len(rows)==21','len(rows)==24');change('len(errors)==1134','len(errors)==1296')
  change('metric_checks==1260 and count_checks==3360','metric_checks==1440 and count_checks==3840')
  change('==567','==648');change('seven-scene','eight-scene')
 ast.parse(after)
 def fn(s,n):return ast.get_source_segment(s,next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==n))
 if kind=='panels':assert fn(before,'aggregate_panels')==fn(after,'aggregate_panels');proof['aggregate_panels_source_exact']=True
 if kind=='verify_numeric':assert fn(before,'verify_aggregate')==fn(after,'verify_aggregate');proof['verify_aggregate_source_exact']=True
 reb[kind]=dict(source=source.relative_to(R).as_posix(),sha256=sha(source),effective_sha256=hashlib.sha256(after.encode()).hexdigest(),replacements=edits)
 diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='accepted_C70/'+kind+'.py',tofile='effective_C80/'+kind+'.py'))
 write(kind+'.py','"""Thin entry reusing pinned accepted C70 source; no scientific metric implementation."""\nfrom loader import load,expose\n_body=load('+repr(kind)+')\nexpose(_body,globals())\nif __name__=="__main__":\n    _body.main()\n' if kind!='panels' else '"""Original C70 panels with only exact eight-scene expectation added."""\nfrom loader import load,expose\n_body=load("panels")\nexpose(_body,globals())\n')
write('REBINDS.json',reb);write('SOURCE_DIFF.patch',''.join(diff))
inputs=read(OLD/'source/INPUTS.json');pins=dict(inputs['files'])
extra=[OLD/'ROOT_VERIFICATION.json',OLD/'ACTUAL_FILES_SHA256.json',OLD/'source/FILES_SHA256.json',prior]+list((OLD/'snapshot').iterdir())+[OLD/'source'/n for n in ['build.py','panels.py','verify_numeric.py']]+[P/n for n in ['FILES_SHA256.json','execution_candidate/EXECUTION_SOURCE_SHA256.json','inventory_actual180_Full100refs.json','bridge.py','HANDOFF.json']]+[R/'tmp/celeba_mechanism_valid_C_after60_20261010/inventory_actual170_Full100refs.json']
for p in extra:
 if p.is_file():pins[p.relative_to(R).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
for n,pin in pins.items():assert sha(R/n)==pin['sha256'] and (R/n).stat().st_size==pin['bytes']
expected=[f'minus_C_non-IID_FedSA_seed{s}' for s in range(91001,91011)]
write('INPUTS.json',dict(status='SOURCE_PREPARED_WAIT_ACTUAL_AFTER70_C10_ROOT_ADOPTION',files=pins,full900_path=inputs['full900_path'],prior170_adoption=prior.relative_to(R).as_posix(),prior170_adoption_sha256=sha(prior),old_C70_root_sha256=sha(OLD/'ROOT_VERIFICATION.json'),expected_added_ids=expected,new_statistics_generated=False,new_three_view_acceptance=0))
write('C10_BINDING_TEMPLATE.json',dict(stage=P.relative_to(R).as_posix(),science_sha256=sha(P/'FILES_SHA256.json'),execution_sha256=sha(P/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),inventory_sha256=sha(P/'inventory_actual180_Full100refs.json'),adoption=None,adoption_sha256=None))
write('SOURCE_REUSE.json',dict(proof,original_statistics=True,original_strict_receipt_join=True,original_Full_normalizer=True,source_inputs=len(pins),expected_records=160,expected_pairs=80,expected_scenes=8,expected_mean_SD_scalars=1296,expected_display_cells=648,expected_metrics=1440,expected_count_checks=3840,old140_records_and_order_preserved=True,old1134_statistics_preserved=True,old567_display_cells_preserved=True,old162_IID_aggregate_preserved=True,no_eight_scene_aggregate=True,future_adoption_sha256=None))
print(json.dumps({'status':'PREPARED_SOURCE_NO_OUTPUT_NO_FUTURE_ADOPTION','input_pins':len(pins)}))
