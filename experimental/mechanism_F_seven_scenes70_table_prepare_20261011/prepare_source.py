"""F60 -> F70 source/identity preparation only; no statistics, F reads or SSH."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/celeba_F_six_scenes60_table_prepare_20261011'
OLD='docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_six_scenes60_20261011'
NEW='docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_seven_scenes70_20261011'
ROOT360='tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011/ROOT_ADOPTION.json'
ROOT370='tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
names=['build.py','binding.py','panels.py','verify_saved.py','verify_numeric.py','root_adopt_actual.py']
sources={n:(P/n).read_bytes().decode() for n in names};edits={n:[] for n in names}
def change(n,a,b):
 assert a in sources[n],(n,a)
 count=sources[n].count(a);sources[n]=sources[n].replace(a,b);edits[n].append(dict(old=a,new=b,count=count))
def simultaneous(n,pairs):
 import re
 old=sources[n];mapping=dict(pairs)
 for a in mapping:assert a in old,(n,a)
 pattern='|'.join(re.escape(a) for a in sorted(mapping,key=len,reverse=True))
 sources[n]=re.sub(pattern,lambda m:mapping[m.group()],old)
 edits[n].append(dict(simultaneous=pairs))
oldroot=read(R/OLD/'ROOT_VERIFICATION.json')
assert sha(R/OLD/'ROOT_VERIFICATION.json')=='cbba6a6cdfb9b861202422f867c847800474961cd02784dd207f87f51c386164'
assert oldroot['records']==120 and oldroot['paired_models']==60 and oldroot['original_fsum_verification']['mean_sd_scalars']==972
prior=read(R/ROOT360);prior_index=prior['records_index_path'];prior_index_sha=prior['records_index_sha256']
assert sha(R/ROOT360)=='ca558d6e558d26afe6b172b65eded83b51ad02eedddc0f9e2d2cbadf7cdf5a11'
root=read(R/ROOT370);assert sha(R/ROOT370)=='a69fd80942b585a89a64d83b306ec121a1d300957c91d321eb3abf1340c20781'
assert (root['prior_accepted'],root['new_accepted'],root['cumulative_accepted'])==(360,10,370)
assert root['records_index_sha256']=='9d8c0651c453b4d3676c7bf63b541b7fd132cca1e9ae5391b3614a270919cedc'
IDS=[f'minus_F_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
assert root['accepted_new_ids']==IDS and root['original360_unchanged']
assert root['pending_native_closed_ids']==['minus_F_non-IID_FedSA_seed91001']
for n in ('build.py','root_adopt_actual.py'):
 if n=='build.py':change(n,'three_view_F_IID_five_scenes50_20261011','three_view_F_six_scenes60_20261011')
 else:simultaneous(n,[('three_view_F_six_scenes60_20261011','three_view_F_seven_scenes70_20261011'),('three_view_F_IID_five_scenes50_20261011','three_view_F_six_scenes60_20261011')])
n='binding.py'
for a,b in [('Exact accepted350 + ten non-IID Benign records -> six F scenes','Exact accepted360 + ten non-IID F Flip records -> seven F scenes'),('tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json',ROOT360),('3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4',sha(R/ROOT360)),('tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/MECHANISM350_INDEX.json',prior_index),('5b24956b4f14707e2a284fe73ffb538545b4aad218c833e31427720402ff02b3',prior_index_sha),('minus_F_non-IID_Benign_seed','minus_F_non-IID_F Flip_seed'),('(350,10,360)','(360,10,370)'),('Exact350+10 replay360','Exact360+10 replay370'),('original350_unchanged','original360_unchanged'),('Exactly ten non-IID Benign','Exactly ten non-IID F Flip'),("==350,'Parent350","==360,'Parent360"),("==360,'Old350 order or exact360","==370,'Old360 order or exact370"),('Accepted350 parent','Accepted360 parent'),('>=360','>=370'),('must cover360','must cover370'),('replay360 is not bound','replay370 is not bound'),('ACTUAL_REPLAY360_BOUND_FOR_F_SIX_SCENES60','ACTUAL_REPLAY370_BOUND_FOR_F_SEVEN_SCENES70'),('parent350 drift','parent360 drift')]:change(n,a,b)
old_selector="row['distribution']=='IID' and row['attack'] in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')"
new_selector="(row['distribution'],row['attack']) in {('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA'),('non-IID','Benign')}"
n='build.py'
for a,b in [('Read adopted non-IID Benign10 receipts; preserve old F50','Read adopted non-IID F Flip10 receipts; preserve old F60'),("full['attack']==row['attack']=='Benign'","full['attack']==row['attack']=='F Flip'"),('Exact paired seed/non-IID Benign/alpha5','Exact paired seed/non-IID F Flip/alpha5'),('len(old_records)==100','len(old_records)==120'),('Old five-scene100','Old six-scene120'),("len({r['id'] for r in records})==120","len({r['id'] for r in records})==140"),('Exact six-scene120','Exact seven-scene140'),("checks['mean_sd_scalars']==972 and checks['receipt_metrics_from_group_counts']==1080","checks['mean_sd_scalars']==1134 and checks['receipt_metrics_from_group_counts']==1260"),(old_selector,new_selector),('Old810 five-scene','Old972 six-scene'),('[:100]','[:120]'),('Old100 object','Old120 object'),('five IID scenes and non-IID Benign, three views','five IID scenes and non-IID Benign/F Flip, three views'),('non-IID Benign uses alpha5','non-IID scenes use alpha5'),('display_checks==486','display_checks==567'),('Only the five IID scenes and non-IID Benign seeds91001–91010 enter this six-scene table.','Only the five IID scenes and non-IID Benign/F Flip seeds91001–91010 enter this seven-scene table. The extra observed non-IID FedSA seed91001 is not adopted into this scope.'),('The other four non-IID minus_F scenes','The other three non-IID minus_F scenes'),('display_mean_sd_cells=486,table_record_count=120,preserved_records=120,paired_models=60,old100_record_JSON_bytes_and_order_exact=True,old810_scalars_exact=True,old405_cells_preserved=True','display_mean_sd_cells=567,table_record_count=140,preserved_records=140,paired_models=70,old120_record_JSON_bytes_and_order_exact=True,old972_scalars_exact=True,old486_cells_preserved=True'),('ACTUAL_SIX_F_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING','ACTUAL_SEVEN_F_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING'),('complete_scenes=6,paired_models=60','complete_scenes=7,paired_models=70'),('displayed_records=120,preserved_records=120','displayed_records=140,preserved_records=140'),('actual_F60_root_adoption','actual_F70_root_adoption'),('ACTUAL_F60_SIX_SCENE_TABLE_BUILT_ROOT_REVIEW_PENDING','ACTUAL_F70_SEVEN_SCENE_TABLE_BUILT_ROOT_REVIEW_PENDING'),('records=120,displayed_models=120,stats=972,cells=486,metrics_from_counts=1080,count_checks=2880','records=140,displayed_models=140,stats=1134,cells=567,metrics_from_counts=1260,count_checks=3360')]:change(n,a,b)
n='panels.py'
change(n,"('non-IID','Benign')]","('non-IID','Benign'),('non-IID','F Flip')]")
change(n,'Only exact five IID scenes plus non-IID Benign10','Only exact five IID scenes plus non-IID Benign10/F Flip10')
n='verify_numeric.py'
for a,b in [('==120','==140'),("('non-IID','Benign')}","('non-IID','Benign'),('non-IID','F Flip')}"),('len(rows)==18','len(rows)==21'),('len(errors)==972','len(errors)==1134'),('metric_checks==1080 and count_checks==2880','metric_checks==1260 and count_checks==3360')]:change(n,a,b)
n='verify_saved.py';oldsel="r['distribution']=='IID' and r['attack'] in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')";newsel="(r['distribution'],r['attack']) in {('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA'),('non-IID','Benign')}"
for a,b in [('len(records)==120','len(records)==140'),("len({r['id'] for r in records})==120","len({r['id'] for r in records})==140"),('Exact120 saved','Exact140 saved'),('records[:100]','records[:120]'),('Old100 objects','Old120 objects'),('records[100:]','records[120:]'),('Exact new non-IID Benign','Exact new non-IID F Flip'),('==(6,60,120,120)','==(7,70,140,140)'),("checks['mean_sd_scalars']==972 and checks['receipt_metrics_from_group_counts']==1080 and checks['base_confusion_counts_structurally_checked']==2880","checks['mean_sd_scalars']==1134 and checks['receipt_metrics_from_group_counts']==1260 and checks['base_confusion_counts_structurally_checked']==3360"),('actual_F60_root_adoption','actual_F70_root_adoption'),("('non-IID','Benign')]","('non-IID','Benign'),('non-IID','F Flip')]"),(oldsel,newsel),('Old five-scene record','Old six-scene record'),('Old five-scene810','Old six-scene972'),('[:100]','[:120]'),('Old100 serialized','Old120 serialized'),('cells==486','cells==567')]:change(n,a,b)
n='root_adopt_actual.py'
for a,b in [('already checked F60','already checked F70'),('ACTUAL_REPLAY360_BOUND_FOR_F_SIX_SCENES60','ACTUAL_REPLAY370_BOUND_FOR_F_SEVEN_SCENES70'),('(350,10,360)','(360,10,370)'),('original350_unchanged','original360_unchanged'),('==(972,486,1080,2880)','==(1134,567,1260,3360)'),('records[:100]','records[:120]'),('len(records)==120','len(records)==140'),(oldsel,newsel),('cf89d438b80f48df617c994084a69d8d913cf7dc35c65c762d8800517b89590b',sha(R/OLD/'ROOT_VERIFICATION.json')),('c7a6e30ef60485ee4955052ec26abfa2b709e9bcde92ff6e4a18fa871f9da44f',oldroot['files']['tables.json']['sha256']),("==('non-IID','Benign')","==('non-IID','F Flip')"),('ROOT_F60_SIX_SCENES','ROOT_F70_SEVEN_SCENES'),('complete_scenes=6,paired_models=60,records=120,preserved_pairs=60,old100_record_objects_order_exact=True,old810_statistical_scalars_exact=True,old405_display_cells_preserved=True','complete_scenes=7,paired_models=70,records=140,preserved_pairs=70,old120_record_objects_order_exact=True,old972_statistical_scalars_exact=True,old486_display_cells_preserved=True'),('actual_views360','actual_views370'),('nonIID_F_scenes_remaining=4','nonIID_F_scenes_remaining=3'),('new_nonIID_Benign_native10_rows','new_nonIID_FFlip_native10_rows'),('Five IID scenes plus non-IID Benign only.','Five IID scenes plus non-IID Benign and F Flip only.'),('actual future views360 root','actual views370 root'),('Accepted five IID scenes plus non-IID Benign Full','Accepted five IID scenes plus non-IID Benign/F Flip Full'),('Four non-IID F scenes','Three non-IID F scenes'),('ROOT_COMPACT_F60','ROOT_COMPACT_F70'),('records=120,paired=60','records=140,paired=70')]:change(n,a,b)
for n,s in sources.items():
 compile(s,str(H/n),'exec');original=(P/n).read_bytes().decode();delta=list(difflib.ndiff(original.splitlines(True),s.splitlines(True)))
 assert ''.join(difflib.restore(delta,1)).encode()==(P/n).read_bytes()
 assert ''.join(difflib.restore(delta,2))==s
 assert not (H/n).exists();(H/n).write_bytes(s.encode())
inputs=read(P/'SOURCE_INPUTS.json');inputs['C_panels']=(H/'panels.py').relative_to(R).as_posix();inputs['C_numeric']=(H/'verify_numeric.py').relative_to(R).as_posix();inputs['old_F60_root']=OLD+'/ROOT_VERIFICATION.json'
for p in [*(P/n for n in names),P/'FILES_SHA256.json',R/OLD/'ROOT_VERIFICATION.json',R/ROOT360,R/prior_index,H/'panels.py',H/'verify_numeric.py']:
 inputs['files'][p.relative_to(R).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
# Source preparation reads the root's accepted metadata pins, not table/model arrays or F files.
for name in ('records.json','tables.json','TABLES.md'):
 inputs['files'][OLD+'/'+name]=oldroot['files'][name]
(H/'SOURCE_INPUTS.json').write_text(json.dumps(inputs,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
(H/'INVERSE_EDITS.json').write_text(json.dumps(edits,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
(H/'SOURCE_DIFF.patch').write_text(''.join(''.join(difflib.unified_diff((P/n).read_bytes().decode().splitlines(True),sources[n].splitlines(True),fromfile=P.name+'/'+n,tofile=H.name+'/'+n)) for n in names),encoding='utf8',newline='')
template=dict(status='SOURCE_ONLY_NOT_BOUND',root_adopted=False,required_status='ACTUAL_REPLAY370_BOUND_FOR_F_SEVEN_SCENES70',adoption=None,adoption_sha256=None,index=None,index_sha256=None,native_root=None,native_root_sha256=None,root_status=None,source_seal_sha256=None)
(H/'ROOT_BINDING.template.json').write_text(json.dumps(template,indent=2)+'\n')
candidate=dict(template,status=template['required_status'],root_adopted=True,adoption=ROOT370,adoption_sha256=sha(R/ROOT370),index=root['records_index_path'],index_sha256=root['records_index_sha256'],native_root=root['native_root_path'],native_root_sha256=root['native_root_sha256'],root_status=root['status'],note='Actual replay370 identity only. F70 table/source review/build/verify/adoption not executed. source seal bound only after source checks. Full raw inspection has extra FedSA1; exact accepted10 IDs determine this table.')
(H/'ROOT_BINDING.actual_candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
print(json.dumps({'status':'SIX_SOURCES_PREPARED_NOT_EXECUTED','actual_replay_root370':candidate['adoption_sha256'],'future_table_pairs':70,'extra_FedSA1_excluded':True}))
