"""Source-only bounded F50 -> F60 derivation; never imports or runs table code."""
from pathlib import Path
import ast,hashlib,json,difflib
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
P=R/'tmp/celeba_F_IID_five_scenes50_table_prepared_20261011'
OLD='docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_five_scenes50_20261011'
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
names=['build.py','binding.py','panels.py','verify_saved.py','verify_numeric.py','root_adopt_actual.py']
sources={n:(P/n).read_bytes().decode() for n in names};changes={n:[] for n in names}
def change(n,a,b):
    assert a in sources[n],(n,a)
    sources[n]=sources[n].replace(a,b);changes[n].append(dict(old=a,new=b))
for n in names:
    if 'three_view_F_IID_four_scenes40_20261011' in sources[n]:change(n,'three_view_F_IID_four_scenes40_20261011','three_view_F_IID_five_scenes50_20261011')

n='binding.py'
for a,b in [
 ('Exact accepted340 + ten Sp-DFA records -> five F scenes','Exact accepted350 + ten non-IID Benign records -> six F scenes'),
 ('remaining_F_SDFA10_root_adoption','remaining_F_SpDFA10_root_adoption'),
 ('80a7f38890c2a7f8c75c0472489315bb2acb9e43c9e28d686a799cfea9709a02','3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4'),
 ('MECHANISM340_INDEX','MECHANISM350_INDEX'),
 ('a92ca53bf7beb7fac0951381ba1807ff0ee4f6d74d040826b0137cfb700d1d99','5b24956b4f14707e2a284fe73ffb538545b4aad218c833e31427720402ff02b3'),
 ('minus_F_IID_Sp-DFA_seed','minus_F_non-IID_Benign_seed'),
 ('(340,10,350)','(350,10,360)'),('Exact340+10 replay350','Exact350+10 replay360'),
 ('original340_unchanged','original350_unchanged'),('Exactly ten IID Sp-DFA seeds','Exactly ten non-IID Benign seeds'),
 ('==340','==350'),('Parent340','Parent350'),('==350,\'Old340','==360,\'Old350'),('exact350','exact360'),('Accepted340','Accepted350'),
 ('>=350','>=360'),('must cover350','must cover360'),('replay350','replay360'),
 ('ACTUAL_REPLAY350_BOUND_FOR_F_IID_FIVE_SCENES50','ACTUAL_REPLAY360_BOUND_FOR_F_SIX_SCENES60'),('parent340','parent350')]:change(n,a,b)

n='build.py'
for a,b in [
 ('Read adopted Sp-DFA10 receipts; preserve old F40','Read adopted non-IID Benign10 receipts; preserve old F50'),
 ("full['distribution']==row['distribution']=='IID' and full['attack']==row['attack']=='Sp-DFA' and row['actual_alpha']==5000", "full['distribution']==row['distribution']=='non-IID' and full['attack']==row['attack']=='Benign' and row['actual_alpha']==5"),
 ('Exact paired seed/IID Sp-DFA/alpha5000','Exact paired seed/non-IID Benign/alpha5'),
 ('len(old_records)==80','len(old_records)==100'),('Old four-scene80','Old five-scene100'),
 ("len({r['id'] for r in records})==100","len({r['id'] for r in records})==120"),('Exact five-scene100','Exact six-scene120'),
 ("checks['mean_sd_scalars']==810 and checks['receipt_metrics_from_group_counts']==900","checks['mean_sd_scalars']==972 and checks['receipt_metrics_from_group_counts']==1080"),
 ("row['attack'] in ('Benign','F Flip','FedSA','S-DFA')","row['distribution']=='IID' and row['attack'] in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')"),
 ('Old648 four-scene','Old810 five-scene'),('[:80]','[:100]'),('Old80 object','Old100 object'),
 ('# CelebA Full–minus_F: IID Benign, F Flip, FedSA, S-DFA and Sp-DFA, three views','# CelebA Full–minus_F: five IID scenes and non-IID Benign, three views'),
 ('IID is frozen Dirichlet alpha5000.','IID is frozen Dirichlet alpha5000; non-IID Benign uses alpha5.'),
 ('display_checks==405','display_checks==486'),
 ('Only IID Benign, IID F Flip, IID FedSA, IID S-DFA and IID Sp-DFA seeds91001–91010 enter this five-scene table.','Only the five IID scenes and non-IID Benign seeds91001–91010 enter this six-scene table.'),
 ('The other five non-IID minus_F scenes','The other four non-IID minus_F scenes'),
 ('display_mean_sd_cells=405,table_record_count=100,preserved_records=100,paired_models=50,old80_record_JSON_bytes_and_order_exact=True,old648_scalars_exact=True,old324_cells_preserved=True','display_mean_sd_cells=486,table_record_count=120,preserved_records=120,paired_models=60,old100_record_JSON_bytes_and_order_exact=True,old810_scalars_exact=True,old405_cells_preserved=True'),
 ('ACTUAL_FIVE_F_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING','ACTUAL_SIX_F_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING'),
 ('complete_scenes=5,paired_models=50','complete_scenes=6,paired_models=60'),
 ('displayed_records=100,preserved_records=100','displayed_records=120,preserved_records=120'),
 ('actual_F50_root_adoption','actual_F60_root_adoption'),
 ('ACTUAL_F50_FIVE_SCENE_TABLE_BUILT_ROOT_REVIEW_PENDING','ACTUAL_F60_SIX_SCENE_TABLE_BUILT_ROOT_REVIEW_PENDING'),
 ('records=100,displayed_models=100,stats=810,cells=405,metrics_from_counts=900,count_checks=2400','records=120,displayed_models=120,stats=972,cells=486,metrics_from_counts=1080,count_checks=2880')]:change(n,a,b)

n='panels.py'
change(n,"('IID','Sp-DFA')]","('IID','Sp-DFA'),('non-IID','Benign')]")
change(n,'Only exact IID Benign10, F Flip10, FedSA10, S-DFA10 and Sp-DFA10 scenes are publishable','Only exact five IID scenes plus non-IID Benign10 are publishable')

n='verify_numeric.py'
for a,b in [('==100','==120'),("('IID','Sp-DFA')}","('IID','Sp-DFA'),('non-IID','Benign')}"),('len(rows)==15','len(rows)==18'),('len(errors)==810','len(errors)==972'),('metric_checks==900 and count_checks==2400','metric_checks==1080 and count_checks==2880')]:change(n,a,b)

n='verify_saved.py'
for a,b in [
 ('len(records)==100', 'len(records)==120'),("len({r['id'] for r in records})==100","len({r['id'] for r in records})==120"),('Exact100 saved','Exact120 saved'),
 ('records[:80]','records[:100]'),('Old80 objects','Old100 objects'),('records[80:]','records[100:]'),('Exact new Sp-DFA','Exact new non-IID Benign'),
 ('==(5,50,100,100)','==(6,60,120,120)'),
 ("checks['mean_sd_scalars']==810 and checks['receipt_metrics_from_group_counts']==900 and checks['base_confusion_counts_structurally_checked']==2400","checks['mean_sd_scalars']==972 and checks['receipt_metrics_from_group_counts']==1080 and checks['base_confusion_counts_structurally_checked']==2880"),
 ('actual_F50_root_adoption','actual_F60_root_adoption'),
 ("('IID','Sp-DFA')]","('IID','Sp-DFA'),('non-IID','Benign')]"),
 ("r['variant']=='minus_F' and r['attack'] in ('Benign','F Flip','FedSA','S-DFA')","r['variant']=='minus_F' and r['distribution']=='IID' and r['attack'] in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')"),
 ('Old four-scene record','Old five-scene record'),
 ("r['attack'] in ('Benign','F Flip','FedSA','S-DFA')","r['distribution']=='IID' and r['attack'] in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')"),
 ('Old four-scene648','Old five-scene810'),('[:80]','[:100]'),('Old80 serialized','Old100 serialized'),
 ("line.startswith('| IID / ')","line.startswith(('| IID / ','| non-IID / '))"),('cells==405','cells==486')]:change(n,a,b)

n='root_adopt_actual.py'
# This original adopter was written after F50 finished and contains actual pins,
# devices and interpretation. F60 must bind those from future checked inputs.
change(n,'already checked F50','already checked F60')
change(n,'import json, hashlib, datetime, shutil','import json, hashlib, datetime, shutil, argparse\n\nparser=argparse.ArgumentParser()\nparser.add_argument("--binding-sha256",required=True)\nparser.add_argument("--verifier-sha256",required=True)\nparser.add_argument("--root-source-review-sha256",required=True)\nargs=parser.parse_args()')
change(n,"d=T/'celeba_mechanism_v1/three_view_F_IID_five_scenes50_20261011'","d=T/'celeba_mechanism_v1/three_view_F_six_scenes60_20261011'")
change(n,"assert not d.exists()","assert not d.exists()\nassert H(b/'ROOT_BINDING.json')==args.binding_sha256\nassert H(b/'ACTUAL_VERIFY.stdout.json')==args.verifier_sha256\nassert H(b/'ROOT_SOURCE_REVIEW.json')==args.root_source_review_sha256\nbinding=R(b/'ROOT_BINDING.json')\nassert binding['root_adopted'] is True and binding['status']=='ACTUAL_REPLAY360_BOUND_FOR_F_SIX_SCENES60'\nassert H(b/'FILES_SHA256.json')==binding['source_seal_sha256']\nfor name,pin in R(b/'FILES_SHA256.json')['files'].items():\n    assert H(b/name)==pin['sha256'] and (b/name).stat().st_size==pin['bytes']\nfor key in ('adoption','index','native_root'):\n    assert H(ROOT/binding[key])==binding[key+'_sha256']\nactualroot=R(ROOT/binding['adoption'])\nassert actualroot['records_index_sha256']==binding['index_sha256'] and actualroot['native_root_sha256']==binding['native_root_sha256']\nassert (actualroot['prior_accepted'],actualroot['new_accepted'],actualroot['cumulative_accepted'])==(350,10,360) and actualroot['original350_unchanged'] is True")
for a,b in [
 ('==(810,405,900,2400)','==(972,486,1080,2880)'),('records[:80]','records[:100]'),('len(records)==100','len(records)==120'),
 ("r['attack']!='Sp-DFA'","r['distribution']=='IID' and r['attack'] in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')"),
 ("assert H(b/'ROOT_BINDING.json')=='3c259365f013aa9bf23ee279438c95b2e64644a7c8e0f20d81921734b4fed440'\nassert H(ROOT/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json')=='3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4'","assert H(old/'ROOT_VERIFICATION.json')=='cf89d438b80f48df617c994084a69d8d913cf7dc35c65c762d8800517b89590b'\nassert H(old/'tables.json')=='c7a6e30ef60485ee4955052ec26abfa2b709e9bcde92ff6e4a18fa871f9da44f'"),
 ("if r['attack']=='Sp-DFA'","if (r['distribution'],r['attack'])==('non-IID','Benign')"),
 ('ROOT_F50_FIVE_IID_SCENES_COUNTS_PAIRED_STATISTICS_DISPLAY_AND_PRIOR_ROWS_PASS','ROOT_F60_SIX_SCENES_COUNTS_PAIRED_STATISTICS_DISPLAY_AND_PRIOR_ROWS_PASS'),
 ('complete_scenes=5,paired_models=50,records=100,preserved_pairs=50,old80_record_objects_order_exact=True,old648_statistical_scalars_exact=True,old324_display_cells_preserved=True','complete_scenes=6,paired_models=60,records=120,preserved_pairs=60,old100_record_objects_order_exact=True,old810_statistical_scalars_exact=True,old405_display_cells_preserved=True'),
 ("actual_views350_root_sha256=H(ROOT/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json'),actual_views350_index_sha256=H(ROOT/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/MECHANISM350_INDEX.json')","actual_views360_root_sha256=binding['adoption_sha256'],actual_views360_index_sha256=binding['index_sha256']"),
 ("Full_replay_devices={'cpu':3,'cuda:0':47},minus_F_replay_devices={'cpu':50},native_shared_metrics_and_counts_exact=True,nonIID_F_scenes_remaining=5","Full_replay_devices=tables['replay_devices']['Full'],minus_F_replay_devices=tables['replay_devices']['minus_F'],native_shared_metrics_and_counts_exact=R(b/'checks.json')['native_shared_metrics_and_counts_exact'],nonIID_F_scenes_remaining=4"),
 ('new_SpDFA_native10_rows','new_nonIID_Benign_native10_rows'),
 ('Five IID scenes only. SpDFA native/shared minusF improves ACC and AEOD means but worsens ASPD; raw both disparities worsen while ACC slightly rises. Retain all directions; no significance, universal necessity or isolated-causality claim. Mixed replay devices/selection history retained.','Five IID scenes plus non-IID Benign only. All fixed10/9/6 panels and all signed paired differences retained; no cross-scene aggregate, significance, universal necessity or isolated-causality claim. Mixed replay devices/selection history retained.'),
 ("root_reader_note='An earlier inline metadata adopter failed to parse before executing any statement. No outputs or scientific results were affected; this saved adopter is the actual adoption source.'","root_reader_note='Adopts only outputs pinned to the actual future views360 root and independently saved verifier; source preparation itself is not statistical acceptance.'"),
 ('Accepted five-IID-scene Full versus minus_F table.','Accepted five IID scenes plus non-IID Benign Full versus minus_F table.'),
 ('Native/shared identical here. Five non-IID F scenes and four other controls remain incomplete.','See actual checks for native/shared equality. Four non-IID F scenes and four other controls remain incomplete.'),
 ('ROOT_COMPACT_F50_TABLE_DELIVERY','ROOT_COMPACT_F60_TABLE_DELIVERY'),('records=100,paired=50','records=120,paired=60')]:change(n,a,b)

# Static source-only checks: compile source strings, reverse each exact edit, and
# compare original scientific bodies; no import, main(), statistics or checker.
for n,s in sources.items():
    compile(s,str(D/n),'exec');old=(P/n).read_bytes().decode();forward=old
    for ch in changes[n]:forward=forward.replace(ch['old'],ch['new'])
    assert forward==s,('only declared edits',n)
    ndiff=list(difflib.ndiff(old.splitlines(True),s.splitlines(True)))
    assert ''.join(difflib.restore(ndiff,1))==old and ''.join(difflib.restore(ndiff,2))==s,('inverse',n)
    assert not (D/n).exists();(D/n).write_bytes(s.encode())
def nodes(n,klass):return [x for x in ast.walk(ast.parse(sources[n])) if isinstance(x,klass)]
def seg(s,node):return ast.get_source_segment(s,node)
oldbuild=(P/'build.py').read_text(encoding='utf8');newbuild=sources['build.py']
oldloop=next(x for x in ast.walk(ast.parse(oldbuild)) if isinstance(x,ast.For) and ast.unparse(x.target)=='rid')
newloop=next(x for x in ast.walk(ast.parse(newbuild)) if isinstance(x,ast.For) and ast.unparse(x.target)=='rid')
oldlooptext=seg(oldbuild,oldloop);newlooptext=seg(newbuild,newloop)
for ch in reversed(changes['build.py']):newlooptext=newlooptext.replace(ch['new'],ch['old'])
assert newlooptext==oldlooptext
oldnum=(P/'verify_numeric.py').read_text(encoding='utf8');newnum=sources['verify_numeric.py']
for target in ['row','record']:
    a=next(x for x in ast.walk(ast.parse(oldnum)) if isinstance(x,ast.For) and ast.unparse(x.target)==target)
    b=next(x for x in ast.walk(ast.parse(newnum)) if isinstance(x,ast.For) and ast.unparse(x.target)==target)
    assert seg(oldnum,a)==seg(newnum,b),target
for name in ['need','write','function','variant_module']:
    a=next(x for x in ast.parse(oldbuild).body if isinstance(x,ast.FunctionDef) and x.name==name)
    b=next(x for x in ast.parse(newbuild).body if isinstance(x,ast.FunctionDef) and x.name==name)
    assert seg(oldbuild,a)==seg(newbuild,b)
inputs=read(P/'SOURCE_INPUTS.json')
inputs['C_panels']=(D/'panels.py').relative_to(R).as_posix();inputs['C_numeric']=(D/'verify_numeric.py').relative_to(R).as_posix()
inputs['old_F50_root']=OLD+'/ROOT_VERIFICATION.json'
for p in [*(P/n for n in names),P/'FILES_SHA256.json',*(R/OLD/n for n in ['ROOT_VERIFICATION.json','records.json','tables.json','TABLES.md']),R/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json',R/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/MECHANISM350_INDEX.json',D/'panels.py',D/'verify_numeric.py']:
    inputs['files'][p.relative_to(R).as_posix()]=dict(sha256=H(p.read_bytes()),bytes=p.stat().st_size)
assert inputs['files'][OLD+'/ROOT_VERIFICATION.json']['sha256']=='cf89d438b80f48df617c994084a69d8d913cf7dc35c65c762d8800517b89590b'
assert inputs['files'][OLD+'/tables.json']['sha256']=='c7a6e30ef60485ee4955052ec26abfa2b709e9bcde92ff6e4a18fa871f9da44f'
template=read(P/'ROOT_BINDING.template.json');template['required_status']='ACTUAL_REPLAY360_BOUND_FOR_F_SIX_SCENES60';template['note']='No ROOT_BINDING.json or numeric execution before actual adopted views360 root. Future pins intentionally null. Exact accepted350 parent plus ten non-IID Benign IDs; native>=360 does not enlarge table scope.'
for name,obj in [('SOURCE_INPUTS.json',inputs),('ROOT_BINDING.template.json',template),('INVERSE_EDITS.json',changes)]:
    (D/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
diff=''.join(''.join(difflib.unified_diff((P/n).read_bytes().decode().splitlines(True),sources[n].splitlines(True),fromfile=P.name+'/'+n,tofile=D.name+'/'+n)) for n in names)
(D/'SOURCE_DIFF.patch').write_text(diff,encoding='utf8')
check=dict(status='SOURCE_ONLY_COMPILE_AND_EXACT_METADATA_INVERSE_PASS',scientific_execution=False,statistics_run=False,checker_run=False,source_files=6,
 all_six_sources_declared_metadata_forward_edits_and_ndiff_reverse_restore_original_bytes=True,build_record_loop_only_distribution_attack_alpha_identity_mapping=True,
 numeric_row_mean_sd_body_byte_exact=True,numeric_record_confusion_metric_body_byte_exact=True,original_build_helper_functions_byte_exact=True,
 original_statistic_summarize_source=inputs['evidence'],fixed10_9_6_seed_panels_preserved=True,
 old_scene_selector='distribution IID AND one of five original attacks; cannot include new non-IID Benign',old_scope=dict(records=100,statistical_scalars=810,display_cells=405),
 expected_scope_not_actual=dict(records=120,paired_models=60,scenes=6,statistical_scalars=972,display_cells=486,metrics_from_counts=1080,base_confusion_counts=2880),
 binding_created=False,future_adoption_sha256=None,root_accepted=False,arrays_or_models_read=False,source_input_closure='Prior SOURCE_INPUTS pins retained; new original source and F50 compact metadata hashes added. No model or prediction-array contents read.',
 initial_lookup_note='Two guessed template filenames did not exist; actual ROOT_BINDING.template.json was then read. Read-only discovery only, no source/science execution failure.')
(D/'SOURCE_CHECK.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(check))
