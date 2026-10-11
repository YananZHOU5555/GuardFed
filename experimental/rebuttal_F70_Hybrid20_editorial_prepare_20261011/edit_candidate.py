"""Bounded editorial update; formats saved statistics, never recomputes them."""
from pathlib import Path
import hashlib, json, re, difflib

BASE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REV = BASE / 'docs/server_deployment_20260923/revision_20260923'
PARENT = REV / 'rebuttal_working_F60_FL81_gradient64_20261011.md'
OUT = REV / 'rebuttal_working_F70_FL81_Hybrid20_20261011.md'
F = BASE / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_seven_scenes70_20261011'
H = BASE / 'outputs/guardfed_tables/celeba_hybrid_three_view_two_scenes20_20261011'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj): (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
expected = {
 PARENT: 'e76df8e53b7379a437bf09426c4b576443022f7766ae77d4ff444c456fd650b6',
 F/'ROOT_VERIFICATION.json': 'b3af4a2a55d92e1d611822ebbba7e7af65f49c57aa659aa327e9bf4801f20535',
 F/'TABLES.md': 'f9bf31190f0fb9e8e3be65fa60d4ffb2fb500764ccdd6e05135c9991adfaa53e',
 F/'tables.json': '5cdc61b2c75a8571bfb3e51b5d31517a3c3139b110bee802c17688d0f493e94d',
 H/'ROOT_VERIFICATION.json': 'c221f009c6cbbddd1c8046d049856d955be4de558e31b276069a2f7b5e22791c',
 H/'tables.json': '1b282e89973368d85f915856db94fcc1bc9511b7adedd1f7b0608a189b2ceb00',
 H/'TABLES.md': 'a9510a7c1e13fb7dbfceccd068997a0ff32079dc452ae2d900a598b458b54c66',
 H/'ENVIRONMENT_SCOPE.json': '62b6f9c1d4d092e3a31aa6ccce6a2bd77b2c7b9ac0aaedcc55eb6301c5edfb53',
 BASE/'tmp/fl_native_after77_SDFA9_prepared_20261011/ROOT_ADOPTION_REVIEW.json': '904e3ee036dc4df66459f804db1903211b873ffb5de714ccb97f3900e39c824a',
 BASE/'tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json': 'a69fd80942b585a89a64d83b306ec121a1d300957c91d321eb3abf1340c20781',
 BASE/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T014706Z/ROOT_DELTA_VERIFICATION.json': '4ae0a66ca2cc48289b9a7a18272b83652c73c9da16b2d65989ecd5a1166d002e',
}
for p,s in expected.items(): assert sha(p)==s,(p,sha(p))
ft,ht,fr,he = read(F/'tables.json'),read(H/'tables.json'),read(F/'ROOT_VERIFICATION.json'),read(H/'ENVIRONMENT_SCOPE.json')
assert (fr['complete_scenes'],fr['paired_models'],fr['records'])==(7,70,140)
assert fr['old120_record_objects_order_exact'] and fr['old972_statistical_scalars_exact'] and fr['old486_display_cells_preserved']
assert (ht['complete_scenes'],ht['unique_checkpoints'])==(2,20)
assert he['native_raw_exact_all_records'] and he['replay_devices']=={'cpu':20}
facts=[]; signs=[]
def fmt(row,metric,path,signed=False):
 v=row[metric]; digits=3 if metric=='accuracy_pct' else 5
 text=f"{v['mean']:+.{digits}f}" if signed else f"{v['mean']:.{digits}f}"
 text+=f" ± {v['sample_sd_ddof1']:.{digits}f}"
 facts.append({'source':str((F if path.startswith('F:') else H)/'tables.json'),'pointer':path[2:]+'/'+metric,'value':v,'formatted':text})
 return text
metrics=['accuracy_pct','aeod','aspd']
frows=[]
for i,p in enumerate(ft['panels']):
 r=p['rows'][20]
 assert (r['distribution'],r['attack'],r['variant'],r['n'])==('non-IID','F Flip','minus_F minus Full',[10,9,6][i%3])
 expected_sign=[1,-1,1] if i<3 or i>=6 else [1,1,1]
 actual=[1 if r[m]['mean']>0 else -1 if r[m]['mean']<0 else 0 for m in metrics]
 assert actual==expected_sign
 signs.append({'source':'F','panel':i,'n':r['n'],'signs':actual})
 if i<6:
  frows.append('| '+('Native/shared' if i<3 else 'Raw')+f" | {r['n']} | "+' | '.join(fmt(r,m,f'F:/panels/{i}/rows/20',True) for m in metrics)+' |')
for i in range(3):
 assert ft['panels'][i]['rows'][20]==ft['panels'][i+6]['rows'][20]
hrows=[]
for off in range(3):
 n=ht['panels'][9+off]['rows'][0]; r=ht['panels'][12+off]['rows'][0]; c=ht['panels'][15+off]['rows'][0]
 assert n==r and n['n']==[10,9,6][off]
 assert all(c[m]['mean']<n[m]['mean'] for m in metrics)
 signs.append({'source':'Hybrid','n':n['n'],'shared_vs_native_mean_directions':['lower','lower','lower']})
for i,label in [(9,'Native/raw'),(15,'Shared calibration')]:
 r=ht['panels'][i]['rows'][0]
 hrows.append('| '+label+' | '+' | '.join(fmt(r,m,f'H:/panels/{i}/rows/0') for m in metrics)+' |')

original=PARENT.read_bytes(); text=original.decode('utf-8'); edits=[]
def change(old,new):
 global text
 assert old!=new and text.count(old)==1,(old[:100],text.count(old))
 text=text.replace(old,new); edits.append({'old':old,'new':new,'count':1})
change('accepted six-scene F-deletion, eight-complete-scene FLGMM and IID Benign hybrid-control evidence','accepted seven-scene F-deletion, eight-complete-scene FLGMM and two-scene hybrid-control evidence')
change('360 native and 360 three-view mechanism checkpoints, F60','370 native and 370 three-view mechanism checkpoints, F70')
change('and the ten-checkpoint hybrid scene.','and the 20-checkpoint hybrid comparison in IID Benign and F Flip. Separate FLGMM native acceptance covers 86 new coverage checkpoints plus four reused screen checkpoints; this does not expand its 81-record three-view table.')
change('—and non-IID Benign, each with ten paired seeds. An additional ten-checkpoint IID Benign hybrid-control comparison separates native/raw prediction from shared calibration.','—and non-IID Benign and F Flip, each with ten paired seeds. An additional 20-checkpoint hybrid-control comparison covers IID Benign and F Flip, with ten seeds per scene, and separates native/raw prediction from shared calibration.')
change('A further FLGMM extension has 81 accepted checkpoints:','A further FLGMM extension has 81 accepted three-view checkpoints:')
change('These are reported separately; non-IID S-DFA and Sp-DFA still lack complete accepted ten-seed coverage.','These are reported separately; non-IID S-DFA and Sp-DFA still lack complete accepted ten-seed three-view coverage.')
old='The [six-scene F extension]('+str(F.parent/'three_view_F_six_scenes60_20261011/TABLES.md').replace('\\','/')+') additionally includes non-IID Benign (α=5) with ten paired seeds under the same checkpoint and prediction-view rules.'
new='The [seven-scene F extension]('+str(F/'TABLES.md').replace('\\','/')+') additionally includes non-IID Benign and F Flip (α=5), each with ten paired seeds under the same checkpoint and prediction-view rules.'
change(old,new)
change('The other four non-IID F scenes and four other image controls remain unfinished;','The other three non-IID F scenes and four other image controls remain unfinished;')
change('U100 + C100 + A100 + F60 = 360 deletion checkpoints','U100 + C100 + A100 + F70 = 370 deletion checkpoints')
change('both cover these 360/800 deletion checkpoints','both cover these 370/800 deletion checkpoints')
change('Across the six-scene extension, Full has 5 CPU and 55 GPU evaluations, whereas all 60 F-deletion evaluations use CPU. The corresponding training provenance is 59 cu128 and one cu130 Full checkpoint, versus 60 cu128 deletion checkpoints.','Across the seven-scene extension, Full has 5 CPU and 65 GPU evaluations, whereas all 70 F-deletion evaluations use CPU. The corresponding training provenance is 69 cu128 and one cu130 Full checkpoint, versus 70 cu128 deletion checkpoints.')
fblock='The added non-IID F Flip scene shows a mixed trade-off and near-equal ten-seed mean accuracy. Its saved paired differences (minus_F−Full, mean±sample SD, ddof=1) are:\n\n| Prediction view | n | ΔACC (percentage points) | ΔAEOD | ΔASPD |\n|---|---:|---:|---:|---:|\n'+'\n'.join(frows)+'\n\nThe ten-seed native/shared accuracy difference is only +0.006 percentage points, with a paired sample SD of 0.846 percentage points; this descriptive near-tie is not an equivalence test. Removing F lowers mean AEOD but raises mean ASPD in all three native/shared panels. Raw prediction instead gives higher mean accuracy and higher values of both disparities after deletion in each panel. Thus the AEOD direction depends on the prediction rule, while the nine- and six-seed panels are retained rather than selected for a favorable result. Native/shared outputs coincide and are not independent confirmations. These matched-checkpoint results do not establish per-seed dominance, significance, isolated causality or universal necessity of F.\n\n'
anchor='Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes.'
change(anchor,fblock+anchor)
change('Non-IID S-DFA and Sp-DFA therefore remain incomplete. Neither these partial additions nor the hybrid scene are folded into the earlier nine-method denominator.','Non-IID S-DFA and Sp-DFA therefore remain incomplete in the three-view comparison. Separately, native checkpoint acceptance now covers 86 new coverage runs plus the same four reused screen checkpoints, including nine additional non-IID S-DFA seeds (91002–91010). Those nine are not included in the 81-record three-view result or its eight-scene table. Neither these additions nor the hybrid scenes are folded into the earlier nine-method denominator.')
change('All three fixed seed panels exhibit lower mean ACC together with lower mean disparities after shared calibration.','For this Benign scene, all three fixed seed panels exhibit lower mean ACC together with lower mean disparities after shared calibration.')
hblock='The accepted [two-scene hybrid table]('+str(H/'TABLES.md').replace('\\','/')+') adds IID F Flip, giving 20 checkpoints across two complete scenes. For the ten F Flip seeds, the saved means and sample SDs (ddof=1) are:\n\n| Prediction view | ACC (%) | AEOD | ASPD |\n|---|---:|---:|---:|\n'+'\n'.join(hrows)+'\n\nThe fixed nine- and six-seed F Flip panels also have lower mean accuracy and lower mean disparities under shared calibration. This is a calibration trade-off, not evidence that aggregation alone provides the disparity reduction. Native/raw outputs are identical for all 20 checkpoints; the three views of a checkpoint are not independent model replicates. All ten F Flip checkpoints are coverage runs, while the retained Benign seed91001 checkpoint is a reused screen result. The recipe and original validation-selection history remain unchanged, including selection seed91001; the nine- and six-seed subsets do not create an untouched test. All 20 checkpoints retain cu128 training/replay provenance and CPU replay. The saved evidence retains the original Windows failure and diagnostic differences; Linux whole-replay verification and Windows saved-output verification do not establish cross-platform bitwise recalibration. The other eight hybrid scenes remain outside this accepted two-scene table.\n\n'
anchor='The hybrid is a project control, not a faithful implementation of an external named method.'
change(anchor,anchor+'\n\n'+hblock.rstrip())
change('alongside A100, F60, the IID Benign hybrid-control evidence and the gradient search. That fixed Git64 release stops at F50; the new non-IID Benign F-deletion evidence and this F60/FLGMM prose expansion have not yet been published in it.','alongside A100, F70, the two-scene hybrid-control evidence and the gradient search. That fixed Git64 release stops at F50; it does not include the subsequent non-IID F-deletion scenes, the two-scene hybrid extension or this F70/Hybrid20 prose update. This new author-review draft has not yet been published.')
change('FLGMM has 77 new coverage checkpoints plus four reused screen checkpoints; its eight complete scenes contain 80 three-view records, with the non-IID S-DFA seed91001 record retained separately. Complete non-IID S-DFA and Sp-DFA coverage remains pending.','FLGMM native acceptance covers 86 new coverage checkpoints plus four reused screen checkpoints, whereas its accepted three-view evidence still covers 81: 80 in eight complete scenes and the non-IID S-DFA seed91001 record separately. Complete non-IID S-DFA and Sp-DFA three-view coverage remains pending.')
change('The hybrid IID Benign three-view result completes one scene, not the entire method comparison.','The hybrid three-view table completes IID Benign and F Flip (20 checkpoints); eight other scenes remain. No completed 14-job deployment-gate stage or completed expansion is claimed here.')
change("At this draft's accepted cutoff, 360 deletion checkpoints have complete three-view evidence: U100, C100 and A100 each cover all ten cells, and F60 covers all five IID scenes plus non-IID Benign with ten paired seeds per scene. The remaining 440 planned mechanism-control checkpoints comprise 40 runs for the other four non-IID F-deletion scenes and 100 runs for each of four other controls. Native acceptance is also 360/800, leaving 440 without native acceptance; that is not a claim that all 440 remain untrained.","At this draft's accepted cutoff, 370 deletion checkpoints have complete three-view evidence: U100, C100 and A100 each cover all ten cells, and F70 covers all five IID scenes plus non-IID Benign and F Flip with ten paired seeds per scene. The remaining 430 planned mechanism-control checkpoints comprise 30 runs for the other three non-IID F-deletion scenes and 100 runs for each of four other controls. Native acceptance is also 370/800, leaving 430 without native acceptance; that is not a claim that all 430 remain untrained.")
anchor='- [F-deletion extension: five IID scenes plus non-IID Benign, matched raw/native/shared tables]('
change(anchor,'- [Historical F60 extension: five IID scenes plus non-IID Benign, matched raw/native/shared tables](')
anchor='- [Completed 64-run Fed-NGA/Huber validation-search records and retained negative outcomes]('
change(anchor,'- [Current F70 extension: five IID scenes plus non-IID Benign and F Flip, all fixed panels and paired differences]('+str(F/'TABLES.md').replace('\\','/')+').\n'+anchor)
anchor='- [Method, root-update pseudocode, theory, notation and manuscript insertion candidates]('
change(anchor,'- [Current hybrid control: IID Benign and F Flip, 20 checkpoints and three prediction views]('+str(H/'TABLES.md').replace('\\','/')+').\n'+anchor)

candidate=text.encode('utf-8'); reverse=text
for e in reversed(edits):
 assert reverse.count(e['new'])==1
 reverse=reverse.replace(e['new'],e['old'])
assert reverse.encode('utf-8')==original
quotes=lambda b:[x for x in b.splitlines(keepends=True) if x.startswith(b'>')]
blocks=lambda b:re.findall(rb'(?:^>[^\r\n]*(?:\r?\n|$))+',b,re.M)
headers=lambda b:[x for x in b.splitlines(keepends=True) if x.startswith(b'#')]
assert quotes(original)==quotes(candidate) and len(quotes(candidate))==38
assert blocks(original)==blocks(candidate) and len(blocks(candidate))==24
assert headers(original)==headers(candidate)
# Unedited original paragraphs include all old negative results and technical limits.
unchanged=[p for p in original.decode('utf-8').split('\n\n') if p and not any(e['old'] in p or p in e['old'] for e in edits)]
assert all(p in text for p in unchanged)
for term in ['27/32','12/32','identity','virtual','primary','test']:
 assert term in text,term
links=re.findall(r'\]\((E:/[^)]+)\)',text)
assert all(Path(p).exists() for p in links)
assert sha(PARENT)==expected[PARENT]
assert not OUT.exists(), 'Candidate already exists; preserve before further editing'
OUT.write_bytes(candidate)
save('EDITS.json',edits)
(HERE/'DIFF.patch').write_text(''.join(difflib.unified_diff(original.decode('utf-8').splitlines(True),text.splitlines(True),fromfile=str(PARENT),tofile=str(OUT))),encoding='utf-8')
save('SOURCE_PINS.json',[{'path':str(p),'sha256':s,'bytes':p.stat().st_size} for p,s in expected.items()])
save('NUMERICAL_BINDINGS.json',{'kind':'Existing saved mean/SD formatting and directional comparisons only; no statistics recomputed','formatted_pairs':facts,'direction_guards':signs,'F_scope':{'native':370,'views':370,'F_pairs':70,'scenes':7,'remaining_F':30,'other_controls':400},'Hybrid_scope':{'checkpoints':20,'scenes':2,'native_raw_exact':True},'FL_scope':{'native_new':86,'screen_reused_separately':4,'three_views':81,'table_records':80,'complete_scenes':8},'gradient_search':64,'new_science':False})
proof={'status':'PASS_AUTHOR_EDITORIAL_CHECKS_ROOT_REVIEW_PENDING','root_accepted':False,'candidate_path':str(OUT),'candidate_sha256':sha(OUT),'parent_path':str(PARENT),'parent_sha256':sha(PARENT),'edits':len(edits),'exact_original_quote_blocks':24,'exact_original_quote_lines':38,'headings_order_raw_exact':True,'inverse_parent_bytes_exact':True,'old_unaffected_paragraphs_exact':len(unchanged),'existing_local_links_checked':len(links),'saved_mean_SD_pairs_formatted':len(facts),'F_nine_panel_direction_guards':9,'Hybrid_three_panel_mean_direction_guards':3,'source_pins_sha256':sha(HERE/'SOURCE_PINS.json'),'numerical_bindings_sha256':sha(HERE/'NUMERICAL_BINDINGS.json'),'diff_sha256':sha(HERE/'DIFF.patch'),'edit_script_sha256':sha(Path(__file__)),'no_science_or_statistics_execution':True,'old_F60_unchanged':True,'final_test':False,'whole_benchmark_complete':False,'manuscript_applied':False,'published':False}
save('EDITORIAL_VERIFICATION.json',proof)
save('HANDOFF.json',{'status':'CANDIDATE_READY_FOR_ROOT_EDITORIAL_REVIEW','root_accepted':False,'candidate':str(OUT),'candidate_sha256':sha(OUT),'proof_sha256':sha(HERE/'EDITORIAL_VERIFICATION.json'),'scope':'Native/views370; F70 seven scenes; Hybrid20 two scenes; FL86 new native+4 separate screen but views81/table80; gradient64 unchanged. No completed14gate/fullbenchmark/finaltest claim.','reading_order':['DIFF.patch','NUMERICAL_BINDINGS.json','EDITORIAL_VERIFICATION.json'],'limitations':'This script checks editorial identity and saved values, not original scientific validity; accepted roots are pinned. No SSH, F access, models, arrays, scientific checker, STATE or Git.'})
seal={str(p.relative_to(BASE)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='FILES.json'}
seal[str(OUT.relative_to(BASE)).replace('\\','/')]={'sha256':sha(OUT),'bytes':OUT.stat().st_size}
save('FILES.json',seal)
print(json.dumps({'candidate_sha256':sha(OUT),'proof_sha256':sha(HERE/'EDITORIAL_VERIFICATION.json'),'seal_sha256':sha(HERE/'FILES.json'),'edits':len(edits)},indent=2))
