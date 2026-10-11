"""Bounded editorial merge of adopted FL81 evidence; no scientific computation."""
from pathlib import Path
import json, hashlib, re, difflib

R = Path(__file__).resolve().parents[2]
O = Path(__file__).resolve().parent
H = lambda b: hashlib.sha256(b).hexdigest()
parent = R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F50_gradient64_20261011.md'
target = R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F50_FL81_gradient64_20261011.md'
original = parent.read_bytes()
assert H(original) == 'b6e1c35be9dd0bea4867ff8486c23f7b7d39d0a6966e449dda5ec0664d515670'
pins = {
 'FL_native_root': ('tmp/fl_native_after67_FedSA10_prepared_20261011/ROOT_ADOPTION_REVIEW.json','5fdf135fd13665df6d528818db81765ad3567d5d754b18c538b1772f3994c80d'),
 'FL_three_view_root': ('tmp/fl_FedSA10_three_view_prepared_20261011/ROOT_SCIENTIFIC_ADOPTION.json','adc7726ca05561c6903ebb153e34fb3dc69a1df0118d87e6bf7dfa3e31655799'),
 'FL_table_root': ('outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/ROOT_VERIFICATION.json','bf44b6e8d7a46ea0ffe14077939bd9c3886901c4063c5c6c8ff20a21415cee66'),
 'FL_tables': ('outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/tables.json','707d8d4b806be30e17f2c5e3119833c93994cfba35f2c408990f27d15a983bbd'),
 'FL_table_md': ('outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md','b0dbaee96c41c7e36e6e835213264e2ec46b7ff89e5780ac9e700e3196a866ce'),
 'FL_records': ('outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/records81.json','54a0af5b4f2cf8da2c3e9e2092b347db37bbf0edf9923b6a2eab39291a9cf0f9'),
}
sources = {}
for role, (path, sha) in pins.items():
    b = (R/path).read_bytes(); assert H(b) == sha, role
    if path.endswith('.json'): sources[role] = json.loads(b)
native, root, tr = [sources[k] for k in ['FL_native_root','FL_three_view_root','FL_table_root']]
assert (native['accepted_total'],native['reused_separately']) == (77,4)
assert (root['FLGMM_total_three_view_records'],tr['complete_scene_records'],tr['retained_partial_records'],tr['scenes']) == (81,80,1,8)
assert root['Linux_whole_original_saved_check_pass'] and root['Windows_saved_outputs_audit_pass']
assert root['Windows_saved_outputs_audit_fit_calls']==0 and not root['Windows_original47_exact_refit_pass']
tables = sources['FL_tables']; facts = []
def scene(view, panel):
    pi = next(i for i,p in enumerate(tables['panels']) if p['view']==view and p['panel']==panel)
    p = tables['panels'][pi]
    si = next(i for i,s in enumerate(p['scenes']) if (s['distribution'],s['attack'])==('non-IID','FedSA'))
    return p['scenes'][si], f'/panels/{pi}/scenes/{si}'
rows=[]
for view,label in [('native','Native/raw'),('shared_calibration','Shared calibration')]:
    s,pointer=scene(view,'ten'); assert s['n']==10
    displays=[]
    for metric in ['accuracy','aeod','aspd']:
        m=s['metrics'][metric]; displays.append(m['display'])
        facts.append(dict(source='FL_tables',pointer=pointer+'/metrics/'+metric,mean=m['mean'],sample_sd=m['sample_sd'],display=m['display']))
    rows.append('| '+label+' | 10 | '+' | '.join(displays)+' |')
for panel in ['ten','nonselection_nine','matching_six']:
    a,_=scene('native',panel); b,_=scene('shared_calibration',panel); raw,_=scene('raw',panel)
    assert raw['metrics']==a['metrics']
    assert all(b['metrics'][m]['mean']<a['metrics'][m]['mean'] for m in ['accuracy','aeod','aspd'])
records=sources['FL_records']
assert records['retained_partial_ids']==['FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed91001_screen']
assert len(records['records'])==81
assert all(x['same_checkpoint_all_views'] and x['terminal_round']==70 and x['evaluation_split']=='valid' for x in records['records'])

text=original.decode('utf8'); changes=[]
def replace(old,new):
    global text
    assert text.count(old)==1,old[:100]
    text=text.replace(old,new); changes.append({'old':old,'new':new})

replace('This unified working version incorporates the accepted five-scene F-deletion and IID Benign hybrid-control evidence, together with the completed Fed-NGA/Huber validation search, directly into the complete responses.',
        'This unified working version incorporates the accepted five-scene F-deletion, eight-complete-scene FLGMM and IID Benign hybrid-control evidence, together with the completed Fed-NGA/Huber validation search, directly into the complete responses. Its fixed evidence cutoff is 350 native and 350 three-view mechanism checkpoints, F50, FLGMM 81 three-view checkpoints (80 in eight complete scenes and one retained partial record), the 64-run gradient search, and the ten-checkpoint hybrid scene.')
replace('Contrary COMPAS and CelebA findings are retained; these results do not establish that every component is indispensable.',
        'The additional FLGMM evidence covers eight complete scenes with ten seeds each; a separate partial-scene checkpoint is retained without being included in those scene statistics. Its shared-calibration results also expose an accuracy–disparity trade-off (R3.8). Contrary COMPAS and CelebA findings are retained; these results do not establish that every component is indispensable.')
replace('The latter contains 900 records: nine methods × two distributions × five scenarios × ten shared seeds.',
        'The latter contains 900 records: nine methods × two distributions × five scenarios × ten shared seeds. The separately accepted FLGMM extension adds eight complete ten-seed scenes and one partial record (R3.8); it is not a complete additional 100-record method and does not change the denominator of the nine-method summaries below.')
replace('A separate native-method table adds LoGoFair, yielding 1,000 records across ten methods; it does not expand the nine-method common-calibration comparison to ten methods.',
        'A separate native-method table adds LoGoFair, yielding 1,000 records across ten methods; it does not expand the nine-method common-calibration comparison to ten methods. A further FLGMM extension has 81 accepted checkpoints: 80 across all five IID scenes and non-IID Benign, F Flip and FedSA, plus one partial non-IID S-DFA checkpoint. These are reported separately; non-IID S-DFA and Sp-DFA still lack complete accepted ten-seed coverage.')
flparagraph='''The [FLGMM extension](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md) contains 77 newly accepted coverage checkpoints and four explicitly reused validation-screen checkpoints. Its 81 saved three-view records comprise eight complete ten-seed scenes: all five IID scenarios, plus non-IID Benign, F Flip and FedSA. The remaining non-IID S-DFA seed91001 screen checkpoint is preserved separately and excluded from complete-scene statistics. Non-IID S-DFA and Sp-DFA therefore remain incomplete. Neither these partial additions nor the hybrid scene are folded into the earlier nine-method denominator.

For FLGMM under non-IID FedSA, the ten-seed results are mean±sample SD (ddof=1):

| Prediction rule | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Shared calibration lowers both measured disparities but also lowers accuracy in this scene. The fixed nine- and six-seed panels retain all three directions. These are the same round-70 checkpoints across prediction rules; raw and native outputs coincide here. The comparison therefore shows a prediction-rule-dependent trade-off, not evidence that aggregation alone improves fairness, a significant effect, or a win on every seed. AEOD denotes the absolute TPR gap, not full equalized odds. Reused screening seeds and the fixed sensitivity subsets retain their selection history and do not constitute an untouched confirmation set.

The saved-output evidence combines Linux verification of the original root calibration with independent Windows checks of saved predictions, counts and metrics without refitting. The earlier Windows refit failure is retained; complementary checks do not establish cross-platform bitwise recalibration equivalence. This extension remains validation-only and does not complete the 17-method benchmark.

'''
anchor="The [hybrid control's IID Benign table]"
replace(anchor,flparagraph+anchor)
replace('The current draft adds complete A100, five-scene F-deletion, IID Benign hybrid-control evidence and the completed 64-run validation search while retaining the A90 findings;',
        'The current draft adds complete A100, five-scene F-deletion, the FLGMM eight-scene table with its separate partial record, IID Benign hybrid-control evidence and the completed 64-run validation search while retaining the A90 findings;')
replace('FLGMM and the cosine/fairness hybrid still lack complete accepted fixed-recipe coverage;',
        'FLGMM has 77 new coverage checkpoints plus four reused screen checkpoints; its eight complete scenes contain 80 three-view records, with the non-IID S-DFA seed91001 record retained separately. Complete non-IID S-DFA and Sp-DFA coverage remains pending. FLGMM and the cosine/fairness hybrid therefore still lack complete accepted fixed-recipe coverage;')
replace('- [Hybrid control: IID Benign, ten checkpoints and raw/native/shared prediction]',
        '- [FLGMM: eight complete validation scenes and fixed 10/9/6-seed raw/native/shared tables](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md), with [adopted scope and retained partial record](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/ROOT_VERIFICATION.json).\n- [Hybrid control: IID Benign, ten checkpoints and raw/native/shared prediction]')

candidate=text.encode('utf8'); back=text
for ch in reversed(changes):
    assert back.count(ch['new'])==1;back=back.replace(ch['new'],ch['old'])
assert back.encode('utf8')==original
quote=lambda b:[x for x in b.splitlines(keepends=True) if x.startswith(b'>')]
blocks=lambda b:re.findall(rb'(?:^>[^\r\n]*(?:\r?\n|$))+',b,re.M)
assert quote(candidate)==quote(original) and len(quote(candidate))==38
assert blocks(candidate)==blocks(original) and len(blocks(candidate))==24
assert [x for x in candidate.splitlines() if x.startswith(b'#')]==[x for x in original.splitlines() if x.startswith(b'#')]
links=re.findall(r'\]\((E:/[^)]+)\)',text)
assert all(Path(x).is_file() for x in links)
# All unchanged paragraphs remain byte-for-byte, including every F50 trade-off,
# Huber/NGA negative outcome, selection-history and unfinished-work boundary.
untouched=[]
for para in original.decode('utf8').split('\n\n'):
    if para and not any(ch['old'] in para or para in ch['old'] for ch in changes):
        assert para in text
        untouched.append(H(para.encode()))
assert '27/32 and 12/32' in text and 'No final-test performance is supplied here.' in text
assert parent.read_bytes()==original and not target.exists()
target.write_bytes(candidate)
(O/'CHANGES.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(O/'DIFF.patch').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),text.splitlines(True),fromfile=parent.name,tofile=target.name)),encoding='utf8')
proof=dict(status='AUTHOR_EDITORIAL_SOURCE_AND_SCOPE_CHECK_PASS_NOT_ROOT_ADOPTION',root_accepted=False,
 parent_path=parent.relative_to(R).as_posix(),parent_sha256=H(original),draft_path=target.relative_to(R).as_posix(),draft_sha256=H(candidate),
 source_pins={k:dict(path=p,sha256=h) for k,(p,h) in pins.items()},editor_source_sha256=H(Path(__file__).read_bytes()),
 edits=len(changes),reverse_full_parent_bytes_exact=True,original24_quote_blocks_and38_lines_exact=True,all_headings_order_bytes_exact=True,
 local_links_checked=len(links),unchanged_paragraphs_preserved=len(untouched),unchanged_paragraph_sha256=untouched,
 new_mean_SD_pairs=facts,new_direction_checks='non-IID FedSA native-to-shared: ACC lower, AEOD lower, ASPD lower in each existing10/9/6panel; comparisons of saved means only',
 cutoff=dict(native=350,three_view=350,F_pairs=50,F_scenes=5,FL_native_new=77,FL_screen_reuse_separate=4,FL_three_view=81,FL_complete_scene_records=80,FL_complete_scenes=8,FL_retained_partial_records=1,gradient_search=64,Hybrid_three_view=10),
 no_F60_or_gradient14_results_added=True,original_scientific_statistics_not_recomputed=True,new_scientific_runs=0,arrays_or_models_read=False,
 final_test=False,submitted_manuscript_edited=False,whole_benchmark_complete=False,
 limits=['Author-review candidate only; parent and canonical science unchanged.','Existing saved JSON mean/SD display and scope only; no scientific checker, fit, CNN, SSH or Git.','Original Windows refit failure, Huber27/NGA12 constant outcomes, all F50 reversals, adaptation/selection history and unfinished P1-P6 retained.'])
(O/'EDITORIAL_VERIFICATION.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'draft_sha256':H(candidate),'editorial_proof_sha256':H((O/'EDITORIAL_VERIFICATION.json').read_bytes()),'edits':len(changes),'mean_SD_pairs':len(facts)},ensure_ascii=False))
