"""Editorial F60 increment from adopted saved summaries; no statistics run."""
from pathlib import Path
import json,hashlib,re,difflib
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
P=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F50_FL81_gradient64_20261011.md'
original=P.read_bytes();assert H(original)=='879eabe84687f498d56ebd3dabbaed52d7bb5b8aa7eeeda191f46649a952fbef'
F='docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_six_scenes60_20261011/'
pins={
 'F60_root':(F+'ROOT_VERIFICATION.json','cbba6a6cdfb9b861202422f867c847800474961cd02784dd207f87f51c386164'),
 'F60_tables':(F+'tables.json','497126f23be22533830d00389799c28238a4456382862d802a48c27cb2307b59'),
 'F60_table_md':(F+'TABLES.md','56353691ab7a3a798f2cc7e485977f464c415e30dad603906db684ab817f1f58'),
 'views360_root':('tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011/ROOT_ADOPTION.json','ca558d6e558d26afe6b172b65eded83b51ad02eedddc0f9e2d2cbadf7cdf5a11'),
 'native360_root':('docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T002530Z/ROOT_DELTA_VERIFICATION.json','396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418'),
 'Git64_receipt':('docs/server_deployment_20260923/training_20260923/publication_closed_increment64_verified_20261011.json','ceb120bf2ebc7fe8d635c2fd554f1c69787f70dc0e112398600ca265cdc9add2'),
 'parent_editorial_check':('tmp/rebuttal_FL81_F50_update_20261011/EDITORIAL_VERIFICATION_V2.json','3068cf1c3d5eaa9202872f14590bd12a3c58bce9217553ce7b583c03d8c477d7'),
}
sources={}
for k,(p,h) in pins.items():
    b=(R/p).read_bytes();assert H(b)==h,k
    if p.endswith('.json'):sources[k]=json.loads(b)
root=sources['F60_root'];tables=sources['F60_tables']
assert (root['paired_models'],root['records'],root['complete_scenes'])==(60,120,6)
assert root['old100_record_objects_order_exact'] and root['old810_statistical_scalars_exact'] and root['old405_display_cells_preserved']
assert root['actual_views360_root_sha256']==pins['views360_root'][1]
assert sources['views360_root']['native_root_sha256']==pins['native360_root'][1]
assert sources['native360_root']['total_new_strict_and_offserver']==sources['views360_root']['cumulative_accepted']==360
assert tables['replay_devices']=={'Full':{'cpu':5,'cuda:0':55},'minus_F':{'cpu':60}}
assert tables['training_torch']=={'Full':{'2.11.0+cu128':59,'2.11.0+cu130':1},'minus_F':{'2.11.0+cu128':60}}
assert not root['primary_endpoint_selected'] and not root['final_test']
facts=[];signs=[];rows=[]
def get(view,n):
    pi=next(i for i,p in enumerate(tables['panels']) if p['view']==view and len(p['seeds'])==n)
    p=tables['panels'][pi];ri=next(i for i,r in enumerate(p['rows']) if (r['distribution'],r['attack'],r['variant'])==('non-IID','Benign','minus_F minus Full'))
    return p['rows'][ri],f'/panels/{pi}/rows/{ri}'
for view in ['native','raw','shared_calibration']:
    for n in [10,9,6]:
        row,ptr=get(view,n)
        actual=[1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ['accuracy_pct','aeod','aspd']]
        assert actual==([1,1,1] if n in [10,9] else [-1,1,1]),(view,n)
        signs.append(dict(view=view,n=n,pointer=ptr,minusF_minusFull_signs=actual))
        if view=='shared_calibration':assert row==get('native',n)[0]
        else:
            values=[]
            for m in ['accuracy_pct','aeod','aspd']:
                x=row[m];digits=3 if m=='accuracy_pct' else 5
                val=f"{x['mean']:+.{digits}f} ± {x['sample_sd_ddof1']:.{digits}f}"
                facts.append(dict(source='F60_tables',pointer=ptr+'/'+m,mean=x['mean'],sample_sd=x['sample_sd_ddof1'],display=val))
                values.append(val)
            rows.append('| '+('Native/shared' if view=='native' else 'Raw')+' | '+str(n)+' | '+' | '.join(values)+' |')
text=original.decode();edits=[]
def change(old,new):
    global text
    assert text.count(old)==1,old[:90]
    text=text.replace(old,new);edits.append(dict(old=old,new=new,count=1))
change('accepted five-scene F-deletion, eight-complete-scene FLGMM','accepted six-scene F-deletion, eight-complete-scene FLGMM')
change('350 native and 350 three-view mechanism checkpoints, F50','360 native and 360 three-view mechanism checkpoints, F60')
change('plus F-deletion results for all five IID scenarios—Benign, F Flip, FedSA, S-DFA and Sp-DFA—each with ten paired seeds.',
       'plus F-deletion results for all five IID scenarios—Benign, F Flip, FedSA, S-DFA and Sp-DFA—and non-IID Benign, each with ten paired seeds.')
change('The five non-IID F scenes and four other image controls remain unfinished; this is not F100 or an all-component completion claim.',
       'The [six-scene F extension](E:/OneDrive/文档/GuardFed/'+F+'TABLES.md) additionally includes non-IID Benign (α=5) with ten paired seeds under the same checkpoint and prediction-view rules. The other four non-IID F scenes and four other image controls remain unfinished; this is not F100 or an all-component completion claim.')
change('U100 + C100 + A100 + F50 = 350 deletion checkpoints','U100 + C100 + A100 + F60 = 360 deletion checkpoints')
change('Native and three-view acceptance both cover these 350/800 deletion checkpoints at this cutoff.','Native and three-view acceptance both cover these 360/800 deletion checkpoints at this cutoff.')
change('For these five F scenes, the reused Full predictions comprise 3 CPU and 47 GPU evaluations, whereas all 50 F-deletion evaluations use CPU; the selected models all retain their recorded Torch 2.11.0+cu128 training provenance.',
       'Within the original five IID F scenes, the reused Full predictions comprise 3 CPU and 47 GPU evaluations, whereas all 50 F-deletion evaluations use CPU; those selected models all retain their recorded Torch 2.11.0+cu128 training provenance. Across the six-scene extension, Full has 5 CPU and 55 GPU evaluations, whereas all 60 F-deletion evaluations use CPU. The corresponding training provenance is 59 cu128 and one cu130 Full checkpoint, versus 60 cu128 deletion checkpoints.')
newscene='''The added non-IID Benign scene retains another utility–disparity trade-off. The table below reports the existing paired differences (minus_F−Full, mean±sample SD, ddof=1) for each fixed panel:

| Prediction view | n | ΔACC (percentage points) | ΔAEOD | ΔASPD |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

In both the ten- and nine-seed panels, removing F increases mean accuracy while worsening both disparities, for raw as well as native/shared predictions. The six-seed panel reverses the accuracy direction: Full has higher mean accuracy and lower mean disparities in all three views. Native and shared results coincide and are not independent confirmations. These are descriptive paired means from matched round-70 checkpoints, not claims about every seed or statistical significance. We retain all panels, including the accuracy reversal; neither a selected subset nor the mixed environments identifies an isolated causal effect. The primary endpoint remains undecided, and no cross-scene F aggregate or untouched-test confirmation is supplied.

'''
change('Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes.',newscene+'Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes.')
change('alongside A100, F50, the IID Benign hybrid-control evidence and the gradient search. This FLGMM prose expansion has not yet been published in that fixed commit.',
       'alongside A100, F60, the IID Benign hybrid-control evidence and the gradient search. That fixed Git64 release stops at F50; the new non-IID Benign F-deletion evidence and this F60/FLGMM prose expansion have not yet been published in it.')
change("At this draft's accepted cutoff, 350 deletion checkpoints have complete three-view evidence: U100, C100 and A100 each cover all ten cells, and F50 covers all five IID scenes with ten paired seeds per scene. The remaining 450 planned mechanism-control checkpoints comprise 50 non-IID F-deletion runs and 100 runs for each of four other controls. Native acceptance is also 350/800, leaving 450 without native acceptance; that is not a claim that all 450 remain untrained.",
       "At this draft's accepted cutoff, 360 deletion checkpoints have complete three-view evidence: U100, C100 and A100 each cover all ten cells, and F60 covers all five IID scenes plus non-IID Benign with ten paired seeds per scene. The remaining 440 planned mechanism-control checkpoints comprise 40 runs for the other four non-IID F-deletion scenes and 100 runs for each of four other controls. Native acceptance is also 360/800, leaving 440 without native acceptance; that is not a claim that all 440 remain untrained.")
anchor='- [Completed 64-run Fed-NGA/Huber validation-search records and retained negative outcomes]'
change(anchor,'- [F-deletion extension: five IID scenes plus non-IID Benign, matched raw/native/shared tables](E:/OneDrive/文档/GuardFed/'+F+'TABLES.md).\n'+anchor)

candidate=text.encode();back=text
for e in reversed(edits):
    assert back.count(e['new'])==1;back=back.replace(e['new'],e['old'],1)
assert back.encode()==original
q=lambda b:[x for x in b.splitlines(keepends=True) if x.startswith(b'>')]
blocks=lambda b:re.findall(rb'(?:^>[^\r\n]*(?:\r?\n|$))+',b,re.M)
assert q(candidate)==q(original) and len(q(candidate))==38
assert blocks(candidate)==blocks(original) and len(blocks(candidate))==24
assert [x for x in candidate.splitlines() if x.startswith(b'#')]==[x for x in original.splitlines() if x.startswith(b'#')]
# Preserve the entire old F50 scientific interpretation, FL81 response/table,
# and Huber/NGA/LoGoFair limitations, rather than only checking their numbers.
oldtext=original.decode();preserved={}
for name,start,end in [
 ('all_F50_negative_and_positive_interpretations','The new IID F-deletion comparison reinforces','Correlated scores,'),
 ('FL81_Hybrid_gradient64_LoGoFair','The [FLGMM extension]','### R3.9'),
 ('Git64_fixed_published_F50','**Response.** A verified revision snapshot','The earlier [817f6fd snapshot]')]:
    a=oldtext.index(start);b=oldtext.index(end,a);chunk=oldtext[a:b];assert chunk in text;preserved[name]=H(chunk.encode())
for p in oldtext.split('\n\n'):
    if p and not any(e['old'] in p or p in e['old'] for e in edits):assert p in text
links=re.findall(r'\]\((E:/[^)]+)\)',text);assert all(Path(x).is_file() for x in links)
assert '27/32 and 12/32' in text and 'No final-test performance is supplied here.' in text
assert P.read_bytes()==original
target=D/'rebuttal_working_F60_FL81_gradient64_20261011.md';assert not target.exists();target.write_bytes(candidate)
(D/'EDITS.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(D/'DIFF.patch').write_text(''.join(difflib.unified_diff(oldtext.splitlines(True),text.splitlines(True),fromfile=P.name,tofile=target.name)),encoding='utf8')
proof=dict(status='AUTHOR_F60_EDITORIAL_CANDIDATE_SOURCE_AND_QUOTE_CHECK_PASS_NOT_ROOT_ADOPTION',root_accepted=False,
 parent_path=P.relative_to(R).as_posix(),parent_sha256=H(original),candidate_path=target.relative_to(R).as_posix(),candidate_sha256=H(candidate),source_sha256=H(Path(__file__).read_bytes()),
 source_pins={k:dict(path=p,sha256=h) for k,(p,h) in pins.items()},edits=len(edits),all_count1=True,reverse_full_parent_bytes_exact=True,
 original24_quote_blocks38_lines_bytes_and_order_exact=True,all_headings_bytes_and_order_exact=True,local_links_checked=len(links),
 preserved_scientific_blocks=preserved,displayed_new_paired_mean_SD_pairs=facts,nine_actual_panel_direction_bindings=signs,
 new_environment_binding=dict(replay_devices=tables['replay_devices'],training_torch=tables['training_torch']),
 content_cutoff=dict(native=360,three_view=360,F_pairs=60,F_scenes=6,FL_native_new=77,FL_screen_reuse_separate=4,FL_three_view=81,FL_complete_table_records=80,FL_complete_scenes=8,gradient_search=64,Hybrid_three_view=10),
 Git64_fixed_cutoff=dict(native=350,three_view=350,F_pairs=50,FL_three_view=81,gradient_search=64),candidate_published=False,
 statistics_recomputed=False,scientific_checker_run=False,arrays_models_read=False,SSH=False,Git=False,STATE_modified=False,final_test=False,submitted_manuscript_edited=False,whole_rebuttal_complete=False,
 limits=['Existing saved means/SD and root scope only; formatting and sign comparisons are editorial checks, not a new statistical analysis.','Original numerical F60 tools were prepared by this agent; current text is author editing, not an independent science re-verification.','No primary endpoint/significance/every-seed/causal-isolation claim; validation selection and mixed replay/training environments retained.'])
(D/'EDITORIAL_VERIFICATION.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
handoff=dict(status='F60_COMPLETE24_CANDIDATE_READY_FOR_ROOT_REVIEW',candidate_path=target.relative_to(R).as_posix(),candidate_sha256=H(candidate),editorial_proof_sha256=H((D/'EDITORIAL_VERIFICATION.json').read_bytes()),diff_sha256=H((D/'DIFF.patch').read_bytes()),edits_sha256=H((D/'EDITS.json').read_bytes()),root_accepted=False,canonical_written=False,changes='Current mechanism360/F60; new non-IID Benign paired10/9/6 tradeoffs and mixed environment; fixed FL81 and historical Git64F50 unchanged.')
(D/'HANDOFF.json').write_text(json.dumps(handoff,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(handoff))
