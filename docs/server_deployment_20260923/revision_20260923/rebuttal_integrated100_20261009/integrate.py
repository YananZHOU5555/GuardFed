"""Bounded source-pinned prose integration; no statistical recomputation or network."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse,unquote
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
R=H.parents[1]
P=R/'tmp/guardfed_rebuttal_integrated71_v2_20261009'
C=R/'docs/server_deployment_20260923/revision_20260923'
T=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1'
S=T/'three_view_interim100_20261009'
N=T/'native_interim100_20261009'
V=R/'outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009'
B=R/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
PDF=R/'outputs/guardfed_tables/celeba_nine_method_three_view_pdf_20261009'
F=R/'outputs/guardfed_figures/synthetic_terminal_candidate_20261009'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
REFS=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def load(p):return json.loads(Path(p).read_bytes())
def save(name,data):
    text=data if isinstance(data,str) else json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with (H/name).open('x',encoding='utf8') as out:out.write(text)
def link(label,path):return '['+label+']('+path.as_posix()+')'
def quote_blocks(text):return re.findall(r'\*\*Original comment \(verbatim\)\.\*\*\n\n(.*?)\n\n\*\*Response\.\*\*',text,re.S)
def cell(tables,view,dist,attack,variant,metric,field='mean',digits=5,sign=False):
    pi,p=next((i,p) for i,p in enumerate(tables['panels']) if p['view']==view and len(p['seeds'])==10)
    ri,row=next((i,r) for i,r in enumerate(p['rows']) if (r['distribution'],r['attack'],r['variant'])==(dist,attack,variant))
    value=row[metric][field];display=format(value,('+' if sign else '')+'.'+str(digits)+'f')
    REFS.append(dict(source=(S/'snapshot100/tables.json').as_posix(),json_pointer=f'/panels/{pi}/rows/{ri}/{metric}/{field}',value=value,display=display,scale=1))
    return display
def attr(stats,path,digits=4,scale=1,sign=False):
    value=stats
    for part in path:value=value[part]
    display=format(value*scale,('+' if sign else '')+'.'+str(digits)+'f')
    REFS.append(dict(source=(V/'statistics.json').as_posix(),json_pointer='/'+'/'.join(path),value=value,display=display,scale=scale))
    return display
def delta(tables,view,metric,digits,sign=True):
    return cell(tables,view,'non-IID','Sp-DFA','minus_U minus Full',metric,digits=digits,sign=sign)+'±'+cell(tables,view,'non-IID','Sp-DFA','minus_U minus Full',metric,'sample_sd_ddof1',digits)

def main():
    pins=load(H/'INPUTS.json')['files']
    for name,pin in pins.items():need(sha(R/name)==pin['sha256'] and (R/name).stat().st_size==pin['bytes'],'Source drift '+name)
    prior={row['path']:row for row in load(P/'FILES_SHA256.json')['members']}
    for name,pin in prior.items():
        p=P/name;need(sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Prior sealed member changed '+name)
    original={n:(P/n).read_text(encoding='utf8') for n in NAMES}
    for n in NAMES:need((P/n).read_bytes()==(C/'rebuttal_integrated71_v2_20261009'/n).read_bytes(),'Canonical v2 copy mismatch')
    root=load(S/'ROOT_REVIEW.json');tables=load(S/'snapshot100/tables.json');records=load(S/'snapshot100/records.json')['records'];stats=load(V/'statistics.json')
    need(root['accepted_three_view100']==100 and root['complete_scenes']==10 and len(records)==200,'Wrong accepted mechanism scope')
    need(root['source_seal_sha256']=='766c08939e7f1d9fa6ab46e233ba7d2859bb0d3b9a0f418d73f5942b8eec0e88','Wrong final table source')
    need(tables['native_shared_identical_records']==200 and all(r['views']['native']==r['views']['shared_calibration'] for r in records),'Native/shared identity unsupported')
    need(tables['replay_devices']=={'Full':{'cpu':5,'cuda:0':95},'minus_U':{'cpu':100}},'Replay provenance changed')
    need(tables['training_torch']=={'Full':{'2.11.0+cu128':98,'2.11.0+cu130':2},'minus_U':{'2.11.0+cu128':100}},'Training provenance changed')
    counts=stats['panels']['ten']['GuardFed_positive_mean_advantage_baseline_count']
    need(counts['native']=={'accuracy':6,'aeod':8,'aspd':7} and counts['shared_calibration']=={'accuracy':7,'aeod':4,'aspd':1},'Calibration direction counts changed')
    need(load(V/'ROOT_REVIEW.json')['scalar_checks']==2052 and load(N/'ROOT_REVIEW.json')['accepted_native_controls']==104,'Accepted descriptive/native scope changed')
    need(load(F/'ROOT_REVIEW.json')['original_same_round_triplets_checked']==260 and load(F/'ROOT_REVIEW.json')['mean_scalars_checked']==78,'Figure candidate source changed')
    directions={}
    for p in tables['panels']:
        if len(p['seeds'])==10:
            rows=[r for r in p['rows'] if r['variant']=='minus_U minus Full']
            directions[p['view']]={m:{'negative':sum(r[m]['mean']<0 for r in rows),'positive':sum(r[m]['mean']>0 for r in rows),'source_rows':[dict(distribution=r['distribution'],attack=r['attack'],mean=r[m]['mean']) for r in rows]} for m in ('accuracy_pct','aeod','aspd')}
    need([directions['native'][m]['negative'] for m in ('accuracy_pct','aeod','aspd')]==[10,2,10],'Native direction wording unsupported')
    need([directions['raw'][m]['negative'] for m in ('accuracy_pct','aeod','aspd')]==[10,6,9],'Raw direction wording unsupported')
    dacc=[r['mean'] for r in directions['native']['accuracy_pct']['source_rows']]
    acc_range=f'{-max(dacc):.3f}–{-min(dacc):.3f}'
    note=('**DO_NOT_SUBMIT_BEFORE_FULL_COHORT. Integrated author-review copy; fixed accepted evidence snapshots.** '
          'The U-deletion statements use '+link('the ten-scene three-view snapshot',S/'snapshot100/TABLES.md')+', independently accepted at '+root['checked_utc']+' in '+link('the root review',S/'ROOT_REVIEW.json')+'. '
          'It contains 100 minus_U and 100 paired historical Full checkpoints: both distributions, all five scenarios and ten matched seeds, with the same fixed nine-/six-seed sensitivity panels. '
          'The separately accepted '+link('native100 snapshot',N/'rendered/TABLES.md')+' contains these U100 controls plus four partial minus_C records; C4 does not enter the U three-view table. '
          'This version supersedes the seven-scene writing copy without altering its evidence. Other image interventions, the remaining eight-method benchmark coverage, the frozen final evaluation and submitted-manuscript integration remain pending (P1–P6). The native/shared primary endpoint is not selected.')
    replay=('**Accepted nine-method validation evidence.** The '+link('900-record table packet',B/'README.md')+' and '+link('nine-page three-view PDF',PDF/'celeba_nine_method_three_view.pdf')+' cover nine methods × two distributions × five scenarios × ten shared seeds, using one round-70 checkpoint per ID. '
            'They include ten-, nine- and six-seed panels with mean ± sample SD (ddof=1). The '+link('accepted calibration analysis',V/'REPORT.md')+' has an independent 2,052-scalar '+link('root check',V/'ROOT_REVIEW.json')+'. '
            'The nine-method replay comprises 434 CPU and 466 GPU records; training comprises 886 cu128 and 14 cu130 records. These are exposed validation data with recipe-selection seed 91001 in the ten-seed panel, not a uniform-device final comparison or an untouched test. '
            'No new inference or threshold fitting was performed for this writing update.')
    change=['panels','ten','view_changes_target_minus_raw','GuardFed-AD2+','native']
    calibration=('The three prediction views separate the fixed trained model from its output rule. Raw uses argmax; native uses GuardFed’s original clean-root group thresholds and argmax for the eight baselines. '
        'Shared calibration applies the same frozen fitting rule to every model, with a separate threshold pair fitted from each model’s clean-training-root predictions. GuardFed’s native and shared outputs therefore coincide; the shared comparison also calibrates the baselines. '
        'For each method and seed, the analysis first averages the ten distribution–scenario cells, then summarizes across seeds. On these fixed GuardFed checkpoints, native minus raw changes ACC by '+attr(stats,change+['accuracy','mean'],4,100,True)+'±'+attr(stats,change+['accuracy','sample_sd'],4,100)+' percentage points, AEOD by '+attr(stats,change+['aeod','mean'],6,1,True)+'±'+attr(stats,change+['aeod','sample_sd'],6)+' and ASPD by '+attr(stats,change+['aspd','mean'],6,1,True)+'±'+attr(stats,change+['aspd','sample_sd'],6)+'. '
        'In the native view, GuardFed has better cross-scenario mean ACC/AEOD/ASPD than 6/8, 8/8 and 7/8 baselines; under shared calibration the counts are 7/8, 4/8 and 1/8. These are directions of paired mean comparisons, not seed win rates or significance tests. '
        'Shared-calibration accuracy remains '+attr(stats,['panels','ten','GuardFed_advantage','FLTrust','shared_calibration','accuracy','mean'],4,100)+' percentage points relative to FLTrust, with slightly higher AEOD and lower ASPD. '
        'Thus native performance describes the complete pipeline; its disparity advantage cannot be assigned wholly to aggregation. '+link('All method/view means and paired differences',V/'REPORT.md')+' retain the unfavorable comparisons.')
    iid_values={v:'/'.join(cell(tables,'native','IID','Benign',v,m,digits=3 if m=='accuracy_pct' else 5) for m in ('accuracy_pct','aeod','aspd')) for v in ('Full','minus_U')}
    u_response=('The 280-record tabular analysis, including the 260 new runs and 20 historical Adult Full controls, improves on the grouped deletions without establishing universal component necessity: all 12 COMPAS deletion conditions have higher mean accuracy than Full, and six improve all three means. We retain these negative findings. '
        'The image U-deletion comparison is now complete over all ten distribution–scenario cells, with 100 minus_U and 100 matched Full checkpoints. The U mask is applied after every candidate override; removing its score contribution leaves the remaining mechanisms enabled. '
        'Every paired result matches scene and seed, and all three metrics and views use the same round-70 checkpoint. For IID Benign, native ACC(%)/AEOD/ASPD changes from Full’s '+iid_values['Full']+' to minus_U’s '+iid_values['minus_U']+'. '
        +link('The complete U table',S/'snapshot100/TABLES.md')+' supplies each scene’s means, sample SDs and paired differences for all three views and the same 10/9/6-seed subsets. '
        'This closes one of the eight image controls, rather than the full mechanism study. C, A, F, V, norm treatment, hard filtering and candidate selection are outside this snapshot’s conclusions.')
    u_boundary=('**Image comparison boundary.** Native and shared-calibration metrics and saved group confusion counts coincide for all 200 U/Full records; these are parallel views, not independent repetitions or evidence of an additional calibration gain. '
        'Full replay comprises five CPU and 95 GPU models; all 100 minus_U replays use CPU. Full training includes 98 cu128 and two cu130 checkpoints, whereas controls use cu128 in the current driver 595 environment; historical and current drivers were not held equal. '
        'This limits causal attribution and does not establish a uniform-device final fairness comparison. The fixed sensitivity panels exclude selection seed 91001 or retain 91005–91010 identically for both procedures. Prior validation and official-test exposure remain; neither panel is an untouched confirmation set. '
        'The nine-method calibration control in R1.3 provides the complementary comparison in which all baselines receive the common root-only rule; deleting a score term must not be conflated with deleting prediction calibration.')
    u_interpret=('The complete U comparison strengthens the case for reporting trade-offs. In the ten-seed native panel, deleting U lowers ACC by '+acc_range+' percentage points and lowers ASPD in all ten scenes; AEOD rises in eight and falls in IID Sp-DFA and non-IID S-DFA. '
        'Raw AEOD instead falls in six scenes and rises in four: IID Sp-DFA and non-IID F Flip, FedSA and Sp-DFA. '
        'In the newly closed non-IID Sp-DFA scene, the paired minus_U−Full native/shared differences are '+delta(tables,'native','accuracy_pct',3)+' percentage points in ACC, '+delta(tables,'native','aeod',5)+' in AEOD and '+delta(tables,'native','aspd',5)+' in ASPD. '
        'The raw differences are '+delta(tables,'raw','accuracy_pct',3)+' percentage points, '+delta(tables,'raw','aeod',5)+' and '+delta(tables,'raw','aspd',5)+', respectively. Full has better raw means on all three metrics in this scene, while the native U-deletion control has lower ASPD. '
        'These are paired means ± sample SD, not significance claims. The prediction-rule dependence complements the COMPAS counterexamples and the nine-method calibration control (R1.3). Candidate compensation, score redundancy and root-estimation variability remain hypotheses, not isolated causes; the ten-seed directions do not guarantee the same result in every sensitivity panel or component.')
    figure=('A separate '+link('four-panel terminal-metric candidate',F/'fig3_terminal_candidate.pdf')+' now makes a possible Fig.3 correction reviewable. It plots ACC against AEOD and ASPD from coherent round-70 triplets, retaining all 13 settings per dataset and all ten scenarios at the single historical seed 123. '
        'The '+link('independent check',F/'ROOT_REVIEW.json')+' verifies 260 source triplets and 78 plotted means. In these terminal records, COMPAS TVAE with 1% real plus 9% synthetic root data is worse than the 10% real baseline on all three metrics. '
        'This preserves contrary evidence without a disputed FairScore or metric-wise extrema. It is a proposed redraw, not recovery of the original plotting source or author adoption; historical test exposure, missing checkpoint binary identity and unresolved generator execution remain P4/P5.')
    synthetic=link('coherent terminal correction candidate',F/'fig3_terminal_candidate.pdf')
    r_updates={
        '**Manuscript:**':next(p for p in original[NAMES[0]].split('\n\n') if p.startswith('**Manuscript:**')).replace('**Evidence cut-off:** 9 October 2026 (Australia/Sydney).','**Evidence snapshot:** 9 October 2026, 18:35 UTC (10 October 2026, Australia/Sydney).'),
        '**DO_NOT_SUBMIT_BEFORE_FULL_COHORT.':note,
        '**Accepted nine-method three-view replay extension':replay,
        'The completed additions broaden':('The completed additions broaden the evaluation beyond the original tabular tasks and grouped ablation, while exposing limitations. The nine-method native comparison combines aggregation and prediction calibration; the common-calibration control changes the disparity comparison, as detailed in R1.3. We preserve these findings rather than claim universal dominance.'),
        'The new image mechanism evidence':('The U-deletion study now covers all ten CelebA distribution–scenario cells with matched ten-seed controls. It associates retaining U with higher native mean accuracy, but deleting U lowers ASPD and has mixed AEOD effects. R3.2/R3.7 give the same-checkpoint three-view evidence and its boundary. The other seven image controls remain pending, and the unfavorable COMPAS tabular results remain part of the response.'),
        'Across both distributions and five scenarios':next(p for p in original[NAMES[0]].split('\n\n') if p.startswith('Across both distributions and five scenarios'))+'\n\n'+calibration,
        'The [Fig.3/PCA recovery audit]':next(p for p in original[NAMES[0]].split('\n\n') if p.startswith('The [Fig.3/PCA recovery audit]'))+'\n\n'+figure,
        '**Response.** The completed additions include both requested types':next(p for p in original[NAMES[0]].split('\n\n') if p.startswith('**Response.** The completed additions include both requested types')).replace('We replace “Mini-Benchmark”','We propose replacing “Mini-Benchmark”'),
        'The 280-record tabular analysis':u_response,
        '**Interim image comparability.**':u_boundary,
        'The intermediate image comparison':u_interpret,
    }
    pending=next(p for p in original[NAMES[0]].split('\n\n') if p.startswith('| Item |'))
    lines=pending.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| P2 —'):lines[i]='| P2 — CelebA mechanisms; still pending | U100 has accepted/offserver same-checkpoint raw/native/shared results for all ten scenes, with fixed 10/9/6-seed panels and paired differences. Complete and accept the other seven image controls; preserve Full/source/partition identities and the calibration/device/runtime boundary. The frozen native104 snapshot also contains four partial minus_C records, excluded from the U table. | R3.2; R3.7; mechanism attribution |'
        if line.startswith('| P4 —'):lines[i]=line.replace('no redraw or final approval is claimed.','a coherent terminal-metric redraw candidate is available for review, without original-source recovery or final adoption.')
        if line.startswith('| P5 —'):lines[i]=line.replace('Insert candidate text and accepted tables,','Obtain the submitted-version source project; the discovered paper.md is a historical source with different title/method/experiment structure and missing bibliography/figure dependencies. Insert candidate text and accepted tables,')
    r_updates['| Item |']='\n'.join(lines)
    mechanism=('**Image U-deletion evidence.** We pair 100 CelebA models trained without the utility-score contribution U with 100 historical Full terminal models across both distributions and all five scenarios. Each scene uses ten matched seeds; all three metrics and prediction views use the same round-70 checkpoint per model. '
        'The ten-seed native panel shows an accuracy decrease of '+acc_range+' percentage points after deletion and lower ASPD in all ten scenes, while AEOD rises in eight and falls in IID Sp-DFA and non-IID S-DFA. Raw AEOD falls in six scenes and rises in four. '
        'For non-IID Sp-DFA, the paired native/shared ACC/AEOD/ASPD differences (minus_U−Full) are '+delta(tables,'native','accuracy_pct',3)+' percentage points / '+delta(tables,'native','aeod',5)+' / '+delta(tables,'native','aspd',5)+', versus raw '+delta(tables,'raw','accuracy_pct',3)+' percentage points / '+delta(tables,'raw','aeod',5)+' / '+delta(tables,'raw','aspd',5)+'. '
        +link('The full three-view table',S/'snapshot100/TABLES.md')+' reports all per-scene means, sample SDs and paired differences, including unfavorable results. Native/shared complete metrics and group counts coincide for all 200 records; these are not independent calibration replications. '
        'This one intervention supports a utility–disparity trade-off, not necessity of every term or a causal explanation based on candidate compensation, redundancy or estimation variability. The native104 snapshot also contains four partial minus_C records; they are not included here. Other image controls remain pending.')
    body_boundary=('**Comparison boundary.** AEOD is absolute TPR gap, not full equalized odds. Native preserves each procedure’s original prediction rule; raw argmax and shared root-only calibration are parallel outputs of the same checkpoints. The Full replay mixes five CPU and 95 GPU models, with 98 cu128 and two cu130 training builds; all 100 controls use CPU replay and cu128 training in the current driver 595 environment. Historical/current driver equality was not established. '
        'Thus this valid-only U study is not a uniform-device final fairness comparison or a completed all-component image ablation. The nine-seed panel excluding recipe-selection seed 91001 and the six-seed panel retaining 91005–91010 apply the same rules to both procedures; previously exposed validation data do not become untouched confirmation sets by subsetting. The native/shared main endpoint and final evaluation remain for the authors to decide.')
    m_updates={
        '**DO_NOT_SUBMIT_BEFORE_FULL_COHORT.':note,
        '**Accepted nine-method three-view replay extension':replay,
        '**Prediction calibration.**':next(p for p in original[NAMES[1]].split('\n\n') if p.startswith('**Prediction calibration.**')).replace('A separate same-checkpoint common-calibration control is reported to distinguish the postprocessing effect.','The common-calibration control fits a separate threshold pair for each model from its clean-training-root predictions using the same frozen rule. Raw argmax uses margin > 0 with ties assigned to class 0; group thresholds use margin ≥ threshold. GuardFed native already uses this rule, whereas the eight baseline native outputs use argmax.'),
        'The submitted Fig.3 marker':next(p for p in original[NAMES[1]].split('\n\n') if p.startswith('The submitted Fig.3 marker')).replace('no corrected figure is represented as already produced.','the separately supplied '+synthetic+' is a proposed redraw, not recovery of the original plotting source or author adoption.'),
        '**Author decision pending;':next(p for p in original[NAMES[1]].split('\n\n') if p.startswith('**Author decision pending;'))+' The '+link('terminal-metric candidate',F/'fig3_terminal_candidate.pdf')+' uses all 260 coherent round-70 triplets and 78 verified means at the single historical seed 123; its COMPAS TVAE 1% real plus 9% synthetic result is worse than 10% real on all three terminal metrics. This additional counterexample does not certify historical generator or checkpoint-binary identity.',
        '**Historical calibration control paragraph.**':next(p for p in original[NAMES[1]].split('\n\n') if p.startswith('**Historical calibration control paragraph.**'))+'\n\n'+'**Nine-method same-checkpoint calibration control.** '+calibration,
        '**Image mechanism evidence at':mechanism,
        '**Comparison boundary.**':body_boundary,
    }
    checklist=next(p for p in original[NAMES[1]].split('\n\n') if p.startswith('1. Replace the earlier fairness-aware'))
    m_updates['1. Replace the earlier fairness-aware']=checklist.replace('6. Compile the final manuscript,','6. Obtain the submitted-version source project: the discovered paper.md is a historical source with different title/method/experiment structure, and its bibliography/figure dependencies were not recovered. Compile the final manuscript,')
    outputs={};ledger={};updates={NAMES[0]:r_updates,NAMES[1]:m_updates};edits=[]
    for name in NAMES:
        chunks=re.split(r'(\n\s*\n)',original[name]);used=set();rows=[]
        for i in range(0,len(chunks),2):
            before=chunks[i];matches=[key for key in updates[name] if before.startswith(key)];need(len(matches)<=1,'Ambiguous update')
            if matches:
                key=matches[0];need(key not in used,'Repeated anchor');used.add(key);chunks[i]=updates[name][key]
                edits.append(dict(file=name,old_paragraph=i//2+1,anchor=key,before=before,after=chunks[i]))
            rows.append(dict(old_paragraph=i//2+1,old_sha256=hashlib.sha256(before.encode()).hexdigest(),new_sha256=hashlib.sha256(chunks[i].encode()).hexdigest(),unchanged=before==chunks[i]))
        need(used==set(updates[name]),'Missing update anchor')
        text=''.join(chunks)
        outputs[name]=text;ledger[name]=rows
    oldquotes=quote_blocks(original[NAMES[0]]);newquotes=quote_blocks(outputs[NAMES[0]]);comments=load(C/'rebuttal_20261009/comment_source_map.json')['comments']
    need(len(oldquotes)==len(newquotes)==len(comments)==24 and oldquotes==newquotes,'Original comment blocks changed')
    comment_checks=[]
    order=[r['id'] for r in load(P/'COMMENT_CONSISTENCY.json')['comments']]
    need(len(order)==len(set(order))==24 and set(order)==set(comments),'Prior comment identity order drift')
    for identity,quote in zip(order,newquotes):
        source=comments[identity]
        recovered='\n'.join(line[2:] if line.startswith('> ') else line[1:] if line.startswith('>') else line for line in quote.splitlines())
        need(recovered==source,'Comment source mismatch '+identity)
        comment_checks.append(dict(id=identity,quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),verbatim_source_exact=True))
    for name,text in outputs.items():
        need('DO_NOT_SUBMIT_BEFORE_FULL_COHORT' in text,'Submission boundary missing')
        for stale in ['three_view_interim71_20261009','seven complete scenes','65 GPU','all 140 displayed','0.309–1.384']:
            need(stale not in text,'Stale scope '+stale)
        need('was revised' not in text.lower() and 'we have revised' not in text.lower(),'Unsupported source-edit claim')
    for name in NAMES:
        if name==NAMES[0]:
            for prefix in ['**Response.** We completed separate deletions','**Response.** We agree that emphasizing only','Calibration can also reverse','**Response.** We agree that a low-accuracy',"The historical Table II requires",'The audit also found method-name','The completed CelebA matrix now contains nine methods']:
                paragraph=next(p for p in original[name].split('\n\n') if p.startswith(prefix));need(paragraph in outputs[name],'Original negative/identity paragraph changed')
        else:
            for prefix in ['**Ablation paragraph.**','**Historical-table correction.**','**Selection and uncertainty.**']:
                paragraph=next(p for p in original[name].split('\n\n') if p.startswith(prefix));need(paragraph in outputs[name],'Original negative/statistics paragraph changed')
    links=[]
    for name,text in outputs.items():
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)',text):
            target=match.group(2).strip('<>');parsed=urlparse(target)
            if parsed.scheme in ('http','https'):need(bool(parsed.netloc),'Invalid external URL');exists=None
            else:
                p=Path(unquote(target.split('#')[0]));p=p if p.is_absolute() else H/p;exists=p.exists();need(exists,'Missing local link '+target)
            links.append(dict(file=name,label=match.group(1),target=target,local_exists=exists,network_checked=False))
    for ref in REFS:
        source=load(Path(ref['source']));x=source
        for part in ref['json_pointer'].strip('/').split('/'):x=x[int(part)] if isinstance(x,list) else x[part]
        need(x==ref['value'],'Numeric pointer mismatch');need(ref['display'] in '\n'.join(outputs.values()),'Numeric reference absent from prose')
    diff=''.join(''.join(difflib.unified_diff(original[n].splitlines(True),outputs[n].splitlines(True),fromfile='integrated71_v2/'+n,tofile='integrated100/'+n)) for n in NAMES)
    for name,text in outputs.items():save(name,text)
    save('UPDATE_DIFF.patch',diff);save('UNCHANGED_PARAGRAPHS.json',ledger)
    save('COMMENT_CONSISTENCY.json',dict(status='ALL24_ORIGINAL_COMMENT_BLOCKS_AND_CANONICAL_V2_EXACT',comments=comment_checks,comment_source_sha256=sha(C/'rebuttal_20261009/comment_source_map.json')))
    save('NUMERIC_REFERENCES.json',dict(status='SOURCE_VALUES_FORMATTED_ONLY_NO_NEW_STATISTICS',references=REFS,native_direction_source_rows=directions,native_ACC_decrease_range=acc_range,calibration_direction_counts_source_pointer='/panels/ten/GuardFed_positive_mean_advantage_baseline_count',calibration_direction_counts=counts,source_table_sha256=sha(S/'snapshot100/tables.json'),attribution_source_sha256=sha(V/'statistics.json')))
    save('LINK_CHECKS.json',dict(status='ALL_LOCAL_MARKDOWN_LINKS_EXIST_EXTERNAL_SYNTAX_ONLY',links=links,network_access=False))
    save('SOURCE_MAP.json',dict(status='EXACT_SNAPSHOT_AND_MINIMAL_PROSE_REPLACEMENT_MAP',input_files=pins,changes=edits,old_evidence_modified=False,scientific_statistics_recomputed=False))
    save('verification.json',dict(status='FULL24_ENGLISH_RESPONSE_AND_COMPLETE_INSERTIONS_SOURCE_NUMBERS_COMMENTS_LINKS_PASS',comments_verbatim=24,source_values_referenced=len(REFS),complete_U_scenes=10,paired_U_Full=100,records=200,native_shared_identical_records=200,all_original_COMPAS_counterexamples_and_TableII_actual_n_retained=True,old_280_tabular_evidence_retained=True,adapted_method_identities_retained=True,pending_items=['P1','P2','P3','P4','P5','P6'],native_shared_primary_pending=True,submission_gate='DO_NOT_SUBMIT_BEFORE_FULL_COHORT',main_manuscript_changed=False,new_scientific_statistics=False,network_CNN_training_test_Git_STATE_operations=False,local_links=len(links),paragraphs={n:dict(prior=len(ledger[n]),unchanged=sum(r['unchanged'] for r in ledger[n]),changed=sum(not r['unchanged'] for r in ledger[n])) for n in NAMES}))
    print(json.dumps(dict(status='PASS',comments=24,numeric_references=len(REFS),links=len(links),changed_paragraphs=len(edits))))

if __name__=='__main__':main()
