"""Minimal reversible full-text extension; no experiments or statistical fitting."""
from pathlib import Path
import collections,difflib,hashlib,json,re,sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
R=D.parents[3]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C100_20261010'
ADAPT=OLD.parent/'rebuttal_adaptations_20261010'
A=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(name,obj):
    with (D/name).open('x',encoding='utf8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def write(name,value):
    with (D/name).open('x',encoding='utf8',newline='\n') as f:f.write(value)

def prepare_A20_and_adaptations():
    assert not any((D/n).exists() for n in NAMES), 'Fresh full-document candidate required'
    assert [sha(OLD/n) for n in NAMES]==['2f70f682d085a228576720fcfde9f145461634be7b80cfe4b3a107f930e6611f','316cc83884963d4a52ac3dea650af66c6728d68ea64dcb1d33fa5abea151301c']
    assert sha(A/'ROOT_VERIFICATION.json')=='dbc2f8fe481c2a049f4e556a500c29d3122fc8392d69e4ab2bc131c0e3f4b3c0'
    assert sha(A/'FILES_SHA256.json')=='3841a7498dad4a3dc6a0bf3b8262510082cd042671850b5f56320a079c012ef8'
    root=read(A/'ROOT_VERIFICATION.json');tables=read(A/'tables.json')
    assert (root['paired_models'],root['complete_scenes'],root['preserved_records'])==(20,2,40)
    assert root['root_adoption'] and not root['test'] and not root['manuscript_applied']
    for name,pin in read(A/'FILES_SHA256.json')['files'].items():assert sha(A/name)==pin['sha256'] and (A/name).stat().st_size==pin['bytes']
    assert read(ADAPT/'ROOT_REVIEW.json')['root_text_reviewed'] and read(ADAPT/'ROOT_REVIEW.json')['check_exit_code']==0
    assert sha(ADAPT/'FILES_SHA256.json')=='f87eb44eb85ac2acc333641e2764d72055191793cc96edd329683a3333fcc44d'
    for name,pin in read(ADAPT/'FILES_SHA256.json')['files'].items():assert sha(ADAPT/name)==pin['sha256'] and (ADAPT/name).stat().st_size==pin['bytes']
    cells=[]
    def triplet(panel,row):
        target=tables['panels'][panel]['rows'][row];values=[]
        for metric in ('accuracy_pct','aeod','aspd'):
            value=target[metric];dp=3 if metric=='accuracy_pct' else 5
            display=f"{value['mean']:+.{dp}f} ± {value['sample_sd_ddof1']:.{dp}f}"
            cells.append(dict(source=(A/'tables.json').relative_to(R).as_posix(),pointer=f'/panels/{panel}/rows/{row}/{metric}',value=value,decimal_places=dp,display=display,units='percentage points' if metric=='accuracy_pct' else 'absolute gap'))
            values.append(display)
        return ' / '.join(values)
    atext=('**Image A-deletion evidence: two complete IID scenes.** We now separately compare 20 minus_A models with 20 original Full checkpoints for IID Benign and IID F Flip, ten matched seeds per scene. The A mask removes the score contribution while geometric hard filtering remains enabled; prediction calibration is retained. Each model uses the same round-70 checkpoint across raw, native and shared-calibration views on 19,867 validation images. The [accepted two-scene table]('+ (A/'TABLES.md').as_posix()+') and [root verification]('+(A/'ROOT_VERIFICATION.json').as_posix()+') give all Full/A means and paired differences. The ten-seed paired differences below are minus_A−Full means ± sample SD (ddof=1); ACC differences are percentage points and AEOD/ASPD are absolute gaps.\n\n'
        '| Scene | Native/shared ΔACC / ΔAEOD / ΔASPD | Raw ΔACC / ΔAEOD / ΔASPD |\n|---|---|---|\n'
        f'| IID Benign | {triplet(0,2)} | {triplet(3,2)} |\n'
        f'| IID F Flip | {triplet(0,5)} | {triplet(3,5)} |\n\n'
        'Deleting A lowers Benign accuracy in the ten-seed panel, slightly raises AEOD and lowers ASPD in both views. Under F Flip, its native/shared accuracy and AEOD differences are close to zero, with lower ASPD; raw deletion instead raises accuracy and both disparities. Near-zero displayed differences do not establish equivalence, and these descriptive means do not establish significance or component necessity.\n\n'
        f'**A-deletion subset sensitivity and comparison boundary.** For F Flip, the fixed nine-/six-seed native/shared paired triplets are {triplet(1,5)} and {triplet(2,5)}, respectively. Both accuracy and AEOD change from positive ten-seed differences to negative differences in these subsets, while ASPD stays negative. Raw F Flip accuracy is positive with ten/nine seeds and negative with six, whereas both raw disparity differences remain positive. Benign native/shared AEOD is positive with ten/nine seeds and negative with six; raw Benign accuracy reverses in the nine-seed subset. All predefined 10/9/6 panels use the same paired seeds for every metric; none is chosen for a favorable direction. Full replay comprises two CPU and 18 GPU models, whereas all 20 A replays use CPU; both training cohorts report cu128. Per-record historical/current environment and driver provenance is retained without claiming device equivalence. Native/shared metrics and group counts coincide for these 40 records and are not independent replications. Seed 91001 selection, exposed validation and prior official-test exposure remain disclosed; the final test under a frozen final protocol has not been run. These two IID scenes do not close A100, the other eight A scenes or the six still-incomplete image-control variants. No scene-pooled mean, causal-isolation, universal-necessity or primary-endpoint claim follows.')
    changes=[];texts={n:(OLD/n).read_text('utf8') for n in NAMES}
    def edit(name,before,after,reason):
        assert texts[name].count(before)==1,(name,reason)
        texts[name]=texts[name].replace(before,after,1)
        changes.append(dict(document=name,reason=reason,old=before,new=after,old_sha256=hashlib.sha256(before.encode()).hexdigest(),new_sha256=hashlib.sha256(after.encode()).hexdigest()))
    for item in read(ADAPT/'SOURCE_CHANGES.json')['changes']:
        name=Path(item['document']).name
        edit(name,item['old'],item['new'],'Apply the actual root-reviewed Huber/LoGo adaptation patch verbatim')
    for name in NAMES:
        p=next(p for p in texts[name].split('\n\n') if p.startswith('**Current C100 comparability boundary.**'))
        edit(name,p,p+'\n\n'+atext,'Integrate actual adopted A20 evidence after the unchanged complete C100 boundary')
        p=next(p for p in texts[name].split('\n\n') if p.startswith('**AUTHOR_REVIEW'))
        edit(name,p,p+f' The separately [accepted A20 comparison]({(A/"TABLES.md").as_posix()}) adds only IID Benign and IID F Flip, each with ten paired seeds and the same 10/9/6 sensitivity panels. A remains incomplete over eight scenes; U100 and C100 are unchanged.','Identify the exact additional partial-control scope')
    name=NAMES[0]
    anchor='### R3.8 — Explain attacks and comparison methods'
    edit(name,anchor,'The A20 comparison in R3.2 adds an image-domain example of the same limitation: its mean directions depend on the prediction rule and fixed seed panel. It preserves accuracy–disparity trade-offs and does not explain them by an isolated causal mechanism. The tabular COMPAS counterexamples above remain unchanged.\n\n'+anchor,'Connect A trade-offs to R3.7 without repeating the numerical table')
    p=next(p for p in texts[name].split('\n\n') if p.startswith('| Item |'))
    updated=p.replace('Complete and accept the six other image controls.','A20 now covers only IID Benign and IID F Flip with 20 paired checkpoints; complete its other eight scenes and the other five incomplete image controls.')
    assert p!=updated
    edit(name,p,updated,'Update pending mechanism scope without marking A100 complete')
    paths=[OLD/n for n in NAMES]+[OLD/'ROOT_REVIEW.json',OLD/'FILES_SHA256.json',OLD/'SOURCE_POINTERS.json',ADAPT/'ROOT_REVIEW.json',ADAPT/'FILES_SHA256.json',ADAPT/'SOURCE_CHANGES.json',ADAPT/'SOURCE_POINTERS.json',A/'ROOT_VERIFICATION.json',A/'FILES_SHA256.json',A/'tables.json',A/'TABLES.md',A/'SOURCE_BINDINGS.json',A/'records.json']
    pins={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths}
    pins.update(read(ADAPT/'SOURCE_POINTERS.json')['source_pins'])
    facts=[dict(source=(A/'ROOT_VERIFICATION.json').relative_to(R).as_posix(),pointer='/'+key,value=root[key]) for key in ('paired_models','complete_scenes','preserved_records','root_adoption','test','replay_devices','training_torch')]
    save('SOURCE_POINTERS.json',dict(source_pins=pins,numeric_cells=cells,json_facts=facts,adaptation_json_facts=read(ADAPT/'SOURCE_POINTERS.json')['json_facts'],adaptation_source_segments=read(ADAPT/'SOURCE_POINTERS.json')['source_segments'],A_table_directory=A.relative_to(R).as_posix(),LoGo100_root_adoption_received=False,final_primary_endpoint='PENDING_AUTHOR'))
    save('SOURCE_CHANGES.json',dict(status='A20_AND_APPROVED_ADAPTATIONS_FULL_DOCUMENT_PREPARATION_WAITING_FOR_ACTUAL_LOGO100_ROOT',source_pins=pins,changes=changes,full_documents_written=True,LoGo100_root_adoption_received=False))
    for name,text in texts.items():write(name,text)
    write('UPDATE_DIFF.patch',''.join(''.join(difflib.unified_diff((OLD/name).read_text('utf8').splitlines(True),texts[name].splitlines(True),fromfile='adopted_C100/'+name,tofile='A20_LoGo100_author_review/'+name)) for name in NAMES))
    print(json.dumps(dict(status='A20_AND_ADAPTATION_FULL_DRAFTS_PREPARED_LOGO100_ROOT_STILL_REQUIRED',full_documents=2,changes=len(changes),A_mean_SD_cells=len(cells),LoGo100_accepted_claim_added=False)))

def integrate_actual_LoGo100():
    L=R/'docs/server_deployment_20260923/training_20260923/celeba_logofair100_accepted_20261010'
    assert not (D/'FILES_SHA256.json').exists(), 'Sealed documents may not be revised'
    assert sha(L/'ROOT_ADOPTION.json')=='1529a852b3bd02561d274fdea832186706bb09725c8d02594d09114f44f977c2'
    root=read(L/'ROOT_ADOPTION.json');summary=read(L/'SUMMARY100.json');records=read(L/'records100.json')
    assert root['status']=='ROOT_LOGOFAIR_FIXED_RECIPE100_STRICT_SAVED_PREDICTION_AND_TABLES_ADOPTED'
    assert (root['accepted_count'],root['root_adopted'],root['new_accepted'],root['reused'])==(100,100,96,4)
    assert root['fit_seed']==summary['fit_seed']==1719 and root['virtual_cohorts']==20 and not root['true_client_fairness']
    assert root['selected_recipe']['id']=='LoGoFair-DP_07' and not summary['recipe_search_or_score_ranking_performed']
    assert not root['final_test'] and not root['primary_endpoint_selected'] and not root['whole_rebuttal_complete']
    assert sha(L/'SUMMARY100.json')=='e09cf1139f63661528bfaf7ee730ec558dec72107982bbee07d74d654955acb1'
    assert sha(L/'records100.json')=='fdc7c4f2402e26fdaa7b34bbfa792ceccafc5e3d32759d77940def2ed1fbc98d'
    assert sha(L/'TABLES.md')=='1cd87594eb25f953581a4153a64b4888e4e10c332f972988667810ac3c1c91a8'
    assert sha(R/root['independent_review_path'])==root['independent_review_sha256']=='b4fc447ba86c2fd40b54a1100c1ca2cdba890546b383feb5ddfa6bae19958610'
    for name,digest in root['files_sha256'].items():assert sha(L/name)==digest
    assert len(records)==len({r['id'] for r in records})==100 and root['constant_prediction_ids']==summary['constant_prediction_ids']==['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001']
    bindings=read(D/'SOURCE_POINTERS.json');delta=read(D/'SOURCE_CHANGES.json')
    assert not bindings['LoGo100_root_adoption_received'] and not delta['LoGo100_root_adoption_received']
    cells=bindings['numeric_cells']
    def cell(panel,metric):
        target=summary['panels'][panel]['per_scene'][0]
        assert (target['distribution'],target['attack'])==('IID','Benign')
        value=target[metric];dp=3 if metric=='accuracy_pct' else 5
        display=f"{value['mean']:.{dp}f} ± {value['sample_sd_ddof1']:.{dp}f}"
        cells.append(dict(source=(L/'SUMMARY100.json').relative_to(R).as_posix(),pointer=f'/panels/{panel}/per_scene/0/{metric}',value=value,decimal_places=dp,display=display,mean_signed=False,units='percent' if metric=='accuracy_pct' else 'absolute gap'))
        return display
    constant_index=next(i for i,r in enumerate(records) if r['id']==root['constant_prediction_ids'][0])
    constant=records[constant_index]
    assert constant['constant_prediction']==0 and constant['metrics']['aeod']==constant['metrics']['aspd']==constant['metrics']['positive_rate']==0
    constant_acc=f"{100*constant['metrics']['accuracy']:.3f}"
    bindings['scalar_displays']=[dict(source=(L/'records100.json').relative_to(R).as_posix(),pointer=f'/{constant_index}/metrics/accuracy',value=constant['metrics']['accuracy'],multiply=100,decimal_places=3,display=constant_acc,units='percent')]
    source_paragraph=(f'**Accepted fixed-recipe LoGoFair validation coverage.** The [independently adopted LoGoFair100 table]({(L/"TABLES.md").as_posix()}) and [root adoption]({(L/"ROOT_ADOPTION.json").as_posix()}) now cover both distributions and all five scenarios with ten model seeds: 96 new postprocessing records and four explicitly reused records. The original LoGoFair-DP_07 recipe selected in the validation search is fixed; this coverage run performs no new recipe ranking. DP here means demographic parity, not differential privacy. The official postprocessor is fitted only with root labels, using the declared20 image-ID virtual cohorts and fixed fit seed1719; validation is used for the original selection and subsequent reporting, not fitting. Its30 postprocessing rounds operate on accepted round-70 FedAvg checkpoints/score caches without new CNN training. These virtual cohorts do not represent real training clients.\n\n'
        f'**LoGoFair constant prediction and uncertainty.** The retained IID Benign seed91001 predicts every validation image negative: ACC is {constant_acc}%, while AEOD and ASPD are both zero. Those zero gaps do not show useful fairness with preserved predictive utility. IID Benign native ACC is {cell("ten_seed","accuracy_pct")}% across ten model seeds, compared with {cell("exclude_selection","accuracy_pct")}% in the fixed nine-seed panel and {cell("matching_six","accuracy_pct")}% in the fixed six-seed panel. The ten-seed AEOD/ASPD means are {cell("ten_seed","aeod")} / {cell("ten_seed","aspd")}. All values are means ± sample SD (ddof=1), and the full table retains the all-negative run and every unfavorable scene. The constant outcome is part of the large ten-seed spread; excluding selection seed91001 is a predefined sensitivity analysis, not permission to discard an unfavorable result or choose a better-looking panel. The SD varies model seeds conditional on fit seed1719, and does not estimate variability over postprocessor-fit seeds.\n\n'
        'The fixed10/9/6 panels remain descriptive because validation and selection seeds were previously exposed. LoGoFair postprocessing reports CPU Torch2.8.0+cpu; the accepted FedAvg source histories include98 cu128 and two cu130 training checkpoints and mixed runtime/device provenance. The accepted float32-margin-to-probability adapter is retained; equivalence to fresh softmax in a uniform runtime is not established. Earlier official-test results from the 240-run initial cohort had already been inspected, in addition to test attribute/split metadata exposure. This is post-initial-test validation development, not a never-viewed holdout or the frozen final test. The complete native LoGoFair result must remain distinct from the FedAvg `valid_native_prediction` cache and raw/shared backbone diagnostics. This fixed-recipe acceptance does not by itself create a ten-method three-view table, complete the17-method benchmark, select the final primary endpoint or certify the whole revision.')
    source_paragraph=source_paragraph.replace('declared20','declared 20').replace('seed1719','seed 1719').replace('Its30','Its 30').replace('seed91001','seed 91001').replace('fixed10/9/6','fixed 10/9/6').replace('Torch2.8.0','Torch 2.8.0').replace('include98','include 98').replace('the17-method','the 17-method')
    T=R/'outputs/guardfed_tables/celeba_ten_method_native_20261010'
    assert sha(T/'ROOT_REVIEW.json')=='a6009db154213944482ed194e5df81c0a6ce353dfc6821b984b222a18f5eeeec'
    native=read(T/'ROOT_REVIEW.json')
    assert native['status']=='ROOT_TEN_METHOD_NATIVE1000_DESCRIPTIVE_TABLE_ADOPTED'
    assert (native['records'],native['methods'],native['distributions'],native['scenes_per_distribution'],native['seeds'])==(1000,10,2,5,[10,9,6])
    assert native['original_nine_method810_metric_objects_exact'] and not native['final_test'] and not native['full17_complete'] and not native['primary_endpoint_selected']
    assert sha(T/'TABLES.md')==native['files_sha256']['TABLES.md']=='c6a38590e39e6ebafa2732498d7cabb479a161d27faee8016c48aecce2eef5a5'
    source_paragraph+=f'\n\n**Ten-method native validation comparison.** The separately [accepted native1000 table]({(T/"TABLES.md").as_posix()}), verified in [its root review]({(T/"ROOT_REVIEW.json").as_posix()}), combines the unchanged nine-method native records with the100 accepted LoGoFair outputs: ten methods, both distributions, all five scenarios and fixed10/9/6 model-seed panels. It preserves each method’s original calibrated/postprocessed decision rule, including the official fitted demographic-parity LoGoFair predictor. This descriptive native comparison does not replace the900-record raw/native/shared comparison or its nine-method common-calibration attribution. It is not a1000-record three-view/shared comparison, an aggregation-only causal comparison, a uniform-runtime evaluation or a completed17-method benchmark. In the ten-scene seed-first native means, GuardFed-AD2+ has higher accuracy and lower AEOD, whereas LoGoFair has lower ASPD. These outcomes show a utility–disparity trade-off, not superiority on every metric or over every method. Scenarios are averaged within each model seed before summarizing across seeds; scenarios are not independent seeds. No final primary endpoint or final-test result is supplied.'
    for a,b in [('native1000','native 1000'),('the100 accepted','the 100 accepted'),('fixed10/9/6','fixed 10/9/6'),('the900-record','the 900-record'),('a1000-record','a 1000-record'),('completed17-method','completed 17-method')]:source_paragraph=source_paragraph.replace(a,b)
    texts={n:(D/n).read_text('utf8') for n in NAMES}
    def edit(name,before,after,reason):
        assert texts[name].count(before)==1,(name,reason)
        texts[name]=texts[name].replace(before,after,1)
        delta['changes'].append(dict(document=name,reason=reason,old=before,new=after,old_sha256=hashlib.sha256(before.encode()).hexdigest(),new_sha256=hashlib.sha256(after.encode()).hexdigest()))
    for name in NAMES:
        p=next(p for p in texts[name].split('\n\n') if p.startswith('**Remaining-baseline implementation status at this evidence cutoff.**'))
        old_sentence='This search acceptance does not establish a complete 100-cell LoGoFair evaluation; that fixed-recipe stage remains pending.'
        assert old_sentence in p
        new_sentence='That fixed recipe has now passed independent root acceptance for100 validation records, reported separately below; its acceptance does not transfer to uncompleted methods.'
        edit(name,p,p.replace(old_sentence,new_sentence.replace('for100','for 100')),'Replace superseded pending LoGo100 status with actual root acceptance')
        p=next(p for p in texts[name].split('\n\n') if p.startswith('**LoGoFair prediction scope.**'))
        updated=p.replace('official DP postprocessor','official demographic-parity (DP) postprocessor')
        edit(name,p,updated+'\n\n'+source_paragraph,'Integrate independently accepted native LoGo100 evidence and retain predictor scope')
        p=next(p for p in texts[name].split('\n\n') if p.startswith('**AUTHOR_REVIEW'))
        updated=p.replace('The remaining eight-method benchmark coverage, frozen final evaluation and submitted-manuscript integration remain pending','Completion and comparable presentation of the target-method benchmark, frozen final evaluation and submitted-manuscript integration remain pending')
        updated+=f' Separately, [LoGoFair100 native validation results]({(L/"TABLES.md").as_posix()}) are now accepted under the fixed DP adaptation. They do not enlarge the earlier900-record three-view packet or imply complete17-method coverage.'
        updated=updated.replace('earlier900','earlier 900').replace('complete17','complete 17')
        updated+=f' The [ten-method native table]({(T/"TABLES.md").as_posix()}) is separately accepted; it preserves each method’s original prediction rule and is not a ten-method shared-calibration or three-view comparison.'
        edit(name,p,updated,'Update front-matter cutoff without combining incompatible predictor views')
    name=NAMES[0]
    for before,after in [('The current table contains nine methods; the remaining eight methods are P1.','The original three-view table contains nine methods; the separately accepted ten-method native table below does not close the remaining target-method coverage/comparability gap in P1.'),('We do not use “extended benchmark” to conceal the remaining eight-method coverage gap (P1).','We do not use “extended benchmark” to conceal the remaining target-method coverage/comparability gap (P1).')]:
        edit(name,before,after,'Align earlier overview with separately accepted LoGo100 scope')
    edit(name,'The current expanded CelebA comparison contains','The accepted nine-method three-view CelebA comparison contains','Distinguish the original900 three-view packet from the new native1000 comparison')
    edit(name,'The completed CelebA matrix now contains nine methods','The completed nine-method three-view CelebA matrix contains nine methods','Scope the previous matrix accurately after native1000 adoption')
    p=next(p for p in texts[name].split('\n\n') if p.startswith('| ID |'))
    row=f'\n| E14 | Separately accepted ten-method native validation table:1,000 records, both distributions × five scenarios × ten model seeds | Adds fixed-recipe LoGoFair native100 to the unchanged nine-method native records, with fixed10/9/6 panels. Uses original per-method postprocessing; the900-record three-view/common-calibration packet remains separate. [Table]({(T/"TABLES.md").as_posix()}); [root review]({(T/"ROOT_REVIEW.json").as_posix()}). Not a17-method or final-test result. |'
    for a,b in [('table:1,000','table: 1,000'),('native100','native 100'),('fixed10/9/6','fixed 10/9/6'),('the900-record','the 900-record'),('a17-method','a 17-method')]:row=row.replace(a,b)
    edit(name,p,p+row,'Add the actually adopted native1000 evidence without replacing E3/E6')
    before='The remaining eight methods are LoGoFair, Fed-NGA, FedWA, Huber-BRFL, FLGMM, SmartFL, FedDNA, and the cosine/fairness hybrid (P1). They define an 800-cell target extension not yet completed in the accepted comparison;'
    after='The original eight-method target extension comprises LoGoFair, Fed-NGA, FedWA, Huber-BRFL, FLGMM, SmartFL, FedDNA, and the cosine/fairness hybrid (P1), totaling800 cells. LoGoFair now has separately accepted fixed-recipe native100 coverage; completion and comparable presentation of the other seven methods remain pending. That original800-cell extension is not yet complete in the accepted comparison;'
    edit(name,before,after.replace('totaling800','totaling 800').replace('native100','native 100').replace('original800','original 800'),'Preserve original800 target as historical scope, without calling accepted LoGo100 pending')
    p=next(p for p in texts[name].split('\n\n') if p.startswith('| Item |'))
    updated=p.replace('eight methods × two distributions × five scenarios × ten seeds (800 records), with reuse explicit','the original eight methods × two distributions × five scenarios × ten seeds target (800 records), with reuse explicit. LoGoFair native100 is now independently accepted under its fixed adaptation; complete the remaining seven methods and align predictor scope before the final comparison')
    edit(name,p,updated.replace('native100','native 100'),'Update P1 while retaining all P1–P6 pending obligations')
    aggregate_path=T/'seed_first_aggregates.json'
    assert sha(aggregate_path)=='bc9c935e5b54ecd7488b714b92d356a3988d8ad417b9a158c1d516f12932267b'
    aggregate=read(aggregate_path)
    for i,method in [(26,'GuardFed-AD2+'),(29,'LoGoFair-DP')]:
        assert aggregate['ten'][i]['method']==method and aggregate['ten'][i]['scope']=='balanced_all10'
        for metric in ('accuracy','aeod','aspd'):bindings['json_facts'].append(dict(source=aggregate_path.relative_to(R).as_posix(),pointer=f'/ten/{i}/{metric}/mean',value=aggregate['ten'][i][metric]['mean']))
    bindings['tradeoff_source_pointers']={m:[f'/ten/26/{m}/mean',f'/ten/29/{m}/mean'] for m in ('accuracy','aeod','aspd')}
    paths=[L/n for n in ('ROOT_ADOPTION.json','SUMMARY100.json','records100.json','TABLES.md','ACCEPTANCE100.json')]+[R/root['independent_review_path'],T/'ROOT_REVIEW.json',T/'TABLES.md',T/'SOURCE_BINDINGS.json',T/'NUMERIC_VERIFICATION.json',aggregate_path]
    for p in paths:bindings['source_pins'][p.relative_to(R).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
    for key in ('accepted_count','root_adopted','new_accepted','reused','fit_seed','virtual_cohorts','true_client_fairness','selected_recipe','constant_prediction_ids','all_negative_and_constant_results_retained','final_test','primary_endpoint_selected','whole_rebuttal_complete'):
        bindings['json_facts'].append(dict(source=(L/'ROOT_ADOPTION.json').relative_to(R).as_posix(),pointer='/'+key,value=root[key]))
    for key in ('fit_seed','model_seed_panels','constant_prediction_ids','all_negative_and_constant_records_retained','recipe_search_or_score_ranking_performed'):
        bindings['json_facts'].append(dict(source=(L/'SUMMARY100.json').relative_to(R).as_posix(),pointer='/'+key,value=summary[key]))
    for key in ('records','methods','distributions','scenes_per_distribution','seeds','original_nine_method810_metric_objects_exact','final_test','full17_complete','primary_endpoint_selected'):
        bindings['json_facts'].append(dict(source=(T/'ROOT_REVIEW.json').relative_to(R).as_posix(),pointer='/'+key,value=native[key]))
    for key in ('constant_prediction','seed','fit_seed','reused_screen_record','metrics'):
        bindings['json_facts'].append(dict(source=(L/'records100.json').relative_to(R).as_posix(),pointer=f'/{constant_index}/'+key,value=constant[key]))
    bindings['derived_source_counts']=[dict(source=(L/'records100.json').relative_to(R).as_posix(),field='pretrained_source_runtime/torch_version',expected={'2.11.0+cu128':98,'2.11.0+cu130':2}),dict(source=(L/'records100.json').relative_to(R).as_posix(),field='environment/torch',expected={'2.8.0+cpu':100})]
    bindings['LoGo100_root_adoption_received']=True;bindings['LoGo100_root_adoption_sha256']=sha(L/'ROOT_ADOPTION.json')
    delta.update(status='COMPLETE_A20_LOGO100_AUTHOR_REVIEW_CANDIDATE_UNAPPLIED',LoGo100_root_adoption_received=True,source_pins=bindings['source_pins'])
    for n,obj in [('SOURCE_CHANGES.json',delta),('SOURCE_POINTERS.json',bindings)]:
        with (D/n).open('w',encoding='utf8',newline='\n') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
    for name,text in texts.items():(D/name).write_text(text,encoding='utf8',newline='\n')
    (D/'UPDATE_DIFF.patch').write_text(''.join(''.join(difflib.unified_diff((OLD/name).read_text('utf8').splitlines(True),texts[name].splitlines(True),fromfile='adopted_C100/'+name,tofile='A20_LoGo100_author_review/'+name)) for name in NAMES),encoding='utf8',newline='\n')
    print(json.dumps(dict(status=delta['status'],full_documents=2,changes=len(delta['changes']),mean_SD_cells=len(cells),actual_LoGo100_root=bindings['LoGo100_root_adoption_sha256'])))

if __name__=='__main__':
    if sys.argv[1:]==['prepare_A20']:prepare_A20_and_adaptations()
    elif sys.argv[1:]==['integrate_actual_LoGo100']:integrate_actual_LoGo100()
    else:raise ValueError('Use prepare_A20 once, then integrate_actual_LoGo100 only with its actual adopted closure.')
