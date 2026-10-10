"""Reversible, local prose-only C20-to-C50 integration; reads accepted statistics."""
from pathlib import Path
import difflib,hashlib,json,re,sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent; R=D.parents[1]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009'
SHORT=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_C50_update_20261010'
TABLE=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
url=lambda p:p.as_posix()

def write(path,value):
    with path.open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    assert sha(OLD/NAMES[0])=='452cd7e7240933e396befc1826f36c4ad3660b3d572bdec9dcd267067d0cab08'
    assert sha(OLD/NAMES[1])=='1175e69371bbf8ebf3915cac7c804e78829d97965a68fdf147db284ef27982e7'
    assert sha(SHORT/'ROOT_REVIEW.json')=='cb57e51f7ba426583f7975a78078c79b59149e475e63901ba4079c218e1354ff'
    assert sha(TABLE/'ROOT_VERIFICATION.json')=='811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
    basis=read(SHORT/'SOURCE_POINTERS.json'); tables=read(TABLE/'snapshot/tables.json')
    source_pins={k:v for k,v in basis['source_pins'].items() if 'rebuttal_C40_addendum_prepared' not in k}
    for path in [OLD/NAMES[0],OLD/NAMES[1],OLD/'SOURCE_CHANGES.json',SHORT/'ROOT_REVIEW.json',SHORT/'C50_REVIEWER_ADDENDUM.md',SHORT/'C50_MANUSCRIPT_INSERTIONS.md',SHORT/'SOURCE_POINTERS.json',R/'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/comment_source_map.json',R/'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/reviewer_comments_verbatim.md']:
        source_pins[path.relative_to(R).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
    for path,pin in source_pins.items():assert sha(R/path)==pin['sha256'] and (R/path).stat().st_size==pin['bytes']
    table_url=url(TABLE/'snapshot/TABLES.md'); root_url=url(TABLE/'ROOT_VERIFICATION.json'); cross_url=url(TABLE/'snapshot/cross_scene_seed_first.json')
    def cell(view,n,scene,metric,spacing=' ± '):
        panel=next(p for p in tables['panels'] if p['view']==view and len(p['seeds'])==n)
        row=next(r for r in panel['rows'] if r['attack']==scene and r['variant']=='minus_C minus Full')
        p=3 if metric=='accuracy_pct' else 5
        return f"{row[metric]['mean']:+.{p}f}{spacing}{row[metric]['sample_sd_ddof1']:.{p}f}"
    sp_native=' / '.join(cell('native',10,'Sp-DFA',m) for m in ['accuracy_pct','aeod','aspd'])
    sp_raw=' / '.join(cell('raw',10,'Sp-DFA',m) for m in ['accuracy_pct','aeod','aspd'])
    sp9=cell('native',9,'Sp-DFA','accuracy_pct');sp6=cell('native',6,'Sp-DFA','accuracy_pct')
    originals={n:(OLD/n).read_bytes().decode('utf8') for n in NAMES}; output=dict(originals); changes=[]
    def replace(name,old,new,label):
        assert output[name].count(old)==1,(name,label)
        assert old!=new
        changes.append(dict(document=name,label=label,old=old,new=new,original_line=originals[name][:originals[name].index(old)].count('\n')+1))
        output[name]=output[name].replace(old,new,1)
    old_common=originals[NAMES[0]].splitlines()[6]
    new_common=old_common.replace('three_view_C_two_scenes_20261009','three_view_C_five_scenes_20261010').replace('2026-10-09T20:30:49.596523+00:00','2026-10-09T23:55:40.717121+00:00').replace('contains 20 minus_C and 20 matched historical Full checkpoints for exactly two scenes: IID Benign and IID F Flip','contains 50 minus_C and 50 matched historical Full checkpoints for five IID scenes: Benign, F Flip, FedSA, S-DFA and Sp-DFA').replace('The other eight C scenes and six other image controls remain incomplete','The five non-IID C scenes and six other image controls remain incomplete')
    for name in NAMES:replace(name,old_common,new_common,'C50 snapshot coverage; U100 and pending endpoint unchanged')
    replace(NAMES[0],'**Evidence snapshot:** 9 October 2026, 20:30 UTC (10 October 2026, Australia/Sydney).','**Evidence snapshot:** 9 October 2026, 23:55 UTC (10 October 2026, Australia/Sydney).','Latest adopted C50 evidence time')
    old=originals[NAMES[0]].splitlines()[48]
    new=old.replace('The C-deletion extension now covers IID Benign and IID F Flip only','The C-deletion extension now covers all five IID scenes: Benign, F Flip, FedSA, S-DFA and Sp-DFA').replace('The other eight C scenes and six other image controls remain incomplete','The five non-IID C scenes and six other image controls remain incomplete')
    replace(NAMES[0],old,new,'AE C coverage; existing U and COMPAS counterexamples unchanged')
    old=originals[NAMES[0]].splitlines()[266]
    new=old.replace('three_view_C_two_scenes_20261009','three_view_C_five_scenes_20261010').replace('adds 20 minus_C checkpoints paired by scene and seed with 20 existing Full checkpoints, restricted to IID Benign and IID F Flip','contains 50 minus_C checkpoints paired by scene and seed with 50 existing Full checkpoints across IID Benign, F Flip, FedSA, S-DFA and Sp-DFA').replace('C is not complete over the other eight scenes','C remains incomplete over the five non-IID scenes')
    new+=f'\n\nFor IID Sp-DFA, the ten-seed paired native/shared ACC/AEOD/ASPD differences are {sp_native}, versus {sp_raw} in raw (ACC in percentage points; mean ± sample SD, ddof=1). Deleting C raises mean accuracy in both views, lowers native/shared AEOD but raises ASPD, and raises both raw disparity means. Calibration remains enabled after C deletion; these are conditional trade-offs, not an isolated intervention on every aggregation operation.'
    replace(NAMES[0],old,new,'R3.2 five-scene C evidence and Sp-DFA trade-off; original Benign/F Flip values retained')
    old_tail='For the separate C20 comparison, Full replay uses two CPU and 18 GPU checkpoints and all 20 minus_C replays use CPU; both training cohorts use cu128. These C20 counts do not replace the historical U100 or nine-method environment disclosures. Native/shared metrics and group counts also coincide for all 40 C/Full records, so the views are not independent replications.'
    new_tail='For the separate C50 comparison, Full replay uses three CPU and 47 GPU checkpoints and all 50 minus_C replays use CPU; both training cohorts use PyTorch 2.11.0+cu128, but historical/current driver equality was not established. These C50 counts do not replace the historical U100 or nine-method environment disclosures. Native/shared metrics and group counts also coincide for all 100 C/Full records, so the views are not independent replications. C50 evaluation uses 19,867 validation images; the final test under a frozen final protocol has not been run.'
    replace(NAMES[0],old_tail,new_tail,'C50 actual devices, environment, valid/test boundary; U100 boundary unchanged')
    old=originals[NAMES[0]].splitlines()[320]
    new=old.replace('The two completed C scenes','The five completed IID C scenes').replace('They do not extrapolate to the other eight C scenes or the other incomplete image controls.','They do not extrapolate to the five non-IID C scenes or the other incomplete image controls.')
    new+=f'\n\nThe added scenes also retain counterexamples and subset dependence. In the ten-seed native/shared panel, deleting C under IID FedSA raises ACC and ASPD while lowering AEOD; under IID S-DFA it lowers ACC and raises both gaps. For FedSA, native/shared AEOD changes from a lower ten-seed mean to a higher mean in both sensitivity panels, while ASPD changes from a higher mean to lower means. In the raw nine-seed FedSA panel, deletion improves all three means, whereas in the raw six-seed panel it raises both gaps. S-DFA native/shared ACC reverses from a lower ten-/nine-seed mean to a higher six-seed mean. For Sp-DFA, the native/shared paired ACC difference is {sp9} percentage points for nine seeds and {sp6} for six seeds; lower AEOD and higher ASPD remain in both subsets. Raw Sp-DFA retains ACC-up/AEOD-up/ASPD-up in both subsets. The fixed panels are reported together, without selecting the favorable panel.\n\nThe [five-scene table]({table_url}) reports every Full/C mean and paired difference for raw, native and shared-calibration views and identical 10/9/6-seed panels. The separate [cross-scene summary]({cross_url}) first averages the five IID scenes within each seed, then calculates across-seed mean and sample SD; its sample size is the seed count. Its ten-seed native/shared mean trades lower AEOD for higher ASPD after C deletion, while raw raises both disparity means. These descriptions preserve the earlier Benign, F Flip and COMPAS counterexamples and do not establish uniform component benefit, necessity, significance or a pure aggregation causal effect.'
    replace(NAMES[0],old,new,'R3.7 all-scene trade-offs, preselected subset reversals and seed-first scope')
    old=originals[NAMES[0]].splitlines()[369]
    new=old.replace('C20 has the same three views for IID Benign and IID F Flip only, with 20 matched Full controls','C50 has the same three views for all five IID scenes (Benign, F Flip, FedSA, S-DFA and Sp-DFA), with 50 matched Full controls').replace('the other eight C scenes','the five non-IID C scenes')
    replace(NAMES[0],old,new,'P2 partial C50 is not full C100 or whole-mechanism completion')
    replace(NAMES[1],'The separate C20 extension below does not change these U100 records or conclusions','The separate C50 extension below does not change these U100 records or conclusions','U100 evidence unchanged; reference updated to C50')
    old=originals[NAMES[1]].splitlines()[148]
    new=old.replace('**Image C-deletion evidence, two scenes only.** We separately compare 20 minus_C models with 20 existing Full checkpoints for IID Benign and IID F Flip, ten matched seeds per scene.','**Image C-deletion evidence, five IID scenes only.** We separately compare 50 minus_C models with 50 existing Full checkpoints for IID Benign, F Flip, FedSA, S-DFA and Sp-DFA, ten matched seeds per scene.').replace('[The accepted two-scene table]','[The accepted five-scene table]').replace('three_view_C_two_scenes_20261009','three_view_C_five_scenes_20261010').replace('all 40 records','all 100 records').replace('The other eight C scenes and six other image controls','The five non-IID C scenes and six other image controls').replace('These two scenes do not support','These five IID scenes do not support')
    new+=f'\n\n**Additional C-scene trade-offs.** For Sp-DFA, ten-seed paired native/shared ACC/AEOD/ASPD differences are {sp_native}, versus {sp_raw} in raw, with ACC in percentage points and all values mean ± sample SD (ddof=1). Deletion lowers native/shared AEOD but raises ASPD; raw raises both disparity means. The native/shared ACC difference reverses from {sp9} for nine seeds to {sp6} for six seeds, while lower AEOD and higher ASPD persist. For FedSA, deleting C raises native/shared ten-seed ACC and ASPD while lowering AEOD; both gap directions reverse in the fixed nine-/six-seed subsets. Raw nine-seed FedSA improves all three means after deletion, whereas raw six-seed FedSA raises both gaps. S-DFA native/shared deletion lowers ACC and raises both gaps in the ten-seed panel; its ACC direction reverses in the six-seed subset. All unfavorable outcomes and subset reversals remain in the full table, without choosing a favorable view or panel.\n\n**Cross-scene interpretation.** The [seed-first summary]({cross_url}) averages the five IID scenes within each seed before computing across-seed mean and sample SD. Its sample size remains the seed count. The ten-seed native/shared mean trades lower AEOD for higher ASPD after deletion; raw raises both gaps. This descriptive summary does not establish component necessity, significance or a pure aggregation causal effect. The earlier COMPAS counterexamples and U100 findings remain unchanged.'
    replace(NAMES[1],old,new,'C50 manuscript candidate, retaining old C20 values and new counterexamples')
    old='For the separate C20 comparison, Full replay uses two CPU and 18 GPU checkpoints, all 20 minus_C replays use CPU, and both training cohorts use cu128; these counts do not replace the U100 or nine-method historical build disclosures.'
    new='For the separate C50 comparison, Full replay uses three CPU and 47 GPU checkpoints, all 50 minus_C replays use CPU, and both training cohorts use PyTorch 2.11.0+cu128; historical/current driver equality was not established, and these counts do not replace the U100 or nine-method historical build disclosures. C50 evaluation uses 19,867 validation images, with prior validation and official-test exposure disclosed; the final test under a frozen final protocol has not been run.'
    replace(NAMES[1],old,new,'C50 manuscript device/environment and validation/test boundary')
    for name,text in output.items():
        with (D/name).open('xb') as f:f.write(text.encode('utf8'))
    for item in changes:
        item['new_line']=output[item['document']][:output[item['document']].index(item['new'])].count('\n')+1
        item['old_sha256']=hashlib.sha256(item['old'].encode('utf8')).hexdigest();item['new_sha256']=hashlib.sha256(item['new'].encode('utf8')).hexdigest()
    patch=''.join(''.join(difflib.unified_diff(originals[n].splitlines(keepends=True),output[n].splitlines(keepends=True),fromfile=(OLD/n).relative_to(R).as_posix(),tofile=(D/n).relative_to(R).as_posix())) for n in NAMES)
    with (D/'UPDATE_DIFF.patch').open('x',encoding='utf8',newline='\n') as f:f.write(patch)
    write(D/'SOURCE_CHANGES.json',dict(status='C50_COMPLETE_AUTHOR_REVIEW_TEXTUAL_DELTA_ONLY',source_pins=source_pins,changes=changes,unchanged_material_preserved=True,reverse_diff_must_recover_original_bytes=True,manuscript_applied=False,new_statistics=False))
    # Bind every C display cell occurring in changed prose to accepted C50 table pointers.
    bindings=[]
    for pi,p in enumerate(tables['panels']):
        if p['view']=='shared_calibration':continue  # native/shared identity is separately source-bound
        for ri,row in enumerate(p['rows']):
            if row['variant']!='minus_C minus Full':continue
            for metric in ['accuracy_pct','aeod','aspd']:
                precision=3 if metric=='accuracy_pct' else 5; v=row[metric]
                for spacing in ['±',' ± ']:
                    display=f"{v['mean']:+.{precision}f}{spacing}{v['sample_sd_ddof1']:.{precision}f}"
                    occurs={}
                    for n,text in output.items():
                        ls=[i for i,l in enumerate(text.splitlines(),1) if display in l and any(c['document']==n and c['new_line']<=i<c['new_line']+c['new'].count('\n')+1 for c in changes)]
                        if ls:occurs[n]=ls
                    if occurs:bindings.append(dict(source_path=(TABLE/'snapshot/tables.json').relative_to(R).as_posix(),pointer=f'/panels/{pi}/rows/{ri}/{metric}',value=v,decimal_places=precision,spacing=spacing,display=display,view=p['view'],panel_n=len(p['seeds']),scene=row['attack'],metric=metric,document_lines=occurs))
    pointers=dict(status='SOURCE_BOUND_C50_INTEGRATED_AUTHOR_REVIEW_PROSE_ONLY',root_adoption_sha256=sha(TABLE/'ROOT_VERIFICATION.json'),source_pins=source_pins,quoted_mean_SD_cells=bindings,fact_bindings=basis['fact_bindings'],direction_bindings=basis['direction_bindings'],accepted_short_update_sha256=sha(SHORT/'ROOT_REVIEW.json'),new_statistics=False,manuscript_applied=False)
    write(D/'SOURCE_POINTERS.json',pointers)
    print(json.dumps(dict(status='WRITTEN_PENDING_SOURCE_AND_TEXT_CHECK',changed_spans=len(changes),display_pointer_bindings=len(bindings),source_pins=len(source_pins))))

if __name__=='__main__':main()
