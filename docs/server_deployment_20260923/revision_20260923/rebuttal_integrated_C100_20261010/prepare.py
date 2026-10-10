"""Reversible complete-author-draft extension from adopted C60 to adopted C100."""
from pathlib import Path
import json,hashlib,difflib
D=Path(__file__).resolve().parent;R=D.parents[1]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010'
TABLE=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010'
LOGO=R/'tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json'
DECISION=R/'tmp/celeba_gradient_screen64_v2_20261010/AUTHOR_DECISIONS.json'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
    with (D/name).open('w',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    assert sha(TABLE/'ROOT_VERIFICATION.json')=='0bed6d372c63a60a978f8faa1353ec3257dcba66cc236b01906abd7097a0d0e7'
    assert [sha(OLD/n) for n in NAMES]==['ee946d604ff56ed047071a53da12ff9ad9a23c32f9f7b3a3c5a83feeaabdc200','200ff6fa300f78bdaf9ae177c601b76343349973e758b9a1ce9fcbb9db20aa48']
    root=read(TABLE/'ROOT_VERIFICATION.json');tables=read(TABLE/'snapshot/tables.json');aggregate=read(TABLE/'snapshot/cross_scene_additional.json')
    assert (root['unique_records'],root['paired_models'],root['complete_scenes'])==(200,100,10)
    assert not root['test'] and not root['whole_rebuttal_complete']
    cells=[]
    def triplet(source,panel,row):
        tree=tables if source=='tables' else aggregate
        base=f'/panels/{panel}/rows/{row}' if source=='tables' else f'/{source}/{panel}/rows/{row}'
        target=tree['panels'][panel]['rows'][row] if source=='tables' else tree[source][panel]['rows'][row]
        output=[]
        for metric in ('accuracy_pct','aeod','aspd'):
            value=target[metric];dp=3 if metric=='accuracy_pct' else 5
            display=f"{value['mean']:+.{dp}f} ± {value['sample_sd_ddof1']:.{dp}f}"
            cells.append(dict(source='tables.json' if source=='tables' else 'cross_scene_additional.json',pointer=base+'/'+metric,value=value,decimal_places=dp,display=display))
            output.append(display)
        return ' / '.join(output)
    url=(TABLE/'snapshot/TABLES.md').as_posix();rooturl=(TABLE/'ROOT_VERIFICATION.json').as_posix()
    latest=(f'The current [complete C100 table]({url}), independently adopted in [the C100 root verification]({rooturl}), extends those unchanged historical subsets to 100 minus_C and 100 paired Full checkpoints: both distributions, all five scenarios and ten matched seeds. The fixed 10/9/6 raw/native/shared panels are complete for U and C; six other image-control variants remain incomplete. This completes two controls, not the all-component mechanism study.')
    attackrows=[]
    for attack,row in [('F Flip',20),('FedSA',23),('S-DFA',26),('Sp-DFA',29)]:
        attackrows.append(f'| non-IID {attack} | {triplet("tables",0,row)} | {triplet("tables",3,row)} |')
    numerical=('**Complete non-IID C-deletion extension.** The previously accepted C70/C80 increments added non-IID F Flip and FedSA; C100 now also includes all ten matched seeds for S-DFA and Sp-DFA. The original five-IID and non-IID Benign values above remain unchanged. The following table reports paired minus_C−Full means ± sample SD (ddof=1), with ACC differences in percentage points and gaps on [0,1], from the same round-70 checkpoints and 19,867 validation images. Native and shared-calibration values coincide in these accepted records.\n\n'
        '| Scene | Native/shared ΔACC / ΔAEOD / ΔASPD | Raw ΔACC / ΔAEOD / ΔASPD |\n|---|---|---|\n'+'\n'.join(attackrows)+'\n\n'
        'At ten seeds, non-IID S-DFA favors deletion on all three recorded metrics in both raw and native/shared views. Sp-DFA instead has a calibrated accuracy–ASPD trade-off: deletion improves accuracy and slightly lowers AEOD but increases ASPD; its raw means favor deletion on all three metrics. These retained counterexamples do not establish that C is indispensable or that any component improves every metric.')
    sensitivity=(f'**Fixed subset sensitivity for the newly completed C attacks.** For non-IID S-DFA, native/shared minus_C−Full triplets are {triplet("tables",1,26)} with nine seeds and {triplet("tables",2,26)} with six. For Sp-DFA they are {triplet("tables",1,29)} and {triplet("tables",2,29)}, respectively. Accuracy gains persist, but AEOD changes from a negative ten-seed difference to positive differences in these subsets; the S-DFA nine-seed value rounds to zero and is not evidence of equivalence. Sp-DFA calibrated ASPD remains higher after deletion. Raw ASPD changes from negative ten-/nine-seed differences to positive six-seed differences for both attacks. All three fixed panels are retained without selecting favorable seeds or attaching a significance claim.')
    aggregation=(f'**Balanced C100 seed-first summary.** The original five-IID seed-first artifact is unchanged. The additional [distribution and balanced summaries]({(TABLE/"snapshot/cross_scene_additional.json").as_posix()}) first average scenarios within each seed and then summarize the seed-level means; the sample size is ten, nine or six seeds, never the scenario count. At ten seeds, the five-non-IID native/shared paired triplet is {triplet("nonIID_five_scene_panels",0,2)}. For the balanced ten-scene set, native/shared gives {triplet("balanced_ten_scene_panels",0,2)} and raw gives {triplet("balanced_ten_scene_panels",3,2)}. Deleting C therefore raises mean accuracy while the calibrated ASPD mean increases. This supplementary balanced summary retains the scene tables and does not choose a primary endpoint, establish necessity or identify an isolated aggregation effect.')
    boundary=('**Current C100 comparability boundary.** Full replay uses five CPU and 95 GPU records; 100 minus_C replays use CPU. Full training includes 98 cu128 and two cu130 checkpoints, whereas all 100 C checkpoints report cu128. Historical/current driver equality and CPU/GPU numerical equivalence are not established. Native/shared metrics and saved group counts coincide for all 200 records; the views are not independent replications. These complete-cohort counts extend the earlier C50/C60 disclosures rather than changing those historical records. Root-only fitting, the fixed configuration-selection history including seed91001, prior validation and official-test exposure remain disclosed. AEOD is the absolute TPR gap, not full equalized odds. The six other image controls, remaining-method full coverage, frozen final evaluation and manuscript integration remain pending; no final test or full revision completion is claimed.')
    methodupdate=(f'**Remaining-baseline implementation status at this evidence cutoff.** The [author decision]({DECISION.as_posix()}) accepts Huber’s practical CNN adaptation with parameter domain R^p and identity projection; it does not inherit a convex-domain or covering-number guarantee. The [complete LoGoFair32 validation search]({LOGO.as_posix()}) has been independently accepted and selects LoGoFair-DP_07 by the original four-condition mean score and candidate tie rule. It uses the author-authorized image-ID hash into 20 declared virtual cohorts, the official DP objective and root-only fitting, not real training-client fairness. Its fit seed is fixed at 1719 and its search uses only seed 91001. This search acceptance does not establish a complete 100-cell LoGoFair evaluation; that fixed-recipe stage remains pending. These updates do not enlarge the accepted nine-method/900-record comparison to the 17-method target or erase constant-prediction and unfavorable results.')
    changes=[];diff=[]
    for name in NAMES:
        old=(OLD/name).read_text('utf8');parts=[]
        for p in old.split('\n\n'):
            new=p
            if p.startswith('>'):parts.append(p);continue
            if '**Evidence snapshot:**' in p:new=p+' Extended here with the independently adopted C100 table of 10 October 2026, 05:19 UTC; baseline search status is separately source-bound.'
            if p.startswith('**AUTHOR_REVIEW'):
                marker='It contains 60 minus_C and 60 historical Full checkpoints;'
                at=p.index(marker);end=p.index('The remaining eight-method benchmark coverage',at)
                new=p[:at]+'That historical C60 subset contains 60 minus_C and 60 Full checkpoints. '+latest+' '+p[end:]
            else:
                new=new.replace('The remaining four non-IID C scenes and six other image controls remain incomplete.','C100 now closes the remaining four non-IID C scenes; the six other image controls remain incomplete.')
                new=new.replace('C remains incomplete over the four non-IID attack scenes;','The complete C100 extension below now covers those four non-IID attack scenes;')
                new=new.replace('They do not extrapolate to the four remaining non-IID C scenes or the other incomplete image controls.','They do not by themselves extrapolate across scenes or to the other incomplete image controls; the complete C100 results below provide the actual non-IID comparisons.')
                new=new.replace('The remaining four non-IID C scenes and six other image controls remain incomplete, leaving seven image-control variants incomplete.','The complete C100 extension below closes the four remaining non-IID C scenes, leaving six other image-control variants incomplete.')
            if p.startswith('**C60 scope and comparability.'):
                new=p.replace('**C60 scope and comparability.** Only five IID scenarios and non-IID Benign are complete; the four non-IID attack scenarios and six other image controls remain incomplete.','**Historical C60 scope and comparability.** This earlier snapshot contains five IID scenarios and non-IID Benign; its scope is superseded by the complete C100 extension below.')
                new+='\n\n'+numerical+'\n\n'+sensitivity+'\n\n'+aggregation+'\n\n'+boundary
            if p.startswith('**Image U-deletion evidence.'):
                new=new.replace('The separate C50 extension below does not change these U100 records or conclusions; the other image-control variants remain incomplete.','The separate complete C100 extension below does not change these U100 records or conclusions; six other image-control variants remain incomplete.')
            if p.startswith('**Cross-scene interpretation.'):
                new=p.replace('This remains the original five-IID-scene summary; adding non-IID Benign does not create a balanced six-scene or full non-IID summary.','This original five-IID-scene summary is preserved; the separate C100 seed-first summary above now covers five non-IID and balanced ten-scene sets.')
            if p.startswith('The completed CelebA matrix now contains nine methods'):
                new=p.replace('They account for 800 missing matrix records;','They define an 800-cell target extension not yet completed in the accepted comparison;')+'\n\n'+methodupdate
            if p.startswith('**Nine-method same-checkpoint calibration control.'):
                new=p+'\n\n'+methodupdate
            if '| P2 —' in p:
                lines=new.splitlines();lines=[('| P2 — CelebA mechanisms; still pending | U100 and C100 each have accepted/offserver same-checkpoint raw/native/shared results for all ten scenes and fixed 10/9/6-seed panels. C100 preserves all earlier C50/C60 values and adds the remaining non-IID attacks. Complete and accept the six other image controls. Preserve Full/source/partition identities, all unfavorable effects and calibration/device/runtime/selection boundaries. | R3.2; R3.7; mechanism attribution |' if x.startswith('| P2 —') else x) for x in lines];new='\n'.join(lines)
            if new!=p:changes.append(dict(document=name,old=p,new=new,old_sha256=hashlib.sha256(p.encode()).hexdigest(),new_sha256=hashlib.sha256(new.encode()).hexdigest()))
            parts.append(new)
        result='\n\n'.join(parts)
        assert methodupdate in result,(name,'method status insertion point absent')
        with (D/name).open('w',encoding='utf8',newline='\n') as f:f.write(result)
        diff+=difflib.unified_diff(old.splitlines(True),result.splitlines(True),fromfile='accepted_C60/'+name,tofile='prepared_C100/'+name)
    (D/'UPDATE_DIFF.patch').write_text(''.join(diff),encoding='utf8',newline='\n')
    inputs=[OLD/n for n in NAMES]+[OLD/'ROOT_REVIEW.json',OLD/'SOURCE_POINTERS.json',TABLE/'ROOT_VERIFICATION.json',TABLE/'ACTUAL_FILES_SHA256.json',TABLE/'snapshot/tables.json',TABLE/'snapshot/records.json',TABLE/'snapshot/cross_scene_additional.json',TABLE/'snapshot/cross_scene_seed_first.json',LOGO,DECISION]
    pins={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in inputs}
    save('SOURCE_CHANGES.json',dict(changes=changes,source_pins=pins))
    save('SOURCE_POINTERS.json',dict(source_pins=pins,table_directory=TABLE.relative_to(R).as_posix(),numeric_cells=cells,root_adoption_sha256=sha(TABLE/'ROOT_VERIFICATION.json')))
    print(json.dumps(dict(status='PREPARED_C100_COMPLETE_DRAFT',changed_spans=len(changes),new_mean_SD_cells=len(cells))))
if __name__=='__main__':main()
