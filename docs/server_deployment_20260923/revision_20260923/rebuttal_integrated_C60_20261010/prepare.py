"""Minimal reversible C60 prose extension; accepted statistics are read, not recomputed."""
from pathlib import Path
import hashlib,json,difflib
D=Path(__file__).resolve().parent;R=D.parents[1]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010'
TABLE=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010'
ADD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_C60_validation_addendum_20261010.md'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
    with (D/name).open('w',encoding='utf8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    assert sha(TABLE/'ROOT_VERIFICATION.json')=='f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
    assert sha(ADD)=='8a92288f2f620261afa61dcb6339ccdcaad237de203d7a8cae830f1a71399510'
    assert [sha(OLD/n) for n in NAMES]==['aa4701445ace1c7477c08581d3a5d55f8a1b96ae111a23aba5280d65933f3d22','c6321e1b2fbe3b89a5db06a1eb4669659aa2f4c1bbaeac42bac80a8f1dd19f6d']
    root=read(TABLE/'ROOT_VERIFICATION.json');tables=read(TABLE/'snapshot/tables.json')
    assert sha(TABLE/'snapshot/tables.json')==root['tables_sha256'] and not root['whole_rebuttal_complete'] and not root['test']
    url=(TABLE/'snapshot/TABLES.md').as_posix();rooturl=(TABLE/'ROOT_VERIFICATION.json').as_posix()
    cells=[]
    def cell(panel,row,metric,signed=True):
        value=tables['panels'][panel]['rows'][row][metric];precision=3 if metric=='accuracy_pct' else 5
        fmt=('+' if signed else '')+'.'+str(precision)+'f'
        display=f"{format(value['mean'],fmt)} ± {value['sample_sd_ddof1']:.{precision}f}"
        cells.append(dict(pointer=f'/panels/{panel}/rows/{row}/{metric}',value=value,decimal_places=precision,signed=signed,display=display))
        return display
    triplet=lambda p,r,signed=True:' / '.join(cell(p,r,m,signed) for m in ('accuracy_pct','aeod','aspd'))
    full=triplet(0,15,False);minus=triplet(0,16,False);native=triplet(0,17);raw=triplet(3,17);nine=triplet(1,17);six=triplet(2,17)
    extension=(f'The later [six-scene C60 extension]({url}), independently adopted in [its root verification]({rooturl}) at {root["checked_utc"]}, adds ten matched non-IID Benign pairs to these unchanged five IID scenes. It contains 60 minus_C and 60 historical Full checkpoints; it does not complete the non-IID attack coverage. ')
    numerical=(f'**Non-IID Benign C-deletion extension.** The accepted [six-scene table]({url}) adds ten matched non-IID Benign pairs, using the same round-70 checkpoint per model and 19,867 validation images. Native/shared Full ACC/AEOD/ASPD are {full}, compared with {minus} after deleting C (ACC in percent; gaps on [0,1]). The paired differences (minus_C−Full) are {native} in native/shared and {raw} in raw, with ACC differences in percentage points; all triplets are means ± sample SD (ddof=1) across matched seeds. Removing C therefore increases accuracy in both views, lowers the native/shared AEOD mean slightly but increases ASPD, and increases both raw disparity means. These are retained trade-offs, not evidence that C benefits every metric or is universally indispensable. In the fixed nine- and six-seed native/shared panels, the corresponding paired triplets are {nine} and {six}. The accuracy and ASPD directions persist; the six-seed AEOD mean difference is close to zero (the displayed value rounds to zero), not evidence of equivalence. No significance or isolated aggregation-causality claim follows from these descriptive comparisons.')
    boundary=('**C60 scope and comparability.** Only five IID scenarios and non-IID Benign are complete; the four non-IID attack scenarios and six other image controls remain incomplete. Native/shared metrics and saved group counts coincide for all 120 C60 records; the views are not independent replications. Full replay uses five CPU and 55 GPU records, with 59 cu128 and one cu130 training checkpoints; all 60 minus_C records use CPU replay and cu128 training. These C60 counts extend, rather than replace, the original C50, U100 and nine-method environment disclosures. Historical/current driver equality is not established. The original five-IID-scenario seed-first aggregate is preserved byte for byte; no mean over the imbalanced six-scenario set is presented. All raw/native/shared 10/9/6-seed panels, selection history and prior official-test exposure remain disclosed. AEOD is the absolute TPR gap, not full equalized odds. This is exposed validation evidence, not an untouched test, a selected primary endpoint or a completed all-component image study.')
    changes=[];patches=[]
    for name in NAMES:
        old=(OLD/name).read_bytes().decode('utf8');out=[]
        for paragraph in old.split('\n\n'):
            new=paragraph
            if '**Evidence snapshot:**' in paragraph:
                new=paragraph.replace('23:55 UTC (10 October 2026, Australia/Sydney).', '23:55 UTC (10 October 2026, Australia/Sydney), extended with the C60 adoption of 10 October 2026, 00:59 UTC.')
            if paragraph.startswith('**AUTHOR_REVIEW'):
                new=new.replace('The five non-IID C scenes',extension+'The remaining four non-IID C scenes')
            else:
                new=new.replace('The five non-IID C scenes and six other image controls remain incomplete.','The remaining four non-IID C scenes and six other image controls remain incomplete.')
                new=new.replace('The five non-IID C scenes and six other image controls remain incomplete,','The remaining four non-IID C scenes and six other image controls remain incomplete,')
                new=new.replace('C remains incomplete over the five non-IID scenes;','C remains incomplete over the four non-IID attack scenes;')
                new=new.replace('They do not extrapolate to the five non-IID C scenes','They do not extrapolate to the four remaining non-IID C scenes')
            if paragraph.startswith('The U-deletion study now covers'):
                new=new.replace('The remaining four non-IID C scenes','The added non-IID Benign C comparison also retains a trade-off: deletion raises native accuracy and ASPD while slightly lowering the AEOD mean. The remaining four non-IID C scenes')
            if paragraph.startswith('The separately accepted [C-deletion table]') or paragraph.startswith('**Additional C-scene trade-offs.**'):
                new+='\n\n'+numerical+'\n\n'+boundary
            if paragraph.startswith('The five completed IID C scenes provide'):
                new+=' The additional non-IID Benign comparison in R3.2 extends this descriptive trade-off evidence; its accuracy and ASPD increases after deletion persist in the fixed nine-/six-seed panels, while the six-seed AEOD difference is near zero.'
            if paragraph.startswith('**Cross-scene interpretation.**'):
                new+=' This remains the original five-IID-scene summary; adding non-IID Benign does not create a balanced six-scene or full non-IID summary.'
            if '| P2 —' in paragraph:
                new=new.replace('Complete and accept the five non-IID C scenes','The later C60 extension adds ten non-IID Benign pairs while retaining those C50 values. Complete and accept the four remaining non-IID C attack scenes')
            if new!=paragraph:
                assert old.count(paragraph)==1 and new
                changes.append(dict(document=name,old=paragraph,new=new,old_sha256=hashlib.sha256(paragraph.encode()).hexdigest(),new_sha256=hashlib.sha256(new.encode()).hexdigest()))
            out.append(new)
        text='\n\n'.join(out)
        with (D/name).open('w',encoding='utf8',newline='\n') as f:f.write(text)
        patches.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='accepted_C50/'+name,tofile='prepared_C60/'+name))
    (D/'UPDATE_DIFF.patch').write_text(''.join(patches),encoding='utf8',newline='\n')
    sources=[OLD/n for n in NAMES]+[OLD/'ROOT_REVIEW.json',OLD/'SOURCE_POINTERS.json',OLD/'SOURCE_CHANGES.json',TABLE/'ROOT_VERIFICATION.json',TABLE/'snapshot/tables.json',TABLE/'snapshot/records.json',TABLE/'snapshot/cross_scene_seed_first.json',ADD,R/root['independent_review_path']]
    pins={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sources}
    save('SOURCE_CHANGES.json',dict(changes=changes,source_pins=pins))
    save('SOURCE_POINTERS.json',dict(source_pins=pins,table_source=(TABLE/'snapshot/tables.json').relative_to(R).as_posix(),root_adoption_sha256=sha(TABLE/'ROOT_VERIFICATION.json'),numeric_cells=cells,direction_row_pointers=[f'/panels/{p}/rows/17' for p in range(9)]))
    print(json.dumps(dict(status='PREPARED_WRITING_ONLY',documents=2,changed_spans=len(changes),numeric_cells=len(cells))))
if __name__=='__main__':main()
