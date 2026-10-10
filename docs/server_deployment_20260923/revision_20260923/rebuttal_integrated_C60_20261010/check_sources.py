"""Read-only prose/source checks; no inference, fitting or new statistical analysis."""
from pathlib import Path
import hashlib,json,re,collections
D=Path(__file__).resolve().parent;R=D.parents[1]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010'
TABLE=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def pointer(v,path):
    for key in path.strip('/').split('/'):v=v[int(key)] if isinstance(v,list) else v[key]
    return v
def quotes(t):return re.findall(r'\*\*Original comment \(verbatim\)\.\*\*\n\n((?:>[^\n]*\n)+)',t)
def main():
    bindings=read(D/'SOURCE_POINTERS.json');delta=read(D/'SOURCE_CHANGES.json');texts={n:(D/n).read_bytes().decode() for n in NAMES}
    assert bindings['source_pins']==delta['source_pins']
    for path,pin in bindings['source_pins'].items():assert sha(R/path)==pin['sha256'] and (R/path).stat().st_size==pin['bytes']
    for n,text in texts.items():
        restored=text
        for edit in reversed(delta['changes']):
            if edit['document']!=n:continue
            assert restored.count(edit['new'])==1
            for key in ('old','new'):assert hashlib.sha256(edit[key].encode()).hexdigest()==edit[key+'_sha256']
            restored=restored.replace(edit['new'],edit['old'],1)
        assert restored.encode()==(OLD/n).read_bytes()
        # Every old numerical mean/SD display remains literally present, including counterexamples.
        oldcells=collections.Counter(re.findall(r'[+\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?',(OLD/n).read_text('utf8')))
        newcells=collections.Counter(re.findall(r'[+\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?',text))
        assert all(newcells[k]>=v for k,v in oldcells.items())
        for phrase in ('AUTHOR_REVIEW — DO_NOT_SUBMIT_BEFORE_FULL_COHORT','four non-IID attack scenarios','six other image controls','not full equalized odds','official-test exposure','same round-70 checkpoint','no mean over the imbalanced six-scenario set','59 cu128 and one cu130','all 60 minus_C records use CPU','not evidence of equivalence','not independent replications'):
            assert phrase in text,(n,phrase)
        assert 'The five non-IID C scenes and six other image controls remain incomplete' not in text
        assert 'final test under a frozen final protocol has not been run' in text
        for marker in ('434 CPU and 466 GPU','886 cu128 and 14 cu130','98 cu128 and two cu130','three CPU and 47 GPU','0.64838/0.06130/0.04196','0.65999/0.05352/0.03752','0.250984','0.239597','0.044831','0.049143'):
            assert marker in text,(n,marker)
    assert len(quotes(texts[NAMES[0]]))==24 and quotes(texts[NAMES[0]])==quotes((OLD/NAMES[0]).read_text('utf8'))
    assert all(f'| P{i} —' in texts[NAMES[0]] for i in range(1,7))
    p2=next(x for x in texts[NAMES[0]].splitlines() if x.startswith('| P2 —'))
    assert 'C60 extension adds ten non-IID Benign pairs' in p2 and 'four remaining non-IID C attack scenes' in p2
    tables=read(R/bindings['table_source']);cells=bindings['numeric_cells'];locations=[]
    for cell in cells:
        value=pointer(tables,cell['pointer']);assert value==cell['value']
        precision=cell['decimal_places'];fmt=('+' if cell['signed'] else '')+'.'+str(precision)+'f'
        display=f"{format(value['mean'],fmt)} ± {value['sample_sd_ddof1']:.{precision}f}"
        assert display==cell['display']
        for n,t in texts.items():
            lines=[i+1 for i,line in enumerate(t.splitlines()) if display in line];assert lines
            locations.append(dict(document=n,pointer=cell['pointer'],display=display,lines=lines))
    # Check actual directions for every view and predefined seed panel, without significance testing.
    directions=[]
    for ptr in bindings['direction_row_pointers']:
        row=pointer(tables,ptr);signs=[1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ('accuracy_pct','aeod','aspd')]
        p=int(ptr.split('/')[2]);assert signs==([1,1,1] if tables['panels'][p]['view']=='raw' else [1,-1,1])
        directions.append(dict(pointer=ptr,signs=signs))
    assert abs(tables['panels'][2]['rows'][17]['aeod']['mean'])<0.000005
    root=read(TABLE/'ROOT_VERIFICATION.json');records=read(TABLE/'snapshot/records.json')['records']
    assert sha(TABLE/'ROOT_VERIFICATION.json')==bindings['root_adoption_sha256']=='f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
    assert (root['unique_records'],root['paired_models'],root['complete_scenes'])==(120,60,6)
    assert root['replay_devices']=={'Full':{'cpu':5,'cuda:0':55},'minus_C':{'cpu':60}}
    assert root['training_torch']=={'Full':{'2.11.0+cu128':59,'2.11.0+cu130':1},'minus_C':{'2.11.0+cu128':60}}
    assert root['primary_endpoint']=='PENDING_AUTHOR' and not root['test'] and not root['whole_rebuttal_complete']
    assert all(r['views']['native']==r['views']['shared_calibration'] for r in records)
    assert all(r['data_contract']['evaluation_split']=='valid' and r['data_contract']['actual_evaluation_rows']==19867 for r in records)
    links=[]
    for n,t in texts.items():
        inherited=set(re.findall(r'\]\(([^)]+)\)',(OLD/n).read_text('utf8')))
        for link in re.findall(r'\]\(([^)]+)\)',t):
            if link.startswith(('http:','https:')):assert link in inherited
            else:assert Path(link).is_absolute() and Path(link).is_file(),link
            links.append(dict(document=n,target=link,inherited=link in inherited))
    result=dict(status='PASS_COMPLETE_C60_AUTHOR_REVIEW_TEXT_SOURCE_CHECKS_ONLY',full_documents=2,original_comments_verbatim_and_order=24,reverse_reconstruction_exact=2,changed_spans=len(delta['changes']),new_mean_SD_cells=len(cells),new_scalar_pointer_checks=2*len(cells),numeric_locations=locations,direction_checks=3*len(directions),directions=directions,source_pins=len(bindings['source_pins']),links_checked=len(links),links=links,all_original_numeric_displays_preserved=True,old_COMPAS_U100_900_environment_disclosures_retained=True,C60_records=120,C60_pairs=60,C60_complete_scenes=6,remaining_nonIID_C_scenes=4,remaining_other_image_controls=6,original_five_IID_seed_first_aggregate_unchanged=True,P1_P6_pending=True,native_shared_primary_endpoint='PENDING_AUTHOR',manuscript_applied=False,whole_rebuttal_complete=False,test=False,new_inference=0,new_training=0,new_statistics=False,canonical_or_STATE_or_Git_modified=False,documents_sha256={n:sha(D/n) for n in NAMES})
    with (D/'CHECK_RESULTS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('links','numeric_locations','directions')}))
if __name__=='__main__':main()
