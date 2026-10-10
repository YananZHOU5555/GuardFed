"""Original C60 reversible/pointer checks, with exact adopted C100 scope checks."""
from pathlib import Path
import hashlib,json,re,collections
D=Path(__file__).resolve().parent;R=D.parents[1]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010'
TABLE=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010'
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
    for name,text in texts.items():
        restored=text
        for edit in reversed(delta['changes']):
            if edit['document']!=name:continue
            assert restored.count(edit['new'])==1
            for key in ('old','new'):assert hashlib.sha256(edit[key].encode()).hexdigest()==edit[key+'_sha256']
            restored=restored.replace(edit['new'],edit['old'],1)
        assert restored.encode()==(OLD/name).read_bytes()
        oldcells=collections.Counter(re.findall(r'[+\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?',(OLD/name).read_text('utf8')))
        newcells=collections.Counter(re.findall(r'[+\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?',text))
        assert all(newcells[k]>=v for k,v in oldcells.items())
        for phrase in ('AUTHOR_REVIEW — DO_NOT_SUBMIT_BEFORE_FULL_COHORT','six other image-control variants remain incomplete','not full equalized odds','official-test exposure','same round-70 checkpoint','not evidence of equivalence','not independent replications','identity projection','not real training-client fairness','LoGoFair-DP_07','does not establish a complete 100-cell LoGoFair evaluation','scenario count','first average scenarios within each seed'):
            assert phrase in text,(name,phrase)
        for stale in ('C remains incomplete over the four','seven image-control variants incomplete','seven variants are still incomplete','four non-IID attack scenarios and six other image controls remain incomplete'):
            assert stale not in text,(name,stale)
        assert 'final test under a frozen final protocol has not been run' in text
        for marker in ('434 CPU and 466 GPU','886 cu128 and 14 cu130','98 cu128 and two cu130','three CPU and 47 GPU','59 cu128 and one cu130','0.64838/0.06130/0.04196','0.65999/0.05352/0.03752','0.250984','0.239597','0.044831','0.049143'):
            assert marker in text,(name,marker)
    assert len(quotes(texts[NAMES[0]]))==24 and quotes(texts[NAMES[0]])==quotes((OLD/NAMES[0]).read_text('utf8'))
    assert all(f'| P{i} —' in texts[NAMES[0]] for i in range(1,7))
    assert 'U100 and C100 each have accepted/offserver' in texts[NAMES[0]]
    cells=bindings['numeric_cells'];locations=[]
    for cell in cells:
        value=pointer(read(TABLE/'snapshot'/cell['source']),cell['pointer']);assert value==cell['value']
        dp=cell['decimal_places'];display=f"{value['mean']:+.{dp}f} ± {value['sample_sd_ddof1']:.{dp}f}"
        assert display==cell['display']
        for name,text in texts.items():
            lines=[i+1 for i,line in enumerate(text.splitlines()) if display in line];assert lines
            locations.append(dict(document=name,source=cell['source'],pointer=cell['pointer'],display=display,lines=lines))
    tables=read(TABLE/'snapshot/tables.json');directions=[]
    expected={('native',10,26):[1,-1,-1],('native',9,26):[1,1,-1],('native',6,26):[1,1,-1],('native',10,29):[1,-1,1],('native',9,29):[1,1,1],('native',6,29):[1,1,1],('raw',10,26):[1,-1,-1],('raw',9,26):[1,-1,-1],('raw',6,26):[1,-1,1],('raw',10,29):[1,-1,-1],('raw',9,29):[1,-1,-1],('raw',6,29):[1,-1,1]}
    for p in tables['panels']:
        for row in (26,29):
            signs=[1 if p['rows'][row][m]['mean']>0 else -1 if p['rows'][row][m]['mean']<0 else 0 for m in ('accuracy_pct','aeod','aspd')]
            assert signs==expected[('native' if p['view']=='shared_calibration' else p['view'],len(p['seeds']),row)]
            directions.append(dict(view=p['view'],n=len(p['seeds']),row=row,signs=signs))
    assert 0<tables['panels'][1]['rows'][26]['aeod']['mean']<0.000005
    root=read(TABLE/'ROOT_VERIFICATION.json');records=read(TABLE/'snapshot/records.json')['records']
    assert sha(TABLE/'ROOT_VERIFICATION.json')==bindings['root_adoption_sha256']=='0bed6d372c63a60a978f8faa1353ec3257dcba66cc236b01906abd7097a0d0e7'
    assert (root['unique_records'],root['paired_models'],root['complete_scenes'])==(200,100,10)
    assert root['replay_devices']=={'Full':{'cpu':5,'cuda:0':95},'minus_C':{'cpu':100}}
    assert root['training_torch']=={'Full':{'2.11.0+cu128':98,'2.11.0+cu130':2},'minus_C':{'2.11.0+cu128':100}}
    assert all(r['views']['native']==r['views']['shared_calibration'] for r in records)
    assert root['primary_endpoint']=='PENDING_AUTHOR' and not root['test'] and not root['whole_rebuttal_complete']
    links=[]
    for name,text in texts.items():
        inherited=set(re.findall(r'\]\(([^)]+)\)',(OLD/name).read_text('utf8')))
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if link.startswith(('http:','https:')):assert link in inherited
            else:assert Path(link).is_absolute() and Path(link).is_file(),link
            links.append(dict(document=name,target=link,inherited=link in inherited))
    result=dict(status='PASS_COMPLETE_C100_AUTHOR_REVIEW_TEXT_SOURCE_CHECKS_ONLY',full_documents=2,original_comments_verbatim_and_order=24,reverse_reconstruction_exact=2,changed_spans=len(delta['changes']),new_mean_SD_cells=len(cells),new_scalar_pointer_checks=2*len(cells),numeric_locations=locations,direction_checks=3*len(directions),directions=directions,source_pins=len(bindings['source_pins']),links_checked=len(links),links=links,all_original_numeric_displays_preserved=True,old_COMPAS_U100_900_environment_disclosures_retained=True,C100_records=200,C100_pairs=100,C100_complete_scenes=10,remaining_nonIID_C_scenes=0,remaining_other_image_controls=6,original_five_IID_seed_first_aggregate_unchanged=True,P1_P6_pending=True,native_shared_primary_endpoint='PENDING_AUTHOR',manuscript_applied=False,whole_rebuttal_complete=False,test=False,new_inference=0,new_training=0,new_statistics=False,canonical_or_STATE_or_Git_modified=False,documents_sha256={n:sha(D/n) for n in NAMES})
    with (D/'CHECK_RESULTS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('links','numeric_locations','directions')}))
if __name__=='__main__':main()
