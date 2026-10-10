"""Original reversible-text/JSON-pointer checks extended to actual A20/LoGo100."""
from pathlib import Path
import collections,difflib,hashlib,json,re,sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;R=D.parents[3]
OLD=D.parent/'rebuttal_integrated_C100_20261010'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def pointer(value,path):
    for key in path.strip('/').split('/'):value=value[int(key)] if isinstance(value,list) else value[key]
    return value
def quotes(text):return re.findall(r'\*\*Original comment \(verbatim\)\.\*\*\n\n((?:>[^\n]*\n)+)',text)

def check():
    b=read(D/'SOURCE_POINTERS.json');delta=read(D/'SOURCE_CHANGES.json')
    assert b['LoGo100_root_adoption_received'] and delta['LoGo100_root_adoption_received']
    assert b['source_pins']==delta['source_pins']
    for path,pin in b['source_pins'].items():assert sha(R/path)==pin['sha256'] and (R/path).stat().st_size==pin['bytes'],path
    for fact in b['json_facts']+b['adaptation_json_facts']:assert pointer(read(R/fact['source']),fact['pointer'])==fact['value'],fact
    for segment in b['adaptation_source_segments']:
        value='\n'.join((R/segment['source']).read_text('utf8').splitlines()[segment['first_line']-1:segment['last_line']])
        assert value==segment['text'] and hashlib.sha256(value.encode()).hexdigest()==segment['sha256']
    for item in b['derived_source_counts']:
        assert dict(collections.Counter(pointer(r,'/'+item['field']) for r in read(R/item['source'])))==item['expected']
    texts={n:(D/n).read_text('utf8') for n in NAMES};diff=[];links=[];locations=[]
    base_quotes=quotes((OLD/NAMES[0]).read_text('utf8'))
    assert len(base_quotes)==24 and quotes(texts[NAMES[0]])==base_quotes
    letter=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/reviewer_comments_verbatim.md'
    letter_text=' '.join(letter.read_text('utf8').split())
    for quote in base_quotes:
        unquoted='\n'.join(line[2:] if line.startswith('> ') else line[1:] for line in quote.splitlines())
        assert ' '.join(unquoted.split()) in letter_text,unquoted[:120]
    for name,text in texts.items():
        original=(OLD/name).read_text('utf8');restored=text
        for edit in reversed(delta['changes']):
            if edit['document']!=name:continue
            assert restored.count(edit['new'])==1,(name,edit['reason'])
            for key in ('old','new'):assert hashlib.sha256(edit[key].encode()).hexdigest()==edit[key+'_sha256']
            restored=restored.replace(edit['new'],edit['old'],1)
        assert restored.encode()==(OLD/name).read_bytes(),name
        oldcells=collections.Counter(re.findall(r'[+\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?',original))
        newcells=collections.Counter(re.findall(r'[+\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?',text))
        assert all(newcells[k]>=v for k,v in oldcells.items()),name
        strip_links=lambda value:re.sub(r'\]\([^)]*\)',']()',value)
        olddec=collections.Counter(re.findall(r'(?<!\w)[+\-]?\d+\.\d+',strip_links(original)))
        newdec=collections.Counter(re.findall(r'(?<!\w)[+\-]?\d+\.\d+',strip_links(text)))
        assert all(newdec[k]>=v for k,v in olddec.items()),name
        for phrase in ('AUTHOR_REVIEW — DO_NOT_SUBMIT_BEFORE_FULL_COHORT','identity projection','without transferring the original convex-domain or covering-number guarantee',
            'does not evaluate fairness across real training clients','DP here means demographic parity, not differential privacy','`valid_native_prediction`','`prediction`',
            'must be labelled separately from LoGoFair native','fixed fit seed 1719','predicts every validation image negative','does not estimate variability over postprocessor-fit seeds','post-initial-test validation development','240-run initial cohort had already been inspected',
            'Full replay comprises two CPU and 18 GPU models','A remains incomplete over eight scenes','other eight A scenes','not full equalized odds','official-test exposure',
            'final test under a frozen final protocol has not been run','not a ten-method shared-calibration or three-view comparison','six other image-control variants remain incomplete',
            'first average scenarios within each seed','GuardFed-AD2+ has higher accuracy and lower AEOD, whereas LoGoFair has lower ASPD',
            'No final primary endpoint or final-test result is supplied','0.64838/0.06130/0.04196','0.65999/0.05352/0.03752','0.250984','0.239597','0.044831','0.049143',
            '434 CPU and 466 GPU','886 cu128 and 14 cu130','98 cu128 and two cu130'):
            assert phrase in text,(name,phrase)
        for stale in ('that fixed-recipe stage remains pending','The remaining eight methods are LoGoFair','remaining eight-method coverage gap'):
            assert stale not in text,(name,stale)
        inherited=set(re.findall(r'\]\(([^)]+)\)',original))
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if link.startswith(('http:','https:')):assert link in inherited
            else:assert Path(link).is_absolute() and Path(link).is_file(),link
            links.append(dict(document=name,target=link,inherited=link in inherited))
        diff.extend(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='adopted_C100/'+name,tofile='A20_LoGo100_author_review/'+name))
    assert ''.join(diff)==(D/'UPDATE_DIFF.patch').read_text('utf8')
    assert all(f'| P{i} —' in texts[NAMES[0]] for i in range(1,7))
    for cell in b['numeric_cells']:
        value=pointer(read(R/cell['source']),cell['pointer']);assert value==cell['value']
        dp=cell['decimal_places'];sign='+' if cell.get('mean_signed',True) else ''
        display=f"{value['mean']:{sign}.{dp}f} ± {value['sample_sd_ddof1']:.{dp}f}"
        assert display==cell['display']
        for name,text in texts.items():
            lines=[i+1 for i,line in enumerate(text.splitlines()) if display in line];assert lines,(name,display)
            locations.append(dict(document=name,source=cell['source'],pointer=cell['pointer'],display=display,lines=lines,units=cell['units']))
    for cell in b['scalar_displays']:
        value=pointer(read(R/cell['source']),cell['pointer']);assert value==cell['value']
        display=f"{cell['multiply']*value:.{cell['decimal_places']}f}";assert display==cell['display']
        assert all(display+'%' in text for text in texts.values())
    a=read(R/b['A_table_directory']/'tables.json');directions=[]
    expected={('native',10,'Benign'):[-1,1,-1],('native',9,'Benign'):[-1,1,-1],('native',6,'Benign'):[-1,-1,-1],
        ('native',10,'F Flip'):[1,1,-1],('native',9,'F Flip'):[-1,-1,-1],('native',6,'F Flip'):[-1,-1,-1],
        ('raw',10,'Benign'):[-1,1,-1],('raw',9,'Benign'):[1,1,1],('raw',6,'Benign'):[-1,-1,-1],
        ('raw',10,'F Flip'):[1,1,1],('raw',9,'F Flip'):[1,1,1],('raw',6,'F Flip'):[-1,1,1]}
    for panel in a['panels']:
        for row in panel['rows']:
            if row['variant']!='minus_A minus Full':continue
            signs=[1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ('accuracy_pct','aeod','aspd')]
            view='native' if panel['view']=='shared_calibration' else panel['view']
            assert signs==expected[view,len(panel['seeds']),row['attack']]
            directions.append(dict(view=panel['view'],n=len(panel['seeds']),attack=row['attack'],signs=signs))
    aggregate=read(R/'outputs/guardfed_tables/celeba_ten_method_native_20261010/seed_first_aggregates.json')
    ad2,logo=aggregate['ten'][26],aggregate['ten'][29]
    assert (ad2['method'],logo['method'],ad2['scope'],logo['scope'])==('GuardFed-AD2+','LoGoFair-DP','balanced_all10','balanced_all10')
    assert ad2['accuracy']['mean']>logo['accuracy']['mean'] and ad2['aeod']['mean']<logo['aeod']['mean'] and logo['aspd']['mean']<ad2['aspd']['mean']
    result=dict(status='PASS_COMPLETE_A20_LOGO100_AUTHOR_REVIEW_TEXT_SOURCE_CHECKS_ONLY',full_documents=2,original_comments_verbatim_and_order=24,
        original_comment_letter_blocks_checked=24,reverse_reconstruction_exact=2,changed_spans=len(delta['changes']),all_original_mean_SD_and_decimal_values_preserved=True,
        numeric_mean_SD_cells=len(b['numeric_cells']),numeric_mean_SD_scalar_pointers=2*len(b['numeric_cells']),scalar_displays=len(b['scalar_displays']),numeric_locations=locations,
        fact_pointers=len(b['json_facts'])+len(b['adaptation_json_facts']),adaptation_source_segments=len(b['adaptation_source_segments']),derived_environment_counts=len(b['derived_source_counts']),
        A_direction_sign_checks=3*len(directions),A_directions=directions,ten_scene_native_tradeoff_mean_checks=6,source_pins=len(b['source_pins']),links_checked=len(links),links=links,
        A20_complete_scenes=2,A100_complete=False,LoGo100_root_adopted=True,ten_method_native_records=1000,nine_method_three_view_records=900,full17_complete=False,
        old_COMPAS_U100_C100_900_environment_disclosures_retained=True,P1_P6_pending=True,final_primary_endpoint='PENDING_AUTHOR',AUTHOR_REVIEW=True,DO_NOT_SUBMIT=True,
        manuscript_applied=False,whole_rebuttal_complete=False,new_statistics=False,new_inference=0,new_fits=0,new_training=0,final_test=False,STATE_or_Git_modified=False)
    if sys.argv[1:]==['write']:
        with (D/'CHECK_RESULTS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    elif sys.argv[1:]:raise ValueError('Use write once or no args for read-only checking')
    print(json.dumps({k:v for k,v in result.items() if k not in ('numeric_locations','A_directions','links')}))
    return result

if __name__=='__main__':check()
