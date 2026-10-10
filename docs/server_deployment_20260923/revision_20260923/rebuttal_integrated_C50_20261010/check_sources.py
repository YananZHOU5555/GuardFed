"""Prose delta/source/comment/display/link checks; no scientific statistics."""
import argparse,datetime,hashlib,itertools,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;R=D.parents[1]
OLD=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009'
TABLE=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010'
NAMES=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def pointer(value,path):
    for key in path.lstrip('/').split('/') if path else []:
        key=key.replace('~1','/').replace('~0','~');value=value[int(key)] if isinstance(value,list) else value[key]
    return value

def quoted_comments(text):
    quotes=re.findall(r'\*\*Original comment \(verbatim\)\.\*\*\n\n((?:>[^\n]*\n)+)',text)
    return ['\n'.join(line[2:] if line.startswith('> ') else line[1:] for line in q.rstrip('\n').splitlines()) for q in quotes]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    assert not args.report.exists()
    source=read(D/'SOURCE_POINTERS.json');delta=read(D/'SOURCE_CHANGES.json');data={}
    assert source['source_pins']==delta['source_pins']
    for path,pin in source['source_pins'].items():
        p=R/path;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],path
        if p.suffix=='.json':data[path]=read(p)
    texts={n:(D/n).read_bytes().decode('utf8') for n in NAMES}
    # Exact reverse reconstruction proves every unedited byte and old number was retained.
    for name,text in texts.items():
        restored=text
        for change in reversed(delta['changes']):
            if change['document']!=name:continue
            assert restored.count(change['new'])==1
            assert hashlib.sha256(change['old'].encode('utf8')).hexdigest()==change['old_sha256']
            assert hashlib.sha256(change['new'].encode('utf8')).hexdigest()==change['new_sha256']
            restored=restored.replace(change['new'],change['old'],1)
            assert change['new'].splitlines()[0] in text.splitlines()[change['new_line']-1]
        assert restored.encode('utf8')==(OLD/name).read_bytes(),name
    mapped=set();quoted_cells=set()
    for binding in source['quoted_mean_SD_cells']:
        value=pointer(data[binding['source_path']],binding['pointer']);precision=binding['decimal_places'];spacing=binding['spacing']
        assert value==binding['value']
        display=f"{value['mean']:+.{precision}f}{spacing}{value['sample_sd_ddof1']:.{precision}f}"
        assert display==binding['display']
        quoted_cells.add((binding['pointer'],display))
        for name,lines in binding['document_lines'].items():
            for line in lines:
                assert display in texts[name].splitlines()[line-1]
                mapped.add((name,line,display))
    for change in delta['changes']:
        for offset,line in enumerate(change['new'].splitlines()):
            for cell in re.findall(r'[+\-]?\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?',line):
                assert (change['document'],change['new_line']+offset,cell) in mapped,('unbound C mean/SD cell',cell)
    for binding in source['fact_bindings']:assert pointer(data[binding['source_path']],binding['pointer'])==binding['value'],binding['label']
    for binding in source['direction_bindings']:
        row=pointer(data[binding['source_path']],binding['pointer'])
        assert [1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ['accuracy_pct','aeod','aspd']]==binding['signs']
    map_path=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/comment_source_map.json'
    verbatim=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/reviewer_comments_verbatim.md'
    comments=read(map_path)['comments'];original=verbatim.read_text('utf8')
    old_comment_order=[c['id'] for c in read(OLD/'SOURCE_CHANGES.json')['comments']]
    assert len(comments)==len(old_comment_order)==24 and set(old_comment_order)==set(comments)
    assert quoted_comments(texts[NAMES[0]])==[comments[key] for key in old_comment_order]
    assert quoted_comments(texts[NAMES[0]])==quoted_comments((OLD/NAMES[0]).read_text('utf8'))
    assert all(comment in original for comment in comments.values())
    root=read(TABLE/'ROOT_VERIFICATION.json');tables=read(TABLE/'snapshot/tables.json');records=read(TABLE/'snapshot/records.json')['records']
    assert sha(TABLE/'ROOT_VERIFICATION.json')==source['root_adoption_sha256']=='811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
    assert root['status']=='ROOT_C50_FIVE_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert (root['unique_records'],root['paired_models'],root['complete_scenes'])==(100,50,5)
    assert root['test'] is False and root['primary_endpoint']=='PENDING_AUTHOR' and root['whole_rebuttal_complete'] is False
    assert root['replay_devices']=={'Full':{'cpu':3,'cuda:0':47},'minus_C':{'cpu':50}}
    assert root['training_torch']=={'Full':{'2.11.0+cu128':50},'minus_C':{'2.11.0+cu128':50}}
    assert len(records)==len({r['id'] for r in records})==100
    assert {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in records}==set(itertools.product(('Full','minus_C'),('IID',),('Benign','F Flip','FedSA','S-DFA','Sp-DFA'),range(91001,91011)))
    assert all(r['data_contract']['evaluation_split']=='valid' and r['data_contract']['actual_evaluation_rows']==19867 for r in records)
    assert all(r['views']['native']==r['views']['shared_calibration'] for r in records)
    seedsets=[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
    assert len(tables['panels'])==9
    for p in tables['panels']:
        assert p['seeds'] in seedsets and all(r['n']==len(p['seeds']) and r['seeds']==p['seeds'] and r['complete'] for r in p['rows'])
    coverage=read(TABLE/'snapshot/coverage.json')
    assert all(len(rows)==10 and sum(r['complete'] for r in rows)==5 and all((r['distribution']=='IID' and r['n']==10) if r['complete'] else (r['distribution']=='non-IID' and r['n']==0) for r in rows) for rows in coverage.values())
    short=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_C50_update_20261010/C50_REVIEWER_ADDENDUM.md'
    short_text=short.read_text('utf8')
    for phrase in ['six other image-control variants','historical/current driver equality was not established','official-test exposure']:
        assert phrase in short_text
    # Source-bound changed-scope phrases and every pending register are visible in the complete copies.
    for name,text in texts.items():
        for phrase in ['AUTHOR_REVIEW — DO_NOT_SUBMIT_BEFORE_FULL_COHORT','five non-IID C scenes','six other image controls','seven image-control variants','not full equalized odds','official-test exposure','final test under a frozen final protocol has not been run','three CPU and 47 GPU','all 50 minus_C replays use CPU','PyTorch 2.11.0+cu128','historical/current driver equality was not established','The native/shared']:
            assert phrase in text,(name,phrase)
        assert 'three_view_C_two_scenes_20261009' not in text and 'C20' not in text
        assert 'five IID scenes' in text and 'within each seed' in text
        assert 'raw nine-seed fedsa' in text.lower()
        assert 'six-seed' in text and 'Sp-DFA' in text
    assert all(f'| P{i} —' in texts[NAMES[0]] for i in range(1,7))
    for phrase in ['all 12 deletion conditions','six have better means on all three metrics','0.64838/0.06130/0.04196','0.65999/0.05352/0.03752','0.250984','0.239597','0.044831','0.049143']:
        assert phrase in texts[NAMES[0]]
    # These original U100/nine-method paragraphs are outside the changed spans.
    for name in NAMES:
        old=(OLD/name).read_text('utf8')
        for line in old.splitlines():
            if line.startswith('**Accepted nine-method validation evidence.**'):assert line in texts[name]
        for marker in ['434 CPU and 466 GPU','886 cu128 and 14 cu130','100 minus_U','98 cu128 and two cu130','five CPU and 95 GPU']:
            assert marker in texts[name],(name,marker)
    links=[];external_links=0
    for name,text in texts.items():
        inherited=set(re.findall(r'\]\(([^)]+)\)',(OLD/name).read_text('utf8')))
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:')):
                assert target in inherited,('unverified new external link',target)
                external_links+=1
                links.append(dict(document=name,target=target,check='inherited exact URL; no online revalidation'))
            else:
                assert Path(target).is_absolute() and Path(target).is_file(),target
                links.append(dict(document=name,target=target,check='actual local file exists'))
    for path,pin in source['source_pins'].items():assert sha(R/path)==pin['sha256']
    result=dict(status='PASS_COMPLETE_24_COMMENT_C50_AUTHOR_REVIEW_TEXT_AND_SOURCES_ONLY',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_author_review_documents=2,original_comments_verbatim=24,original_comments_order_exact=True,prior_documents_reverse_diff_exact=2,changed_spans=len(delta['changes']),new_and_retained_C_display_pointer_bindings=len(quoted_cells),C_scalar_pointer_checks=2*len(quoted_cells),C_scope_fact_pointer_checks=len(source['fact_bindings']),direction_checks=len(source['direction_bindings']),source_files_pinned=len(source['source_pins']),links_checked=len(links),inherited_external_links_exact_not_revalidated_online=external_links,links=links,C50_unique_records=100,C50_matched_pairs=50,C50_complete_IID_scenes=5,C_nonIID_scenes_incomplete=5,other_image_controls_incomplete=6,seven_variants_incomplete=True,native_shared_equal_records=100,old_COMPAS_U100_900_calibration_numbers_preserved=True,Sp_tradeoff_and_preselected_9_to_6_reversal_retained=True,FedSA_raw9_all_three_mean_counterexample_retained=True,P1_P6_still_pending=True,primary_endpoint='PENDING_AUTHOR',manuscript_applied=False,AUTHOR_REVIEW_DO_NOT_SUBMIT=True,new_statistics=False,new_inference=0,new_training=0,SSH=False,STATE_or_Git_modified=False,source_inputs_unchanged=True,source_pointers_sha256=sha(D/'SOURCE_POINTERS.json'),documents_sha256={n:sha(D/n) for n in NAMES},scope='Writing/source/display/scope checks only; accepted arithmetic and archive proofs are read, not repeated.')
    with args.report.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='links'},ensure_ascii=False))

if __name__=='__main__':main()
