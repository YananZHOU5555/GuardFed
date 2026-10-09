"""Bind the C30 author-review addendum to its actual table and quoted sources."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,sys
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/guardfed_rebuttal_C30_20261009'
TARGET=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C30_addendum_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(SOURCE/'FILES_SHA256.json')=='8276afbd89a40041d4204dcc9cd3245092c005d6e79a4077fe4ca28a058566ad'
files=read(SOURCE/'FILES_SHA256.json')['files'];assert len(files)==8
for name,pin in files.items():assert sha(SOURCE/name)==pin['sha256'] and (SOURCE/name).stat().st_size==pin['bytes']
handoff=read(SOURCE/'HANDOFF.json');pointers=read(SOURCE/'SOURCE_POINTERS.json');checked=read(SOURCE/'CHECK_RESULTS.json')
assert handoff['root_table_sha256']==pointers['root_adoption_sha256']=='2cce0519555efbff75f559875c4c1afe63e07d242cb8e3ebe6af644efc95961d'
assert sha(ROOT/handoff['root_table_path'])==handoff['root_table_sha256']
assert handoff['author_review_only'] and handoff['DO_NOT_SUBMIT'] and not handoff['manuscript_applied'] and not handoff['old_24_comment_response_modified']
data={}
for name,pin in pointers['source_pins'].items():
    path=ROOT/name;assert sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes']
    if path.suffix=='.json':data[name]=read(path)
documents={n:(SOURCE/n).read_text('utf8') for n in handoff['new_documents']}
def pointer(value,path):
    for part in path.lstrip('/').split('/') if path else []:
        part=part.replace('~1','/').replace('~0','~');value=value[int(part)] if isinstance(value,list) else value[part]
    return value
for row in pointers['quoted_scalar_bindings']:
    actual=pointer(data[row['source_path']],row['pointer']);assert actual==row['value'] and type(actual)==type(row['value'])
    for line in row['document_lines']:assert row['paired_cell'] in documents[row['document']].splitlines()[line-1]
for row in pointers['quoted_mean_SD_cells']:
    actual=pointer(data[row['source_path']],row['base_pointer']);places=row['decimal_places']
    display=f"{actual['mean']:+.{places}f} ± {actual['sample_sd_ddof1']:.{places}f}";assert display==row['display']
    for line in row['document_lines']:assert display in documents[row['document']].splitlines()[line-1]
for row in pointers['fact_bindings']:
    assert pointer(data[row['source_path']],row['pointer'])==row['value']
    for name,phrase in row['document_phrases'].items():assert phrase in documents[name]
for row in pointers['verbatim_comment_bindings']:
    assert (ROOT/row['source_path']).read_text('utf-8-sig').splitlines()[row['source_line']-1]==row['text'] and row['text'] in documents[row['document']]
links=[]
for name,text in documents.items():
    assert 'AUTHOR_REVIEW / DO_NOT_SUBMIT' in text and 'not applied' in text and 'seven C' in text and 'six other image-control variants' in text
    for target in re.findall(r'\]\(([^)]+)\)',text):assert Path(target).is_absolute() and Path(target).is_file();links.append((name,target))
assert (len(pointers['source_pins']),len(pointers['quoted_scalar_bindings']),len(pointers['quoted_mean_SD_cells']),len(pointers['fact_bindings']),len(links))==(11,98,49,25,9)
assert checked['status']=='PASS_C30_AUTHOR_REVIEW_WRITING_NUMERIC_POINTERS_SCOPE_AND_LINKS' and checked['source_pointers_sha256']==sha(SOURCE/'SOURCE_POINTERS.json')
assert checked['verbatim_original_comment_excerpts_exact']==2 and checked['negative_direction_checks_passed']==7
assert not TARGET.exists();TARGET.mkdir()
for name in list(files)+['FILES_SHA256.json']:
    shutil.copyfile(SOURCE/name,TARGET/name);assert sha(SOURCE/name)==sha(TARGET/name)
proof=dict(status='ROOT_C30_AUTHOR_REVIEW_ADDENDUM_QUOTED_VALUES_SCOPE_AND_SOURCE_PINS_PASS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'),root_table_sha256=handoff['root_table_sha256'],
    scalar_pointer_checks=98,display_cells_checked=49,scope_fact_checks=25,source_pins_checked=11,links_checked=9,verbatim_comment_excerpts=2,
    old_24_comment_draft_unchanged=True,complete_C_scenes=3,excluded_partial_C_records=6,all_negative_results_retained=True,
    addendum_sha256=sha(TARGET/'C30_REVIEWER_ADDENDUM.md'),insertions_sha256=sha(TARGET/'C30_MANUSCRIPT_INSERTIONS.md'),
    author_review_only=True,manuscript_applied=False,final_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False,new_CNN=0,new_training=0)
with (TARGET/'ROOT_REVIEW.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof|dict(root_path=str(TARGET/'ROOT_REVIEW.json'),root_sha256=sha(TARGET/'ROOT_REVIEW.json'))))
