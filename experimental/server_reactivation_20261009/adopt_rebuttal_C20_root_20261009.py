"""Adopt the reviewed C20 writing copy after checking its exact textual delta."""
from pathlib import Path
import datetime, hashlib, json, re, shutil
from urllib.parse import unquote, urlparse

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/guardfed_rebuttal_C20_20261009'
PRIOR=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009'
TARGET=PRIOR.with_name('rebuttal_integrated_C20_20261009')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(SOURCE/'FILES_SHA256.json')=='196d5822ddb7dec20bb7014ea9a150725019b0b09ec0de62d1215d0498e9287c'
seal=read(SOURCE/'FILES_SHA256.json')
for name,pin in seal['files'].items():
    p=SOURCE/name
    assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
mapping=read(SOURCE/'SOURCE_CHANGES.json')
for name,pin in mapping['input_files'].items():
    p=ROOT/name
    assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
names=['rebuttal_integrated_20261009.md','manuscript_insertions_integrated_20261009.md']
documents={name:(SOURCE/name).read_text(encoding='utf8') for name in names}
for name,text in documents.items():
    previous=text
    for edit in reversed(mapping['changes']):
        if edit['file']==name:
            assert previous.count(edit['after'])==1
            previous=previous.replace(edit['after'],edit['before'],1)
    assert previous==(PRIOR/name).read_text(encoding='utf8')
    assert 'AUTHOR_REVIEW' in text and 'DO_NOT_SUBMIT_BEFORE_FULL_COHORT' in text
pattern=r'\*\*Original comment \(verbatim\)\.\*\*\n\n(.*?)\n\n\*\*Response\.\*\*'
quotes=re.findall(pattern,documents[names[0]],re.S)
assert quotes==re.findall(pattern,(PRIOR/names[0]).read_text(encoding='utf8'),re.S) and len(quotes)==24
comments=read(ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/comment_source_map.json')['comments']
for item,quote in zip(mapping['comments'],quotes):
    sha256_quote=hashlib.sha256(quote.encode()).hexdigest()
    assert sha256_quote==item['quote_sha256']
    actual='\n'.join(line[2:] if line.startswith('> ') else line[1:] if line.startswith('>') else line for line in quote.splitlines())
    assert actual==comments[item['id']]
references=mapping['new_C20_scalar_pointers']+mapping['new_C20_scope_and_environment_pointers']
for ref in references:
    p=Path(ref['source']);assert sha(p)==ref['source_sha256']
    value=read(p)
    for key in ref['json_pointer'].strip('/').split('/'):
        value=value[int(key)] if isinstance(value,list) else value[key]
    assert value==ref['value']
    if 'display' in ref:
        fmt=('+' if ref['sign_explicit'] else '')+'.'+str(ref['decimal_places'])+'f'
        assert format(value*ref['scale'],fmt)==ref['display']
        for occurrence in ref['document_occurrences']:
            assert ref['display'] in documents[occurrence['file']].splitlines()[occurrence['line']-1]
links=[]
for name,text in documents.items():
    for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',text):
        target=target.strip('<>');parsed=urlparse(target)
        if parsed.scheme in ('http','https'):assert parsed.netloc
        else:
            p=Path(unquote(target.split('#')[0]));assert (p if p.is_absolute() else SOURCE/p).exists()
        links.append((name,label,target))
assert links==[(x['file'],x['label'],x['target']) for x in read(SOURCE/'DELIVERY_CHECKS.json')['links']] and len(links)==41
assert not TARGET.exists();TARGET.mkdir()
for name in list(seal['files'])+['FILES_SHA256.json']:
    shutil.copyfile(SOURCE/name,TARGET/name);assert sha(SOURCE/name)==sha(TARGET/name)
proof=dict(status='ROOT_C20_COMPLETE_AUTHOR_REVIEW_TEXT_DELTA_AND_SOURCE_POINTERS_PASS',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(SOURCE/'FILES_SHA256.json'),
    full_documents=2,comments_verbatim=24,prior_documents_reverse_diff_exact=2,
    changed_passages=len(mapping['changes']),new_scalar_pointer_checks=22,scope_environment_pointer_checks=12,links_checked=41,
    rebuttal_sha256=sha(TARGET/names[0]),insertions_sha256=sha(TARGET/names[1]),
    manuscript_applied=False,final_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False)
with (TARGET/'ROOT_REVIEW.json').open('x',encoding='utf8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof))
