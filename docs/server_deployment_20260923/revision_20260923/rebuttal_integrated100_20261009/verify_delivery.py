"""Read-only writing checks: exact prior recovery, comments, citations and numbers."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse,unquote
H=Path(__file__).resolve().parent
R=H.parents[1]
P=R/'tmp/guardfed_rebuttal_integrated71_v2_20261009'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    mapping=read(H/'SOURCE_MAP.json');ledger=read(H/'UNCHANGED_PARAGRAPHS.json')
    for name,pin in mapping['input_files'].items():
        p=R/name;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
    documents={name:(H/name).read_text(encoding='utf8') for name in ledger}
    for name,text in documents.items():
        recovered=text
        for edit in mapping['changes']:
            if edit['file']==name:
                assert recovered.count(edit['after'])==1;recovered=recovered.replace(edit['after'],edit['before'],1)
        old=(P/name).read_text(encoding='utf8');assert recovered==old
        paragraphs=re.split(r'\n\s*\n',old)
        for row in ledger[name]:
            before=paragraphs[row['old_paragraph']-1]
            assert hashlib.sha256(before.encode()).hexdigest()==row['old_sha256']
            if row['unchanged']:assert row['old_sha256']==row['new_sha256'] and before in text
            else:
                edit=next(e for e in mapping['changes'] if (e['file'],e['old_paragraph'])==(name,row['old_paragraph']))
                assert hashlib.sha256(edit['after'].encode()).hexdigest()==row['new_sha256']
    pattern=r'\*\*Original comment \(verbatim\)\.\*\*\n\n(.*?)\n\n\*\*Response\.\*\*'
    rebuttal=documents['rebuttal_integrated_20261009.md'];quotes=re.findall(pattern,rebuttal,re.S)
    assert quotes==re.findall(pattern,(P/'rebuttal_integrated_20261009.md').read_text(encoding='utf8'),re.S) and len(quotes)==24
    comments=read(R/'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/comment_source_map.json')['comments']
    for proof,quote in zip(read(H/'COMMENT_CONSISTENCY.json')['comments'],quotes):
        assert proof['quote_sha256']==hashlib.sha256(quote.encode()).hexdigest()
        actual='\n'.join(line[2:] if line.startswith('> ') else line[1:] if line.startswith('>') else line for line in quote.splitlines())
        assert actual==comments[proof['id']]
    numerical=read(H/'NUMERIC_REFERENCES.json');sources={};refs=numerical['references']
    for ref in refs:
        p=Path(ref['source']);sources.setdefault(str(p),read(p));value=sources[str(p)]
        for key in ref['json_pointer'].strip('/').split('/'):value=value[int(key)] if isinstance(value,list) else value[key]
        assert value==ref['value']
        digits=len(ref['display'].split('.')[-1]);assert float(ref['display'])==round(value*ref['scale'],digits)
        assert ref['display'] in '\n'.join(documents.values())
    actual_links=[]
    for name,text in documents.items():
        for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',text):
            target=target.strip('<>');parsed=urlparse(target)
            if parsed.scheme in ('http','https'):assert parsed.netloc
            else:
                p=Path(unquote(target.split('#')[0]));assert (p if p.is_absolute() else H/p).exists()
            actual_links.append((name,label,target))
        assert 'DO_NOT_SUBMIT_BEFORE_FULL_COHORT' in text
        assert 'was revised' not in text.lower() and 'we have revised' not in text.lower()
    recorded=[(r['file'],r['label'],r['target']) for r in read(H/'LINK_CHECKS.json')['links']]
    assert actual_links==recorded and len(actual_links)==37
    for pending in ('P1','P2','P3','P4','P5','P6'):assert '| '+pending+' —' in rebuttal
    assert '| P2 — CelebA mechanisms; still pending |' in rebuttal
    assert 'six improve all three means' in rebuttal and 'all 12 COMPAS deletion conditions' in rebuttal
    assert '294 with n=1, 180 with n=10, and six with n=3' in rebuttal and 'SD is unavailable, not zero' in rebuttal
    assert '0.250984 versus Full 0.239597' in rebuttal and '0.044831 versus 0.049143' in rebuttal
    print(json.dumps(dict(status='READ_ONLY_DELIVERY_CHECKS_PASS',exact_prior_documents_recovered=2,comments_verbatim=24,source_value_format_checks=len(refs),links_checked=len(actual_links),changed_prior_paragraphs=len(mapping['changes']),unchanged_prior_paragraphs=sum(row['unchanged'] for rows in ledger.values() for row in rows),P1_P6_retained_pending=True,COMPAS_TableII_and_SD_counterexamples_retained=True,statistics_recomputed=False,network=False,source_inputs_unchanged=True)))

if __name__=='__main__':main()
