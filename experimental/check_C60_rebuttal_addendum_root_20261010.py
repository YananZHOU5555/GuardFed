from pathlib import Path
import hashlib,json,re
R=Path(__file__).resolve().parents[1]
P=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_C60_validation_addendum_20261010.md'
B=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010'
text=P.read_text(encoding='utf8')
table=json.loads((B/'snapshot/tables.json').read_bytes())
panels={(p['view'],len(p['seeds'])):p for p in table['panels']}
def rows(view,n):
    return {r['variant']:r for r in panels[view,n]['rows'] if (r['distribution'],r['attack'])==('non-IID','Benign')}
a=rows('native',10)
values=[format(a[v][metric][stat],fmt) for v in ('Full','minus_C')
        for metric,fmt in [('accuracy_pct','.3f'),('aeod','.5f'),('aspd','.5f')]
        for stat in ('mean','sample_sd_ddof1')]
values.append(format(a['minus_C minus Full']['accuracy_pct']['mean'],'.3f'))
raw=rows('raw',10)['minus_C minus Full']
values.extend(format(raw[k]['mean'],fmt) for k,fmt in [('accuracy_pct','.3f'),('aeod','.5f'),('aspd','.5f')])
assert all(v in text for v in values)
assert all(rows('native',n)['minus_C minus Full'][k]['mean']>0 for n in (9,6) for k in ('accuracy_pct','aspd'))
assert abs(rows('native',6)['minus_C minus Full']['aeod']['mean'])<.00001
for n in (10,9,6):assert rows('native',n)==rows('shared_calibration',n)
links=re.findall(r'\]\(([^)]+)\)',text)
assert len(links)==3 and all((P.parent/link).resolve().is_file() for link in links)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B/'ROOT_VERIFICATION.json') in text
result=dict(status='C60_AUTHOR_REVIEW_ADDENDUM_VALUES_LINKS_AND_SCOPE_PASS',
    scalar_render_checks=len(values),links=len(links),sha256=sha(P),
    C60_root_sha256=sha(B/'ROOT_VERIFICATION.json'),
    full_24_comment_response_modified=False,test=False)
P.with_suffix('.verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
