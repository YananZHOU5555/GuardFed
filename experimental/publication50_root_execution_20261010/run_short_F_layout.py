"""Path-length recovery only: original sealed publisher, shorter F output root."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,traceback
R=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
P=R/'tmp/publication50_actual_scope_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'FILES_SHA256.json')=='fc42fa79466ab920db0631c47588eaed6f525638fa0d50822f4f5af6b001a550'
for name,pin in json.loads((P/'FILES_SHA256.json').read_bytes())['files'].items():
    assert sha(P/name)==pin['sha256'] and (P/name).stat().st_size==pin['bytes']
s=importlib.util.spec_from_file_location('original_sealed50_for_short_path',P/'publish_increment50.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
old=Path('F:/YananResearchStorage/GuardFed/git_publication/increment50')
new=Path('F:/YananResearchStorage/GuardFed/git_publication/i50')
assert m.ns['OUTPUT_ROOT']==old and m.ns['ROOT']==R and m.original_source.__globals__['ROOT']==R
assert new.parent==old.parent and new.drive.upper()=='F:'
m.ns['OUTPUT_ROOT']=new
a=argparse.ArgumentParser(description=__doc__);a.add_argument('action',choices=['freeze','stage']);v=a.parse_args()
if v.action=='freeze':
    path=P/'ACTUAL_SPEC.json';expected='5b7ca1b50c27f9b7559895e82cd082930fcb348dbac90c74dfe75009358df93b';name='i2'
    d=json.loads(path.read_bytes());longest=max(len(str(new/name/'source'/e['source'])) for e in d['files'])
    assert longest<=252 and not (new/name).exists()
else:
    path=new/'i2/FROZEN_INPUTS.json';expected=json.loads((HERE/'SHORT_F_FREEZE.json').read_bytes())['frozen_inputs_sha256'];name='s2'
    assert not (new/name).exists()
try:
    result=(m.freeze if v.action=='freeze' else m.stage)(path,expected,name)
    if v.action=='freeze':
        p=new/name/'FROZEN_INPUTS.json'
        receipt=dict(status='ORIGINAL50_FREEZE_PASS_SHORT_F_LAYOUT_ONLY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(P/'FILES_SHA256.json'),actual_spec_sha256=expected,old_partial_retained=str(old/'inputs001'),output_root=str(new),frozen_inputs_path=str(p),frozen_inputs_sha256=sha(p),files=len(result['files']),max_snapshot_path_characters=longest,science_changed=False,storage_and_alternates_ROOT=str(m.ns['ROOT']),wrapper_sha256=sha(__file__))
        with (HERE/'SHORT_F_FREEZE.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
        print(json.dumps(receipt))
except BaseException:
    p=HERE/('SHORT_F_'+v.action.upper()+'_FAILURE.json')
    with p.open('x',encoding='utf-8') as f:json.dump(dict(traceback=traceback.format_exc(),no_auto_retry=True),f,indent=2)
    raise
