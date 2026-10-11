"""Original native collector with a metadata-only exact10 guard before backup."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];H=Path(__file__).resolve().parent
P=R/'tmp/celeba_native_after288_20261011/run_once.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
GUARD="""    if argv[len(prefix)]=='inspect':
        inspected=json.loads(Path(INSPECTION+'/inspection.json').read_bytes())
        assert sha(PRIOR_INSPECTION)==PRIOR_INSPECTION_SHA
        prior_inspected=json.loads(Path(PRIOR_INSPECTION).read_bytes())
        scope=dict(prior_native=prior_inspected['new_count'],observed_native=inspected['new_count'],fixed_ids=EXACT_IDS,actual_delta=sorted(set(inspected['accepted_new_ids'])-set(prior_inspected['accepted_new_ids'])),extra_ids=sorted(set(inspected['accepted_new_ids'])-set(prior_inspected['accepted_new_ids'])-set(EXACT_IDS)),missing_ids=sorted(set(EXACT_IDS)-set(inspected['accepted_new_ids'])),backup_dispatched=False)
        (base/'EXACT_DELTA_SCOPE.json').write_text(json.dumps(scope,indent=2))
        assert prior_inspected['new_count']==360 and len(prior_inspected['accepted_new_ids'])==360
        assert len(EXACT_IDS)==len(set(EXACT_IDS))==10 and not set(EXACT_IDS)&set(prior_inspected['accepted_new_ids'])
        assert set(prior_inspected['accepted_new_ids'])<=set(inspected['accepted_new_ids'])
        assert not inspected['invalid'] and inspected['new_count']==len(inspected['accepted_new_ids']) and inspected['new_count']>=370
        assert set(EXACT_IDS)<=set(scope['actual_delta']) and not scope['missing_ids'], 'Fixed10 incomplete/refused; preserve full inspection; no backup or retry'
"""
def derive():
    assert sha(P)=='9745525453d797b72a36aee66bade4fb3d3db04345fecd93535b68f44d142e88'
    assert sha(H/'PARENT_LEDGER.json')=='67c32aced70438bf7bf614c5862792f1887ddeb4fbede5e4da06cc157fd992ef'
    scope=json.loads((H/'FROZEN_SCOPE.json').read_bytes())
    assert scope['candidate_ids']==[f'minus_F_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
    assert sha(R/scope['parent_root_path'])==scope['parent_root_sha256']=='396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418'
    assert sha(H/'fixed_backup.py')==scope['fixed_backup_source_sha256']
    text=P.read_text('utf8')
    for old,new in [('native_delta_after288_20261011','native_delta_after360_nonIID_FFlip10_20261011'),('celeba_native_after288_20261011','mechanism_F_nonIID_FFlip10_prepare_20261011')]:
        assert text.count(old)>0;text=text.replace(old,new)
    anchor='    if result.returncode:break\n'
    assert text.count(anchor)==1;text=text.replace(anchor,anchor+GUARD)
    anchor='bindings=dict(TOOL=TOOL,HASH=HASH,'
    assert text.count(anchor)==1
    text=text.replace(anchor,"bindings=dict(FIXED_BACKUP_SOURCE=(H/'fixed_backup.py').read_text('utf8'),EXACT_IDS=SCOPE['candidate_ids'],PRIOR_INSPECTION=SCOPE['parent_remote_inspection_path'],PRIOR_INSPECTION_SHA=SCOPE['parent_inspection_sha256'],TOOL=TOOL,HASH=HASH,")
    old_call = '    result=subprocess.run(argv,capture_output=True,text=True)\n'
    new_call = '    if argv[len(prefix)]=="backup": argv=prefix[:-1]+["-c",FIXED_BACKUP_SOURCE]+argv[len(prefix)+1:]\n' + old_call
    assert text.count(old_call)==1
    text=text.replace(old_call,new_call)
    old_count = "'total_new_strict_and_offserver':report['new_count'],"
    new_count = "'total_new_strict_and_offserver':SCOPE['parent_native']+len(closed['accepted_new_ids']),'observed_native_count':report['new_count'],'observed_not_backed_up_ids':sorted(set(report['accepted_new_ids'])-set(SCOPE['candidate_ids'])-set(json.loads(Path(ROOT/SCOPE['parent_inspection_path']).read_bytes())['accepted_new_ids'])),'fixed_scope_only':True,"
    assert text.count(old_count)==1
    text=text.replace(old_count,new_count)
    return text,scope
if __name__=='__main__':
    text,scope=derive()
    exec(compile(text,str(P)+':FIXED10_SELECTION_METADATA_AFTER360','exec'),dict(__name__='__main__',__file__=__file__,SCOPE=scope))
