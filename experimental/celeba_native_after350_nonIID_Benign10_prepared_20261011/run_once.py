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
        assert prior_inspected['new_count']==350 and len(prior_inspected['accepted_new_ids'])==350
        assert len(EXACT_IDS)==len(set(EXACT_IDS))==10 and not set(EXACT_IDS)&set(prior_inspected['accepted_new_ids'])
        assert set(prior_inspected['accepted_new_ids'])<=set(inspected['accepted_new_ids'])
        assert not inspected['invalid'] and inspected['new_count']==len(inspected['accepted_new_ids'])==360
        assert scope['actual_delta']==EXACT_IDS and not scope['extra_ids'] and not scope['missing_ids'], 'Exact10 cutoff refused; preserve inspection; no backup or retry'
"""
def derive():
    assert sha(P)=='9745525453d797b72a36aee66bade4fb3d3db04345fecd93535b68f44d142e88'
    assert sha(H/'PARENT_LEDGER.json')=='cfd096af7e787f419774e140ede4ebaadefe5770e5768c5754bd8932894050d0'
    scope=json.loads((H/'FROZEN_SCOPE.json').read_bytes())
    assert scope['candidate_ids']==[f'minus_F_non-IID_Benign_seed{s}' for s in range(91001,91011)]
    assert sha(R/scope['parent_root_path'])==scope['parent_root_sha256']=='706b528831145c2a2402eb040f1e8a19279b09b29171e99082aa1c410db69770'
    text=P.read_text('utf8')
    for old,new in [('native_delta_after288_20261011','native_delta_after350_nonIID_Benign10_20261011'),('celeba_native_after288_20261011','celeba_native_after350_nonIID_Benign10_prepared_20261011')]:
        assert text.count(old)>0;text=text.replace(old,new)
    anchor='    if result.returncode:break\n'
    assert text.count(anchor)==1;text=text.replace(anchor,anchor+GUARD)
    anchor='bindings=dict(TOOL=TOOL,HASH=HASH,'
    assert text.count(anchor)==1
    text=text.replace(anchor,"bindings=dict(EXACT_IDS=SCOPE['candidate_ids'],PRIOR_INSPECTION=SCOPE['parent_remote_inspection_path'],PRIOR_INSPECTION_SHA=SCOPE['parent_inspection_sha256'],TOOL=TOOL,HASH=HASH,")
    return text,scope
if __name__=='__main__':
    text,scope=derive()
    exec(compile(text,str(P)+':EXACT10_METADATA_GUARD_AFTER350','exec'),dict(__name__='__main__',__file__=__file__,SCOPE=scope))
