from pathlib import Path
import base64, hashlib, json
R=Path(__file__).resolve().parents[1]
D=R/'tmp/FL100_cuda100_saved_closure_prepare_20261011'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(D/'FILES_SHA256.json')=='04f93dbb60fd74cbb059d8135f016064f9319a6444edd945582b3424b4394ae4'
for n,pin in read(D/'FILES_SHA256.json')['files'].items():
    p=D/n;assert p.name==n and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
    if n.endswith('.py'):compile(p.read_bytes(),str(p),'exec')
for n in ['audit_saved.py','transport.py','metadata_gate.py']:
    v=read(D/(n+'.inverse.json'));b=(D/n).read_bytes()
    assert hashlib.sha256(b).hexdigest()==v['derived_sha256']
    for op in reversed(v['operations']):
        before=base64.b64decode(op['before_b64']);after=base64.b64decode(op['after_b64'])
        assert b.count(after)==op['count'],n
        b=b.replace(after,before)
    parent=R/v['parent'];assert sha(parent)==v['parent_sha256'] and b==parent.read_bytes(),n
for n in ['original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt','original_fl_audit.py.txt','saved_fit_binding.py']:
    assert (D/n).read_bytes()==(R/'tmp/FL100_cuda3_saved_closure_prepare_20261011'/n).read_bytes(),n
selection=read(D/'SELECTED100_INPUTS.json');ids=selection['ids']
assert len(ids)==len(set(ids))==100 and len(selection['canary_ids'])==3 and len(selection['remaining_ids'])==97
assert set(selection['canary_ids']).isdisjoint(selection['remaining_ids']) and set(selection['canary_ids']+selection['remaining_ids'])==set(ids)
overlay=(D/'prior3_overlay.py').read_text(encoding='utf-8')
assert "new_transport['remote']['reused_member_pins']==keep" in overlay
assert 'root[\'records\'][i][\'task_acceptance_sha256\']' in overlay
assert 'not(set(keep)&set(new_members))' in overlay
proof={'status':'ROOT_SOURCE_REVIEW_PASS_FL100_ORIGINAL_SCIENCE_DELTA97_REUSE3',
    'source_adoptable':True,'source_seal_sha256':sha(D/'FILES_SHA256.json'),
    'three_complete_source_inverses_verified':True,'five_original_science_files_byte_exact':True,
    'no_changed_fit_prediction_metric_or_tolerance_math':True,
    'delta_policy':'Hash all100 live; transport97 new arrays only; retain28 prior3 F members at exact original paths with original root/transport/SHA links; reject overlap.',
    'target_records':100,'reused':3,'new':97,'target_metrics':900,'target_counts':2400,'target_rules':300,
    'actual_completion_not_inferred':True,'actual_transport':False,'actual_audit':False,'scientific_adoption':False,
    'original_native_tolerance':1e-12,'native_max_difference_must_be_measured':True,'test':False}
p=D/'ROOT_SOURCE_REVIEW.json';assert not p.exists()
p.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'source_adoptable':True}))
