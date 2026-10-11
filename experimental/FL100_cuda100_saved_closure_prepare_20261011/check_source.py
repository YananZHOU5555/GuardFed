"""Pure source/metadata checks; no remote/F/NumPy/Torch/array or scientific execution."""
from pathlib import Path
import ast,base64,hashlib,json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
for name in ['transport.py','audit_saved.py','metadata_gate.py']:
    inverse=read(H/(name+'.inverse.json'));s=(H/name).read_bytes();assert sha(H/name)==inverse['derived_sha256']
    for op in reversed(inverse['operations']):
        a=base64.b64decode(op['after_b64']);b=base64.b64decode(op['before_b64']);assert s.count(a)==op['count'];s=s.replace(a,b)
    assert s==(P/name).read_bytes() and hashlib.sha256(s).hexdigest()==inverse['parent_sha256']
old=(P/'audit_saved.py').read_bytes();loop=old[old.index(b'        for identity in IDS:'):old.index(b'        for rel,pin in members.items():')];assert (H/'audit_saved.py').read_bytes().count(loop)==1
for name in ['original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt','saved_fit_binding.py']:assert (P/name).read_bytes()==(H/name).read_bytes()
tree=ast.parse((H/'transport.py').read_text('utf8'));remote=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets));compile(remote,'remote future body','exec')
f=read(H/'SELECTED100_INPUTS.json');oldf=read(P/'SELECTED3_INPUTS.json');assert len(f['ids'])==len(set(f['ids']))==100 and f['canary_ids']==oldf['ids'] and len(f['remaining_ids'])==97 and not(set(f['canary_ids'])&set(f['remaining_ids'])) and set(f['canary_ids']+f['remaining_ids'])==set(f['ids'])
for r in oldf['rows']:assert next(v for v in f['rows'] if v['route']['id']==r['route']['id'])==r
assert {(r['original_row']['distribution'],r['original_row']['attack'],r['original_row']['seed']) for r in f['rows']}=={(d,a,s) for d in ['IID','non-IID'] for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA'] for s in range(91001,91011)}
from prior3_overlay import Fixed3Paths,old_member
x=Fixed3Paths(Path('SYNTHETIC_NEW'),Path('SYNTHETIC_OLD'),f['canary_ids']);positive=0
for i in f['ids']:
    prefix=Path('SYNTHETIC_OLD') if i in f['canary_ids'] else Path('SYNTHETIC_NEW');assert x/('base/runs/'+i)==prefix/'base/runs'/i;positive+=1
assert x/'source/metadata.npz'==Path('SYNTHETIC_OLD/source/metadata.npz')
assert not old_member('base/runs/'+f['canary_ids'][0]+'_foreign/validation_predictions.npz',f['canary_ids'])
denied=0
for rel in ['../bad','base/../bad','/absolute','C:/foreign']:
    try:x/rel
    except ValueError:denied+=1
    else:raise AssertionError('Traversal accepted')
report=dict(status='SOURCE_SCOPE100_DIFF97_REUSE3_CHECK_PASS_NOT_COMPLETION',full_raw_inverse_files=3,scientific_loop_raw_exact=True,science_source_files_raw_exact=5,metadata_routing_positive=positive+2,traversal_rejected=denied,ids=100,new97=97,reused3=3,duplicate_count=0,source_packages=sorted({r['route']['identity_source'] for r in f['rows']}),actual_completion=None,SSH=False,F=False,scientific_execution=False)
with (H/'SOURCE_CHECK.json').open('x',encoding='utf8') as o:json.dump(report,o,indent=2);o.write('\n')
print(json.dumps(report))
