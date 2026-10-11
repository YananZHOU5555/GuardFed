"""Compile, inverse and synthetic metadata only; no remote/F/scientific imports."""
from pathlib import Path
import ast,hashlib,json,copy,sys,re,types
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
inverses=[]
for p in H.glob('*.INVERSE.json'):
 d=read(p);target=H/p.name.removesuffix('.INVERSE.json');s=target.read_bytes().decode('utf8')
 assert sha(target)==d['derived_sha256'] and sha(Path(d['parent']))==d['parent_sha256']
 for span in reversed(d['spans']):assert s.count(span['after'])==span['count'];s=s.replace(span['after'],span['before'])
 assert s.encode()==Path(d['parent']).read_bytes();inverses.append(target.name)
assert set(inverses)=={'audit_saved.py','metadata_gate.py','remote_export.py','transport.py','check_metadata_chain.py'}
for n in ['original_metric_functions.py.txt','original_saved_decision_block.py.txt','original_prefix_functions.py.txt','SOURCE_REUSE.json']:
 assert (H/n).read_bytes()==(P/n).read_bytes()
def loop(p):
 raw=p.read_bytes();start=raw.index(b'        for row,v in zip(refs[\'records\'],items):');end=raw.index(b'        for p,s in originals:',start);return raw[start:end]
assert loop(H/'audit_saved.py')==loop(P/'audit_saved.py')
assert ast.dump(ast.parse('def wrapper():\n'+loop(H/'audit_saved.py').decode()),include_attributes=False)==ast.dump(ast.parse('def wrapper():\n'+loop(P/'audit_saved.py').decode()),include_attributes=False)
cohort=R/'tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011/EXACT100.json'
assert (H/'EXACT100_REFS.json').read_bytes()==cohort.read_bytes() and sha(cohort)=='52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'
rows=read(cohort)['records'];identities=[]
for r in rows:
 m=re.fullmatch(r'LoGoFair-DP_07_FedAvg_(IID|non-IID)_(Benign|F Flip|FedSA|S-DFA|Sp-DFA)_seed(910\d\d)',r['post_id']);assert m
 identities.append((m[1],m[2].replace('F Flip','F-Flip'),int(m[3])))
assert len(set(identities))==100 and set(identities)=={(d,a,s) for d in ['IID','non-IID'] for a in ['Benign','F-Flip','FedSA','S-DFA','Sp-DFA'] for s in range(91001,91011)}
import metadata_gate as g
from prior2_overlay import merge_maps
assert g.IDS==[r['post_id'] for r in rows] and len(g.REMAINING_IDS)==98 and not(set(g.REMAINING_IDS)&set(g.CANARY_IDS))
assert "auth['canary_ids']==CANARY_IDS" in (H/'metadata_gate.py').read_text('utf8')
assert "job['seed']==int(row['post_id'].rsplit('_seed',1)[1])" in (H/'metadata_gate.py').read_text('utf8')
x=read(H/'ROOT_INPUTS.template.json');reject=0
try:g.input_gate(x)
except (TypeError,ValueError):reject+=1
else:raise AssertionError('Unbound input accepted')
B='/workspace/guardfed_checks/uniform_cuda_LoGo100_SYNTHETIC'
pin=lambda p:dict(path=p,sha256='a'*64)
x.update(server_base=B,service='guardfed_SYNTHETIC_CANARY',remaining_service='guardfed_SYNTHETIC_REMAINING',root_queue_auth=pin(B+'/ROOT_QUEUE_AUTH.json'),deployment_seal=pin(B+'/DEPLOYMENT_SEAL.json'),task_accepted={i:pin(B+'/tasks/'+i+'/ACCEPTED.json') for i in g.IDS},canary_complete=pin(B+'/canary/COMPLETE.json'),remaining_complete=pin(B+'/remaining/COMPLETE.json'),root_canary_gate=pin(B+'/ROOT_CANARY_ACCEPTANCE.json'))
g.input_gate(x)
for mutate in [lambda q:q['exact_ids'].reverse(),lambda q:q['task_accepted'].pop(g.IDS[0]),lambda q:q['FedAvg_CUDA_root'].update(sha256='b'*64),lambda q:q['remaining_complete'].update(path=B+'/canary/COMPLETE.json'),lambda q:q.update(prior2_root_sha256='b'*64),lambda q:q.update(remaining_service=None)]:
 q=copy.deepcopy(x);mutate(q)
 try:g.input_gate(q)
 except (TypeError,ValueError):reject+=1
 else:raise AssertionError('Tampered scope accepted')
new={'/workspace/new':'runs/new/scores_predictions.npz'};old={'/workspace/old':'runs/old/scores_predictions.npz'};nm={next(iter(new.values())):{'sha256':'a'*64,'bytes':1}};om={next(iter(old.values())):{'sha256':'b'*64,'bytes':2}}
merged=merge_maps(new,old,nm,om);assert len(merged[0])==len(merged[1])==2
map_reject=0
for args in [(new,new,nm,nm),(new,old,nm,nm),({'/workspace/new':'../wrong'},old,{'../wrong':{}},om),(new,old,{},om)]:
 try:merge_maps(*args)
 except ValueError:map_reject+=1
 else:raise AssertionError('Bad overlay accepted')
remote=ast.parse((H/'remote_export.py').read_bytes());checks=[n for n in ast.walk(remote) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='need']
def predicate(marker):
 node=next(n.args[0] for n in checks if len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value==marker)
 return compile(ast.Expression(node),'<metadata only>','eval')
service_test=predicate('Service not normally exited');service_ok=0;service_no=0
for rc,state,ok in [(3,'guardfed_SYNTHETIC EXITED timestamp',True),(0,'guardfed_SYNTHETIC EXITED timestamp',True),(0,'guardfed_SYNTHETIC RUNNING pid',False),(3,'other EXITED',False),(1,'guardfed_SYNTHETIC EXITED',False),(3,'',False)]:
 actual=eval(service_test,{'service':types.SimpleNamespace(returncode=rc,stdout=state),'service_name':'guardfed_SYNTHETIC'})
 assert actual==ok;service_ok+=ok;service_no+=not ok
comp_test=predicate('Incomplete/disordered Linux phase');auth={'exact_cohort':{'sha256':g.COHORT_SHA}};good=dict(ids=g.REMAINING_IDS,cohort=auth['exact_cohort'],server_saved_accepted=True)
assert eval(comp_test,{'complete':good,'ids':g.REMAINING_IDS,'auth':auth})
complete_reject=0
for q in [dict(good,ids=g.REMAINING_IDS[:-1]),dict(good,ids=list(reversed(g.REMAINING_IDS))),dict(good,server_saved_accepted=False),dict(good,cohort={'sha256':'b'*64})]:
 assert not eval(comp_test,{'complete':q,'ids':g.REMAINING_IDS,'auth':auth});complete_reject+=1
report=dict(status='SOURCE_ONLY_PASS_NOT_RUNTIME_OR_SCIENTIFIC_ACCEPTANCE',compile=True,full_raw_inverse=inverses,scientific_loop_raw_AST_exact=True,original_science_files_raw_exact=3,original_cohort100_byte_exact=True,grid='2 distributions x 5 attacks x 10 seeds',prior2_payloads=len(g.PRIOR_MEMBERS),new_scores=98,old_scores_referenced=2,scope_positive=1,scope_rejections=reject,overlay_positive=1,overlay_rejections=map_reject,service_positive=service_ok,service_rejections=service_no,completion_positive=1,completion_rejections=complete_reject,SSH=False,F_access=False,arrays=False,CNN=False,fit=False,actual_inputs_bound=False)
with (H/'SOURCE_CHECK.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
