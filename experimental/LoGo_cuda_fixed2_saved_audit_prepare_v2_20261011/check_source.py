"""Compile/exact-source and stdlib metadata fixtures only; never invoke transport/audit."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'SOURCE_ONLY_CHECK_PASS_NOT_RUNTIME_OR_SCIENCE','scientific_execution':False,'SSH':False,'F_access':False}
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
pins=json.loads((H/'SOURCE_REUSE.json').read_bytes())
for rel,p in pins['source_pins'].items():assert sha(R/rel)==p['sha256']
funcs=(H/'original_metric_functions.py.txt').read_bytes()
for entry in pins['functions']:
    p=R/entry['source'];lines=p.read_bytes().splitlines(keepends=True);node=next(n for n in ast.parse(p.read_text('utf8')).body if isinstance(n,ast.FunctionDef) and n.name==entry['name']);raw=b''.join(lines[node.lineno-1:node.end_lineno])
    assert hashlib.sha256(raw).hexdigest()==entry['raw_sha256'] and funcs.count(raw)==1
    orig=ast.dump(node,include_attributes=False);new=next(n for n in ast.parse(funcs).body if isinstance(n,ast.FunctionDef) and n.name==entry['name']);assert ast.dump(new,include_attributes=False)==orig
block=(H/'original_saved_decision_block.py.txt').read_bytes();assert hashlib.sha256(block).hexdigest()==pins['saved_decision_block_raw_sha256'] and (H/'audit_saved.py').read_bytes().count(block)==1
inverse=json.loads((H/'TRANSPORT_INVERSE.json').read_bytes());parent=(R/inverse['parent']).read_text('utf8');start=inverse['start_marker'];tail=(H/'transport.py').read_text('utf8').split(start,1)[1];original=parent.split(start,1)[1]
assert tail.replace('LOGO_FIXED2','OLD800_FIXED3_ATTEMPT1')==original
compile((H/'original_metric_functions.py.txt').read_bytes(),'metrics','exec')
spec=importlib.util.spec_from_file_location('source_only_logo_metadata',H/'metadata_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
x=json.loads((H/'ROOT_INPUTS.template.json').read_bytes());denied=0
try:g.input_gate(x)
except (ValueError,TypeError):denied+=1
else:raise AssertionError('Unbound template accepted')
base='/workspace/guardfed_checks/uniform_cuda_LoGo100_SYNTHETIC_NOT_AUTH'
x.update(server_base=base,service='guardfed_SYNTHETIC_ONLY',root_queue_auth={'path':base+'/ROOT_QUEUE_AUTH.json','sha256':'a'*64},deployment_seal={'path':base+'/DEPLOYMENT_SEAL.json','sha256':'b'*64},task_accepted={i:{'path':base+'/tasks/'+i+'/ACCEPTED.json','sha256':'c'*64} for i in g.IDS})
g.input_gate(x)
for mutation in [lambda q:q['exact_ids'].reverse(),lambda q:q['task_accepted'].pop(g.IDS[0]),lambda q:q['FedAvg_CUDA_root'].update(sha256='d'*64),lambda q:q['FedAvg_CUDA_index'].update(sha256='d'*64),lambda q:q.update(server_base=base+'/../bad'),lambda q:q['root_queue_auth'].update(path='/wrong'),lambda q:q['task_accepted'][g.IDS[0]].update(sha256=None)]:
    q=copy.deepcopy(x);mutation(q)
    try:g.input_gate(q)
    except (ValueError,TypeError):denied+=1
    else:raise AssertionError('Tampered inputs accepted')
report.update(original_metric_functions_raw_AST_exact=4,original_saved_decision_block_raw_exact=True,transport_tail_single_filename_inverse=True,metadata_positive=1,metadata_rejections=denied,actual_inputs_bound=False)
with (H/'SOURCE_CHECK.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
