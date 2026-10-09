"""Bounded refusal/schema/policy checks only; no Torch, images, inference or GPU calls."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from driver import approve,finite,read,digest,CORE_SHA,OLD_GATE_SHA
from writer_policy import sanitize_result,check_sidecar
from summarize import rank

HERE=Path(__file__).resolve().parent
checks=[]
def reject(name,call):
    try:call()
    except (ValueError,AssertionError,KeyError,TypeError):checks.append(dict(name=name,rejected=True))
    else:raise AssertionError('Expected refusal: '+name)
def pass_(name):checks.append(dict(name=name,pass_=True))

gate=read(HERE/'gate_scope.json');screen=read(HERE/'screen_scope.json');protocol=read(HERE/'runtime_protocol.json')
assert gate['status']==screen['status']=='PREPARED_NOT_FROZEN' and len(gate['jobs'])==4 and len(screen['jobs'])==32
assert not (HERE/'gate_runs').exists() and not (HERE/'screen_runs').exists()
reject('unfrozen_gate_rejected_before_missing_approval_or_Torch',lambda:approve('gate',None,None))
reject('unfrozen_screen_rejected_before_missing_approval_or_Torch',lambda:approve('screen',None,None))
sys.path.insert(0,str(HERE/'scientific_snapshot'))
import worker
rows=[]
for entry in screen['jobs']:
    job=read(HERE/entry['job']);assert digest(HERE/entry['job'])==entry['job_sha256']
    worker.validate_job({k:v for k,v in job.items() if k!='runtime_protocol_sha256'},protocol,require_frozen=False)
    assert job['runtime_protocol_sha256']==digest(HERE/'runtime_protocol.json')
    for name,sha in job['adapter_source_hashes'].items():assert digest(HERE/'scientific_snapshot'/name)==sha
    rows.append(dict(id=job['id'],candidate=job['tuning_candidate'],distribution=job['distribution'],attack=job['attack'],metrics=dict(accuracy=.5,aeod=.1,aspd=.2)))
pass_('32_new_job_bytes_original_validator_recipe_and_five_source_hashes')
for key,value in [('rounds',69),('seed',91002),('client_alpha',1.0),('learning_rate',.1),('ad2_calibration_enabled',True),('celeba_evaluation_split','test')]:
    bad=copy.deepcopy(job);bad['config'][key]=value
    reject('original_screen_validator_refuses_'+key,lambda bad=bad:worker.validate_job({k:v for k,v in bad.items() if k!='runtime_protocol_sha256'},protocol,require_frozen=False))

for distribution in ('IID','non-IID'):
    for horizon in (3,70):
        example=copy.deepcopy(job);example['distribution']=distribution;example['attack']='S-DFA';example['config']['rounds']=horizon
        audits=[dict(client_id=i,is_malicious=i<4,samples=100+i,attack_types=['fflip','foe'] if i<4 else [],fflip_mode='all_unprivileged',fflip_overwrite_ratio=1.0,fflip_requires_full_flip=False,label_changed_count=0,fflip_label_corr_after=float('nan') if i<4 else .2) for i in range(20)]
        value=dict(config=copy.deepcopy(example['config']),method=example['method'],attack='S-DFA',attack_audit=audits,metrics=dict(accuracy=.5,aeod=.1,aspd=.2),finite_payload=[None,True,-.3,'x'])
        clean,changes=sanitize_result(value,example,'a'*64,'a'*64)
        assert len(changes)==4 and all(clean['attack_audit'][i]['fflip_label_corr_after'] is None for i in range(4))
        assert clean['metrics']==value['metrics'] and clean['finite_payload']==value['finite_payload'] and all(math.isnan(value['attack_audit'][i]['fflip_label_corr_after']) for i in range(4))
        sidecar=dict(status='EXPLICIT_UNDEFINED_DIAGNOSTIC_ONLY',job_sha256='a'*64,original_gate_sha256=OLD_GATE_SHA,original_core_sha256=CORE_SHA,undefined_values=changes)
        check_sidecar(clean,sidecar,example,'a'*64,OLD_GATE_SHA,CORE_SHA)
        pass_(f'{distribution}_{horizon}_exact_four_NaN_null_bits_cause_roundtrip_input_unchanged')
        for name,key,badvalue in [('main_metric','metrics',dict(accuracy=float('nan'))),('unknown_field','unknown',float('nan')),('weight','weights',[float('inf')])]:
            bad=copy.deepcopy(value);bad[key]=badvalue
            reject(f'{distribution}_{horizon}_{name}_nonfinite_refused',lambda bad=bad:sanitize_result(bad,example,'a'*64,'a'*64))
        malformed=copy.deepcopy(sidecar);malformed['undefined_values'][0]['path']=['metrics','accuracy']
        reject(f'{distribution}_{horizon}_unknown_sidecar_path_refused',lambda:check_sidecar(clean,malformed,example,'a'*64,OLD_GATE_SHA,CORE_SHA))
reject('Benign_unknown_nonfinite_refused',lambda:finite({'attack':'Benign','extra':float('nan')}))
finite({'attack':'Benign','metrics':[.5,0.,.1],'config_none':None});pass_('Benign_finite_structure_preserved')
ranking=rank(rows,protocol['candidates']);assert ranking['selected_recipe']==min(c['id'] for c in protocol['candidates']) and len(ranking['all_candidates'])==8 and len(ranking['three_metric_Pareto'])==8
pass_('four_condition_mean_exact_tie_lexicographic_all_candidates_no_false_Pareto')
reject('incomplete31_terminals_refused',lambda:rank(rows[:-1],protocol['candidates']))
reject('duplicate_or_foreign_condition_refused',lambda:rank(rows[:-1]+[rows[0]],protocol['candidates']))

def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
old=functions(HERE/'original_writer_policy.py');new=functions(HERE/'writer_policy.py')
assert old['check_sidecar']==new['check_sidecar'] and old['require']==new['require']
oldnode=next(n for n in ast.parse((HERE/'original_writer_policy.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sanitize_result')
newnode=next(n for n in ast.parse((HERE/'writer_policy.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sanitize_result')
assert [ast.dump(n,include_attributes=False) for n in oldnode.body[4:]]==[ast.dump(n,include_attributes=False) for n in newnode.body[4:]]
pass_('writer_recursive_visitor_bit_reason_and_sidecar_body_unchanged_only_explicit_job_domain_header_extended')
assert 'torch' not in sys.modules
result=dict(status='PASS_PREPARED_SCHEMA_AND_REFUSAL_ONLY_NO_REAL_EXECUTION',checks=checks,check_count=len(checks),GPU_calls=0,Torch_imported=False,image_loads=0,new_training=0,new_inference=0)
with (HERE/'selfcheck.json').open('x',encoding='utf-8') as out:out.write(json.dumps(result,indent=2)+'\n')
print(result['status'],len(checks))
