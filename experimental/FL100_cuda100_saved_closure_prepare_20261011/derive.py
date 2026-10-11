"""Source-only exact scope derivation; never invoke generated transport or audit."""
from pathlib import Path
import ast,base64,hashlib,json
R=Path.cwd();H=Path(__file__).resolve().parent;P=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011';F=R/'tmp/FL100_uniform_cuda_formal_adapter_prepare_20261011'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def put(n,v):(H/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
old=read(P/'SELECTED3_INPUTS.json');cohort=read(F/'EXACT100.json');pins=read(F/'SOURCE_PINS.json');ids=[r['id'] for r in cohort['records']];can=old['ids'];remaining=[i for i in ids if i not in can];assert len(ids)==len(set(ids))==100 and len(remaining)==97
rows=[]
for route in cohort['records']:
    pkg=pins['packages'][route['identity_source']];p=R/pkg['manifest']['path'];assert p.resolve().drive.upper()=='E:' and sha(p)==pkg['manifest']['sha256'];row=read(p)['records'][route['identity_index']];assert row['id']==route['id'];rows.append(dict(route=route,original_row=row,package=pkg))
for row in old['rows']:assert next(r for r in rows if r['route']['id']==row['route']['id'])==row
frozen=dict(old,status='FROZEN_FL100_SCOPE_NOT_CUDA_COMPLETION',ids=ids,rows=rows,canary_ids=can,remaining_ids=remaining)
put('SELECTED100_INPUTS.json',frozen)
prior={}
for role,name in [('root','root_adoption/ROOT_SCIENTIFIC_ADOPTION.json'),('transport','OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json'),('audit','actual/SAVED_OUTPUTS_AUDIT.json')]:
    p=P/name;prior[role]={'path':str(p.resolve()),'sha256':sha(p)}
put('PRIOR3_PINS.json',prior)
for n in ['original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt','saved_fit_binding.py']:(H/n).write_bytes((P/n).read_bytes())
sourcepins=read(P/'SOURCE_PINS.json');sourcepins['frozen_selection_sha256']=sha(H/'SELECTED100_INPUTS.json');sourcepins['parent_actual_fixed3_source_seal']={'path':str(P/'FILES_SHA256.json'),'sha256':sha(P/'FILES_SHA256.json')};put('SOURCE_PINS.json',sourcepins)
def derive(name,edits):
    raw=(P/name).read_bytes();s=raw;ops=[]
    for before,after in edits:
        b=before.encode();a=after.encode();count=s.count(b);assert count>0,(name,before[:100]);s=s.replace(b,a);ops.append(dict(before_b64=base64.b64encode(b).decode(),after_b64=base64.b64encode(a).decode(),count=count))
    restored=s
    for op in reversed(ops):
        a=base64.b64decode(op['after_b64']);b=base64.b64decode(op['before_b64']);assert restored.count(a)==op['count'];restored=restored.replace(a,b)
    assert restored==raw
    (H/name).write_bytes(s);put(name+'.inverse.json',dict(parent=str((P/name).relative_to(R)),parent_sha256=sha(P/name),derived_sha256=sha(H/name),operations=ops,full_raw_inverse_exact=True))
canrepr=repr(can);idrepr=repr(ids)
completion="remaining=read(member('base/remaining/COMPLETE.json'));need(sha(x/'base/remaining/COMPLETE.json')==inp['remaining_complete_sha256'] and remaining['ids']==frozen['remaining_ids'] and remaining['server_saved_accepted'] is True and remaining['cohort']==auth['exact_cohort'] and remaining['root_offserver_adopted'] is False,'Incomplete remaining97')\n    need(set(complete['ids']).isdisjoint(remaining['ids']) and set(complete['ids']+remaining['ids'])==set(ids),'Wrong3+97 partition')\n    "
derive('metadata_gate.py',[
    ("x=Path(extract);base=", "x=extract;base="),
    ("auth['canary_ids']==ids and len(auth['ordered_ids'])==100", "auth['canary_ids']==frozen['canary_ids'] and auth['ordered_ids']==ids and len(ids)==100"),
    ("complete['ids']==ids", "complete['ids']==frozen['canary_ids']"),
    ("need(set(inp['accepted_sha256'])==set(ids)", completion+"need(set(inp['accepted_sha256'])==set(ids)")])
science=(P/'audit_saved.py').read_bytes();loop= science[science.index(b'        for identity in IDS:'):science.index(b'        for rel,pin in members.items():')]
derive('audit_saved.py',[
    ('IDS='+canrepr,'IDS='+idrepr),('SELECTED3_INPUTS.json','SELECTED100_INPUTS.json'),
    ("    return inp,t,extract,members,volume", "    from prior3_overlay import join_prior\n    extract,members=join_prior(inp,extract,members)\n    return inp,t,extract,members,volume"),
    ("FL100_FIXED3_CUDA_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED", "FL100_ALL100_CUDA_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED"),
    ('metric_values_checked=27,integer_base_counts_checked=72,prediction_rules_checked=9','metric_values_checked=900,integer_base_counts_checked=2400,prediction_rules_checked=300'),
    ("scope='FL fixed3 only; original Linux whole plus SHA transport and zero-fit saved-output consistency. No CNN/fit/test/FL100-completion claim.'", "scope='Exact FL100=adopted3+new97 once each; original Linux whole plus SHA delta transport and zero-fit saved-output consistency. Not root adopted; no CNN/fit/test.',prior3_root_sha256=read(HERE/'PRIOR3_PINS.json')['root']['sha256'],new_records=97,reused_records=3,native_max_abs_difference=max(r['native_comparison']['max_abs_difference'] for r in rows)"),
    ('records=3,sha256','records=100,sha256')])
assert (H/'audit_saved.py').read_bytes().count(loop)==1
remote_remaining="remaining=read(B/'remaining/COMPLETE.json');assert sha(B/'remaining/COMPLETE.json')==inp['remaining_complete_sha256']\nassert remaining['ids']==frozen['remaining_ids'] and remaining['server_saved_accepted'] is True and remaining['cohort']==auth['exact_cohort'] and remaining['root_offserver_adopted'] is False\nassert not (B/'remaining/QUEUE_FAILURE.json').exists()\n"
filter_old="\n# Exact prior3 plus original metadata remain in their adopted F archive; download only97 arrays.\nold_ids=set(frozen['canary_ids'])\nreused={n:p for n,p in files.items() if n=='source/metadata.npz' or any(n.startswith('base/tasks/'+i+'/') or n.startswith('base/runs/'+i+'/') for i in old_ids)}\nreused_pins={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in reused.items()}\nassert set(old_ids)==set(inp['prior3_accepted_sha256'])\nfor i in old_ids:assert inp['accepted_sha256'][i]==inp['prior3_accepted_sha256'][i]\nfiles={n:p for n,p in files.items() if n not in reused}\nassert len([n for n in files if n.endswith('/validation_predictions.npz')])==97\n"
derive('transport.py',[
    ('tmp/FL100_cuda3_saved_closure_prepare_20261011','tmp/FL100_cuda100_saved_closure_prepare_20261011'),('SELECTED3_INPUTS.json','SELECTED100_INPUTS.json'),
    ("['deployment_sha256','cohort_sha256','canary_complete_sha256']", "['deployment_sha256','cohort_sha256','canary_complete_sha256','remaining_complete_sha256']"),
    ("canary3_attempt001", "all100_delta97_attempt001"),('check_bulk_storage(50*1024**2)','check_bulk_storage(250*1024**2)'),
    ("auth['canary_ids']==ids", "auth['canary_ids']==frozen['canary_ids']"),
    ("complete['ids']==ids", "complete['ids']==frozen['canary_ids']"),
    ("assert not (B/'canary/QUEUE_FAILURE.json').exists()", "assert not (B/'canary/QUEUE_FAILURE.json').exists()\n"+remote_remaining),
    ("and 'canary' in argv", "and any(phase in argv for phase in ('canary','remaining'))"),
    ('FL100_fixed3_saved_audit_attempt001','FL100_all100_delta97_saved_audit_attempt001'),('FL100_FIXED3_CUDA_SELECTION_NOT_ROOT_ADOPTED','FL100_ALL100_CUDA_SELECTION_NOT_ROOT_ADOPTED'),
    ("'worker.py','canary/COMPLETE.json'", "'worker.py','canary/COMPLETE.json','remaining/COMPLETE.json'"),
    ("members={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}",filter_old+"members={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}"),
    ("assert all(sha(p)==members[n]['sha256'] for n,p in files.items())", "assert all(sha(p)==members[n]['sha256'] for n,p in files.items())\nassert all(sha(p)==reused_pins[n]['sha256'] for n,p in reused.items())"),
    ("member_index_sha256=sha(index),ids=ids", "member_index_sha256=sha(index),ids=ids,new_arrays=97,reused_arrays=3,reused_member_pins=reused_pins"),
    ('OLD800_FIXED3_ATTEMPT1','FL100_DELTA97'),('timeout=90)','timeout=300)')])
root=read(Path(prior['root']['path']));template=read(P/'ROOT_INPUTS.template.json');template.update(exact_ids=ids,accepted_sha256={i:None for i in ids},remaining_complete_sha256=None,prior3_accepted_sha256={i:root['records'][i]['task_acceptance_sha256'] for i in can},prior3_root=prior['root']);put('ROOT_INPUTS.template.json',template)
put('DERIVATION_CHECK.json',dict(status='SOURCE_ONLY_RAW_INVERSE_PASS',full_inverse_files=['metadata_gate.py','audit_saved.py','transport.py'],per_record_scientific_loop_raw_sha256=hashlib.sha256(loop).hexdigest(),per_record_scientific_loop_raw_exact=True,ids=100,reused=3,new=97,original_three_rows_exact=True,all_seven_packages=sorted({r['route']['identity_source'] for r in rows}),future_completion=None,new_scientific_execution=False))
print(json.dumps({'prepared':100,'new_delta':97,'reused':3,'science_raw_exact':True}))
