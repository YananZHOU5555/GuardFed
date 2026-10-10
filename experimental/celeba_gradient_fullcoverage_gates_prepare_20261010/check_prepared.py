"""Pure source/metadata checks, deliberately never binding candidates or importing Torch."""
import ast,copy,json,sys
from metadata import HERE,OLD,H,coverage_source,jobs_for,validate_jobs
from adapter import render_gate_functions,check_attack_assignment


def functions(text):
    return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}


def run():
    coverage=coverage_source();protocol,_,_,_=coverage.inputs();derived,_=coverage.render()
    original=functions((coverage.OLD/coverage.BRIDGE/'worker.py').read_text(encoding='utf8'))
    actual=functions(derived['worker.py']);science=set(original)-{'validate_job','run'}
    assert len(science)==14 and all(original[n]==actual[n] for n in science)
    assert (OLD/'snapshot/gradient_bridge_20261009/worker.py').read_bytes()==(coverage.OLD/coverage.BRIDGE/'worker.py').read_bytes()
    gate,diff=render_gate_functions();old=functions((OLD/'gate.py').read_text(encoding='utf8'))
    # Inside the gate, numerical gradient/sign/aggregation oracles stay source-identical;
    # only the expected root-call predicate in real_aggregate changes.
    def nested(source):return {n.name:ast.get_source_segment(source,n) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)}
    oldn,newn=nested(old['run_one']),nested(gate['run_one'])
    exact=['retained_bundle','forbidden','root_adam','real_gradient','real_attack','progress']
    assert all(oldn[k]==newn[k] for k in exact)
    assert oldn['real_aggregate'].replace("int(job['attack'] == 'S-DFA')","int(job['attack'] in {'FedSA', 'S-DFA', 'Sp-DFA'})")==newn['real_aggregate']
    candidates=[next(c for c in protocol['candidates'] if c['method']==m) for m in coverage.METHODS]
    jobs=[j for c in candidates for j in jobs_for(c,protocol)];validate_jobs(jobs,candidates,protocol)
    rejected=0
    mutations=[lambda j:j[0]['config'].update(seed=91001),lambda j:j[0]['config'].update(rounds=70),
        lambda j:j[0]['config'].update(client_alpha=5000),lambda j:j[0].update(attack='S-DFA'),
        lambda j:j[0].update(implementation='other'),lambda j:j[0]['adapter'].update(server_eta=99),
        lambda j:j[0].update(source_hashes={}),lambda j:j.append(copy.deepcopy(j[0])),lambda j:j.pop()]
    for change in mutations:
        changed=copy.deepcopy(jobs);change(changed)
        try:validate_jobs(changed,candidates,protocol)
        except AssertionError:rejected+=1
        else:raise AssertionError('Bad metadata passed')
    for attack in ('Benign','F Flip','FedSA','Sp-DFA'):
        audit=[]
        for cid in range(20):
            types=[] if cid>=4 or attack=='Benign' else {'F Flip':['fflip'],'FedSA':['foe'],'Sp-DFA':['fflip'] if cid<2 else ['foe']}[attack]
            audit.append(dict(client_id=cid,attack_types=types,label_changed_count=0,foe_mode='fedsa',foe_impl='gradient sign-conjugacy'))
        check_attack_assignment(attack,audit)
        for field,value in [('client_id',99),('label_changed_count',1),('attack_types',['unknown'])]:
            bad=copy.deepcopy(audit);bad[0][field]=value
            try:check_attack_assignment(attack,bad)
            except AssertionError:rejected+=1
            else:raise AssertionError('Bad attack assignment passed')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    report=dict(status='SOURCE_AND_METADATA_ONLY_PASS',whole_science_functions_exact=sorted(science),
        gate_nested_functions_exact=exact,aggregation_oracle_only_root_expectation_changed=True,
        fixture_jobs=14,fixture_rounds=42,metadata_refusals=rejected,attack_assignments=4,
        selected_recipe=None,real_jobs_bound=0,real_image_gates_run=0,execution_authorized=False,
        source_identity=dict(original_gate=H((OLD/'gate.py').read_bytes()),screen_worker=H((coverage.OLD/coverage.BRIDGE/'worker.py').read_bytes())))
    return report,diff


if __name__=='__main__':
    report,diff=run()
    for name,text in [('SELF_CHECK.json',json.dumps(report,indent=2)+'\n'),('SOURCE_DIFF.patch',diff)]:
        with (HERE/name).open('x',encoding='utf8') as f:f.write(text)
    print(json.dumps(report))
