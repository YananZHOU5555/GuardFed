"""Tiny metadata checks only: no Torch imports, model files, training or real selection."""
from pathlib import Path
import ast,copy,json,math,sys
sys.dont_write_bytecode=True
import prepare as p

def source_functions(text):
    return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}

def run():
    protocol,manifest,pins,score=p.inputs();derived,diff=p.render()
    old=source_functions((p.OLD/p.BRIDGE/'worker.py').read_text(encoding='utf8'));new=source_functions(derived['worker.py'])
    unchanged=set(old)-{'validate_job','run'}
    assert all(old[n]==new[n] for n in unchanged)
    changed=[n for n in old if old[n]!=new[n]];assert set(changed)=={'validate_job','run'}
    ast.parse(derived['accept_result.py'])
    ns=dict(math=math,METHODS=set(p.METHODS),OBJECTIVES={'original_unweighted_ce','shared_reweighted_ce'},
        ROOT_REFERENCES={'same_point_unweighted_gradient','frozen_root_localadam_delta'},
        COMPONENTS=p.read(p.OLD/'jobs'/manifest['jobs'][0]['job'])['component_hashes'],
        LOCAL_SOURCES={'worker.py','accept_result.py','protocol.json'})
    for n in ('validate_adapter','validate_job'):exec(compile(new[n],'<metadata-function>','exec'),ns)
    fixture=copy.deepcopy(protocol);fixture.update(status='PREPARED_NOT_FROZEN',attacks=list(p.ATTACKS),component_hashes=ns['COMPONENTS'])
    rejected=0;valid=0
    for method in p.METHODS:
        c=next(c for c in protocol['candidates'] if c['method']==method)
        jobs=p.grid(c,fixture,{k:'0'*64 for k in ns['LOCAL_SOURCES']})
        for j in jobs:ns['validate_job'](j,fixture,False);valid+=1
        for transform in [lambda j:j['config'].update(seed=91011),lambda j:j['config'].update(client_alpha=9),
          lambda j:j.update(attack='FOE'),lambda j:j['adapter'].update(server_eta=99),
          lambda j:j.update(evidence_stage='validation_screen'),lambda j:j['config'].update(seed=91001)]:
            j=copy.deepcopy(next(j for j in jobs if j['attack']=='Benign'));transform(j)
            try:ns['validate_job'](j,fixture,False)
            except ValueError:rejected+=1
            else:raise AssertionError('mutated identity accepted')
        try:ns['validate_job'](jobs[0],fixture)
        except ValueError:rejected+=1
        else:raise AssertionError('PREPARED executed')
    # Equal-score synthetic records exercise complete64 and lexical tie only.
    records=[]
    for e in manifest['jobs']:
        j=p.read(p.OLD/'jobs'/e['job']);records.append(dict(id=e['id'],method=j['method'],candidate=j['tuning_candidate'],
            distribution=j['distribution'],attack=j['attack'],seed=91001,rounds=70,alpha=j['config']['client_alpha'],evaluation_split='valid',
            job_sha256=e['job_sha256'],strict_pass=True,offserver_verified=True,source_hashes=j['source_hashes'],
            local_hashes=j['local_hashes'],component_hashes=j['component_hashes'],result_sha256='0'*64,model_sha256='0'*64,
            acceptance_sha256='0'*64,output='FIXTURE_NOT_REAL',metrics={'accuracy':.5,'aeod':0.,'aspd':0.}))
    winners,_=p.selected(records,protocol,manifest,score)
    assert winners=={m:min(c['id'] for c in protocol['candidates'] if c['method']==m) for m in p.METHODS}
    for bad in (records[:-1],records[:-1]+[records[0]]):
        try:p.selected(bad,protocol,manifest,score)
        except AssertionError:rejected+=1
        else:raise AssertionError('partial/duplicate64 accepted')
    assert 'torch' not in sys.modules
    return dict(status='SOURCE_METADATA_ONLY_PASS',unchanged_whole_worker_functions=sorted(unchanged),
        changed_worker_functions=changed,valid_new_job_fixtures=valid,refusals=rejected,
        complete64_lexical_tie_fixture=True,actual_recipe_selected=False,real_image_gates_run=0,
        summary_run=False,torch_imported=False,execution_authorized=False),diff

if __name__=='__main__':
    result,diff=run()
    for name,value in [('SELF_CHECK.json',json.dumps(result,indent=2)+'\n'),('SOURCE_DIFF.patch',diff)]:
        with (p.HERE/name).open('x',encoding='utf8') as f:f.write(value)
    print(json.dumps(result))
