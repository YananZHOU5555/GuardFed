"""Pure metadata/AST checks; no scientific imports, model, stage binding or job files."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'celeba_flgmm_screen_20261009_v2_frozen_release'
sys.path.insert(0,str(HERE/'source'))
from worker import validate_job
from prepare_jobs import definitions,make_job
from bind_stage import validate_selection


def main():
    old=json.loads((OLD/'source/protocol.json').read_bytes())
    manifest=json.loads((OLD/'jobs/manifest.json').read_bytes())
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    hashes={name:digest(HERE/'source'/name) for name in ['flgmm_adapter.py','worker.py','prepare_jobs.py','protocol.json','sources/flgmm_pinned.py','sources/flgmm_fedavg.py','sources/flgmm_license.txt']}
    positives=0;refused=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,AssertionError,KeyError):refused.append(name)
        else:raise AssertionError('Expected refusal: '+name)
    for candidate in old['candidates']:
        protocol=dict(copy.deepcopy(old),scope='flgmm_selected_100_valid_only',version='fixture_only_not_materialized',selected_recipe=candidate,candidates=[candidate],
                      seeds=list(range(91001,91011)),attacks=['Benign','F Flip','FedSA','S-DFA','Sp-DFA'])
        jobs=list(definitions(protocol,hashes));assert len(jobs)==96
        observed=set()
        for j in jobs:
            validate_job(j,protocol);positives+=1
            observed.add((j['distribution'],j['attack'],j['config']['seed']))
        reused={(d,a,91001) for d in old['distributions'] for a in old['attacks']}
        assert observed.isdisjoint(reused) and len(observed|reused)==100
        for attack in protocol['attacks']:
            j=make_job(protocol,hashes,'non-IID',attack,91001,'preflight');validate_job(j,protocol);positives+=1
        job=copy.deepcopy(jobs[0])
        for name,change in [('screen_phase',lambda x:x.update(phase='screen')),('wrong_seed',lambda x:x['config'].update(seed=92000)),
                            ('bool_seed',lambda x:x['config'].update(seed=True)),('round69',lambda x:x['config'].update(rounds=69)),
                            ('test_split',lambda x:x['config'].update(celeba_evaluation_split='test')),('wrong_alpha',lambda x:x['config'].update(client_alpha=7.)),
                            ('changed_recipe',lambda x:x['adapter'].update(control_width=99.)),('source_drift',lambda x:x['source_hashes'].update({'src/celeba_data.py':'0'*64})),
                            ('budget_change',lambda x:x['config'].update(local_epochs=2)),('old_stage',lambda x:x.update(evidence_stage='validation_screen'))]:
            bad=copy.deepcopy(job);change(bad);reject(candidate['id']+':'+name,lambda:validate_job(bad,protocol))
        bad=make_job(protocol,hashes,'IID','Benign',91001,'fullcoverage')
        reject(candidate['id']+':duplicate_reused_cell',lambda:validate_job(bad,protocol))
        pending=dict(protocol,status='PREPARED_NOT_FROZEN')
        reject(candidate['id']+':not_frozen',lambda:validate_job(job,pending))
    # Exercise actual old validator with only the declared horizon constant patch.
    source=(OLD/'source/worker.py').read_text();tree=ast.parse(source)
    reference_tree=ast.parse((HERE/'canary_reference.py').read_text())
    function=next(n for n in reference_tree.body if isinstance(n,ast.FunctionDef) and n.name=='horizon_functions')
    ns={'ast':ast,'copy':copy};exec(compile(ast.Module(body=[function],type_ignores=[]),'horizon_fixture','exec'),ns)
    patched=ns['horizon_functions'](source)
    validator=next(n for n in patched.body if n.name=='validate_job')
    scope={'METHOD':'FLGMM-author-code'};exec(compile(ast.Module(body=[validator],type_ignores=[]),'old_validator3','exec'),scope)
    count=0
    for item in manifest['jobs']:
        if item['distribution']!='non-IID':continue
        job=json.loads((OLD/'jobs'/item['job']).read_bytes());job['config']['rounds']=3
        scope['validate_job'](job,old);count+=1
        job['config']['rounds']=70;reject('reference_wrong_horizon:'+item['id'],lambda:scope['validate_job'](job,old))
    # No invented accepted summary passes; acceptance/selection are not run here.
    reject('missing_actual_summary',lambda:validate_selection({}, {}, old, manifest))
    reject('template_cannot_run',lambda:validate_job({},json.loads((HERE/'source/protocol.json').read_bytes())))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    for path in HERE.rglob('*.py'):ast.parse(path.read_text())
    result=dict(status='PASS_PURE_METADATA_AST_ONLY',new_and_gate_positive_cases=positives,
        old_reference_3round_validator_positive_cases=count,refusal_count=len(refused),refusals=refused,
        grid_per_candidate={'new':96,'reused':4,'total':100},selected_recipe=None,formal_jobs_generated=0,
        stage_bound=False,protocol_frozen=False,scientific_imports=0,real_image_canaries=0)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
