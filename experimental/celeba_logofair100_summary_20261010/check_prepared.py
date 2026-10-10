"""Metadata/source and tiny synthetic arithmetic only; never execute the100 review."""
import ast, copy, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import summary100 as b


def main():
    b.need(not (b.HERE / 'SELF_CHECK.json').exists(), 'No overwrite/retry')
    metadata = b.sources(); inventory = b.read(b.SOURCE / 'CACHE_IDENTITIES100.json')['references']
    candidate = b.read(b.ADOPTION)['selected_recipe']; jobs = []; reused = []
    for r in inventory:
        entry = dict(id=candidate['id']+'_'+r['cell_id'], cell_id=r['cell_id'])
        (reused if r['seed']==91001 and r['attack'] in ('Benign','S-DFA') else jobs).append(entry)
    manifest = dict(jobs=jobs, reused_jobs=reused, candidate=candidate, scientific_stage='fixed_recipe_validation_postprocessing100', new_CNN=0, final_test=False)
    index = dict(status='LOCAL_STRICT96_PLUS4_ROOT_REVIEW_PENDING', records=copy.deepcopy(jobs), reused_jobs=copy.deepcopy(reused))
    b.grid(manifest, index, inventory, candidate); refused = []
    def reject(name, call):
        try: call()
        except ValueError: refused.append(name)
        else: raise AssertionError('Must refuse: '+name)
    def mutated(name, change):
        m, i = copy.deepcopy(manifest), copy.deepcopy(index); change(m, i)
        reject(name, lambda: b.grid(m, i, inventory, candidate))
    mutated('partial95new', lambda m,i:m['jobs'].pop())
    mutated('missing_old4', lambda m,i:m['reused_jobs'].pop())
    mutated('duplicate_newID', lambda m,i:m['jobs'].__setitem__(1,m['jobs'][0]))
    mutated('foreign_ID', lambda m,i:m['jobs'][0].__setitem__('id','foreign'))
    mutated('cell_alias_drift', lambda m,i:m['jobs'][0].__setitem__('cell_id','wrong'))
    mutated('partial_index', lambda m,i:i['records'].pop())
    mutated('reordered_index', lambda m,i:i['records'].reverse())
    mutated('wrong_recipe', lambda m,i:m.__setitem__('candidate',dict(candidate,id='LoGoFair-DP_06')))
    mutated('final_test_claim', lambda m,i:m.__setitem__('final_test',True))
    reject('wrong_external_SHA', lambda:b.pinned(b.HERE/'INPUT_PINS.json','0'*64))
    adopt=b.read(b.ADOPTION); strict=b.read(adopt['strict_index_path'])
    actual=next(r for r in strict['records'] if r['id']=='LoGoFair-DP_07_FedAvg_IID_Benign_seed91001')
    result=b.pinned(actual['result'],actual['result_sha256']); job=result['job']; row=next(r for r in inventory if r['id']==job['baseline_id'])
    b.record_identity(result,job,row,candidate)
    for name,key,value in [('checkpoint_drift','checkpoint_sha256','0'*64),('cache_drift','accepted_margin_cache_sha256','0'*64),('wrong_fitseed','fit_seed',1720)]:
        altered=copy.deepcopy(result);altered[key]=value
        reject(name,lambda altered=altered:b.record_identity(altered,job,row,candidate))
    altered=copy.deepcopy(result);altered['history'].pop()
    reject('incomplete29postrounds',lambda:b.record_identity(altered,job,row,candidate))
    changed_job=copy.deepcopy(job);changed_job['settings']['post_lr']*=2
    reject('changed_fixed_settings',lambda:b.record_identity(result,changed_job,row,candidate))
    reject('saved_metric_difference_over_original_bound',lambda:b.saved_metric_error(dict(accuracy=.5,aeod=.01,aspd=.02),dict(accuracy=.5+2e-12,aeod=.01,aspd=.02)))
    reject('nonfinite_saved_metric',lambda:b.saved_metric_error(dict(accuracy=.5,aeod=math.nan,aspd=.02),dict(accuracy=.5,aeod=.01,aspd=.02)))
    old=(b.SCREEN/'snapshot/logofair_bridge_20261010/bridge.py').read_text(encoding='utf8'); new=metadata.bridge_source()
    oldtree,newtree=ast.parse(old),ast.parse(new)
    functions=b.read(b.SOURCE/'SOURCE_REUSE.json')['science_functions_source_exact']
    for name in functions:
        a=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        c=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b.need(ast.get_source_segment(old,a)==ast.get_source_segment(new,c),'Original science function drift: '+name)
    synthetic=[]
    for scene,(d,a) in enumerate(b.SCENES):
        for j,s in enumerate(b.SEEDS):synthetic.append(dict(id=f'fixture_{d}_{a}_{s}',distribution=d,attack=a,seed=s,metrics=dict(accuracy=.6+j*.001+scene*.002,aeod=.02+j*.0001+scene*.0002,aspd=.1-j*.0002+scene*.0001),constant_prediction=0 if scene==0 else None))
    summary=b.describe(synthetic); checks=0
    for panel,seeds in b.PANELS.items():
        actual=summary['panels'][panel];b.need(actual['cross_scene']['n']==len(seeds) and all(r['n']==len(seeds) for r in actual['per_scene']),'Seed denominator changed')
        for metric,outkey,scale in [('accuracy','accuracy_pct',100),('aeod','aeod',1),('aspd','aspd',1)]:
            seedvalues=[math.fsum(r['metrics'][metric]*scale for r in synthetic if r['seed']==s)/10 for s in seeds]
            mu=math.fsum(seedvalues)/len(seeds);sd=math.sqrt(math.fsum((x-mu)**2 for x in seedvalues)/(len(seeds)-1))
            b.need(abs(mu-actual['cross_scene'][outkey]['mean'])<=1e-12 and abs(sd-actual['cross_scene'][outkey]['sample_sd_ddof1'])<=1e-12,'Seed-first mean/sampleSD regression');checks+=2
    b.need(len(summary['constant_prediction_ids'])==10,'Constant-negative fixtures were dropped')
    reject('partial99_summary',lambda:b.describe(synthetic[:-1]))
    reject('duplicate_summary_cell',lambda:b.describe(synthetic[:-1]+[synthetic[0]]))
    b.need('torch' not in sys.modules and 'netcal' not in sys.modules,'No real runtime/checker/fit import in source test')
    proof=dict(status='SOURCE_ONLY_METADATA_AND_SYNTHETIC_CHECKS_PASS_NOT_EXECUTED',metadata_refusals=len(refused),refused=refused,
               positive_grid_fixture='96+4 exact actual inventory identities; no actual100 results',original_accepted32_metadata_fixture=True,
               original_science_functions_source_exact=functions,synthetic_seed_first_mean_SD_checks=checks,constant_negative_fixture_retained=True,
               actual100_review_executed=False,actual100_statistics_generated=False,real_arrays_or_models_read=False,new_fits=0,new_CNN=0,new_bulk_files=0)
    (b.HERE/'SELF_CHECK.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=proof['status'],refusals=len(refused),synthetic_checks=checks)))


if __name__=='__main__':
    try: main()
    except BaseException as e:
        import traceback
        p=b.HERE/'SELF_CHECK_FAILURE.json'
        if not p.exists():p.write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),indent=2)+'\n',encoding='utf8')
        raise
