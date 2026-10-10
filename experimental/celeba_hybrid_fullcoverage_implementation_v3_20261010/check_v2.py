"""Local metadata only: no bind_stage execution, job creation or scientific imports."""
import ast,copy,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/'tmp/celeba_hybrid_fullcoverage_implementation_20261010'
SCREEN=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
SUMMARY=ROOT/'tmp/celeba_hybrid32_final_collection_20261010/SUMMARY32.json'
ADOPTION=SCREEN/'accepted_delta_after27_20261010/ROOT32_SUMMARY_ADOPTION.json'
ROOT_SHA='6fcbdc41c7af01815e15995e0cb3688672404bf3b96844dc8d9bffa21028587f'
SUMMARY_SHA='46b5f8fdca9536166ed868e50d4c7bc2578f8a1100ffea97878cf044c95748ae'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def function(p,name):
    s=p.read_text();return ast.get_source_segment(s,next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))
def check():
    if sys.flags.optimize:raise RuntimeError('Optimized Python refused')
    assert sha(PARENT/'FILES_SHA256.json')=='f92342afff7b271073108c637d2c599f2a2717f49dad7eb08066198bccdb128e'
    assert sha(ROOT/'tmp/celeba_hybrid_fullcoverage_source_review_20261010/FILES_SHA256.json')=='0db75b3c3bbd5f282f1e6fed0b8ccc4ed1199bcf8b97989d66f18a672b81e1ee'
    oldseal=read(PARENT/'FILES_SHA256.json');changed=[]
    for n,r in oldseal['files'].items():
        assert sha(PARENT/n)==r['sha256'] and (PARENT/n).stat().st_size==r['bytes']
        if sha(HERE/n)!=r['sha256']:changed.append(n)
    assert sorted(changed)==['bind_stage.py','metadata_contract.py']
    for n in ('original_rank','coverage_layout'):assert function(HERE/'metadata_contract.py',n)==function(PARENT/'metadata_contract.py',n)
    before=ast.parse((PARENT/'bind_stage.py').read_text());after=ast.parse((HERE/'bind_stage.py').read_text());calls=[n for n in ast.walk(after) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='check_complete_summary'];assert len(calls)==1
    assert [k.arg for k in calls[0].keywords]==['summary_path','root_path','root_sha256'];calls[0].keywords=[]
    assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    oldtail=(PARENT/'metadata_contract.py').read_text().split("    require(summary['seed_n']",1)[1].split("    return ranked['selected_recipe']",1)[0]
    newtail=(HERE/'metadata_contract.py').read_text().split("    require(summary['seed_n']",1)[1].split('    if adopted is not None:',1)[0]
    assert oldtail==newtail
    assert sha(SUMMARY)==SUMMARY_SHA and sha(ADOPTION)==ROOT_SHA
    module=load(HERE/'metadata_contract.py','hybrid_v2_metadata');old=load(PARENT/'metadata_contract.py','hybrid_old_metadata')
    summary=read(SUMMARY);protocol=read(SCREEN/'runtime_protocol.json');scope=read(SCREEN/'screen_scope.json')
    winner=module.check_complete_summary(summary,protocol,scope,summary_path=SUMMARY,root_path=ADOPTION,root_sha256=ROOT_SHA)
    assert winner==read(ADOPTION)['selected_recipe']=='CosineFairness_lam20.0_tau0.1_lr0.001'
    rejects=[]
    def reject(label,fn):
        try:fn()
        except (ValueError,KeyError,TypeError) as e:rejects.append({'case':label,'error':type(e).__name__+': '+str(e)})
        else:raise AssertionError('Accepted invalid '+label)
    reject('unchanged_parent_rejects_actual_status',lambda:old.check_complete_summary(summary,protocol,scope))
    reject('actual_status_without_root',lambda:module.check_complete_summary(summary,protocol,scope))
    reject('wrong_external_root_sha',lambda:module.check_complete_summary(summary,protocol,scope,summary_path=SUMMARY,root_path=ADOPTION,root_sha256='0'*64))
    with tempfile.TemporaryDirectory(prefix='TEST_ONLY_',dir=HERE) as directory:
        f=Path(directory)/'TEST_ONLY_MUTATED_ROOT.json'
        for key,value in [('status','PENDING'),('accepted_total',31),('all32_offserver_verified',False),('final_test',True),('summary_sha256','0'*64),('source_seal_sha256','0'*64),('selected_recipe','wrong')]:
            root=read(ADOPTION);root[key]=value;f.write_text(json.dumps(root),encoding='utf8')
            reject('root_'+key,lambda:module.check_complete_summary(summary,protocol,scope,summary_path=SUMMARY,root_path=f,root_sha256=sha(f)))
        altered=copy.deepcopy(summary);altered['selected_recipe']='wrong'
        reject('supplied_summary_not_original_file',lambda:module.check_complete_summary(altered,protocol,scope,summary_path=SUMMARY,root_path=ADOPTION,root_sha256=ROOT_SHA))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert sha(SUMMARY)==SUMMARY_SHA and sha(ADOPTION)==ROOT_SHA
    return dict(status='PASS_ACTUAL_ROOT_BOUND_METADATA_SMOKE_ONLY',parent_members=19,unchanged_members=17,changed_members=changed,original_rank_and_coverage_source_exact=True,original_selection_loop_source_exact=True,binder_AST_only_three_keyword_arguments=True,summary_path=str(SUMMARY),summary_sha256=SUMMARY_SHA,root_path=str(ADOPTION),root_sha256=ROOT_SHA,selected_recipe=winner,refusals=rejects,refusal_count=len(rejects),torch_imported=False,numpy_imported=False,jobs_generated=0,gates_executed=0,training_started=False,binding_executed=False,execution_authorized=False)
if __name__=='__main__':
    result=check();out=HERE/'V2_CHECK.json'
    if out.exists():raise RuntimeError('Preserve existing check output')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))
