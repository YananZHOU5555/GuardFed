"""Metadata-only publisher guards; never invokes plan, stage, Git mutation or experiments."""
from pathlib import Path
import ast,copy,hashlib,json,sys
D=Path(__file__).resolve().parent;R=D.parents[1]
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    text=(D/'publish_increment38.py').read_text('utf8');tree=ast.parse(text)
    names={'closed_guard','reply_guard','delta_guard'};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    import re
    env=dict(sys=sys,re=re,PARENT='8c71a8f76ac69fbbf9aa84d77a6264d294509daf',COUNTS=dict(native=160,three_view=160,FL_new=14,Hybrid=21,baseline_valid=900))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'guard_functions_only','exec'),env)
    reply=read(R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010/ROOT_REVIEW.json')
    hybrid=read(R/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after19_20261010/ROOT_ADOPTION_REVIEW.json')
    # Fixture only: no future FL closure or accepted count is asserted by this preparation.
    fl=dict(status='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS',accepted_before=12,accepted_new=2,accepted_total=14,accepted_job_ids=[f'fixture_{i}' for i in range(14)],accepted_new_ids=['fixture_12','fixture_13'],old_models_repacked=0,root_new_CNN=0,final_test=False,negative_results_preserved=True,planned_new=96,reused_separately=4)
    closed=dict(status='ROOT_CLOSED_INCREMENT38_INPUTS',parent_commit=env['PARENT'],counts=env['COUNTS'].copy(),closure_pins={k:dict(path='fixture',sha256='0'*64) for k in ('C60_reply_root','FL_root','Hybrid_root','state','formal_live','previous_publication')})
    env['closed_guard'](closed);env['reply_guard'](reply);env['delta_guard'](fl,hybrid)
    refused=0
    def reject(fn,obj,key,value):
        nonlocal refused
        changed=copy.deepcopy(obj);changed[key]=value
        try:fn(changed)
        except (AssertionError,KeyError,TypeError):refused+=1
        else:raise AssertionError(('guard did not refuse',key))
    for k,v in [('status','PREPARED'),('parent_commit','wrong'),('counts',{}),('closure_pins',{})]:reject(env['closed_guard'],closed,k,v)
    for k,v in [('status','PREPARED'),('complete_C_scenes',5),('root_C60_table_sha256','wrong'),('prior_C50_reply_root_sha256','wrong'),('changed_spans',9),('author_review_only',False),('manuscript_applied',True),('final_test',True),('whole_rebuttal_complete',True)]:reject(env['reply_guard'],reply,k,v)
    for k,v in [('accepted_before',11),('accepted_new',3),('accepted_total',15),('accepted_job_ids',[]),('accepted_new_ids',[]),('old_models_repacked',1),('root_new_CNN',1),('final_test',True),('negative_results_preserved',False),('planned_new',100),('reused_separately',0)]:reject(lambda f:env['delta_guard'](f,hybrid),fl,k,v)
    for k,v in [('accepted_before',18),('accepted_new',1),('accepted_total',20),('new_inference',1),('selection_performed',True),('scientific_changes',True),('final_test',True),('formal100_started',True)]:reject(lambda h:env['delta_guard'](fl,h),hybrid,k,v)
    old=(R/'tmp/publication_increment37_prepared_20261010/publish_increment37.py').read_text('utf8');marker='    try:\n        for name,source in sources.items():'
    assert text[text.index(marker):]==old[old.index(marker):]
    verify=(D/'verify_increment38.py').read_text('utf8');oldverify=(R/'tmp/publication_increment37_prepared_20261010/verify_increment37.py').read_text('utf8')
    first="    payload = git('cat-file'";last="    result = dict("
    assert verify[verify.index(first):verify.index(last)]==oldverify[oldverify.index(first):oldverify.index(last)]
    for path in (D/'publish_increment38.py',D/'verify_increment38.py'):compile(path.read_text('utf8'),str(path),'exec')
    prep=read(D/'PREPARED_INPUTS.json')
    for rel,digest in prep['reference_pins'].items():assert sha(R/rel)==digest
    result=dict(status='PASS_PUBLICATION38_SOURCE_METADATA_ONLY_NOT_EXECUTED',actual_C60_reply_and_Hybrid_roots_used=True,FL_positive_fixture_not_actual_closure=True,positive_guard_calls=3,refusal_checks=refused,copy_forceadd_attributes_index_failure_body_byteexact37=True,commit_blob_parser_byteexact37=True,actual_plan_executed=False,Git_mutations=False,SSH=False,CNN=False,actual_publication=False,reference_pins_checked=len(prep['reference_pins']))
    (D/'SELF_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(result))
if __name__=='__main__':main()
