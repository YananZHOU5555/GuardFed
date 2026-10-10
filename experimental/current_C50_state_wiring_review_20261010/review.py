"""Bounded read-only local C50 wiring review. Does not execute any updater."""
import ast,datetime,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
TRAIN=R/'docs/server_deployment_20260923/training_20260923'
SOURCE=[R/'tmp'/n for n in ['update_reactivation_state_20261009.py','update_completion_current_20261009.py','update_overview_closure100_root_20261009.py']]
DOCS=[TRAIN/'TRAINING_STATE.json',TRAIN/'RUNNING.md',TRAIN/'REBUTTAL_COMPLETION_20261009.md',R/'docs/返修实验总览.md']
BASE=R/'tmp/celeba_mechanism_valid_C_after50_20261010/execution_candidate'
START=BASE/'ROOT_STARTUP_OBSERVATION.json';DEPLOY=BASE/'deployment_receipt.json'
REPLY=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010'
PROOF=REPLY/'ROOT_REVIEW.json'

def main():
    pins={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in SOURCE+DOCS+[START,DEPLOY,PROOF,REPLY/'FILES_SHA256.json',REPLY/'rebuttal_integrated_20261009.md',REPLY/'manuscript_insertions_integrated_20261009.md']}
    assert sha(START)=='45e955f2031c53430b7dbb914b3e90db771f1c03008d5dca2d50c1dd0cdb133e'
    assert sha(DEPLOY)=='2b65d3b01e424f93c6efed1c7d547a86e0bd64d4d0b3ba09ad98050e70bd83e6'
    assert sha(PROOF)=='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
    state=read(DOCS[0]);main=state['celeba_mechanism_v1'];c=main['C_after50_valid_replay'];latest=state['latest_rebuttal_draft'];proof=read(PROOF);start=read(START)
    assert start['status']=='ROOT_C_AFTER50_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert start['deployment_receipt_sha256']==sha(DEPLOY) and start['original150_not_rerun'] and not start['test_inference']
    assert start['execution_seal_sha256']==read(DEPLOY)['execution_seal_sha256']=='073bfde67b2286f6e29fe5f6e2c1f7580f74465b4b0a4e42c583ab0332aa6595'
    assert start['scientific_offserver_new_accepted']==0
    assert main['three_view_new_models_offserver_verified']==main['three_view_new_models_accepted']==150
    assert main['three_view_counts_by_variant']=={'minus_U':100,'minus_C':50}
    accepted=main['three_view_accepted_ids'];expected=[f'minus_C_non-IID_Benign_seed{i}' for i in range(91001,91007)]
    assert len(accepted)==len(set(accepted))==150 and not set(expected)&set(accepted)
    assert (c['selected_count'],c['prior_three_view_models'],c['offserver_new_accepted'])==(6,150,0)
    assert c['status']=='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING' and c['execution_started'] and c['startup_root_proof_sha256']==sha(START)
    assert c['CPU_affinity']==list(range(112,120)) and c['compute_threads']==8 and c['nice']==10 and c['IO']=='idle' and c['CUDA_visible']==''
    assert c['new_training']==c['new_Full_inference']==0 and c['final_test'] is False
    assert latest['entry']==(REPLY/'rebuttal_integrated_20261009.md').relative_to(R).as_posix()
    assert latest['manuscript_candidate']==(REPLY/'manuscript_insertions_integrated_20261009.md').relative_to(R).as_posix()
    assert latest['root_proof_sha256']==sha(PROOF) and latest['complete_C_scenes']==5
    assert (latest['new_C_scalar_pointer_checks'],latest['new_C_scope_environment_checks'],latest['links_checked'],latest['changed_passages'])==(38,19,44,11)
    assert (proof['original_comments_verbatim'],proof['C_scalar_pointer_checks'],proof['C_scope_fact_pointer_checks'],proof['links_checked'],proof['changed_spans'])==(24,38,19,44,11)
    assert latest['author_review_only'] and not latest['manuscript_applied'] and not latest['whole_rebuttal_complete'] and not proof['final_test']
    assert sha(REPLY/'FILES_SHA256.json')==proof['source_seal_sha256']==latest['source_seal_sha256']
    assert sha(REPLY/'rebuttal_integrated_20261009.md')==proof['rebuttal_sha256'] and sha(REPLY/'manuscript_insertions_integrated_20261009.md')==proof['insertions_sha256']
    source=[p.read_text('utf8') for p in SOURCE]
    for p,text in zip(SOURCE,source):ast.parse(text,filename=str(p))
    branch=source[0].split('C56_replay_dir=',1)[1].split('C30_dir=',1)[0]
    for guard in ["if C56_roots:","C56_proof['status']=='ROOT_C_AFTER50_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'","==(150,6,156)","C56_proof['accepted_new_ids']==expected56","C56_proof['original150_unchanged']","C56_proof['prior150_root_adoption_sha256']","C56_proof['startup_observation_sha256']","==(54,144,18)","len(prior_ids)==150","three_view_new_models_offserver_verified=156"]:assert guard in branch
    assert source[2].index("if state['latest_rebuttal_draft'].get('complete_C_scenes')==5:")>source[2].index("if state['latest_rebuttal_addendum']['complete_C_scenes']==5:")
    current={DOCS[1]:DOCS[1].read_text('utf8').split('# HISTORICAL: Nine-method coverage COMPLETE')[0],DOCS[2]:DOCS[2].read_text('utf8').split('## Current accepted increment',1)[1].split('## Historical accepted increment')[0],DOCS[3]:DOCS[3].read_text('utf8').split('以下为 2026-10-04')[0]}
    selected_lines={}
    for path,text in current.items():
        assert 'strict off-server acceptance remains0' in text or '严格离机接受仍0' in text
        if path==DOCS[1]:line=next(l for l in text.splitlines() if '英文24意见完整草稿入口' in l)
        elif path==DOCS[3]:line=next(l for l in text.splitlines() if l.startswith('| 英文回复 |'))
        else:line=next(l for l in text.splitlines() if 'The24-comment complete author-review reply' in l)
        assert 'rebuttal_integrated_C50_20261010/' in line and 'rebuttal_integrated_C20_20261009/' not in line
        selected_lines[path.relative_to(R).as_posix()]=dict(line=path.read_text('utf8').splitlines().index(line)+1,text_sha256=hashlib.sha256(line.encode('utf8')).hexdigest())
    result=dict(status='PASS_BOUNDED_C_AFTER50_PENDING_AND_COMPLETE_C50_LATEST_WIRING',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),findings=[],source_pins=pins,
        source_review_ranges={'tmp/update_reactivation_state_20261009.py':[[2201,2236],[2334,2354],[2389,2395],[2489,2500]],'tmp/update_completion_current_20261009.py':[[91,102],[227,230]],'tmp/update_overview_closure100_root_20261009.py':[[77,97],[133,138]]},
        observed=dict(native_strict_offserver=main['scientific_results_offserver_verified'],three_view_strict_offserver=150,three_view_by_variant=main['three_view_counts_by_variant'],C_after50_selected=6,C_after50_offserver_new_accepted=0,pending_IDs_excluded_from_accepted=True,complete_C50_latest_reply=True,latest_reply_root_sha256=sha(PROOF),actual_startup_sha256=sha(START),actual_deployment_sha256=sha(DEPLOY)),
        checks=dict(three_updater_sources_parsed_without_execution=True,startup_and_deployment_actual_pins_match=True,three_views_156_guarded_by_future_actual_exact6_root_adoption=True,current_native_and_views_denominators_distinct=True,complete_C50_overrides_historical_C20_and_addendum=True,current_latest_reply_links=selected_lines,metadata24_comments_38_C_scalars_19_scope_44_links_11_edits_match_adopted_root=True,author_review_manuscript_unapplied_final_test_pending_preserved=True),
        limitations=['Source inspection and current local state wiring only; no updater was executed.','No server, CNN, strict acceptor, archive/tensor or scientific arithmetic check repeated.','Historical C20 branches/text remain historical, not the current latest reply.','Startup proof is a fixed observed snapshot; no new live-server observation is claimed.'],
        local_writer_attempt_failure='Initial inline report writer had an invalid numeric-leading Python keyword; SyntaxError occurred before any statement/file write. Corrected only the local report writer; no shared updater or scientific output was affected.',shared_files_modified=False,SSH=False,CNN=False,Git=False,canonical_modified=False)
    for name,pin in pins.items():assert sha(R/name)==pin['sha256']
    with (D/'ROOT_INDEPENDENT_REVIEW.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],review_sha256=sha(D/'ROOT_INDEPENDENT_REVIEW.json'),source_pins=len(pins),findings=0,native=result['observed']['native_strict_offserver'],views=150,C_after50_new_accepted=0)))

if __name__=='__main__':main()
