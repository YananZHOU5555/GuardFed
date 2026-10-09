"""Local structural fixtures only; no SSH, acceptor, model, backup or actual selection."""
from pathlib import Path
import ast,copy,hashlib,json,statistics,sys
from closure_guard import EXPECTED_CHAIN,EXPECTED_PACKAGE,SOURCE_PINS,validate_previous,validate_snapshot,validate_live
from summary32 import summarize_records,validate_final_proof
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch/accepted_delta_after19_v2_20261009';RELEASE=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_frozen_release'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
previous=read(HERE/'PREVIOUS_CHAIN.json');manifest=read(HERE/'manifest.json');protocol=read(HERE/'protocol.json')
checks=[]
def refuse(label,call):
 try:call()
 except (AssertionError,KeyError,TypeError):checks.append(label)
 else:raise AssertionError('Refusal missing: '+label)
wanted=validate_previous(previous,sha(HERE/'PREVIOUS_CHAIN.json'),EXPECTED_PACKAGE,manifest);assert wanted==read(HERE/'EXACT_DELTA.json')['selected_ids']
assert 'package_sha256' not in previous
refuse('wrong_prior_SHA',lambda:validate_previous(previous,'0'*64,EXPECTED_PACKAGE,manifest))
refuse('wrong_package_SHA',lambda:validate_previous(previous,EXPECTED_CHAIN,'0'*64,manifest))
missing=copy.deepcopy(previous);missing.pop('accepted_job_ids');refuse('missing_actual_prior_field',lambda:validate_previous(missing,EXPECTED_CHAIN,EXPECTED_PACKAGE,manifest))
duplicate=copy.deepcopy(previous);duplicate['accepted_job_ids'][-1]=duplicate['accepted_job_ids'][0];refuse('duplicate_prior_ID',lambda:validate_previous(duplicate,EXPECTED_CHAIN,EXPECTED_PACKAGE,manifest))
snapshot=dict(source_sha256=SOURCE_PINS.copy(),queue=dict(completed=32,pending=0,active=[]),failure_paths=[],rows=[dict(id=e['id'],active=False,progress=dict(round=70),result_exists=True,acceptance_exists=True,screen_identity_exists=True) for e in manifest['jobs']])
validate_snapshot(snapshot,manifest)
for label,change in [
 ('29_terminal_2active_1pending',lambda s:s.update(queue=dict(completed=29,pending=1,active=[dict(id='STRUCTURAL_FIXTURE_ACTIVE_A'),dict(id='STRUCTURAL_FIXTURE_ACTIVE_B')]))),
 ('active_producer',lambda s:s['rows'][0].update(active=True)),
 ('round69',lambda s:s['rows'][0]['progress'].update(round=69)),
 ('missing_acceptance',lambda s:s['rows'][0].update(acceptance_exists=False)),
 ('preserved_failure',lambda s:s.update(failure_paths=['STRUCTURAL_FIXTURE_FAILURE'])),
 ('wrong_source_SHA',lambda s:s['source_sha256'].update({'frozen_score.py':'0'*64}))]:
 changed=copy.deepcopy(snapshot);change(changed);refuse(label,lambda:validate_snapshot(changed,manifest))
live=dict(queue=snapshot['queue'],flgmm=dict(rows=[dict(id=e['id'],active=False,progress=dict(round=70),terminal_acceptance=True) for e in manifest['jobs']]))
validate_live(live,manifest,[])
refuse('live_job_process_even_after_queue_idle',lambda:validate_live(live,manifest,[dict(argv=['python','run_one.py','--job-id',wanted[0]])]))
fixture=[]
for e in manifest['jobs']:
 high=e['distribution']=='non-IID';metrics=dict(accuracy=.8,aeod=.2 if high else 0.,aspd=.2 if high else 0.)
 fixture.append(dict(id=e['id'],candidate=e['tuning_candidate'],distribution=e['distribution'],attack=e['attack'],seed=91001,rounds=70,metrics=metrics))
result=summarize_records(fixture,manifest,protocol)
assert len(result['candidates'])==len(result['three_metric_pareto'])==8 and len(result['records'])==32
assert result['selected_per_method'][manifest['method']]['candidate']==min(c['id'] for c in protocol['candidates'])
from frozen_score import score
group=fixture[:4];expected=statistics.mean(score(r['metrics']) for r in group)
assert result['candidates'][0]['score']==expected and expected!=score({k:statistics.mean(r['metrics'][k] for r in group) for k in ('accuracy','aeod','aspd')})
varied=copy.deepcopy(fixture);last=max(c['id'] for c in protocol['candidates'])
for row in varied:
 if row['candidate']==last:row['metrics']=dict(accuracy=.99,aeod=1.,aspd=1.)
varied_result=summarize_records(varied,manifest,protocol)
assert varied_result['accuracy_champion']['candidate']==last and varied_result['selected_per_method'][manifest['method']]['candidate']!=last and len(varied_result['candidates'])==8
refuse('summary31_records',lambda:summarize_records(fixture[:-1],manifest,protocol))
bad=copy.deepcopy(fixture);bad[0]['seed']=91002;refuse('summary_wrong_seed',lambda:summarize_records(bad,manifest,protocol))
bad=copy.deepcopy(fixture);bad[0]['metrics']['accuracy']=float('nan');refuse('summary_nonfinite',lambda:summarize_records(bad,manifest,protocol))
root_fixture=dict(status='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS',accepted_before=26,accepted_new=6,accepted_total=32,previous_chain_sha256=EXPECTED_CHAIN,new_inference=0,final_test=False)
validate_final_proof(root_fixture)
for total in (26,29,31):
 bad=dict(root_fixture,accepted_total=total);refuse('summary_not_root_accepted32_'+str(total),lambda:validate_final_proof(bad))
old=(OLD/'collect_delta.py').read_bytes();new=(HERE/'collect_delta.py').read_bytes()
loop=lambda raw:raw[raw.index(b'    for identity in wanted:'):raw.index(b'    after=repo_identity')]
assert loop(old)==loop(new)
oldv=(OLD/'verify_delta_offserver.py').read_bytes();newv=(HERE/'verify_delta_offserver.py').read_bytes()
vloop=lambda raw:raw[raw.index(b"    for identity in receipt['accepted_new_ids']:"):raw.index(b"    assert len(results)==receipt['accepted_new']")]
assert vloop(oldv)==vloop(newv)
assert (HERE/'frozen_score.py').read_bytes()==(RELEASE/'frozen_score.py').read_bytes()
for p in HERE.glob('*.py'):compile(ast.parse(p.read_text(encoding='utf8')),p.name,'exec')
assert not any(name in sys.modules for name in ('screen_common','accept_result','torch'))
assert not any((HERE/name).exists() for name in ('AUTHORIZED_SNAPSHOT.json','EXECUTION_BINDINGS.json','PARTIAL_ACCEPTANCE.json','OFFSERVER_ACCEPTANCE.json','SUMMARY32.json'))
proof=dict(status='PREPARED_LOCAL_STRUCTURAL_AND_SOURCE_CHECKS_PASS_NOT_EXECUTED',positive_actual_prior26_schema=True,positive_synthetic32_gate=True,
 positive_synthetic_recipe_math=True,fixture_results_are_not_experimental_outputs=True,refusals=checks,
 original_per_job_scientific_loop_raw_bytes_exact=True,strict_loop_raw_sha256=hashlib.sha256(loop(old)).hexdigest(),
 original_offserver_per_job_loop_raw_bytes_exact=True,original_frozen_score_bytes_exact=True,
 per_condition_score_then_four_condition_mean=True,exact_tie_lexical=True,accuracy_champion_and_all8_candidates_retained=True,
 acceptor_or_model_imported=False,actual_snapshot_created=False,actual_new_accepted=0,actual_recipe_selected=False,
 no_SSH_no_backup_no_CNN_no_STATE_no_Git=True)
with (HERE/'PREPARATION_CHECKS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof))
