"""Local preparation and refusal checks; no scientific imports or inference."""
import copy
import json
from pathlib import Path
import tempfile

import batch
from resource_extra import add_known_roles

HERE = Path(__file__).resolve().parent
checks = []
scope = batch.identities(check_new_seal=False)
template = batch.read(HERE / 'APPROVAL_TEMPLATE.json')
approval = dict(template, status='APPROVED_REMAINING_SEVEN_MECHANISM_VALID_REPLAY_ONLY',scope_sha256='scope-sha',execution_seal_sha256='seal-sha')
batch.check_approval(approval,scope,'scope-sha','seal-sha')
checks.append(dict(name='actual_seven_original_checkpoint_job_result_pairs_and_scope_validated',pass_=True))
def refuses(name, callback):
    try:
        callback()
    except (ValueError, AssertionError):
        checks.append(dict(name=name,rejected=True))
    else:
        raise AssertionError('Expected refusal: '+name)
for name,key,value in [
    ('missing_root_approval','status','PREPARED_NOT_APPROVED'),
    ('already_accepted_single_seed91002','selected_ids',['minus_U_IID_Benign_seed91002']),
    ('duplicate_terminal','selected_ids',batch.SELECTED[:-1]+[batch.SELECTED[0]]),
    ('pending_seed91009','selected_ids',batch.SELECTED[:-1]+['minus_U_IID_Benign_seed91009']),
    ('Full_not_new_inference','selected_ids',batch.SELECTED[:-1]+['GuardFed-AD2+_IID_Benign_seed91001']),
    ('wrong_cpu','allowed_cpus',list(range(8,16))),
    ('higher_concurrency','max_processes',2),
    ('test_split','target_split','test'),
    ('relaxed_native_tolerance','native_tolerance',1e-6),
    ('new_Full_inference','new_full_inference',1),
    ('automatic_retry','automatic_retry_authorized',True),
    ('another_inventory','inventory_sha256','0'*64),
    ('another_scientific_bridge','bridge_sha256','0'*64)]:
    altered=copy.deepcopy(approval); altered[key]=value
    refuses(name,lambda altered=altered: batch.check_approval(altered,scope,'scope-sha','seal-sha'))
altered=copy.deepcopy(approval);altered['outputs'][batch.SELECTED[0]]='/workspace/GuardFed-celeba-expanded/result'
refuses('output_points_to_original_repo',lambda:batch.check_approval(altered,scope,'scope-sha','seal-sha'))
with tempfile.TemporaryDirectory(prefix='mechanism7_refusal_') as directory:
    output=Path(directory)/'terminal'; batch.fresh_output(output)
    output.mkdir(); (output/'original_evidence.txt').write_text('preserve')
    refuses('nonempty_or_partial_output_preserved',lambda:batch.fresh_output(output))
    assert (output/'original_evidence.txt').read_text()=='preserve'
    failed=Path(directory)/'failed'; failed.with_name(failed.name+'.bridge_failure.json').write_text('{}')
    refuses('failure_sentinel_preserved',lambda:batch.fresh_output(failed))
base=dict(tracked_compute=[dict(pid=100,role='formal_GPU_worker')],nominal_compute_threads_including_this8=104,actual_quota_cores=122.87999)
extra=[dict(pid=101,role='Hybrid_repair_CPU8',task='hybrid',compute_threads=8,cpus=list(range(8,16))),
       dict(pid=102,role='FLscreen_GPU_CPU1',task='FL1',compute_threads=1,cpus=list(range(128))),
       dict(pid=103,role='FLscreen_GPU_CPU1',task='FL2',compute_threads=1,cpus=list(range(128)))]
assert add_known_roles(base,extra)['total_effective_nominal']==114
checks.append(dict(name='88baseline_8formal_8Hybrid_2FL_8new_nominal114_not_measured_usage',pass_=True))
for name, rows in [
    ('duplicate_compute_pid',extra+[copy.deepcopy(extra[0])]),
    ('duplicate_compute_task',[extra[0],dict(extra[1],task='hybrid')]),
    ('wrong_Hybrid_affinity',[dict(extra[0],cpus=list(range(112,120)))]),
    ('wrong_FL_one_thread',[dict(extra[1],compute_threads=8)]),
    ('too_many_FLworkers',extra+[dict(extra[1],pid=104,task='FL3')]),
    ('restricted_FL_overlaps112',[dict(extra[1],cpus=list(range(112,120)))])]:
    refuses(name,lambda rows=rows:add_known_roles(base,rows))
refuses('quota_exceeded',lambda:add_known_roles(dict(base,nominal_compute_threads_including_this8=120),extra))
original=batch.read(batch.PREPARED/'inventory_actual8_Full100refs.json')
assert len(scope['selected_records'])==7 and {row['id'] for row in scope['selected_records']}==set(batch.SELECTED)
assert not set(batch.SELECTED).intersection(original['pending_new_ids_no_checkpoint'])
assert len(original['full_references'])==100 and scope['paired_Full_display']['missing_paired_display_does_not_invalidate_mechanism_terminal']
checks.append(dict(name='no_pending_SHA_no_repeated_single_Full_display_missing_explicit',pass_=True))
target=HERE/'selfcheck.json';assert not target.exists()
target.write_bytes((json.dumps(dict(status='PASS_LOCAL_PREPARATION_ONLY',checks=checks,check_count=len(checks),
    training_runs=0,inference_runs=0,scientific_body_rewritten=False),indent=2)+'\n').encode())
print('PASS local checks',len(checks),'no images or scientific imports')
