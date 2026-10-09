"""Bind the unchanged original32 validation configurations to a bounded dispatch approval."""
from pathlib import Path
import datetime
import hashlib
import itertools
import json
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
OLD=ROOT/'tmp/celeba_hybrid_gpu_prepared_20261009'
OUT=ROOT/'tmp/celeba_hybrid_screen32_root_approval_20261009'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(NEW/'FILES_SHA256.json')=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
files=read(NEW/'FILES_SHA256.json')['files']
for n,h in files.items():assert sha(NEW/n)==h,n
assert len(files)==69
unchanged=['body.py','driver.py','writer_policy.py','scientific_snapshot/worker.py','scientific_snapshot/adapters.py',
    'scientific_snapshot/accept_result.py','scientific_snapshot/prepare_jobs.py','scientific_snapshot/protocol.json','summarize.py']
for n in unchanged:assert (NEW/n).read_bytes()==(OLD/n).read_bytes(),n
old,new=read(OLD/'runtime_protocol.json'),read(NEW/'runtime_protocol.json')
assert {k for k in set(old)|set(new) if old.get(k)!=new.get(k)}=={'status','runtime_execution_identity','limits'}
assert new['status']=='FROZEN' and sha(NEW/'runtime_protocol.json')=='bcc66477d22096eaf647e31f065f59ed6727716dcf01db814d5edcbe4ad525f1'
scope,previous=read(NEW/'screen_scope.json'),read(OLD/'screen_scope.json')
assert sha(NEW/'screen_scope.json')=='d76d5fdff375c58b3b42354fc4256e19b26b29f3fdb4387530f0ae8617920f94'
assert scope['status']=='FROZEN_HYBRID_VALID_SCREEN_ONLY' and len(scope['jobs'])==32
prior={r['id']:r for r in previous['jobs']};seen=set();grid=set()
for entry in scope['jobs']:
    job=read(NEW/entry['job']);original=read(OLD/prior[entry['id']]['job'])
    assert sha(NEW/entry['job'])==entry['job_sha256']
    assert {k:v for k,v in job.items() if k!='runtime_protocol_sha256'}=={k:v for k,v in original.items() if k!='runtime_protocol_sha256'}
    config,adapter=job['config'],job['adapter']
    assert config['rounds']==70 and config['seed']==91001 and config['device']=='cuda'
    assert config['celeba_evaluation_split']=='valid' and config['ad2_calibration_enabled'] is False
    assert config['guardfed_fairness_lambda']==adapter['fairness_lambda'] and config['trust_threshold']==adapter['threshold']
    grid.add((config['learning_rate'],adapter['fairness_lambda'],adapter['threshold'],job['distribution'],job['attack']))
    seen.add(job['id'])
assert len(seen)==32 and grid==set(itertools.product((.0005,.001),(5.,20.),(.1,.2),('IID','non-IID'),('Benign','S-DFA')))
for name in ('screen_runs','gate_runs','screen_failure.json','screen_complete.json','APPROVED_screen.json'):assert not (NEW/name).exists()
proof=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/HYBRID_CUDA_FOUR_ROOT_VERIFICATION.json'
assert read(proof)['actual_cuda_canaries_accepted']==4 and read(proof)['paired_model_tensors_exact']
draft=NEW/'execution_attachments/APPROVED_screen_DRAFT.json'
assert sha(draft)=='d274827517148a9557be959737f6b3b6c29c0ea4e5badda427d87eadbe3cf9fb'
approval=dict(status='ROOT_APPROVED_EXACT_ORIGINAL32_VALID_SCREEN_WITH_FRESH_DISPATCH_PREFLIGHT_REQUIRED',
    approved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(NEW/'FILES_SHA256.json'),
    scope_sha256=sha(NEW/'screen_scope.json'),runtime_protocol_sha256=sha(NEW/'runtime_protocol.json'),
    exact_ids=[r['id'] for r in scope['jobs']],unchanged_sources=unchanged,original32_configurations_unchanged=True,
    root_cuda_four_proof_sha256=sha(proof),actual_cuda_gate_canaries_accepted=4,
    cpu_threads=1,max_processes=1,allowed_cpus=[104],nice=10,idle_io=True,cuda_visible_device='0',
    gpu_uuid='GPU-da357477-30a7-fddc-344b-a20513b9a2d0',min_free_gpu_memory_mib=4096,
    required_fresh_resource_age_seconds=90,require_recovery_action_none=True,require_no_duplicate=True,
    require_no_restricted_cpu_overlap=True,require_actual_quota_and_existing_role_budget=True,
    selection='Original four-condition mean score, exact ties candidate lexical order; preserve accuracy champion, Pareto and all candidates',
    seed_n=1,no_sample_std_or_significance=True,test_authorized=False,formal100_authorized=False,
    automatic_or_partial_retry_authorized=False,alter_existing_services_or_environments_authorized=False,
    negative_and_constant_predictions_must_be_preserved=True,goal_complete=False,
    root_review_note='Initial readonly review assumed a top-level seed; actual frozen schema correctly binds config.seed. No package or scientific defect, no execution occurred.')
OUT.mkdir(exist_ok=False)
with (OUT/'APPROVED.json').open('x',encoding='utf8') as stream:json.dump(approval,stream,indent=2);stream.write('\n')
(OUT/'APPROVED.sha256').write_text(sha(OUT/'APPROVED.json')+'\n',encoding='ascii')
print(json.dumps(dict(status=approval['status'],source_members_verified=69,original_jobs_verified=32,approval_sha256=sha(OUT/'APPROVED.json'))))
