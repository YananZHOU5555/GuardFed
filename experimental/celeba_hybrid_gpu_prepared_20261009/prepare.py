"""Materialize CUDA gate4 and original candidate32 drafts; no training, imports or self-freeze."""
import itertools
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as out:out.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def materialize(old_gate_scope,old_gate_root):
    assert not (HERE/'gate_jobs').exists() and not (HERE/'screen_jobs').exists()
    protocol_path=HERE/'runtime_protocol.json'
    if not protocol_path.exists():
        protocol=read(HERE/'scientific_snapshot/protocol.json')
        assert protocol['status']=='PREPARED_NOT_FROZEN'
        protocol['runtime_source_note']='Original5 source bytes remain scientific_snapshot; this additional runtime protocol SHA is bound independently in every job/scope.'
        protocol['limits']=['New CUDA gate4 and70round candidate32 have not executed; CPU4 closure is a separate required SHA-bound prerequisite.',
            'Custom project control, not an external-method reproduction or GuardFed-AD2+.',
            'No partial-output restart guarantee; preserve all failures/negative or constant predictions.',
            'PREPARED is not authorization; new frozen scope/protocol and external exact approval are required.']
        save(protocol_path,protocol)
    protocol=read(protocol_path)
    assert protocol['status'] in {'PREPARED_NOT_FROZEN','FROZEN'}
    adapter_names=['accept_result.py','adapters.py','prepare_jobs.py','protocol.json','worker.py']
    adapters={name:digest(HERE/'scientific_snapshot'/name) for name in adapter_names}
    local={str(path.relative_to(HERE)).replace('\\','/'):digest(path) for path in sorted(HERE.rglob('*'))
        if path.is_file() and path.suffix in {'.py','.sh','.conf','.json'} and path.name not in {'FILES_SHA256.json','selfcheck.json','gate_scope.json','screen_scope.json'}
        and 'jobs' not in path.parts and path.parent.name!='__pycache__'}
    # Snapshot's own seal is an input authority, not a seal of the new package.
    local['scientific_snapshot/FILES_SHA256.json']=digest(HERE/'scientific_snapshot/FILES_SHA256.json')
    common=dict(status='PREPARED_NOT_FROZEN',max_processes=1,cpu_threads=1,test_evaluation_authorized=False,
        protected_source_hashes=old_gate_scope['protected_source_hashes'],local_hashes=local,
        guide_sha256=old_gate_scope['guide_sha256'],runtime_protocol='runtime_protocol.json',
        runtime_protocol_sha256=digest(protocol_path),original_cpu_gate_sha256=old_gate_scope['local_hashes']['gate.py'],
        original_CPU_gate_not_CUDA_or70_evidence=True,new_method_or_attack_logic=False,
        resource_proposal=dict(max_new_GPU_processes=1,compute_threads=1,proposed_dedicated_cpu=[104],nice=10,io='idle',
            GPU='root chooses existing physical0 or1 after live headroom/identity check; no restart or preemption',
            worst_declared_nominal_if_remaining7_still_present=115,not_measured_cpu_utilization=True),
        metadata_boundary='Original loader may materialize test-tail attributes; no test image inference/fitting/selection, not untouched-test claim.')
    gate_jobs=[];exact={}
    for original in old_gate_scope['jobs']:
        job=read(old_gate_root/original['job']);identity=job['id'].replace('_cpu_gate3','_cuda_gate3')
        job=dict(job,id=identity,config=dict(job['config'],device='cuda',experiment_suite='celeba_hybrid_cuda_pipeline_gate_20261009_v1',experiment_tag=identity),
            runtime_protocol_sha256=digest(protocol_path))
        path=HERE/'gate_jobs'/(identity+'.json');save(path,job);exact[identity]=job
        gate_jobs.append(dict(id=identity,job=str(path.relative_to(HERE)).replace('\\','/'),job_sha256=digest(path),output='gate_runs/'+identity))
    gate=dict(common,kind='cuda_pipeline_gate',scope_file='gate_scope.json',rounds=3,run_parent='gate_runs',jobs=gate_jobs,exact_gate_jobs=exact,
        evidence_stage='real_image_cuda_pipeline_gate_only',formal_screen_status='PREPARED_NOT_FROZEN',
        claim_limit='Four full-image CUDA three-round canaries only; not CPU-CUDA equivalence,70round or final-test evidence.')
    save(HERE/'gate_scope.json',gate)
    jobs=[]
    for candidate,distribution,attack in itertools.product(protocol['candidates'],protocol['distributions'],protocol['attacks']):
        identity=f"{candidate['id']}_{distribution}_{attack}_seed91001_screen"
        config=dict(protocol['base_config'],rounds=70,seed=91001,client_alpha=protocol['distributions'][distribution],learning_rate=candidate['learning_rate'],
            guardfed_fairness_lambda=candidate['adapter']['fairness_lambda'],trust_threshold=candidate['adapter']['threshold'],experiment_suite=protocol['version'],experiment_tag=identity)
        job=dict(id=identity,dataset='celeba',method='CosineFairnessHybrid',distribution=distribution,attack=attack,phase='screen',evidence_stage='validation_screen',
            config=config,adapter=candidate['adapter'],tuning_candidate=candidate['id'],source_hashes=protocol['source_hashes'],adapter_source_hashes=adapters,
            runtime_protocol_sha256=digest(protocol_path))
        path=HERE/'screen_jobs'/(identity+'.json');save(path,job)
        jobs.append(dict(id=identity,job=str(path.relative_to(HERE)).replace('\\','/'),job_sha256=digest(path),output='screen_runs/'+identity,tuning_candidate=candidate['id'],distribution=distribution,attack=attack))
    assert len(jobs)==len({j['id'] for j in jobs})==32
    screen=dict(common,kind='validation_screen',scope_file='screen_scope.json',rounds=70,run_parent='screen_runs',jobs=jobs,
        evidence_stage='validation_screen',score=protocol['score'],selection_rule=protocol['selection_rule'],
        claim_limit='Original8 candidates,4conditions,seed91001 validation screen. n1; not independent test,multi-seed confirmation or full17-row table.')
    save(HERE/'screen_scope.json',screen)
    save(HERE/'manifest_draft.json',dict(status='PREPARED_NOT_FROZEN',execution_started=False,gate_jobs=gate_jobs,screen_jobs=jobs,
        gpu_gates_completed=0,screen_completed=0,source_scope_hashes={kind:digest(HERE/(kind+'_scope.json')) for kind in ('gate','screen')}))
    print('PREPARED_NOT_FROZEN four CUDA gate jobs and32 original validation screen jobs; no execution')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--original-gate-root',type=Path,required=True);a=p.parse_args()
    materialize(read(a.original_gate_root/'scope.json'),a.original_gate_root)
