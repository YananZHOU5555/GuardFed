"""Reuse the accepted CPU image gate; only seed/root-audit/stage boundary is adapted."""
from pathlib import Path
import argparse,ast,difflib,hashlib,importlib.util,json,os,sys,time,traceback
sys.dont_write_bytecode=True
from metadata import HERE,OLD,STAGE,PINS,H,read,coverage_source,validate_jobs
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'

def render_gate_functions():
    text=(OLD/'gate.py').read_text(encoding='utf8');assert H((OLD/'gate.py').read_bytes())==PINS['gate.py']
    functions={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    changes={
      'run_one':{"int(job['attack'] == 'S-DFA')":"int(job['attack'] in {'FedSA', 'S-DFA', 'Sp-DFA'})",
        'core.set_seed(91001, deterministic_image=True)':"core.set_seed(job['config']['seed'], deterministic_image=True)",
        'Full-image exploratory CPU three-round gate only; formal five decisions unresolved; no test/GPU/70-round result':
        'Same-horizon exploratory CPU3 coverage interface gate only; selected recipe and frozen192 bound; no test/GPU/70-round result'},
      'checked':{
        "(91001, 3, job['config']['client_alpha'], 20, 4)":"(job['config']['seed'], 3, job['config']['client_alpha'], 20, 4)",
        "attacked = job['attack'] == 'S-DFA'":"attacked = job['attack'] in {'FedSA', 'S-DFA', 'Sp-DFA'}",
        "    if job['attack'] == 'S-DFA':\n        require(all(x['attack_types'] == ['fflip', 'foe'] and x['foe_mode'] == 'fedsa' and\n                    'gradient sign-conjugacy' in x['foe_impl'] for x in result['attack_audit'][:4]), 'S-DFA semantics changed')":
        "    check_attack_assignment(job['attack'], result['attack_audit'])"}}
    derived={};diff=[]
    for name,edits in changes.items():
        old=functions[name];new=old
        for a,b in edits.items():assert new.count(a)==1,(name,a);new=new.replace(a,b)
        ast.parse(new);derived[name]=new
        diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='accepted_gate/'+name,tofile='coverage_gate/'+name))
    return derived,''.join(diff)

def check_attack_assignment(attack,audits):
    assert attack in ('Benign','F Flip','FedSA','Sp-DFA') and len(audits)==20
    for cid,row in enumerate(audits):
        expected=[] if cid>=4 or attack=='Benign' else {'F Flip':['fflip'],'FedSA':['foe'],
            'Sp-DFA':['fflip'] if cid<2 else ['foe']}[attack]
        assert row['client_id']==cid and row['attack_types']==expected
        assert not row.get('label_changed_count',0)
        if 'foe' in expected:assert row['foe_mode']=='fedsa' and 'gradient sign-conjugacy' in row['foe_impl']

def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def load_context(stage,repo):
    coverage_source();stage,repo=Path(stage).resolve(),Path(repo).resolve();scope=read(stage/'scope.json')
    assert scope['status']=='PREPARED_EXACT14_NOT_DISPATCHED' and scope['scientific_table_records']==0 and scope['test'] is False
    for p,h in scope['external_identities'].items():assert H(Path(p).read_bytes())==h,p
    assert scope['rounds']==3 and scope['seed']==91002 and scope['max_workers']==1 and scope['cpu_threads']==8 and scope['device']=='cpu'
    jobs=[]
    for e in scope['jobs']:
        assert e['job']=='jobs/'+e['id']+'.json' and e['output']=='runs/'+e['id']
        assert H((stage/e['job']).read_bytes())==e['job_sha256'];jobs.append(read(stage/e['job']))
    validate_jobs(jobs,scope['candidates'],scope['original_protocol'])
    gate=load(OLD/'gate.py','gate');gate.HERE=stage;gate.REPO=repo;gate.STAGE=STAGE
    gate.check_attack_assignment=check_attack_assignment
    for name,source in render_gate_functions()[0].items():exec(compile(source,'<minimal-gate-boundary:'+name+'>','exec'),gate.__dict__)
    load(OLD/'shared_cache_wrapper.py','coverage_shared_inputs').bind_shared_paths()
    gate.hashes(stage,scope['local_hashes'])
    return gate,scope

def runtime_receipt(stage,scope,approval,sha):
    assert H(Path(approval).read_bytes())==sha;receipt=read(approval)
    assert receipt['status']=='ROOT_APPROVED_GRADIENT14_CPU3_CANARIES_ONLY'
    assert receipt['scope_sha256']==H((stage/'scope.json').read_bytes())
    assert receipt['jobs']=={e['id']:e['job_sha256'] for e in scope['jobs']}
    assert receipt['test_authorized'] is False and receipt['coverage192_authorized'] is False and receipt['automatic_retry'] is False
    assert 0<=time.time()-receipt['measured_unix']<=120
    cpus=receipt['exclusive_cpu_ids'];assert len(cpus)==len(set(cpus))==8 and all(type(x) is int and x>=0 for x in cpus)
    assert receipt['no_restricted_cpu_overlap'] and receipt['old_gradient64_exited'] and receipt['protected_main_healthy']
    assert receipt['no_duplicate_gate_worker'] and receipt['protected_main_growth_or_completed']
    assert receipt['nominal_existing_compute_threads']+8<=receipt['actual_cpu_quota'] and receipt['ram_available_bytes']>=8*1024**3
    return receipt

def run(stage,repo,job_id,approval,sha):
    assert sys.platform=='linux','This adapter has no local Windows model-output path'
    stage=Path(stage).resolve();assert stage.is_relative_to(Path('/workspace'))
    gate,scope=load_context(stage,repo);receipt=runtime_receipt(stage,scope,approval,sha)
    assert not list(stage.glob('failure*.json')),'Prior outer failure requires review'
    entry=next(e for e in scope['jobs'] if e['id']==job_id)
    index=scope['jobs'].index(entry)
    for previous in scope['jobs'][:index]:
        previous_out=stage/previous['output']
        assert (previous_out/'item_receipt.json').is_file() and not list(previous_out.glob('failure*.json'))
    attempt=stage/('attempt_'+H(job_id.encode())+'.json')
    with attempt.open('x',encoding='utf8') as f:
        json.dump(dict(id=job_id,started_unix=time.time(),approval_sha256=sha,automatic_retry=False),f)
    assert H(Path('/etc/vast-agents-guide.md').read_bytes())==GUIDE
    assert os.sched_getaffinity(0)==set(receipt['exclusive_cpu_ids']) and os.getpriority(os.PRIO_PROCESS,0)>=10
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert os.environ.get(n)=='8'
    assert not (stage/entry['output']).exists(),'No partial/retry overwrite'
    gate.hashes(Path(repo),scope['protected_source_hashes'])
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
    job=read(stage/entry['job']);worker=load(Path(scope['worker_paths'][job['method']][job['implementation']]),'gate_scientific_worker')
    core=worker.load_core(repo);components=worker.load_components()
    assert torch.get_num_threads()==8 and torch.get_num_interop_threads()==1 and not torch.cuda.is_initialized()
    gate.run_one(entry,scope,receipt,worker,core,components)
    # Observation only: no draws or numerical changes; original artifact receipt remains untouched.
    import random,numpy as np
    ns=np.random.get_state()
    rng=dict(python=H(repr(random.getstate()).encode()),numpy=H(ns[0].encode()+ns[1].tobytes()+repr(ns[2:]).encode()),
        torch_cpu=H(torch.get_rng_state().numpy().tobytes()))
    out=stage/entry['output'];gate.write(out/'rng_after.json',rng)
    for p,h in scope['external_identities'].items():assert H(Path(p).read_bytes())==h
    gate.write(out/'item_receipt.json',dict(status='CPU3_GATE_ORIGINAL_STRICT_AND_EXTERNAL_IDENTITY_PASS',
        acceptance_sha256=H((out/'acceptance.json').read_bytes()),rng_sha256=H((out/'rng_after.json').read_bytes()),
        scientific_table_records=0,scope_sha256=H((stage/'scope.json').read_bytes())))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True);a.add_argument('--repo',required=True);a.add_argument('--job',required=True)
    a.add_argument('--approval',required=True);a.add_argument('--approval-sha256',required=True);x=a.parse_args()
    try:run(x.stage,x.repo,x.job,x.approval,x.approval_sha256)
    except BaseException as error:
        failure=Path(x.stage)/('failure_outer_'+H(x.job.encode())+'.json')
        if failure.parent.is_dir() and not failure.exists():
            with failure.open('x',encoding='utf8') as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc()),f,indent=2)
        raise
