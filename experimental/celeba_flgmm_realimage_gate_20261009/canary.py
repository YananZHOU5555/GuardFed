"""Two full-data, CPU-only real-image CANARY jobs; no formal screen launch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sealed_group_b"))
from worker import METHOD, LABEL, aggregation_wrapper, digest, load_core, write_json


def snapshot(path):
    stage = Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
    progress = []
    for p in stage.glob('runs/*/progress.json'):
        r = json.loads(p.read_text())
        if time.time() - r.get('updated_unix', 0) < 300:
            progress.append({k:r.get(k) for k in ['job_id','pid','round','updated_unix']})
    result = dict(time=time.time(), formal_active_progress=progress,
        supervisor=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_formal'],capture_output=True,text=True).stdout.strip(),
        gpus=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),
        cgroup={name:(Path('/sys/fs/cgroup')/name).read_text() for name in
                ['cpu.max','cpu.stat','memory.current','memory.max','memory.events']},
        disk_free=shutil.disk_usage('/workspace').free)
    write_json(path, result)
    return result


def verify_freeze(repo):
    receipt = json.loads((HERE/'FREEZE.json').read_text())
    assert receipt['status'] == 'FROZEN' and receipt['scope'] == 'two_3round_real_image_CANARY_only'
    for name, expected in receipt['local_hashes'].items():
        assert digest(HERE/name) == expected, ('canary freeze changed',name)
    scope=json.loads((HERE/'SCOPE.json').read_text())
    assert scope['formal_screen_authorized'] is False and scope['cpu_threads'] == 8
    for name, expected in scope['source_hashes'].items():
        assert digest(repo/name) == expected, ('core/data changed',name)
    return receipt,scope


def checked_result(job, out):
    import torch
    from flgmm_adapter import FLGMMAdapter
    if (out/'failure.json').exists():
        raise ValueError('Failure preserved; no automatic reuse')
    evidence=json.loads((out/'acceptance.json').read_text())
    assert evidence['status']=='PASS' and evidence['evidence_stage']=='CANARY_REAL_IMAGE_ONLY'
    for name,sha in evidence['artifact_hashes'].items():
        assert digest(out/name)==sha, name
    r=json.loads((out/'result.json').read_text())
    assert r['revision_job']==job and r['config']==job['config']
    assert r['evidence_stage']=='CANARY_REAL_IMAGE_ONLY' and r['status']=='canary_complete'
    for key in ['method','dataset','distribution','attack']:
        assert r[key]==job[key],key
    assert r['seed']==91001 and r['rounds']==3
    assert r['metrics']==r['trajectory_metrics'][-1]['metrics']
    for name in ['trajectory_metrics','round_summaries']:
        assert [x['round'] for x in r[name]]==[1,2,3]
    assert all(math.isfinite(v) and 0<=v<=1 for row in r['trajectory_metrics'] for v in row['metrics'].values())
    c=r['data_contract']['image_data_contract']
    assert (c['evaluation_split'],c['actual_train_rows'],c['actual_evaluation_rows'],c['train_eval_disjoint'],c['root_client_disjoint'])==('valid',162770,19867,True,True)
    assert r['evaluation_stats']['prediction_count']==19867
    diagnostics=json.loads((out/'diagnostics.json').read_text())
    assert [x['aggregate']['stage'] for x in diagnostics]==['per_round_gmm','fit_control_limit','monitor']
    for i,row in enumerate(diagnostics,1):
        assert row['aggregate']==r['round_summaries'][i-1]['aggregate']
        state=json.loads((out/f'round_{i:03d}_state.json').read_text())
        controller=FLGMMAdapter(range(20),**job['adapter']);controller.load_state_dict(state)
        assert controller.round_index==i and controller.ucl==row['aggregate']['ucl']
    model=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    assert model and all(torch.isfinite(v).all() for v in model.values())
    return r


def run_job(repo, job, receipt, scope):
    import torch
    out=HERE/'runs'/job['id'];out.mkdir(parents=True,exist_ok=False)
    started=time.time();cpu_before=resource.getrusage(resource.RUSAGE_SELF)
    try:
        assert job['evidence_stage']=='CANARY_REAL_IMAGE_ONLY' and job['adapter']==dict(warmup_rounds=1,control_width=3.)
        cfg_expected=dict(scope['base_config'],rounds=3,seed=91001,device='cpu',learning_rate=.001,
            client_alpha={'IID':5000.,'non-IID':5.}[job['distribution']],experiment_suite=scope['version'],experiment_tag=job['id'])
        assert job['config']==cfg_expected
        write_json(out/'job.json',job)
        core=load_core(repo);cfg=core.ExperimentConfig(**job['config'])
        assert torch.get_num_threads()==8 and not torch.cuda.is_initialized()
        provenance=dict(job_sha256=digest(HERE/'jobs'/f"{job['id']}.json"),freeze_sha256=digest(HERE/'FREEZE.json'),
            source_hashes=scope['source_hashes'],local_hashes=receipt['local_hashes'],
            python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device='cpu',threads=8,
            real_data=True,formal_table_eligible=False,training_resume_supported=False)
        write_json(out/'provenance.json',provenance)
        original=core.aggregate_round;wrapped=aggregation_wrapper(original,job['adapter'],out)
        core.aggregate_round=wrapped
        def progress(item):
            idx=item['round']
            shutil.copyfile(out/'state.json',out/f'round_{idx:03d}_state.json')
            write_json(out/'progress.json',dict(item,job_id=job['id'],pid=os.getpid(),updated_unix=time.time(),elapsed_sec=time.time()-started))
            print(json.dumps(dict(job_id=job['id'],round=idx,elapsed_sec=time.time()-started,metrics=item['metrics'])),flush=True)
        try:
            result=core.run_experiment('celeba',job['distribution'],METHOD,job['attack'],cfg,
                'real_image_CANARY_only',torch.device('cpu'),progress_callback=progress,checkpoint_path=out/'model.pt')
        finally:core.aggregate_round=original
        assert wrapped.controller.round_index==3 and torch.are_deterministic_algorithms_enabled()
        assert not torch.cuda.is_initialized()
        result.update(status='canary_complete',evidence_stage='CANARY_REAL_IMAGE_ONLY',revision_job=job,
                      method_impl_note=LABEL,provenance=provenance)
        write_json(out/'result.json',result)
        names=['model.pt','state.json','diagnostics.json','result.json','provenance.json','job.json',
               'round_001_state.json','round_002_state.json','round_003_state.json']
        now=resource.getrusage(resource.RUSAGE_SELF)
        evidence=dict(status='PASS',evidence_stage='CANARY_REAL_IMAGE_ONLY',formal_table_eligible=False,
            artifact_hashes={n:digest(out/n) for n in names},elapsed_sec=time.time()-started,
            cpu_user_seconds=now.ru_utime-cpu_before.ru_utime,cpu_system_seconds=now.ru_stime-cpu_before.ru_stime,
            max_rss_kib=now.ru_maxrss,threads=8,limitations=['CPU not GPU equivalence','Tg1 is stage-coverage only, not formal recipe','No optimizer/full RNG checkpoint or training resume guarantee'])
        write_json(out/'acceptance.json',evidence)
        checked_result(job,out)
        print(json.dumps(dict(accepted=job['id'],elapsed_sec=evidence['elapsed_sec'])),flush=True)
    except BaseException as exc:
        write_json(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),failed_unix=time.time()))
        raise


def main(repo):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='8'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    receipt,scope=verify_freeze(repo)
    snapshot(HERE/'resources_before.json')
    for name in scope['jobs']:
        job=json.loads((HERE/name).read_text())
        run_job(repo,job,receipt,scope)
        snapshot(HERE/f"resources_after_{job['id']}.json")
    write_json(HERE/'COMPLETE.json',dict(status='PASS',completed_canaries=2,formal_screen_jobs_started=0,
                                      formal_table_eligible=False,finished_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    main(p.parse_args().repo.resolve())
