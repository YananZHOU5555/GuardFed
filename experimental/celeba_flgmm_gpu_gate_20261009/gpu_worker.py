"""One parent-frozen, three-round GPU CANARY; never a scientific screen job."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sealed_group_b'))
from worker import METHOD, LABEL, aggregation_wrapper, digest, load_core, write_json


def validate_inputs(repo, job_path):
    freeze = json.loads((HERE / 'FREEZE.json').read_text())
    assert freeze['status'] == 'FROZEN' and freeze['scope'] == 'four_3round_GPU_CANARY_only'
    assert freeze['prerequisites']['cpu_offserver_verified'] is True
    for name, expected in freeze['local_hashes'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert digest(HERE / name) == expected, name
    scope = json.loads((HERE / 'SCOPE.json').read_text())
    assert scope['status'] == 'PREPARED_PENDING_PARENT_REVIEW'
    assert scope['formal_screen_authorized'] is False and scope['cpu_threads'] == 1
    relative = job_path.resolve().relative_to(HERE).as_posix()
    assert relative in scope['jobs'] and relative in freeze['local_hashes']
    job = json.loads(job_path.read_text())
    expected = dict(scope['base_config'], rounds=3, seed=91001, device='cuda', learning_rate=.001,
        client_alpha={'IID': 5000., 'non-IID': 5.}[job['distribution']],
        experiment_suite=scope['version'], experiment_tag=job['id'])
    assert job['config'] == expected and job['adapter'] == dict(warmup_rounds=1, control_width=3.)
    assert (job['dataset'], job['method'], job['evidence_stage']) == ('celeba', METHOD, 'GPU_CANARY_REAL_IMAGE_ONLY')
    assert job['repeat'] in (1, 2)
    assert (job['distribution'], job['attack']) in [('IID', 'Benign'), ('non-IID', 'S-DFA')]
    for name, expected in scope['source_hashes'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert digest(repo / name) == expected, name
    return freeze, scope, job


def run(repo, job_path, out):
    freeze, scope, job = validate_inputs(repo, job_path)
    # Each supervisor child must expose exactly one physical GPU. Logical cuda:0
    # then refers to that GPU, and its RNG state is independent of other workers.
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert visible in ('0', '1'), 'Require explicit one-device CUDA_VISIBLE_DEVICES=0 or 1'
    assert visible == str(job['gpu'])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    generators = []
    original_rng = np.random.default_rng

    def track_rng(*args, **kwargs):
        generator = original_rng(*args, **kwargs)
        if all(generator is not previous for previous in generators):
            generators.append(generator)
        return generator

    def json_value(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {k: json_value(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [json_value(v) for v in value]
        return value

    def save_rng(prefix):
        torch.cuda.synchronize()
        write_json(out / f'{prefix}_rng.json', json_value(dict(python=random.getstate(),
            numpy_legacy=np.random.get_state(),
            numpy_generators=[g.bit_generator.state for g in generators])))
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()), out / f'{prefix}_rng.pt')

    try:
        write_json(out / 'job.json', job)
        np.random.default_rng = track_rng  # records actual generators; returns them unchanged
        core = load_core(repo)
        cfg = core.ExperimentConfig(**job['config'])
        core.set_seed(cfg.seed, deterministic_image=True)
        provenance = dict(job_sha256=digest(job_path), freeze_sha256=digest(HERE / 'FREEZE.json'),
            source_hashes=scope['source_hashes'], local_hashes=freeze['local_hashes'],
            python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0), visible_gpu=visible, device='cuda:0', threads=1,
            real_data=True, formal_table_eligible=False, training_resume_supported=False,
            rng_capture='global Python/NumPy/Torch CPU+visible CUDA and actual default_rng generators')
        write_json(out / 'provenance.json', provenance)
        original = core.aggregate_round
        wrapped = aggregation_wrapper(original, job['adapter'], out)
        core.aggregate_round = wrapped

        def progress(item):
            number = item['round']
            shutil.copyfile(out / 'state.json', out / f'round_{number:03d}_state.json')
            save_rng(f'round_{number:03d}')
            write_json(out / 'progress.json', dict(item, job_id=job['id'], pid=os.getpid(),
                updated_unix=time.time(), elapsed_sec=time.time() - started))
            print(json.dumps(dict(job_id=job['id'], round=number, metrics=item['metrics'])), flush=True)

        try:
            result = core.run_experiment('celeba', job['distribution'], METHOD, job['attack'], cfg,
                'GPU_CANARY_only', torch.device('cuda:0'), progress_callback=progress, checkpoint_path=out / 'model.pt')
        finally:
            core.aggregate_round = original
        assert torch.are_deterministic_algorithms_enabled() and wrapped.controller.round_index == 3
        save_rng('final')
        after = {name: digest(repo / name) for name in scope['source_hashes']}
        write_json(out / 'source_verification.json', dict(before=scope['source_hashes'], after=after,
            consistent=after == scope['source_hashes']))
        assert after == scope['source_hashes'], 'Source/data changed during GPU CANARY; evidence preserved'
        result.update(status='canary_complete', evidence_stage='GPU_CANARY_REAL_IMAGE_ONLY',
            revision_job=job, method_impl_note=LABEL, provenance=provenance)
        write_json(out / 'result.json', result)
        names = ['model.pt', 'state.json', 'diagnostics.json', 'result.json', 'provenance.json', 'job.json', 'source_verification.json']
        names += [f'round_{i:03d}_state.json' for i in range(1, 4)]
        names += [f'{prefix}_rng.{suffix}' for prefix in ['round_001', 'round_002', 'round_003', 'final'] for suffix in ['json', 'pt']]
        from verify_gpu import check_one
        write_json(out / 'acceptance.json', dict(status='RECORDED_PENDING_CHECK', formal_table_eligible=False,
            artifact_hashes={name: digest(out / name) for name in names}, elapsed_sec=time.time() - started))
        check_one(HERE, out, require_pass=False)
        evidence = json.loads((out / 'acceptance.json').read_text())
        evidence['status'] = 'PASS'
        write_json(out / 'acceptance.json', evidence)
        print(json.dumps(dict(accepted=job['id'], elapsed_sec=evidence['elapsed_sec'])), flush=True)
    except BaseException as error:
        write_json(out / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), failed_unix=time.time()))
        raise
    finally:
        np.random.default_rng = original_rng


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.job.resolve(), args.out.resolve())
