"""CPU import/recorder regression only; fixtures are not image-training evidence."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def probe(repo, output):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from rng_capture import RNGCapture
    sys.path.insert(0, str(HERE / 'sealed_group_b'))
    from worker import load_core
    recorder = RNGCapture()
    recorder.install()
    try:
        core = load_core(repo)
        recorder.begin_training()
        imported = recorder.import_evidence()
        initial = recorder.snapshot()
        assert not initial['training_states']
        core.set_seed(91001, deterministic_image=True)
        generator = np.random.default_rng(91001)
        generator.random(5)  # labelled recorder test; no dataset/model execution
        explicit = recorder.snapshot()
        assert len(explicit['training_states']) == 1
        assert explicit['training_origins'] == [dict(reason='default_rng_called_after_import', import_index=None)]
        recorder.imported[0].random()  # direct post-import use must enter training registry
        imported_use = recorder.snapshot()
        assert len(imported_use['training_states']) == 2
        assert imported_use['training_origins'][1]['import_index'] == 0
        assert 0 in imported_use['import_advanced_indices']
        assert not torch.cuda.is_initialized()
        write(output, dict(scope='CPU_IMPORT_AND_SYNTHETIC_RECORDER_TEST_ONLY',
            imported=imported, initial=initial, explicit=explicit, imported_use=imported_use,
            no_bundle_or_model_training=True, no_cuda_context=True))
    finally:
        recorder.restore()


def fixtures(output, first, second):
    import torch
    from verify_gpu import compare_repeat
    paths = [output / 'fixture1', output / 'fixture2']
    for path, data in zip(paths, (first, second)):
        path.mkdir()
        torch.save({'unit_test_tensor': torch.tensor([1.0])}, path / 'model.pt')
        write(path / 'result.json', {name: [] for name in ('metrics', 'trajectory_metrics', 'last10_metrics', 'round_summaries', 'attack_audit', 'evaluation_stats')})
        write(path / 'diagnostics.json', [])
        write(path / 'import_rng_evidence.json', data['imported'])
        for prefix in ('round_001', 'round_002', 'round_003', 'final'):
            write(path / f'{prefix}_rng.json', dict(python='unit fixture', numpy_legacy='unit fixture',
                numpy_generators=data['explicit']['training_states']))
            write(path / f'{prefix}_rng_scope.json', data['explicit'])
            torch.save(dict(cpu=torch.tensor([1], dtype=torch.uint8), cuda=[torch.tensor([2], dtype=torch.uint8)]), path / f'{prefix}_rng.pt')
        for number in range(1, 4): write(path / f'round_{number:03d}_state.json', dict(unit_test_only=True))
    exact = compare_repeat(*paths)
    assert exact['exact']
    target = paths[1] / 'final_rng.json'
    bad = json.loads(target.read_text())
    bad['numpy_generators'][0]['state']['state'] += 1
    write(target, bad)
    rejected = compare_repeat(*paths)
    assert not rejected['exact'] and not rejected['checks']['final_rng_json']
    assert rejected['checks']['model_tensors']
    return dict(import_only_differences_retained=exact['import_rng_disclosure'],
        unchanged_training_rng_passes=True, changed_training_rng_fails=True,
        fixture_scope='Only comparator unit fixture; never passed to check_one or paper tables')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.probe:
        probe(args.repo, args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for number in (1, 2):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--repo', str(args.repo),
                '--out', str(args.out / f'probe{number}.json'), '--probe'], env=env, check=True)
        a, b = [json.loads((args.out / f'probe{i}.json').read_text()) for i in (1, 2)]
        assert len(a['imported']['records']) == len(b['imported']['records']) == 12
        assert all(any(frame['function'] == '_generate_example' for frame in r['callsite']) for r in a['imported']['records'])
        assert a['explicit']['training_states'] == b['explicit']['training_states']
        assert a['imported_use']['training_states'] == b['imported_use']['training_states']
        result = dict(status='PASS',real_gpu_runs=0,real_image_training_runs=0,
            import_generators_per_process=12, training_registry_empty_at_boundary=True,
            explicit_training_generator_captured=True, directly_used_import_generator_captured=True,
            comparator=fixtures(args.out, a, b))
        write(args.out / 'REGRESSION.json', result)
        print(json.dumps(result))
