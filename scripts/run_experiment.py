"""Run one fully specified experiment and preserve its terminal checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import reproduce_paper_tables as core

METHODS = {'FedAvg', 'FairFed', 'Median', 'FLTrust', 'FairGuard', 'GuardFed',
           'GuardFed-AD2', 'GuardFed-AD2+'}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_job(job, output):
    if job['dataset'] not in {'adult', 'compas', 'celeba'}:
        raise ValueError('Unsupported dataset')
    if job['distribution'] not in core.DISTRIBUTIONS or job['method'] not in METHODS:
        raise ValueError('Unsupported distribution or method; external baseline operators have separate interfaces')
    if job['attack'] not in core.ALL_ATTACKS:
        raise ValueError('Unsupported attack')
    if job['dataset'] == 'celeba' and job['config'].get('celeba_evaluation_split') != 'valid':
        raise ValueError('The released CelebA recipes evaluate the validation split')
    config = core.ExperimentConfig(**job['config'])
    if config.rounds < 1:
        raise ValueError('rounds must be positive')
    for name, expected in job.get('source_hashes', {}).items():
        path = (core.ROOT / name).resolve()
        if not path.is_relative_to(core.ROOT.resolve()):
            raise ValueError('Source reference is outside this repository')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source identity mismatch: {name}')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    device = core.choose_device(config.device)
    output.mkdir(parents=True)
    try:
        result = core.run_experiment(
            job['dataset'], job['distribution'], job['method'], job['attack'],
            config, 'release', device,
            progress_callback=lambda row: write_json(output/'progress.json', row),
            checkpoint_path=output/'model.pt')
        expected_rounds = list(range(1, config.rounds + 1))
        if [row['round'] for row in result['trajectory_metrics']] != expected_rounds:
            raise ValueError('Incomplete experiment trajectory')
        result['checkpoint_sha256'] = hashlib.sha256((output/'model.pt').read_bytes()).hexdigest()
        result['runtime'] = {'python': platform.python_version(), 'torch': core.torch.__version__}
        result['source_hashes'] = {name: hashlib.sha256((core.ROOT/name).read_bytes()).hexdigest()
                                   for name in ['scripts/run_experiment.py', 'scripts/reproduce_paper_tables.py',
                                                'src/data_loader.py', 'src/celeba_data.py']}
        write_json(output/'result.json', result)
        with (output/'raw_results.jsonl').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result) + '\n')
        return result
    except Exception as error:
        write_json(output/'failure.json', {'error': str(error), 'type': type(error).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    core.torch.set_num_threads(1)
    core.torch.set_num_interop_threads(1)
    result = run_job(json.loads(args.job.read_text(encoding='utf-8')), args.output)
    print(json.dumps(result['metrics']))


if __name__ == '__main__':
    main()
