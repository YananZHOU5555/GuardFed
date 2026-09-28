"""Freeze 700 validation cells (644 new, 56 strictly checked reuse); never launch training."""
import ast
import copy
import importlib.util
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/GuardFed-celeba-expanded')
OUT = ROOT / 'results/revision_20260926/celeba_fullcoverage_v1'
PROTOCOL = ROOT / 'deployment/CELEBA_FULLCOVERAGE_PROTOCOL.md'
PREVIOUS = ROOT / 'results/revision_20260925/celeba_seedcheck_v1/manifest.json'
DISTRIBUTIONS = {'IID': 5000.0, 'non-IID': 5.0}
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
SEEDS = list(range(91001, 91011))
METHODS = ['FedAvg', 'FairFed', 'Median', 'FLTrust', 'FairGuard',
           'FLTrust+FairGuard', 'GuardFed-AD2+']
CONFIG_AXES = {'seed', 'client_alpha', 'experiment_suite', 'experiment_tag'}
RUNTIME_KEYS = {'visible_gpu', 'cpu_threads', 'torch_version', 'python_version',
                'checkpoint_sha256', 'finished_unix'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def recipe(config):
    return {k: v for k, v in config.items() if k not in CONFIG_AXES}


def key(job):
    return job['method'], job['distribution'], job['attack'], job['config']['seed']


def main():
    assert PROTOCOL.is_file() and PROTOCOL.stat().st_size > 500, 'Upload and freeze PROTOCOL first'
    assert not OUT.exists(), 'Refuse to overwrite a frozen or partial stage'
    spec = importlib.util.spec_from_file_location('frozen_runner', ROOT / 'scripts/run_revision_ablation.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    digest = runner.digest
    prior = read(PREVIOUS)
    frozen = dict(prior['source_hashes'])
    for name, expected in frozen.items():
        assert digest(ROOT / name) == expected, ('frozen source/data changed', name)

    # Inspect only literal declarations and the small attack router; do not import torch/core.
    tree = ast.parse((ROOT / 'scripts/reproduce_paper_tables.py').read_text())
    distribution_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'DISTRIBUTIONS' for t in n.targets))
    assert ast.literal_eval(distribution_node.value) == DISTRIBUTIONS
    router = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'attack_types_for_client')
    router.returns = None
    for arg in router.args.args:
        arg.annotation = None
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[router], type_ignores=[])), '<attack-router>', 'exec'), scope)
    for attack in ATTACKS:
        assignments = [scope['attack_types_for_client'](attack, cid, list(range(4))) for cid in range(20)]
        assert all(not a for a in assignments[4:])
        expected = {'Benign': [[], [], [], []], 'F Flip': [['fflip']] * 4,
                    'FedSA': [['foe']] * 4, 'S-DFA': [['fflip', 'foe']] * 4,
                    'Sp-DFA': [['fflip'], ['fflip'], ['foe'], ['foe']]}
        assert assignments[:4] == expected[attack], attack

    old_jobs = [read(Path(p)) for p in prior['jobs']] + prior['reused_jobs']
    assert len(old_jobs) == 56
    reused, reuse_audit = {}, []
    for job in old_jobs:
        result = runner.checked_result(job)
        assert result is not None, ('missing reuse result', job['id'])
        assert not (Path(job['output']) / 'failure.json').exists()
        assert key(job) not in reused, ('duplicate reuse', key(job))
        assert job['dataset'] == 'celeba' and job['distribution'] == 'non-IID'
        assert job['attack'] in ['Benign', 'S-DFA'] and job['method'] in METHODS
        cfg = job['config']
        assert cfg['seed'] in SEEDS[:4] and cfg['rounds'] == 70 and cfg['client_alpha'] == 5.0
        assert cfg['celeba_evaluation_split'] == 'valid'
        assert cfg['celeba_train_limit'] == cfg['celeba_eval_limit'] == 0
        assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
        assert result['alpha'] == 5.0 and result['seed'] == cfg['seed']
        for field in ['dataset', 'distribution', 'method', 'attack']:
            assert result[field] == job[field]
        contract = result['data_contract']['image_data_contract']
        assert contract['evaluation_split'] == 'valid'
        assert contract['actual_train_rows'] == 162770 and contract['actual_evaluation_rows'] == 19867
        for name, expected_hash in job['source_hashes'].items():
            if name in frozen:
                assert expected_hash == frozen[name], ('reuse source/data mismatch', name, job['id'])
        reused[key(job)] = copy.deepcopy(job)
        reuse_audit.append(dict(cell=list(key(job)), job_id=job['id'], output=job['output'],
                               checkpoint_sha256=result['revision_job']['checkpoint_sha256'],
                               result_sha256=digest(Path(job['output']) / 'result.json'),
                               torch_version=result['revision_job']['torch_version']))
    templates = {m: reused[(m, 'non-IID', 'Benign', 91001)] for m in METHODS}
    for job in old_jobs:
        assert recipe(job['config']) == recipe(templates[job['method']]['config']), ('recipe drift', job['id'])
        assert job['tuning_candidate'] == templates[job['method']]['tuning_candidate']
    for path in [Path(__file__).resolve(), PROTOCOL, PREVIOUS]:
        frozen[str(path.relative_to(ROOT))] = digest(path)

    jobs, coverage = [], []
    for seed, distribution, attack, method in itertools.product(SEEDS, DISTRIBUTIONS, ATTACKS, METHODS):
        cell = (method, distribution, attack, seed)
        if cell in reused:
            job = reused[cell]
            coverage.append(dict(cell=list(cell), status='reused_verified', job_id=job['id'], output=job['output']))
            continue
        template = templates[method]
        job = {k: copy.deepcopy(v) for k, v in template.items() if k not in RUNTIME_KEYS}
        job_id = f"{method}_{distribution}_{attack.replace(' ', '-')}_seed{seed}"
        job.update(id=job_id, distribution=distribution, attack=attack,
                   output=str(OUT / 'runs' / job_id), source_hashes=frozen,
                   evidence_stage='fixed_recipe_validation_full_coverage',
                   source_selection_job_id=template['id'])
        job['config'].update(seed=seed, client_alpha=DISTRIBUTIONS[distribution],
                             experiment_suite='celeba_fullcoverage_v1', experiment_tag=job_id)
        assert recipe(job['config']) == recipe(template['config'])
        changed = {k for k in template['config'] if template['config'][k] != job['config'][k]}
        assert changed <= CONFIG_AXES
        jobs.append(job)
        coverage.append(dict(cell=list(cell), status='pending_new', job_id=job_id, output=job['output']))
    expected = set(itertools.product(METHODS, DISTRIBUTIONS, ATTACKS, SEEDS))
    assert {tuple(c['cell']) for c in coverage} == expected and len(coverage) == 700
    assert len(jobs) == 644 and len({j['id'] for j in jobs}) == 644
    assert set(map(key, jobs)).isdisjoint(reused)
    assert all(n == 10 for n in Counter(tuple(c['cell'][:3]) for c in coverage).values())
    # Do not leave any partial queue until all reuse/source/config gates have passed.
    (OUT / 'jobs').mkdir(parents=True)
    paths, job_hashes = [], {}
    for job in jobs:
        path = OUT / 'jobs' / (job['id'] + '.json')
        write(path, job)
        paths.append(str(path))
        job_hashes[str(path)] = digest(path)
    manifest = dict(protocol='celeba_fullcoverage_v1', stage='fixed_recipe_validation_full_coverage',
                    created_utc=datetime.now(timezone.utc).isoformat(), jobs=paths, output=str(OUT),
                    new_run_count=644, reused_full_count=56, total_cell_count=700,
                    reused_jobs=list(reused.values()), reuse_audit=reuse_audit,
                    source_hashes=frozen, job_sha256s=job_hashes, concurrency=8,
                    methods=METHODS, distributions=DISTRIBUTIONS, attacks=ATTACKS, seeds=SEEDS,
                    attack_display_labels={'F Flip': 'F-Flip'}, evaluation_split='valid',
                    recipe_selection_seed=91001, nonselection_seeds=SEEDS[1:],
                    prospectively_unobserved_seeds=SEEDS[4:],
                    selected_recipes=[dict(method=m, candidate=templates[m]['tuning_candidate']) for m in METHODS],
                    limitations=['Fixed non-IID Benign/S-DFA-selected recipes transferred to every cell; no IID retuning.',
                                 'Validation coverage, not untouched-test confirmation.',
                                 'Reused seed91001 used for recipe selection; report seeds91002..91010 separately.',
                                 'Reused14 on cu130; reused42 and new644 on cu128. First-round migration canaries do not prove full-run equivalence.',
                                 'Runner summaries cover only644 new jobs; final700 aggregation must explicitly merge reused_jobs.',
                                 'FairFed/FairGuard/hybrid are project adaptations; three official adapters remain separate.'])
    write(OUT / 'manifest.json', manifest)
    write(OUT / 'coverage_audit.json', dict(cells=coverage, new_runs=644, reused_runs=56,
                                           reuse_audit=reuse_audit, complete_matrix=True,
                                           manifest_sha256=digest(OUT / 'manifest.json')))
    print(json.dumps(dict(manifest=str(OUT / 'manifest.json'), new_runs=644, reused_runs=56,
                          total_cells=700, manifest_sha256=digest(OUT / 'manifest.json'))))


if __name__ == '__main__':
    main()
