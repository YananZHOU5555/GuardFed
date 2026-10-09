"""Prepare missing-only CelebA mechanism jobs; this command never dispatches."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from adapter import VARIANTS

STAGE = 'celeba_mechanism_v1'
REMOTE = PurePosixPath('/workspace/GuardFed-celeba-expanded')
BASE = REMOTE / 'deployment/celeba_mechanism_20261009'
DEST = REMOTE / 'results/revision_20261009' / STAGE
ATTACKS = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
DISTRIBUTIONS = {'IID': 5000.0, 'non-IID': 5.0}
SEEDS = list(range(91001, 91011))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(project, output):
    old = project / 'docs/server_deployment_20260923/training_20260923/celeba_fullcoverage_v1/manifest.json'
    accepted_path = project / 'outputs/guardfed_tables/celeba_nine_method_final_20261004/source_stageA700_snapshot.json'
    previous = json.loads(old.read_text(encoding='utf-8-sig'))
    accepted = json.loads(accepted_path.read_text(encoding='utf-8-sig'))
    assert accepted['complete'] and accepted['accepted_total'] == 700
    full = [r for r in accepted['all_conditions'] if r['method'] == 'GuardFed-AD2+']
    expected = {(d, a, s) for d in DISTRIBUTIONS for a in ATTACKS for s in SEEDS}
    assert len(full) == 100 and {(r['distribution'], r['attack'], r['seed']) for r in full} == expected
    template = next(j for j in previous['reused_jobs'] if j['method'] == 'GuardFed-AD2+')
    assert template['tuning_candidate'] == 'lr0.0005_drop0.005'
    files = [Path(__file__), Path(__file__).with_name('adapter.py'), Path(__file__).with_name('worker.py'),
             Path(__file__).with_name('runner.py')]
    adapters = {str(BASE / f.name): digest(f) for f in files}
    protocol = output / 'PROTOCOL.md'
    assert protocol.is_file()
    jobs = []
    gates = []
    for variant in VARIANTS:
        for dist, alpha in DISTRIBUTIONS.items():
            for attack in ATTACKS:
                for seed in SEEDS:
                    if variant == 'Full':
                        continue
                    identity = f'{variant}_{dist}_{attack}_seed{seed}'
                    cfg = copy.deepcopy(template['config'])
                    cfg.update(seed=seed, client_alpha=alpha, ablation_component=variant[-1] if variant.startswith('minus_') else 'none',
                               experiment_suite=STAGE, experiment_tag=identity, full_round_diagnostics=True)
                    job = {'id': identity, 'dataset': 'celeba', 'distribution': dist, 'attack': attack,
                           'method': 'GuardFed-AD2+', 'variant': variant, 'config': cfg,
                           'output': str(DEST / 'runs' / identity), 'source_hashes': previous['source_hashes'],
                           'adapter_hashes': adapters, 'tuning_candidate': template['tuning_candidate'],
                           'evidence_stage': 'fixed_recipe_mechanism_validation', 'protocol_sha256': digest(protocol)}
                    path = output / 'jobs' / f'{identity}.json'
                    save(path, job)
                    jobs.append({'id': identity, 'variant': variant, 'job': str(DEST / 'jobs' / path.name),
                                 'job_sha256': digest(path), 'output': job['output']})
            # A same-horizon 3-round image gate is required for every variant/distribution.
            cfg = copy.deepcopy(template['config'])
            cfg.update(seed=91001, client_alpha=alpha, rounds=3,
                       ablation_component=variant[-1] if variant.startswith('minus_') else 'none', full_round_diagnostics=True,
                       experiment_suite=STAGE + '_pipeline_only', experiment_tag=f'{variant}_{dist}_gate3')
            identity = f'{variant}_{dist}_S-DFA_seed91001_gate3'
            job = {'id': identity, 'dataset': 'celeba', 'distribution': dist, 'attack': 'S-DFA',
                   'method': 'GuardFed-AD2+', 'variant': variant, 'config': cfg,
                   'output': str(DEST / 'preflight/runs' / identity), 'source_hashes': previous['source_hashes'],
                   'adapter_hashes': adapters, 'tuning_candidate': template['tuning_candidate'],
                   'evidence_stage': 'pipeline_canary_only', 'protocol_sha256': digest(protocol)}
            path = output / 'preflight/jobs' / f'{identity}.json'; save(path, job)
            gates.append({'id': identity, 'variant': variant, 'job': str(DEST / 'preflight/jobs' / path.name),
                          'job_sha256': digest(path), 'output': job['output']})
    assert len(jobs) == 800 and len(gates) == 18
    assert len({x['id'] for x in jobs + gates}) == 818
    references = []
    for gate in gates:
        if gate['variant'] != 'Full':
            continue
        identity = gate['id'] + '_unchanged_reference'
        payload = json.loads((output / 'preflight/jobs' / (gate['id'] + '.json')).read_text())
        payload.update(id=identity, output=str(DEST / 'preflight/runs' / identity), reference_to=gate['id'])
        path = output / 'preflight/jobs' / (identity + '.json'); save(path, payload)
        references.append({'id': identity, 'reference_to': gate['id'],
                           'job': str(DEST / 'preflight/jobs' / path.name),
                           'job_sha256': digest(path), 'output': payload['output']})
    assert len(references) == 2
    manifest = {'stage': STAGE, 'status': 'PREPARED_PENDING_SERVER_AND_REAL_IMAGE_GATES',
                'authorized_objective': 'Complete rebuttal experiments; user goal20261009',
                'planned_new': 800, 'planned_reused': 100, 'total_model_records': 900,
                'variants': list(VARIANTS), 'distributions': DISTRIBUTIONS, 'attacks': ATTACKS,
                'seeds': SEEDS, 'rounds': 70, 'concurrency': 8, 'evaluation_split': 'valid',
                'jobs': jobs, 'preflight_jobs': gates, 'reference_jobs': references, 'reused_full': full,
                'base_manifest_sha256': digest(old), 'accepted_snapshot_sha256': digest(accepted_path),
                'protocol_sha256': digest(protocol), 'source_hashes': previous['source_hashes'],
                'adapter_hashes': adapters, 'server_data_hashes_verified_now': False,
                'real_image_gates_passed': False, 'new_training_started': False,
                'dispatch_condition': 'Live source/data/config/reused-checkpoint checks,18 image gates and Full exact same-horizon regression; freeze separate dispatch receipt before queue',
                'negative_results_retained': True, 'same_seed_and_terminal_checkpoint_for_all_metrics': True}
    save(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); m = prepare(args.project.resolve(), args.output.resolve())
    print(json.dumps({k: m[k] for k in ['status', 'planned_new', 'planned_reused', 'total_model_records', 'new_training_started']}))
