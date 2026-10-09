"""Verify frozen results and rank all validation recipes without training."""
import collections
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/GuardFed-celeba-expanded')
OUT = ROOT / 'deployment/expanded_final'
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('frozen_runner', ROOT / 'scripts/run_revision_ablation.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
manifests = [
    ROOT / 'results/revision_20260924/celeba_expanded_screen_v2/manifest.json',
    Path('/workspace/GuardFed-celeba-tuning/results/revision_20260924/celeba_tuning_screen_v1/manifest.json'),
]
all_rows, jobs, seen, hashes = [], {}, set(), {}
for manifest_path, expected_count in zip(manifests, [82, 40]):
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['jobs']) == expected_count
    for name, expected in manifest['source_hashes'].items():
        actual = hashes.setdefault(name, runner.digest(ROOT / name))
        assert actual == expected, name
    for job_path in manifest['jobs']:
        job = json.loads(Path(job_path).read_text())
        result = runner.checked_result(job)
        assert result is not None, job['id']
        assert not (Path(job['output']) / 'failure.json').exists()
        cfg = job['config']
        assert cfg['rounds'] == result['rounds'] == 70
        assert cfg['seed'] == result['seed'] == 91001
        assert cfg['celeba_evaluation_split'] == 'valid'
        assert result['data_contract']['image_data_contract']['evaluation_split'] == 'valid'
        assert result['data_contract']['train_rows'] == 162770
        assert result['evaluation_stats']['prediction_count'] == 19867
        assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
        assert result['revision_job']['tuning_candidate'] == job['tuning_candidate']
        candidate = 'lr' + str(cfg['learning_rate'])
        if job['method'] == 'GuardFed-AD2+':
            candidate += '_drop' + format(cfg['ad2_calibration_max_acc_drop'], 'g')
        assert candidate == job['tuning_candidate'], (candidate, job['tuning_candidate'])
        key = (job['method'], candidate, job['attack'], cfg['seed'])
        assert key not in seen, ('duplicate condition', key)
        seen.add(key)
        metrics = result['metrics']
        assert all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in ['accuracy', 'aeod', 'aspd'])
        row = dict(id=job['id'], method=job['method'], candidate=candidate,
                   attack=job['attack'], seed=91001, **metrics,
                   checkpoint_sha256=result['revision_job']['checkpoint_sha256'],
                   torch_version=result['revision_job']['torch_version'],
                   output=job['output'], config=cfg)
        all_rows.append(row)
        jobs[job['id']] = job

def score(row):
    gap = max(row['aeod'], row['aspd'])
    return row['accuracy'] - .35 * (.45 * row['aeod'] + .45 * row['aspd'] + .10 * gap) - .10 * max(0, gap - .06)

groups = collections.defaultdict(list)
for row in all_rows:
    groups[row['method'], row['candidate']].append(row)
ranking = []
for (method, candidate), conditions in groups.items():
    assert len(conditions) == 2 and {x['attack'] for x in conditions} == {'Benign', 'S-DFA'}
    assert conditions[0]['seed'] == conditions[1]['seed'] == 91001
    c0, c1 = [dict(x['config']) for x in conditions]
    c0.pop('experiment_tag'); c1.pop('experiment_tag')
    assert c0 == c1, (method, candidate, 'cross-condition config mismatch')
    row = dict(method=method, candidate=candidate,
               **{k: sum(x[k] for x in conditions) / 2 for k in ['accuracy', 'aeod', 'aspd']},
               score=sum(score(x) for x in conditions) / 2,
               conditions=conditions,
               degenerate_zero_gap=any(x['accuracy'] < .6 and x['aeod'] == x['aspd'] == 0 for x in conditions))
    ranking.append(row)

def dominates(a, b):
    return a['accuracy'] >= b['accuracy'] and a['aeod'] <= b['aeod'] and a['aspd'] <= b['aspd'] and any(a[k] != b[k] for k in ['accuracy', 'aeod', 'aspd'])

ranking.sort(key=lambda x: (-x['score'], x['method'], x['candidate']))
for row in ranking:
    row['pareto'] = not any(dominates(other, row) for other in ranking)
methods = sorted({r['method'] for r in ranking})
champions = [max((r for r in ranking if r['method'] == m), key=lambda x: x['accuracy']) for m in methods]
score_winners = [next(r for r in ranking if r['method'] == m) for m in methods]
guard = next(r for r in score_winners if r['method'] == 'GuardFed-AD2+')
comparisons = [dict(method=r['method'], guard_dominates=dominates(guard, r),
                    accuracy_difference=guard['accuracy']-r['accuracy'],
                    aeod_difference=guard['aeod']-r['aeod'], aspd_difference=guard['aspd']-r['aspd'])
               for r in score_winners if r['method'] != 'GuardFed-AD2+']
assert len(all_rows) == 122 and len(ranking) == 61 and len(methods) == 7
payload = dict(ranking=ranking, score_winners=score_winners, accuracy_champions=champions,
               pareto=[r for r in ranking if r['pareto']], all_conditions=all_rows,
               guard_comparison_to_score_winners=comparisons,
               joint_dominance_over_all_baseline_candidates=all(dominates(guard, r) for r in ranking if r['method'] != 'GuardFed-AD2+'),
               n_per_condition=1, seed=91001, evaluation_split='valid',
               note='Condition means, not seed means; no sample std or significance. Prior test exposure disclosed. cu130/cu128 runtime split retained. FairFed/FairGuard/hybrid are project adaptations.')
(OUT/'ranking.json').write_text(json.dumps(payload, indent=2))
verification = dict(checked_utc=datetime.now(timezone.utc).isoformat(), expanded_accepted=82,
                    prior_tuning_accepted=40, unique_runs=122, complete_recipes=61,
                    current_failed=0, historical_failed_attempts=3, active=0,
                    source_data_hashes=hashes, checkpoint_config_round70_split_seed_candidate_checks=True,
                    environments=dict(collections.Counter(r['torch_version'] for r in all_rows)),
                    accepted_ids=[r['id'] for r in all_rows],
                    scientific_limitations=payload['note'])
(OUT/'verification.json').write_text(json.dumps(verification, indent=2))
lines = ['# CelebA expanded validation stage complete', '',
         '82 new +40 previous =122 unique runs,61 recipes,7 methods. Every run accepted at70rounds; n=1, seed91001, official valid only.', '',
         'Ranking averages frozen per-condition scores across Benign/S-DFA; metrics below are condition means. All candidates and negative results retained in ranking.json.', '',
         '|Method|Score-winning recipe|ACC|AEOD|ASPD|Score|', '|---|---|---:|---:|---:|---:|']
for row in sorted(score_winners, key=lambda x: -x['score']):
    lines.append(f"|{row['method']}|{row['candidate']}|{row['accuracy']:.6f}|{row['aeod']:.6f}|{row['aspd']:.6f}|{row['score']:.6f}|")
lines += ['', '## Interpretation and next authorized boundary', '', payload['note'],
          'Full82 acceptance is not completion of all baseline integration. Official FedAA/LoGoFair/Fed-NGA end-to-end work remains. No new training/test confirmation launched.',
          'Next: complete adapter fidelity/integration and isolated canaries; then freeze bounded validation searches. For the7 current methods, a concrete follow-up proposal is score-winning recipes ×development seeds91002/91003/91004 ×Benign/S-DFA =42 runs; not launched or represented as authorized formal confirmation.',
          'Three old CUDA failure attempts are preserved. Eight resumed jobs used the new cu128 host after4 exact first-round migration checks; this does not prove70round environment equivalence.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(verified=verification, winners=[{k:v for k,v in r.items() if k not in ['conditions']} for r in score_winners], comparisons=comparisons), indent=2))
