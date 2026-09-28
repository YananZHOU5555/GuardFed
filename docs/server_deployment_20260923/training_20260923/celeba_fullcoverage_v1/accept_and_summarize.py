"""Read-only acceptance of frozen jobs; write a separate, explicitly partial/final summary."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import importlib.util
import json
import math
import statistics

ROOT = Path('/workspace/GuardFed-celeba-expanded')
STAGE = ROOT / 'results/revision_20260926/celeba_fullcoverage_v1'
OUT = ROOT / 'deployment/fullcoverage_acceptance'
spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/run_revision_ablation.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
manifest = json.loads((STAGE / 'manifest.json').read_text())
assert manifest['total_cell_count'] == 700
jobs = [json.loads(Path(p).read_text()) for p in manifest['jobs']]
assert len(jobs) == 644 and len(manifest['reused_jobs']) == 56
rows, missing, failures, accepted_new = [], [], [], []
seen = set()
for origin, cohort in [('new', jobs), ('reused', manifest['reused_jobs'])]:
    for job in cohort:
        key = (job['method'], job['distribution'], job['attack'], job['config']['seed'])
        assert key not in seen
        seen.add(key)
        path = Path(job['output'])
        if (path / 'failure.json').exists():
            failures.append(job['id'])
        result = runner.checked_result(job)
        if result is None:
            assert origin == 'new', ('Lost accepted reuse', job['id'])
            missing.append(job['id'])
            continue
        cfg = job['config']
        assert not (path / 'failure.json').exists(), ('Unresolved failure plus result', job['id'])
        assert result['rounds'] == cfg['rounds'] == 70
        assert result['seed'] == cfg['seed']
        assert result['alpha'] == cfg['client_alpha'] == manifest['distributions'][job['distribution']]
        assert all(result[k] == job[k] for k in ['dataset', 'method', 'distribution', 'attack'])
        assert result['config'] == cfg
        assert len(result['trajectory_metrics']) == 70
        assert [r['round'] for r in result['trajectory_metrics']] == list(range(1, 71))
        assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
        contract = result['data_contract']['image_data_contract']
        assert cfg['celeba_evaluation_split'] == contract['evaluation_split'] == 'valid'
        assert contract['actual_train_rows'] == 162770 and contract['actual_evaluation_rows'] == 19867
        assert result['evaluation_stats']['prediction_count'] == 19867
        assert all(math.isfinite(result['metrics'][k]) and 0 <= result['metrics'][k] <= 1 for k in ['accuracy', 'aeod', 'aspd'])
        row = dict(id=job['id'], origin=origin, method=job['method'], distribution=job['distribution'], attack=job['attack'], seed=cfg['seed'], candidate=job['tuning_candidate'], **result['metrics'], checkpoint_sha256=result['revision_job']['checkpoint_sha256'], torch_version=result['revision_job']['torch_version'], output=str(path))
        gap = max(row['aeod'], row['aspd'])
        row['score'] = row['accuracy'] - .35*(.45*row['aeod']+.45*row['aspd']+.10*gap)-.10*max(0,gap-.06)
        rows.append(row)
        if origin == 'new': accepted_new.append(job['id'])
assert len(seen) == 700
subsets = {}
for label, seeds in [('all_ten', manifest['seeds']), ('nonselection_nine', manifest['nonselection_seeds']), ('prospective_six', manifest['prospectively_unobserved_seeds'])]:
    grouped = collections.defaultdict(list)
    for row in rows:
        if row['seed'] in seeds:
            grouped[(row['method'], row['distribution'], row['attack'])].append(row)
    summaries = []
    for (method, distribution, attack), group in sorted(grouped.items()):
        summaries.append(dict(method=method, distribution=distribution, attack=attack, n=len(group), expected_n=len(seeds), complete=len(group)==len(seeds), seeds=sorted(r['seed'] for r in group), **{k:dict(mean=statistics.mean(r[k] for r in group), sample_sd=statistics.stdev(r[k] for r in group) if len(group)>1 else None) for k in ['accuracy','aeod','aspd','score']}))
    subsets[label] = summaries
record = dict(checked_utc=datetime.now(timezone.utc).isoformat(), complete=len(rows)==700 and not failures, accepted_new=len(accepted_new), reused_verified=56, accepted_total=len(rows), planned_total=700, new_total=644, missing_ids=missing, failed_ids=failures, accepted_new_ids=accepted_new, environments=dict(collections.Counter(r['torch_version'] for r in rows)), source_manifest_sha256=runner.digest(STAGE/'manifest.json'), all_conditions=rows, summaries=subsets, note='Validation cohort; selection seed and runtime differences disclosed. Partial groups are not complete ten-seed comparisons.')
OUT.mkdir(exist_ok=True)
(OUT/'acceptance_summary.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['checked_utc','complete','accepted_new','reused_verified','accepted_total','planned_total','failed_ids']}))
