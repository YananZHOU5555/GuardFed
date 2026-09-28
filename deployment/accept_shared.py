"""Accept frozen checkpoint-only calibration results; write independent final summaries."""
import collections
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

ROOT = Path('/workspace/GuardFed-celeba-expanded')
STAGE = ROOT / 'results/revision_20260928/celeba_shared_calibration_v1'
METRICS = ('accuracy', 'aeod', 'aspd')
VIEWS = ('raw', 'shared_calibration')
SUBSETS = {'all_ten': list(range(91001, 91011)),
           'nonselection_nine': list(range(91002, 91011)),
           'prospective_six': list(range(91005, 91011))}
DIST = ('IID', 'non-IID')
ATTACKS = ('Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA')
EPS = 1e-12


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def stats(values):
    require(len(values) >= 2 and all(math.isfinite(v) for v in values), 'Invalid statistical inputs')
    return {'n': len(values), 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}


def condition_key(row):
    return (row['method'], row['distribution'], row['attack'], row['seed'])


def main():
    require(not sys.flags.optimize, 'Do not disable assertions used by checked_output')
    manifest_path = STAGE / 'manifest.json'
    manifest = read(manifest_path)
    manifest_hash = digest(manifest_path)
    require(digest(STAGE / 'PROTOCOL.md') == manifest['protocol_sha256'], 'Protocol drift')
    for name, expected in manifest['evaluation_source_hashes'].items():
        require(digest(ROOT / name) == expected, 'Evaluation source drift: ' + name)
    for name, expected in manifest['data_hashes'].items():
        require(digest(ROOT / name) == expected, 'Data drift: ' + name)
    failures = list((STAGE / 'runs').glob('*/failure.json')) + list(STAGE.glob('shard_failure_*.json'))
    require(not failures, 'Preserved unresolved failures: ' + str(failures))
    jobs = manifest['jobs']
    require(manifest['total'] == len(jobs) == 700, 'Expected exactly 700 jobs')
    ids = {j['id'] for j in jobs}
    require(len(ids) == 700, 'Duplicate job IDs')
    available = {p.parent.name for p in (STAGE / 'runs').glob('*/evaluation.json')}
    require(available == ids, f'Evaluation ID mismatch; missing={sorted(ids-available)}, extra={sorted(available-ids)}')

    spec = importlib.util.spec_from_file_location('shared_evaluator', ROOT / 'deployment/evaluate_shared.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    import numpy as np
    import reproduce_paper_tables as core

    rows = []
    hashes = {}
    max_native_delta = 0.0
    for index, job in enumerate(jobs):
        result = evaluator.checked_output(job, manifest_hash, core, np)
        require(result is not None, 'Missing result: ' + job['id'])
        original = read(Path(job['output']) / 'result.json')
        prior_contract = original['data_contract']['image_data_contract']
        for key in ('root_image_ids_sha256', 'train_image_ids_sha256', 'evaluation_image_ids_sha256',
                    'cache_manifest_sha256', 'evaluation_split'):
            require(result['root_contract'][key] == prior_contract[key], 'Data identity mismatch: ' + job['id'] + '/' + key)
        require(result['root_contract']['evaluation_split'] == 'valid', 'Unexpected evaluation split')
        require(original['rounds'] == result['round'] == 70, 'Not final round70')
        require(original['data_contract']['train_rows'] == 162770 and original['data_contract']['test_rows'] == 19867, 'Sample count mismatch')
        require(result['torch_version'] == '2.11.0+cu128', 'Unexpected replay environment')
        runtime = original.get('runtime', {})
        original_torch = runtime.get('torch_version', original.get('torch_version'))
        # The source acceptance ledger is the authoritative runtime record if runtime
        # lives under a version-specific original result field.
        with np.load(STAGE / 'runs' / job['id'] / 'margins.npz') as cache:
            for split in ('root', 'valid'):
                labels = cache[split + '_y']
                sensitive = cache[split + '_sensitive']
                margins = cache[split + '_margins']
                require(labels.ndim == sensitive.ndim == margins.ndim == 1, 'Cache shape mismatch')
                require(len(labels) == len(sensitive) == len(margins), 'Cache row mismatch')
                require(np.isfinite(margins).all(), 'Nonfinite cached margins')
                require(set(np.unique(labels)) <= {0, 1} and set(np.unique(sensitive)) <= {0, 1}, 'Nonbinary cache metadata')
                for group in (0, 1):
                    counts = result[split + '_counts'][str(group)]
                    require(counts['n'] == int((sensitive == group).sum()), 'Group count mismatch')
                    require(counts['positives'] == int(((sensitive == group) & (labels == 1)).sum()), 'Positive count mismatch')
                    require(counts['n'] > 0 and counts['positives'] > 0, 'Undefined fairness denominator')
                if split == 'valid':
                    require(len(labels) == 19867, 'Incomplete validation set')
                    raw = core.compute_metrics(labels, (margins > 0).astype(int), sensitive)
                    shared = core.metrics_from_group_thresholds(labels, margins, sensitive,
                                {int(k): v for k, v in result['thresholds'].items()})
                    for view, recomputed in (('raw', raw), ('shared_calibration', shared)):
                        for metric in (*METRICS, 'positive_rate', 'majority_accuracy'):
                            require(math.isfinite(result[view][metric]) and abs(result[view][metric] - recomputed[metric]) <= EPS,
                                    'Metric/cache mismatch: ' + job['id'] + '/' + view + '/' + metric)
                        require(result[view]['prediction_count'] == 19867, 'Prediction count mismatch')
        reference_view = 'shared_calibration' if job['method'] == 'GuardFed-AD2+' else 'raw'
        for metric in (*METRICS, 'positive_rate', 'majority_accuracy'):
            require(math.isfinite(result['native'][metric]), 'Nonfinite native metric')
            require(abs(result['native'][metric] - result[reference_view][metric]) <= EPS,
                    'Native endpoint mismatch: ' + job['id'] + '/' + metric)
        for metric in METRICS:
            difference = result['native'][metric] - original['metrics'][metric]
            max_native_delta = max(max_native_delta, abs(difference))
            require(abs(difference) <= EPS and abs(result['native_replay_differences'][metric] - difference) <= EPS, 'Native replay gate mismatch')
        row = {key: result[key] for key in ('id', 'method', 'distribution', 'attack', 'seed', 'round', 'evaluation_split',
               'checkpoint_sha256', 'original_result_sha256', 'cache_sha256', 'raw', 'shared_calibration', 'native',
               'thresholds', 'root_counts', 'valid_counts', 'torch_version')}
        row['original_torch_version'] = original_torch
        row['native_replay_max_abs_difference'] = max(abs(result['native_replay_differences'][m]) for m in METRICS)
        hashes[job['id']] = digest(STAGE / 'runs' / job['id'] / 'evaluation.json')
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f'Checked {index + 1}/700', flush=True)

    # Preserve original training environments from the strictly accepted source cohort.
    source_path = ROOT / 'deployment/fullcoverage_acceptance/acceptance_summary.json'
    require(digest(source_path) == manifest['source_acceptance_sha256'], 'Source acceptance changed')
    source = read(source_path)
    accepted = {r['id']: r for r in source['all_conditions']}
    require(set(accepted) == ids, 'Source cohort ID mismatch')
    for row in rows:
        prior = accepted[row['id']]
        require(condition_key(row) == condition_key(prior) and row['checkpoint_sha256'] == prior['checkpoint_sha256'], 'Source cohort identity mismatch')
        row['original_torch_version'] = prior['torch_version']
    training_environments = dict(collections.Counter(r['original_torch_version'] for r in rows))
    require(training_environments == {'2.11.0+cu128': 686, '2.11.0+cu130': 14}, 'Original environment counts changed')
    methods = sorted({r['method'] for r in rows})
    require(len(methods) == 7, 'Expected seven methods')
    lookup = {condition_key(r): r for r in rows}
    require(len(lookup) == 700 and set(lookup) == set(itertools.product(methods, DIST, ATTACKS, SUBSETS['all_ten'])), 'Incomplete factorial cohort')
    per_condition = {}
    overall = {}
    paired = {}
    for subset, seeds in SUBSETS.items():
        per_condition[subset] = []
        overall[subset] = []
        paired[subset] = {}
        for method, dist, attack, view in itertools.product(methods, DIST, ATTACKS, VIEWS):
            selected = [lookup[(method, dist, attack, seed)][view] for seed in seeds]
            per_condition[subset].append({'method': method, 'distribution': dist, 'attack': attack,
                'view': view, 'n': len(seeds), 'seeds': seeds,
                **{metric: stats([r[metric] for r in selected]) for metric in (*METRICS, 'positive_rate')}})
        seed_means = {}
        for method, view in itertools.product(methods, VIEWS):
            seed_means[(method, view)] = {seed: {metric: statistics.mean(lookup[(method, dist, attack, seed)][view][metric]
                for dist in DIST for attack in ATTACKS) for metric in METRICS} for seed in seeds}
            overall[subset].append({'method': method, 'view': view, 'n': len(seeds), 'seeds': seeds,
                **{metric: stats([seed_means[(method, view)][seed][metric] for seed in seeds]) for metric in METRICS}})
        for view in VIEWS:
            paired[subset][view] = {}
            for metric in METRICS:
                differences = [seed_means[('GuardFed-AD2+', view)][seed][metric] - seed_means[('FLTrust', view)][seed][metric] for seed in seeds]
                paired[subset][view][metric] = {**stats(differences), 'seed_differences': dict(zip(seeds, differences)),
                    'guardfed_better_seeds': sum(d > 0 if metric == 'accuracy' else d < 0 for d in differences),
                    'ties': sum(d == 0 for d in differences)}
    degeneracy = []
    for method, view in itertools.product(methods, VIEWS):
        selected = [r for r in rows if r['method'] == method]
        degeneracy.append({'method': method, 'view': view, 'records': len(selected),
            'constant_prediction_ids': [r['id'] for r in selected if r[view]['positive_rate'] in (0, 1)],
            'both_zero_gap_ids': [r['id'] for r in selected if r[view]['aeod'] == 0 and r[view]['aspd'] == 0],
            'positive_rate_min': min(r[view]['positive_rate'] for r in selected),
            'positive_rate_max': max(r[view]['positive_rate'] for r in selected)})
    require(digest(manifest_path) == manifest_hash, 'Manifest changed during acceptance')
    require(not list((STAGE / 'runs').glob('*/failure.json')) and not list(STAGE.glob('shard_failure_*.json')), 'Failure appeared during acceptance')
    summary = {'complete': True, 'checked_utc': datetime.now(timezone.utc).isoformat(), 'records': 700, 'outcomes': 1400,
        'manifest_sha256': manifest_hash, 'protocol_sha256': manifest['protocol_sha256'],
        'source_acceptance_sha256': manifest['source_acceptance_sha256'], 'original_training_environments': training_environments,
        'replay_environment': '2.11.0+cu128', 'native_replay_max_abs_difference': max_native_delta,
        'native_endpoints_match': True, 'failure_count': 0, 'calibration': manifest['calibration'],
        'per_condition': per_condition, 'overall_seed_first': overall, 'paired_guardfed_minus_fltrust': paired,
        'degeneracy': degeneracy, 'evaluation_json_sha256': hashes,
        'limitations': ['Validation only, not untouched test', 'seed91001 selected recipes; first four seeds previously observed',
            'Fourteen original cu130 checkpoints replayed in cu128; all native metrics matched; does not prove training equivalence',
            'FairFed/FairGuard/FLTrust+FairGuard are project adaptations', 'AEOD is absolute TPR gap, not full equalized odds',
            'This is a checkpoint-level common calibration control, not a retrained common algorithm or a significance test',
            'Scenes averaged within seed before aggregate statistics; sample standard deviation ddof1']}
    lines = ['# CelebA 共享校准：700模型终轮评价', '',
        '已验收700个原始终轮模型、1400组raw/shared结果；所有原生指标复现通过。只评价既有模型，无新增训练。', '',
        '共享校准统一采用训练root拟合：acc_floor、最大准确率下降0.005、budget0.06、41分位点。valid仅用于评价。', '',
        '下表先对每个种子的两分布×五场景取均值，再跨10种子计算均值±样本标准差；ACC为百分比，AEOD/ASPD为0–1。', '',
        '| 方法 | 评价口径 | ACC (%) | AEOD | ASPD |', '|---|---|---:|---:|---:|']
    for row in overall['all_ten']:
        lines.append(f"| {row['method']} | {row['view']} | {100*row['accuracy']['mean']:.3f} ± {100*row['accuracy']['sample_sd']:.3f} | {row['aeod']['mean']:.5f} ± {row['aeod']['sample_sd']:.5f} | {row['aspd']['mean']:.5f} ± {row['aspd']['sample_sd']:.5f} |")
    lines += ['', '## GuardFed−FLTrust共享种子配对差异', '',
        'ACC差值正向有利，公平差距负向有利；±为配对差的样本标准差，不作显著性结论。', '',
        '| 子集 | 口径 | ΔACC (百分点) | ΔAEOD | ΔASPD | ACC/AEOD/ASPD胜出种子数 |', '|---|---|---:|---:|---:|---|']
    for subset, seeds in SUBSETS.items():
        for view in VIEWS:
            p = paired[subset][view]
            lines.append(f"| {subset} (n={len(seeds)}) | {view} | {100*p['accuracy']['mean']:+.3f} ± {100*p['accuracy']['sample_sd']:.3f} | {p['aeod']['mean']:+.5f} ± {p['aeod']['sample_sd']:.5f} | {p['aspd']['mean']:+.5f} ± {p['aspd']['sample_sd']:.5f} | {p['accuracy']['guardfed_better_seeds']}/{p['aeod']['guardfed_better_seeds']}/{p['aspd']['guardfed_better_seeds']} |")
    lines += ['', '所有单次预测率、零差距、恒定预测及负结果均保留。原生GuardFed等于shared端点，其他六方法原生等于raw端点。不得将原生表直接当共同校准后的聚合能力比较。', '',
        '14个旧cu130模型目前在cu128复现原生指标一致；这只证明当前指标复现，不证明原训练轨迹一致。seed91001参与配置选择，另列排除该种子的9种子及此前未观察的6种子。', '',
        '此结果为valid-only，三种基线仍为项目适配；尚不替代其余基线可信实现、训练机制消融和最后冻结评价。AEOD表示绝对TPR差距。', '']
    final = STAGE / 'final'
    save(final / 'accepted.json', {'manifest_sha256': manifest_hash, 'records': rows})
    summary['accepted_sha256'] = digest(final / 'accepted.json')
    save(final / 'summary.json', summary)
    temporary = final / 'analysis.md.tmp'
    temporary.write_text('\n'.join(lines))
    temporary.replace(final / 'analysis.md')
    print(json.dumps({'complete': True, 'accepted': 700, 'max_native_delta': max_native_delta,
                      'summary': str(final / 'summary.json'), 'paired_ten': paired['all_ten']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
