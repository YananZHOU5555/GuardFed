#!/usr/bin/env python3
"""Independent pre-attack tabular allocation audit; never run training.

Replays sensitive-group counts and cross-checks original archived joint audits.
The latter are recovered measurements, not newly inferred client label counts.
"""
import ast
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
SEEDS = [123, 456, 789, 1001, 2024, 3141, 4242, 5050, 6060, 7070]
ALPHAS = [1.0, 0.5, 0.1]
CELLS = ['0|0', '0|1', '1|0', '1|1']
MANIFEST_SHAS = {
    'adult': 'c505319330aaff636f5cc7773fd5ebbd9eae148d82aac0f627d44efa0406e984',
    'compas': '163a53cfdf81e7c05951628b0ae9a45264d6fdafc4e9ef88cc0df290cc5ad6bb',
}


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, message):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12, message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def reconstruct_sensitive_counts(lengths, alpha, seed, clients=20):
    rng = np.random.default_rng(seed)
    counts = {}
    for group in [1, 0]:
        indices = np.arange(lengths[group], dtype=np.int64)
        rng.shuffle(indices)
        cuts = (np.cumsum(rng.dirichlet([alpha] * clients)) * len(indices)).astype(int)[:-1]
        counts[group] = np.diff(np.concatenate([[0], cuts, [len(indices)]])).astype(int).tolist()
    return counts


def shuffle_independence(length, seed):
    one, two = np.random.default_rng(seed), np.random.default_rng(seed)
    a = np.arange(length, dtype=np.int64); b = a * 7 + 31
    one.shuffle(a); two.shuffle(b)
    require(one.bit_generator.state == two.bit_generator.state, 'Shuffle content changed RNG state')
    require(np.array_equal(b, a * 7 + 31), 'Shuffle content changed swap pattern')


def stats(values):
    a = np.asarray(values, dtype=float)
    require(len(a) == 10, 'Expected ten independent shared seeds per summary')
    constant = bool(np.all(a == a[0]))
    return {'n_seeds': 10, 'mean': float(a[0] if constant else a.mean()), 'sample_sd_ddof1': 0.0 if constant else float(a.std(ddof=1)),
            'min': float(a.min()), 'max': float(a.max())}


def check_sources(manifests):
    receipts = read(OUT / 'frozen_source_receipts.json')
    excerpts, records, functions = [], [], {}
    for record in receipts:
        dataset, name, path = record['dataset'], record['git_path'], Path(record['output'])
        require(file_sha(path) == record['sha256'] == manifests[dataset]['source_hashes'][name], 'Frozen source SHA mismatch')
        text = path.read_text(encoding='utf-8'); tree = ast.parse(text)
        if name == 'src/data_loader.py':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DatasetLoader')
            fn = '_load_adult' if dataset == 'adult' else '_load_compas'
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn)
            segment = ast.get_source_segment(text, node)
            records.append({'dataset': dataset, 'function': 'DatasetLoader.' + fn, 'line': node.lineno, 'end_line': node.end_lineno,
                            'source_sha256': hashlib.sha256(segment.encode()).hexdigest(),
                            'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
            excerpts.append(f'# {dataset} {name}:{node.lineno}, SHA {record["sha256"]}\n{segment}')
        if name != 'scripts/reproduce_paper_tables.py':
            continue
        nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        wanted = ['create_client_data_dict', 'load_bundle', 'sample_server_dataframe', 'client_runtime_data',
                  'run_experiment', 'attack_types_for_client', 'train_local_model']
        for fn in wanted:
            node = nodes[fn]; segment = ast.get_source_segment(text, node)
            records.append({'dataset': dataset, 'function': fn, 'line': node.lineno, 'end_line': node.end_lineno,
                            'source_sha256': hashlib.sha256(segment.encode()).hexdigest(),
                            'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
            excerpts.append(f'# {dataset} {name}:{node.lineno}, SHA {record["sha256"]}\n{segment}')
        create = nodes['create_client_data_dict']
        loop = next(n for n in create.body if isinstance(n, ast.For))
        require(ast.literal_eval(loop.iter) == [1, 0], 'Changed sensitive-group order')
        calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(create) if isinstance(n, ast.Call)]
        require(next(line for line, n in calls if n == 'rng.shuffle') < next(line for line, n in calls if n == 'rng.dirichlet'), 'Changed RNG ordering')
        run_calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(nodes['run_experiment']) if isinstance(n, ast.Call)]
        require(next(line for line, n in run_calls if n == 'load_bundle') < next(line for line, n in run_calls if n == 'client_runtime_data'), 'Attack precedes data partition')
        for fn in ['load_bundle', 'create_client_data_dict', 'sample_server_dataframe']:
            args = {a.arg for a in nodes[fn].args.args}
            attrs = {n.attr for n in ast.walk(nodes[fn]) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'config'}
            require('attack' not in args and not attrs.intersection({'experiment_tag', 'fflip_mode', 'foe_mode', 'sdfa_foe_mode', 'spdfa_foe_mode'}), 'Partition code reads attack')
        # Execute only the pinned allocation function on explicitly fabricated rows.
        # This tests counts; these toy labels are NEVER reported as original labels.
        namespace = {'np': np, 'pd': pd, 'torch': torch, 'Any': Any, 'Dict': Dict, 'List': List}
        exec(compile(ast.Module(body=[create], type_ignores=[]), str(path), 'exec'), namespace)
        functions[dataset] = namespace['create_client_data_dict']
    (OUT / 'frozen_source_function_excerpts.txt').write_text('\n\n'.join(excerpts) + '\n', encoding='utf-8')
    return receipts, records, functions


def main():
    torch.set_num_threads(1)
    restoration = read(OUT / 'restoration_receipt.json')
    require(file_sha(Path(restoration['archive'])) == restoration['archive_sha256'], 'Restoration archive changed')
    for member in restoration['members']:
        if member['output']:
            p = Path(member['output'])
            require(file_sha(p) == member['sha256'] and p.stat().st_size == member['bytes'], 'Restored member byte/SHA mismatch')
    manifests = {d: read(OUT / 'manifests' / (d + '.json')) for d in ['adult', 'compas']}
    for d, manifest in manifests.items():
        require(file_sha(OUT / 'manifests' / (d + '.json')) == MANIFEST_SHAS[d], 'Original acceptance manifest mismatch')
        require(manifest['seeds'] == SEEDS, 'Changed seed list')
    source_receipts, source_ast, frozen_functions = check_sources(manifests)
    preparer = OUT / 'original_audits/compas__prepare_compas_heterogeneity.py'
    require(file_sha(preparer) == manifests['compas']['source_hashes']['deployment/prepare_compas_heterogeneity.py'], 'COMPAS audit producer SHA mismatch')
    original = {'adult': read(OUT / 'original_audits/adult__client_distribution_audit.json')['reports'],
                'compas': read(OUT / 'original_audits/compas__client_distribution_audit.json')}
    original_map = {(d, float(r.get('client_alpha', r.get('alpha'))), r['seed']): r for d, rows in original.items() for r in rows}
    require(len(original_map) == 60, 'Archived partition-audit coverage')
    cross_records = read(OUT / 'cross_attack_audit_sources.json')
    cross_map = {(r['archived_result_excerpt']['dataset'], float(r['archived_result_excerpt']['alpha']), r['archived_result_excerpt']['seed']): r for r in cross_records}
    require(len(cross_map) == 60, 'S-DFA cross-check coverage')
    member_map = {m['output']: m for m in restoration['members'] if m['output']}
    expected = {(d, alpha, seed) for d in ['adult', 'compas'] for alpha in ALPHAS for seed in SEEDS}
    found, partitions, client_rows, roots = set(), [], [], {}
    checks = collections.Counter(); shuffle_checks = set(); job_hashes = []
    for path in sorted((OUT / 'raw_results').glob('*.json')):
        raw = read(path); d, alpha, seed = raw['dataset'], float(raw['alpha']), raw['seed']; key = (d, alpha, seed)
        require(key in expected and key not in found, 'Unexpected/duplicate partition'); found.add(key)
        cfg = raw['config']; contract = raw['data_contract']; audit = contract['server_sampling_audit']; n_clients = cfg['num_clients']
        require(alpha == cfg['client_alpha'] and seed == cfg['seed'] and n_clients == 20 and cfg['num_malicious'] == 4, 'Partition identity')
        require(raw['attack'] == 'Benign' and raw['method'] == 'GuardFed-AD2+' and raw['distribution'] == 'non-IID', 'Original result identity')
        require(raw['rounds'] == cfg['rounds'] == 70 and [r['round'] for r in raw['trajectory_metrics']] == list(range(1, 71)), 'Incomplete source run')
        require(len(raw['round_summaries']) == 70, 'Incomplete source round diagnostics')
        require(cfg['server_sampling'] == 'stratified_sensitive' and cfg['synthetic_ratio'] == 0 and contract['root_synthetic_rows'] == 0, 'Changed root sampling')
        if d == 'compas':
            require(cfg['compas_preprocessing_version'] == contract['preprocessing_version'] == 'train_only', 'COMPAS preprocessing cohort mismatch')
            require(cfg['root_sensitive_noise'] == cfg['root_label_noise'] == 0, 'Unexpected root noise')
        root_n, train_n = contract['root_clean_rows'], contract['train_rows']
        root_cells, global_cells = audit['server_group_counts'], audit['global_group_counts']
        require(set(root_cells) == set(global_cells) == set(CELLS), 'Missing root/global cells')
        require(sum(root_cells.values()) == root_n == audit['server_rows'] and sum(global_cells.values()) == train_n, 'Root/global totals')
        root_sensitive = {s: sum(root_cells[f'{s}|{y}'] for y in [0, 1]) for s in [0, 1]}
        global_sensitive = {s: sum(global_cells[f'{s}|{y}'] for y in [0, 1]) for s in [0, 1]}
        root_labels = {y: sum(root_cells[f'{s}|{y}'] for s in [0, 1]) for y in [0, 1]}
        require(root_sensitive == {int(s): v for s, v in audit['server_sensitive_counts'].items()}, 'Root sensitive marginals')
        require(root_labels == {int(y): v for y, v in audit['server_label_counts'].items()}, 'Root label marginals')
        lengths = {s: global_sensitive[s] - root_sensitive[s] for s in [0, 1]}
        require(sum(lengths.values()) == train_n - root_n, 'Client pool total')
        counts = reconstruct_sensitive_counts(lengths, alpha, seed)
        # Frozen-code equivalence with arbitrary int64 index values and dummy labels.
        fake_s = np.concatenate([np.zeros(lengths[0], dtype=int), np.ones(lengths[1], dtype=int)])
        fake = pd.DataFrame({'sensitive': fake_s, 'label': np.zeros(len(fake_s), dtype=int), 'dummy': np.zeros(len(fake_s))})
        frozen = frozen_functions[d](fake, ['dummy'], 'label', 'sensitive', 20, alpha, torch.device('cpu'), seed)
        for s in [0, 1]:
            token = (lengths[s], seed)
            if token not in shuffle_checks:
                shuffle_independence(*token); shuffle_checks.add(token)
            require(sum(counts[s]) == lengths[s], 'Reconstructed sensitive conservation')
            require([int(np.sum(frozen[cid]['sensitive'] == s)) for cid in range(20)] == counts[s], 'Count helper differs from frozen AST function')
        checks['frozen_AST_mock_partitions'] += 1
        cross = cross_map[key]; sdfa = cross['archived_result_excerpt']
        require(sdfa['attack'] == 'S-DFA' and sdfa['method'] == raw['method'] and cross['trajectory_rounds'] == list(range(1, 71)), 'S-DFA source identity')
        require(sdfa['data_contract'] == contract, 'S-DFA pre-attack data/root differs')
        require({k: v for k, v in sdfa['config'].items() if k != 'experiment_tag'} == {k: v for k, v in cfg.items() if k != 'experiment_tag'}, 'S-DFA partition config differs')
        for record in [raw, sdfa]:
            job_id = record['revision_job']['id']; job_path = OUT / 'jobs' / (d + '__' + job_id + '.json'); job = read(job_path)
            require(job['config'] == record['config'], 'Frozen job config mismatch')
            require(job['source_hashes'] == record['revision_job']['source_hashes'] == manifests[d]['source_hashes'], 'Frozen source/data identity mismatch')
            require(job['output'] == record['revision_job']['output'] and job['id'] == job_id, 'Frozen job provenance mismatch')
            require(job['output'].replace('/runs/', '/jobs/') + '.json' in manifests[d]['jobs'], 'Job not in original manifest')
            for attr in ['method', 'attack', 'dataset', 'distribution']:
                require(job[attr] == record[attr], 'Frozen job scientific identity mismatch')
            job_hashes.append({'dataset': d, 'id': job_id, 'sha256': file_sha(job_path)})
            checks['original_jobs_matched'] += 1
        archived = original_map[key]
        require(archived['train_rows'] == train_n, 'Original audit training cohort differs')
        require(archived['empty_client_ids'] == [c for c in range(20) if counts[0][c] + counts[1][c] == 0], 'Original empty client IDs differ')
        if d == 'adult':
            require(archived['source_hashes'] == manifests[d]['source_hashes'], 'Original Adult audit source pin differs')
            require(archived['server_sampling_audit'] == audit and archived['root_cross_counts'] == root_cells, 'Original Adult root audit differs')
        else:
            require(archived['root_noise_audit'] == contract['root_noise_audit'] and archived['preprocessing'] == 'train_only', 'Original COMPAS root audit differs')
        original_clients = {c['client_id']: c for c in archived['clients']}
        require(set(original_clients) == set(range(20)), 'Original audit client IDs incomplete')
        require([a['client_id'] for a in raw['attack_audit']] == [a['client_id'] for a in sdfa['attack_audit']] == list(range(20)), 'Result attack-audit client IDs')
        partition_clients, joint_totals = [], {cell: 0 for cell in CELLS}
        for cid in range(20):
            n0, n1 = counts[0][cid], counts[1][cid]; total = n0 + n1; orig = original_clients[cid]
            benign_a, sdfa_a = raw['attack_audit'][cid], sdfa['attack_audit'][cid]
            require(total == benign_a['samples'] == sdfa_a['samples'] == orig.get('n', orig.get('samples')), 'Reconstructed client total differs')
            require(benign_a['is_malicious'] == sdfa_a['is_malicious'] == (cid < 4) and benign_a['attack_types'] == [], 'Nominal malicious flags')
            joint = orig.get('sensitive_label_counts', orig.get('group_label_counts'))
            require(set(joint) == set(CELLS) and all(isinstance(v, int) and v >= 0 for v in joint.values()), 'Invalid original joint measurement')
            require(n0 == joint['0|0'] + joint['0|1'] and n1 == joint['1|0'] + joint['1|1'], 'Archived joint sensitive marginals differ from replay')
            for cell in CELLS:
                joint_totals[cell] += joint[cell]
            if d == 'adult':
                require(orig['sensitive_counts'] == {'0': n0, '1': n1}, 'Adult explicit sensitive counts differ')
                require(orig['label_counts'] == {str(y): joint[f'0|{y}'] + joint[f'1|{y}'] for y in [0, 1]}, 'Adult label marginal inconsistent')
                require(orig['empty_client'] == (total == 0) and orig['empty_sensitive_groups'] == [str(s) for s in [0, 1] if counts[s][cid] == 0], 'Adult missing-group tags inconsistent')
            elif total:
                close(orig['sensitive_one_share'], n1 / total, 'COMPAS sensitive ratio inconsistent')
                close(orig['positive_label_share'], (joint['0|1'] + joint['1|1']) / total, 'COMPAS label ratio inconsistent')
            else:
                require(orig['sensitive_one_share'] is None and orig['positive_label_share'] is None, 'Empty client ratio must be undefined')
            if cid < 4:
                require(sdfa_a['attack_types'] == ['fflip', 'foe'] and sdfa_a['fflip_mode'] == 'all_unprivileged', 'Changed S-DFA attack recipe')
                require(sdfa_a['fflip_changed'] == n1 and sdfa_a['label_changed_count'] == 0, 'S-DFA sensitive-count cross-check failed')
                checks['sdfa_fflip_changed_matches_sensitive1'] += 1
            checks['client_totals_match_two_results_and_original_audit'] += 1
            checks['sensitive_counts_match_archived_joint_marginals'] += 2
            row = {'dataset': d, 'alpha': alpha, 'seed': seed, 'client_id': cid,
                   'samples_replayed': total, 'sensitive0_replayed': n0, 'sensitive1_replayed': n1,
                   'sensitive1_fraction_nonempty': n1 / total if total else None,
                   'empty_client': total == 0, 'missing_sensitive0': n0 == 0, 'missing_sensitive1': n1 == 0,
                   'nonempty_single_sensitive_group': total > 0 and min(n0, n1) == 0,
                   'nominal_first4_under_attack': cid < 4,
                   'archived_label0': joint['0|0'] + joint['1|0'], 'archived_label1': joint['0|1'] + joint['1|1'],
                   'archived_positive_label_fraction_nonempty': (joint['0|1'] + joint['1|1']) / total if total else None,
                   **{'archived_joint_' + cell.replace('|', '_'): joint[cell] for cell in CELLS},
                   'archived_missing_joint_cells': sum(v == 0 for v in joint.values()),
                   'archived_tensor_sha_not_recomputed': orig.get('partition_tensor_sha256'),
                   'source_result_sha256': member_map[str(path.resolve())]['sha256']}
            client_rows.append(row); partition_clients.append(row)
        require(joint_totals == {cell: global_cells[cell] - root_cells[cell] for cell in CELLS}, 'Original client joint totals do not conserve population-root')
        totals = np.array([c['samples_replayed'] for c in partition_clients]); nonempty = [c for c in partition_clients if not c['empty_client']]
        sens = np.array([c['sensitive1_fraction_nonempty'] for c in nonempty]); label = np.array([c['archived_positive_label_fraction_nonempty'] for c in nonempty])
        row = {'dataset': d, 'alpha': alpha, 'seed': seed, 'distribution_label': raw['distribution'],
               'train_rows': train_n, 'root_rows': root_n, 'client_pool_rows': int(totals.sum()),
               'client_pool_sensitive0': lengths[0], 'client_pool_sensitive1': lengths[1],
               'client_samples_min_including_empty': int(totals.min()), 'client_samples_max': int(totals.max()),
               'client_samples_min_nonempty': int(totals[totals > 0].min()), 'client_samples_cv_population_sd': float(totals.std(ddof=0) / totals.mean()),
               'empty_clients': int(np.sum(totals == 0)), 'nonempty_clients': len(nonempty),
               'clients_missing_any_sensitive_including_empty': sum(c['missing_sensitive0'] or c['missing_sensitive1'] for c in partition_clients),
               'nonempty_single_sensitive_group_clients': sum(c['nonempty_single_sensitive_group'] for c in partition_clients),
               'clients_missing_any_joint_cell_including_empty_archived': sum(c['archived_missing_joint_cells'] > 0 for c in partition_clients),
               'nonempty_missing_any_joint_cell_archived': sum(c['archived_missing_joint_cells'] > 0 for c in nonempty),
               'sensitive1_fraction_min_nonempty': float(sens.min()), 'sensitive1_fraction_max_nonempty': float(sens.max()),
               'sensitive1_fraction_population_sd_nonempty': float(sens.std(ddof=0)),
               'label1_fraction_min_nonempty_archived': float(label.min()), 'label1_fraction_max_nonempty_archived': float(label.max()),
               'first4_nominal_sample_fraction': float(totals[:4].sum() / totals.sum()),
               'first4_sensitive0_population_coverage': sum(counts[0][:4]) / lengths[0],
               'first4_sensitive1_population_coverage': sum(counts[1][:4]) / lengths[1],
               'first4_nonempty_clients': int(np.sum(totals[:4] > 0)),
               'sdfa_changed_sensitive_annotations_first4': sum(counts[1][:4]),
               'source_result_sha256': member_map[str(path.resolve())]['sha256'],
               'source_sdfa_result_sha256': cross['member']['sha256']}
        partitions.append(row)
        joint_tvd = .5 * sum(abs(root_cells[cell] / root_n - global_cells[cell] / train_n) for cell in CELLS)
        sensitive_tvd = .5 * sum(abs(root_sensitive[s] / root_n - global_sensitive[s] / train_n) for s in [0, 1])
        label_tvd = .5 * sum(abs(root_labels[y] / root_n - sum(global_cells[f'{s}|{y}'] for s in [0, 1]) / train_n) for y in [0, 1])
        for value, tag in [(joint_tvd, 'group_tvd'), (sensitive_tvd, 'sensitive_tvd'), (label_tvd, 'label_tvd')]:
            close(value, audit[tag], 'Root TVD does not match archived calculation'); checks['root_tvd_matches'] += 1
        root_row = {'dataset': d, 'seed': seed, 'root_rows': root_n, 'train_rows': train_n,
                    **{'root_' + cell.replace('|', '_'): root_cells[cell] for cell in CELLS},
                    **{'global_' + cell.replace('|', '_'): global_cells[cell] for cell in CELLS},
                    'root_min_joint_support': min(root_cells.values()), 'root_joint_tvd': joint_tvd,
                    'root_sensitive_tvd': sensitive_tvd, 'root_label_tvd': label_tvd,
                    'root_sensitive1_fraction': root_sensitive[1] / root_n,
                    'root_label1_fraction': root_labels[1] / root_n,
                    'clean_root_tensor_sha_not_recomputed': contract.get('root_noise_audit', {}).get('clean_root_sha256')}
        if (d, seed) in roots:
            require(roots[d, seed] == root_row, 'Same dataset/seed root changes across alpha')
        roots[d, seed] = root_row
        checks['accepted_unique_partitions'] += 1
    require(found == expected, 'Incomplete 60-partition coverage')
    numeric_keys = [k for k, v in partitions[0].items() if isinstance(v, (int, float)) and k not in ['alpha', 'seed']]
    summary = {f'{d}|alpha={alpha:g}': {k: stats([r[k] for r in partitions if r['dataset'] == d and r['alpha'] == alpha]) for k in numeric_keys}
               for d in ['adult', 'compas'] for alpha in ALPHAS}
    root_keys = ['root_min_joint_support', 'root_joint_tvd', 'root_sensitive_tvd', 'root_label_tvd', 'root_sensitive1_fraction', 'root_label1_fraction']
    root_summary = {d: {k: stats([r[k] for r in roots.values() if r['dataset'] == d]) for k in root_keys} for d in ['adult', 'compas']}
    require(checks['client_totals_match_two_results_and_original_audit'] == 1200 and checks['sdfa_fflip_changed_matches_sensitive1'] == 240, 'Incomplete independent count checks')
    acceptance = {'status': 'PASS', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'scope': '60 pre-attack sensitive-allocation replays; original joint audits restored and consistency checked',
                  'checks': dict(checks), 'shuffle_content_independence_tests': len(shuffle_checks),
                  'source_archive': {k: restoration[k] for k in ['archive', 'archive_sha256']},
                  'frozen_source_receipts': source_receipts, 'frozen_source_functions': source_ast,
                  'frozen_jobs': job_hashes, 'manifest_sha256s': MANIFEST_SHAS,
                  'partition_summaries_across_10_shared_seeds': summary, 'root_summaries_across_10_unique_seeds': root_summary,
                  'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__},
                  'boundaries': ['No original tabular features/raw metadata or tensors loaded; no new training or server access.',
                                 'Client sensitive counts independently replayed; archived label/joint counts are recovered prior measurements, not regenerated from original metadata.',
                                 'Archived client/root tensor hashes are preserved but were not independently recomputed.',
                                 'Empty clients retained. Sensitive and label ratios of empty clients are undefined, not zero.',
                                 'First four means nominal attacked IDs under S-DFA; Benign applies no attack.',
                                 'Strong alpha sensitivity cohorts are distinct from historical Table II and CelebA; no metrics pooled.']}
    # Pin the intended helper explicitly instead of relying on AST body position.
    self_tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    helper = next(n for n in self_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reconstruct_sensitive_counts')
    acceptance['count_helper_ast_sha256'] = hashlib.sha256(ast.dump(helper, include_attributes=False).encode()).hexdigest()
    write_csv(OUT / 'per_partition_60.csv', partitions)
    write_csv(OUT / 'per_client_1200.csv', client_rows)
    write_csv(OUT / 'root_support_20_unique_seeds.csv', list(roots.values()))
    write_json(OUT / 'acceptance.json', acceptance)
    print(json.dumps({'status': 'PASS', 'checks': dict(checks), 'source_archive_sha256': restoration['archive_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
