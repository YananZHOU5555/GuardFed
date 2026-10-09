"""Replay frozen CelebA train partitions on CPU without accessing evaluation labels.

Remote mode pins current repo metadata/code/results against archived contracts,
decodes only training prefixes of Smiling/Male, and exports a train-only receipt.
Offline mode replays that receipt and the pinned archived function source.
Neither mode imports/runs the full loader, images, model, attacks or training.
"""
from __future__ import annotations
import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

OUT = Path(__file__).resolve().parent
CELLS = ('0|0', '0|1', '1|0', '1|1')  # Male | Smiling


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_dump(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def supports(s, y):
    return {f'{g}|{l}': int(((s == g) & (y == l)).sum()) for g in (0, 1) for l in (0, 1)}


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'n_seeds': len(a), 'mean': float(a.mean()), 'sample_sd': float(a.std(ddof=1)),
            'min': float(a.min()), 'max': float(a.max())}


def load_frozen_functions(source, pins):
    require(sha(source) == pins['scripts/reproduce_paper_tables.py']['sha256'], 'Frozen core source mismatch')
    text = source.read_text(encoding='utf-8')
    names = {'sample_server_dataframe', 'create_client_data_dict', '_adjust_counts_to_total', 'apply_root_noise'}
    selected = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({n.name for n in selected} == names, 'Missing frozen AST function')
    module = ast.Module(body=[ast.parse('from __future__ import annotations').body[0]] + selected, type_ignores=[])
    namespace = {'np': np, 'pd': pd, 'torch': torch, 'math': math, 'hashlib': hashlib}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    records = []
    excerpts = []
    for node in selected:
        segment = ast.get_source_segment(text, node)
        records.append({'name': node.name, 'source_text_sha256': hashlib.sha256(segment.encode()).hexdigest(),
                        'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
                        'line': node.lineno, 'end_line': node.end_lineno})
        excerpts.append(segment)
    return namespace, records, '\n\n'.join(excerpts) + '\n'


def read_train_prefix(npz, name, total, count):
    """Read just the C-order NPY header and first count scalar entries of a ZIP member.

    The requested Smiling/Male payload ends at the final official train row.
    Validation/test label payloads are not requested or materialized.
    """
    with zipfile.ZipFile(npz) as archive:
        with archive.open(name + '.npy') as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                raise ValueError(f'Unsupported NPY header: {version}')
            require(shape == (total,) and not fortran and not dtype.hasobject, 'Unsafe/non-vector metadata')
            payload = member.read(count * dtype.itemsize)
            require(len(payload) == count * dtype.itemsize, 'Short training metadata prefix')
            a = np.frombuffer(payload, dtype=dtype).copy()
            return a, {'member': name + '.npy', 'shape': list(shape), 'dtype': str(dtype),
                       'requested_label_entries': count, 'requested_payload_bytes': len(payload),
                       'first_included_position': 0, 'last_included_position': count - 1,
                       'evaluation_label_entries_requested': 0, 'train_prefix_sha256': array_sha(a)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    refs = json.loads((OUT / 'reference_contracts.json').read_text(encoding='utf-8'))
    pins = refs['source_pins']
    contracts = refs['contracts']
    require(len(contracts) == 20, 'Expected20 original contracts')
    before = {}
    if args.offline:
        source = OUT / pins['scripts/reproduce_paper_tables.py']['local_file']
        loader = OUT / pins['src/celeba_data.py']['local_file']
        with np.load(OUT / 'train_only_audit_metadata.npz', allow_pickle=False) as f:
            ids, split, labels, sensitive = (f[k] for k in ('image_id', 'split', 'train_Smiling', 'train_Male'))
        metadata_receipt = json.loads((OUT / 'remote_metadata_receipt.json').read_text())
        runtime_mode = 'offline_independent_replay'
    else:
        require(args.repo is not None, 'Remote requires --repo')
        source = args.repo / 'scripts/reproduce_paper_tables.py'
        loader = args.repo / 'src/celeba_data.py'
        cache = args.repo / 'data/celeba/derived/rgb64_v1'
        metadata = cache / 'metadata.npz'
        manifest = cache / 'manifest.json'
        required_files = [source, loader, metadata, manifest]
        for path in required_files:
            before[str(path)] = sha(path)
        require(before[str(loader)] == pins['src/celeba_data.py']['sha256'], 'Frozen loader mismatch')
        cache_manifest = json.loads(manifest.read_text())
        require(cache_manifest['complete'], 'Cache incomplete')
        for rec in contracts:
            sh = rec['source_hashes']
            for rel, path in [('data/celeba/derived/rgb64_v1/metadata.npz', metadata),
                              ('data/celeba/derived/rgb64_v1/manifest.json', manifest)]:
                require(sh[rel] == before[str(path)], 'Original data hash differs from current source')
            im = rec['data_contract']['image_data_contract']
            require(im['official_metadata'] == cache_manifest['official_metadata'], 'Official metadata identity mismatch')
            result_path = Path(rec['reference_output']) / 'result.json'
            digest = sha(result_path)
            require(digest == rec['raw_sha256'] and result_path.stat().st_size == rec['raw_bytes'], 'Original Full reference changed')
            before[str(result_path)] = digest
        with np.load(metadata, allow_pickle=False) as f:
            ids, split = f['image_id'], f['split']  # no labels from this npz access
        train_pos = np.flatnonzero(split == 0)
        require(np.array_equal(train_pos, np.arange(162770)), 'Train must be contiguous prefix for selective label reads')
        labels, lr = read_train_prefix(metadata, 'Smiling', len(ids), len(train_pos))
        sensitive, sr = read_train_prefix(metadata, 'Male', len(ids), len(train_pos))
        metadata_receipt = {
            'metadata_path': str(metadata), 'metadata_sha256': before[str(metadata)],
            'cache_manifest_sha256': before[str(manifest)], 'official_metadata': cache_manifest['official_metadata'],
            'label_read_strategy': 'NPY headers then C-order train prefix only; no evaluation label entries requested',
            'label_prefix_reads': [lr, sr], 'images_loaded': False,
            'full_loader_executed': False, 'train_rows': len(labels), 'total_id_rows': len(ids),
            'train_image_id_array_sha256': array_sha(ids[train_pos]),
            'official_split_counts': {str(i): int((split == i).sum()) for i in (0, 1, 2)},
        }
        dump(OUT / 'remote_metadata_receipt.json', metadata_receipt)
        np.savez_compressed(OUT / 'train_only_audit_metadata.npz', image_id=ids, split=split,
                            train_Smiling=labels, train_Male=sensitive)
        runtime_mode = 'remote_readonly_cpu'
    require(sha(loader) == pins['src/celeba_data.py']['sha256'], 'Loader pin changed')
    funcs, ast_records, excerpts = load_frozen_functions(source, pins)
    require(np.array_equal(ids, np.arange(1, len(ids) + 1)), 'Official image IDs no longer monotone')
    train_pos, eval_pos = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    require(len(labels) == len(sensitive) == len(train_pos) == 162770, 'Train metadata sizes')
    require(set(np.unique(labels)) == set(np.unique(sensitive)) == {0, 1}, 'Metadata is not binary')
    require(not np.intersect1d(ids[train_pos], ids[eval_pos]).size, 'Train/valid IDs overlap')
    frame = pd.DataFrame({'image_id': ids[train_pos], 'Smiling': labels, 'Male': sensitive})
    global_counts = supports(sensitive, labels)
    prior_clients = {}
    with (OUT / 'prior_sensitive_clients_400.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            prior_clients[(r['distribution'], int(r['seed']), int(r['client_id']))] = r
    require(sha(OUT / 'prior_sensitive_clients_400.csv') == refs['prior_sensitive_csv_sha256'], 'Prior400 source changed')
    client_rows, root_rows, partition_rows, checks = [], [], [], []
    for ref in sorted(contracts, key=lambda r: (r['distribution'], r['seed'])):
        seed, distribution = ref['seed'], ref['distribution']
        cfg = SimpleNamespace(**ref['config'])
        require(cfg.celeba_evaluation_split == 'valid' and cfg.celeba_train_limit == cfg.celeba_eval_limit == 0, 'Subset/evaluation config mismatch')
        require(cfg.server_sampling == 'stratified_sensitive' and cfg.synthetic_ratio == 0, 'Unexpected root sampling/synthesis')
        require(cfg.num_clients == 20 and cfg.num_malicious == 4 and cfg.root_label_noise == cfg.root_sensitive_noise == 0, 'Expected clean 20/4')
        alpha = 5000. if distribution == 'IID' else 5.
        require(ref['alpha'] == cfg.client_alpha == alpha, 'Actual alpha mismatch')
        original = ref['data_contract']
        im = original['image_data_contract']
        require(array_sha(ids[train_pos]) == im['train_image_ids_sha256'], 'Train image-ID hash mismatch')
        require(array_sha(ids[eval_pos]) == im['evaluation_image_ids_sha256'], 'Valid image-ID hash mismatch')
        require(metadata_receipt['cache_manifest_sha256'] == im['cache_manifest_sha256'], 'Cache identity mismatch')
        root_df, sample_audit = funcs['sample_server_dataframe'](frame, 'Smiling', 'Male', cfg)
        require(sample_audit == original['server_sampling_audit'], 'Root sampling audit differs from archived contract')
        client_df = frame.drop(root_df.index).reset_index(drop=True)
        clients = funcs['create_client_data_dict'](client_df, ['image_id'], 'Smiling', 'Male', 20, alpha, torch.device('cpu'), seed)
        # The frozen loader applies noise after client splitting; zero-noise contract is checked in full.
        root_clean = root_df.reset_index(drop=True)
        root_df, noise_audit = funcs['apply_root_noise'](root_clean, 'Smiling', 'Male', cfg)
        require(noise_audit == original['root_noise_audit'], 'Clean root hash/count/noise audit mismatch')
        root_ids = root_df['image_id'].to_numpy(dtype=np.int64)
        root_y = root_df['Smiling'].to_numpy(dtype=int)
        root_s = root_df['Male'].to_numpy(dtype=int)
        rc = supports(root_s, root_y)
        require(array_sha(root_ids) == im['root_image_ids_sha256'], 'Root image-ID hash mismatch')
        require(rc == sample_audit['server_group_counts'], 'Root four cells differ')
        pool = supports(client_df['Male'].to_numpy(), client_df['Smiling'].to_numpy())
        require(all(pool[k] + rc[k] == global_counts[k] for k in CELLS), 'Root+pool four-cell conservation')
        unions, groups, rows = [], [], []
        for cid, client in clients.items():
            ci = client['X'][:, 0].numpy().astype(np.int64)
            cy = client['y'].numpy()
            cs = client['sensitive']
            n = len(cy)
            cc = supports(cs, cy)
            require(n == im['client_sample_counts'][cid], 'Original client total mismatch')
            old = prior_clients[(distribution, seed, cid)]
            require(n == int(old['sample_count']), 'Sp-DFA prior total mismatch')
            require(int((cs == 0).sum()) == int(old['sensitive_Male0_count']) and
                    int((cs == 1).sum()) == int(old['sensitive_Male1_count']), 'Sp-DFA prior sensitive margins mismatch')
            require(len(np.unique(ci)) == n and np.isin(ci, ids[train_pos]).all(), 'Duplicate/nontrain client IDs')
            # Independently resolve the source train labels for every client ID.
            require(np.array_equal(cy, labels[ci - 1]) and np.array_equal(cs, sensitive[ci - 1]), 'Client labels differ from train-only metadata')
            row = {'distribution': distribution, 'alpha': alpha, 'seed': seed, 'client_id': cid,
                   'sample_count': n, **{f'Male{g}_Smiling{l}': cc[f'{g}|{l}'] for g in (0, 1) for l in (0, 1)},
                   'Male0_count': int((cs == 0).sum()), 'Male1_count': int((cs == 1).sum()),
                   'Smiling0_count': int((cy == 0).sum()), 'Smiling1_count': int((cy == 1).sum()),
                   'Male1_fraction': float((cs == 1).mean()) if n else 0.,
                   'Smiling1_fraction': float((cy == 1).mean()) if n else 0.,
                   'joint_tvd_from_client_pool': 0.5 * sum(abs(cc[k] / n - pool[k] / len(client_df)) for k in CELLS) if n else 0.,
                   'empty_client': n == 0, 'missing_sensitive_group': min((cs == 0).sum(), (cs == 1).sum()) == 0,
                   'missing_label': min((cy == 0).sum(), (cy == 1).sum()) == 0,
                   'missing_joint_cell': min(cc.values()) == 0, 'minimum_joint_support': min(cc.values()),
                   'nominal_attack_client_first4': cid < 4,
                   'ordered_image_ids_sha256': array_sha(ci),
                   'ordered_joint_rows_sha256': array_sha(np.column_stack([ci, cs, cy]).astype('<i8')),
                   'reference_id': ref['id'], 'reference_result_sha256': ref['raw_sha256']}
            rows.append(row); client_rows.append(row); unions.append(ci); groups.append(cc)
        union = np.concatenate(unions)
        require(len(np.unique(union)) == len(union) == 146493, 'Client/client overlap or incomplete pool')
        require(not np.intersect1d(root_ids, union).size, 'Root/client overlap')
        require(np.array_equal(np.sort(np.concatenate([root_ids, union])), ids[train_pos]), 'Root+clients not exact train partition')
        require(not np.intersect1d(np.concatenate([root_ids, union]), ids[eval_pos]).size, 'Training membership leaks to eval')
        require(all(sum(g[k] for g in groups) == pool[k] for k in CELLS), 'Client four-cell pool conservation')
        root_rows.append({'distribution': distribution, 'alpha': alpha, 'seed': seed, 'root_rows': len(root_ids),
                          **{f'Male{g}_Smiling{l}': rc[f'{g}|{l}'] for g in (0, 1) for l in (0, 1)},
                          'minimum_joint_support': min(rc.values()),
                          'joint_tvd_from_training': sample_audit['group_tvd'],
                          'ordered_image_ids_sha256': array_sha(root_ids),
                          'ordered_joint_rows_sha256': array_sha(np.column_stack([root_ids, root_s, root_y]).astype('<i8')),
                          'reference_id': ref['id'], 'reference_result_sha256': ref['raw_sha256']})
        first = rows[:4]
        partition = {'distribution': distribution, 'alpha': alpha, 'seed': seed, 'client_rows': len(union),
                     'client_sample_min': min(r['sample_count'] for r in rows), 'client_sample_max': max(r['sample_count'] for r in rows),
                     'Male1_fraction_min': min(r['Male1_fraction'] for r in rows), 'Male1_fraction_max': max(r['Male1_fraction'] for r in rows),
                     'Smiling1_fraction_min': min(r['Smiling1_fraction'] for r in rows), 'Smiling1_fraction_max': max(r['Smiling1_fraction'] for r in rows),
                     'Smiling1_fraction_population_sd': float(np.std([r['Smiling1_fraction'] for r in rows], ddof=0)),
                     'joint_tvd_mean': float(np.mean([r['joint_tvd_from_client_pool'] for r in rows])),
                     'joint_tvd_max': max(r['joint_tvd_from_client_pool'] for r in rows),
                     'minimum_client_joint_support': min(r['minimum_joint_support'] for r in rows),
                     'empty_clients': sum(r['empty_client'] for r in rows),
                     'missing_sensitive_group_clients': sum(r['missing_sensitive_group'] for r in rows),
                     'missing_label_clients': sum(r['missing_label'] for r in rows),
                     'missing_joint_cell_clients': sum(r['missing_joint_cell'] for r in rows),
                     'first4_sample_coverage': sum(r['sample_count'] for r in first) / len(union),
                     **{f'first4_Male{g}_Smiling{l}_coverage': sum(r[f'Male{g}_Smiling{l}'] for r in first) / pool[f'{g}|{l}']
                        for g in (0, 1) for l in (0, 1)},
                     'root_image_ids_sha256': array_sha(root_ids), 'client_union_sorted_ids_sha256': array_sha(np.sort(union))}
        partition_rows.append(partition)
        checks.append({'reference_id': ref['id'], 'raw_result_sha256': ref['raw_sha256'],
                       'train_hash_match': True, 'valid_id_hash_match': True, 'root_id_hash_match': True,
                       'root_sampling_audit_exact': True, 'root_noise_audit_exact': True,
                       'client_sample_counts_exact': 20, 'client_sensitive_margin_exact': 40,
                       'client_train_label_resolutions_exact': 20, 'sample_disjoint_complete': True})
    unique_roots = []
    for seed in range(91001, 91011):
        pair = [r for r in root_rows if r['seed'] == seed]
        require(len(pair) == 2 and pair[0]['ordered_joint_rows_sha256'] == pair[1]['ordered_joint_rows_sha256'], 'Root differs across distribution')
        unique_roots.append(pair[0])
    metrics = ('Smiling1_fraction_min', 'Smiling1_fraction_max', 'Smiling1_fraction_population_sd',
               'joint_tvd_mean', 'joint_tvd_max', 'minimum_client_joint_support', 'first4_sample_coverage',
               'first4_Male0_Smiling0_coverage', 'first4_Male0_Smiling1_coverage',
               'first4_Male1_Smiling0_coverage', 'first4_Male1_Smiling1_coverage')
    summary = {dist: {k: stats([r[k] for r in partition_rows if r['distribution'] == dist]) for k in metrics}
               for dist in ('IID', 'non-IID')}
    if not args.offline:
        after = {path: sha(path) for path in before}
        require(before == after, 'Protected source/data/original result changed during audit')
    else:
        after = {}
    prefix = 'offline_' if args.offline else ''
    csv_dump(OUT / (prefix + 'per_client_400.csv'), client_rows)
    csv_dump(OUT / (prefix + 'per_root_20.csv'), root_rows)
    csv_dump(OUT / (prefix + 'per_partition_20.csv'), partition_rows)
    acceptance = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'PASS',
        'mode': runtime_mode, 'partitions': 20, 'shared_seed_count': 10, 'clients_per_partition': 20,
        'client_rows': len(client_rows), 'four_cell_counts': 4 * len(client_rows),
        'archived_benign_contracts': 20, 'prior_Sp_DFA_sensitive_count_matches': 800,
        'original_client_total_matches': 400, 'all_client_train_label_memberships_matched': True,
        'root_train_valid_hash_checks': 60, 'sample_disjoint_complete_partitions': 20,
        'frozen_source_pins': pins, 'frozen_function_ast_records': ast_records,
        'helper_script_sha256': sha(Path(__file__)), 'reference_contracts_sha256': sha(OUT / 'reference_contracts.json'),
        'metadata_receipt': metadata_receipt, 'train_only_export_sha256': sha(OUT / 'train_only_audit_metadata.npz'),
        'runtime': {'python': __import__('sys').version, 'numpy': np.__version__, 'pandas': pd.__version__,
                    'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
                    'torch_interop_threads': torch.get_num_interop_threads(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']},
        'protected_before_after': {'before': before, 'after': after, 'unchanged': before == after},
        'global_train_joint_counts_Male_Smiling': global_counts,
        'client_pool_joint_counts_Male_Smiling_seed91001': {k: sum(r[f'Male{k[0]}_Smiling{k[2]}'] for r in client_rows[:20]) for k in CELLS},
        'checks': checks, 'summary_across_10_seeds': summary,
        'root_summary_across_10_unique_seeds': {'minimum_support': stats([r['minimum_joint_support'] for r in unique_roots]),
                                             'joint_tvd': stats([r['joint_tvd_from_training'] for r in unique_roots])},
        'client_occurrence_totals_by_distribution': {dist: {
            k: sum(int(r[k]) for r in client_rows if r['distribution'] == dist)
            for k in ('empty_client', 'missing_sensitive_group', 'missing_label', 'missing_joint_cell')}
            for dist in ('IID', 'non-IID')},
        'partition_variable': 'Male sensitive group, not target label Smiling',
        'evaluation_labels_accessed': False, 'images_loaded': False, 'model_inference': False,
        'training_started': False, 'old_results_modified': False,
        'scope': 'Clean pre-attack train partitions at frozen selected recipe; descriptive allocation evidence only.',
        'limitations': ['400 clients are nested in20 partitions sharing10 seeds; no400-independent-sample claims.',
                        'No predictions, final evaluation or evidence of method superiority.',
                        'Ordered client IDs and train labels are reconstructed; original result contracts stored only root/train/eval ID hashes and client totals.',
                        'New ordered client/joint hashes are replay receipts, not previously archived original client-ID hashes.'],
    }
    dump(OUT / (prefix + 'acceptance.json'), acceptance)
    if not args.offline:
        (OUT / 'frozen_functions.txt').write_text(excerpts, encoding='utf-8')
        (OUT / 'frozen_loader.txt').write_text(loader.read_text(encoding='utf-8'), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'mode': runtime_mode, 'partitions': 20, 'client_rows': 400,
                      'missing': acceptance['client_occurrence_totals_by_distribution'], 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
