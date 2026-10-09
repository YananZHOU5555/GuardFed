"""Verify off-server transfer and independently reconstruct all clean joint counts.

Uses NumPy indices only; no torch, training module, pixels or evaluation labels.
The NumPy root replay is independent of the archived pandas .sample function.
"""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

OUT = Path(__file__).resolve().parent
EXPECTED_REMOTE_ARCHIVE_SHA = 'fb44638650ee77bc06d204770eff8f54e9a171a9d8465d4936d6f44a6edeb63d'
EXPECTED_REMOTE_SCRIPT_SHA = '06d5815964b0d29352a7c419fdce8935f4a3585c9647e66a49d0fe4a76ff86d6'
EXPECTED_REMOTE_ACCEPTANCE_SHA = '35f6bae31d6df1af84af05238a84462caada5d4fef6147886201c1f846e79bf1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def arrsha(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def unpack_remote():
    archive = OUT / 'remote_audit_results.tar.gz'
    assert sha(archive) == EXPECTED_REMOTE_ARCHIVE_SHA
    members = []
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            assert member.isfile() and Path(member.name).name == member.name, 'Unsafe member'
            blob = tar.extractfile(member).read()
            target = OUT / ('remote_run.log' if member.name == 'run.log' else member.name)
            digest = hashlib.sha256(blob).hexdigest()
            if target.exists():
                assert target.read_bytes() == blob, f'Existing file differs: {target}'
            else:
                target.write_bytes(blob)
            members.append({'member': member.name, 'output': target.name, 'sha256': digest, 'bytes': len(blob)})
    assert sha(OUT / 'audit_joint_partitions.py') == EXPECTED_REMOTE_SCRIPT_SHA
    assert sha(OUT / 'acceptance.json') == EXPECTED_REMOTE_ACCEPTANCE_SHA
    receipt = {'archive': archive.name, 'archive_sha256': EXPECTED_REMOTE_ARCHIVE_SHA,
               'archive_bytes': archive.stat().st_size, 'members_verified': len(members), 'members': members,
               'remote_host': '89.22.197.55:60350', 'remote_directory': '/workspace/guardfed_checks/celeba_joint_partition_20261009',
               'sha256_received_over_ssh': True, 'safe_single_file_members_only': True}
    dump(OUT / 'transfer_verification.json', receipt)


def validate():
    refs = json.loads((OUT / 'reference_contracts.json').read_text())
    remote = json.loads((OUT / 'acceptance.json').read_text())
    assert remote['status'] == 'PASS' and remote['partitions'] == 20 and remote['client_rows'] == 400
    assert remote['helper_script_sha256'] == sha(OUT / 'audit_joint_partitions.py')
    assert remote['reference_contracts_sha256'] == sha(OUT / 'reference_contracts.json')
    assert remote['protected_before_after']['before'] == remote['protected_before_after']['after']
    assert len(remote['protected_before_after']['before']) == 24
    assert not remote['evaluation_labels_accessed'] and not remote['images_loaded']
    assert not remote['training_started'] and not remote['model_inference'] and not remote['old_results_modified']
    assert sha(OUT / 'train_only_audit_metadata.npz') == remote['train_only_export_sha256']
    for rec in refs['source_pins'].values():
        assert sha(OUT / rec['local_file']) == rec['sha256']
    with np.load(OUT / 'train_only_audit_metadata.npz', allow_pickle=False) as f:
        assert set(f.files) == {'image_id', 'split', 'train_Smiling', 'train_Male'}
        ids, split, y, s = (f[k] for k in ('image_id', 'split', 'train_Smiling', 'train_Male'))
    n = int((split == 0).sum())
    assert len(y) == len(s) == n == 162770
    assert len(ids) == len(split) == 202599 and np.array_equal(ids, np.arange(1, len(ids) + 1))
    assert np.array_equal(np.flatnonzero(split == 0), np.arange(n))
    train_ids = ids[:n]
    with (OUT / 'per_client_400.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with (OUT / 'per_root_20.csv').open(encoding='utf-8') as f:
        roots = list(csv.DictReader(f))
    assert len(rows) == 400 and len(roots) == 20
    row_map = {(r['distribution'], int(r['seed']), int(r['client_id'])): r for r in rows}
    root_map = {(r['distribution'], int(r['seed'])): r for r in roots}
    assert len(row_map) == 400 and len(root_map) == 20
    direct_checks = []
    for rec in refs['contracts']:
        dist, seed, alpha = rec['distribution'], rec['seed'], rec['alpha']
        ic = rec['data_contract']['image_data_contract']
        # pandas GroupBy.sort defaults True; each sample(random_state=seed)
        # creates a fresh RandomState. Independent NumPy index-only replay.
        picks = []
        for group in (0, 1):
            indices = np.flatnonzero(s == group)
            sample = np.random.RandomState(seed).choice(len(indices), size=round(0.1 * len(indices)), replace=False)
            picks.append(indices[sample])
        root_pos = np.sort(np.concatenate(picks))
        root_ids = train_ids[root_pos].astype(np.int64)
        root_s, root_y = s[root_pos], y[root_pos]
        root_record = root_map[(dist, seed)]
        assert arrsha(root_ids) == ic['root_image_ids_sha256'] == root_record['ordered_image_ids_sha256']
        assert arrsha(np.column_stack([root_ids, root_s, root_y]).astype('<i8')) == root_record['ordered_joint_rows_sha256']
        for group in (0, 1):
            for label in (0, 1):
                assert int(((root_s == group) & (root_y == label)).sum()) == int(root_record[f'Male{group}_Smiling{label}'])
        mask = np.ones(n, dtype=bool)
        mask[root_pos] = False
        pool_ids = train_ids[mask].astype(np.int64)
        pool_s, pool_y = s[mask], y[mask]
        rng = np.random.default_rng(seed)
        parts = [[] for _ in range(20)]
        for group in (1, 0):
            indices = np.flatnonzero(pool_s == group)
            rng.shuffle(indices)
            cuts = (np.cumsum(rng.dirichlet([alpha] * 20)) * len(indices)).astype(int)[:-1]
            for cid, part in enumerate(np.split(indices, cuts)):
                parts[cid].append(part)
        union = []
        for cid in range(20):
            positions = np.concatenate(parts[cid])
            ci, cs, cy = pool_ids[positions], pool_s[positions], pool_y[positions]
            row = row_map[(dist, seed, cid)]
            assert len(ci) == int(row['sample_count']) == ic['client_sample_counts'][cid]
            assert arrsha(ci) == row['ordered_image_ids_sha256']
            assert arrsha(np.column_stack([ci, cs, cy]).astype('<i8')) == row['ordered_joint_rows_sha256']
            for group in (0, 1):
                for label in (0, 1):
                    assert int(((cs == group) & (cy == label)).sum()) == int(row[f'Male{group}_Smiling{label}'])
            union.append(ci)
        union = np.concatenate(union)
        assert len(np.unique(union)) == len(union) == 146493
        assert np.intersect1d(root_ids, union).size == 0
        assert np.array_equal(np.sort(np.concatenate([root_ids, union])), train_ids)
        assert arrsha(train_ids) == ic['train_image_ids_sha256']
        assert arrsha(ids[split == 1]) == ic['evaluation_image_ids_sha256']
        direct_checks.append({'distribution': dist, 'seed': seed, 'alpha': alpha,
                              'all20_ordered_client_id_hashes_match': True, 'all80_joint_cells_match': True,
                              'root_joint_hash_match': True, 'complete_disjoint_partition': True})
    # Independent arithmetic recomputation from the exported per-client rows.
    with (OUT / 'per_partition_20.csv').open(encoding='utf-8') as f:
        pr = list(csv.DictReader(f))
    max_summary_error = 0.
    for dist in ('IID', 'non-IID'):
        partition = [r for r in pr if r['distribution'] == dist]
        assert len(partition) == 10
        for field, source in remote['summary_across_10_seeds'][dist].items():
            a = np.asarray([float(r[field]) for r in partition])
            calculated = {'mean': a.mean(), 'sample_sd': a.std(ddof=1), 'min': a.min(), 'max': a.max()}
            for k, v in calculated.items():
                error = abs(float(v) - source[k])
                max_summary_error = max(max_summary_error, error)
                assert error <= 1e-12, (dist, field, k, error)
        # Recompute each partition's label/joint diagnostics from400 raw rows.
        for p in partition:
            subset = [r for r in rows if r['distribution'] == dist and r['seed'] == p['seed']]
            assert len(subset) == 20
            total = sum(int(r['sample_count']) for r in subset)
            cells = {f'Male{g}_Smiling{l}': sum(int(r[f'Male{g}_Smiling{l}']) for r in subset) for g in (0, 1) for l in (0, 1)}
            label_rates = np.array([int(r['Smiling1_count']) / int(r['sample_count']) for r in subset])
            assert abs(label_rates.std(ddof=0) - float(p['Smiling1_fraction_population_sd'])) <= 1e-12
            joint = []
            for r in subset:
                joint.append(0.5 * sum(abs(int(r[k]) / int(r['sample_count']) - cells[k] / total) for k in cells))
                assert abs(joint[-1] - float(r['joint_tvd_from_client_pool'])) <= 1e-12
            assert abs(np.mean(joint) - float(p['joint_tvd_mean'])) <= 1e-12
            assert abs(max(joint) - float(p['joint_tvd_max'])) <= 1e-12
            for g in (0, 1):
                for l in (0, 1):
                    k = f'Male{g}_Smiling{l}'
                    frac = sum(int(r[k]) for r in subset[:4]) / cells[k]
                    assert abs(frac - float(p[f'first4_{k}_coverage'])) <= 1e-12
    result = {'status': 'PASS', 'validator_script_sha256': sha(Path(__file__)),
              'remote_acceptance_sha256': sha(OUT / 'acceptance.json'),
              'independent_implementation': 'NumPy RandomState root index sampling plus default_rng sensitive-group Dirichlet splits; no archived function execution',
              'source_train_metadata_only': True, 'evaluation_label_entries': 0,
              'independent_client_ordered_id_hash_checks': 400,
              'independent_client_ordered_joint_hash_checks': 400,
              'independent_client_joint_cell_checks': 1600,
              'independent_root_joint_cell_checks': 80,
              'independent_root_joint_row_hash_checks': 20,
              'recomputed_seed_summary_scalar_checks': 88,
              'maximum_seed_summary_arithmetic_error': max_summary_error,
              'all24_protected_original_paths_unchanged': True, 'partitions': direct_checks}
    dump(OUT / 'independent_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'partitions'}, indent=2))


if __name__ == '__main__':
    unpack_remote()
    validate()
