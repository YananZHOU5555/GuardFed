"""Prepare immutable contracts from the already verified local Full100 restore archive."""
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAINING = OUT.parent
REPO = TRAINING.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    restore_path = TRAINING / 'server_reactivation_20261009/restore_bundle.json'
    restore = json.loads(restore_path.read_text(encoding='utf-8'))
    archive = restore_path.parent / restore['archive']
    assert sha(archive) == restore['sha256']
    manifest_path = TRAINING / 'celeba_mechanism_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    chosen = [r for r in manifest['reused_full'] if r['attack'] == 'Benign']
    assert len(chosen) == 20
    contracts = []
    with tarfile.open(archive, 'r:gz') as tar:
        for r in chosen:
            member = r['output'].removeprefix('/workspace/') + '/result.json'
            raw = tar.extractfile(member).read()
            expected = restore['members'][member]
            assert hashlib.sha256(raw).hexdigest() == expected['sha256']
            assert len(raw) == expected['bytes']
            d = json.loads(raw)
            assert d['dataset'] == 'celeba' and d['method'] == 'GuardFed-AD2+' and d['attack'] == 'Benign'
            assert d['rounds'] == 70 and d['seed'] == r['seed'] and d['distribution'] == r['distribution']
            assert d['revision_job']['checkpoint_sha256'] == r['checkpoint_sha256']
            contracts.append({
                'id': r['id'], 'reference_output': r['output'], 'archive_member': member,
                'raw_sha256': expected['sha256'], 'raw_bytes': expected['bytes'],
                'config': d['config'], 'data_contract': d['data_contract'],
                'source_hashes': d['revision_job']['source_hashes'],
                'seed': d['seed'], 'distribution': d['distribution'], 'alpha': d['alpha'],
                'checkpoint_sha256': r['checkpoint_sha256'],
                'attack_audit': d['attack_audit'],
            })
    old = TRAINING / 'celeba_partition_audit_20261009'
    old_acceptance = json.loads((old / 'acceptance.json').read_text(encoding='utf-8'))
    (OUT / 'sources').mkdir(exist_ok=True)
    sources = {}
    for relative, rec in old_acceptance['source_pins'].items():
        src = Path(rec['path'])
        assert sha(src) == rec['sha256']
        dest = OUT / 'sources' / Path(relative).name
        shutil.copyfile(src, dest)
        sources[relative] = {'sha256': rec['sha256'], 'local_file': dest.relative_to(OUT).as_posix()}
    shutil.copyfile(old / 'per_client_400.csv', OUT / 'prior_sensitive_clients_400.csv')
    shutil.copyfile(old / 'per_partition_20.csv', OUT / 'prior_partitions_20.csv')
    payload = {
        'stage': 'clean_training_metadata_joint_partition_audit',
        'manifest_source': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'restore_receipt': str(restore_path), 'restore_receipt_sha256': sha(restore_path),
        'archive': str(archive), 'archive_sha256': restore['sha256'], 'archive_bytes': archive.stat().st_size,
        'source_pins': sources,
        'prior_audit_acceptance': str(old / 'acceptance.json'),
        'prior_audit_acceptance_sha256': sha(old / 'acceptance.json'),
        'prior_sensitive_csv_sha256': sha(OUT / 'prior_sensitive_clients_400.csv'),
        'prior_partition_csv_sha256': sha(OUT / 'prior_partitions_20.csv'),
        'contracts': contracts,
    }
    (OUT / 'reference_contracts.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'contracts': len(contracts), 'archive_sha256': restore['sha256'], 'source_pins': sources}, indent=2))


if __name__ == '__main__':
    main()
