#!/usr/bin/env python3
"""Restore a narrowly selected, read-only tabular partition evidence bundle.

No training, metadata reconstruction, Git mutation, or extractall is used.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

OUT = Path(__file__).resolve().parent
ARCHIVE = OUT.parent / 'revision_verified740_20260923T090026Z.tar.gz'
ARCHIVE_SHA = '55b50d489a4ee1824a8637c7da61e25157b50f500ddbd2fc5119711ba16c399b'
COHORTS = {
    'adult': ('GuardFed-revision', '9e62b7887be73a4c3ce600bb8d6df79596fba0e1'),
    'compas': ('GuardFed-next', 'b1a808b0c8016634e032a3d1914e55782da44fdd'),
}


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_write(relative, blob):
    target = (OUT / relative).resolve()
    assert OUT.resolve() in target.parents
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    return str(target)


def main():
    assert file_sha(ARCHIVE) == ARCHIVE_SHA, 'Original archive changed'
    members, cross_attack = [], []
    with tarfile.open(ARCHIVE, 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name
            dataset = next((d for d, (prefix, _) in COHORTS.items()
                            if name.startswith(prefix + '/results/revision_20260923/' + d + '_heterogeneity_v1/')), None)
            support = name == 'GuardFed-next/deployment/prepare_compas_heterogeneity.py'
            if dataset is None and not support:
                continue
            if '/pilot/' in name:
                continue
            relative = None
            if name.endswith('/manifest.json'):
                relative = 'manifests/' + dataset + '.json'
            elif name.endswith('/client_distribution_audit.json'):
                relative = 'original_audits/' + dataset + '__client_distribution_audit.json'
            elif name.endswith('/audit_client_distributions.py'):
                relative = 'original_audits/' + dataset + '__audit_client_distributions.py'
            elif support:
                relative = 'original_audits/compas__prepare_compas_heterogeneity.py'
            elif '/jobs/' in name and 'GuardFed-AD2+' in name and name.endswith('.json'):
                relative = 'jobs/' + dataset + '__' + Path(name).name
            elif '/runs/' in name and 'GuardFed-AD2+' in name and name.endswith('/result.json'):
                if '_Benign_' in name:
                    relative = 'raw_results/' + dataset + '__' + Path(name).parent.name + '.json'
                elif '_SDFA_' not in name and '_S-DFA_' not in name:
                    continue
            else:
                continue
            blob = archive.extractfile(member).read()
            digest = hashlib.sha256(blob).hexdigest()
            record = {'dataset': dataset or 'compas', 'archive_member': name,
                      'bytes': len(blob), 'sha256': digest, 'output': None}
            if relative:
                record['output'] = safe_write(relative, blob)
            else:
                raw = json.loads(blob)
                keep = ['run_id', 'dataset', 'distribution', 'alpha', 'method', 'attack', 'seed', 'rounds',
                        'num_clients', 'num_malicious', 'config', 'data_contract', 'attack_audit', 'revision_job']
                cross_attack.append({'member': record, 'archived_result_excerpt': {k: raw[k] for k in keep},
                                     'terminal_round': raw['trajectory_metrics'][-1]['round'],
                                     'trajectory_rounds': [x['round'] for x in raw['trajectory_metrics']]})
            members.append(record)
    assert len(list((OUT / 'raw_results').glob('*.json'))) == 60
    assert len(cross_attack) == 60
    source_receipts = []
    for dataset, (_, commit) in COHORTS.items():
        for path in ['scripts/reproduce_paper_tables.py', 'src/data_loader.py', 'scripts/run_revision_ablation.py']:
            result = subprocess.run(['git', 'show', commit + ':' + path], capture_output=True, check=True)
            output = safe_write('frozen_sources/' + dataset + '__' + path.replace('/', '__'), result.stdout)
            source_receipts.append({'dataset': dataset, 'git_commit': commit, 'git_path': path,
                                    'output': output, 'sha256': hashlib.sha256(result.stdout).hexdigest()})
    for name, data in [('cross_attack_audit_sources.json', cross_attack),
                       ('frozen_source_receipts.json', source_receipts),
                       ('restoration_receipt.json', {'archive': str(ARCHIVE.resolve()), 'archive_sha256': ARCHIVE_SHA,
                                                    'members': members, 'full_benign_results': 60,
                                                    'sdfa_result_excerpts': 60,
                                                    'note': 'S-DFA records are explicit excerpts; source member hash covers the full original blob.'})]:
        (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Restored 60 Benign raw results, 120 jobs, 60 explicit S-DFA excerpts, manifests, audits and pinned source.')


if __name__ == '__main__':
    main()
