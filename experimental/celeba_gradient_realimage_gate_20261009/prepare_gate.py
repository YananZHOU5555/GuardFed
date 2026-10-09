"""Local preparation only; no SSH, resource approval, freeze or training."""
import argparse
from pathlib import Path
import shutil

from gate import HERE, STAGE, digest, read, require, write


def prepare(project):
    require(not (HERE / 'scope.json').exists(), 'Preserve existing preparation; no overwrite')
    baseline = project / 'tmp/celeba_baselines'
    source = baseline / 'gradient_bridge_20261009'
    inventory = read(source / 'FILES_SHA256.json')
    copies = []
    for name in ('worker.py', 'protocol.json', 'accept_result.py', 'prepare.py', 'REPORT.md', 'FILES_SHA256.json'):
        copies.append((source / name, Path('snapshot/gradient_bridge_20261009') / name,
                       inventory['file_sha256'].get(name, digest(source / name))))
    for name, expected in inventory['external_dependency_sha256'].items():
        if name.startswith('../'):
            continue
        copies.append((baseline / name, Path('snapshot') / name, expected))
    original_job = source / 'screen_jobs_draft/FedNGA_eta0.03_IID_Benign_seed91001_screen.json'
    copies.append((original_job, Path('sealed_formal_job.json'), inventory['file_sha256'][str(original_job.relative_to(source)).replace('\\', '/')]))
    copied = {}
    for original, relative, expected in copies:
        require(digest(original) == expected, 'Sealed source changed: ' + str(original))
        destination = HERE / relative
        require(not destination.exists(), 'Do not overwrite snapshot')
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, destination)
        require(digest(destination) == expected, 'Byte-copy mismatch')
        copied[relative.as_posix()] = expected
    protocol = read(HERE / 'snapshot/gradient_bridge_20261009/protocol.json')
    protection = read(project / 'tmp/celeba_hybrid_realimage_gate_20261009/scope.json')['protected_source_hashes']
    for relative, expected in protocol['source_hashes'].items():
        require(protection[relative] == expected, 'Known protected source identity differs')
    write(HERE / 'sealed_copy_receipt.json', dict(original_inventory_sha256=digest(source / 'FILES_SHA256.json'),
        copied_hashes=copied, original_source_modified=False, original_protocol_status=protocol['status']))
    entries = []
    for candidate_id in ('FedNGA_eta0.03', 'Huber_eta0.03_T00.1_M1'):
        candidate = next(x for x in protocol['candidates'] if x['id'] == candidate_id)
        for distribution, attack in (('IID', 'Benign'), ('non-IID', 'S-DFA')):
            identity = candidate_id + '_' + distribution + '_' + attack + '_seed91001_cpu_gate3'
            config = dict(protocol['base_config'], seed=91001, rounds=3, device='cpu',
                client_alpha=protocol['distributions'][distribution], use_reweighting=False,
                experiment_suite='celeba_gradient_realimage_gate_20261009', experiment_tag=identity)
            job = dict(id=identity, dataset='celeba', method=candidate['method'], distribution=distribution, attack=attack,
                pilot_candidate=candidate_id, adapter=candidate['adapter'], config=config, evidence_stage=STAGE,
                scientific_table_records=0, source_hashes=protocol['source_hashes'])
            relative = 'jobs/' + identity + '.json'; write(HERE / relative, job)
            entries.append(dict(id=identity, job=relative, job_sha256=digest(HERE / relative), output='runs/' + identity))
    local = dict(copied)
    for name in ('gate.py', 'prepare_gate.py', 'check_preparation.py', 'sealed_copy_receipt.json', 'REPORT.md'):
        local[name] = digest(HERE / name)
    scope = dict(status='PREPARED_NOT_FROZEN', execution_started=False, evidence_stage=STAGE, jobs=entries,
        formal_protocol_sha256=digest(HERE / 'snapshot/gradient_bridge_20261009/protocol.json'),
        original_formal_decisions_unresolved=5, scientific_table_records=0, test_evaluation_authorized=False,
        max_concurrent_cpu_processes=1, cpu_threads=8, cpu_allocation='PENDING_PARENT_RESOURCE_ASSIGNMENT',
        protected_hybrid_cpu_ids=list(range(8, 16)),
        expected_torch='2.11.0+cu128', expected_cuda_build='12.8',
        protected_source_hashes=protection, local_hashes=local,
        guide_sha256='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa',
        pilot_choices=dict(objective='original_unweighted_ce', root_reference='frozen_root_localadam_delta',
            server_eta=0.03, huber_t0=0.1, huber_m=1.0, projection='identity_Rp',
            meaning='Explicit existing draft interior recipes for bounded pipeline diagnosis, not tuned optimal or formal protocol approval'),
        restrictions='No execution until separate matching dispatch/resource receipt; no GPU/test/64screen/automatic retry')
    write(HERE / 'scope.json', scope)
    write(HERE / 'dispatch_receipt.PENDING.json', dict(status='PENDING_NOT_AUTHORIZED',
        scope_sha256=digest(HERE / 'scope.json'), jobs={x['id']: x['job_sha256'] for x in entries},
        source_hashes=protection, local_hashes=local, exclusive_cpu_ids=None,
        verified_no_overlap_with_live_cpu_gates=False, formal_decisions_approved=False, screen64_authorized=False,
        test_authorized=False, authorized_by=None, authorized_at_utc=None))
    print('PREPARED_NOT_FROZEN: four real-image jobs; no allocation, dispatch, training or result')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--project', type=Path, required=True)
    prepare(parser.parse_args().project.resolve())
