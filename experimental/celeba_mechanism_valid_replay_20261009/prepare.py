"""Build an actual8-terminal /100-reference inventory without model inference."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import sys
sys.dont_write_bytecode = True
import bridge as b

WORKSPACE = Path(__file__).resolve().parents[2]
BACKUPS = 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
INSPECTION = BACKUPS + '/mechanism_inspection_new8_v4_20261009T074500Z/inspection.json'
BASELINE = 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json'
MANIFEST = 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json'
PROTOCOL = 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/PROTOCOL.md'
EVIDENCE = 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
CHAIN = [
    ('incremental_new5_v3_20261009T073000Z.tar.gz', 'c7f3f4d14bc9a1ca61570945fc3f0d38840b889368363b76815899461fbb9e4c',
     'incremental_new5_offserver_verification.json', '8c66ad61d24e2870b7b582504deceb1791f7fdc8864aa97bfb3ba1b3c1ef4a3f'),
    ('incremental_new3_total8_v4_20261009T074500Z.tar.gz', 'c0754e1129e5615daec99e885f718a6dd06835f6c19b004e5ad95e50e92d485b',
     'incremental_new3_total8_offserver_verification.json', 'b10556da06d21aeb7061c662dc845e4f34d8b21520c81adb86392c9ae3e99104'),
]


def prepare(workspace, output):
    workspace, output = Path(workspace), Path(output)
    pins = {BASELINE: b.BASELINE_INVENTORY_SHA, MANIFEST: b.MANIFEST_SHA,
            INSPECTION: b.INSPECTION_SHA, EVIDENCE: b.EVIDENCE_V4_SHA,
            'tmp/celeba_final_valid_replay_20261009/replay.py': b.V2_SHA,
            'tmp/celeba_final_valid_replay_20261009/v3/replay_v3.py': b.V3_SHA,
            'tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py': b.EVALUATOR_SHA,
            BACKUPS + '/verified_ledger.json': '9d8384425cdae353cbfed213c8b14c5c0e5772b6f53385e553c153ba588d215f'}
    for rel, sha in pins.items():
        b.require(b.digest(workspace / rel) == sha, 'Input drift: ' + rel)
    evidence = b.load('bridge_prepare_v4', workspace / EVIDENCE, b.EVIDENCE_V4_SHA)
    inspection = b.read(workspace / INSPECTION)
    manifest = b.read(workspace / MANIFEST)
    baseline = b.read(workspace / BASELINE)
    b.require(b.digest(workspace / PROTOCOL) == manifest['protocol_sha256'], 'Mechanism protocol changed')
    pins[PROTOCOL] = manifest['protocol_sha256']
    b.require(inspection['source_script_sha256'] == b.EVIDENCE_V4_SHA and inspection['manifest_sha256'] == b.MANIFEST_SHA
              and inspection['full_inventory_sha256'] == b.BASELINE_INVENTORY_SHA, 'Inspection authority changed')
    b.require(not inspection['invalid'] and inspection['new_count'] == 8 and inspection['reused_count'] == 100, 'Wrong accepted inspection cohort')
    rows = {r['id']: r for r in inspection['records'] if r['role'] == 'new'}
    b.require(len(rows) == 8 and set(rows) == set(inspection['accepted_new_ids']), 'Duplicate/missing accepted rows')
    full = {b.cell(r): r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+'}
    b.require(len(full) == 100 and {r['id'] for r in full.values()} == set(inspection['accepted_reused_ids']), 'Full reference set differs')
    entries = {r['id']: r for r in manifest['jobs']}
    ledger = b.read(workspace / BACKUPS / 'verified_ledger.json')
    b.require(len(ledger['entries']) == 2 and ledger['manifest_sha256'] == b.MANIFEST_SHA, 'Wrong ledger snapshot')
    accepted, chain_proofs, artifacts = set(), [], {}
    previous = None
    for ledger_entry, (archive_name, receipt_sha, verification_name, verification_sha) in zip(ledger['entries'], CHAIN):
        archive_rel = BACKUPS + '/' + archive_name
        receipt_rel = archive_rel + '.receipt.json'
        verify_rel = BACKUPS + '/' + verification_name
        archive, receipt = workspace / archive_rel, b.read(workspace / receipt_rel)
        b.require(b.digest(workspace / receipt_rel) == receipt_sha == ledger_entry['receipt_sha256'], 'Receipt SHA differs from ledger')
        b.require(PurePosixPath(ledger_entry['archive']).name == archive_name and PurePosixPath(ledger_entry['receipt']).name == archive_name + '.receipt.json', 'Wrong ledger artifact path')
        b.require(b.digest(workspace / verify_rel) == verification_sha, 'Offserver verification changed')
        proof = b.read(workspace / verify_rel)
        b.require(proof['pass'] and proof['different_host_observed'] and proof['source_host'] == receipt['source_host']
                  and proof['archive_sha256'] == receipt['archive_sha256'] and proof['accepted_new_ids'] == receipt['accepted_new_ids'], 'Offserver acceptance does not bind receipt/archive')
        b.require(receipt['previous_receipt_sha256'] == previous and receipt['manifest_sha256'] == b.MANIFEST_SHA, 'Broken incremental chain')
        b.require(not accepted.intersection(receipt['accepted_new_ids']) and set(receipt['accepted_new_ids']) <= set(rows), 'Duplicate/undeclared archived terminal')
        verified = evidence.verify_archive(archive, receipt)
        b.require(verified['members_verified'] == proof['members_verified'] and receipt['reused_full_weights_repacked'] == 0 and not receipt['failure_identities'], 'Incomplete/invalid incremental evidence')
        pins.update({archive_rel: receipt['archive_sha256'], receipt_rel: receipt_sha, verify_rel: verification_sha})
        with tarfile.open(archive, 'r:gz') as tar:
            backup_inventory = json.load(tar.extractfile('backup_inventory.json'))
            b.require(backup_inventory['reused_full_weights_repacked'] == 0, 'Full weights were repacked')
            archived_inspection = tar.extractfile('sourcefreeze/inspection.json').read()
            b.require(hashlib.sha256(archived_inspection).hexdigest() == receipt['inspection_sha256'], 'Archive inspection receipt differs')
            b.require(hashlib.sha256(tar.extractfile('sourcefreeze/manifest.json').read()).hexdigest() == b.MANIFEST_SHA, 'Archived manifest differs')
            for model_id in receipt['accepted_new_ids']:
                result = json.load(tar.extractfile('runs/' + model_id + '/result.json'))
                job = json.load(tar.extractfile('jobs/' + model_id + '.json'))
                row, entry = rows[model_id], entries[model_id]
                control = full[(job['distribution'], job['attack'], job['config']['seed'])]
                # These unchanged v4 checks use real terminal JSON, no tensors,
                # images, model forward calls or scientific result reacceptance.
                evidence.terminal_checks(result, job, manifest, job['variant'])
                evidence.partition_identity(result, control)
                b.require(result['revision_job']['variant'] == job['variant'] and result['config'] == job['config'], 'Original variant/config mismatch')
                b.require(result['revision_job']['checkpoint_sha256'] == row['checkpoint_sha256'] and result['revision_job']['torch_version'] == '2.11.0+cu128', 'Terminal checkpoint/environment mismatch')
                b.require(job['adapter_hashes'] == result['revision_job']['adapter_hashes'] == manifest['adapter_hashes']
                          and job['source_hashes'] == manifest['source_hashes'] and job['protocol_sha256'] == manifest['protocol_sha256'], 'Archived adapter/source/protocol differs')
                refs = {}
                for kind, member in [('checkpoint', 'runs/' + model_id + '/model.pt'), ('result', 'runs/' + model_id + '/result.json'), ('raw_job', 'jobs/' + model_id + '.json')]:
                    refs[kind] = {'archive': archive_rel, 'archive_sha256': receipt['archive_sha256'],
                                  'member': member, **backup_inventory['members'][member]}
                record = {'id': model_id, 'variant': job['variant'], 'method': job['method'], 'source_method': result['method'],
                          'distribution': job['distribution'], 'actual_alpha': result['alpha'], 'attack': job['attack'], 'seed': result['seed'],
                          'terminal_round': 70, 'original_split': 'valid', 'original_n_eval': 19867,
                          'config': job['config'], 'config_canonical_sha256': b.canonical(job['config']),
                          'source_hashes': job['source_hashes'], 'adapter_source_hashes': job['adapter_hashes'],
                          'protocol_sha256': job['protocol_sha256'], 'data_contract': result['data_contract']['image_data_contract'],
                          'original_remote_output': job['output'], 'original_job': entry['job'],
                          'runtime_paths': {'checkpoint': job['output'] + '/model.pt', 'result': job['output'] + '/result.json', 'raw_job': entry['job']},
                          'training_torch': result['revision_job']['torch_version'], 'prior_validation_metrics': result['metrics'],
                          'native_prediction_rule': 'root_fitted_group_thresholds_from_original_recipe',
                          'paired_full': b.full_reference(control), 'manifest_entry': copy.deepcopy(entry),
                          'accepted_v4_row': copy.deepcopy(row), **refs}
                artifacts[model_id] = record
        accepted.update(receipt['accepted_new_ids'])
        chain_proofs.append({'archive': archive_rel, 'archive_sha256': receipt['archive_sha256'],
                             'receipt': receipt_rel, 'receipt_sha256': receipt_sha, 'verification': verify_rel,
                             'verification_sha256': verification_sha, 'accepted_new_ids': receipt['accepted_new_ids'],
                             'members_verified_now_without_inference': verified['members_verified']})
        previous = receipt_sha
    b.require(accepted == set(rows), 'Offserver archive chain omits an accepted terminal')
    inventory = {'schema': 'celeba_mechanism_valid_replay_inventory_v1', 'scope': b.SCOPE, 'status': 'PREPARED_NOT_DISPATCHED',
                 'baseline_inventory_sha256': b.BASELINE_INVENTORY_SHA, 'mechanism_manifest_sha256': b.MANIFEST_SHA,
                 'mechanism_inspection_sha256': b.INSPECTION_SHA, 'mechanism_acceptance_source_sha256': b.EVIDENCE_V4_SHA,
                 'mechanism_protocol_sha256': manifest['protocol_sha256'], 'mechanism_adapter_hashes': manifest['adapter_hashes'],
                 'mechanism_source_hashes': manifest['source_hashes'], 'native_tolerance': b.TOLERANCE, 'views': b.VIEWS,
                 'records': [artifacts[k] for k in sorted(artifacts)],
                 'full_references': sorted((b.full_reference(r) for r in full.values()), key=lambda r: r['id']),
                 'pending_new_ids_no_checkpoint': sorted(set(entries) - accepted),
                 'counts': {'planned_new': 800, 'actual_accepted_new': 8, 'Full_reference_only': 100, 'pending_new_without_checkpoint': 792},
                 'backup_chain': chain_proofs, 'input_pins': pins,
                 'new_image_inference_performed': False, 'new_training_performed': False, 'full_weights_repacked': 0,
                 'all900_mechanism_views_complete': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN',
                 'label_boundary': 'Original training loader materialized full-split Smiling/Male metadata. Reused replay reads only train+valid scalar prefixes and never test images/labels for inference/fitting/scoring; no untouched-test claim.',
                 'history': 'Full100:98cu128+2cu130; new8:cu128; CPU replay runtime not assumed CUDA-equivalent. Seed91001 selected recipe; remaining validation seeds previously observed.',
                 'activation': 'No dispatcher supplied. Parent code review and separately SHA-approved CPU allocation required before bind_runtime/replay_one.'}
    b.validate_inventory(inventory, baseline)
    for rel, sha in pins.items():
        b.require(b.digest(workspace / rel) == sha, 'Input changed during inventory preparation: ' + rel)
    b.save_new(output, inventory)
    return {'status': 'PREPARED_NOT_DISPATCHED', 'inventory_sha256': b.digest(output), 'actual_new': 8, 'Full_reference_only': 100,
            'missing_new_without_checkpoint': 792, 'archive_members_checked': sum(r['members_verified_now_without_inference'] for r in chain_proofs),
            'image_inference_performed': False, 'training_performed': False, 'Full_weights_copied': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=WORKSPACE)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'inventory_actual8_Full100refs.json')
    args = p.parse_args()
    result = prepare(args.workspace, args.output)
    b.save_new(args.output.with_name('preparation_check.json'), result)
    print(json.dumps(result))
