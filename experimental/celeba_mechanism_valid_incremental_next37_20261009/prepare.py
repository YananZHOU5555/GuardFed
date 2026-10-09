"""Build an actual60-terminal /100-reference inventory without model inference."""
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
BACKUPS = 'tmp/revision-publish-20260928/docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
INSPECTION = BACKUPS + '/mechanism_inspection_v4_root_delta_20261009T133319Z/inspection.json'
BASELINE = 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json'
MANIFEST = 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json'
PROTOCOL = 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/PROTOCOL.md'
EVIDENCE = 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
LEDGER = 'tmp/celeba_mechanism_valid_incremental_next37_20261009/inputs/verified_ledger_60.json'
CHAIN = [('incremental_new5_v3_20261009T073000Z.tar.gz', 'c7f3f4d14bc9a1ca61570945fc3f0d38840b889368363b76815899461fbb9e4c', 'incremental_new5_offserver_verification.json', '8c66ad61d24e2870b7b582504deceb1791f7fdc8864aa97bfb3ba1b3c1ef4a3f'), ('incremental_new3_total8_v4_20261009T074500Z.tar.gz', 'c0754e1129e5615daec99e885f718a6dd06835f6c19b004e5ad95e50e92d485b', 'incremental_new3_total8_offserver_verification.json', 'b10556da06d21aeb7061c662dc845e4f34d8b21520c81adb86392c9ae3e99104'), ('incremental_v4_20261009T081900Z.tar.gz', 'e17f2dae47faa3654497cac9e2120ef82dce9e6fcf592e0f2078d8d4c4f4aae2', 'incremental_v4_20261009T081900Z_offserver_verification.json', '534cff45c37d2f62b0e9ca42ce984effba0e64d00977359c663ca04e1dbc8bd3'), ('incremental_v4_20261009T084000Z.tar.gz', 'e93cd94fdea947ed1b816148fead116d6ee620f8553b50e1d4f55e2e8714ca50', 'incremental_v4_20261009T084000Z_offserver_verification.json', '648c04a198df0483bb6dde88e4835d5865b2162891738bbf2c95b40b24892fa2'), ('incremental_v4_20261009T092000Z.tar.gz', '82a5b55db58be3fa4ffd978d8a00545689f1435bc3eefa5ffdfbc883b7a8493b', 'incremental_v4_20261009T092000Z_offserver_verification.json', '4678daab66b2657f2e2345362f2d919cc64fd97b397042f0f7c4e8e50296a86c'), ('incremental_v4_20261009T101427Z.tar.gz', '3207ecb140d33130ae77880f58df91be83bf60100e278dfbf1c43ea1955f8f87', 'incremental_v4_20261009T101427Z_offserver_verification.json', 'd221004d69b38c80fd754d65e14d7e20abbfac85c182d665424ab188b7c9a05a'), ('incremental_v4_20261009T105813Z.tar.gz', '00ba1beb3e92909961817ad206156b6e1ac025f180f4e11afef5cbab276dc196', 'incremental_v4_20261009T105813Z_offserver_verification.json', '2745bc027d32bb1c0bcc5346c4eda811823913261e3361402c4646e9d4b5469c'), ('root_live_20261009T113229Z_incremental_new2_total40.tar.gz', '3f494748e233b4a446baefd52e96e12431b169bda0a3d6181a0e40b42b01cffd', 'root_live_20261009T113229Z_offserver_verification.json', '168cd0407d87a95bc010dcaf2eac283be40c9a327519d4cee74b8cfefc19896b'), ('root_delta_20261009T124918Z.tar.gz', 'a0e741547ecf190d9fa5f7029c0aa2cb46d21718c0726350f13d10f6816b75cc', 'root_delta_20261009T124918Z_offserver_verification.json', '286cf283761300b0d233c07d07a6f1d8cc6cc4e9bfcb9696513ecbb6e50755c3'), ('root_delta_20261009T133319Z.tar.gz', '13a27d11ea591aa543341bbd549bd4e8584019e28d16631fcea77a1dbfe4b48b', 'root_delta_20261009T133319Z_offserver_verification.json', '64621342864155c8175d655f3e00883802984fc1ed38363f7ff65f3438fc97dc')]


def prepare(workspace, output):
    workspace, output = Path(workspace), Path(output)
    pins = {BASELINE: b.BASELINE_INVENTORY_SHA, MANIFEST: b.MANIFEST_SHA,
            INSPECTION: b.INSPECTION_SHA, EVIDENCE: b.EVIDENCE_V4_SHA,
            'tmp/celeba_final_valid_replay_20261009/replay.py': b.V2_SHA,
            'tmp/celeba_final_valid_replay_20261009/v3/replay_v3.py': b.V3_SHA,
            'tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py': b.EVALUATOR_SHA,
            LEDGER: '0bf78e803b5e77f1becd383e10010275ad0d146fcc7bb32b0dc12f014790fe9f'}
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
    b.require(not inspection['invalid'] and inspection['new_count'] == 60 and inspection['reused_count'] == 100, 'Wrong accepted inspection cohort')
    rows = {r['id']: r for r in inspection['records'] if r['role'] == 'new'}
    b.require(len(rows) == 60 and set(rows) == set(inspection['accepted_new_ids']), 'Duplicate/missing accepted rows')
    full = {b.cell(r): r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+'}
    b.require(len(full) == 100 and {r['id'] for r in full.values()} == set(inspection['accepted_reused_ids']), 'Full reference set differs')
    entries = {r['id']: r for r in manifest['jobs']}
    ledger = b.read(workspace / LEDGER)
    b.require(len(ledger['entries']) == 10 and ledger['manifest_sha256'] == b.MANIFEST_SHA, 'Wrong ledger snapshot')
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
    from closed_twenty_three import verify_closed_twenty_three
    closure = verify_closed_twenty_three(workspace, evidence, artifacts)
    pins.update(closure['input_pins'])
    inventory = {'schema': 'celeba_mechanism_valid_replay_inventory_next37', 'scope': b.SCOPE, 'status': 'PREPARED_NOT_DISPATCHED',
                 'baseline_inventory_sha256': b.BASELINE_INVENTORY_SHA, 'mechanism_manifest_sha256': b.MANIFEST_SHA,
                 'mechanism_inspection_sha256': b.INSPECTION_SHA, 'mechanism_acceptance_source_sha256': b.EVIDENCE_V4_SHA,
                 'mechanism_protocol_sha256': manifest['protocol_sha256'], 'mechanism_adapter_hashes': manifest['adapter_hashes'],
                 'mechanism_source_hashes': manifest['source_hashes'], 'native_tolerance': b.TOLERANCE, 'views': b.VIEWS,
                 'records': [artifacts[k] for k in sorted(artifacts)],
                 'closed_replay_ids': b.CLOSED_IDS, 'selected_replay_ids': b.REPLAY_IDS, 'closed_replay_evidence': closure,
                 'full_references': sorted((b.full_reference(r) for r in full.values()), key=lambda r: r['id']),
                 'pending_new_ids_no_checkpoint': sorted(set(entries) - accepted),
                 'counts': {'planned_new': 800, 'actual_accepted_new': 60, 'Full_reference_only': 100, 'pending_new_without_checkpoint': 740},
                 'backup_chain': chain_proofs, 'input_pins': pins,
                 'new_image_inference_performed': False, 'new_training_performed': False, 'full_weights_repacked': 0,
                 'all900_mechanism_views_complete': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN',
                 'label_boundary': 'Original training loader materialized full-split Smiling/Male metadata. Reused replay reads only train+valid scalar prefixes and never test images/labels for inference/fitting/scoring; no untouched-test claim.',
                 'history': 'Full100:98cu128+2cu130; new60:cu128; CPU replay runtime not assumed CUDA-equivalent. Seed91001 selected recipe; remaining validation seeds previously observed.',
                 'activation': 'No dispatcher supplied. Parent code review and separately SHA-approved CPU allocation required before bind_runtime/replay_one.'}
    b.validate_inventory(inventory, baseline)
    for rel, sha in pins.items():
        b.require(b.digest(workspace / rel) == sha, 'Input changed during inventory preparation: ' + rel)
    b.save_new(output, inventory)
    return {'status': 'PREPARED_NOT_DISPATCHED', 'inventory_sha256': b.digest(output), 'actual_new': 60, 'Full_reference_only': 100,
            'missing_new_without_checkpoint': 740, 'archive_members_checked': sum(r['members_verified_now_without_inference'] for r in chain_proofs),
            'image_inference_performed': False, 'training_performed': False, 'Full_weights_copied': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=WORKSPACE)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'inventory_actual60_Full100refs.json')
    args = p.parse_args()
    result = prepare(args.workspace, args.output)
    b.save_new(args.output.with_name('preparation_check.json'), result)
    print(json.dumps(result))
