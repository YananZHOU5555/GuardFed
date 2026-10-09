"""Prepare exact11 from a verified native71 snapshot. No scientific runtime import."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import sys
sys.dont_write_bytecode = True
import bridge as b
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PARENT = 'tmp/celeba_mechanism_valid_incremental_next37_20261009'
BACKUPS = 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG = 'root_delta_20261009T144558Z'
INSPECTION = BACKUPS + '/mechanism_inspection_v4_' + TAG + '/inspection.json'
BASELINE = 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json'
MANIFEST = 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json'
PROTOCOL = 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/PROTOCOL.md'
EVIDENCE = 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
PREVIOUS = 'tmp/celeba_mechanism_valid_incremental_next8_20261009'
PREVIOUS_SEAL_SHA = 'ecc09388f61e11fc768c6e667e57e615d984477aaf3c6221c3aad7b5ebc4f83f'
PREVIOUS_INVENTORY_SHA = '97a4ab5d086a959877147026bac9eb8f7f24f3d76a31747f9c66b524cf0bf188'
DELTA_IDS = ['minus_U_non-IID_F Flip_seed91009', 'minus_U_non-IID_F Flip_seed91010', 'minus_U_non-IID_FedSA_seed91001']
PARENT_SEAL_SHA = '95978fa42c28e9b4ff5b855b33c2dda56edc2b14fcfd56c3a29b0a9ba98135fd'
PARENT_INVENTORY_SHA = '0f837e22a1dd316fadbae85464f2dedbf074fcd5de4c96a3c974f426c7ee55d7'
ARCHIVE_SHA = 'b9de831bed4854445a0b5ad7b82426c5b7aa6dded4e40cb1a951c3033abc9aee'
RECEIPT_SHA = 'cf52d46862401e062251b456e6fb524398c088516e64527c86068a2500eb087f'
OFFSERVER_SHA = 'c110e3b32e9fc076b097c18716b7a31d83a34359b9fe43f7021afcfbeaa8a8f4'
PARENT_REPLAY_PROOF = PARENT + '/execution_candidate/backups/incremental_20261009T144055Z/ROOT_ADOPTION_REVIEW.json'
PARENT_REPLAY_PROOF_SHA = 'a7a563390de455914b33cf62064d674347e0c26754a778b9a344ac99bdc4fac3'
ROOT_SHA = '4202579dcf804c62e1f2356b07b7c529f7e7237deaa02344117c0fc843e509aa'


def prepare():
    workspace, output = WORKSPACE, HERE / 'inventory_actual71_Full100refs.json'
    archive_rel = BACKUPS + '/' + TAG + '.tar.gz'
    receipt_rel = archive_rel + '.receipt.json'
    verify_rel = BACKUPS + '/' + TAG + '_offserver_verification.json'
    root_rel = BACKUPS + '/' + TAG + '/ROOT_DELTA_VERIFICATION.json'
    pins = {BASELINE: b.BASELINE_INVENTORY_SHA, MANIFEST: b.MANIFEST_SHA,
            INSPECTION: b.INSPECTION_SHA, EVIDENCE: b.EVIDENCE_V4_SHA,
            'tmp/celeba_final_valid_replay_20261009/replay.py': b.V2_SHA,
            'tmp/celeba_final_valid_replay_20261009/v3/replay_v3.py': b.V3_SHA,
            'tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py': b.EVALUATOR_SHA,
            PREVIOUS + '/FILES_SHA256.json': PREVIOUS_SEAL_SHA, PREVIOUS + '/inventory_actual68_Full100refs.json': PREVIOUS_INVENTORY_SHA,
            PARENT + '/FILES_SHA256.json': PARENT_SEAL_SHA, PARENT_REPLAY_PROOF: PARENT_REPLAY_PROOF_SHA,
            PARENT + '/inventory_actual60_Full100refs.json': PARENT_INVENTORY_SHA,
            archive_rel: ARCHIVE_SHA, receipt_rel: RECEIPT_SHA, verify_rel: OFFSERVER_SHA, root_rel: ROOT_SHA}
    for rel, sha in pins.items():
        b.require(b.digest(workspace / rel) == sha, 'Input drift: ' + rel)
    parent_seal = b.read(workspace / PARENT / 'FILES_SHA256.json')
    b.require(len(parent_seal['members']) == 16, 'Parent science package membership changed')
    for member in parent_seal['members']:
        path = workspace / PARENT / member['path']
        b.require(b.digest(path) == member['sha256'] and path.stat().st_size == member['size'], 'Parent source/member drift')
        pins[PARENT + '/' + member['path']] = member['sha256']
    for member in b.read(workspace / PREVIOUS / 'FILES_SHA256.json')['members']:
        path = workspace / PREVIOUS / member['path']
        b.require(b.digest(path) == member['sha256'] and path.stat().st_size == member['size'], 'Previous prepared member drift')
        pins[PREVIOUS + '/' + member['path']] = member['sha256']
    prior_replay = b.read(workspace / PARENT_REPLAY_PROOF)
    b.require(prior_replay['status'] == 'ROOT_NEXT37_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
              and prior_replay['cumulative_three_view_models'] == 60 and prior_replay['prior_three_view_models'] == 28
              and prior_replay['accepted_new'] == 32 and prior_replay['science_seal_sha256'] == PARENT_SEAL_SHA
              and prior_replay['source_scope_complete'] and prior_replay['all_native_differences_zero']
              and prior_replay['original23_unchanged'], 'Prior60 root closure receipt mismatch')
    prior = b.read(workspace / PREVIOUS / 'inventory_actual68_Full100refs.json')
    b.require(set(prior_replay['accepted_new_ids']) <= set(b.EXCLUDED_PRIOR_IDS), 'Foreign prior replay IDs')
    b.require(prior['excluded_prior_replay_ids'] == b.EXCLUDED_PRIOR_IDS and len(prior['records']) == 68, 'Prior60 exclusion mismatch')
    evidence = b.load('next11_original_evidence_v4', workspace / EVIDENCE, b.EVIDENCE_V4_SHA)
    inspection, manifest, baseline = (b.read(workspace / p) for p in (INSPECTION, MANIFEST, BASELINE))
    b.require(b.digest(workspace / PROTOCOL) == manifest['protocol_sha256'], 'Mechanism protocol changed')
    pins[PROTOCOL] = manifest['protocol_sha256']
    b.require(inspection['source_script_sha256'] == b.EVIDENCE_V4_SHA and inspection['manifest_sha256'] == b.MANIFEST_SHA
              and inspection['full_inventory_sha256'] == b.BASELINE_INVENTORY_SHA, 'Inspection authority changed')
    b.require(not inspection['invalid'] and inspection['new_count'] == 71 and inspection['reused_count'] == 100, 'Wrong native71 snapshot')
    rows = {r['id']: r for r in inspection['records'] if r['role'] == 'new'}
    b.require(len(rows) == 71 and set(rows) == set(inspection['accepted_new_ids']) == set(b.ACCEPTED_IDS), 'Actual71 IDs mismatch')
    full = {b.cell(r): r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+'}
    b.require(len(full) == 100 and {r['id'] for r in full.values()} == set(inspection['accepted_reused_ids']), 'Full reference set differs')
    entries = {r['id']: r for r in manifest['jobs']}
    artifacts = {r['id']: copy.deepcopy(r) for r in prior['records']}
    for model_id, record in artifacts.items():
        b.require(record['accepted_v4_row'] == rows[model_id], 'Prior native acceptance identity differs in new inspection')
    archive, receipt, proof, root = workspace / archive_rel, b.read(workspace / receipt_rel), b.read(workspace / verify_rel), b.read(workspace / root_rel)
    b.require(receipt['accepted_new_ids'] == proof['accepted_new_ids'] == root['new_ids'] == DELTA_IDS, 'Exact3 new native scope mismatch')
    b.require(root['status'] == 'ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and root['total_new_strict_and_offserver'] == 71
              and root['archive_sha256'] == ARCHIVE_SHA and root['receipt_sha256'] == RECEIPT_SHA
              and root['offserver_proof_sha256'] == OFFSERVER_SHA and root['inspection_sha256'] == b.INSPECTION_SHA,
              'Root native acceptance receipt mismatch')
    b.require(root['previous_ledger_sha256'] == 'b790825e946dcef375d2b31d4c661733fcd016c723455b76f86e8361b32e46a6', 'Parent ledger chain mismatch')
    b.require(proof['pass'] and proof['different_host_observed'] and proof['source_host'] == receipt['source_host']
              and proof['archive_sha256'] == receipt['archive_sha256'] == ARCHIVE_SHA, 'Offserver receipt mismatch')
    b.require(receipt['previous_receipt_sha256'] == prior['backup_chain'][-1]['receipt_sha256']
              and receipt['manifest_sha256'] == b.MANIFEST_SHA and receipt['inspection_sha256'] == b.INSPECTION_SHA,
              'Broken native backup chain')
    b.require(not set(artifacts).intersection(DELTA_IDS) and set(rows) - set(artifacts) == set(DELTA_IDS), 'New native delta differs from exact3')
    verified = evidence.verify_archive(archive, receipt)
    b.require(verified['members_verified'] == proof['members_verified'] and receipt['reused_full_weights_repacked'] == 0
              and not receipt['failure_identities'], 'Incomplete/invalid incremental evidence')
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
    inventory = copy.deepcopy(prior)
    inventory.update(schema='celeba_mechanism_valid_replay_inventory_next11', scope=b.SCOPE, status='PREPARED_NOT_APPROVED',
                     mechanism_inspection_sha256=b.INSPECTION_SHA,
                     records=[artifacts[k] for k in sorted(artifacts)],
                     excluded_prior_replay_ids=b.EXCLUDED_PRIOR_IDS, selected_replay_ids=b.REPLAY_IDS,
                     pending_new_ids_no_checkpoint=sorted(set(entries) - set(artifacts)),
                     counts={'planned_new': 800, 'actual_accepted_new': 71, 'Full_reference_only': 100, 'pending_new_without_checkpoint': 729},
                     input_pins=pins, history='Full100:98cu128+2cu130; native71:cu128. No new replay runtime is claimed; CPU/CUDA equivalence is not assumed.',
                     activation='PREPARED ONLY. Prior60 root adoption is recorded separately; exact11 source-bound external approval is still required. No runtime installer or dispatch is supplied.')
    inventory['prior_replay_boundary'] = {'status': 'PRIOR60_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND', 'prior_inventory_sha256': PARENT_INVENTORY_SHA,
                                          'prior_science_seal_sha256': PARENT_SEAL_SHA, 'root_adoption_receipt': PARENT_REPLAY_PROOF,
                                          'root_adoption_receipt_sha256': PARENT_REPLAY_PROOF_SHA, 'next11_execution_approved': False}
    inventory['backup_chain'].append({'archive': archive_rel, 'archive_sha256': ARCHIVE_SHA,
                                     'receipt': receipt_rel, 'receipt_sha256': RECEIPT_SHA, 'verification': verify_rel,
                                     'verification_sha256': OFFSERVER_SHA, 'accepted_new_ids': DELTA_IDS,
                                     'members_verified_now_without_inference': verified['members_verified']})
    b.validate_inventory(inventory, baseline)
    b.require(all(artifacts[r['id']] == r for r in prior['records']), 'Prior68 records reconstructed/modified')
    for rel, sha in pins.items():
        b.require(b.digest(workspace / rel) == sha, 'Input changed during preparation: ' + rel)
    b.require('torch' not in sys.modules and 'numpy' not in sys.modules, 'Scientific runtime imported')
    b.save_new(output, inventory)
    return {'status': 'PREPARED_NOT_APPROVED', 'inventory_sha256': b.digest(output), 'actual_native_records': 71, 'selected_replay_count': 11,
            'excluded_prior_count': 60, 'prior60_root_replay_adoption_receipt_bound': True, 'next11_execution_approved': False, 'Full_reference_only': 100,
            'pending_native_without_checkpoint': 729, 'new_archive_members_verified': verified['members_verified'],
            'parent_source_members_verified': 16, 'prior68_records_exactly_preserved': True, 'prior60_records_exactly_preserved': True, 'previous_prepared_seal_sha256': PREVIOUS_SEAL_SHA,
            'original_bytes_boundary': 'New3 result/job/model archive members verified byte-exact; inventory is a reconstructed identity document. Original accepted runtime is copied from result, never inferred.',
            'image_inference_performed': False, 'training_performed': False, 'tensor_load_performed': False, 'Full_weights_copied': 0}


if __name__ == '__main__':
    result = prepare()
    b.save_new(HERE / 'preparation_check.json', result)
    print(json.dumps(result))
