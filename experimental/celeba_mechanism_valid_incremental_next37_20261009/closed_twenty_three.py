"""Freeze prior23 identities from already accepted source/proofs; no new result acceptance."""
import hashlib
import json
from pathlib import Path
import tarfile
import bridge as b

PARENT = 'tmp/celeba_mechanism_valid_incremental_v2_20261009'
EXECUTION = 'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009'


def verify_closed_twenty_three(workspace, evidence, actual_records):
    workspace = Path(workspace); pins = {}; records = {}; proofs = []
    def pinned(rel, expected):
        path = workspace / rel
        b.require(b.digest(path) == expected, 'Closed23 frozen source/proof changed: ' + rel)
        pins[rel] = expected
        return path
    parent = b.read(pinned(PARENT + '/inventory_actual23_Full100refs.json', '288d2afb260f7eb77bcccba82e7edf6dbfe0519cd5bcc42fb546496236eeadbc'))
    pinned(PARENT + '/FILES_SHA256.json', '70d0d920c4c5351c42efc9968fe3c38eed431d208b94bc8af486ba49d869a42d')
    pinned(PARENT + '/bridge.py', '2950199b48e1b10a131ee8dafb992e44e002915b475976fa4525a63f25cc35b3')
    # The original complete15 delivery is a frozen identity input, not a new acceptance.
    delivery = b.read(pinned(EXECUTION + '/FINAL_DELIVERY.json', DELIVERY_SHA))
    b.require(delivery['status'] == 'COMPLETE_15_STRICT_VALID_THREE_VIEWS_OFFSERVER_VERIFIED'
              and delivery['accepted_new'] == 15 and delivery['mechanism_replayed_from_this_snapshot_total'] == 23
              and delivery['native_tolerance'] == b.TOLERANCE and delivery['max_abs_native_difference'] <= b.TOLERANCE,
              'Original complete15 proof is incomplete')
    pinned(EXECUTION + '/EXECUTION_SOURCE_SHA256.json', delivery['execution_seal_sha256'])
    pinned(EXECUTION + '/APPROVED.json', delivery['actual_approval_sha256'])
    old = {r['id']: r for r in parent['records']}
    b.require(set(old) == set(b.CLOSED_IDS), 'Closed23 source snapshot differs')
    for identity, previous in old.items():
        current = actual_records[identity]
        for kind in ('checkpoint', 'result', 'raw_job'):
            b.require(all(previous[kind][k] == current[kind][k] for k in ('member', 'sha256', 'bytes')), 'Closed checkpoint/result/job changed')
        for key in ('config', 'source_hashes', 'adapter_source_hashes', 'data_contract', 'prior_validation_metrics', 'paired_full', 'runtime_paths'):
            b.require(previous[key] == current[key], 'Closed23 scientific identity changed: ' + key)
    closure = parent['closed_replay_evidence']
    for rel, expected in closure['input_pins'].items(): pinned(rel, expected)
    for old_record in closure['records']:
        identity = old_record['id']
        with tarfile.open(workspace / old_record['archive']) as bundle:
            data = bundle.extractfile(old_record['strict_acceptance_member']).read()
        b.require(hashlib.sha256(data).hexdigest() == old_record['strict_acceptance_sha256'], 'Closed8 strict member changed')
        accepted = json.loads(data)
        b.require(accepted['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and accepted['id'] == identity
                  and accepted['checkpoint_sha256'] == actual_records[identity]['checkpoint']['sha256']
                  and accepted['native_comparison']['accepted'] and set(accepted['views']) == set(b.VIEWS), 'Closed8 terminal/views differ')
        records[identity] = old_record
    proofs.extend(closure['archive_proofs'])
    previous_receipt = None; prior_ids = []
    for chain in delivery['archive_chain']:
        prefix = EXECUTION + '/' + chain['directory']
        receipt_path = pinned(prefix + '/backup_receipt.json', chain['receipt_sha256'])
        proof_path = pinned(prefix + '/OFFSERVER_VERIFICATION.json', chain['offserver_proof_sha256'])
        archive = pinned(prefix + '/incremental_valid_three_views.tar.gz', chain['archive_sha256'])
        receipt, proof = b.read(receipt_path), b.read(proof_path)
        ids = receipt['accepted_new_ids']
        b.require(proof['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
                  and proof['accepted_new_ids'] == ids == chain['new_ids'] and proof['archive_sha256'] == receipt['archive_sha256'] == chain['archive_sha256']
                  and proof['source_seal_sha256'] == delivery['execution_seal_sha256'], 'Closed15 offserver archive identity differs')
        b.require(receipt['previous_backup_receipt_sha256'] == previous_receipt and not set(ids).intersection(records)
                  and receipt['all_accepted_ids'] == prior_ids + ids, 'Closed15 backup chain/duplicate differs')
        with tarfile.open(archive) as bundle:
            data = bundle.extractfile('backup_inventory.json').read(); inventory = json.loads(data)
            b.require(hashlib.sha256(data).hexdigest() == receipt['inventory_sha256'] and inventory['accepted_new_ids'] == ids
                      and len(bundle.getnames()) == len(set(bundle.getnames())) == receipt['members']
                      and set(bundle.getnames()) == set(inventory['members']) | {'backup_inventory.json'}
                      and inventory['models_repacked'] == 0, 'Closed15 archive manifest changed')
            for identity in ids:
                member = 'runs/' + identity + '/strict_acceptance.json'; data = bundle.extractfile(member).read(); accepted = json.loads(data)
                b.require(hashlib.sha256(data).hexdigest() == inventory['members'][member]['sha256']
                          and accepted['status'] == 'MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and accepted['id'] == identity
                          and accepted['checkpoint_sha256'] == actual_records[identity]['checkpoint']['sha256']
                          and accepted['inventory_sha256'] == '288d2afb260f7eb77bcccba82e7edf6dbfe0519cd5bcc42fb546496236eeadbc'
                          and set(accepted['views']) == set(b.VIEWS) and accepted['native_comparison']['accepted'], 'Closed15 strict terminal identity differs')
                records[identity] = dict(id=identity, checkpoint_sha256=accepted['checkpoint_sha256'], strict_acceptance_member=member,
                    strict_acceptance_sha256=hashlib.sha256(data).hexdigest(), archive=prefix + '/incremental_valid_three_views.tar.gz', archive_sha256=chain['archive_sha256'])
        proofs.append(dict(archive=prefix + '/incremental_valid_three_views.tar.gz', archive_sha256=chain['archive_sha256'], accepted_ids=ids,
                           proof_sha256=chain['offserver_proof_sha256'], members_verified_by_original_offserver_proof=proof['members_verified']))
        previous_receipt = chain['receipt_sha256']; prior_ids += ids
    b.require(set(records) == set(b.CLOSED_IDS) and prior_ids == delivery['selected_ids'], 'Closed23 proof coverage differs')
    return dict(status='PRIOR23_OFFSERVER_IDENTITIES_FROZEN_NO_NEW_ACCEPTANCE_OR_INFERENCE', records=[records[i] for i in b.CLOSED_IDS],
                archive_proofs=proofs, input_pins=pins, original_complete15_delivery_sha256=DELIVERY_SHA)


DELIVERY_SHA = '8098ae66620adc85c3168aa16df78880f3462691708f5e91655b7e216d32cf16'
