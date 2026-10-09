"""Bind the eight completed prior replays to existing offserver archive evidence."""
import hashlib
import json
from pathlib import Path
import tarfile

import bridge as b

OLD = 'tmp/celeba_mechanism_valid_replay_20261009/'
COHORTS = [
    dict(folder=OLD+'recovery_attempt03_20261009T085000Z/accepted_one_backup_20261009/',
         archive='accepted_one_valid_replay.tar.gz', receipt='60b50dea5641ac3d1e7c3a0ca5a948754aeb3556af1e9bb038b9fef8dfe1211a',
         verification='a546db15564b45adec76ffa411d654fbd684f94f92d6ca704b6c7887f11a7518',
         count=1),
    dict(folder=OLD+'remaining_seven_completed_backup_20261009/',
         archive='remaining_seven_valid_three_views.tar.gz', receipt='bbecaf432c0f6651b39f69f358edd3290563d42cd0952cc5d108c2fe100b156d',
         verification='b3d0a55ed50db5448235495517b172ddb7cdc09c46f659925c10e3bcb89e29ce',
         count=7),
]


def verify_closed_eight(workspace, evidence, actual_records):
    pins, proofs, records = {}, [], {}
    for cohort in COHORTS:
        prefix=cohort['folder']
        receipt_path=prefix+'backup_receipt.json'
        verification_path=prefix+'offserver_verification.json'
        b.require(b.digest(workspace/receipt_path)==cohort['receipt'], 'Prior replay backup receipt changed')
        b.require(b.digest(workspace/verification_path)==cohort['verification'], 'Prior replay offserver proof changed')
        receipt, proof=b.read(workspace/receipt_path), b.read(workspace/verification_path)
        b.require(proof['pass'] and proof['different_host_observed'] and proof['archive_sha256']==receipt['archive_sha256']
                  and proof['accepted_new_ids']==receipt['accepted_new_ids'], 'Prior replay offserver chain mismatch')
        archive_path=prefix+cohort['archive']
        checked=evidence.verify_archive(workspace/archive_path,receipt)
        b.require(checked['members_verified']==proof['members_verified'] and len(receipt['accepted_new_ids'])==cohort['count'], 'Prior replay cohort incomplete')
        with tarfile.open(workspace/archive_path) as archive:
            for identity in receipt['accepted_new_ids']:
                member=('execution_attachments_v2/strict_acceptance.json' if cohort['count']==1
                        else 'runs/'+identity+'/strict_acceptance.json')
                data=archive.extractfile(member).read()
                accepted=json.loads(data)
                b.require(identity not in records and identity in b.CLOSED_IDS, 'Duplicate/foreign already-replayed ID')
                b.require(accepted['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and accepted['id']==identity,
                          'Prior record lacks strict three-view acceptance')
                b.require(accepted['checkpoint_sha256']==actual_records[identity]['checkpoint']['sha256'], 'Prior replay uses another terminal')
                b.require(set(accepted['views'])==set(b.VIEWS) and accepted['native_comparison']['accepted'], 'Prior views/native acceptance failed')
                records[identity]=dict(id=identity,checkpoint_sha256=accepted['checkpoint_sha256'],
                    strict_acceptance_member=member,strict_acceptance_sha256=hashlib.sha256(data).hexdigest(),
                    archive=archive_path,archive_sha256=receipt['archive_sha256'])
        pins.update({receipt_path:cohort['receipt'],verification_path:cohort['verification'],archive_path:receipt['archive_sha256']})
        proofs.append(dict(archive=archive_path,archive_sha256=receipt['archive_sha256'],
                           accepted_ids=receipt['accepted_new_ids'],members_verified=checked['members_verified']))
    b.require(set(records)==set(b.CLOSED_IDS), 'Eight closed replay IDs are not covered')
    return dict(status='PRIOR_EIGHT_OFFSERVER_IDENTITIES_VERIFIED_NO_NEW_INFERENCE',
                records=[records[identity] for identity in b.CLOSED_IDS], archive_proofs=proofs, input_pins=pins)
