"""Actual prior19 schema/receipt/root-proof links, without model inference or old member replay."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
EXPECTED_CHAIN = '84be50b9c7dc44ceb3bf34e1bedd0c3ce36e5186b9020e1d6935b9cdc0c95dcd'
EXPECTED_PACKAGE = 'aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())


def validate(chain, chain_sha, package_sha):
    assert chain_sha == EXPECTED_CHAIN
    assert package_sha == EXPECTED_PACKAGE
    assert chain['status'] == 'PARTIAL_STRICT_OFFSERVER_ROOT_RECORD_REVIEW_ADOPTED'
    assert chain['accepted_total'] == len(chain['accepted_job_ids']) == len(set(chain['accepted_job_ids'])) == 19
    assert chain['planned'] == 32 and chain['selected_recipe'] is None
    assert chain['test_evaluated'] is False and chain['formal100_started'] is False


def main():
    chain = read(HERE/'PREVIOUS_CHAIN.json')
    snapshot = read(HERE/'AUTHORIZED_SNAPSHOT.json')
    package = snapshot['source_sha256']['PACKAGE_SHA256.json']
    validate(chain, sha(HERE/'PREVIOUS_CHAIN.json'), package)
    assert 'package_sha256' not in chain  # The adopted schema intentionally lacks this key.
    latest=read(HERE/'PREVIOUS_LATEST.json')
    assert latest['chain_sha256']==EXPECTED_CHAIN and latest['accepted']==19 and latest['planned']==32
    assert (BASE/'LATEST_BACKUP.json').read_bytes() == (HERE/'PREVIOUS_LATEST.json').read_bytes()
    assert sha(BASE/latest['chain_file']) == EXPECTED_CHAIN
    previous=BASE/chain['delta_dir']
    root_proof=ROOT/chain['root_adoption_path']
    assert sha(root_proof) == chain['root_adoption_sha256']
    root=read(root_proof);receipt=read(previous/'BACKUP_SHA256.json')
    assert root['status']=='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
    assert root['accepted_total']==receipt['accepted_total']==19
    assert root['archive_sha256']==receipt['archive_sha256']==chain['archive_sha256']==sha(previous/chain['archive'])
    assert root['members_verified']==receipt['archived_member_count']==chain['archive_members']==70
    assert root['strict_receipt_sha256']==chain['server_strict_sha256']==sha(previous/'PARTIAL_ACCEPTANCE.json')
    assert root['offserver_proof_sha256']==chain['offserver_proof_sha256']==sha(previous/'OFFSERVER_ACCEPTANCE.json')
    assert receipt['inventory_sha256']==chain['inventory_sha256']==sha(previous/'MEMBERS.json')
    assert receipt['package_sha256']==package==sha(BASE.parents[0]/'celeba_flgmm_screen_20261009_v2_frozen_release/PACKAGE_SHA256.json')
    refusals=[]
    for label,c,h,p in [('wrong_prior_SHA',chain,'0'*64,package),('wrong_package',chain,EXPECTED_CHAIN,'0'*64)]:
        try: validate(c,h,p)
        except AssertionError: refusals.append(label)
        else: raise AssertionError('Must reject '+label)
    missing=copy.deepcopy(chain);missing.pop('accepted_job_ids')
    try: validate(missing,EXPECTED_CHAIN,package)
    except KeyError: refusals.append('missing_real_prior_field')
    else: raise AssertionError('Missing prior IDs accepted')
    proof=dict(status='ACTUAL_PRIOR19_SCHEMA_ROOT_RECEIPT_ARCHIVE_LINK_PASS',previous_chain_sha256=EXPECTED_CHAIN,
        package_sha256=package,prior19_package_field_absent_and_not_used=True,accepted_previous=19,
        prior_root_proof_sha256=sha(root_proof),prior_backup_receipt_sha256=sha(previous/'BACKUP_SHA256.json'),
        prior_archive_sha256=chain['archive_sha256'],prior_offserver_proof_sha256=chain['offserver_proof_sha256'],
        previous_latest_sha256=sha(HERE/'PREVIOUS_LATEST.json'),authorized_snapshot_sha256=sha(HERE/'AUTHORIZED_SNAPSHOT.json'),
        refusals=refusals,no_old_member_recheck=True,no_CNN=True,canonical_unchanged=True)
    with (HERE/'PRIOR_SCHEMA_CHECK.json').open('x',encoding='utf-8') as f:
        json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(proof))


if __name__=='__main__': main()
