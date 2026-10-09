"""Independent local member verification and original single-result acceptance."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(batch, release, receipt_sha):
    receipt_path = batch/'BACKUP_SHA256.json'
    assert sha(receipt_path) == receipt_sha
    receipt = json.loads(receipt_path.read_text())
    archive = batch/'accepted_first_two.tar.gz'
    assert sha(archive) == receipt['archive_sha256'] and archive.stat().st_size == receipt['archive_size']
    inventory = json.loads((batch/'MEMBERS.json').read_text())
    assert sha(batch/'MEMBERS.json') == receipt['inventory_sha256']
    assert sha(batch/'PARTIAL_ACCEPTANCE.json') == receipt['acceptance_sha256']
    expected = inventory['members']
    out = batch/'restored'
    assert not out.exists(), 'Do not overwrite an offserver restoration'
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers()) == receipt['archived_member_count']
        assert len(set(tar.getnames())) == len(tar.getnames())
        assert set(tar.getnames()) == set(expected)|{'MEMBERS.json'}
        values = []
        for member in tar:
            assert member.isfile() and (out/member.name).resolve().is_relative_to(out.resolve())
            data = tar.extractfile(member).read()
            spec = (dict(sha256=receipt['inventory_sha256'],size=(batch/'MEMBERS.json').stat().st_size)
                    if member.name=='MEMBERS.json' else expected[member.name])
            assert len(data)==spec['size'] and hashlib.sha256(data).hexdigest()==spec['sha256']
            values.append((member.name,data))
    out.mkdir()
    for name, data in values:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        assert sha(target)==(receipt['inventory_sha256'] if name=='MEMBERS.json' else expected[name]['sha256'])
    sys.path.insert(0,str(release))
    from screen_common import local_identity, accepted
    protocol, manifest = local_identity()
    assert sha(release/'PACKAGE_SHA256.json')==receipt['package_sha256']
    results=[]
    for identity in receipt['accepted_new_ids']:
        item=next(row for row in manifest['jobs'] if row['id']==identity)
        result=accepted(item,out/'runs'/identity)
        assert result is not None
        results.append(dict(id=identity,rounds=result['rounds'],seed=result['seed'],
            distribution=result['distribution'],attack=result['attack'],metrics=result['metrics'],
            checkpoint_sha256=sha(out/'runs'/identity/'model.pt')))
    assert len(results)==2
    proof=dict(status='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED',accepted=2,planned=32,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=results,
        archive_sha256=receipt['archive_sha256'],archived_member_count=receipt['archived_member_count'],
        inventory_sha256=receipt['inventory_sha256'],server_backup_receipt_sha256=receipt_sha,
        package_sha256=receipt['package_sha256'],original_checked_result_replayed_locally=True,
        server_source_data_recheck='before/after identical21 pinnedSHA, recorded in PARTIAL_ACCEPTANCE',
        candidate_selection_performed=False,final_test=False,formal100=False)
    target=batch/'OFFSERVER_ACCEPTANCE.json'
    with target.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(proof,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=proof['status'],proof_sha256=sha(target),records=results)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--receipt-sha',required=True)
    args=parser.parse_args()
    verify(args.batch.resolve(),args.release.resolve(),args.receipt_sha)
