"""Same member-SHA and unchanged original terminal checker as first-two backup."""
import datetime
import hashlib
import json
import platform
from pathlib import Path
import sys
import tarfile

BATCH=Path(__file__).resolve().parent
RELEASE=BATCH.parents[1]/'celeba_flgmm_screen_20261009_v2_frozen_release'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def execute():
    receipt=read(BATCH/'BACKUP_SHA256.json')
    assert receipt['previous_chain_sha256']=='84be50b9c7dc44ceb3bf34e1bedd0c3ce36e5186b9020e1d6935b9cdc0c95dcd'
    assert receipt['accepted_total']==19+receipt['accepted_new']
    assert not set(receipt['accepted_new_ids'])&set(read(BATCH/'PREVIOUS_CHAIN.json')['accepted_job_ids'])
    archive=BATCH/'accepted_delta_after19.tar.gz'
    assert sha(archive)==receipt['archive_sha256'] and archive.stat().st_size==receipt['archive_size']
    assert sha(BATCH/'MEMBERS.json')==receipt['inventory_sha256']
    assert sha(BATCH/'PARTIAL_ACCEPTANCE.json')==receipt['acceptance_sha256']
    expected=read(BATCH/'MEMBERS.json')['members']; out=BATCH/'restored'
    assert not out.exists(), 'Do not overwrite prior restoration'
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==receipt['archived_member_count']
        assert len(set(tar.getnames()))==len(tar.getnames())
        assert set(tar.getnames())==set(expected)|{'MEMBERS.json'}
        values=[]
        for member in tar:
            assert member.isfile() and (out/member.name).resolve().is_relative_to(out.resolve())
            data=tar.extractfile(member).read()
            spec=dict(sha256=receipt['inventory_sha256'],size=(BATCH/'MEMBERS.json').stat().st_size) if member.name=='MEMBERS.json' else expected[member.name]
            assert len(data)==spec['size'] and hashlib.sha256(data).hexdigest()==spec['sha256']
            values.append((member.name,data))
    out.mkdir()
    for name,data in values:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        assert sha(target)==(receipt['inventory_sha256'] if name=='MEMBERS.json' else expected[name]['sha256'])
    sys.path.insert(0,str(RELEASE))
    from screen_common import local_identity,accepted
    protocol,manifest=local_identity()
    assert sha(RELEASE/'PACKAGE_SHA256.json')==receipt['package_sha256']
    results=[]
    for identity in receipt['accepted_new_ids']:
        item=next(row for row in manifest['jobs'] if row['id']==identity)
        result=accepted(item,out/'runs'/identity); assert result is not None
        results.append(dict(id=identity,rounds=result['rounds'],seed=result['seed'],distribution=result['distribution'],
            attack=result['attack'],metrics=result['metrics'],evaluation_stats=result['evaluation_stats'],
            checkpoint_sha256=sha(out/'runs'/identity/'model.pt')))
    assert len(results)==receipt['accepted_new']
    acceptance=read(BATCH/'PARTIAL_ACCEPTANCE.json'); assert acceptance['source_host']!=platform.node()
    proof=dict(status='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED',accepted_new=receipt['accepted_new'],accepted_total=receipt['accepted_total'],planned=32,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=results,
        archive_sha256=receipt['archive_sha256'],archived_member_count=receipt['archived_member_count'],inventory_sha256=receipt['inventory_sha256'],
        server_backup_receipt_sha256=sha(BATCH/'BACKUP_SHA256.json'),package_sha256=receipt['package_sha256'],
        original_checked_result_replayed_locally=True,source_host=acceptance['source_host'],verification_host=platform.node(),
        different_host_observed=True,no_old_models_repackaged=True,previous_chain_sha256=receipt['previous_chain_sha256'],
        candidate_selection_performed=False,final_test=False,formal100=False)
    with (BATCH/'OFFSERVER_ACCEPTANCE.json').open('x',encoding='utf8',newline='\n') as stream:
        json.dump(proof,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=proof['status'],proof_sha256=sha(BATCH/'OFFSERVER_ACCEPTANCE.json'),records=results)))
if __name__=='__main__': execute()
