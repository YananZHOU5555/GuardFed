"""Offserver member hashes plus original saved-array count rules; no image inference."""
from pathlib import Path,PurePosixPath
import hashlib,json,sys,tarfile
from verify_saved_increment import verify

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main(dest):
    receipt=read(dest/'backup_receipt.json');archive=dest/'incremental_valid_three_views.tar.gz'
    assert sha(archive)==receipt['archive_sha256']
    extract=dest/'verified_extract';extract.mkdir(exist_ok=False)
    with tarfile.open(archive) as t:
        b=t.extractfile('backup_inventory.json').read();assert hashlib.sha256(b).hexdigest()==receipt['inventory_sha256'];inv=json.loads(b)
        assert len(t.getnames())==len(set(t.getnames()))==receipt['members']
        assert set(t.getnames())==set(inv['members'])|{'backup_inventory.json'}
        assert inv['accepted_new_ids']==receipt['accepted_new_ids']
        assert inv['models_repacked']==inv['new_training']==inv['new_test_inference']==0
        for m in t.getmembers():
            assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
            data=t.extractfile(m).read()
            if m.name in inv['members']:
                row=inv['members'][m.name];assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            f=extract/m.name;f.parent.mkdir(parents=True,exist_ok=True)
            with f.open('xb') as out:out.write(data)
    expected_previous=receipt['previous_backup_receipt_sha256']
    if expected_previous:
        previous=inv['previous_backup'];assert previous['receipt_sha256']==expected_previous
        matches=[p for p in (HERE/'backups').glob('*/backup_receipt.json') if sha(p)==expected_previous]
        assert len(matches)==1 and (matches[0].parent/'OFFSERVER_VERIFICATION.json').is_file()
        assert not set(receipt['accepted_new_ids'])&set(read(matches[0])['all_accepted_ids'])
        assert set(receipt['all_accepted_ids'])==set(receipt['accepted_new_ids'])|set(read(matches[0])['all_accepted_ids'])
    else:assert set(receipt['all_accepted_ids'])==set(receipt['accepted_new_ids'])
    proof=verify(extract/'runs',HERE.parent/'inventory_actual101_Full100refs.json',
        HERE.parents[2]/'tmp/celeba_final_valid_replay_20261009/verification_inputs/original_valid_cache.npz')
    proof.update(archive_sha256=receipt['archive_sha256'],inventory_sha256=receipt['inventory_sha256'],members_verified=len(inv['members']),
        accepted_new_ids=receipt['accepted_new_ids'],all_accepted_ids=receipt['all_accepted_ids'],previous_backup_receipt_sha256=expected_previous,
        source_seal_sha256=inv['execution_seal_sha256'],verification_script_sha256=sha(__file__))
    with (dest/'OFFSERVER_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':proof['status'],'accepted_new':len(receipt['accepted_new_ids']),'accepted_total':len(receipt['all_accepted_ids']),
        'max_abs_native_difference':max(r['native_max_abs_difference'] for r in proof['records']),'members_verified':proof['members_verified'],
        'proof_sha256':sha(dest/'OFFSERVER_VERIFICATION.json')}))
if __name__=='__main__':main(Path(sys.argv[1]))
