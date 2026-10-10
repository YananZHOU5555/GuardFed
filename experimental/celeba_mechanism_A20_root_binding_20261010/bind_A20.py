"""Root-only A20 metadata binding of the SHA-pinned A12 adopter; preparation is not adoption."""
from pathlib import Path
import argparse,difflib,hashlib,json

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ORIGINAL=ROOT/'tmp/adopt_mechanism_A12_root_20261010.py'
ORIGINAL_SHA='c6295afa5cea744f5ef18d1ca96d6c3ca3400d5aee5921019b3d8f6816f3f3af'
PRIOR_ROOT_SHA='1221482d564a2c735b0de0680fe8a42512c9fe5774a9d157bd3bfda5dd9c858b'
PRIOR_INDEX_SHA='9a90f3d74a27d9ca4225b850797faec3c3aac2b6e2f83f87d3afe49c21912496'
EXPECTED=[f'minus_A_IID_F Flip_seed{s}' for s in range(91003,91011)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def render(args):
    assert sha(ORIGINAL)==ORIGINAL_SHA
    assert args.archive_members==74
    for key in ('delivery_seal_sha256','storage_index_sha256','archive_sha256','receipt_sha256','offserver_sha256'):
        value=getattr(args,key);assert len(value)==64 and set(value)<=set('0123456789abcdef'),key
    old=ORIGINAL.read_text('utf8');text=old
    def edit(a,b):
        nonlocal text
        assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b)
    edit('Join twelve original saved-output checks to the accepted native212 restore chain.',
         'Join eight original saved-output checks to the accepted native220 restore chain.')
    edit('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})')
    edit("HERE = ROOT / 'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010'",
         "HERE = ROOT / 'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010'")
    edit("DELIVERY = ROOT / 'tmp/celeba_remaining620_A12_transport_20261010'",
         "DELIVERY = ROOT / 'tmp/celeba_remaining620_A20_transport_20261010'")
    edit("expected = [f'minus_A_IID_Benign_seed{s}' for s in range(91001, 91011)] + [f'minus_A_IID_F Flip_seed{s}' for s in (91001, 91002)]",
         "expected = [f'minus_A_IID_F Flip_seed{s}' for s in range(91003, 91011)]")
    edit("prepared['actual_export'] is False","prepared['science_transport_unchanged'] is True and prepared['CPU'] == 110")
    edit('(108, 288, 36)','(72, 192, 24)')
    edit('970f43d8dfbc0eff46189e71245b6af982d23e72fe980fcd6191316edb02625a',args.archive_sha256)
    edit("assert sha(storage['receipt']) == storage['receipt_sha256']",
         f"assert sha(storage['receipt']) == storage['receipt_sha256'] == {args.receipt_sha256!r}")
    edit('b62949a65a140921c22fe56acbd0906d5288d1b4aee78152342801ac3b1e0092',args.offserver_sha256)
    edit("'original_A12_archive_verifier'","'original_A20_archive_verifier'")
    edit("archive_check['members_verified'] == 110",f"archive_check['members_verified'] == {args.archive_members}")
    edit('celeba_mechanism_remaining620_C100_root_adoption_20261010/ROOT_ADOPTION.json',
         'celeba_mechanism_remaining620_A12_root_adoption_20261010/ROOT_ADOPTION.json')
    edit('2ac1d2f200d9671de5271f24b0cbb3a0772afb88c16ae6ccf71805ed1588ee46',PRIOR_ROOT_SHA)
    edit("prior_proof['cumulative_accepted'] == 200 and prior_proof['C100_table_adopted'] is False",
         "prior_proof['cumulative_accepted'] == 212 and prior_proof['A_table_adopted'] is False")
    edit("assert sha(prior_path) == prior_proof['records_index_sha256']",
         f"assert sha(prior_path) == prior_proof['records_index_sha256'] == {PRIOR_INDEX_SHA!r}")
    edit("len(prior['all_ids']) == 200","len(prior['all_ids']) == 212")
    edit('celeba_remaining620_C19_transport_20261010/RAW_STORAGE_INDEX.json',
         'celeba_remaining620_A12_transport_20261010/RAW_STORAGE_INDEX.json')
    for a,b in [
        ('root_delta_20261010T060527Z','root_delta_20261010T070642Z'),
        ('7179d0d832392cf9f322098d8d28f5c53e3beec90791d7ecd997fc659dca7683','6199af1014fe91bc8f25c80a9175af93e1c441cb736f9fa42cb62ed8873595ad'),
        ('796fa3fc3a7b673c9267696e6b3e122e85fef2a2ec8db0dfbd6fcddaeb18c415','d6dd77d74ed1a29e69e51cad761145a8bb14840fda44d40764647f1328f60bed'),
        ('root_delta_20261010T062622Z','root_delta_20261010T072729Z'),
        ('b608f2188e8ecca16e514344e613337eb315cd478e90946a6bbada54fe5dfbe9','ee4d7a53fe2c594c19da64b9feee48f1e9008efb76a76b01fdd1daf11933667b'),
        ('566b44092d4261afc63ccd512765408b74820721eb327b3179c61ffde585ae48','a396108987b67fab13376c00c678be79f9eb2fe57ace6b628a425b48474ce98d'),
    ]:
        assert text.count(a) in (1,2,3);text=text.replace(a,b)
    edit('13894351b4ea323204ce7bc389ad478cd3fdc746c973d5119de70ebf6b23c0fb',
         'b6efce88e9d3336ed35c17d461762a460701731442818652ed8c94c12433a069')
    edit('len(native_rows) == 312','len(native_rows) == 320')
    edit('member_checks == 24','member_checks == 16')
    edit('len(all_ids) == len(set(all_ids)) == 212','len(all_ids) == len(set(all_ids)) == 220')
    edit("status='ACCEPTED212_REPLAY_ID_INDEX'","status='ACCEPTED220_REPLAY_ID_INDEX'")
    edit("HERE / 'MECHANISM212_INDEX.json'","HERE / 'MECHANISM220_INDEX.json'")
    edit('ROOT_A12_SAVED_ARRAYS_AND_NATIVE212_RESTORE_CHAIN_ADOPTED',
         'ROOT_A20_SAVED_ARRAYS_AND_NATIVE220_RESTORE_CHAIN_ADOPTED')
    edit('new_accepted=12','new_accepted=8')
    edit('prior_accepted=200, cumulative_accepted=212, remaining620_new_accepted=32',
         'prior_accepted=212, cumulative_accepted=220, remaining620_new_accepted=40')
    edit('archive_members=110',f'archive_members={args.archive_members}')
    edit('exact_native_records_checked=12, native_members_rehashed=24',
         'exact_native_records_checked=8, native_members_rehashed=16')
    edit('independent_metrics=108, independent_counts=288, prediction_rules=36',
         'independent_metrics=72, independent_counts=192, prediction_rules=24')
    edit('original200_unchanged=True','original212_unchanged=True')
    edit("complete_A_scenes=[['IID', 'Benign']], partial_A_scenes=[dict(distribution='IID', attack='F Flip', seeds=[91001, 91002])]",
         "complete_A_scenes=[['IID', 'Benign'], ['IID', 'F Flip']], partial_A_scenes=[]")
    edit('accepted=212, A_table_adopted=False','accepted=220, A_table_adopted=False')
    # External delivery pins are checked before the unchanged original archive/saved checks.
    prelude=f'''assert sha(DELIVERY / 'FILES_SHA256.json') == {args.delivery_seal_sha256!r}
for name, pin in read(DELIVERY / 'FILES_SHA256.json')['files'].items():
    path = DELIVERY / name
    assert path.resolve().is_relative_to(DELIVERY.resolve()) and path.is_file()
    assert sha(path) == pin['sha256'] and path.stat().st_size == pin['bytes']
assert sha(DELIVERY / 'RAW_STORAGE_INDEX.json') == {args.storage_index_sha256!r}
assert not HERE.exists(), 'Preserve any previous root-adoption attempt'
native_first = read(NATIVE / 'root_delta_20261010T070642Z/ROOT_DELTA_VERIFICATION.json')
native_last = read(NATIVE / 'root_delta_20261010T072729Z/ROOT_DELTA_VERIFICATION.json')
assert native_first['total_new_strict_and_offserver'] == 218 and native_last['total_new_strict_and_offserver'] == 220
assert native_first['new_ids'] == [f'minus_A_IID_F Flip_seed{{s}}' for s in range(91003, 91009)]
assert native_last['new_ids'] == [f'minus_A_IID_F Flip_seed{{s}}' for s in (91009, 91010)]
assert native_first['previous_ledger_sha256'] == '95f68e9e219c99e7958444b77722e8820c923f59389c9d4111eb2ac2e77dec04'
assert native_last['previous_ledger_sha256'] == native_first['ledger_sha256']
'''
    edit('volume = check_bulk_storage()',prelude+'\nvolume = check_bulk_storage()')
    compile(text,'<A20-original-adopter-metadata-rebind>','exec')
    return text,''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='original_A12_adopter.py',tofile='A20_metadata_rebind.py'))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('delivery-seal-sha256','storage-index-sha256','archive-sha256','receipt-sha256','offserver-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--archive-members',required=True,type=int)
    return p

if __name__=='__main__':
    args=parser().parse_args();source,_=render(args)
    exec(compile(source,'<A20-original-adopter-metadata-rebind>','exec'),{'__file__':str(__file__),'__name__':'__main__'})
