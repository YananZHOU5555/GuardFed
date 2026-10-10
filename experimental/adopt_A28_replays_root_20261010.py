"""Adopt independently joined exact8 saved-array records, retaining source bytes."""
from pathlib import Path
import datetime,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'tmp/celeba_mechanism_A28_independent_join_20261010'
DEST=ROOT/'tmp/celeba_mechanism_remaining620_A28_root_adoption_20261010'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not DEST.exists()
    assert sha(SRC/'FILES_SHA256.json')=='1c863d8192cbd42a8ff37124db5c8ad7adfb7c79173d2848a413bb0db3f04599'
    for name,pin in read(SRC/'FILES_SHA256.json')['files'].items():
        p=SRC/name;assert p.resolve().is_relative_to(SRC.resolve()) and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
    rp=SRC/'REVIEW.json';r=read(rp)
    assert sha(rp)=='19bc6712c517ed746de0a7832faf68d52dd0150606f1ae0ce5ffa0c7e995cb99'
    assert r['status']=='INDEPENDENT_A28_IDENTITY_RESTORE_CHAIN_PASS_ROOT_ADOPTABLE_NO_ADOPTION' and not r['blocking_findings']
    assert (r['prior_accepted'],r['checked_new_ids'],r['proposed_cumulative'])==(220,8,228)
    assert r['native_model_result_members_rehashed']==16 and r['new_source_config_data_checkpoint_receipt_identities_exact']==8
    assert r['prior220_index_prefix_exact'] and r['prior33_ledger_prefix_exact'] and r['Full100_native_tail_exact']
    p=ROOT/r['proposed_index_path'];index=read(p)
    assert sha(p)==r['proposed_index_sha256']=='765ea715defea1e54aebbee0115f38c926ee022f47f520d151d1417c6c8592a9'
    prior=ROOT/index['prior_index_path'];assert sha(prior)==index['prior_index_sha256']=='af414a0ac6705230c53d324cd1f51e7a76b893dabd7416bb6914a6690b6fdddc'
    expected=[f'minus_A_IID_FedSA_seed{s}' for s in range(91001,91009)]
    assert index['all_ids']==read(prior)['all_ids']+expected and len(set(index['all_ids']))==228 and r['selected_ids']==expected
    handoff=read(ROOT/'tmp/celeba_remaining620_A28_transport_20261010/HANDOFF.json')
    assert (handoff['metrics'],handoff['counts'],handoff['rules'],handoff['native_max_abs_difference'])==(72,192,24,0)
    assert handoff['archive_sha256']==r['archive_sha256'] and handoff['offserver_sha256']==r['offserver_sha256']
    assert sha(Path(handoff['receipt_path']))==r['receipt_sha256'] and sha(Path(handoff['offserver_path']))==r['offserver_sha256']
    native_root=Path(handoff['native228_root']);assert sha(native_root)==handoff['native228_root_sha256']=='8df618fda9dde965a43d5a325fc4010a322189585dc45302d3166ab87ea1ed60'
    DEST.mkdir();shutil.copyfile(p,DEST/'MECHANISM228_INDEX.json');assert sha(DEST/'MECHANISM228_INDEX.json')==sha(p)
    proof=dict(status='ROOT_A28_SAVED_ARRAYS_AND_NATIVE228_RESTORE_CHAIN_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        accepted_new_ids=expected,new_accepted=8,prior_accepted=220,cumulative_accepted=228,remaining620_new_accepted=48,
        records_index_path=(DEST/'MECHANISM228_INDEX.json').relative_to(ROOT).as_posix(),records_index_sha256=sha(p),
        independent_review_path=rp.relative_to(ROOT).as_posix(),independent_review_sha256=sha(rp),independent_review_seal_sha256=sha(SRC/'FILES_SHA256.json'),
        archive_path=handoff['archive_path'],archive_sha256=r['archive_sha256'],archive_members=74,
        offserver_proof_path=handoff['offserver_path'],offserver_proof_sha256=r['offserver_sha256'],
        native_root_path=native_root.relative_to(ROOT).as_posix(),native_root_sha256=sha(native_root),native_archives=r['native_archives'],
        native_inspection_sha256=r['native_inspection_sha256'],native_ledger_sha256=r['native_ledger_sha256'],native_members_rehashed=16,exact_native_records_checked=8,
        independent_metrics=72,independent_counts=192,prediction_rules=24,native_max_abs_difference=0,original220_unchanged=True,
        source_seal_sha256=handoff['source_seal_sha256'],complete_A_scenes=[['IID','Benign'],['IID','F Flip']],
        partial_A_scenes=[dict(distribution='IID',attack='FedSA',seeds=list(range(91001,91009)),n=8,target_n=10)],
        new_scene_table_created=False,Full_inference=0,new_CNN=0,new_fit=0,new_training=0,test=False,whole_rebuttal_complete=False)
    with (DEST/'ROOT_ADOPTION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(dict(accepted=228,A=28,partial_FedSA=8,root_sha256=sha(DEST/'ROOT_ADOPTION.json'))))
if __name__=='__main__':main()
