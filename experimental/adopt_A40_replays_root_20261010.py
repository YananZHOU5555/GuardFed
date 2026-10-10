"""Adopt only four saved-array records against native243; derived from A36 adopter."""
from pathlib import Path
import datetime, hashlib, json, shutil
from guardfed_local_storage import check_bulk_storage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tmp/celeba_mechanism_remaining620_A40_20261010'
DEST = ROOT / 'tmp/celeba_mechanism_remaining620_A40_root_adoption_20261010'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not DEST.exists()
    storage = check_bulk_storage()
    seal = SRC / 'FINAL_FILES_SHA256.json'
    assert sha(seal) == '3c0be0ae4a77592e8ea6b70bc229b0a2efcffab411e183af8a6f125b7fd4bb8a'
    for name, pin in read(seal)['files'].items():
        p = SRC / name
        assert p.resolve().is_relative_to(SRC.resolve())
        assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes']
    ready = read(SRC / 'ROOT_READY.json')
    assert sha(SRC / 'ROOT_READY.json') == 'c90d3a19bc123c5d95c9f5e70c3c4c09d518b4dbf85d0704c5f20fb837833e1e'
    assert not ready['blocking_findings'] and not ready['actual_root_adoption_performed']
    rp = SRC / 'REVIEW.json'; r = read(rp)
    assert sha(rp) == '2f67c560be4884881a21e08dc7c8eaff1363e920e07cb0d27820ab0ba114b7b9'
    assert r['status'] == 'INDEPENDENT_A40_IDENTITY_RESTORE_CHAIN_PASS_ROOT_ADOPTABLE_NO_ADOPTION' and not r['blocking_findings']
    assert (r['prior_accepted'], r['checked_new_ids'], r['proposed_cumulative']) == (236, 4, 240)
    assert r['native_model_result_members_rehashed'] == 8 and r['new_source_config_data_checkpoint_receipt_identities_exact'] == 4
    assert all(r[k] for k in ('prior236_index_prefix_exact', 'prior35_ledger_prefix_exact', 'Full100_native_tail_exact'))
    p = SRC / 'MECHANISM240_INDEX.json'; index = read(p)
    assert sha(p) == r['proposed_index_sha256'] == 'aa0df30db62549f1be808f188fa9a0f9479b9daa812406cd98ff438b2068f8cc'
    prior = ROOT / index['prior_index_path']
    assert sha(prior) == index['prior_index_sha256'] == 'f378bab97b5a2fba436370f4508f122198559d05920dac2ac9075d83ef7f6e59'
    expected = [f'minus_A_IID_S-DFA_seed{s}' for s in range(91007,91011)]
    assert index['all_ids'] == read(prior)['all_ids'] + expected and len(set(index['all_ids'])) == 240 and r['selected_ids'] == expected
    h = read(SRC / 'HANDOFF.json')
    assert (h['metrics'], h['counts'], h['rules'], h['native_max_abs_difference']) == (36,96,12,0)
    assert (h['prior_transported'], h['new_transported'], h['total_transported']) == (56,4,60)
    for field, pinfield in [('archive_path','archive_sha256'), ('receipt_path','receipt_sha256'), ('offserver_path','offserver_sha256')]:
        assert sha(h[field]) == h[pinfield] == r[pinfield]
    assert h['archive_members'] == ready['archive_members'] == 38
    native_root = Path(h['native243_root'])
    assert sha(native_root) == h['native243_root_sha256'] == 'd6042fdd8868ae9d9f9b69238755ab665a65619bab6b47731101af6e1141910b'
    for native in r['native_archives']:
        assert sha(native['path']) == native['sha256'] and sha(ROOT / native['root_verification_path']) == native['root_verification_sha256']
    assert (ready['independent_metric_checks'], ready['independent_confusion_count_checks'], ready['prediction_rule_checks']) == (36,96,12)
    assert ready['CNN'] == ready['training'] == ready['fit'] == ready['Full_inference'] == ready['new_test'] == ready['models_repacked'] == 0
    assert ready['actual_export_runtime'][0]['CPU'] == ready['actual_export_runtime'][1]['CPU'] == [111]
    assert all(len([i for i in index['all_ids'] if i.startswith('minus_A_IID_'+a+'_seed')])==10 for a in ('Benign','F Flip','FedSA','S-DFA'))
    DEST.mkdir(); shutil.copyfile(p, DEST / p.name)
    assert sha(DEST / p.name) == sha(p)
    proof = dict(status='ROOT_A40_SAVED_ARRAYS_AND_NATIVE243_RESTORE_CHAIN_ADOPTED', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        accepted_new_ids=expected, new_accepted=4, prior_accepted=236, cumulative_accepted=240, remaining620_new_accepted=60,
        records_index_path=(DEST/p.name).relative_to(ROOT).as_posix(), records_index_sha256=sha(p),
        independent_review_path=rp.relative_to(ROOT).as_posix(), independent_review_sha256=sha(rp), independent_review_seal_sha256=sha(seal),
        root_ready_path=(SRC/'ROOT_READY.json').relative_to(ROOT).as_posix(), root_ready_sha256=sha(SRC/'ROOT_READY.json'),
        archive_path=h['archive_path'], archive_sha256=r['archive_sha256'], archive_members=38,
        offserver_proof_path=h['offserver_path'], offserver_proof_sha256=r['offserver_sha256'],
        native_root_path=native_root.relative_to(ROOT).as_posix(), native_root_sha256=sha(native_root), native_archives=r['native_archives'],
        native_inspection_path=index['native_inspection_path'], native_inspection_sha256=r['native_inspection_sha256'], native_ledger_sha256=r['native_ledger_sha256'],
        native_members_rehashed=8, exact_native_records_checked=4, independent_metrics=36, independent_counts=96, prediction_rules=12,
        native_max_abs_difference=0, original236_unchanged=True, source_seal_sha256=h['source_seal_sha256'],
        complete_A_scenes=[['IID',a] for a in ('Benign','F Flip','FedSA','S-DFA')], partial_A_scenes=[],
        new_scene_table_created=False, Full_inference=0, new_CNN=0, new_fit=0, new_training=0, test=False, whole_rebuttal_complete=False,
        fresh_F_volume=storage, source_adopter_path='tmp/adopt_A36_replays_root_20261010.py', source_adopter_sha256=sha(ROOT/'tmp/adopt_A36_replays_root_20261010.py'),
        scope='Only four fixed S-DFA records, native243 identity and unchanged236 prefix; original strict and saved-array predicates retained. No Sp-DFA inclusion.')
    with (DEST/'ROOT_ADOPTION.json').open('x', encoding='utf8') as f: json.dump(proof,f,indent=2); f.write('\n')
    print(json.dumps(dict(accepted=240,A=40,complete_IID_scenes=4,root_sha256=sha(DEST/'ROOT_ADOPTION.json'))))

if __name__ == '__main__': main()
