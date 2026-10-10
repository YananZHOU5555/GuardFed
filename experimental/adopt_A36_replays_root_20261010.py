"""Adopt the exact eight saved-array records against the native236 restore chain."""
from pathlib import Path
import datetime, hashlib, json, shutil
from guardfed_local_storage import check_bulk_storage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tmp/celeba_mechanism_remaining620_A36_20261010'
DEST = ROOT / 'tmp/celeba_mechanism_remaining620_A36_root_adoption_20261010'
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not DEST.exists()
    check_bulk_storage()
    seal = SRC / 'FINAL_FILES_SHA256.json'
    assert sha(seal) == '6d8eab7de1c748fe4b8b3f9275f4d1b93bb2612441d2b37e79ec3d7cb2c256c9'
    for name, pin in read(seal)['files'].items():
        p = SRC / name
        assert p.resolve().is_relative_to(SRC.resolve())
        assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes']
    ready = read(SRC / 'ROOT_READY.json')
    assert sha(SRC / 'ROOT_READY.json') == 'a023d509191fd473ada52bd524d2ebe8834d59aceb39515bfacdfc24afec79ff'
    assert not ready['blocking_findings'] and not ready['actual_root_adoption_performed']
    rp = SRC / 'REVIEW.json'; r = read(rp)
    assert sha(rp) == '89f24c8f322e992dc47dd471b285962b4cff980a3e879d593e71e76c40034ff6'
    assert r['status'] == 'INDEPENDENT_A36_IDENTITY_RESTORE_CHAIN_PASS_ROOT_ADOPTABLE_NO_ADOPTION' and not r['blocking_findings']
    assert (r['prior_accepted'], r['checked_new_ids'], r['proposed_cumulative']) == (228, 8, 236)
    assert r['native_model_result_members_rehashed'] == 16 and r['new_source_config_data_checkpoint_receipt_identities_exact'] == 8
    assert all(r[k] for k in ('prior228_index_prefix_exact', 'prior34_ledger_prefix_exact', 'Full100_native_tail_exact'))
    p = SRC / 'MECHANISM236_INDEX.json'; index = read(p)
    assert sha(p) == r['proposed_index_sha256'] == 'f378bab97b5a2fba436370f4508f122198559d05920dac2ac9075d83ef7f6e59'
    prior = ROOT / index['prior_index_path']
    assert sha(prior) == index['prior_index_sha256'] == '765ea715defea1e54aebbee0115f38c926ee022f47f520d151d1417c6c8592a9'
    expected = [f'minus_A_IID_FedSA_seed{s}' for s in (91009,91010)] + [f'minus_A_IID_S-DFA_seed{s}' for s in range(91001,91007)]
    assert index['all_ids'] == read(prior)['all_ids'] + expected and len(set(index['all_ids'])) == 236 and r['selected_ids'] == expected
    h = read(SRC / 'HANDOFF.json')
    assert (h['metrics'], h['counts'], h['rules'], h['native_max_abs_difference']) == (72,192,24,0)
    assert (h['prior_transported'], h['new_transported'], h['total_transported']) == (48,8,56)
    for field, pinfield in [('archive_path','archive_sha256'), ('receipt_path','receipt_sha256'), ('offserver_path','offserver_sha256')]:
        assert sha(h[field]) == h[pinfield] == r[pinfield]
    native_root = Path(h['native236_root'])
    assert sha(native_root) == h['native236_root_sha256'] == '6d9594230da5f11b26a3df74db1b698811e8f6cdbb3368a9b467cdf92b2467cd'
    for native in r['native_archives']:
        assert sha(native['path']) == native['sha256'] and sha(ROOT / native['root_verification_path']) == native['root_verification_sha256']
    assert (ready['independent_metric_checks'], ready['independent_confusion_count_checks'], ready['prediction_rule_checks']) == (72,192,24)
    assert ready['CNN'] == ready['training'] == ready['fit'] == ready['Full_inference'] == ready['new_test'] == ready['models_repacked'] == 0
    assert ready['actual_export_runtime'][0]['CPU'] == ready['actual_export_runtime'][1]['CPU'] == [111]
    DEST.mkdir(); shutil.copyfile(p, DEST / p.name)
    assert sha(DEST / p.name) == sha(p)
    proof = dict(status='ROOT_A36_SAVED_ARRAYS_AND_NATIVE236_RESTORE_CHAIN_ADOPTED', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        accepted_new_ids=expected, new_accepted=8, prior_accepted=228, cumulative_accepted=236, remaining620_new_accepted=56,
        records_index_path=(DEST/p.name).relative_to(ROOT).as_posix(), records_index_sha256=sha(p),
        independent_review_path=rp.relative_to(ROOT).as_posix(), independent_review_sha256=sha(rp), independent_review_seal_sha256=sha(seal),
        root_ready_path=(SRC/'ROOT_READY.json').relative_to(ROOT).as_posix(), root_ready_sha256=sha(SRC/'ROOT_READY.json'),
        archive_path=h['archive_path'], archive_sha256=r['archive_sha256'], archive_members=74,
        offserver_proof_path=h['offserver_path'], offserver_proof_sha256=r['offserver_sha256'],
        native_root_path=native_root.relative_to(ROOT).as_posix(), native_root_sha256=sha(native_root), native_archives=r['native_archives'],
        native_inspection_path=index['native_inspection_path'], native_inspection_sha256=r['native_inspection_sha256'], native_ledger_sha256=r['native_ledger_sha256'],
        native_members_rehashed=16, exact_native_records_checked=8, independent_metrics=72, independent_counts=192, prediction_rules=24,
        native_max_abs_difference=0, original228_unchanged=True, source_seal_sha256=h['source_seal_sha256'],
        complete_A_scenes=[['IID','Benign'],['IID','F Flip'],['IID','FedSA']], partial_A_scenes=[r['partial_A_scene']],
        new_scene_table_created=False, Full_inference=0, new_CNN=0, new_fit=0, new_training=0, test=False, whole_rebuttal_complete=False,
        console_display_limitation=ready['console_display_correction'])
    with (DEST/'ROOT_ADOPTION.json').open('x', encoding='utf8') as f: json.dump(proof,f,indent=2); f.write('\n')
    print(json.dumps(dict(accepted=236,A=36,partial_SDFA=6,root_sha256=sha(DEST/'ROOT_ADOPTION.json'))))

if __name__ == '__main__': main()
