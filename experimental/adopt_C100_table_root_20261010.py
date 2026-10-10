"""Adopt the actual independently reviewed C100 snapshot without new science."""
import argparse, datetime, hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tmp'))
from adopt_C70_table_root_20261010 import files_in_seal

SOURCE = ROOT/'tmp/celeba_mechanism_C100_table_20261010'
REVIEW = ROOT/'tmp/celeba_mechanism_C100_independent_review_20261010'
OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_eight_scenes_20261010'
DEST = OLD.parent/'three_view_C_full100_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

def main(review_seal):
    assert not sys.flags.optimize and not DEST.exists()
    source_names = files_in_seal(SOURCE, 'ACTUAL_FILES_SHA256.json', '7dc195a38f0062f910bb01e32708fb8446d963e01612d3262b69fba9cfd9f6cf')
    snap_names = files_in_seal(SOURCE/'snapshot', 'FILES_SHA256.json', '00a6a877e861391d587f360ac4c7252a51f22d00520784f52631266507a9da8b')
    review_names = files_in_seal(REVIEW, 'FILES_SHA256.json', review_seal)
    assert len(source_names) == 28 and len(snap_names) == 10
    proof = read(REVIEW/'ROOT_ARITHMETIC_REVIEW.json')
    assert sha(REVIEW/'ROOT_ARITHMETIC_REVIEW.json') == 'b8f835ecfbd42f9a6de4e3f9b954ac40baeb0dd13df0d29b29e81420e033836c'
    assert proof['status'] == 'INDEPENDENT_C100_TEN_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
    assert proof['actual_delivery_seal_sha256'] == sha(SOURCE/'ACTUAL_FILES_SHA256.json')
    assert proof['snapshot_seal_sha256'] == sha(SOURCE/'snapshot/FILES_SHA256.json')
    assert proof['candidate_adoptable'] and not proof['findings']
    fields = ('unique_records', 'paired_models', 'complete_scenes', 'mean_SD_scalars_recomputed', 'display_cells', 'count_metrics_recomputed', 'confusion_count_checks', 'paired_seed_metric_checks', 'additional_seed_first_scalars_recomputed')
    assert tuple(proof[k] for k in fields) == (200, 100, 10, 1620, 810, 1800, 4800, 900, 324)
    preserved = ('old160_record_JSON_bytes_and_order_exact', 'old1296_scalars_exact', 'old648_display_cells_exact', 'old162_IID_aggregate_bytes_exact')
    assert all(proof[k] is True for k in preserved)
    assert proof['max_abs_difference'] <= 1e-12 and proof['count_metric_max_abs_difference'] == 0
    assert proof['new_CNN'] == proof['new_training'] == proof['new_Full_inference'] == 0 and proof['test'] is False
    for key in ('canonical_modified', 'adoption_performed', 'STATE_modified', 'Git_used'):
        assert proof[key] is False
    joins = proof['source_connections']
    root_accept = ROOT/'tmp/celeba_mechanism_remaining620_C100_root_adoption_20261010/ROOT_ADOPTION.json'
    idx = root_accept.parent/'MECHANISM200_INDEX.json'
    assert sha(root_accept) == joins['actual_root200_sha256'] == '2ac1d2f200d9671de5271f24b0cbb3a0772afb88c16ae6ccf71805ed1588ee46'
    assert sha(idx) == joins['index200_sha256'] == '0940513f702c42ce9451d42ba6d66cbc9ab70868c90ca2137612b8555ca5dc96'
    assert joins['all100_C_native_metric_checkpoint_joins'] and joins['Full100_actual900_reference_source_joins']
    records = read(SOURCE/'snapshot/records.json')['records']
    old_records = read(OLD/'snapshot/records.json')['records']
    assert records[:160] == old_records and len(records) == len({r['id'] for r in records}) == 200
    scenes = [(d, a) for d in ('IID', 'non-IID') for a in ('Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA')]
    assert {(r['variant'], r['distribution'], r['attack'], r['seed']) for r in records} == {(v,d,a,s) for v in ('Full','minus_C') for d,a in scenes for s in range(91001,91011)}
    tables = read(SOURCE/'snapshot/tables.json')
    assert (tables['unique_records'], tables['paired_models'], tables['complete_scenes']) == (200, 100, 10)
    assert tables['full_nonIID_coverage'] and not tables['primary_endpoint_selected'] and not tables['final_test']
    assert {(p['view'], tuple(p['seeds'])) for p in tables['panels']} == {(v, tuple(s)) for v in ('raw','native','shared_calibration') for s in (range(91001,91011),range(91002,91011),range(91005,91011))}
    assert (SOURCE/'snapshot/cross_scene_seed_first.json').read_bytes() == (OLD/'snapshot/cross_scene_seed_first.json').read_bytes()
    for name, scene_n in (('IID',5),('nonIID',5),('balanced',10)):
        assert proof['aggregates'][name]['scalars'] == 162
        assert proof['aggregates'][name]['scenes_per_seed'] == scene_n and proof['aggregates'][name]['n_is_seed_count']
    DEST.mkdir()
    # Copy only sealed source and compact accepted records/reports; no arrays/models.
    for base,prefix,names in ((SOURCE,'source',source_names),(SOURCE/'snapshot','snapshot',snap_names),(REVIEW,'independent_review',review_names)):
        for name in names:
            if prefix == 'source' and name.startswith('snapshot/'):
                continue
            target=DEST/prefix/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(base/name,target); assert sha(target)==sha(base/name)
    adopted = dict(status='ROOT_C100_TEN_SCENE_THREE_VIEW_TABLES_ADOPTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **{k:proof[k] for k in fields}, **{k:proof[k] for k in preserved},
        independent_review_sha256=sha(REVIEW/'ROOT_ARITHMETIC_REVIEW.json'),independent_review_seal_sha256=review_seal,
        actual_delivery_seal_sha256=sha(SOURCE/'ACTUAL_FILES_SHA256.json'),snapshot_seal_sha256=sha(SOURCE/'snapshot/FILES_SHA256.json'),
        source_acceptance_path=root_accept.relative_to(ROOT).as_posix(),source_acceptance_sha256=sha(root_accept),accepted_index_sha256=sha(idx),
        replay_devices=proof['replay_devices'],training_torch=proof['training_torch'],
        seed_panels=[10,9,6],IID_complete_scenes=5,nonIID_complete_scenes=5,full_nonIID_coverage=True,
        aggregate_scope='IID5/nonIID5/balanced10: scene average within seed, then mean and sampleSD across seeds',
        all_negative_results_retained=True,AEOD_definition=proof['AEOD_definition'],primary_endpoint='PENDING_AUTHOR',
        auxiliary_schema_failure_preserved=proof['auxiliary_schema_failure_preserved'],
        reviewer_authored_C100_table_builder=False,reviewer_authored_C100_numeric_verifier=False,
        reviewer_authored_reused_evaluator_parent=True,reviewer_authored_transport_packaging=True,
        root_reviewed_transport_separately=True,new_CNN=0,new_training=0,new_fits=0,test=False,
        other_six_controls_complete=False,whole_rebuttal_complete=False,incorporated_into_full_rebuttal=False,
        canonical_table=(DEST/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
    (DEST/'ROOT_VERIFICATION.json').write_text(json.dumps(adopted,indent=2)+'\n',encoding='utf8')
    seal={p.relative_to(DEST).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'ACTUAL_FILES_SHA256.json').write_text(json.dumps(dict(files=seal),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(root_sha256=sha(DEST/'ROOT_VERIFICATION.json'),actual_seal_sha256=sha(DEST/'ACTUAL_FILES_SHA256.json'),members=len(seal),table=adopted['canonical_table'])))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--review-seal',required=True);main(p.parse_args().review_seal)
