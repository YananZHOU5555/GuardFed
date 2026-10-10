"""Prepare-only C60 adoption helper; execution needs actual independent numeric review.
Derived from adopt_C_five_scene_table_root_20261010.py. No build/inference/Git/STATE.
Expected review contract: INDEPENDENT_C60_SIX_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION;
120 records/60 pairs/6 scenes/972 mean-SD scalars/486 cells/1080 metrics/2880 counts,
162 preserved IID aggregate scalars. External actual review and handoff SHAs are mandatory.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse, datetime, hashlib, json, shutil, sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_C_six_scenes_prepared_20261010'
OLD=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010'
DEST=OLD.parent/'three_view_C_six_scenes_20261010'
SOURCE_REVIEW=ROOT/'tmp/celeba_mechanism_C60_source_review_20261010/ROOT_INDEPENDENT_REVIEW.json'
SOURCE_SEAL='4a1a2f9d71bd432e58229f98cf7f98b1fdba9e8b60dad4c9338c8801b240067c'
SOURCE_REVIEW_SHA='1ccc136c6dd5e132a8c90f73e4b8c981650390866befbabadcf4228158ef74f4'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
FIELDS=('unique_records','paired_models','complete_scenes','mean_SD_scalars_recomputed','display_cells','count_metrics_recomputed','confusion_count_checks','cross_scene_mean_SD_scalars_recomputed')
EXPECTED=(120,60,6,972,486,1080,2880,162)


def review_guard(review,handoff_sha,delivery_sha,C4_sha):
    assert review['status']=='INDEPENDENT_C60_SIX_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
    assert review['actual_handoff_sha256']==handoff_sha and review['actual_delivery_seal_sha256']==delivery_sha
    assert review['actual_C4_root_adoption_sha256']==C4_sha
    assert review['source_prepared_seal_sha256']==SOURCE_SEAL
    assert review['adoption_performed'] is False and review['canonical_modified'] is False
    assert review['new_CNN']==review['new_training']==review['new_Full_inference']==0 and review['test'] is False
    assert review['primary_endpoint']=='PENDING_AUTHOR' and review['n_is_seed_count'] is True
    assert review['other_nonIID_C_scenes_complete'] is False and review['whole_rebuttal_complete'] is False
    assert tuple(review[k] for k in FIELDS)==EXPECTED
    for k in ('old100_record_JSON_bytes_and_order_exact','old810_scalars_exact','old405_display_cells_exact','old162_IID_aggregate_bytes_exact'):
        assert review[k] is True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff-sha256',required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--review-sha256',required=True)
    args=parser.parse_args()
    assert not sys.flags.optimize and not DEST.exists() and not DEST.is_symlink()
    review_path=args.review.resolve()
    assert review_path.is_relative_to(ROOT/'tmp/celeba_mechanism_C60_root_arithmetic_review_20261010')
    assert sha(review_path)==args.review_sha256 and sha(BASE/'ACTUAL_HANDOFF.json')==args.handoff_sha256
    assert sha(BASE/'FILES_SHA256.json')==SOURCE_SEAL and sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA
    sr=read(SOURCE_REVIEW)
    assert sr['status']=='PASS_C60_SOURCE_READY_FOR_ROOT_BUILD_ONLY_AFTER_ACTUAL_EXACT4_ADOPTION' and sr['source_adoptable'] is True
    assert sr['source_seal_sha256']==SOURCE_SEAL and sr['actual_C60_statistics_accepted'] is False
    for row in read(BASE/'FILES_SHA256.json')['members']:
        assert sha(BASE/row['path'])==row['sha256'] and (BASE/row['path']).stat().st_size==row['size']
    review,handoff=read(review_path),read(BASE/'ACTUAL_HANDOFF.json')
    seal=BASE/'ACTUAL_FILES_SHA256.json';files=read(seal)['files']
    assert files['ACTUAL_HANDOFF.json']['sha256']==args.handoff_sha256
    assert handoff['source_prepared_seal_sha256']==SOURCE_SEAL
    assert tuple(handoff[k] for k in ('unique_records','Full','minus_C','complete_scenes','scene_statistics','display_cells','count_metrics','cross_scene_seed_first_statistics'))==(120,60,60,6,972,486,1080,162)
    for name,pin in files.items():
        rel=PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts and rel.suffix not in ('.pt','.pth') and '\\' not in name and ':' not in name
        path=BASE/name
        assert not path.is_symlink() and path.resolve().is_relative_to(BASE.resolve()) and sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes']
    snap=BASE/'snapshot';bindings=read(snap/'SOURCE_BINDINGS.json');binding=bindings['actual_C4_binding']
    C4=(ROOT/binding['adoption']).resolve()
    assert C4.name=='ROOT_ADOPTION_REVIEW.json' and C4.parent.parent==ROOT/'tmp/celeba_mechanism_valid_C_after56_20261010/execution_candidate/backups'
    assert sha(C4)==binding['adoption_sha256']==handoff['actual_C4_adoption_sha256']
    c4=read(C4)
    assert c4['status']=='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (c4['prior_three_view_models'],c4['accepted_new'],c4['cumulative_three_view_models'])==(156,4,160)
    assert c4['accepted_new_ids']==[f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
    assert c4['all_native_differences_zero'] and c4['original156_unchanged'] and c4['source_scope_complete']
    assert c4['science_seal_sha256']==binding['science_sha256'] and c4['execution_seal_sha256']==binding['execution_sha256']
    assert c4['new_training']==c4['new_Full_inference']==0 and c4['test_inference'] is False
    review_guard(review,args.handoff_sha256,sha(seal),sha(C4))
    for name,pin in read(snap/'FILES_SHA256.json')['files'].items():
        assert sha(snap/name)==pin['sha256'] and (snap/name).stat().st_size==pin['bytes']
        assert files['snapshot/'+name]['sha256']==pin['sha256']
    for name,key in [('records.json','records_sha256'),('tables.json','tables_sha256'),('TABLES.md','display_sha256'),('cross_scene_seed_first.json','cross_scene_seed_first_sha256')]:
        assert sha(snap/name)==review[key]==handoff['snapshot_files'][name]
    assert bindings['original_C50_root_sha256']==sha(OLD/'ROOT_VERIFICATION.json')=='811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
    assert handoff['inference']==handoff['training']==bindings['new_inference']==0 and handoff['test'] is False and handoff['negative_results_preserved'] is True
    records=read(snap/'records.json')['records'];old=read(OLD/'snapshot/records.json')['records'];tables=read(snap/'tables.json')
    assert records[:100]==old and len(records)==len({r['id'] for r in records})==120
    scenes=[('IID',a) for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]+[('non-IID','Benign')]
    assert {(r['variant'],r['distribution'],r['attack'],r['seed']) for r in records}=={(v,d,a,s) for v in ('Full','minus_C') for d,a in scenes for s in range(91001,91011)}
    assert (tables['unique_records'],tables['paired_models'],tables['complete_scenes'])==(120,60,6)
    assert tables['nonIID_complete_scenes']==['Benign'] and tables['full_nonIID_coverage'] is False
    assert tables['primary_endpoint_selected'] is False and tables['final_test'] is False and tables['new_inference']==0
    assert {(p['view'],tuple(p['seeds'])) for p in tables['panels']}=={(v,tuple(ss)) for v in ('native','raw','shared_calibration') for ss in (range(91001,91011),range(91002,91011),range(91005,91011))}
    assert (snap/'cross_scene_seed_first.json').read_bytes()==(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()
    # Copy only after actual independent arithmetic/provenance and all SHA gates pass.
    DEST.mkdir()
    for name in list(files)+['ACTUAL_FILES_SHA256.json']:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target);assert sha(target)==sha(BASE/name)
    proof=dict(status='ROOT_C60_SIX_SCENE_THREE_VIEW_TABLES_ADOPTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        independent_review_path=review_path.relative_to(ROOT).as_posix(),independent_review_sha256=sha(review_path),
        independent_source_review_sha256=SOURCE_REVIEW_SHA,prepared_source_seal_sha256=SOURCE_SEAL,actual_delivery_seal_sha256=sha(seal),actual_members_verified=len(files),C4_root_adoption_sha256=sha(C4),
        records_sha256=sha(snap/'records.json'),tables_sha256=sha(snap/'tables.json'),display_sha256=sha(snap/'TABLES.md'),aggregate_sha256=sha(snap/'cross_scene_seed_first.json'),**{k:review[k] for k in FIELDS},
        original_five_scene100_records_exact=True,original_five_scene810_statistics_exact=True,original_five_scene405_cells_preserved=True,original_IID_aggregate162_scalars_bytes_exact=True,
        replay_devices={v:dict(Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in ('Full','minus_C')},training_torch={v:dict(Counter(r['training_torch'] for r in records if r['variant']==v)) for v in ('Full','minus_C')},seed_panels=[10,9,6],
        all_negative_results_retained=True,new_CNN=0,new_training=0,new_Full_inference=0,test=False,primary_endpoint='PENDING_AUTHOR',IID_complete_scenes=5,nonIID_complete_scenes=['Benign'],full_nonIID_coverage=False,aggregate_scope='Original five IID scenes only; no imbalanced six-scene mean',other_C_scenes_complete=False,whole_rebuttal_complete=False,
        canonical_table=(DEST/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
    with (DEST/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(dict(root_sha256=sha(DEST/'ROOT_VERIFICATION.json'),table=proof['canonical_table'])))

if __name__=='__main__':main()
