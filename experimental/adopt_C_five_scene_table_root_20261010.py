"""Adopt the exact independently reviewed C50 validation-table bytes."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, shutil, sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'tmp/celeba_mechanism_three_view_C_five_scenes_prepared_20261010'
OLD = ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_four_scenes_20261010'
DEST = OLD.parent/'three_view_C_five_scenes_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--handoff-sha256',required=True)
parser.add_argument('--review',type=Path,required=True)
parser.add_argument('--review-sha256',required=True)
args = parser.parse_args()
assert not sys.flags.optimize and not DEST.exists()
review_path=args.review.resolve()
assert review_path.is_relative_to(ROOT/'tmp/celeba_mechanism_C50_root_arithmetic_review_20261010')
assert sha(review_path)==args.review_sha256 and sha(BASE/'ACTUAL_HANDOFF.json')==args.handoff_sha256
review,handoff=read(review_path),read(BASE/'ACTUAL_HANDOFF.json')
assert review['status']=='INDEPENDENT_C50_FIVE_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
assert review['actual_handoff_sha256']==args.handoff_sha256
fields=('unique_records','paired_models','complete_scenes','mean_SD_scalars_recomputed','display_cells','count_metrics_recomputed','cross_scene_mean_SD_scalars_recomputed')
assert tuple(review[k] for k in fields)==(100,50,5,810,405,900,162)
for key in ('old80_record_JSON_bytes_and_order_exact','old648_scalars_exact','old324_display_cells_exact'):
    assert review[key] is True
seal=BASE/'ACTUAL_FILES_SHA256.json'
assert sha(seal)==review['actual_delivery_seal_sha256']
files=read(seal)['files']
assert files['ACTUAL_HANDOFF.json']['sha256']==args.handoff_sha256
for name,pin in files.items():
    rel=PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and rel.suffix not in ('.pt','.pth')
    path=BASE/name
    assert not path.is_symlink() and sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes']
bindings=read(BASE/'snapshot/SOURCE_BINDINGS.json')
assert bindings['actual_C3_adoption_sha256']==handoff['actual_C3_adoption_sha256']==review['actual_C3_root_adoption_sha256']
assert sha(Path(bindings['actual_C3_adoption']))==bindings['actual_C3_adoption_sha256']
assert bindings['original_four_scene_root_sha256']==sha(OLD/'ROOT_VERIFICATION.json')=='eb08dbc328339e8293133bbd35f60e627ca70b7436ec3b20871bc149e6047172'
assert handoff['inference']==handoff['training']==bindings['new_inference']==0 and not handoff['test']
assert read(BASE/'snapshot/records.json')['records'][:80]==read(OLD/'snapshot/records.json')['records']
DEST.mkdir()
for name in list(files)+['ACTUAL_FILES_SHA256.json']:
    target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(BASE/name,target)
    assert sha(target)==sha(BASE/name)
tables=read(BASE/'snapshot/tables.json')
proof=dict(status='ROOT_C50_FIVE_SCENE_THREE_VIEW_TABLES_ADOPTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    independent_review_path=review_path.relative_to(ROOT).as_posix(),independent_review_sha256=sha(review_path),
    actual_delivery_seal_sha256=sha(seal),actual_members_verified=len(files),C3_root_adoption_sha256=handoff['actual_C3_adoption_sha256'],
    records_sha256=sha(BASE/'snapshot/records.json'),tables_sha256=sha(BASE/'snapshot/tables.json'),display_sha256=sha(BASE/'snapshot/TABLES.md'),
    aggregate_sha256=sha(BASE/'snapshot/cross_scene_seed_first.json'),**{k:review[k] for k in fields},
    original_four_scene80_records_exact=True,original_four_scene648_statistics_exact=True,original_four_scene324_cells_preserved=True,
    replay_devices=tables['replay_devices'],training_torch=tables['training_torch'],seed_panels=[10,9,6],
    all_negative_results_retained=True,new_CNN=0,new_training=0,new_Full_inference=0,test=False,
    primary_endpoint='PENDING_AUTHOR',other_C_scenes_complete=False,whole_rebuttal_complete=False,
    canonical_table=(DEST/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
with (DEST/'ROOT_VERIFICATION.json').open('x',encoding='utf8') as stream:
    json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(root_sha256=sha(DEST/'ROOT_VERIFICATION.json'),table=proof['canonical_table'])))
