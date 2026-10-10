"""Seal a real, already built C60 snapshot without accepting its science for publication."""
from pathlib import Path
from collections import Counter
import datetime,hashlib,json
R=Path(__file__).resolve().parents[1]
B=R/'tmp/celeba_mechanism_C_six_scenes_prepared_20261010';S=B/'snapshot'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def write(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2);f.write('\n')
assert sha(B/'FILES_SHA256.json')=='4a1a2f9d71bd432e58229f98cf7f98b1fdba9e8b60dad4c9338c8801b240067c'
for row in read(B/'FILES_SHA256.json')['members']:
 assert sha(B/row['path'])==row['sha256'] and (B/row['path']).stat().st_size==row['size']
for name,pin in read(S/'FILES_SHA256.json')['files'].items():assert sha(S/name)==pin['sha256']
records=read(S/'records.json')['records'];table=read(S/'tables.json');verified=read(S/'verification.json')
binding=read(S/'SOURCE_BINDINGS.json')['actual_C4_binding'];adoption=R/binding['adoption']
assert sha(adoption)==binding['adoption_sha256']
proof=read(adoption)
assert proof['status']=='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
assert (proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models'])==(156,4,160)
assert proof['all_native_differences_zero'] and proof['original156_unchanged'] and not proof['test_inference']
assert (verified['mean_sd_scalars'],verified['display_mean_sd_cells'],verified['receipt_metrics_from_group_counts'],verified['base_confusion_counts_structurally_checked'],verified['preserved_IID_seed_first']['mean_sd_scalars'])==(972,486,1080,2880,162)
assert len(records)==len({r['id'] for r in records})==120
counts=Counter(r['variant'] for r in records);assert counts=={'Full':60,'minus_C':60}
assert table['complete_scenes']==6 and not table['full_nonIID_coverage'] and not table['final_test']
actual=dict(status='C60_ACTUAL_BUILT_PENDING_INDEPENDENT_ROOT_REVIEW',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_prepared_seal_sha256=sha(B/'FILES_SHA256.json'),actual_C4_adoption_path=adoption.relative_to(R).as_posix(),actual_C4_adoption_sha256=sha(adoption),unique_records=len(records),**counts,complete_scenes=table['complete_scenes'],scene_statistics=verified['mean_sd_scalars'],display_cells=verified['display_mean_sd_cells'],count_metrics=verified['receipt_metrics_from_group_counts'],confusion_count_checks=verified['base_confusion_counts_structurally_checked'],cross_scene_seed_first_statistics=verified['preserved_IID_seed_first']['mean_sd_scalars'],snapshot_files={p.name:sha(p) for p in sorted(S.iterdir()) if p.is_file()},old100_records_unchanged=verified['old100_records_bytes_exact'],old810_statistics_unchanged=verified['old810_statistics_exact'],old405_display_cells_unchanged=verified['old405_cells_exact'],old162_IID_aggregate_bytes_exact=verified['old162_IID_seed_first_bytes_exact'],inference=0,training=0,test=False,canonical_modified=False,negative_results_preserved=True)
write(B/'ACTUAL_HANDOFF.json',actual)
paths=[B/row['path'] for row in read(B/'FILES_SHA256.json')['members']]+[B/'FILES_SHA256.json',B/'ACTUAL_HANDOFF.json']+list(S.iterdir())
assert all(p.is_file() and not p.is_symlink() for p in paths)
files={p.relative_to(B).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
write(B/'ACTUAL_FILES_SHA256.json',dict(files=files))
print(json.dumps(dict(status=actual['status'],handoff_sha256=sha(B/'ACTUAL_HANDOFF.json'),delivery_sha256=sha(B/'ACTUAL_FILES_SHA256.json'),members=len(files))))
