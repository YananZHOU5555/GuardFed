"""Close this publication's actual compact inputs; no science or Git operations."""
from pathlib import Path
import argparse,datetime,hashlib,json
R=Path(__file__).resolve().parents[1];T=R/'docs/server_deployment_20260923/training_20260923'
H=R/'tmp/publication_increment61_source_prepared_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
save=lambda p,j:p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
p=argparse.ArgumentParser();p.add_argument('--publisher-sha256',required=True);p.add_argument('--independent-review-sha256',required=True);a=p.parse_args()
ind=R/'tmp/publication_increment61_entry_independent_review_20261011/REVIEW.json'
assert sha(ind)==a.independent_review_sha256
ir=read(ind);ep=R/'tmp/increment61_entry_times_corrected_proof_20261011.json';e=read(ep)
assert sha(ep)=='d0e647b980ba010acf8d17c02b799b5c672ee451709b8216301ecef45cc53247'
sp=T/'TRAINING_STATE.json';assert sha(sp)==e['STATE_sha256']=='ab84750840db71ed531a683f3b6e2c681b2e8ab1aac04fa8619a7e58d61649de'
assert ir['status']=='PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW' and ir['STATE_sha256']==sha(sp)
s=read(sp);assert all(sha(R/rel)==v['sha256'] and v['history_exact'] for rel,v in e['entries'].items())
publisher=H/'publish_guardfed_increment61_20261011.py';assert sha(publisher)==a.publisher_sha256
roles={
 'FLGMM_three_view71_root':('tmp/fl_FFlip10_capacity_pool32_20261011/ROOT_SCIENTIFIC_ADOPTION.json','fe400060961fb923cbac79443f735fab6824b06689d45fb3bbf8d56307c95421'),
 'FLGMM_seven_scene70_table_root':('outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/ROOT_VERIFICATION.json','bcf0b16a6838114444080f95b6bd282b16f79f0052bee0598c7c0728e54fc8d8'),
 'Hybrid_three_view10_root':('tmp/hybrid_missing8_root_adoption_prepared_20261011/ROOT_SCIENTIFIC_ADOPTION.json','02b5f2c8fd26808f62a9e94d1901f4587d09d13dcce29b118113ccc0e1374e06'),
 'current_entries_proof':(ep.relative_to(R).as_posix(),sha(ep)),
 'independent_limited_review':(ind.relative_to(R).as_posix(),sha(ind))}
for rel,digest in roles.values():assert sha(R/rel)==digest
review=dict(status='PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW',root_accepted=True,
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),STATE_sha256=sha(sp),entries=e['entries'],
 latest_observation_only=s['latest_five_queue_readonly_observation'],
 source_pins={k:dict(path=v[0],sha256=v[1]) for k,v in roles.items()},
 current_accepted=dict(native_new=320,three_view_models=320,F_paired_models=20,F_complete_scenes=2,
  by_variant=dict(minus_U=100,minus_C=100,minus_A=100,minus_F=20),FL_native_new=67,
  FL_three_view_total=71,FL_complete_table_records=70,FL_complete_table_scenes=7,gradient_accepted=46,
  Hybrid_native_new=12,Hybrid_three_view_total=10,Hybrid_new_three_view_records=9,Hybrid_prior_seed91002_reused=1),
 root_read_changed_current_prefixes=True,distinct_live_observation_times_preserved=True,
 author_adaptations='Huber identity projection empirical CNN adaptation; LoGoFair fixed image-ID virtual20 population cohorts, not actual training clients',
 boundary='Publish adopted FL71/table70 and Hybrid10; native333 collector remains root-pending and excluded. Existing24-comment author-review drafts unchanged, no submitted-manuscript application or final test.',
 no_scientific_recalculation=True,final_test=False,whole_rebuttal_complete=False)
rp=R/'tmp/increment61_root_semantic_review_20261011.json';assert not rp.exists();save(rp,review)
files={}
def add(rel,expected=None):
 q=R/rel;assert not q.is_symlink() and q.is_file() and q.resolve().is_relative_to(R.resolve())
 pin=dict(sha256=sha(q),bytes=q.stat().st_size)
 if expected is not None:assert pin=={k:expected[k] for k in pin},rel
 assert pin['bytes']<2_000_000
 if rel in files:assert files[rel]==pin
 files[rel]=pin
flp=R/'tmp/publication_increment61_review_support_20261011/PROPOSED_NEW_SELECTION_v2.json'
assert sha(flp)=='299b3c3e2576a174a8664503ef2d6298a19241daf2b89a16d3f909c5fe5837a4'
for rel,pin in read(flp)['files'].items():add(rel,pin)
hsp=R/'tmp/publication_increment61_hybrid_support_20261011/PUBLICATION_SUPPORT.json'
assert sha(hsp)=='9d8f17a1ee1b09dab753e5abe73f73d9417bf47ba72bcc3b2c44ff96a4f794b5'
hs=read(hsp)
for pin in hs['actual_closed_proof_paths']:add(pin['path'],pin)
for pkg in hs['closed_source_packages']:
 add(pkg['seal_path'],dict(sha256=pkg['seal_sha256'],bytes=pkg['seal_bytes']))
 for pin in pkg['members']:add(pin['path'],pin)
for pin in hs['F_report_copies']:add(pin['owned_copy_path'],pin)
for pkgdir in ['tmp/publication_increment61_source_prepared_20261011','tmp/publication_increment61_hybrid_support_20261011']:
 seal=R/pkgdir/'FILES_SHA256.json';add(seal.relative_to(R).as_posix())
 for rel,pin in read(seal)['files'].items():add((Path(pkgdir)/rel).as_posix(),pin)
editorial=read(R/s['latest_rebuttal_draft']['root_proof_path'])
for rel,digest in editorial['source_and_documents'].items():add(rel);assert files[rel]['sha256']==digest
for rel in list(e['entries'])+[sp.relative_to(R).as_posix(),s['latest_rebuttal_draft']['root_proof_path'],
 'tmp/update_increment61_entries_20261011.py','tmp/correct_increment61_observation_times_20261011.py',
 'tmp/increment61_entry_update_proof_20261011.json',Path(__file__).relative_to(R).as_posix()]:add(rel)
for rel,digest in roles.values():add(rel);assert files[rel]['sha256']==digest
ndiffs={rel for rel in files if Path(rel).suffix=='.ndiff'}
assert ndiffs=={x['path'] for x in hs['exact_ndiff_suffix_exceptions_if_full_source_seal_closure_required']}
selection=dict(status='ROOT_CLOSED_INCREMENT61_COMPACT_SELECTION',root_accepted=True,
 parent_commit='798ce6ca670a1ca8cd573c6bfaf4850d46f6d258',semantic_review_sha256=sha(rp),STATE_sha256=sha(sp),
 files=dict(sorted(files.items())),files_count=len(files),total_bytes=sum(v['bytes'] for v in files.values()),
 source_seal_closure='Five Hybrid source packages complete including six exact ndiff paths; raw models/arrays/ZIPs remain F/server only.',
 native333_root_pending_excluded=True,final_test=False)
out=H/'COMPACT_SELECTION.json';assert not out.exists();save(out,selection)
print(json.dumps({'semantic_review_sha256':sha(rp),'compact_selection_sha256':sha(out),'files':len(files),'bytes':selection['total_bytes']}))
