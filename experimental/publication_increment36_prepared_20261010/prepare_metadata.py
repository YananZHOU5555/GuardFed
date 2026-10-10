"""Bind immutable existing sources; leave all current closure/state pins to root."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
T='docs/server_deployment_20260923/training_20260923/';N=T+'server_reactivation_20261009/mechanism_science_backups_20261009/';tag='root_delta_20261010T000754Z';C='tmp/celeba_mechanism_valid_C_after50_20261010/'
ready={
 'tmp/publication_increment35_prepared_20261010/publish_increment35.py':'0772f210851449811934614913d2cb40a18cf852eb882d973df2a744b0da012d',
 'tmp/publication_increment35_prepared_20261010/verify_increment35.py':'8267976b8f2b89f14c9c2440e25bc11c8b626049a56f9183dd13c7207f5230b6',
 N+tag+'/ROOT_INDEPENDENT_REVIEW.json':'674a25a2b399837b58747d6a78a0536b9197b287dc9c1b825c8219758544fdb1',
 N+tag+'/ROOT_DELTA_VERIFICATION.json':'48d4d03a2ebc33f00233ba64320cf3bd71fc8610b9ea5b50ac6c14d3279cc9a5',
 N+tag+'/verified_ledger.json':'4dcaa43e2278714684a1cfd4800c165f71d72ebca48c8a1519b3c879f55fbf6c',
 N+'mechanism_inspection_v4_'+tag+'/inspection.json':'9a56b5533c05727dc726fa013e3fdd9df035ec3be633747fff13477f55d83d44',
 N+tag+'.tar.gz':'ab567e5c2e4cdfcfe85001b36c4f78877e618b0a37a4da5e9971f8feea6caaa5',
 N+tag+'.tar.gz.receipt.json':'b1821b4666ebfad1c8a802c78a6d28ecc228f2f326995b7853f8b8e8524fe7db',
 N+tag+'_offserver_verification.json':'b78621abdba3b9c8048ab9f447d35d8056ca70712018e9d2d4f20521cc8e4ed3',
 C+'PACKAGE_SHA256.json':'7191a84dfd2111c0dd7535735a8d016969dd71ec9d9f94c083567aec78de3ba2',
 C+'FILES_SHA256.json':'6e79e65b34a1ff78628884993aa6089fdc8c5d07951ed97ce016ad9b3ca6aaf0',
 C+'execution_candidate/EXECUTION_SOURCE_SHA256.json':'073bfde67b2286f6e29fe5f6e2c1f7580f74465b4b0a4e42c583ab0332aa6595',
 'tmp/celeba_flgmm_fullcoverage_delta_after9_20261010/ROOT_ADOPTION_REVIEW.json':'6feb41c9f2f06980d29865ca03d59e5d2cffeeb2a6f0d6f0209f065cf80caf80',
 T+'publication_closed_increment35_verified_20261010.json':'ece61f6e8207275a37fed750403e886efdf458a05887aef6acc1c30bcf417a32',
 T+'publication_closed_increment35_20261010.json':'b34705719539bd9841feb24eff2615cc2e9b1d74c65b66fc50dd0950e4dff007',
 'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010/ROOT_ADOPTION_REVIEW.json':'9b9340b8f35ff22ddf5476ef023b186a6ce3c0bbdc829b68a3ffbb538673b6f3'}
required=[T+p for p in ['RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','celeba_mechanism_v1/EXECUTION.md','server_reactivation_20261009/MONITOR_HANDOFF.md','server_reactivation_20261009/latest_formal_live.json']]+['docs/返修实验总览.md','tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json']
author='docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010/'
assert sha(R/author/'ROOT_REVIEW.json')=='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
assert sha(R/author/'FILES_SHA256.json')=='427fb0015ef84793d4ced9c49d576563fbfc8c5b0314459a4a7a568fcbe57389'
ar=json.loads((R/author/'ROOT_REVIEW.json').read_bytes());assert ar['author_review_only'] and not ar['manuscript_applied'] and not ar['final_test'] and ar['primary_endpoint']=='PENDING_AUTHOR'
for name,pin in json.loads((R/author/'FILES_SHA256.json').read_bytes())['files'].items():ready[author+name]=pin['sha256'] if isinstance(pin,dict) else pin
ready[author+'FILES_SHA256.json']='427fb0015ef84793d4ced9c49d576563fbfc8c5b0314459a4a7a568fcbe57389';ready[author+'ROOT_REVIEW.json']='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
required+=sorted(p for p in ready if p.startswith(author))
for p,h in ready.items():assert sha(R/p)==h,p
write('PREPARED_INPUTS.json',dict(status='SOURCE_ONLY_REQUIRES_ACTUAL_C6_CLOSURE_AND_ROOT_INJECTED_INPUTS',parent_commit='fbe6027794ce045bdf79254b995c8d2b1de2fb56',branch='codex/revision-evidence-baselines-20260928',ready_sha256=ready,required_extra_paths=required,unchanged_Hybrid_root_sha256='9b9340b8f35ff22ddf5476ef023b186a6ce3c0bbdc829b68a3ffbb538673b6f3',actual_future_C6_root_adoption_sha256=None,C50_table=None,source_preparation_is_publication=False))
write('ROOT_CLOSED_INPUTS_TEMPLATE.json',dict(status='PREPARED_NOT_CLOSED_ROOT_MUST_FILL_ACTUAL_PATHS_AND_SHA',parent_commit='fbe6027794ce045bdf79254b995c8d2b1de2fb56',counts=None,required_closed_counts=dict(native=156,three_view=156,FL_new=11,Hybrid=19,baseline_valid=900),closure_pins={k:dict(path=None,sha256=None) for k in ['C6_adoption','C6_source_review','FL_adoption','Hybrid_adoption','state','formal_live','previous_publication']},extra_pins={p:None for p in required},C50_table=None,note='Root injects actual C6 adoption/source review/operations/terminal and state/live hashes. Hybrid proof is the already-published19 identity bound by publication35; never add its old archive/seal tree. C50 table is already published35 and must stay null. Optional adopted complete author-review directory/adoption may be added only via actual extra_pins after root closure, not prepared drafts or inferred acceptance. The complete C50 author-review canonical seal/ROOT and members are now required actual extra_pins; candidate text is author-review-only, not applied.'))
print('SOURCE_ONLY_INCREMENT36_EXTERNAL_CLOSURE_TEMPLATE_CREATED')
