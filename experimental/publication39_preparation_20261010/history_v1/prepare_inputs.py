from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(n,d):(B/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
T=Path('docs/server_deployment_20260923/training_20260923');C=T/'server_reactivation_20261009';N=C/'mechanism_science_backups_20261009';tag='root_delta_20261010T013518Z'
S=Path('docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010/semantic_review')
F=Path('tmp/celeba_flgmm_fullcoverage_delta_after14_20261010');H=Path('tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after21_20261010');V=Path('tmp/celeba_mechanism_valid_C_after60_20261010')
prior=read(R/'tmp/publication38_preparation_20261010/PREPARED_INPUTS.json')
refs=[Path('tmp/publication38_preparation_20261010/publish_increment38.py'),Path('tmp/publication38_preparation_20261010/verify_increment38.py'),T/'publication_closed_increment38_verified_20261010.json',T/'publication_closed_increment38_20261010.json',N/tag/'ROOT_INDEPENDENT_REVIEW.json',F/'ROOT_ADOPTION_REVIEW.json',H/'ROOT_ADOPTION_REVIEW.json',S/'ROOT_SEMANTIC_REVIEW.json']
ready={}
def add(p):
 p=Path(p);assert (R/p).is_file() and not(R/p).is_symlink();ready[p.as_posix()]=sha(R/p)
def sealed(folder,name):
 d=read(R/folder/name);rows=d.get('members') or [dict(path=k,**(v if isinstance(v,dict) else dict(sha256=v))) for k,v in d['files'].items()]
 for row in rows:
  p=folder/row['path'];assert (R/p).resolve().is_relative_to((R/folder).resolve()) and sha(R/p)==row['sha256'];add(p)
 add(folder/name)
for p in (R/N/tag).iterdir():
 if p.is_file():add(p.relative_to(R))
for p in (R/N/('mechanism_inspection_v4_'+tag)).iterdir():
 if p.is_file():add(p.relative_to(R))
for suffix in ['.tar.gz','.tar.gz.receipt.json','_offserver_verification.json']:add(N/(tag+suffix))
sealed(V,'PACKAGE_SHA256.json')
for p in (R/'tmp/celeba_mechanism_C_after60_source_review_20261010').iterdir():
 if p.is_file():add(p.relative_to(R))
for p in (R/'tmp/celeba_mechanism_C_after60_root_operations_20261010').iterdir():
 if p.is_file():add(p.relative_to(R))
for name in ['APPROVED.json','APPROVED.sha256','deployment_receipt.json','ROOT_APPROVED.json','root_deployment_source.tar.gz','ROOT_STARTUP_OBSERVATION.json','ROOT_STARTUP_OBSERVATION.RAW.json','preflight.json','start_receipt.json']:
 add(V/'execution_candidate'/name)
add(Path('tmp/adopt_native170_review_root_20261010.py'));add(Path('tmp/prepare_native170_and_C60_review_root_20261010.py'))
ledger=read(R/N/tag/'verified_ledger.json');previous_ledger=N/'root_delta_20261010T003657Z/verified_ledger.json'
recovery={previous_ledger.as_posix():sha(R/previous_ledger)}
for e in ledger['entries'][:-1]:
 receipt=N/Path(e['receipt']).name;assert sha(R/receipt)==e['receipt_sha256'];rr=read(R/receipt)
 recovery[receipt.as_posix()]=e['receipt_sha256'];recovery[(N/Path(e['archive']).name).as_posix()]=rr['archive_sha256']
for p in [T/'celeba_mechanism_v1/manifest.json',Path('tmp/celeba_mechanism_evidence_20261009/evidence_v4.py')]:
 dest='experimental/'+p.relative_to('tmp').as_posix() if p.parts[0]=='tmp' else p.as_posix();recovery[dest]=sha(R/p)
extra=[x.replace('accepted_delta_after19_20261010','accepted_delta_after21_20261010') for x in prior['required_extra_paths']]
extra += [(C/'auxiliary_screens_20261010T015053Z.json').as_posix(),(C/'auxiliary_screens_20261010T015053Z.RAW.json').as_posix(),'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/observation_20261010T015106750045Z/SNAPSHOT.json']
assert all((R/p).is_file() for p in extra),[p for p in extra if not(R/p).is_file()]
prep=dict(status='PREPARED_SOURCE_REQUIRES_ACTUAL_AUTHORITATIVE_CLOSED_INPUTS',reference_pins={p.as_posix():sha(R/p) for p in refs},FL_prior_root_sha256=read(R/F/'ROOT_ADOPTION_REVIEW.json')['previous_root_adoption_sha256'],Hybrid_prior_root_sha256=read(R/H/'ROOT_READY_CHAIN_LINK.json')['previous_root_adoption_sha256'],unchanged_C60_reply_root_sha256='b2e3958ed4000fe0365929c94408c411cd47c70e637c5151bb40603ad097098a',semantic_seal_sha256=sha(R/S/'FILES_SHA256.json'),native_previous_ledger=previous_ledger.as_posix(),parent_recovery_blobs=recovery,Hybrid_omitted_body={**prior['Hybrid_omitted_body'],'verified_parent_commit':'3da6cbb1f72e0d750597e8390ab6ac1c2ff7ae44','seal_members_checked_before_omission':57},required_extra_paths=extra,ready_manifest=[dict(path=p,sha256=d) for p,d in sorted(ready.items())],scope='Native170, views160. C_after60 exact10 source/startup only, no accepted replay delta or C70 artifacts.')
save('PREPARED_INPUTS.json',prep)
paths=dict(native_root=N/tag/'ROOT_INDEPENDENT_REVIEW.json',semantic_root=S/'ROOT_SEMANTIC_REVIEW.json',FL_root=F/'ROOT_ADOPTION_REVIEW.json',Hybrid_root=H/'ROOT_ADOPTION_REVIEW.json',state=T/'TRAINING_STATE.json',formal_live=C/'root_live_20261010T015054Z.json',previous_publication=T/'publication_closed_increment38_verified_20261010.json')
c=dict(status='ROOT_CLOSED_INCREMENT39_INPUTS',parent_commit='3da6cbb1f72e0d750597e8390ab6ac1c2ff7ae44',counts=dict(native=170,three_view=160,FL_new=16,Hybrid=22,baseline_valid=900),closure_pins={k:dict(path=p.as_posix(),sha256=sha(R/p)) for k,p in paths.items()},extra_pins={p:sha(R/p) for p in extra})
save('ACTUAL_CLOSED_INPUTS.json',c)
print(json.dumps(dict(ready_files=len(ready),parent_recovery_blobs=len(recovery),closed_sha256=sha(B/'ACTUAL_CLOSED_INPUTS.json'))))
