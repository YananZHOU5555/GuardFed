"""Freeze the root-provided measured version and enumerate only three new evidence archives."""
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
T=Path('docs/server_deployment_20260923/training_20260923');C=T/'server_reactivation_20261009';N=C/'mechanism_science_backups_20261009';tag='root_delta_20261010T023113Z';D=N/tag
F=Path('tmp/celeba_flgmm_fullcoverage_delta_after16_20261010');H=Path('tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after22_20261010')
PARENT='59ec6455c1402ff3bfdac454cbf8de9daf0d216d'
def pin(p):return dict(path=p.as_posix(),sha256=sha(R/p))
rows={};snapshots={};sealed=[];omitted=[]
def add(p):rows[p.as_posix()]=sha(R/p)
def snapshot(p,expected=None):
    target=B/'frozen_inputs'/p
    if target.exists():digest=sha(target);assert expected is None or digest==expected
    else:
        digest=sha(R/p);assert expected is None or digest==expected
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/p,target);assert sha(target)==digest
    rel=target.relative_to(R).as_posix();dest='experimental/'+p.relative_to('tmp').as_posix() if p.parts[0]=='tmp' else p.as_posix();rows[rel]=digest;snapshots[rel]=dest;return dict(path=rel,sha256=digest)
state=snapshot(T/'TRAINING_STATE.json','54be0801ea8dd1763e9f29d21bc763f80cd97cbcad2b1897d49e2374586dde9a')
for p,d in [(T/'RUNNING.md','5efbcd1cf1d635d0681d2f8fa878186605ad98163209b52a938a22ca60fc59db'),(T/'REBUTTAL_COMPLETION_20261009.md','3dc92f7150a29847936494a1e73d5146516f5d50c7b5deda66e351db14f83ead'),(Path('docs/返修实验总览.md'),'b460628423afcc6572ef2a059c0f17ec62bf240c610d0370173f10c0f1ac3a50'),(C/'MONITOR_HANDOFF.md','0815105d63d7ea871ef0f9a0582b6a31b50c2c5f55bc491270fe85e54dcee26b')]:snapshot(p,d)
for p in [T/'celeba_mechanism_v1/EXECUTION.md',Path('tmp/update_reactivation_state_20261009.py'),Path('tmp/update_completion_current_20261009.py'),Path('tmp/update_overview_closure100_root_20261009.py'),Path('tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json'),H.parent/'LATEST_BACKUP.json',H.parent/'BACKUP_CHAIN_accepted_delta_after22_20261010.json',Path('tmp/adopt_Hybrid_after22_root_20261010.py'),Path('tmp/adopt_FL96_after16_delta_root_20261010.py')]:snapshot(p)
live=C/'root_live_20261010T023100Z.json';assert sha(R/live)=='49ddb3588e3ce9dac74791ad81d240948b5dc9d22cbb5796d75fa06c43fa87dd';add(live)
snapshot(C/'latest_formal_live.json',sha(R/live))
for p in (R/D).iterdir():
    if p.is_file():add(p.relative_to(R))
recover={}
def parent(p):recover['experimental/'+p.relative_to('tmp').as_posix() if p.parts[0]=='tmp' else p.as_posix()]=sha(R/p)
for p in [N/'root_delta_20261010T013518Z/verified_ledger.json',N/'root_delta_20261010T013518Z.tar.gz.receipt.json',N/'root_delta_20261010T013518Z/ROOT_DELTA_VERIFICATION.json',Path('tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'),T/'celeba_mechanism_v1/manifest.json',Path('tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/ROOT_ADOPTION_REVIEW.json'),Path('tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/batch/OFFSERVER_ACCEPTANCE.json'),H.parent/'BACKUP_CHAIN_accepted_delta_after21_20261010.json',H.parent/'accepted_delta_after21_20261010/ROOT_ADOPTION_REVIEW.json']:parent(p)
body=H.parent/'accepted_delta_after18_20261010/local_record_bridge_v2/checked_record_body.py';parent(body)
for folder in [F,H]:
    root=read(R/folder/'ROOT_ADOPTION_REVIEW.json');seal=folder/'DELIVERY_FILES_SHA256.json';assert sha(R/seal)==root['delivery_seal_sha256'];obj=read(R/seal);members=obj.get('members') or [dict(path=p,**v) for p,v in obj['files'].items()];sealed.append(dict(**pin(seal),members=len(members)))
    for row in members:
        p=folder/row['path'];assert (R/p).resolve().is_relative_to((R/folder).resolve()) and sha(R/p)==row['sha256']
        if row['path']=='local_record_bridge_v2/checked_record_body.py':
            assert sha(R/p)==sha(R/body);omitted.append(dict(**pin(p),reason='Identical original record body in parent commit',parent_source='experimental/'+body.relative_to('tmp').as_posix(),parent_commit=PARENT));continue
        if row['path']=='collector_transport.tar':
            omitted.append(dict(**pin(p),reason='Upload-only envelope; its seven exact metadata sources are separately retained and inside the accepted result archive. No additional scientific artifact.',recovery='Run the retained transport packer only if a new authorized transport is needed; do not claim regenerated tar bytes equal historical upload.'));continue
        add(p)
    add(seal);add(folder/'ROOT_ADOPTION_REVIEW.json')
add(H/'ROOT_DELIVERY_COPY.json')
for p in [T/'publication_closed_increment40_verified_20261010.json',T/'publication_closed_increment40_20261010.json']:add(p)
nroot=D/'ROOT_DELTA_VERIFICATION.json';n=read(R/nroot);assert sha(R/nroot)=='1f1c7d12bf10aa457ccfdaa9cae48ab3a79843b4960d1621b5c05762d1b64772'
bindings=[]
for p,digest in [(D/(tag+'.tar.gz'),n['archive_sha256']),(D/(tag+'.tar.gz.receipt.json'),n['receipt_sha256']),(D/'OFFSERVER_VERIFICATION.json',n['offserver_proof_sha256']),(D/'verified_ledger.json',n['ledger_sha256'])]:assert sha(R/p)==digest;bindings.append(pin(p))
for folder,items in [(F, [('batch/accepted_delta.tar.gz','archive_sha256'),('batch/OFFSERVER_ACCEPTANCE.json','offserver_acceptance_sha256'),('batch/BACKUP_SHA256.json','server_receipt_sha256')]),(H,[('hybrid_after22_delta.tar.gz','archive_sha256'),('OFFSERVER_MEMBER_TENSOR_PROOF.json','offserver_proof_sha256'),('PARTIAL_ACCEPTANCE.json','strict_receipt_sha256')])]:
    root=read(R/folder/'ROOT_ADOPTION_REVIEW.json')
    for name,key in items:assert sha(R/folder/name)==root[key];bindings.append(pin(folder/name))
c=dict(status='ROOT_CLOSED_INCREMENT41_INPUTS',parent_commit=PARENT,counts=dict(native=180,three_view=170,FL_new=18,Hybrid=23,baseline_valid=900),roots=dict(native=pin(nroot),FL=pin(F/'ROOT_ADOPTION_REVIEW.json'),Hybrid=pin(H/'ROOT_ADOPTION_REVIEW.json'),state=state,live=pin(live),previous_publication=pin(T/'publication_closed_increment40_verified_20261010.json')),artifact_bindings=bindings,native_chain=dict(current=pin(D/'verified_ledger.json'),previous=pin(N/'root_delta_20261010T013518Z/verified_ledger.json'),receipt=pin(D/(tag+'.tar.gz.receipt.json')),offserver=pin(D/'OFFSERVER_VERIFICATION.json')),sealed_deliveries=sealed,parent_recovery_blobs=recover,frozen_snapshot_destinations=snapshots,files=[dict(path=p,sha256=d) for p,d in sorted(rows.items())],omitted_members=omitted,new_archives={('experimental/'+p[len('tmp/'):] if p.startswith('tmp/') else p):d for p,d in rows.items() if p.endswith(('.tar','.tar.gz'))},measurement_utc='2026-10-10T02:30:59Z',C_table_pairs=70,complete_English_reply_C_scenes=6,scope='Root-accepted native180/FL18/Hybrid23 only. Views170. No C80 source or future replay/endpoint/test.')
assert len(c['new_archives'])==3
review=D/'ROOT_INDEPENDENT_REVIEW.json';assert sha(R/review)=='fc650143107f70e2017bb58cc864dd4f2cda2c7ac8e3894558cca81c86fae703'
c['native_independent']=pin(review)
if review.as_posix() not in rows:c['files'].append(pin(review))
c['files'].append(pin(Path('tmp/adopt_native180_review_root_20261010.py')))
with (B/'ACTUAL_CLOSED_INPUTS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(c,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(files=len(rows),new_archives=len(c['new_archives']),frozen_files=len(snapshots),closed_sha256=sha(B/'ACTUAL_CLOSED_INPUTS.json'))))
