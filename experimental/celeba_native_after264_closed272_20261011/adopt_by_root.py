"""Adopt one original-strict/offserver delta and promote its exact recovery ledger."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, os, shlex, shutil, subprocess, tarfile

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp/celeba_native_after264_closed272_20261011'
CANON=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
import sys
sys.path.insert(0,str(ROOT/'tmp'))
from guardfed_local_storage import check_bulk_storage
check_bulk_storage()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
h=read(BASE/'ROOT_READY_HANDOFF.json')
assert sha(BASE/'ROOT_READY_HANDOFF.json')=='94b564b5455073a1508265321f71ef577bb8d5654a16b534388b67e38f26d9a9'
assert sha(BASE/'DELIVERY_FILES_SHA256.json')=='4ded361af36a37fe7e6b0e88037eae9be023d3f63637c1c8aad123567485db70'
for name,pin in read(BASE/'DELIVERY_FILES_SHA256.json')['files'].items():
    p=BASE/name;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
for pathkey,hashkey in [('snapshot_inspection','inspection_sha256'),('archive_path','archive_sha256'),('receipt_path','receipt_sha256'),('offserver_proof_path','offserver_sha256'),('new_ledger_path','new_ledger_sha256')]:assert sha(h[pathkey])==h[hashkey]
parent=read(BASE/'PARENT_ROOT.json');old=read(BASE/'PARENT_LEDGER.json');new=read(h['new_ledger_path'])
assert sha(BASE/'PARENT_ROOT.json')==h['parent_root_sha256']
assert sha(CANON/'verified_ledger.json')==h['parent_ledger_sha256']
assert new['entries'][:-1]==old['entries'] and len(new['entries'])==39
assert new['entries'][-1]['receipt_sha256']==h['receipt_sha256']
report=read(h['snapshot_inspection']);receipt=read(h['receipt_path']);offserver=read(h['offserver_proof_path'])
assert report['new_count']==272 and report['reused_count']==100 and not report['invalid']
assert report['source_script_sha256']==h['original_strict_tool_sha256']=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
assert offserver['pass'] and offserver['different_host_observed'] and offserver['members_verified']==94
assert receipt['accepted_new_ids']==h['new_ids'] and not receipt['failure_identities'] and receipt['reused_full_weights_repacked']==0
expected={'minus_A_non-IID_F Flip_seed91007', 'minus_A_non-IID_F Flip_seed91006', 'minus_A_non-IID_F Flip_seed91008', 'minus_A_non-IID_F Flip_seed91005', 'minus_A_non-IID_F Flip_seed91009', 'minus_A_non-IID_FedSA_seed91001', 'minus_A_non-IID_FedSA_seed91002', 'minus_A_non-IID_F Flip_seed91010'}
assert set(h['new_ids'])==expected and h['parent_accepted']==264 and h['cumulative_strict_offserver']==272
prior=read(CANON/'root_delta_20261010T125824Z/inspection/inspection.json')
oldrows={r['id']:r for r in prior['records']};rows={r['id']:r for r in report['records']}
assert len(rows)==372 and all(rows[i]==r for i,r in oldrows.items())
assert set(rows)-set(oldrows)==set(h['new_ids']) and len(h['new_ids'])==8
with tarfile.open(h['archive_path'],'r:gz') as archive:
    members=archive.getmembers();names=[m.name for m in members]
    assert len(names)==len(set(names))==94
    assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members)
    inventory_bytes=archive.extractfile('backup_inventory.json').read();inventory=json.loads(inventory_bytes)
    assert hashlib.sha256(inventory_bytes).hexdigest()==receipt['inventory_sha256']
    assert set(names)==set(inventory['members'])|{'backup_inventory.json'}
    for name,pin in inventory['members'].items():
        body=archive.extractfile(name).read()
        assert len(body)==pin['bytes'] and hashlib.sha256(body).hexdigest()==pin['sha256']
    for identity in h['new_ids']:
        out='runs/'+identity+'/';result=json.load(archive.extractfile(out+'result.json'));progress=json.load(archive.extractfile(out+'progress.json'));accepted=json.load(archive.extractfile(out+'mechanism_acceptance.json'));job=json.load(archive.extractfile('jobs/'+identity+'.json'))
        assert result['rounds']==progress['round']==accepted['rounds']==70 and accepted['pass'] and progress['job_id']==job['id']==identity
        assert result['seed']==job['config']['seed']==rows[identity]['seed']
        assert accepted['checkpoint_sha256']==rows[identity]['checkpoint_sha256']==inventory['members'][out+'model.pt']['sha256']
        assert accepted['result_sha256']==inventory['members'][out+'result.json']['sha256']
        assert accepted['job_sha256']==inventory['members']['jobs/'+identity+'.json']['sha256']
        assert result['config']['celeba_evaluation_split']=='valid' and rows[identity]['prediction_support']['prediction_count']==19867
        assert rows[identity]['group_denominator_support']=={'0':{'n':11409,'positives':6157},'1':{'n':8458,'positives':3445}}

# All checks above are read-only. Publish compact copies and an atomic exact-ledger promotion once.
source=Path(h['new_ledger_path']).parent;tag=source.name;destination=CANON/tag
assert not destination.exists();shutil.copytree(source,destination)
remote_code="""from pathlib import Path
import hashlib,json,os
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(source)==expected_new and sha(destination)==expected_old
for proc in Path('/proc').glob('[0-9]*'):
 try:
  argv=(proc/'cmdline').read_bytes().split(bytes([0]))
  assert not any(x.endswith(b'/evidence_v4.py') for x in argv),'Existing native inspector/collector'
 except (FileNotFoundError,ProcessLookupError):pass
temporary=Path(destination).with_name('verified_ledger.promote272.tmp')
assert not temporary.exists()
temporary.write_bytes(Path(source).read_bytes());assert sha(temporary)==expected_new
os.replace(temporary,destination)
assert sha(destination)==expected_new
print(json.dumps(dict(status='EXACT_NATIVE272_LEDGER_PROMOTED',old_sha256=expected_old,new_sha256=expected_new,source=source,destination=destination)))
"""
bindings=dict(source=h['private_remote_ledger'],destination='/workspace/guardfed_checks/server_reactivation_20261009/mechanism_science_backups_20261009/verified_ledger.json',expected_old=h['parent_ledger_sha256'],expected_new=h['new_ledger_sha256'])
code='\n'.join(k+'='+repr(v) for k,v in bindings.items())+'\n'+remote_code
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -c '+shlex.quote(code)],capture_output=True,timeout=40)
(destination/'ROOT_REMOTE_LEDGER_PROMOTION.stdout.json').write_bytes(r.stdout);(destination/'ROOT_REMOTE_LEDGER_PROMOTION.stderr.txt').write_bytes(r.stderr)
r.check_returncode();assert json.loads(r.stdout)['new_sha256']==h['new_ledger_sha256']
assert sha(CANON/'verified_ledger.json')==h['parent_ledger_sha256']
temp=CANON/'verified_ledger.promote272.tmp';assert not temp.exists();temp.write_bytes(Path(h['new_ledger_path']).read_bytes());os.replace(temp,CANON/'verified_ledger.json')
assert sha(CANON/'verified_ledger.json')==h['new_ledger_sha256']
proof=read(source/'DELTA_VERIFICATION.json')
proof.update(status='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_adopted=True,root_checked_archive_members=94,old364_records_exact=True,canonical_local_and_remote_ledger_promoted=True,source_handoff_path=(BASE/'ROOT_READY_HANDOFF.json').relative_to(ROOT).as_posix(),source_handoff_sha256=sha(BASE/'ROOT_READY_HANDOFF.json'),source_delivery_seal_sha256=sha(BASE/'DELIVERY_FILES_SHA256.json'))
out=destination/'ROOT_DELTA_VERIFICATION.json';out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=proof['status'],path=out.relative_to(ROOT).as_posix(),sha256=sha(out),new=8,total=272,ledger_sha256=h['new_ledger_sha256'])))
