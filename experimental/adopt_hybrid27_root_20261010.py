"""Adopt the exact four reviewed Hybrid terminals; all bulk stays on F."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'tmp/celeba_hybrid_delta_after23_20261010'
BASE=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
DEST=BASE/'accepted_delta_after23_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def main():
    assert not DEST.exists()
    volume=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,SizeRemaining,HealthStatus | ConvertTo-Json -Compress'],text=True).lstrip('\ufeff'))
    assert volume['DriveLetter']=='F' and volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus'] in ('Healthy',0) and volume['SizeRemaining']>1024**3
    assert sha(SRC/'DELIVERY_FILES_SHA256.json')=='59705f5d592e5bb03009b42288ac86aee3cf94fd18838e368e21c655cfe448ae'
    names=[]
    for item in read(SRC/'DELIVERY_FILES_SHA256.json')['members']:
        p=SRC/item['path'];assert p.resolve().is_relative_to(SRC.resolve()) and p.is_file()
        assert sha(p)==item['sha256'] and p.stat().st_size==item['size'];names.append(item['path'])
    assert len(names)==68 and sha(SRC/'ROOT_READY_CHAIN_LINK.json')=='c26b1e065540ff1d0d20fe3dd6289fc48a192f8639e5b6c83859273b48523096'
    link=read(SRC/'ROOT_READY_CHAIN_LINK.json');latest=read(BASE/'LATEST_BACKUP.json');prior=read(BASE/latest['chain_file'])
    assert sha(BASE/'LATEST_BACKUP.json')==link['previous_latest_sha256']
    assert sha(BASE/latest['chain_file'])==latest['chain_sha256']==link['previous_chain_sha256']
    old=prior['accepted_job_ids'];new=link['accepted_new_ids']
    assert latest['accepted']==len(old)==23 and len(new)==4 and not set(old)&set(new)
    assert link['accepted_job_ids']==old+new and len(set(old+new))==27
    raw=Path(link['raw_storage_directory']);assert raw.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve())
    assert sha(SRC/'RAW_STORAGE_INDEX.json')==link['raw_storage_index_sha256']
    for info in read(SRC/'RAW_STORAGE_INDEX.json')['files'].values():
        p=Path(info['path']);assert p.resolve().is_relative_to(raw.resolve()) and sha(p)==info['sha256'] and p.stat().st_size==info['bytes']
    archive=Path(link['archive_path']);assert sha(archive)==link['archive_sha256']=='011157b30ca5df8220646b7b356464657b8b1b5754ca0aba02c32ab870a89ac8'
    assert sha(SRC/'MEMBERS.json')==link['inventory_sha256'];members=read(SRC/'MEMBERS.json')['members']
    with tarfile.open(archive) as tar:
        files=[m for m in tar.getmembers() if m.isfile()]
        assert len(files)==link['archive_members']==59 and {m.name for m in files}==set(members)|{'MEMBERS.json'}
        for m in files:
            data=tar.extractfile(m).read();expected=members[m.name] if m.name!='MEMBERS.json' else dict(sha256=sha(SRC/'MEMBERS.json'),size=(SRC/'MEMBERS.json').stat().st_size)
            assert len(data)==expected['size'] and hashlib.sha256(data).hexdigest()==expected['sha256']
    strict=read(SRC/'PARTIAL_ACCEPTANCE.json');off=read(SRC/'OFFSERVER_MEMBER_TENSOR_PROOF.json');record=read(SRC/'LOCAL_RECORD_CHECKS.json')
    for name,key in (('PARTIAL_ACCEPTANCE.json','server_strict_sha256'),('OFFSERVER_MEMBER_TENSOR_PROOF.json','offserver_tensor_proof_sha256'),('LOCAL_RECORD_CHECKS.json','offserver_original_record_check_sha256')):assert sha(SRC/name)==link[key]
    assert strict['accepted_new_ids']==off['accepted_new_ids']==new
    assert record['status']=='RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS' and record['server_check_receipt_sha256']==sha(SRC/'PARTIAL_ACCEPTANCE.json')
    assert record['local_CUDA_initialized'] is False and record['local_runtime_not_claimed_equal']
    assert [r['id'] for r in record['records']]==new
    for r in strict['records']:
        folder=raw/'restored/screen_runs'/r['id'];result=read(folder/'result.json')
        assert result['rounds']==70 and result['seed']==91001 and result['config']['celeba_evaluation_split']=='valid'
        assert result['metrics']==r['metrics'] and result['evaluation_stats']==r['evaluation_stats'] and result['evaluation_stats']['prediction_count']==19867
        expected=next(x for x in record['records'] if x['id']==r['id']);assert result['metrics']==expected['metrics'] and sha(folder/'model.pt')==expected['checkpoint_sha256']
    DEST.mkdir()
    for name in names+['DELIVERY_FILES_SHA256.json']:
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SRC/name,target);assert sha(target)==sha(SRC/name)
    proof=dict(status='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),delivery_seal_sha256=sha(SRC/'DELIVERY_FILES_SHA256.json'),ready_link_sha256=sha(SRC/'ROOT_READY_CHAIN_LINK.json'),previous_latest_sha256=sha(BASE/'LATEST_BACKUP.json'),previous_chain_sha256=latest['chain_sha256'],archive_local_path=archive.as_posix(),archive_sha256=sha(archive),members_verified=59,accepted_before=23,accepted_new=4,accepted_total=27,strict_receipt_sha256=sha(SRC/'PARTIAL_ACCEPTANCE.json'),offserver_proof_sha256=sha(SRC/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),record_check_sha256=sha(SRC/'LOCAL_RECORD_CHECKS.json'),original_acceptor_replayed_by_offserver_tool=True,auxiliary_seal_failure_preserved=True,new_inference=0,selection_performed=False,scientific_changes=False,final_test=False,formal100_started=False)
    (DEST/'ROOT_ADOPTION_REVIEW.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
    chain=dict(status='PARTIAL_STRICT_OFFSERVER_ROOT_RECORD_REVIEW_ADOPTED',checked_utc=proof['checked_utc'],accepted_total=27,planned=32,accepted_job_ids=old+new,accepted_new_ids=new,previous_accepted=23,previous_chain_file=latest['chain_file'],previous_chain_sha256=latest['chain_sha256'],delta_dir=DEST.name,archive=archive.as_posix(),archive_sha256=sha(archive),inventory_sha256=sha(SRC/'MEMBERS.json'),archive_members=59,server_strict_sha256=proof['strict_receipt_sha256'],offserver_proof_sha256=proof['offserver_proof_sha256'],root_adoption_path=(DEST/'ROOT_ADOPTION_REVIEW.json').relative_to(ROOT).as_posix(),root_adoption_sha256=sha(DEST/'ROOT_ADOPTION_REVIEW.json'),new_CNN_inference=0,selected_recipe=None,test_evaluated=False,formal100_started=False)
    target=BASE/('BACKUP_CHAIN_'+DEST.name+'.json');assert not target.exists();target.write_text(json.dumps(chain,indent=2)+'\n',encoding='utf8')
    latest=dict(chain_file=target.name,chain_sha256=sha(target),accepted=27,planned=32);(BASE/'LATEST_BACKUP.json').write_text(json.dumps(latest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(root_sha256=proof and sha(DEST/'ROOT_ADOPTION_REVIEW.json'),**latest)))

if __name__=='__main__':main()
