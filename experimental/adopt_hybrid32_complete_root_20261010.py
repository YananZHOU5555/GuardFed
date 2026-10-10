"""Adopt actual final five records and the independently checked frozen32 ranking."""
from pathlib import Path
import datetime,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'tmp/celeba_hybrid32_final_collection_20261010'
BASE=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
DEST=BASE/'accepted_delta_after27_20261010'
REVIEW=ROOT/'tmp/celeba_hybrid32_final_independent_review_20261010'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    assert not DEST.exists()
    for directory,digest in ((SRC,'4370962e33b24e7fe9c5ffb461b1a99adc39621bc46efaaf25deb28c64b26e59'),
                             (REVIEW,'67f1a06b63bf40d5ae7fbdd964a457d6bd0e944cd526ba6cd3fc03232a7bbb2a')):
        assert sha(directory/'FILES_SHA256.json')==digest
        seal=read(directory/'FILES_SHA256.json')
        rows=seal.get('members') or [dict(path=n,sha256=p['sha256'],size=p['bytes']) for n,p in seal['files'].items()]
        for row in rows:
            p=directory/row['path'];assert p.resolve().is_relative_to(directory.resolve())
            assert sha(p)==row['sha256'] and p.stat().st_size==row['size']
    rp=REVIEW/'ROOT_INDEPENDENT_REVIEW.json'
    assert sha(rp)=='40f411717385b14c6d5f8885dba3c0d2c9757a8a67a7d41f2afc709490405789'
    review=read(rp)
    assert review['status']=='PASS_ACTUAL_HYBRID32_FROZEN_SCREEN_ROOT_ADOPTABLE_NO_ADOPTION' and review['blocking_findings']==[]
    assert (review['previous_accepted'],review['new_strict_offserver_verified'],review['total_complete'])==(27,5,32)
    assert review['original_scientific_loop_bytes_exact'] and review['maximum_independent_fsum_difference']<=1e-12
    link=read(SRC/'ROOT_READY_CHAIN_LINK.json'); latest=read(BASE/'LATEST_BACKUP.json');prior=read(BASE/latest['chain_file'])
    assert sha(BASE/'LATEST_BACKUP.json')==link['previous_latest_sha256']
    assert sha(BASE/latest['chain_file'])==latest['chain_sha256']==link['previous_chain_sha256']
    old,new=prior['accepted_job_ids'],link['accepted_new_ids']
    assert latest['accepted']==len(old)==27 and len(new)==5 and not set(old)&set(new)
    assert link['accepted_job_ids']==old+new and set(new)==set(review['exact_new_ids'])
    archive=Path(link['archive_path']);assert archive.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve())
    assert sha(archive)==link['archive_sha256']==review['archive_sha256']
    summary_path=SRC/'SUMMARY32.json';summary=read(summary_path)
    assert sha(summary_path)==review['summary_sha256']=='46b5f8fdca9536166ed868e50d4c7bc2578f8a1100ffea97878cf044c95748ae'
    records=[r for c in summary['all_candidates'] for r in c['records']]
    assert len(records)==len({r['id'] for r in records})==32 and {r['id'] for r in records}==set(old+new)
    assert summary['selected_recipe']==review['selected_recipe']=='CosineFairness_lam20.0_tau0.1_lr0.001'
    candidate=next(c for c in read(BASE/'runtime_protocol.json')['candidates'] if c['id']==summary['selected_recipe'])
    DEST.mkdir()
    for name in ('AUTHORIZED_SNAPSHOT.json','ROOT_READY_CHAIN_LINK.json','RAW_STORAGE_INDEX.json','PARTIAL_ACCEPTANCE.json','OFFSERVER_MEMBER_TENSOR_PROOF.json','LOCAL_RECORD_CHECKS.json','SUMMARY32.json','SUMMARY32.md'):
        shutil.copyfile(SRC/name,DEST/name);assert sha(DEST/name)==sha(SRC/name)
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    delta=dict(status='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS',checked_utc=utc,
        delivery_seal_sha256=sha(SRC/'FILES_SHA256.json'),ready_link_sha256=sha(SRC/'ROOT_READY_CHAIN_LINK.json'),
        independent_review_path=rp.relative_to(ROOT).as_posix(),independent_review_sha256=sha(rp),
        archive_sha256=sha(archive),members_verified=71,accepted_before=27,accepted_new=5,accepted_total=32,
        strict_receipt_sha256=sha(SRC/'PARTIAL_ACCEPTANCE.json'),offserver_proof_sha256=sha(SRC/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),
        record_check_sha256=sha(SRC/'LOCAL_RECORD_CHECKS.json'),selection_performed=False,new_inference=0,scientific_changes=False,final_test=False,formal100_started=False)
    save(DEST/'ROOT_ADOPTION_REVIEW.json',delta)
    root=dict(status='ROOT_HYBRID32_SUMMARY_ADOPTED',checked_utc=utc,accepted_total=32,all32_offserver_verified=True,
        summary_path=summary_path.relative_to(ROOT).as_posix(),summary_sha256=sha(summary_path),
        source_seal_sha256=summary['source_seal_sha256'],selected_recipe=summary['selected_recipe'],selected_candidate=candidate,
        accuracy_champion=summary['accuracy_champion'],three_metric_Pareto=summary['three_metric_Pareto'],
        independent_review_path=rp.relative_to(ROOT).as_posix(),independent_review_sha256=sha(rp),
        delta_root_path=(DEST/'ROOT_ADOPTION_REVIEW.json').relative_to(ROOT).as_posix(),delta_root_sha256=sha(DEST/'ROOT_ADOPTION_REVIEW.json'),
        all_candidate_count=8,seed=91001,conditions_per_candidate=4,sample_SD=False,significance=False,
        negative_and_failed_evidence_preserved=True,final_test=False,formal100_started=False,training_authorized_by_this_record=False)
    save(DEST/'ROOT32_SUMMARY_ADOPTION.json',root)
    chain=dict(status='PARTIAL_STRICT_OFFSERVER_ROOT_RECORD_REVIEW_ADOPTED',checked_utc=utc,accepted_total=32,planned=32,
        accepted_job_ids=old+new,accepted_new_ids=new,previous_accepted=27,previous_chain_file=latest['chain_file'],previous_chain_sha256=latest['chain_sha256'],
        delta_dir=DEST.name,archive=archive.as_posix(),archive_sha256=sha(archive),inventory_sha256=link['inventory_sha256'],archive_members=71,
        root_adoption_path=(DEST/'ROOT_ADOPTION_REVIEW.json').relative_to(ROOT).as_posix(),root_adoption_sha256=sha(DEST/'ROOT_ADOPTION_REVIEW.json'),
        new_CNN_inference=0,selected_recipe=None,test_evaluated=False,formal100_started=False)
    target=BASE/('BACKUP_CHAIN_'+DEST.name+'.json');save(target,chain)
    assert sha(BASE/'LATEST_BACKUP.json')==link['previous_latest_sha256']
    (BASE/'LATEST_BACKUP.json').write_text(json.dumps(dict(chain_file=target.name,chain_sha256=sha(target),accepted=32,planned=32),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(accepted=32,selected_recipe=root['selected_recipe'],root_path=str(DEST/'ROOT32_SUMMARY_ADOPTION.json'),root_sha256=sha(DEST/'ROOT32_SUMMARY_ADOPTION.json'))))

if __name__=='__main__':main()
