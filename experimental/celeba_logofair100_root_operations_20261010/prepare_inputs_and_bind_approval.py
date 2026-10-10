"""Explicit root operation: approve actual fixed mappings and invoke metadata-only bind once."""
import argparse,copy,traceback
from pathlib import Path
from common import *

def validate_support(proof,receipt):
    require(proof['status']=='STAGED100_IDENTITY_AND_FIXED10_POPULATION_SUPPORT_PASS_NEW9_NOT_APPROVED'
        and proof['references_verified']==100 and proof['reference_files_verified']==400
        and proof['ID_members_verified']==9 and proof['new_mappings_PREPARED_NOT_APPROVED']==9
        and proof['original91001_mapping_unchanged'] and proof['root_populations']==10 and proof['valid_population']==1,'Exact100/10 population verification required')
    require(proof['root_approved'] is False and proof['recipe_selected'] is False and proof['new_CNN_calls']==proof['new_fits']==proof['new_score_caches']==0,'Staging is not scientific execution/approval')
    support=proof['population_support'];require([s['seed'] for s in support]==list(range(91001,91011)),'Exact ordered10 populations required')
    for s in support:
        require(s['root_n']==16277 and s['valid_n']==19867 and s['cohorts']==20 and s['all_root_group_label_and_valid_sensitive_support_complete'],'Incomplete original population support')
        m=receipt['mappings'][str(s['seed'])]
        require(s['mapping_sha256']==m['sha256'] and s['mapping_metadata_sha256']==m['metadata_sha256'],'Support/mapping mismatch')
    require(receipt['root_approved'] is False and not receipt['recipe_selected'] and receipt['original91001_mapping_unchanged'],'Original staging must remain unapproved')

def run(a):
    sources();adoption=pinned(ADOPTION,ADOPTION_SHA)
    seal=verify_seal(a.seal,a.seal_sha256);pinned(a.handoff,a.handoff_sha256);proof=pinned(a.verification,a.verification_sha256)
    for p,want in [(a.handoff,a.handoff_sha256),(a.verification,a.verification_sha256)]:
        rel=p.resolve().relative_to(a.seal.resolve().parent).as_posix()
        require(rel in seal['files'] and seal['files'][rel]['sha256']==want,'Handoff/verification absent from actual staging seal')
    receipt=pinned(proof['F_receipt'],proof['F_receipt_sha256']);validate_support(proof,receipt)
    require(proof['source_seal_sha256']==receipt['source_seal_sha256']==SOURCE_SHA
        and proof['original_inventory_sha256']==receipt['cache_identity_sha256']==digest(SOURCE/'CACHE_IDENTITIES100.json'),'Actual inventory/source mismatch')
    rows=read(SOURCE/'CACHE_IDENTITIES100.json')['references'];locations={r['id']:r for r in receipt['references']}
    require(len(locations)==len(receipt['references'])==100 and set(locations)=={r['id'] for r in rows},'Missing/duplicate100 reference')
    require(set(receipt['mappings'])==set(map(str,range(91001,91011))),'Missing population')
    for e in receipt['files']:
        p,_=bulk_path(e['path'],0);require(digest(p)==e['sha256'] and p.stat().st_size==e['bytes'],'Actual staged bytes changed')
    old=read(original.SCREEN/'mapping_metadata.json')
    for seed,m in receipt['mappings'].items():
        bulk_path(m['path'],0);bulk_path(m['metadata'],0)
        require(digest(m['path'])==m['sha256'],'Mapping changed');meta=pinned(m['metadata'],m['metadata_sha256'])
        require(all(meta[k]==old[k] for k in ('semantics','cohorts','domain_hex','rule','author_decision_sha256')),'Fixed population rule changed')
        if seed=='91001':require(m['sha256']==old['mapping_sha256'] and m['metadata_sha256']==digest(original.SCREEN/'mapping_metadata.json'),'Original seed91001 bytes changed')
        else:require(meta['status']=='PREPARED_NOT_APPROVED' and meta['approved'] is False and meta['execution_authorized'] is False,'Nine original metadata must remain prepared')
    stage=FROOT/'stage001';approved=FROOT/'root_approved_inputs001'
    for p in (stage,approved):bulk_path(p,32*1024*1024);require(not p.exists(),'No overwrite/partial bind retry')
    require(a.attempt.resolve().parent==HERE.resolve() and not a.attempt.exists(),'New owned control directory required')
    a.attempt.mkdir();save(a.attempt/'STARTED.json',dict(operation='ROOT_METADATA_BIND_ONLY',staging_seal_sha256=a.seal_sha256,handoff_sha256=a.handoff_sha256,new_fits=0))
    try:
        bulk_path(approved,4*1024*1024);approved.mkdir(parents=True)
        mappings=copy.deepcopy(receipt['mappings'])
        for seed,m in mappings.items():
            if seed=='91001':continue
            meta=read(m['metadata']);meta.update(status='FROZEN',approved=True,execution_authorized=False,
                approval_scope='Root-approved identical image-ID population for fixed-recipe validation100; execution remains separately gated',
                prepared_metadata_sha256=m['metadata_sha256'],root_staging_verification_sha256=a.verification_sha256,
                root_complete32_adoption_sha256=ADOPTION_SHA)
            target=approved/('mapping_metadata_'+seed+'.json');save(target,meta)
            m.update(metadata=target.as_posix(),metadata_sha256=digest(target))
        inputs=approved/'ROOT_INPUTS.json';save(inputs,dict(status='ROOT_ACCEPTED_EXISTING_FEDAVG100_AND_FIXED_COHORT_MAPPINGS',cache_identity_sha256=digest(SOURCE/'CACHE_IDENTITIES100.json'),references=receipt['references'],mappings=mappings,staging_receipt_sha256=proof['F_receipt_sha256'],staging_verification_sha256=a.verification_sha256,original_prepared_metadata_preserved=True,new_CNN=0))
        approval=approved/'BIND_APPROVAL.json';save(approval,dict(status='ROOT_LOGOFAIR_FULLCOVERAGE_BIND_APPROVED',summary_adoption_sha256=ADOPTION_SHA,inputs_sha256=digest(inputs),prepared_source_seal_sha256=SOURCE_SHA,new_fits=96,reused=4,test=False))
        original.bind(adoption['summary_path'],adoption['summary_sha256'],ADOPTION,ADOPTION_SHA,adoption['strict_index_path'],adoption['strict_index_sha256'],inputs,approval,digest(approval),stage)
        save(a.attempt/'BIND_RESULT.json',dict(status='ROOT_LOGOFAIR100_METADATA_BOUND_NOT_STARTED',stage=stage.as_posix(),manifest_sha256=digest(stage/'manifest.json'),source_sha256=digest(stage/'SOURCE_SHA256.json'),inputs=inputs.as_posix(),inputs_sha256=digest(inputs),approval=approval.as_posix(),approval_sha256=digest(approval),source_review_sha256=REVIEW_SHA,summary_adoption_sha256=ADOPTION_SHA,staging_handoff_sha256=a.handoff_sha256,new_fits=0,new_CNN=0))
    except BaseException as e:
        save(a.attempt/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('handoff','seal','verification','attempt'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('handoff','seal','verification'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--execute-bind',action='store_true',required=True)
    run(p.parse_args())
