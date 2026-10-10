"""Bind exact13 metadata only after actual native57 adoption; stdout only, no dispatch."""
import argparse
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parents[2]
P='FLGMM_Tg20_L2.0_lr0.001'
ROOT44='tmp/celeba_flgmm_fullcoverage_delta_after38_20261010/ROOT_ADOPTION_REVIEW.json'
ROOT54='tmp/fl_native44_exact10_20261011/ROOT_ADOPTION_REVIEW.json'
PRIOR='tmp/celeba_flgmm_closed47_root_execution_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pinned(path,expected):
    p=Path(path);p=p if p.is_absolute() else R/p
    if sha(p)!=expected:raise ValueError('SHA mismatch: '+str(p))
    return p,json.loads(p.read_bytes())
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',required=True,type=Path)
    parser.add_argument('--native-root-sha256',required=True)
    a=parser.parse_args()
    if not __debug__:raise ValueError('Do not use -O')
    p44,r44=pinned(ROOT44,'f61a7fa480a62a9394d67a64533568b43e6491ee01d8e1f02005f35a42b6a510')
    p54,r54=pinned(ROOT54,'ead149673512a8c900d570d1f6d7c57e88c0608eb1dc0fb3a5c477d542e5a6a6')
    p57,r57=pinned(a.native_root,a.native_root_sha256)
    pp,prior=pinned(PRIOR,'22fc113add73814ae40accf563ce5f63bd63d50bcf53b7262bdb2b996b016dfe')
    assert r54['previous_root_adoption_sha256']==sha(p44) and (R/r54['previous_root_adoption_path']).resolve()==p44.resolve()
    assert r57['previous_root_adoption_sha256']==sha(p54) and (R/r57['previous_root_adoption_path']).resolve()==p54.resolve()
    assert r57['status']==r54['status']=='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
    assert (r54['accepted_before'],r54['accepted_new'],r54['accepted_total'])==(44,10,54)
    assert (r57['accepted_before'],r57['accepted_new'],r57['accepted_total'])==(54,3,57)
    assert r54['accepted_job_ids']==r44['accepted_job_ids']+r54['accepted_new_ids']
    assert r57['accepted_job_ids']==r54['accepted_job_ids']+r57['accepted_new_ids']
    assert len(set(r57['accepted_job_ids']))==57 and r57['reused_separately']==4 and not r57['final_test']
    assert prior['root_adoption'] and prior['new_three_view_records_accepted']==47 and prior['FLGMM_total_three_view_records']==48
    assert prior['Linux_whole_original_saved_check_pass'] and prior['Windows_saved_outputs_audit_pass']
    assert prior['Windows_saved_outputs_audit_fit_calls']==0 and not prior['Windows_exact_refit_pass'] and not prior['Windows_whole_saved_check_pass']
    old=prior['records']+prior['prior_interface_explicitly_reused'];old_by_id={x['id']:x for x in old}
    assert len(old)==len(old_by_id)==48 and len(prior['prior_interface_explicitly_reused'])==1
    assert set(r44['accepted_job_ids'])<=set(old_by_id) and len(set(old_by_id)-set(r44['accepted_job_ids']))==4
    expected=[P+f'_IID_Sp-DFA_seed{s}_fullcoverage' for s in range(91007,91011)]+[P+f'_non-IID_Benign_seed{s}_fullcoverage' for s in range(91002,91011)]
    delta=[rid for rid in r57['accepted_job_ids'] if rid not in old_by_id]
    assert delta==r54['accepted_new_ids']+r57['accepted_new_ids']==expected and len(delta)==13
    rows=[];sources=[]
    for p,root in ((p54,r54),(p57,r57)):
        index_path,index=pinned(p.parent/'RAW_STORAGE_INDEX.json',root['raw_storage_index_sha256'])
        assert index['internal_drive_fallback'] is False and index['raw_storage_root'].replace('\\','/').startswith('F:/YananResearchStorage/GuardFed/')
        files=index['files']
        assert files['OFFSERVER_ACCEPTANCE.json']['sha256']==root['offserver_acceptance_sha256']
        assert files['BACKUP_SHA256.json']['sha256']==root['server_receipt_sha256']
        for rid in root['accepted_new_ids']:
            artifacts={name:files['restored/runs/'+rid+'/'+name] for name in ('job.json','result.json','model.pt','provenance.json')}
            assert all(len(pin['sha256'])==64 and pin['bytes']>0 for pin in artifacts.values())
            rows.append(dict(id=rid,checkpoint_sha256=artifacts['model.pt']['sha256'],native_root_path=p.relative_to(R).as_posix(),native_root_sha256=sha(p),raw_index_path=index_path.relative_to(R).as_posix(),raw_index_sha256=sha(index_path),raw_storage_root=index['raw_storage_root'],artifacts=artifacts))
        sources.append(dict(root=p.relative_to(R).as_posix(),root_sha256=sha(p),raw_index_sha256=sha(index_path)))
    assert [r['id'] for r in rows]==delta
    # The prior failure remains evidence; no rechecking/refitting the old47 is performed.
    failure=R/'tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/OFFSERVER_ARRAY_REFIT_CHECK.failure.json'
    assert sha(failure)==prior['Windows_original_refit_failure_sha256']
    print(json.dumps(dict(status='EXACT13_METADATA_BOUND_NOT_EVALUATED_NOT_AUTHORIZED',exact_ids=delta,records=rows,sources=sources,prior_three_view_root=PRIOR,prior_three_view_root_sha256=sha(pp),prior_three_view_accepted=48,new_three_view_accepted=0,pending=13,native_new57_plus_screen_reuse4=61,prior_Windows_exact_refit_remains_failed=True,prior_Windows_failure_sha256=sha(failure),runtime_adapter_required=True,dispatch_authorized=False,new_CNN=0,new_fit=0,test=False),indent=2))
if __name__=='__main__':main()
