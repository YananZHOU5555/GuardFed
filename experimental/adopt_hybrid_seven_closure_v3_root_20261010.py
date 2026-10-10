"""Adopt the actual exact-seven archive; no inference or remote mutation."""
from pathlib import Path
import datetime,hashlib,json
from guardfed_local_storage import STORAGE_ROOT,check_bulk_storage
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
COLLECT=HERE/'seven_canary_collection'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    check_bulk_storage()
    handoff=read(COLLECT/'OFFSERVER_HANDOFF.json')
    assert handoff['status']=='ACTUAL_HYBRID7_STRICT_OFFSERVER_PASS_ROOT_PENDING'
    folder=Path(handoff['bulk_path']);assert folder.resolve().is_relative_to(STORAGE_ROOT.resolve())
    proof=folder/'verified/OFFSERVER_VERIFICATION.json';receipt=folder/'BACKUP_RECEIPT.json'
    assert sha(proof)==handoff['offserver_sha256'] and sha(receipt)==handoff['receipt_sha256']
    r=read(receipt);p=read(proof)
    assert p['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON' and p['receipt_sha256']==sha(receipt)
    assert sha(folder/'seven_canaries.tar.gz')==handoff['archive_sha256']==r['archive_sha256']==p['archive_sha256']
    assert p['member_count']==r['archive_members'] and p['CNN_calls']==p['formal_table_samples']==0
    inventory=folder/'verified/MEMBERS.json';assert sha(inventory)==r['inventory_sha256']==p['inventory_sha256']
    pins=read(inventory)['files'];assert len(pins)+1==p['member_count']
    for n,pin in pins.items():
        path=folder/'verified'/n
        assert path.resolve().is_relative_to((folder/'verified').resolve())
        assert sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes']
    stage=folder/'verified/stage';gate=stage/'GATE_ACCEPTANCE.json';local=p['local']
    assert (local['accepted_new'],local['same_horizon_pairs'],local['total_runs'],local['rounds'],local['formal_table_samples'])==(7,2,7,3,0)
    assert sha(gate)==local['gate_sha256']==r['gate_sha256']==handoff['gate_sha256']
    assert local['package_sha256']==r['package_sha256']==handoff['package_sha256']=='a87a050b497a184efbe18b4649ad6bde40b9ea29a16f9e3efddd0ef1156e3b04'
    assert sha(stage/'PACKAGE_SHA256.json')==local['package_sha256']
    manifest=read(stage/'manifest.json');g=read(gate)
    assert g['accepted_ids']==[j['id'] for j in manifest['preflight_jobs']]==[j['id'] for j in local['records']]
    assert len(set(g['accepted_ids']))==7 and g['pairs']==local['pair_ids']
    remote=read(folder/'verified/evidence/REMOTE_STRICT.json')
    assert remote['source_data_before']==remote['source_data_after'] and remote['pair_ids']==local['pair_ids']
    assert remote['server_runtime']['CPU']==[107] and remote['local_CUDA_initialized'] is False
    assert remote['original_compare_source_sha256']==local['original_compare_source_sha256']=='69902e93528ab7f26b1347046d832b1cfd67ccb3f915de806452def1f4b1375b'
    assert local['canary_runner_sha256']=='722c83d5ebb4c211815480cde267a9250e0216c3cbaddbbf6d7a51d4736df29e'
    for pair in g['pairs']:
        assert pair['all_metrics_model_attacks_diagnostics_rng_exact'] is True
        assert sha(stage/'gate_runs'/pair['hybrid']/'model.pt')==sha(stage/'gate_runs'/pair['legacy']/'model.pt')
    startup=HERE/'ROOT_CANARY_STARTUP.json'
    assert sha(startup)=='d90c19c8aedf12a2ac58e37d653b638fcdae8e9b0044d3fcf6487af172a4bbd1'
    authorization=stage/'EXECUTION_AUTHORIZATION.json'
    assert sha(authorization)=='7de75426971da9509ae91b81a081977c5aa4160b6d12e706b0b3cca55eb91882'
    auth=read(authorization);assert auth['scope']=='seven_same_horizon_3round_canaries' and auth['package_sha256']==local['package_sha256']
    assert auth['max_workers']==auth['cpu_threads_per_worker']==1 and auth['allowed_cpus']==[104] and auth['gpu_index']==0 and auth['automatic_retry'] is False and auth['final_test'] is False
    assert sha(folder/'verified/evidence/APPROVAL.json')==auth['root_approval_sha256']==sha(HERE/'ROOT_SEVEN_CANARY_APPROVAL.json')
    preflight=read(folder/'verified/evidence/PREFLIGHT.json')
    assert preflight['service']['stdout'].split()[1]=='EXITED' and preflight['no_producer'] is True
    result=dict(status='ROOT_SEVEN_HYBRID_CANARIES_OFFSERVER_ADOPTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        package_sha256=r['package_sha256'],gate_sha256=sha(gate),offserver_sha256=sha(proof),receipt_sha256=sha(receipt),archive_sha256=r['archive_sha256'],
        offserver_path=proof.as_posix(),gate_path=gate.as_posix(),archive_path=(folder/'seven_canaries.tar.gz').as_posix(),
        inventory_sha256=sha(inventory),archive_members_verified=p['member_count'],
        root_bound_sha256=sha(HERE/'ROOT_BOUND_ADOPTION.json'),root_canary_startup_sha256=sha(startup),
        collection_source_seal_sha256=r['source_seal_sha256'],collection_handoff_sha256=sha(COLLECT/'OFFSERVER_HANDOFF.json'),
        canary_authorization_sha256=sha(authorization),accepted_new_canaries=7,same_horizon_pairs=2,total_canary_runs=7,rounds=3,
        original_saved_comparison_reexecuted=True,source_data_before_after_exact=True,models_repacked_from_70round=0,
        formal_table_samples=0,formal100_started=False,final_test=False,negative_results_retained=True,
        limitations='Three rounds check only early implementation/attacks. Pair comparisons cover saved terminal weights/final RNG summaries and all recorded round metrics/diagnostics; no per-round weights/full-state recovery or universal70-round equivalence. Offserver runtime explicitly differs from training.')
    out=HERE/'ROOT_SEVEN_CANARY_CLOSURE.json'
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=out.relative_to(ROOT).as_posix(),sha256=sha(out),members=p['member_count'],canaries=7,pairs=2)))
if __name__=='__main__':main()
