"""Adopt actually returned metadata; no Torch, SSH, inference or dispatch."""
from pathlib import Path
import datetime, hashlib, importlib.util, json
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
BOUND=HERE/'bound_metadata'; STAGE=BOUND/'stage'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    transfer=read(HERE/'BOUND_TRANSFER_VERIFICATION.json')
    assert transfer['status']=='BOUND_METADATA_MEMBERS_RECEIVED_NOT_TRAINING'
    assert transfer['members']==len(transfer['files'])==147
    for name,pin in transfer['files'].items():
        path=BOUND/name
        assert path.resolve().is_relative_to(BOUND.resolve())
        assert sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes']
    package=read(STAGE/'PACKAGE_SHA256.json')
    assert sha(STAGE/'PACKAGE_SHA256.json')==transfer['package_sha256']=='a87a050b497a184efbe18b4649ad6bde40b9ea29a16f9e3efddd0ef1156e3b04'
    for name,pin in package['files'].items(): assert sha(STAGE/name)==pin
    seal=read(STAGE/'PREPARED_SOURCE_SEAL.json')
    assert sha(STAGE/'PREPARED_SOURCE_SEAL.json')=='bce84aa075a242ec89c2fd016dbac8fb7dea4a1b369074a30fefd5ce6e61d7af'
    for name,pin in seal['files'].items(): assert sha(STAGE/name)==pin['sha256']
    approval=read(HERE/'BIND_APPROVAL.json');binding=read(STAGE/'BINDINGS.json')
    assert sha(HERE/'BIND_APPROVAL.json')==sha(STAGE/'BIND_APPROVAL.json')==binding['bind_approval_sha256']
    assert approval['source_seal_sha256']==binding['source_seal_sha256']==sha(STAGE/'PREPARED_SOURCE_SEAL.json')
    assert approval['execute_authorized'] is False and binding['execution_authorized'] is False
    assert sha(STAGE/'SELECTED_SUMMARY.json')==approval['summary_sha256']
    assert sha(STAGE/'ROOT32_ADOPTION.json')==approval['root32_sha256']
    manifest=read(STAGE/'manifest.json');protocol=read(STAGE/'ORIGINAL_PROTOCOL.json')
    candidate=binding['selected_recipe'];assert candidate['id']==approval['selected_recipe']
    spec=importlib.util.spec_from_file_location('bound_grid_identity',STAGE/'identity.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    jobs=[];gates=[]
    for entries,target in [(manifest['jobs'],jobs),(manifest['preflight_jobs'],gates)]:
        for entry in entries:
            job=read(STAGE/entry['job']);assert sha(STAGE/entry['job'])==entry['job_sha256']
            assert job['id']==entry['id'] and job['distribution']==entry['distribution'] and job['attack']==entry['attack'] and job['config']['seed']==entry['seed']
            module.validate_job(job,protocol,candidate);target.append(job)
    refs=manifest['reused_jobs'];module.validate_grid(jobs,gates,refs)
    summary=read(STAGE/'SELECTED_SUMMARY.json')
    records={r['id']:r for group in summary['all_candidates'] for r in group['records']}
    for ref in refs:
        accepted=ref['accepted_record'];original=records[ref['id']]
        assert accepted==dict(original,rounds=70,job_sha256=ref['job_sha256'],source_hashes=protocol['source_hashes'])
        assert ref['seed']==91001 and ref['source']=='ORIGINAL_HYBRID_SCREEN_ONLY'
    for scope,items in [(read(STAGE/'full_scope.json'),manifest['jobs']),(read(STAGE/'gate_scope.json'),manifest['preflight_jobs'])]:
        assert scope['jobs']==items
        assert scope['protected_source_hashes']==read(BOUND/'BOUND_IDENTITY.json')['source_data_hashes']
        for name,pin in scope['local_hashes'].items(): assert sha(STAGE/name)==pin
    for name in ('EXECUTION_AUTHORIZATION.json','GATE_ACCEPTANCE.json','GATE_FAILURE.json','gate_runs','runs','queue_progress.json'):
        assert not (STAGE/name).exists()
    result=dict(status='ROOT_HYBRID100_BOUND_METADATA_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        package_sha256=transfer['package_sha256'],bound_offserver_sha256=sha(HERE/'BOUND_TRANSFER_VERIFICATION.json'),
        implementation_source_seal_sha256=sha(STAGE/'PREPARED_SOURCE_SEAL.json'),
        new=96,reused=4,canaries=7,received_members=147,new_jobs_checked=96,gate_jobs_checked=7,
        selected_recipe=candidate,summary_sha256=sha(STAGE/'SELECTED_SUMMARY.json'),root32_sha256=sha(STAGE/'ROOT32_ADOPTION.json'),
        bind_approval_sha256=sha(HERE/'BIND_APPROVAL.json'),
        actual_bound_identity_sha256=sha(BOUND/'BOUND_IDENTITY.json'),
        independent_source_review_sha256=approval['independent_source_review_sha256'],
        original_summary_and_ranking_unchanged=True,old_four_explicit_reuse=True,
        prior_failure_preserved=True,execution_authorized=False,final_test=False,
        scope='Actual source/config/data/checkpoint metadata identity; not7 gate acceptance or96 scientific results.')
    with (HERE/'ROOT_BOUND_ADOPTION.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
