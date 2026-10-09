"""Review the locked seven-plus-four terminal deltas; register only after both pass."""
from pathlib import Path
import datetime, hashlib, json, tarfile

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
cases = [
    ('celeba_flgmm_screen_20261009_v2_dispatch', 'accepted_delta_after19_v2_20261009', 19, 7,
     'e18cd7c016af1e0799e22a9dd06f69576934ac9ab831b93401f77e3aa9915c26',
     '2788fdcc0dd5991c376b2dee4adcb9a6ad1f734abcf5d934c4ecf35926dfea78',
     'accepted_delta_after19_v2.tar.gz', 'OFFSERVER_ACCEPTANCE.json', 'runs'),
    ('celeba_hybrid_screen_execution_20261009', 'accepted_delta_after6_20261009', 6, 4,
     '6b28b22bdab9767bffc04be90a26b19214092d8bf74858ba71d4c523f2bb3988',
     '990b3edc5147db79e70eb2e0fa0a5b01e63c738834369f6a44c9f8b69ad3a1b3',
     'hybrid_after6_delta.tar.gz', 'OFFSERVER_MEMBER_TENSOR_PROOF.json', 'screen_runs')]
snapshots = {
    'accepted_delta_after19_v2_20261009': 'f87eb797aef52016d1e9a77de871b5cd4942474545c2e628184f993a2ebfc404',
    'accepted_delta_after6_20261009': 'e27bc9d6dde558ad3470edbb6906f0b44878bac25fca6e610cb8e9237cf52845'}

verified = []
for base_name, delta_name, previous, added, seal_sha, link_sha, archive_name, off_name, runs in cases:
    base = ROOT/'tmp'/base_name; delta = base/delta_name
    seal_path = delta/('FILES_SHA256.json' if runs == 'runs' else 'DELIVERY_FILES_SHA256.json')
    assert sha(seal_path) == seal_sha
    sealed = read(seal_path)
    rows = [dict(path=name, sha256=row['sha256'], size=row['bytes']) for name,row in sealed['artifacts'].items()] if runs == 'runs' else sealed['members']
    for row in rows:
        member = delta/row['path']
        assert member.resolve().is_relative_to(delta.resolve())
        assert sha(member) == row['sha256'] and member.stat().st_size == row['size']
    assert sha(delta/'ROOT_READY_CHAIN_LINK.json') == link_sha
    link = read(delta/'ROOT_READY_CHAIN_LINK.json')
    latest = read(base/'LATEST_BACKUP.json'); previous_chain = base/latest['chain_file']
    assert sha(base/'LATEST_BACKUP.json') == link['previous_latest_sha256'] == sha(delta/'PREVIOUS_LATEST.json')
    assert sha(previous_chain) == latest['chain_sha256'] == link['previous_chain_sha256'] == sha(delta/'PREVIOUS_CHAIN.json')
    prior = read(previous_chain)
    old_ids = prior['accepted_job_ids']; new_ids = link['accepted_new_ids']
    assert latest['accepted'] == len(old_ids) == previous and len(set(new_ids)) == added
    assert not set(old_ids) & set(new_ids)
    assert set(link['accepted_job_ids']) == set(old_ids+new_ids)
    assert link['accepted_total_if_root_adopts'] == len(set(old_ids+new_ids)) == previous+added < 32
    assert link['authorized_snapshot_sha256'] == sha(delta/'AUTHORIZED_SNAPSHOT.json') == snapshots[delta_name]
    archive = delta/archive_name; assert sha(archive) == link['archive_sha256']
    members = read(delta/'MEMBERS.json')['members']
    assert sha(delta/'MEMBERS.json') == link['inventory_sha256']
    with tarfile.open(archive) as tar:
        files = [m for m in tar.getmembers() if m.isfile()]
        assert len(files) == link['archive_members'] == len(members)+1
        assert {m.name for m in files} == set(members)|{'MEMBERS.json'}
        for member in files:
            raw = tar.extractfile(member).read()
            expected = {'sha256': sha(delta/'MEMBERS.json'), 'size': (delta/'MEMBERS.json').stat().st_size} if member.name == 'MEMBERS.json' else members[member.name]
            assert len(raw) == expected['size'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
    strict = read(delta/'PARTIAL_ACCEPTANCE.json'); off = read(delta/off_name)
    assert strict['accepted_new_ids'] == off.get('accepted_new_ids', [r['id'] for r in off['records']]) == new_ids
    assert strict['accepted_new'] == off['accepted_new'] == added
    assert sha(delta/off_name) == link.get('offserver_proof_sha256', link.get('offserver_tensor_proof_sha256'))
    assert sha(delta/'PARTIAL_ACCEPTANCE.json') == link.get('server_acceptance_sha256', link.get('server_strict_sha256'))
    for record in strict['records']:
        out = delta/'restored'/runs/record['id']
        result = read(out/'result.json')
        assert result['dataset'] == 'celeba' and result['seed'] == record['seed'] == 91001
        assert result['rounds'] == record['rounds'] == 70
        assert result['distribution'] == record['distribution'] and result['attack'] == record['attack']
        assert result['metrics'] == record['metrics'] and result['evaluation_stats'] == record['evaluation_stats']
        assert result['evaluation_stats']['prediction_count'] == 19867
        assert result['config']['celeba_evaluation_split'] == 'valid'
        assert sha(out/'model.pt') == record.get('checkpoint_sha256', record.get('model_sha256'))
    if runs == 'screen_runs':
        local = read(delta/'LOCAL_RECORD_CHECKS.json')
        assert sha(delta/'LOCAL_RECORD_CHECKS.json') == link['offserver_original_record_check_sha256']
        assert local['status'] == 'RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS'
        assert local['server_check_receipt_sha256'] == sha(delta/'PARTIAL_ACCEPTANCE.json')
        assert not local['local_CUDA_initialized'] and local['local_runtime_not_claimed_equal']
        old_body = base/'accepted_delta_first_20261009/local_record_bridge_v1/checked_record_body.py'
        assert (delta/'local_record_bridge_v2/checked_record_body.py').read_bytes() == old_body.read_bytes()
    proof = dict(status='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), delivery_seal_sha256=seal_sha,
        ready_link_sha256=link_sha, previous_latest_sha256=sha(base/'LATEST_BACKUP.json'),
        previous_chain_sha256=sha(previous_chain), archive_sha256=sha(archive), members_verified=len(files),
        accepted_before=previous, accepted_new=added, accepted_total=previous+added,
        strict_receipt_sha256=sha(delta/'PARTIAL_ACCEPTANCE.json'), offserver_proof_sha256=sha(delta/off_name),
        original_acceptor_replayed_by_offserver_tool=True, new_inference=0, selection_performed=False,
        scientific_changes=False, final_test=False, formal100_started=False)
    verified.append((base, delta, link, old_ids, new_ids, proof))

for base, delta, link, old_ids, new_ids, proof in verified:
    with (delta/'ROOT_ADOPTION_REVIEW.json').open('x',encoding='utf8',newline='\n') as f:
        json.dump(proof,f,indent=2);f.write('\n')
    chain = dict(status='PARTIAL_STRICT_OFFSERVER_ROOT_RECORD_REVIEW_ADOPTED',
        checked_utc=proof['checked_utc'], accepted_total=len(old_ids+new_ids), planned=32,
        accepted_job_ids=old_ids+new_ids, accepted_new_ids=new_ids, previous_accepted=len(old_ids),
        previous_chain_file=read(base/'LATEST_BACKUP.json')['chain_file'],previous_chain_sha256=proof['previous_chain_sha256'],
        delta_dir=delta.name,archive=next(delta.glob('*.tar.gz')).name,archive_sha256=proof['archive_sha256'],
        inventory_sha256=sha(delta/'MEMBERS.json'),archive_members=proof['members_verified'],
        server_strict_sha256=proof['strict_receipt_sha256'],offserver_proof_sha256=proof['offserver_proof_sha256'],
        root_adoption_path=(delta/'ROOT_ADOPTION_REVIEW.json').relative_to(ROOT).as_posix(),
        root_adoption_sha256=sha(delta/'ROOT_ADOPTION_REVIEW.json'),new_CNN_inference=0,
        selected_recipe=None,test_evaluated=False,formal100_started=False)
    target=base/('BACKUP_CHAIN_'+delta.name+'.json')
    with target.open('x',encoding='utf8',newline='\n') as f:json.dump(chain,f,indent=2);f.write('\n')
    latest=dict(chain_file=target.name,chain_sha256=sha(target),accepted=chain['accepted_total'],planned=32)
    (base/'LATEST_BACKUP.json').write_text(json.dumps(latest,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(dict(base=base.name,**latest)))
