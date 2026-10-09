"""Review the linked seven-ID delta and advance only its canonical backup pointer."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
DELTA=BASE/'accepted_delta_after6_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
link=read(DELTA/'ROOT_READY_CHAIN_LINK.json')
assert sha(DELTA/'ROOT_READY_CHAIN_LINK.json')=='c1479011c053b467cad42129c5345244ae2b9212d6373451f2a9528eac739e54'
previous=BASE/link['previous_chain_file'];old=read(previous)
assert sha(previous)==link['previous_chain_sha256']=='e2e6b6f98abc5d66bc493c2b4e09dc6d9470fca46116778d01e08c69328528a4'
assert link['accepted_previous']==6 and link['accepted_new']==7 and link['accepted_total_if_root_adopts']==13
assert set(old['accepted_job_ids']).isdisjoint(link['accepted_new_ids']) and set(link['accepted_job_ids'])==set(old['accepted_job_ids'])|set(link['accepted_new_ids'])
assert len(set(link['accepted_job_ids']))==13 and link['source_data_before_after_same'] and link['no_training_inference_selection_or_queue_change']
assert sha(DELTA/'OFFSERVER_ACCEPTANCE.json')==link['offserver_proof_sha256']=='e2711f3eac1a4306b8ec8d929b7f732c7a2c70c1420268d04a39b9f9db539003'
proof=read(DELTA/'OFFSERVER_ACCEPTANCE.json')
assert proof['status']=='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED' and proof['original_checked_result_replayed_locally'] and proof['different_host_observed']
assert [r['id'] for r in proof['records']]==link['accepted_new_ids'] and all(r['rounds']==70 and r['seed']==91001 for r in proof['records'])
assert sha(DELTA/'PARTIAL_ACCEPTANCE.json')==link['server_acceptance_sha256']
archive=DELTA/link['archive'];inventory=DELTA/'MEMBERS.json'
assert sha(archive)==link['archive_sha256'] and sha(inventory)==link['inventory_sha256']
members=read(inventory)['members']
with tarfile.open(archive) as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==link['archive_members']==77
    assert set(bundle.getnames())==set(members)|{'MEMBERS.json'}
    for item in bundle:
        rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        data=bundle.extractfile(item).read()
        expected=link['inventory_sha256'] if item.name=='MEMBERS.json' else members[item.name]['sha256']
        assert hashlib.sha256(data).hexdigest()==expected
        if item.name.startswith('runs/'):
            assert rel.parts[1] in set(link['accepted_new_ids'])
for row in proof['records']:
    assert members['runs/'+row['id']+'/model.pt']['sha256']==row['checkpoint_sha256']
root_proof=dict(status='ROOT_FLGMM_NEW7_LINK_ARCHIVE_MEMBER_AND_ORIGINAL_ACCEPTOR_REVIEW_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_chain_sha256=sha(previous),
    reviewed_link_sha256=sha(DELTA/'ROOT_READY_CHAIN_LINK.json'),offserver_proof_sha256=sha(DELTA/'OFFSERVER_ACCEPTANCE.json'),
    archive_sha256=sha(archive),members_verified=77,accepted_before=6,accepted_new=7,accepted_total=13,
    original_acceptor_replayed_by_offserver_tool=True,new_CNN_inference=0,scientific_changes=False,selection_performed=False)
root_path=DELTA/'ROOT_ADOPTION_REVIEW.json'
with root_path.open('x',encoding='utf8') as stream:json.dump(root_proof,stream,indent=2);stream.write('\n')
chain=dict(link,status='PARTIAL_ACCEPTED_OFFSERVER_ROOT_ADOPTED',accepted_total=13,root_adoption_sha256=sha(root_path),
           delta_dir=DELTA.name,root_adoption_path=str(root_path.relative_to(BASE)))
chain_path=BASE/'BACKUP_CHAIN_increment_after6_20261009.json'
with chain_path.open('x',encoding='utf8') as stream:json.dump(chain,stream,indent=2);stream.write('\n')
latest=read(BASE/'LATEST_BACKUP.json');assert latest['chain_sha256']==sha(previous) and latest['accepted']==6
latest.update(chain_file=chain_path.name,chain_sha256=sha(chain_path),accepted=13,previous_chain_sha256=sha(previous))
(BASE/'LATEST_BACKUP.json').write_text(json.dumps(latest,indent=2)+'\n',encoding='utf8')
print(json.dumps(root_proof|{'chain_sha256':sha(chain_path),'root_adoption_sha256':sha(root_path)}))
