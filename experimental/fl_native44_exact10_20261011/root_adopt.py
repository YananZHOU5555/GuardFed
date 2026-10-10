"""Root metadata/archive review of the already executed original strict verifier."""
import datetime, hashlib, json, os, tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write_new(p, obj):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n')

h = read(BASE/'ROOT_READY_HANDOFF.json')
assert sha(BASE/'ROOT_READY_HANDOFF.json') == '69c67366a2322090d758427d3a227eef76b5c111d7012e5deb911b2d11754679'
seal = read(BASE/'DELIVERY_FILES_SHA256.json')
assert sha(BASE/'DELIVERY_FILES_SHA256.json') == 'da2605a87c388ef56ca287c0dc049423e5921c74bec173f5b6a17198c710f008'
assert len(seal['files']) == 67
for name, item in seal['files'].items():
    p = BASE/name
    assert p.resolve().is_relative_to(BASE.resolve())
    assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name
latest_path = ROOT/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json'
assert sha(latest_path) == h['previous_latest_sha256']
latest = read(latest_path)
previous_path = ROOT/h['previous_root_path']
assert sha(previous_path) == h['previous_root_sha256'] == latest['root_adoption_sha256']
previous = read(previous_path)
old_off_path = Path(latest['next_collector_previous_path'])
assert sha(old_off_path) == h['previous_actual_offserver_sha256']
old_off = read(old_off_path)
off_path = Path(h['offserver_acceptance_path'])
assert sha(off_path) == h['offserver_acceptance_sha256']
off = read(off_path)
assert off['status'] == 'PARTIAL_ACCEPTED_OFFSERVER_VERIFIED'
assert (off['accepted_new'], off['accepted_total'], off['planned_new'], off['reused_separately']) == (10,54,96,4)
assert off['new_ids'] == h['accepted_new_ids']
assert off['accepted_job_ids'] == previous['accepted_job_ids']+off['new_ids']
assert old_off['accepted_job_ids'] == previous['accepted_job_ids']
assert len(set(off['accepted_job_ids'])) == 54
assert off['original_checked_result_replayed_locally'] and off['CNN_calls'] == 0 and not off['final_test']
assert off['verification_host'] != off['source_host']
assert off['previous_chain_sha256'] == sha(old_off_path)
assert off['package_sha256'] == previous['source_package_sha256'] == h['source_package_sha256']
archive = Path(h['archive_path']); assert sha(archive) == h['archive_sha256'] == off['archive_sha256']
manifest_path = Path(h['member_manifest_path'])
assert sha(manifest_path) == h['inventory_sha256'] == off['inventory_sha256']
members = read(manifest_path)['members']
receipt = read(BASE/'SERVER_COLLECTOR_RECEIPT.json')
assert sha(BASE/'SERVER_COLLECTOR_RECEIPT.json') == h['server_backup_receipt_sha256'] == off['server_backup_receipt_sha256']
assert receipt['archive_sha256'] == sha(archive) and receipt['accepted_new_ids'] == off['new_ids']
with tarfile.open(archive, 'r:gz') as tf:
    listed = tf.getmembers()
    assert len(listed) == len(members)+1 == h['archive_members'] == 107
    assert len({m.name for m in listed}) == len(listed)
    for m in listed:
        assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
        data = tf.extractfile(m).read()
        if m.name == 'MEMBERS.json': assert data == manifest_path.read_bytes()
        else: assert len(data) == members[m.name]['size'] and hashlib.sha256(data).hexdigest() == members[m.name]['sha256'], m.name
    def archived(name): return json.load(tf.extractfile(name))
    assert hashlib.sha256(tf.extractfile('verify_delta_offserver.py').read()).hexdigest() == 'ecb6627ab90aafa667795d2b3146f6b67abb1924462534640fc31b5dc7b7edfe'
    for record in off['records']:
        jid=record['id']; prefix=f'runs/{jid}/'
        assert jid in off['new_ids'] and record['rounds']==70
        result=archived(prefix+'result.json'); progress=archived(prefix+'progress.json'); accepted=archived(prefix+'acceptance.json')
        assert result['rounds']==progress['round']==accepted['rounds']==70
        assert len(result['round_summaries'])==70
        assert result['metrics']==progress['metrics']==record['metrics']
        assert (accepted['status'],accepted['evaluation_split'],accepted['train_rows'],accepted['evaluation_rows'])==('PASS','valid',162770,19867)
        assert (result['seed'],result['distribution'],result['attack'])==(record['seed'],record['distribution'],record['attack'])
        assert record['checkpoint_sha256']==members[prefix+'model.pt']['sha256']
        assert record['original_acceptance_sha256']==members[prefix+'acceptance.json']['sha256']
        assert record['job_sha256']==members[prefix+'job.json']['sha256']
        assert record['evaluation_stats']['prediction_count']==19867
    assert {n.split('/')[1] for n in members if n.startswith('runs/')} == set(off['new_ids'])
tensor=read(BASE/'SAVED_TENSOR_STATE_CHECK.json')
assert sha(BASE/'SAVED_TENSOR_STATE_CHECK.json')==h['full_saved_tensor_check_sha256']
assert tensor['status']=='SAVED_FULL_STATE_LAYOUT_DTYPE_FINITE_PASS_NO_FORWARD'
assert tensor['offserver_acceptance_sha256']==sha(off_path) and tensor['CNN_forward_calls']==tensor['optimizer_calls']==tensor['data_loads']==0
assert [r['id'] for r in tensor['records']]==off['new_ids']
assert sum(r['tensor_count'] for r in tensor['records'])==80 and sum(r['elements'] for r in tensor['records'])==935060
for r in tensor['records']: assert r['checkpoint_bytes_unchanged'] and all(t['finite'] for t in r['tensors'])
proof=dict(status='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_before=44,accepted_new=10,accepted_total=54,
    accepted_new_ids=off['new_ids'],accepted_job_ids=off['accepted_job_ids'],planned_new=96,reused_separately=4,
    archive_members_verified=107,archive_sha256=sha(archive),offserver_acceptance_sha256=sha(off_path),
    server_receipt_sha256=sha(BASE/'SERVER_COLLECTOR_RECEIPT.json'),delivery_seal_sha256=sha(BASE/'DELIVERY_FILES_SHA256.json'),
    source_package_sha256=off['package_sha256'],previous_root_adoption_path=h['previous_root_path'],previous_root_adoption_sha256=sha(previous_path),
    previous_offserver_path=str(old_off_path),previous_offserver_sha256=sha(old_off_path),archive_local_path=str(archive),
    raw_storage_index_sha256=h['raw_storage_index_sha256'],helper_CPU=110,old_models_repacked=0,root_new_CNN=0,final_test=False,
    original_checker_replayed_by_offserver_verifier=True,original_training_runtime_not_recreated=True,negative_results_preserved=True,
    not_complete_scenario_or_final_summary=True,delivery_files_verified=67,old44_ordered_prefix_exact=True,
    saved_tensor_check_sha256=sha(BASE/'SAVED_TENSOR_STATE_CHECK.json'))
proof_path=BASE/'ROOT_ADOPTION_REVIEW.json';write_new(proof_path,proof)
next_latest=dict(accepted_total=54,root_adoption_path=proof_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(proof_path),
    next_collector_previous_path=off_path.as_posix(),next_collector_previous_sha256=sha(off_path))
assert sha(latest_path)==h['previous_latest_sha256']
temp=latest_path.with_name('LATEST_BACKUP.root54.tmp');write_new(temp,next_latest);os.replace(temp,latest_path)
print(json.dumps(dict(status='ROOT_ADOPTED',accepted_total=54,root_path=proof_path.relative_to(ROOT).as_posix(),root_sha256=sha(proof_path),latest_sha256=sha(latest_path))))
