"""Adopt the externally reviewed exact-four delta using original archive verification."""
from pathlib import Path
import datetime, hashlib, importlib.util, json
from guardfed_local_storage import check_bulk_storage

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'tmp/celeba_gradient64_delta_after1_20261010'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=lambda p:json.loads(Path(p).read_bytes())
check_bulk_storage(0)
assert H(D/'DELIVERY_FILES_SHA256.json')=='833affd3beb8264c139683fc2ab80397b7ec5c7b31a7f70ff1cd4f15fed8f7ba'
for name,pin in R(D/'DELIVERY_FILES_SHA256.json')['files'].items():
    p=(D/name).resolve()
    assert p.is_relative_to(D.resolve()) and H(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
assert H(D/'ROOT_READY_HANDOFF.json')=='bfa52cdd3969c888bcc024a54f00219d6bc484c17fd69d7d3d8e3bb43fecb47e'
handoff=R(D/'ROOT_READY_HANDOFF.json')
assert H(ROOT/handoff['prior_root_path'])==handoff['prior_root_sha256']=='ca4a38076e5080d94069cdabd50a032d84a96a339914761ec568db44a43db978'
assert H(ROOT/handoff['prior_offserver_path'])==handoff['prior_offserver_sha256']
auth=R(D/'AUTHORIZED_SNAPSHOT.json')
assert handoff['accepted_new_ids']==auth['authorized_ids'] and handoff['accepted_before']==1
assert handoff['new_strict_offserver_count']==4 and handoff['strict_offserver_cumulative']==5
prior=R(ROOT/handoff['prior_root_path'])
assert handoff['accepted_job_ids']==prior['accepted_ids']+auth['authorized_ids']
assert len(set(handoff['accepted_job_ids']))==5 and not set(prior['accepted_ids'])&set(auth['authorized_ids'])
assert H(D/'OFFSERVER_ACCEPTANCE.json')==handoff['offserver_sha256']=='b3250c5986728a734dfee38eceef63ab160e65222dc875e8a1f310fc6848e6dd'
off=R(D/'OFFSERVER_ACCEPTANCE.json')
assert off['accepted_new_ids']==auth['authorized_ids'] and off['accepted_total']==5
assert off['original_validator_sha256']=='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9'
index=R(D/'RAW_STORAGE_INDEX.json')
assert H(D/'RAW_STORAGE_INDEX.json')=='9996a0b647f24e63a708cdb7e0297e135831625b3fd3f33097a8e4fc966f48b2'
bulk=Path(index['raw_storage_root']).resolve()
assert bulk.drive.upper()=='F:' and len(index['files'])==135
for name,pin in index['files'].items():
    p=(bulk/name).resolve()
    assert p.is_relative_to(bulk) and H(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
archive=Path(handoff['archive_path']);receipt_path=Path(handoff['receipt_path'])
assert H(archive)==handoff['archive_sha256']=='f1a2133af5b537246570c80d732bfdebc86037b7bc46d87ca6c55d2e070c8c10'
assert H(receipt_path)==handoff['receipt_sha256']
source=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
assert H(source)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
spec=importlib.util.spec_from_file_location('original_archive_verifier',source)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
member_proof=v.verify_archive(archive,R(receipt_path))
assert member_proof['pass'] and member_proof['different_host_observed'] and member_proof['members_verified']==132
assert R(D/'COLLECTOR_RELEASE.json')['CPU106_released']
tensor=R(D/'SAVED_TENSOR_STATE_CHECK.json')
assert tensor['models']==4 and tensor['CNN_forward_calls']==tensor['optimizer_calls']==tensor['data_loads']==0
assert H(D/'SAVED_TENSOR_STATE_CHECK.json')==off['tensor_proof_sha256']
for row in off['records']:
    assert row['rounds']==70 and row['metrics']==dict(accuracy=0.5166859616449389,aeod=0.,aspd=0.)
    assert row['constant_negative_retained'] and row['data_contract']['image_data_contract']['evaluation_split']=='valid'
    assert row['data_contract']['train_rows']==162770 and row['data_contract']['test_rows']==19867
proof=dict(status='ROOT_GRADIENT64_EXACT4_ORIGINAL_STRICT_OFFSERVER_ADOPTED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_before=1,accepted_new=4,accepted_total=5,
    accepted_new_ids=auth['authorized_ids'],accepted_ids=handoff['accepted_job_ids'],
    handoff_path=(D/'ROOT_READY_HANDOFF.json').relative_to(ROOT).as_posix(),handoff_sha256=H(D/'ROOT_READY_HANDOFF.json'),
    delivery_seal_sha256=H(D/'DELIVERY_FILES_SHA256.json'),offserver_sha256=H(D/'OFFSERVER_ACCEPTANCE.json'),
    previous_root_sha256=handoff['prior_root_sha256'],raw_index_sha256=H(D/'RAW_STORAGE_INDEX.json'),
    archive_path=archive.as_posix(),archive_sha256=H(archive),archive_members_root_verified=132,raw_files_root_verified=135,
    original_archive_verifier_sha256=H(source),saved_tensor_proof_sha256=H(D/'SAVED_TENSOR_STATE_CHECK.json'),
    all_negative_results_retained=True,constant_negative_ids=auth['authorized_ids'],
    CPU106_released=True,new_CNN=0,new_training=0,old_models_repacked=0,final_test=False,
    method_champion_claim=False,screen64_complete=False,runtime_equivalence_claim=False,
    scope='Exact4 scientific acceptor and saved-state proofs independently executed before root byte/member adoption; no prediction recomputation')
out=D/'ROOT_ADOPTION_REVIEW.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=H(out),accepted_total=5,new=4)))
