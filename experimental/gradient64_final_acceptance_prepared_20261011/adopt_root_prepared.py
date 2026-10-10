"""PREPARED direct root adopter: exact18/prior46; actual delivery SHA arguments required; not executed."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, math, argparse, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guardfed_local_storage import check_bulk_storage

D=Path(__file__).resolve().parent
ROOT=D.parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--delivery-sha256',required=True)
parser.add_argument('--handoff-sha256',required=True)
a=parser.parse_args()
assert all(len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in (a.delivery_sha256,a.handoff_sha256))
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=lambda p:json.loads(Path(p).read_bytes())
check_bulk_storage(0)
assert not (D/'ROOT_ADOPTION_REVIEW.json').exists()
assert H(D/'DELIVERY_FILES_SHA256.json')==a.delivery_sha256
for name,pin in R(D/'DELIVERY_FILES_SHA256.json')['files'].items():
    p=(D/name).resolve()
    assert p.is_relative_to(D.resolve()) and H(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
assert H(D/'ROOT_READY_HANDOFF.json')==a.handoff_sha256
handoff=R(D/'ROOT_READY_HANDOFF.json')
assert H(ROOT/handoff['prior_root_path'])==handoff['prior_root_sha256']=='e5875e4d6a5e15c95d7fb94686411a106b1db39681341c92e51e9e821867f1e6'
assert H(ROOT/handoff['prior_offserver_path'])==handoff['prior_offserver_sha256']
auth=R(D/'AUTHORIZED_SNAPSHOT.json')
assert handoff['accepted_new_ids']==auth['authorized_ids'] and handoff['accepted_before']==46
assert handoff['new_strict_offserver_count']==18 and handoff['strict_offserver_cumulative']==64
prior=R(ROOT/handoff['prior_root_path'])
assert handoff['accepted_job_ids']==prior['accepted_ids']+auth['authorized_ids']
assert len(set(handoff['accepted_job_ids']))==64 and not set(prior['accepted_ids'])&set(auth['authorized_ids'])
assert handoff['accepted_job_ids']==R(D/'PENDING_BINDING.json')['expected_all64_ids']
assert auth['authorized_ids']==R(D/'PENDING_BINDING.json')['remaining18_ids']
assert H(D/'OFFSERVER_ACCEPTANCE.json')==handoff['offserver_sha256']
off=R(D/'OFFSERVER_ACCEPTANCE.json')
assert off['accepted_new_ids']==auth['authorized_ids'] and off['accepted_total']==64
assert off['original_validator_sha256']=='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9'
index=R(D/'RAW_STORAGE_INDEX.json')
assert H(D/'RAW_STORAGE_INDEX.json')==handoff['raw_storage_index_sha256']
bulk=Path(index['raw_storage_root']).resolve()
assert bulk.drive.upper()=='F:' and len(index['files'])==handoff['archive_members']+3
for name,pin in index['files'].items():
    p=(bulk/name).resolve()
    assert p.is_relative_to(bulk) and H(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
archive=Path(handoff['archive_path']);receipt_path=Path(handoff['receipt_path'])
assert H(archive)==handoff['archive_sha256']
assert H(receipt_path)==handoff['receipt_sha256']
source=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
assert H(source)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
spec=importlib.util.spec_from_file_location('original_archive_verifier',source)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
member_proof=v.verify_archive(archive,R(receipt_path))
assert member_proof['pass'] and member_proof['different_host_observed'] and member_proof['members_verified']==handoff['archive_members']
assert R(D/'COLLECTOR_RELEASE.json')['CPU110_released']
tensor=R(D/'SAVED_TENSOR_STATE_CHECK.json')
assert tensor['models']==18 and tensor['CNN_forward_calls']==tensor['optimizer_calls']==tensor['data_loads']==0
assert H(D/'SAVED_TENSOR_STATE_CHECK.json')==off['tensor_proof_sha256']
for row in off['records']:
    assert row['rounds']==70 and row=={r['id']:r for r in handoff['records']}[row['id']]
    assert all(math.isfinite(x) and 0<=x<=1 for x in row['metrics'].values())
    assert row['constant_negative_retained']==(row['metrics']==dict(accuracy=0.5166859616449389,aeod=0.,aspd=0.))
    assert row['data_contract']['image_data_contract']['evaluation_split']=='valid'
    assert row['data_contract']['train_rows']==162770 and row['data_contract']['test_rows']==19867
assert len(off['records'])==18
assert {r['id'] for r in off['records']}==set(auth['authorized_ids'])
assert sum(r['tensor_count'] for r in tensor['records'])==18*8 and sum(r['elements'] for r in tensor['records'])==18*93506
assert {(r['id'],r['checkpoint_sha256']) for r in tensor['records']}=={(r['id'],r['checkpoint_sha256']) for r in off['records']}
proof=dict(status='ROOT_GRADIENT64_EXACT18_ORIGINAL_STRICT_OFFSERVER_ADOPTED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_before=46,accepted_new=18,accepted_total=64,
    accepted_new_ids=auth['authorized_ids'],accepted_ids=handoff['accepted_job_ids'],
    handoff_path=(D/'ROOT_READY_HANDOFF.json').relative_to(ROOT).as_posix(),handoff_sha256=H(D/'ROOT_READY_HANDOFF.json'),
    delivery_seal_sha256=H(D/'DELIVERY_FILES_SHA256.json'),offserver_sha256=H(D/'OFFSERVER_ACCEPTANCE.json'),
    previous_root_sha256=handoff['prior_root_sha256'],raw_index_sha256=H(D/'RAW_STORAGE_INDEX.json'),
    archive_path=archive.as_posix(),archive_sha256=H(archive),archive_members_root_verified=member_proof['members_verified'],raw_files_root_verified=len(index['files']),
    original_archive_verifier_sha256=H(source),saved_tensor_proof_sha256=H(D/'SAVED_TENSOR_STATE_CHECK.json'),
    all_negative_results_retained=True,constant_negative_ids=[r['id'] for r in off['records'] if r['constant_negative_retained']],
    CPU110_released=True,new_CNN=0,new_training=0,old_models_repacked=0,final_test=False,
    method_champion_claim=False,screen64_complete=True,runtime_equivalence_claim=False,
    scope='Exact18 scientific acceptor and saved-state proofs independently executed before root byte/member adoption; no prediction recomputation')
out=D/'ROOT_ADOPTION_REVIEW.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=H(out),accepted_total=64,new=18)))
