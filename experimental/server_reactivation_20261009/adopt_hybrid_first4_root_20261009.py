"""Register only the four reviewed terminal records, with original evidence unchanged."""
from pathlib import Path
import datetime, hashlib, json
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
DELTA = BASE/'accepted_delta_first_20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
proof = read(DELTA/'ROOT_RECORD_REVIEW.json')
assert sha(DELTA/'ROOT_RECORD_REVIEW.json') == '5715216f9828dde66b917e943737886ec79d2348b7acb316156e05f29912cfaf'
assert proof['status'] == 'RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS'
assert proof['accepted_before'] == 0 and proof['accepted_new'] == 4 and proof['local_CNN_inference'] == 0
assert not proof['local_CUDA_initialized'] and not proof['selection_performed']
ready = read(DELTA/'ROOT_READY_DELIVERY.json')
assert proof['delivery_seal_sha256'] == sha(DELTA/'DELIVERY_FILES_SHA256.json')
assert proof['archive_sha256'] == ready['archive_sha256'] == sha(DELTA/'hybrid_first_delta.tar.gz')
assert proof['source_seal_sha256'] == sha(BASE/'FILES_SHA256.json')
assert [r['id'] for r in proof['records']] == ready['accepted_new_ids']
chain = dict(status='PARTIAL_STRICT_OFFSERVER_ROOT_RECORD_REVIEW_ADOPTED',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    planned=32, accepted_total=4, accepted_new_ids=ready['accepted_new_ids'],
    accepted_job_ids=ready['accepted_new_ids'], previous_accepted=0,
    source_seal_sha256=proof['source_seal_sha256'], archive_sha256=proof['archive_sha256'],
    inventory_sha256=sha(DELTA/'MEMBERS.json'), archive_members=53,
    server_strict_sha256=sha(DELTA/'PARTIAL_ACCEPTANCE.json'),
    offserver_tensor_sha256=sha(DELTA/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),
    root_record_review_sha256=sha(DELTA/'ROOT_RECORD_REVIEW.json'),
    delta_dir=DELTA.name, source_startup_archive_reused=ready['source_startup_archive_sha256'],
    log_semantics=ready['log_note'], local_runtime_not_claimed_equal=True,
    original_acceptor_runtime_metadata_bound_to_server=True,
    scientific_checks_and_null_policy_unchanged=True,
    new_CNN_inference=0, selected_recipe=None, test_evaluated=False, formal100_started=False)
target=BASE/'BACKUP_CHAIN_first4_20261009.json'
with target.open('x',encoding='utf8') as stream:json.dump(chain,stream,indent=2);stream.write('\n')
latest=dict(chain_file=target.name,chain_sha256=sha(target),accepted=4,planned=32)
with (BASE/'LATEST_BACKUP.json').open('x',encoding='utf8') as stream:json.dump(latest,stream,indent=2);stream.write('\n')
print(json.dumps(latest))
