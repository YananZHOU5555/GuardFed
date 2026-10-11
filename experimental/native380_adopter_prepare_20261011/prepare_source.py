"""Source/metadata preparation only. Never import or execute the adopter."""
from pathlib import Path
import ast, datetime, difflib, hashlib, json

R = Path(__file__).resolve().parents[2]
H = Path(__file__).resolve().parent
P = R / 'tmp/mechanism_F_nonIID_FFlip10_prepare_20261011/adopt_root_370.py'
B = R / 'tmp/mechanism_F_nonIID_FedSA10_prepare_20261011'
TAG = 'root_delta_20261011T024637Z'
S = B / TAG
C = R / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
PRIOR = C / 'root_delta_20261011T014706Z'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
def save(name, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    with (H / name).open('xb') as f: f.write(raw)

assert sha(P) == 'd2e4fc9bb212b1b0ff06ff84e2a120d2c5244cbd52566b972066cf06e049b072'
assert sha(PRIOR / 'ROOT_DELTA_VERIFICATION.json') == '4ae0a66ca2cc48289b9a7a18272b83652c73c9da16b2d65989ecd5a1166d002e'
proof = read(S / 'DELTA_VERIFICATION.json')
off = read(S / 'OFFSERVER_VERIFICATION.json')
prior = read(PRIOR / 'ROOT_DELTA_VERIFICATION.json')
old_report = read(PRIOR / 'inspection/inspection.json')
report = read(S / 'inspection/inspection.json')
ids = [f'minus_F_non-IID_FedSA_seed{s}' for s in range(91001, 91011)]
assert proof['new_ids'] == ids and proof['total_new_strict_and_offserver'] == proof['observed_native_count'] == 380
assert proof['observed_not_backed_up_ids'] == [] and off['members_verified'] == 110
assert sha(S / 'OFFSERVER_VERIFICATION.json') == proof['offserver_proof_sha256']
assert sha(S / (TAG + '.tar.gz.receipt.json')) == proof['receipt_sha256']
assert sha(S / 'inspection/inspection.json') == proof['inspection_sha256']
assert sha(S / 'verified_ledger.json') == proof['ledger_sha256']
assert sha(B / 'PARENT_LEDGER.json') == proof['previous_ledger_sha256'] == prior['ledger_sha256']
assert len(read(B / 'PARENT_LEDGER.json')['entries']) == 51
assert read(S / 'verified_ledger.json')['entries'][:-1] == read(B / 'PARENT_LEDGER.json')['entries']
assert len(read(S / 'verified_ledger.json')['entries']) == 52
assert sha(PRIOR / 'inspection/inspection.json') == prior['inspection_sha256']
assert prior['observed_not_backed_up_ids'] == [ids[0]] and len(old_report['records']) == 471
old_rows = [row for row in old_report['records'] if row['id'] not in prior['observed_not_backed_up_ids']]
old_ids = {row['id'] for row in old_rows}
assert len(old_rows) == len(old_ids) == 470
assert [row for row in report['records'] if row['id'] in old_ids] == old_rows
assert set(report['accepted_new_ids']) - (set(old_report['accepted_new_ids']) - set(prior['observed_not_backed_up_ids'])) == set(ids)
seal = read(B / 'FILES_SHA256.json')
for name, pin in seal['files'].items():
    assert (B / name).stat().st_size == pin['bytes'] and sha(B / name) == pin['sha256']

original = P.read_bytes()
text = original.decode('utf8')
projection_old = """old_ids = {row['id'] for row in old_report['records']}
assert len(old_ids) == 460 and [row for row in report['records'] if row['id'] in old_ids] == old_report['records']
assert set(proof['new_ids']) <= set(report['accepted_new_ids']) - set(old_report['accepted_new_ids'])
assert sorted(set(report['accepted_new_ids']) - set(old_report['accepted_new_ids']) - set(proof['new_ids'])) == proof['observed_not_backed_up_ids'] == ACTUAL['pending_closed_ids']"""
projection_new = """prior_root = read(PRIOR / 'ROOT_DELTA_VERIFICATION.json')
assert sha(PRIOR / 'inspection/inspection.json') == prior_root['inspection_sha256']
prior_pending = prior_root['observed_not_backed_up_ids']
assert prior_pending == ['minus_F_non-IID_FedSA_seed91001'] and old_report['new_count'] == 371 and len(old_report['records']) == 471
old_records = [row for row in old_report['records'] if row['id'] not in prior_pending]
old_ids = {row['id'] for row in old_records}
prior_accepted_ids = set(old_report['accepted_new_ids']) - set(prior_pending)
assert len(old_ids) == len(old_records) == 470 and len(prior_accepted_ids) == 370
assert [row for row in report['records'] if row['id'] in old_ids] == old_records
assert set(proof['new_ids']) <= set(report['accepted_new_ids']) - prior_accepted_ids
assert sorted(set(report['accepted_new_ids']) - prior_accepted_ids - set(proof['new_ids'])) == proof['observed_not_backed_up_ids'] == ACTUAL['pending_closed_ids']"""
edits = [
    ('BASE = Path(__file__).resolve().parent', "HERE = Path(__file__).resolve().parent\nBASE = ROOT / 'tmp/mechanism_F_nonIID_FedSA10_prepare_20261011'"),
    ("actual_path=BASE/'ACTUAL_INPUTS.json'", "actual_path=HERE/'ACTUAL_INPUTS.json'"),
    ('ROOT_REVIEWED_ACTUAL_NATIVE_DELTA_AFTER360_NONIID_FFLIP10_READY_TO_ADOPT', 'ROOT_REVIEWED_ACTUAL_NATIVE_DELTA_AFTER370_NONIID_FEDSA10_READY_TO_ADOPT'),
    ('==TOTAL-360 and TOTAL==370', '==TOTAL-370 and TOTAL==380'),
    ('minus_F_non-IID_F Flip_seed', 'minus_F_non-IID_FedSA_seed'),
    ('root_delta_20261011T002530Z', 'root_delta_20261011T014706Z'),
    ('396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418', '4ae0a66ca2cc48289b9a7a18272b83652c73c9da16b2d65989ecd5a1166d002e'),
    ('c5dcabc47c5dd2c930413a296c26bf27aae01286833c3001390f66935b36ceff', '2c741a36168e6604790648948dfef81e0c869b3fb2b486e5f5eabff3c1df98b1'),
    ("BASE / 'COLLECTOR_SOURCE_REVIEW.json'", "HERE / 'ROOT_SOURCE_REVIEW.json'"),
    ("BASE/'COLLECTOR_SOURCE_REVIEW.json'", "HERE/'ROOT_SOURCE_REVIEW.json'"),
    ('67c32aced70438bf7bf614c5862792f1887ddeb4fbede5e4da06cc157fd992ef', '4baa91c13e3310c851291ba65c34ba60af188d14b2d02d5eba72b1193bbfd38a'),
    ("len(old_ledger['entries']) == 50", "len(old_ledger['entries']) == 51"),
    (projection_old, projection_new),
    ('native_delta_after360_nonIID_FFlip10_20261011', 'native_delta_after370_nonIID_FedSA10_20261011'),
    ('verified_ledger.after360_nonIID_FFlip10.root.tmp', 'verified_ledger.after370_nonIID_FedSA10.root.tmp'),
    ('old460_records_exact=True', 'old470_records_exact=True'),
]
for before, after in edits:
    assert text.count(before) >= 1, before
    text = text.replace(before, after)
raw = text.encode('utf8')
inverse = text
for before, after in reversed(edits): inverse = inverse.replace(after, before)
assert inverse.encode('utf8') == original
compile(raw, 'native380_adopter_source_only', 'exec')
block_checks = {}
for label, start, end in [
    ('archive_member_and_per_record_identity', 'archive = Path(proof[', '\ndest = CANON / TAG'),
    ('receipt_and_offserver_lineage', 'receipt_path = SRC /', '\nreport = read('),
    ('source_full_reuse_identity', "for key in ('frozen_source_hashes'", '\nold_ids ='),
]:
    if label == 'source_full_reuse_identity':
        old_block = original.split(start.encode(), 1)[1].split(end.encode(), 1)[0]
        new_block = raw.split(start.encode(), 1)[1].split(b'\nprior_root =', 1)[0]
    else:
        old_block = original.split(start.encode(), 1)[1].split(end.encode(), 1)[0]
        new_block = raw.split(start.encode(), 1)[1].split(end.encode(), 1)[0]
    assert old_block == new_block
    # The two complete executable blocks parse independently; the reuse block begins after its for header.
    ast_equal = None if label == 'source_full_reuse_identity' else ast.dump(ast.parse((start.encode()+old_block).decode()), include_attributes=False) == ast.dump(ast.parse((start.encode()+new_block).decode()), include_attributes=False)
    block_checks[label] = dict(raw_bytes_exact=True, ast_exact=ast_equal, sha256=hashlib.sha256(old_block).hexdigest())
with (H / 'adopt_root.py').open('xb') as f: f.write(raw)
with (H / 'SOURCE_DIFF.patch').open('xb') as f:
    f.write(''.join(difflib.unified_diff(original.decode().splitlines(True), text.splitlines(True), fromfile=P.relative_to(R).as_posix(), tofile='tmp/native380_adopter_prepare_20261011/adopt_root.py')).encode())
save('SOURCE_INVERSE.json', dict(parent_path=P.relative_to(R).as_posix(), parent_sha256=sha(P), source_sha256=sha(H/'adopt_root.py'), complete_raw_inverse_exact=True, edits=[dict(before=a, after=b) for a,b in edits]))
review = dict(status='SOURCE_ONLY_NATIVE380_FIXED10_ADOPTER_READY_FOR_ROOT_REVIEW_NOT_EXECUTED', source_adoptable=True, root_execution_authorized_by_this_preparation=False, adopter_sha256=sha(H/'adopt_root.py'), parent_adopter_sha256=sha(P), source_seal_sha256=sha(B/'FILES_SHA256.json'), collector_source_sha256=sha(B/'run_native_once.py'), actual_delta_path=(S/'DELTA_VERIFICATION.json').relative_to(R).as_posix(), actual_delta_sha256=sha(S/'DELTA_VERIFICATION.json'), prior_root_sha256=sha(PRIOR/'ROOT_DELTA_VERIFICATION.json'), exact_ids=ids, observed_native_count=380, accepted_total_if_root_adopts=380, actual_archive_members_from_original_offserver=110, old470_objects_and_order_exact=True, prior_full471_inspection_preserved=True, prior_pending_excluded_from_old_acceptance=[ids[0]], old51_ledger_prefix_exact=True, full_source_raw_inverse_exact=True, unchanged_blocks=block_checks, no_F_archive_opened=True, no_SSH_or_adopter_execution=True, no_scientific_verification_rerun=True, final_test=False)
save('ROOT_SOURCE_REVIEW.json', review)
actual = dict(status='ROOT_REVIEWED_ACTUAL_NATIVE_DELTA_AFTER370_NONIID_FEDSA10_READY_TO_ADOPT', root_execution_authorized=False, tag=TAG, accepted_total=380, new_ids=ids, archive_members=110, delta_sha256=sha(S/'DELTA_VERIFICATION.json'), source_review_sha256=sha(H/'ROOT_SOURCE_REVIEW.json'), source_seal_sha256=sha(B/'FILES_SHA256.json'), observed_native_count=380, pending_closed_ids=[], **{key:proof[key] for key in ('archive_sha256','inspection_sha256','offserver_proof_sha256','receipt_sha256','ledger_sha256')})
save('ACTUAL_INPUTS.template.json', actual)
readme = '''# native380 adopter — source only\n\n原370 adopter最小元数据派生；尚未执行、未根采用。Root先读SOURCE_DIFF和ROOT_SOURCE_REVIEW，核F卷Yanan 2TB/Healthy/容量及canonical ledger仍为4baa91…，再将ACTUAL_INPUTS.template.json原字节复制成ACTUAL_INPUTS.json，仅经root授权后把root_execution_authorized改true，计算实际SHA。唯一执行入口：\n\n```powershell\npython -B tmp/native380_adopter_prepare_20261011/adopt_root.py --actual-inputs-sha256 <ROOT实际授权的ACTUAL_INPUTS.json SHA256>\n```\n\n新批为non-IID FedSA91001..10，实际原strict/offserver110成员；archive/member/model与receipt检查循环原字节不变。旧370 inspection保留471对象；其中FedSA91001为已记录pending，只从旧接受投影排除，470对象（Full100+旧native370）及顺序精确保持。完整新inspection480对象，ledger51条前缀不变，追加1条至52。原远端/本地原子ledger提升逻辑仅namespace改变。实际归档仍在F，由root运行时读取；本次准备只读E元数据。任何失败保留且不自动重试。没有训练、forward、fit、test或新统计。\n'''
with (H/'README.md').open('xb') as f: f.write(readme.encode())
save('HANDOFF.json', dict(status=review['status'], adopter_path='tmp/native380_adopter_prepare_20261011/adopt_root.py', adopter_sha256=sha(H/'adopt_root.py'), source_review_sha256=sha(H/'ROOT_SOURCE_REVIEW.json'), actual_delta_sha256=sha(S/'DELTA_VERIFICATION.json'), tag=TAG, fixed_ids=ids, accepted_total_if_root_adopts=380, archive_members=110, archive_path=proof['archive_local_path'], archive_sha256=proof['archive_sha256'], receipt_sha256=proof['receipt_sha256'], offserver_sha256=proof['offserver_proof_sha256'], inspection_sha256=proof['inspection_sha256'], private_ledger_path=(S/'verified_ledger.json').relative_to(R).as_posix(), private_ledger_sha256=proof['ledger_sha256'], private_ledger_entries=52, prior_ledger_sha256=proof['previous_ledger_sha256'], executed=False, root_adoption=False, actual_inputs_file_not_yet_created=True))
files = {p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.iterdir()) if p.is_file() and p.name!='FILES_SHA256.json'}
save('FILES_SHA256.json', dict(status='SOURCE_ONLY_NO_EXECUTION_AUTHORIZATION', files=files))
print(json.dumps(dict(adopter_sha256=sha(H/'adopt_root.py'), review_sha256=sha(H/'ROOT_SOURCE_REVIEW.json'), handoff_sha256=sha(H/'HANDOFF.json'), seal_sha256=sha(H/'FILES_SHA256.json'), members=len(files), delta_sha256=sha(S/'DELTA_VERIFICATION.json'), block_checks=block_checks)))
