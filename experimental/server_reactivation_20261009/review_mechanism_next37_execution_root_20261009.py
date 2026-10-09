"""Bind final execution bytes to the already reviewed37 science and root-run refusals."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--scope',type=int,choices=(37,11),default=37);args=parser.parse_args()
profile={37:dict(execution='4f71c60c6966da40d334f0ac17141a7e763520b7fe478df8548809e773971caf',science='95978fa42c28e9b4ff5b855b33c2dda56edc2b14fcfd56c3a29b0a9ba98135fd',members=16,rejections=32,install='bc9330e4d9360a3b806c2be54578adcd3fca729d70f512d798e54ae0f4c70f42'),
         11:dict(execution='9f1252dd7c11abfe7cee297b2028ca9c58f179d964efb39ee6008ea860508b15',science='65f706e8c7c7d7e18e76c8a300dd845297c97b3bd5c6c182bbbb8aad0103b5ff',members=17,rejections=39,install='1ef01d537a7936f86bd191fcb98380f3e074160d8f6e5630b59a6d20d6206f8a')}[args.scope]
BASE=ROOT/f'tmp/celeba_mechanism_valid_incremental_next{args.scope}_20261009'
EX=BASE/'execution_candidate'
OLD=ROOT/'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(EX/'EXECUTION_SOURCE_SHA256.json')==profile['execution']
assert sha(BASE/'FILES_SHA256.json')==profile['science']
members=read(EX/'EXECUTION_SOURCE_SHA256.json')['members'];assert len(members)==profile['members']
for row in members:assert sha(EX/row['path'])==row['sha256'] and (EX/row['path']).stat().st_size==row['size']
assert (OLD/'resource_extra.py').read_bytes()==(EX/'resource_extra.py').read_bytes()
assert sha(EX/'install_once.py')==profile['install']
if args.scope==37:
    assert sha(EX/'EXECUTION_DIFF.patch')=='e86fba0f2ef95b9c22564345d19fd37fec14ecb511a1503cdfdfc0f1a7f7b044'
for row in members:
    if row['path'].endswith('.py'):ast.parse((EX/row['path']).read_text())
result=subprocess.run([sys.executable,'-B',str(EX/'selfcheck.py')],cwd=ROOT,capture_output=True,check=True)
checked=json.loads(result.stdout)
assert checked==read(EX/'selfcheck.json') and checked['rejection_count']==profile['rejections']
assert checked['first_child_failure_calls']==1 and checked['failure_preserved'] and not checked['automatic_retry']
assert not checked['CNN_inference'] and checked['real_subprocesses_started']==0 and checked['saved_array_science_unchanged']
proof=dict(status=f'ROOT_NEXT{args.scope}_EXECUTION_SOURCE_REVIEW_PASS_NOT_DISPATCHED',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    execution_seal_sha256=sha(EX/'EXECUTION_SOURCE_SHA256.json'),execution_members_verified=profile['members'],
    science_seal_sha256=sha(BASE/'FILES_SHA256.json'),scientific_functions_unchanged=True,
    root_scientific_review_sha256=sha(BASE/'root_source_review/ROOT_REVIEW.json'),
    root_execution_rejection_count=profile['rejections'],root_execution_check_report_sha256=sha(EX/'selfcheck.json'),
    full_diff_reviewed_sha256=sha(EX/'EXECUTION_DIFF.patch'),
    reviewed_unchanged_source_data_and_selected_terminal_hash_preflight=True,
    reviewed_selected_terminal_count=args.scope,reviewed_science_member_count=(16 if args.scope==37 else 14),
    reviewed_new_fresh_CPU_owner_scan_and_conservative_extra3=True,
    reviewed_first_archive_includes_actual_external_authority_and_science=True,
    external_template_not_approval=True,actual_Linux_preflight_pending=True,CNN_executed=False,execution_started=False)
target=EX/'ROOT_EXECUTION_REVIEW.json'
with target.open('x',encoding='utf8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(proof|{'proof_sha256':sha(target)}))
