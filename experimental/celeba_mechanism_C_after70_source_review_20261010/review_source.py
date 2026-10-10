"""Thin exact10 metadata rebinding of the accepted after60 source review."""
from pathlib import Path
import ast,hashlib,re,sys,traceback,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
parent=R/'tmp/review_C_after60_source_root_20261010.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='57b59c7f4380302652cb597729235ca40e490747250255bc6528a420a74d0f12'
source=parent.read_text('utf8')
changes={
 "B=R/'tmp/celeba_mechanism_valid_C_after60_20261010'":"B=R/'tmp/celeba_mechanism_valid_C_after70_20261010'",
 "P=R/'tmp/celeba_mechanism_valid_C_after56_20261010'":"P=R/'tmp/celeba_mechanism_valid_C_after60_20261010'",
 "H=R/'tmp/celeba_mechanism_C_after60_source_review_20261010'":"H=R/'tmp/celeba_mechanism_C_after70_source_review_20261010'",
 'H.mkdir(exist_ok=False)':'H.mkdir(exist_ok=True)',
 'minus_C_non-IID_F Flip_seed':'minus_C_non-IID_FedSA_seed',
 '5d35280b387c7ff621d94c00fb5983284edf06f272dba4f5a01dc78fb297f824':'370e75fc5e015048c3cba5c8c13e856a8bbcf8b5a8a46cb158a0468981303159',
 'ed8ecc84781b799e205c9e139dce8e753cb5545139b7e48ef23d5f8d07a50a77':'e08b8617bb30847b380372d8a3bdc3ac5b518136833295f998dc78603e6ad399',
 '12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896':'c93a5c605f7553bb336ba566a18bd87050f3d1e32d2b6320ee9ae15a51681224',
 'c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef':'b4ffdf5f3dc549e397f82db49290cdceb1ee024478cdc4e58365f5c537d144a9',
 'inventory_actual160_Full100refs.json':'inventory_actual170_Full100refs.json',
 'inventory_actual170_Full100refs.json':'inventory_actual180_Full100refs.json',
 'len(before)==160 and len(after)==len(current[\'records\'])==170':'len(before)==170 and len(after)==len(current[\'records\'])==180',
 "read(native)['native_accepted']==170":"read(native)['native_accepted']==180",
 'e45971c7f3db346f54594a7eccfc2f41c4bf2a4a7d015fab020a06dc7a52343e':'fc650143107f70e2017bb58cc864dd4f2cda2c7ac8e3894558cca81c86fae703',
 'native_accepted_snapshot=170,excluded_prior_three_view_ids=160':'native_accepted_snapshot=180,excluded_prior_three_view_ids=170',
 'old160_records_exact':'old170_records_exact',
 'old160_inventory_record_bytes_exact':'old170_inventory_record_bytes_exact',
 'prior160_root_adoption_sha256':'prior170_root_adoption_sha256',
 '21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e':'7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8',
}
assert all(k in source for k in changes)
effective=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m[0]],source)
extra="""
assert sha(B/'HANDOFF.json')=='42537f89df12c762a63393fa438137a4965d336c5ebb6cbee926789fa7f20848'
assert funcs(P/'bridge.py')['require_approval']==funcs(B/'bridge.py')['require_approval']
assert funcs(P/'execution_candidate/backup_completed.py')==funcs(B/'execution_candidate/backup_completed.py')
assert funcs(P/'execution_candidate/verify_saved_increment.py')['verify'].replace('c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef',pin['inventory'])==funcs(B/'execution_candidate/verify_saved_increment.py')['verify']
assert funcs(P/'execution_candidate/verify_backup.py')['main'].replace('inventory_actual170_Full100refs.json','inventory_actual180_Full100refs.json')==funcs(B/'execution_candidate/verify_backup.py')['main']
assert funcs(P/'execution_candidate/install_once.py')['main'].replace('inventory_actual170_Full100refs.json','inventory_actual180_Full100refs.json')==funcs(B/'execution_candidate/install_once.py')['main']
for filename in ['APPROVED_TEMPLATE.json','ROOT_REVIEW_TEMPLATE.json']:
 t=read(B/'execution_candidate'/filename)
 assert t['status']=='PREPARED_NOT_APPROVED' and t['execution_seal_sha256'] is None
 assert t['selected_ids']==IDS and t['compute_threads']==8 and t['max_processes']==1 and t['allowed_cpus']==list(range(112,120))
 assert t['inventory_sha256']==pin['inventory'] and t['bridge_sha256']==sha(B/'bridge.py') and t['native_tolerance']==1e-12
assert read(B/'execution_candidate/ROOT_REVIEW_TEMPLATE.json')['closed170_must_not_replay']==current['excluded_prior_replay_ids']
assert not read(B/'execution_candidate/ROOT_REVIEW_TEMPLATE.json')['execution_authorized_within_existing_user_request']
assert not (B/'execution_candidate/ROOT_APPROVED.json').exists() and not (B/'execution_candidate/EXECUTION_DRAFT.json').exists()
failed=read(B/'REBIND_COMMAND.json');assert failed['exit']==1
assert 'AssertionError' in (B/'REBIND_STDERR.txt').read_text('utf8')
assert read(B/'PREPARE_COMMAND.json')['exit']==0
report.update(original_require_approval_exact10_function_byte_exact=True,original_backup_functions_byte_exact=True,original_saved_array_strict_body_exact_except_inventory_pin=True,installer_main_exact_except_inventory_filename=True,backup_reader_main_exact_except_inventory_filename=True,first_constructor_failure_preserved=True,first_constructor_failure_sha256=sha(B/'REBIND_STDERR.txt'),first_failure_scope='Local source rebind assertion before prepare.py write; native checker/pending-input auxiliary sources had already been written. No replay/CNN/SSH or execution authority.',actual_dispatch_requires_new_root_approval=True,reviewer_authored_parent_after60_source=True,reviewer_authored_current_after70_source=False,parent_review_source_sha256='57b59c7f4380302652cb597729235ca40e490747250255bc6528a420a74d0f12',review_entry_sha256=sha(H/'review_source.py'))
"""
marker="with (H/'ROOT_INDEPENDENT_REVIEW.json').open('x',encoding='utf8') as f:"
assert effective.count(marker)==1;effective=effective.replace(marker,extra+'\n'+marker)
ast.parse(effective)
try:
 exec(compile(effective,str(parent),'exec'),dict(__file__=str(parent),__name__='__main__'))
except BaseException as e:
 with (H/'REVIEW_FAILURE.json').open('x',encoding='utf8') as f:json.dump(dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False),f,indent=2)
 raise
