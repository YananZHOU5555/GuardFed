"""Finish metadata binding from the already saved snapshot; no network calls."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
s=read(H/'SNAPSHOT.json');root=R/'tmp/root_adopt_first_closed_20261010/GRADIENT1_ROOT_ADOPTION.json';prior=read(root)
assert sha(root)=='ca4a38076e5080d94069cdabd50a032d84a96a339914761ec568db44a43db978'
assert read(H/'OWNER.json')==dict(CPU106_free=True,narrow_owners=[],existing_collectors=[])
assert sha(H/'GUIDE.stdout')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert all(read(H/(n+'_COMMAND.json'))['exit_code']==0 for n in ['GUIDE','OWNER','SNAPSHOT'])
assert (H/'OBSERVATION_FAILURE.json').is_file() and "KeyError('failed')"==read(H/'OBSERVATION_FAILURE.json')['error']
assert set(s['queue'])=={'strict_server_completed_ids','offserver_accepted','test'} and s['queue']['test'] is False
assert not s['failure_paths'] and s['service']['returncode']==0 and 'RUNNING' in s['service']['stdout']
pkg=R/'tmp/celeba_gradient_screen64_v2_20261010';manifest=read(pkg/'jobs/manifest.json')
assert sha(pkg/'FILES_SHA256.json')==prior['source_seal_sha256']
allids=[x['id'] for x in manifest['jobs']];completed=s['queue']['strict_server_completed_ids']
assert len(allids)==len(set(allids))==64 and len(completed)==len(set(completed)) and set(completed)<=set(allids)
assert prior['accepted_count']==1 and set(prior['accepted_ids'])<=set(completed)
ids=[x for x in allids if x in completed and x not in prior['accepted_ids']]
for identity in ids:
 row=next(x for x in s['rows'] if x['id']==identity)
 assert row['result'] and row['accepted'] and row['progress']['round']==row['diagnostics_rounds']==70
first=R/'tmp/celeba_gradient64_first_closed_adoption_20261010'
assert sha(first/'OFFSERVER_ACCEPTANCE.json')==prior['offserver_proof_sha256'] and sha(first/'backup_receipt.json')==prior['receipt_sha256']
auth=dict(status='FIXED_ONE_SNAPSHOT_TERMINAL_MINUS_ROOT1_NOT_ACCEPTANCE',snapshot_sha256=sha(H/'SNAPSHOT.json'),snapshot_unix=s['at_unix'],accepted_prior_ids=prior['accepted_ids'],accepted_prior_count=1,authorized_ids=ids,terminal_count=len(completed),manifest_sha256=sha(pkg/'jobs/manifest.json'),source_seal_sha256=sha(pkg/'FILES_SHA256.json'),prior_root_path=root.relative_to(R).as_posix(),prior_root_sha256=sha(root),prior_offserver_path=(first/'OFFSERVER_ACCEPTANCE.json').relative_to(R).as_posix(),prior_offserver_sha256=sha(first/'OFFSERVER_ACCEPTANCE.json'),prior_receipt_sha256=sha(first/'backup_receipt.json'),future_completions_excluded=True,root_source_review_required_before_collect=True,auxiliary_reader_failure_preserved=True,metadata_recovery_only_no_new_snapshot=True)
with (H/'AUTHORIZED_SNAPSHOT.json').open('x',encoding='utf8') as f:json.dump(auth,f,indent=2);f.write('\n')
print(json.dumps(dict(terminal=len(completed),exact_new_ids=ids,snapshot_sha256=auth['snapshot_sha256'],accepted_offserver=0)))
