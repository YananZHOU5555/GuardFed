from pathlib import Path
import datetime,difflib,hashlib,json
R=Path.cwd();H=Path(__file__).resolve().parent;B=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
latest=read(B/'LATEST_BACKUP.json')
assert latest['accepted_total']==12
assert sha(R/latest['root_adoption_path'])==latest['root_adoption_sha256']=='67c355f4cd4fae1d2f015b327fee4b662499987ef9f0307043a1c22ca303a5b0'
assert sha(R/latest['next_collector_previous_path'])==latest['next_collector_previous_sha256']=='83bca8eef7ed5423315ad5bef4b42ccd0335bb750c4214f2e50abcd2db9e8e34'
assert sha(B/'FILES_SHA256.json')=='f6de59a56de25a8d316cf7e05c44eafc9751041757e4dc61c88f7bccfd988472'
for n,p in read(B/'FILES_SHA256.json')['files'].items():assert sha(B/n)==p['sha256']
prior=read(R/latest['next_collector_previous_path']);assert len(prior['accepted_job_ids'])==12==len(set(prior['accepted_job_ids']))
snapshot=R/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/observation_20261010T010714645866Z/SNAPSHOT.json';s=read(snapshot)
assert s['source_members_match'] and not s['changed_source_members'] and not s['failure_paths']
active={r['id'] for r in s['queue']['active']}
terminal=[r['id'] for r in s['rows'] if r['kind']=='new' and r['result_present'] and r['acceptance_present'] and r['progress']['round']==70 and r['id'] not in active]
assert set(prior['accepted_job_ids'])<=set(terminal)
ids=[identity for identity in terminal if identity not in prior['accepted_job_ids']]
expected=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed'+str(seed)+'_fullcoverage' for seed in (91004,91005)]
assert ids==expected
for n,p in [('PREVIOUS_LATEST.json',B/'LATEST_BACKUP.json'),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',R/latest['next_collector_previous_path'])]:
 with (H/n).open('xb') as f:f.write(p.read_bytes())
authorization=dict(status='FIXED_ONE_SNAPSHOT_EXACT_DELTA_NOT_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_path=str(snapshot.relative_to(R)),snapshot_sha256=sha(snapshot),snapshot_utc=s['utc'],prior_root_path=latest['root_adoption_path'],prior_root_sha256=latest['root_adoption_sha256'],prior_offserver_sha256=latest['next_collector_previous_sha256'],prior_count=12,authorized_ids=ids,source_hashes=s['source_hashes'],no_future_terminal_ids_allowed=True,root_adoption_required=True)
with (H/'AUTHORIZED_SNAPSHOT.json').open('x',encoding='utf8') as f:json.dump(authorization,f,indent=2);f.write('\n')
original=(B/'collect_delta.py').read_text(encoding='utf8');marker='    if not wanted:\n';assert original.count(marker)==1
addition='    # Parent-authorized fixed snapshot exact2 only; later terminal IDs remain unaccepted.\n    authorized_ids='+repr(ids)+'\n    wanted=[identity for identity in wanted if identity in authorized_ids]\n'
actual=original.replace(marker,addition+marker);assert actual.replace(addition,'')==original
with (H/'collect_delta.py').open('x',encoding='utf8',newline='\n') as f:f.write(actual)
with (H/'COLLECTOR_SCOPE_DIFF.patch').open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(original.splitlines(True),actual.splitlines(True),fromfile='original/collect_delta.py',tofile='fixed_exact2/collect_delta.py')))
with (H/'verify_delta_offserver.py').open('xb') as f:f.write((B/'verify_delta_offserver.py').read_bytes())
with (H/'SOURCE_REUSE.json').open('x',encoding='utf8') as f:json.dump(dict(original_collector_sha256=sha(B/'collect_delta.py'),collector_sha256=sha(H/'collect_delta.py'),verifier_sha256=sha(H/'verify_delta_offserver.py'),sole_change='Exact two ID filter before original terminal acceptance; scientific strict/archiver/verifier unchanged',filter_removed_source_exact=True,authorized_ids=ids,original_helper_seal_sha256=sha(B/'FILES_SHA256.json')),f,indent=2);f.write('\n')
print(json.dumps(dict(authorized_ids=ids,prior=12,collector_sha256=sha(H/'collect_delta.py'),snapshot_sha256=sha(snapshot))))
